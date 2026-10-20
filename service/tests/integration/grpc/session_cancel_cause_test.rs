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

//! Why a session's token was cancelled decides how its streams end (SDD §12.6,
//! §13.2), and a session missing from the live store is resolved once more
//! through the session recovery hook before anything is cancelled (SDD §7.8).
//!
//! # The two causes
//!
//! * **`Deleted`** — an explicit `DeleteSession`: its own local cancel, its
//!   `le:session:deleted` fan-out, or a canceller that finds the session
//!   missing from the store *and carrying the deletion marker*
//!   `SessionStore::delete` writes. `SubscribeReadLayer` ends `NOT_FOUND
//!   "session was deleted"` (`reason=session_cancel`), `JoinSession` ends `OK`.
//!   Neither is retried: there is nothing to reconnect to.
//! * **`AbsentFromStore`** — a canceller (the `session_tokens` sweeper, or
//!   another stream's setup through `acquire_session_cancel`) finds the session
//!   missing with no marker: a keyspace loss, which a checkpoint may restore.
//!   Both RPCs end `UNAVAILABLE "session is missing from the live store; retry
//!   so it can be recovered"` (`reason=session_absent`), logged at WARN, so the
//!   client retries and the retry's setup can recover the session.
//!
//! The cause tests drive the real cancellers against a real store state — a
//! flushed session (its metadata `DEL`ed, as a keyspace loss leaves it) or a
//! deleted one (`SessionStore::delete`, which marks it) — not a hand-set cause,
//! so what they pin is the marker read that picks the cause, end to end. Two
//! tests call the test seam `cancel_absent_from_store` directly: the
//! shutdown-wins test, which must cancel both tokens with no yield in between,
//! and the test that `DeleteSession` records its `Deleted` cause, which needs
//! a store-absence cancel to reach a token `DeleteSession` already cancelled.
//!
//! # The recovery pass, and the deletion marker that bounds it
//!
//! A stream's setup often resolves its session from a fresh base-path cache
//! entry, which skips the recovery hook; `acquire_session_cancel` then finds it
//! missing and runs the hook once before cancelling. The cross-replica tests
//! here use two or three `LiveEditService`s over one Valkey, each with its own
//! token map and cache and all with an enabled recovery hook over one shared
//! in-memory checkpoint store:
//!
//! * a session flushed from the store is restored on the replica a stream
//!   retries on, even from a warm cache;
//! * a session `DeleteSession` removed is **not** restored, although its
//!   checkpoint is still at the adapter (no checkpoint writer is wired, so the
//!   tombstone never lands) and the fan-out never reached the other replica —
//!   the deletion marker refuses it.
//!
//! **Warm-cache precondition.** A base-path cache entry lives 2 s. The
//! warm-cache tests warm a replica's cache, then act within a few Valkey round
//! trips, and each checks it took the warm path — by the log line of the path
//! it exercises (the pass's INFO or DEBUG line), by a join's admission (a cold
//! lookup refuses the join at setup), or by the refusal arriving within the
//! wait with the budget's message (a cold lookup runs the hook under the 60 s
//! checkpoint fetch budget) — so a run that loses the entry to a runner stall
//! fails naming that, instead of passing through the cold path.
//!
//! # Log capture
//!
//! The teardown lines come from the spawned stream tasks, so the tests that
//! assert them run on a current-thread runtime under a [`ThreadCapture`] and
//! end with [`purge_keys`] (`TestValkey` cannot clean up there). The recovery
//! pass's own line is emitted inside the awaited handler future and is
//! captured with [`capture`] / [`capture_debug`] on a multi-thread runtime.
//!
//! # Wait budget
//!
//! Every wait is a deadline, not a measurement: `SETTLE` (10 s) per catch-up
//! drain (one deadline for the whole drain), awaited message, stream end or
//! polled condition; the helpers' `PRESENCE_WAIT` (10 s) per join read or
//! tracker drain; and `PASS_BUDGET` (10 s) for a direct canceller call's
//! recovery pass. The four tests sharing `stream_endings_follow_the_marker` are
//! the heaviest: a catch-up drain, `SessionConnected`, the canceller, the
//! subscription's end, its exit poll, the join's end, the tracker drain, the
//! `PeerLeft` fence and the tap's close, plus (the fan-out variant) the fan-out
//! fence and the fan-out task's join — about 110 s at every budget's full
//! value, too close to nextest's 120 s hang-killer, where a SIGKILL would print
//! nothing. So those four run under one `SCENARIO_BUDGET` (100 s) deadline and
//! fail with their own message first. The rest sum to at most about 60 s (the
//! shutdown-wins and step-6 tests). The healthy path pays none of it.
//!
//! # A second, silent activation condition
//!
//! The two step-6 tests (`mod step_six`: the join's recovery-pass budget, and
//! shutdown during that pass) drive `JoinRegistrationSeam`, which exists only
//! under **`test-fault-injection`**; without it both tests are compiled out and
//! the run reports two tests fewer.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface, PubsubInterface, SetsInterface};
use futures::{Stream, StreamExt};
use serde_json::Value;
use tonic::{Code, Request, Status};

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointMeta, CheckpointStore,
    SessionCheckpoint, SessionRecoveryHook,
};
use live_edit::error::ServiceError;
use live_edit::grpc::proto::{
    subscribe_read_layer_response, DeleteSessionRequest, SubscribeReadLayerRequest,
    SubscribeReadLayerResponse,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::session::{
    acquire_session_cancel, handle_delete_session, session_absent_status,
    spawn_session_delete_subscriber, subscribe_session_deleted_channel, sweep_session_tokens,
    SessionCancelCause,
};
use live_edit::grpc::session_cache::SessionBasePathCache;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{
    Metrics, STREAM_EXIT_REASONS, STREAM_EXIT_SESSION_ABSENT, STREAM_EXIT_SESSION_CANCEL,
    STREAM_EXIT_SHUTDOWN,
};
use live_edit::presence::{publish_presence_event, PresenceEvent};
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;
use super::log_capture::{
    assert_common_fields, assert_field, capture, capture_debug, purge_keys, Captured,
    ThreadCapture, RPC_FAILURE_MSG,
};

// =============================================================================
// Constants and fixtures
// =============================================================================

/// Upper bound on any single wait. Unreachable on the healthy path.
const SETTLE: Duration = Duration::from_secs(10);

/// The recovery-pass budget handed to `acquire_session_cancel` when a test
/// calls it directly. The healthy pass never approaches it.
const PASS_BUDGET: Duration = Duration::from_secs(10);

/// Whole-test deadline for the heaviest tests, below nextest's 120 s
/// hang-killer (module header, § Wait budget).
const SCENARIO_BUDGET: Duration = Duration::from_secs(100);

/// The layer every session here carries.
const LAYER: &str = "cause-layer";

/// The status message of a session found missing from the live store.
const ABSENT_MESSAGE: &str = "session is missing from the live store; retry so it can be recovered";

/// A service over `tv`, with an enabled session recovery hook over
/// `checkpoints` when given — wired into the base-path cache, as `main.rs`
/// does — and the helpers' disabled hook otherwise. Each call is a separate
/// replica: its own token map, cache and metrics, over the shared keyspace.
fn replica(tv: &TestValkey, checkpoints: Option<Arc<dyn CheckpointStore>>) -> LiveEditService {
    let base = build_service(tv, Arc::new(AllowAllPermissions));
    let Some(store) = checkpoints else {
        return base;
    };
    let mut config = crate::common::test_config();
    config.checkpoint_enabled = true;
    let hook = Arc::new(SessionRecoveryHook::new(
        store,
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::new(config),
        Arc::clone(&base.metrics),
    ));
    LiveEditService {
        session_base_path_cache: SessionBasePathCache::new().with_recovery_hook(Arc::clone(&hook)),
        session_recovery: hook,
        ..base
    }
}

/// The session checkpoint of a live session, read field by field from its
/// store state, with `LAYER` mapped to `object_id`: what the checkpoint writer
/// would have put at the adapter.
async fn checkpoint_of(tv: &TestValkey, session_id: &str, object_id: u64) -> SessionCheckpoint {
    let meta: std::collections::HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().session_meta(session_id))
        .await
        .expect("HGETALL session meta");
    let field = |name: &str| {
        meta.get(name)
            .unwrap_or_else(|| panic!("fixture: session meta has no `{name}`: {meta:?}"))
            .clone()
    };
    let mut checkpoint = SessionCheckpoint {
        schema_version: live_edit::checkpoint::session_payload::SESSION_CHECKPOINT_SCHEMA_VERSION,
        session_id: session_id.to_owned(),
        name: field("name"),
        base_path: field("base_path"),
        created_by: field("created_by"),
        created_at: field("created_at")
            .parse()
            .expect("fixture: created_at is an integer"),
        layer_mappings: std::collections::BTreeMap::from([(LAYER.to_owned(), object_id)]),
        layers_index: std::collections::BTreeSet::from([LAYER.to_owned()]),
        path_name_members: std::collections::BTreeSet::new(),
        path_index_session_id_member: session_id.to_owned(),
    };
    let name = checkpoint.name.clone();
    checkpoint.add_path_name_member(&name, session_id);
    checkpoint
}

/// An in-memory checkpoint store holding `checkpoint`.
async fn store_holding(checkpoint: &SessionCheckpoint) -> Arc<InMemoryCheckpointStore> {
    let store = Arc::new(InMemoryCheckpointStore::new());
    store
        .put_session_checkpoint(
            &checkpoint.session_id,
            checkpoint.to_bytes().expect("to_bytes"),
            CheckpointMeta {
                session_id: checkpoint.session_id.clone(),
                ..CheckpointMeta::default()
            },
        )
        .await
        .expect("put_session_checkpoint");
    store
}

/// Remove `session_id`'s session-scoped state the way a keyspace loss does —
/// metadata, layers index, layer mapping, and its path-index members — and
/// leave no deletion marker. The layer's object state is untouched.
async fn flush_session(tv: &TestValkey, session_id: &str, base_path: &str) {
    let ks = tv.key_schema();
    let pool = &tv.client().pool;
    let removed: i64 = pool
        .del(vec![
            ks.session_meta(session_id),
            ks.layers_index(session_id),
            ks.layer_mapping(session_id, LAYER),
        ])
        .await
        .expect("DEL the session-scoped keys");
    assert!(removed >= 1, "fixture: the session existed");
    let _: i64 = pool
        .srem(ks.path_index(base_path), session_id)
        .await
        .expect("SREM path index");
    let names: Vec<String> = pool
        .smembers(ks.path_names_index(base_path))
        .await
        .expect("SMEMBERS names index");
    for member in names.iter().filter(|m| m.ends_with(session_id)) {
        let _: i64 = pool
            .srem(ks.path_names_index(base_path), member.as_str())
            .await
            .expect("SREM names member");
    }
}

/// Every trace of `session_id` a recovery would write back: its metadata, its
/// path-index and names members, its layers index and its layer mapping.
async fn assert_session_not_resurrected(tv: &TestValkey, session_id: &str, base_path: &str) {
    let ks = tv.key_schema();
    let pool = &tv.client().pool;
    let exists = |key: String| async move {
        let found: bool = pool.exists(&key).await.expect("EXISTS");
        (key, found)
    };
    for (key, found) in [
        exists(ks.session_meta(session_id)).await,
        exists(ks.layers_index(session_id)).await,
        exists(ks.layer_mapping(session_id, LAYER)).await,
    ] {
        assert!(!found, "the deleted session's `{key}` came back");
    }
    let in_path_index: bool = pool
        .sismember(ks.path_index(base_path), session_id)
        .await
        .expect("SISMEMBER path index");
    assert!(
        !in_path_index,
        "the deleted session is back in the path index, so ListSessions shows it again"
    );
    let names: Vec<String> = pool
        .smembers(ks.path_names_index(base_path))
        .await
        .expect("SMEMBERS names index");
    assert!(
        !names.iter().any(|m| m.ends_with(session_id)),
        "the deleted session is back in the names index, holding its name: {names:?}"
    );
}

fn subscribe_request(
    session_id: &str,
    object_id: u64,
    sequence_num: u64,
) -> Request<SubscribeReadLayerRequest> {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: LAYER.to_owned(),
        object_id,
        sequence_num,
        client_id: "cause-subscriber".to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

fn delete_session_request(session_id: &str) -> Request<DeleteSessionRequest> {
    let mut request = Request::new(DeleteSessionRequest {
        session_id: session_id.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

type SubscribeStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Send>>;

/// Drain a subscription's catch-up through its caught-up ack (the one
/// `total_size == 0` header), leaving it idle in steady state. Returns the
/// ack's sequence. One deadline over the whole drain, so its worst case is
/// `SETTLE` however many messages the catch-up turns out to hold.
async fn drain_catch_up(stream: &mut SubscribeStream) -> u64 {
    tokio::time::timeout(SETTLE, async {
        loop {
            let msg = stream
                .next()
                .await
                .expect("the stream ended during catch-up")
                .unwrap_or_else(|s| panic!("catch-up failed: {s:?}"));
            if let Some(subscribe_read_layer_response::ResponseType::Header(h)) = msg.response_type
            {
                if h.total_size == 0 {
                    return h.sequence_num;
                }
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the catch-up did not reach its caught-up ack within {SETTLE:?}"))
}

/// Open a subscription on `svc` and drain its catch-up.
async fn open_idle_subscription(
    svc: &LiveEditService,
    session_id: &str,
    object_id: u64,
) -> SubscribeStream {
    let mut stream: SubscribeStream =
        handle_subscribe_read_layer(svc, subscribe_request(session_id, object_id, 0))
            .await
            .unwrap_or_else(|s| panic!("the subscription must be admitted: {s:?}"))
            .into_inner();
    drain_catch_up(&mut stream).await;
    stream
}

/// Drain to the terminal status under one deadline; `None` for a clean end.
async fn drain_to_terminal(stream: &mut SubscribeStream) -> Option<Status> {
    tokio::time::timeout(SETTLE, async {
        while let Some(item) = stream.next().await {
            if let Err(status) = item {
                return Some(status);
            }
        }
        None
    })
    .await
    .unwrap_or_else(|_| panic!("the subscription did not end within {SETTLE:?}"))
}

fn stream_exits(metrics: &Metrics, reason: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_subscriber_stream_exits_total",
        &[("reason", reason)],
    )
    .unwrap_or_else(|| panic!("the stream-exit series for reason={reason} is absent")) as u64
}

/// Wait until the subscription's tail has counted its exit (it logs its line
/// first), then require exactly one exit, under `reason`.
async fn await_the_one_exit(metrics: &Metrics, reason: &str) {
    let total = || -> u64 {
        STREAM_EXIT_REASONS
            .iter()
            .map(|r| stream_exits(metrics, r))
            .sum()
    };
    let deadline = tokio::time::Instant::now() + SETTLE;
    while total() == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "no stream exit counted within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        stream_exits(metrics, reason),
        1,
        "the subscription's one exit must be counted under `{reason}`; exits: {:?}",
        STREAM_EXIT_REASONS
            .iter()
            .map(|r| (*r, stream_exits(metrics, r)))
            .collect::<Vec<_>>()
    );
    assert_eq!(total(), 1, "exactly one exit");
}

/// The one completion line an endpoint logged.
fn only_failure_of<'c>(captured: &'c Captured, endpoint: &str) -> &'c Value {
    let lines: Vec<&Value> = captured
        .with_message(RPC_FAILURE_MSG)
        .into_iter()
        .filter(|e| e["fields"]["endpoint"] == endpoint)
        .collect();
    assert_eq!(
        lines.len(),
        1,
        "expected exactly one {endpoint} completion line; full capture:\n{}",
        captured.text
    );
    lines[0]
}

/// Teardown's end state for connection `cid`: no member entry, info hash or
/// lease.
async fn assert_join_torn_down(tv: &TestValkey, session_id: &str, cid: &str) {
    let ks = tv.key_schema();
    let reader = store_reader(tv);
    let member: bool = reader
        .sismember(ks.presence_members(session_id), cid)
        .await
        .expect("SISMEMBER");
    assert!(!member, "teardown must remove the member entry");
    for key in [
        ks.presence_user(session_id, cid),
        ks.presence_lease(session_id, cid),
    ] {
        let found: bool = reader.exists(&key).await.expect("EXISTS");
        assert!(!found, "teardown must delete `{key}`");
    }
}

/// An independent observer on one session's presence channel — a peer on
/// another replica.
struct PresenceTap {
    session_id: String,
    rx: tokio::sync::mpsc::UnboundedReceiver<PresenceEvent>,
    task: tokio::task::JoinHandle<()>,
    subscriber: fred::clients::SubscriberClient,
}

impl PresenceTap {
    async fn open(tv: &TestValkey, session_id: &str) -> Self {
        let (rx, task, subscriber) = spawn_external_channel_subscriber(tv, session_id).await;
        Self {
            session_id: session_id.to_owned(),
            rx,
            task,
            subscriber,
        }
    }

    /// How many `PeerLeft`s for `cid` were published before a fence this
    /// publishes now. Call once every join task has finished.
    async fn lefts_for(&mut self, tv: &TestValkey, cid: &str) -> usize {
        self.events_before_fence(tv)
            .await
            .iter()
            .filter(
                |e| matches!(e, PresenceEvent::PeerLeft { connection_id } if connection_id == cid),
            )
            .count()
    }

    /// Every event published on the channel before a fence this publishes now.
    /// Call once every join task has finished.
    async fn events_before_fence(&mut self, tv: &TestValkey) -> Vec<PresenceEvent> {
        const FENCE: &str = "cause-test-fence";
        publish_presence_event(
            &tv.client().pool,
            tv.key_schema(),
            &self.session_id,
            &PresenceEvent::PeerMessage {
                connection_id: FENCE.to_owned(),
                data: Vec::new(),
            },
        )
        .await
        .expect("publishing the fence must succeed");
        let rx = &mut self.rx;
        tokio::time::timeout(SETTLE, async move {
            let mut seen = Vec::new();
            loop {
                match rx.recv().await.expect("the tap's forwarding task ended") {
                    PresenceEvent::PeerMessage { connection_id, .. } if connection_id == FENCE => {
                        return seen;
                    }
                    event => seen.push(event),
                }
            }
        })
        .await
        .unwrap_or_else(|_| panic!("the fence did not reach the tap within {SETTLE:?}"))
    }

    async fn close(self) {
        use fred::interfaces::ClientLike;
        let _ = self.subscriber.quit().await;
        let _ = tokio::time::timeout(SETTLE, self.task).await;
    }
}

// =============================================================================
// The cause, end to end: a real canceller against a real store state
// =============================================================================

/// How the session leaves the live store.
#[derive(Clone, Copy, Debug)]
enum Removal {
    /// Its metadata is `DEL`ed, as a keyspace loss leaves it: no marker.
    Flush,
    /// `SessionStore::delete`, `DeleteSession`'s store step: the marker is
    /// written, but neither the handler's local cancel nor its fan-out runs,
    /// and no layer is deleted, so no layer sentinel can race the token.
    DeleteWithMarker,
}

/// Which canceller finds the session missing.
#[derive(Clone, Copy, Debug)]
enum Canceller {
    /// One pass of the `session_tokens` sweeper.
    Sweeper,
    /// A second stream's setup, through `acquire_session_cancel`: the
    /// cross-replica race, where the store entry is gone and the fan-out has
    /// not reached this replica.
    AnotherSetup,
}

/// An idle `SubscribeReadLayer` and an established `JoinSession` share one
/// session's token on one replica. The session leaves the store by `removal`,
/// `canceller` finds it missing, and both streams must end by the cause the
/// deletion marker picks: an unmarked session `UNAVAILABLE` on both (WARN), a
/// marked one `NOT_FOUND "session was deleted"` / `OK`. The join ends only
/// after its teardown. With the marker and another setup — the cross-replica
/// race — a late `le:session:deleted` fan-out then changes nothing.
///
/// Runs under [`SCENARIO_BUDGET`] (module header, § Wait budget).
async fn stream_endings_follow_the_marker(removal: Removal, canceller: Canceller) {
    tokio::time::timeout(
        SCENARIO_BUDGET,
        stream_endings_follow_the_marker_body(removal, canceller),
    )
    .await
    .unwrap_or_else(|_| {
        panic!("{removal:?} found by {canceller:?} did not finish within {SCENARIO_BUDGET:?}")
    });
}

async fn stream_endings_follow_the_marker_body(removal: Removal, canceller: Canceller) {
    let what = format!("{removal:?} found by {canceller:?}");
    let tv = TestValkey::setup().await;
    let svc = replica(&tv, None);
    let (session_id, object_id) = create_session_and_layer(
        &tv,
        &svc,
        &format!("cause-{removal:?}-{canceller:?}"),
        LAYER,
    )
    .await;
    let mut tap = PresenceTap::open(&tv, &session_id).await;
    let capture = ThreadCapture::start();

    let mut subscription = open_idle_subscription(&svc, &session_id, object_id).await;
    let (mut join, _client) = open_join(&svc, &session_id, "cause-app").await;
    let cid = expect_session_connected(&mut join, SETTLE).await;
    let shared = svc
        .session_tokens
        .get(&session_id)
        .expect("both streams registered the session's token")
        .clone();

    match removal {
        Removal::Flush => flush_session(&tv, &session_id, "/stage").await,
        Removal::DeleteWithMarker => svc
            .session_store
            .delete(&session_id)
            .await
            .expect("SessionStore::delete must succeed"),
    }
    match canceller {
        Canceller::Sweeper => {
            let removed = sweep_session_tokens(
                &svc.session_tokens,
                &svc.session_store,
                &svc.session_base_path_cache,
            )
            .await;
            assert_eq!(removed, 1, "{what}: the sweep removes the session's entry");
        }
        Canceller::AnotherSetup => {
            let result = acquire_session_cancel(
                &svc.session_tokens,
                &svc.session_store,
                &svc.session_base_path_cache,
                PASS_BUDGET,
                &session_id,
            )
            .await;
            assert!(
                matches!(result, Err(ServiceError::NotFound(_))),
                "{what}: the second setup is refused NOT_FOUND; got {result:?}"
            );
        }
    }
    let (want_cause, want_reason) = match removal {
        Removal::Flush => (
            SessionCancelCause::AbsentFromStore,
            STREAM_EXIT_SESSION_ABSENT,
        ),
        Removal::DeleteWithMarker => (SessionCancelCause::Deleted, STREAM_EXIT_SESSION_CANCEL),
    };
    assert_eq!(shared.cause(), want_cause, "{what}: the cancel's cause");

    // SubscribeReadLayer.
    let status = drain_to_terminal(&mut subscription)
        .await
        .unwrap_or_else(|| panic!("{what}: the subscription ended cleanly (gRPC OK)"));
    match removal {
        Removal::Flush => {
            assert_eq!(status.code(), Code::Unavailable, "{what}: {status:?}");
            assert_eq!(status.message(), ABSENT_MESSAGE, "{what}");
        }
        Removal::DeleteWithMarker => {
            assert_eq!(status.code(), Code::NotFound, "{what}: {status:?}");
            assert_eq!(status.message(), "session was deleted", "{what}");
        }
    }
    await_the_one_exit(&svc.metrics, want_reason).await;

    // JoinSession, after its teardown.
    let (_, join_status) = read_to_end(&mut join, &what).await;
    match removal {
        Removal::Flush => {
            let status = join_status
                .unwrap_or_else(|| panic!("{what}: the join ended OK, which Kit never retries"));
            assert_eq!(status.code(), Code::Unavailable, "{what}: {status:?}");
            assert_eq!(
                status.message(),
                session_absent_status().message(),
                "{what}: the same status the subscription got"
            );
        }
        Removal::DeleteWithMarker => assert!(
            join_status.is_none(),
            "{what}: a deleted session ends the join OK (PRD §3.1.2); got {join_status:?}"
        ),
    }
    drain_tracker(&svc).await;
    assert_join_torn_down(&tv, &session_id, &cid).await;
    assert_eq!(
        tap.lefts_for(&tv, &cid).await,
        1,
        "{what}: peers see exactly one PeerLeft, teardown's"
    );

    // The completion lines: WARN for a session missing from the store (it is
    // keyspace loss, worth an operator's attention), INFO for a deletion, and
    // none for the join's OK.
    let captured = capture.snapshot();
    let sub_line = only_failure_of(&captured, "SubscribeReadLayer");
    match removal {
        Removal::Flush => {
            assert_common_fields(
                sub_line,
                "WARN",
                "SubscribeReadLayer",
                "stream",
                "UNAVAILABLE",
            );
            assert_field(sub_line, "reason", STREAM_EXIT_SESSION_ABSENT.into());
            assert_field(sub_line, "status_message", ABSENT_MESSAGE.into());
            let join_line = only_failure_of(&captured, "JoinSession");
            assert_common_fields(join_line, "WARN", "JoinSession", "stream", "UNAVAILABLE");
            assert_field(join_line, "status_message", ABSENT_MESSAGE.into());
        }
        Removal::DeleteWithMarker => {
            assert_common_fields(
                sub_line,
                "INFO",
                "SubscribeReadLayer",
                "stream",
                "NOT_FOUND",
            );
            assert_field(sub_line, "reason", STREAM_EXIT_SESSION_CANCEL.into());
            assert!(
                captured
                    .with_message(RPC_FAILURE_MSG)
                    .iter()
                    .all(|e| e["fields"]["endpoint"] != "JoinSession"),
                "{what}: a join ended OK logs no completion line; capture:\n{}",
                captured.text
            );
        }
    }
    drop(capture);

    // The late fan-out. `apply_session_deleted` runs on the replica's real
    // `le:session:deleted` subscriber; a second entry published after it is
    // the fence (pub/sub delivers one channel in order), so once that entry is
    // gone the session's message has been applied too.
    if let (Removal::DeleteWithMarker, Canceller::AnotherSetup) = (removal, canceller) {
        subscribe_session_deleted_channel(&tv.client().subscriber, tv.key_schema())
            .await
            .expect("subscribe to the session-deleted channel");
        let fan_out_stop = tokio_util::sync::CancellationToken::new();
        let fan_out = spawn_session_delete_subscriber(
            tv.client().subscriber.clone(),
            Arc::clone(&svc.session_tokens),
            svc.session_base_path_cache.clone(),
            tv.key_schema().clone(),
            Arc::clone(&svc.metrics),
            live_edit::grpc::session::SessionRevalidation::new(),
            fan_out_stop.clone(),
        );
        const FENCE_SESSION: &str = "cause-test-fan-out-fence";
        svc.session_tokens.insert(
            FENCE_SESSION.to_owned(),
            live_edit::grpc::session::SessionCancel::new(),
        );
        let channel = tv.key_schema().session_deleted_channel();
        for id in [session_id.as_str(), FENCE_SESSION] {
            let _: i64 = tv
                .client()
                .pool
                .next()
                .publish(&channel, id)
                .await
                .expect("PUBLISH on the session-deleted channel");
        }
        let deadline = tokio::time::Instant::now() + SETTLE;
        while svc.session_tokens.contains_key(FENCE_SESSION) {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the late fan-out was not applied within {SETTLE:?}"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(
            !svc.session_tokens.contains_key(&session_id),
            "the late fan-out must not bring an entry back"
        );
        assert_eq!(
            shared.cause(),
            SessionCancelCause::Deleted,
            "the first recorded cause stands"
        );
        assert_eq!(stream_exits(&svc.metrics, STREAM_EXIT_SESSION_CANCEL), 1);
        assert_eq!(stream_exits(&svc.metrics, STREAM_EXIT_SESSION_ABSENT), 0);
        fan_out_stop.cancel();
        let _ = tokio::time::timeout(SETTLE, fan_out).await;
    }

    tap.close().await;
    purge_keys(&tv).await;
}

/// A session lost from the keyspace, found by the sweeper: both streams end
/// `UNAVAILABLE` (`session_absent`), logged at WARN, so their clients retry
/// into the recovery hook. Before the cause existed they ended as a deletion,
/// which no client retries.
#[tokio::test]
async fn session_absent_via_the_sweeper_ends_both_streams_unavailable() {
    stream_endings_follow_the_marker(Removal::Flush, Canceller::Sweeper).await;
}

/// The same, found by another stream's setup (`acquire_session_cancel`).
#[tokio::test]
async fn session_absent_via_another_setup_ends_both_streams_unavailable() {
    stream_endings_follow_the_marker(Removal::Flush, Canceller::AnotherSetup).await;
}

/// A deleted session (marked), found by the sweeper on a replica that missed
/// the fan-out: the deletion endings, `NOT_FOUND` and `OK`.
#[tokio::test]
async fn deleted_session_via_the_sweeper_ends_with_the_deletion_endings() {
    stream_endings_follow_the_marker(Removal::DeleteWithMarker, Canceller::Sweeper).await;
}

/// The cross-replica race: the store entry is gone, the fan-out has not
/// arrived, a stream is already on the replica, and a second setup finds the
/// session missing. The marker makes the existing streams end as a deletion,
/// and the fan-out arriving afterwards changes nothing.
#[tokio::test]
async fn deleted_session_via_another_setup_before_the_fan_out_ends_with_the_deletion_endings() {
    stream_endings_follow_the_marker(Removal::DeleteWithMarker, Canceller::AnotherSetup).await;
}

/// Shutdown and a session-absent cancel observable together: shutdown wins on
/// both RPCs, `UNAVAILABLE "server shutting down"` at INFO, counted under
/// `shutdown`. Both tokens are cancelled before either stream task runs again
/// (current-thread runtime, no yield in between), so each stream sees both and
/// its shutdown-first check decides.
#[tokio::test]
async fn shutdown_and_a_session_absent_cancel_together_end_both_streams_as_a_shutdown() {
    let tv = TestValkey::setup().await;
    let svc = replica(&tv, None);
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "cause-shutdown-wins", LAYER).await;
    let capture = ThreadCapture::start();

    let mut subscription = open_idle_subscription(&svc, &session_id, object_id).await;
    let (mut join, _client) = open_join(&svc, &session_id, "cause-app").await;
    let _cid = expect_session_connected(&mut join, SETTLE).await;
    let shared = svc
        .session_tokens
        .get(&session_id)
        .expect("both streams registered the session's token")
        .clone();

    svc.shutdown.cancel();
    shared.cancel_absent_from_store();

    let status = drain_to_terminal(&mut subscription)
        .await
        .expect("a shutdown ends the subscription with a status");
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert_eq!(status.message(), "server shutting down");
    await_the_one_exit(&svc.metrics, STREAM_EXIT_SHUTDOWN).await;
    let (_, join_status) = read_to_end(&mut join, "shutdown with a session-absent cancel").await;
    let join_status = join_status.expect("a shutdown ends the join UNAVAILABLE");
    assert_eq!(join_status.code(), Code::Unavailable, "{join_status:?}");
    assert_eq!(join_status.message(), "server shutting down");
    drain_tracker(&svc).await;

    let captured = capture.snapshot();
    let sub_line = only_failure_of(&captured, "SubscribeReadLayer");
    assert_common_fields(
        sub_line,
        "INFO",
        "SubscribeReadLayer",
        "stream",
        "UNAVAILABLE",
    );
    assert_field(sub_line, "reason", STREAM_EXIT_SHUTDOWN.into());
    let join_line = only_failure_of(&captured, "JoinSession");
    assert_common_fields(join_line, "INFO", "JoinSession", "stream", "UNAVAILABLE");
    drop(capture);
    purge_keys(&tv).await;
}

// =============================================================================
// The recovery pass across replicas, and the marker that bounds it
// =============================================================================

/// The case the retryable ending exists for, end to end: a stream on replica A
/// is cancelled `AbsentFromStore` after a keyspace loss, and the client's retry
/// lands on replica B, whose base-path cache still holds the session. B's
/// setup finds the session missing at its re-check, the recovery pass restores
/// it from its checkpoint, and the retry is served — where before the pass it
/// was refused `NOT_FOUND` and the session was lost to that client.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stream_cancelled_as_absent_retries_on_a_warm_replica_and_is_served_after_the_restore() {
    let tv = TestValkey::setup().await;
    let seed = replica(&tv, None);
    let (session_id, object_id) =
        create_session_and_layer(&tv, &seed, "cause-restore", LAYER).await;
    let checkpoints: Arc<dyn CheckpointStore> =
        store_holding(&checkpoint_of(&tv, &session_id, object_id).await).await;
    let replica_a = replica(&tv, Some(Arc::clone(&checkpoints)));
    let replica_b = replica(&tv, Some(Arc::clone(&checkpoints)));

    let mut on_a = open_idle_subscription(&replica_a, &session_id, object_id).await;
    replica_b
        .session_base_path_cache
        .get_base_path(&replica_b.session_store, &session_id)
        .await
        .expect("warming replica B's cache");

    flush_session(&tv, &session_id, "/stage").await;
    let removed = sweep_session_tokens(
        &replica_a.session_tokens,
        &replica_a.session_store,
        &replica_a.session_base_path_cache,
    )
    .await;
    assert_eq!(removed, 1, "A's sweeper finds the session missing");
    let ended = drain_to_terminal(&mut on_a)
        .await
        .expect("A's subscription ends with a status");
    assert_eq!(
        ended.code(),
        Code::Unavailable,
        "the ending the client retries: {ended:?}"
    );

    // The retry, on B.
    let (retried, captured) = capture(handle_subscribe_read_layer(
        &replica_b,
        subscribe_request(&session_id, object_id, 0),
    ))
    .await;
    let mut retried: SubscribeStream = retried
        .unwrap_or_else(|s| {
            panic!("the retry on the warm replica must be served after the restore, got {s:?}")
        })
        .into_inner();
    drain_catch_up(&mut retried).await;

    assert!(
        !captured
            .with_message(
                "acquire_session_cancel: session was missing from the live store at the re-check \
                 and resolved again through the recovery pass"
            )
            .is_empty(),
        "precondition: the retry took the warm-cache path, through the recovery pass. If this \
         fails, B's 2 s cache entry expired before the retry (a runner stall), and the retry was \
         served through the cold path instead; re-run. Capture:\n{}",
        captured.text
    );
    let restored: bool = tv
        .client()
        .pool
        .exists(tv.key_schema().session_meta(&session_id))
        .await
        .expect("EXISTS session meta");
    assert!(restored, "the session is back in the live store");
    drop(retried);
}

/// The decider's race (SDD §22.12): a `DeleteSession` on replica A has removed
/// the session from the store, its fan-out has not reached replicas B and C,
/// and its checkpoint tombstone has not landed — the checkpoint is still at
/// the adapter. B and C still hold the session in their base-path caches, so a
/// `SubscribeReadLayer` on B and a `JoinSession` on C pass their lookups and
/// reach `acquire_session_cancel`, whose recovery pass would restore the
/// session from that checkpoint. The deletion marker refuses it: both end
/// `NOT_FOUND`, and none of the session's state comes back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_session_deleted_on_another_replica_is_not_restored_by_a_warm_replicas_recovery_pass() {
    let tv = TestValkey::setup().await;
    let seed = replica(&tv, None);
    let (session_id, object_id) = create_session_and_layer(&tv, &seed, "cause-race", LAYER).await;
    let checkpoints: Arc<dyn CheckpointStore> =
        store_holding(&checkpoint_of(&tv, &session_id, object_id).await).await;
    let replica_a = replica(&tv, Some(Arc::clone(&checkpoints)));
    let replica_b = replica(&tv, Some(Arc::clone(&checkpoints)));
    let replica_c = replica(&tv, Some(Arc::clone(&checkpoints)));
    // Opened before the caches are warmed, so its connect does not spend the
    // 2 s the warm entries live.
    let mut tap = PresenceTap::open(&tv, &session_id).await;
    for warm in [&replica_b, &replica_c] {
        warm.session_base_path_cache
            .get_base_path(&warm.session_store, &session_id)
            .await
            .expect("warming the cache");
    }

    handle_delete_session(&replica_a, delete_session_request(&session_id))
        .await
        .expect("DeleteSession on A must succeed");
    assert!(
        checkpoints
            .get_session_checkpoint(&session_id)
            .await
            .expect("adapter read")
            .is_some(),
        "precondition: the checkpoint tombstone has not landed (no writer is wired)"
    );

    // SubscribeReadLayer on B.
    let (subscribed, captured) = capture_debug(handle_subscribe_read_layer(
        &replica_b,
        subscribe_request(&session_id, object_id, 0),
    ))
    .await;
    let status = expect_err_status(
        subscribed,
        "a subscription to a deleted session must be refused, not served from a restore",
    );
    assert_eq!(status.code(), Code::NotFound, "{status:?}");
    assert!(
        !captured
            .with_message(
                "acquire_session_cancel: session missing from the live store and not \
                 recoverable; cancelled its streams and returning NotFound"
            )
            .is_empty(),
        "precondition: the refusal came from acquire_session_cancel's recovery pass, not the \
         cold-cache lookup. If this fails, B's 2 s cache entry expired first (a runner stall); \
         re-run. Capture:\n{}",
        captured.text
    );

    // JoinSession on C: admitted on its warm lookup, then refused at step 6,
    // after registering, and torn down. The admission is what proves the warm
    // path: on a cold cache the handler's own lookup refuses the join first.
    let (client_stream, client) = open_join_streaming();
    client
        .send(encode_join_frame(&join_header(&session_id, "cause-app")))
        .expect("the client channel must accept the header frame");
    let mut join = live_edit::grpc::presence::handle_join_session(
        &replica_c,
        make_join_request(client_stream),
    )
    .await
    .unwrap_or_else(|s| {
        panic!(
            "precondition: C's handler admits the join on its warm cache and the refusal comes \
             at step 6. It refused at setup instead ({s:?}), so C's 2 s cache entry expired first \
             (a runner stall) and the lookup took the cold path; re-run"
        )
    })
    .into_inner();
    let (delivered, join_status) = read_to_end(&mut join, "join on a deleted session").await;
    let join_status = join_status.expect("a join on a deleted session must not end OK");
    assert_eq!(join_status.code(), Code::NotFound, "{join_status:?}");
    assert!(
        delivered.is_empty(),
        "no SessionConnected for a deleted session: {delivered:?}"
    );
    drain_tracker(&replica_c).await;
    // It registered (its `PeerJoined`) and tore down (exactly one `PeerLeft`
    // for the same connection): the refusal was step 6's, after registration.
    let events = tap.events_before_fence(&tv).await;
    let joined: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            PresenceEvent::PeerJoined(user) => Some(user.connection_id.as_str()),
            _ => None,
        })
        .collect();
    let [cid] = joined.as_slice() else {
        panic!("the join on C must have registered exactly once; events {events:?}");
    };
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(
                e,
                PresenceEvent::PeerLeft { connection_id } if connection_id == cid
            ))
            .count(),
        1,
        "the refused join tears down with exactly one PeerLeft; events {events:?}"
    );
    tap.close().await;

    assert_session_not_resurrected(&tv, &session_id, "/stage").await;
    for (name, svc) in [("B", &replica_b), ("C", &replica_c)] {
        assert_eq!(
            sample_value(
                &render(&svc.metrics),
                "live_edit_checkpoint_recovery_attempts_total",
                &[
                    ("result", "restored"),
                    ("scope", "session"),
                    ("service", "test")
                ],
            )
            .unwrap_or(0.0),
            0.0,
            "replica {name} restored the deleted session"
        );
    }
}

/// A client retries a `DeleteSession` whose first attempt succeeded — its
/// reply lost, say — while the checkpoint tombstone has not landed. The retry
/// is a clean `NOT_FOUND`, and the session stays deleted. Without the marker
/// the retry's lookup restored the session from its checkpoint and deleted it
/// again, answering `OK` to a delete of a session that no longer existed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_retried_delete_session_before_the_tombstone_lands_is_not_found_and_restores_nothing() {
    let tv = TestValkey::setup().await;
    let seed = replica(&tv, None);
    let (session_id, object_id) =
        create_session_and_layer(&tv, &seed, "cause-retried-delete", LAYER).await;
    let checkpoints: Arc<dyn CheckpointStore> =
        store_holding(&checkpoint_of(&tv, &session_id, object_id).await).await;
    let svc = replica(&tv, Some(Arc::clone(&checkpoints)));

    handle_delete_session(&svc, delete_session_request(&session_id))
        .await
        .expect("the first DeleteSession must succeed");
    let retry = handle_delete_session(&svc, delete_session_request(&session_id)).await;

    let status = expect_err_status(retry, "a retried DeleteSession must not succeed again");
    assert_eq!(status.code(), Code::NotFound, "{status:?}");
    assert!(
        checkpoints
            .get_session_checkpoint(&session_id)
            .await
            .expect("adapter read")
            .is_some(),
        "precondition: the checkpoint tombstone has not landed (no writer is wired)"
    );
    assert_session_not_resurrected(&tv, &session_id, "/stage").await;
}

/// `DeleteSession`'s own cancel records `Deleted` on the session's token — it
/// does not merely leave the cause slot empty, which also reads `Deleted`. The
/// difference shows when a later canceller reaches the same token: a sweeper
/// or another stream's setup that finds the session missing (no marker yet on
/// its read, or a failed one) cancels with `AbsentFromStore`, and only a
/// recorded `Deleted` keeps the streams' deletion endings.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_records_the_deleted_cause_so_a_later_absent_cancel_changes_nothing() {
    let tv = TestValkey::setup().await;
    let svc = replica(&tv, None);
    let (session_id, _) = create_session(&tv, &svc, "cause-delete-records", true).await;
    let shared = svc
        .session_tokens
        .get(&session_id)
        .expect("the session's token is registered on this replica")
        .clone();

    handle_delete_session(&svc, delete_session_request(&session_id))
        .await
        .expect("DeleteSession must succeed");
    assert!(shared.is_cancelled(), "DeleteSession cancels the token");
    shared.cancel_absent_from_store();

    assert_eq!(
        shared.cause(),
        SessionCancelCause::Deleted,
        "DeleteSession must record its cause, so a later store-absence cancel cannot replace it"
    );
}

// =============================================================================
// A slow recovery pass: the budget each handler computes, and shutdown
// =============================================================================

/// The presence lease knobs the budget tests build their services with:
/// `min(refresh / 2, ttl − refresh − 1 ms)` = `min(1 000 ms, 999 ms)`, so the
/// clamp term is the one in force — an arithmetic neither the checkpoint fetch
/// budget (60 s) nor the default presence knobs (7.5 s) can produce.
const SMALL_REFRESH_SECS: u64 = 2;
const SMALL_LEASE_TTL_SECS: u64 = 3;

/// The status a recovery pass that outlives the budget those knobs give ends
/// with, verbatim.
const PASS_ELAPSED_MESSAGE: &str = "session recovery did not finish within 999 ms; retry";

/// A checkpoint store whose session fetch signals that it started and then
/// never completes: a slow adapter under the recovery pass.
struct StalledFetchStore {
    entered: tokio::sync::Notify,
}

#[async_trait]
impl CheckpointStore for StalledFetchStore {
    async fn put_layer_checkpoint(
        &self,
        _object_id: u64,
        _snap_seq: u64,
        _ompf: Bytes,
        _meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        unreachable!("the session recovery hook writes no layer checkpoint")
    }
    async fn get_latest_layer_checkpoint(
        &self,
        _object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        Ok(None)
    }
    async fn list_layer_checkpoints(
        &self,
        _object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        Ok(vec![])
    }
    async fn delete_layer_checkpoint(
        &self,
        _handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        Ok(())
    }
    async fn put_session_checkpoint(
        &self,
        _session_id: &str,
        _body: Bytes,
        _meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        unreachable!("the session recovery hook writes no session checkpoint")
    }
    async fn get_session_checkpoint(
        &self,
        _session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.entered.notify_one();
        std::future::pending().await
    }
    async fn delete_session_checkpoint(&self, _session_id: &str) -> Result<(), CheckpointError> {
        Ok(())
    }
    async fn put_ext_blob(&self, _hash_hex: &str, _bytes: Bytes) -> Result<(), CheckpointError> {
        Ok(())
    }
    async fn get_ext_blob(&self, _hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        Ok(None)
    }
    async fn delete_ext_blob(&self, _hash_hex: &str) -> Result<(), CheckpointError> {
        Ok(())
    }
    async fn put_counters_sidecar(&self, _body: Bytes) -> Result<(), CheckpointError> {
        Ok(())
    }
    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        Ok(None)
    }
    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        Ok(vec![])
    }
    async fn list_referenced_ext_hashes(
        &self,
    ) -> Result<std::collections::HashSet<String>, CheckpointError> {
        Ok(std::collections::HashSet::new())
    }
}

impl StalledFetchStore {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            entered: tokio::sync::Notify::new(),
        })
    }
}

/// `SubscribeReadLayer` computes its recovery-pass budget from the service's
/// presence knobs (`config::session_recovery_pass_budget`), as `JoinSession`
/// does: a subscription whose setup finds its session missing, from a warm
/// cache, into a checkpoint fetch that never returns, is refused `UNAVAILABLE`
/// after exactly the budget those knobs give — 999 ms here — and cancels
/// nothing. A handler that passed another budget (the checkpoint fetch
/// budget's 60 s, say) would give another message, or none within the wait.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscribe_recovery_pass_is_bounded_by_the_budget_the_presence_knobs_give() {
    let tv = TestValkey::setup().await;
    let seed = replica(&tv, None);
    let (session_id, object_id) =
        create_session_and_layer(&tv, &seed, "cause-subscribe-budget", LAYER).await;
    let svc = LiveEditService {
        presence_refresh_interval_secs: SMALL_REFRESH_SECS,
        presence_keepalive_interval_ms: live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
        presence_lease_ttl_secs: SMALL_LEASE_TTL_SECS,
        ..replica(
            &tv,
            Some(StalledFetchStore::new() as Arc<dyn CheckpointStore>),
        )
    };
    svc.session_base_path_cache
        .get_base_path(&svc.session_store, &session_id)
        .await
        .expect("warming the cache");
    let removed: i64 = tv
        .client()
        .pool
        .del(tv.key_schema().session_meta(&session_id))
        .await
        .expect("DEL session meta");
    assert_eq!(removed, 1, "fixture: the session existed");

    let refused = tokio::time::timeout(
        SETTLE,
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, object_id, 0)),
    )
    .await
    .expect(
        "the recovery pass must end at its 999 ms budget, not wait on the stalled adapter. Two \
         causes print this: the handler passed another budget (the checkpoint fetch budget's \
         60 s, say), or the service's 2 s cache entry expired before the subscribe (a runner \
         stall) and its cold-path lookup is waiting out the 60 s fetch budget; if the second, \
         re-run",
    );
    let status = expect_err_status(
        refused,
        "a subscription whose recovery pass never finishes must be refused",
    );
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert_eq!(
        status.message(),
        PASS_ELAPSED_MESSAGE,
        "the pass must be bounded by the budget the presence knobs give"
    );
    assert!(
        svc.session_tokens.contains_key(&session_id),
        "an elapsed pass cancels nothing and leaves the entry"
    );
}

/// Compiled only with `test-fault-injection` (the registration seam).
#[cfg(feature = "test-fault-injection")]
mod step_six {
    use super::*;

    use live_edit::grpc::presence::JoinRegistrationSeam;

    /// A service over `tv` whose recovery hook's fetch never returns, with the
    /// registration seam installed, and optionally the small presence knobs.
    fn stalled_service(
        tv: &TestValkey,
        small_knobs: bool,
    ) -> (
        LiveEditService,
        Arc<StalledFetchStore>,
        Arc<JoinRegistrationSeam>,
    ) {
        let stalled = StalledFetchStore::new();
        let seam = Arc::new(JoinRegistrationSeam::new());
        let base = replica(tv, Some(stalled.clone() as Arc<dyn CheckpointStore>));
        let (refresh, ttl) = if small_knobs {
            (SMALL_REFRESH_SECS, SMALL_LEASE_TTL_SECS)
        } else {
            (
                base.presence_refresh_interval_secs,
                base.presence_lease_ttl_secs,
            )
        };
        let svc = LiveEditService {
            join_registration_seam_for_test: Some(Arc::clone(&seam)),
            presence_refresh_interval_secs: refresh,
            presence_keepalive_interval_ms:
                live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
            presence_lease_ttl_secs: ttl,
            ..base
        };
        (svc, stalled, seam)
    }

    /// Open a join on `session_id`, park it after its registration, flush the
    /// session's metadata, and release it into SR-4 step 6, whose re-check
    /// finds the session missing and runs the recovery pass into the stalled
    /// fetch. Returns the join's stream, its client sender and its connection
    /// id once the fetch has started.
    async fn into_a_stalled_step_six(
        tv: &TestValkey,
        svc: &LiveEditService,
        stalled: &StalledFetchStore,
        seam: &JoinRegistrationSeam,
        session_id: &str,
    ) -> (
        JoinResponseStream,
        tokio::sync::mpsc::UnboundedSender<bytes::Bytes>,
        String,
    ) {
        let gate = &seam.before_session_cancel;
        gate.arm();
        let (join, client) = open_join(svc, session_id, "cause-app").await;
        tokio::time::timeout(SETTLE, gate.wait_until_parked())
            .await
            .expect("the join did not park after its registration");
        let members: Vec<String> = store_reader(tv)
            .smembers(tv.key_schema().presence_members(session_id))
            .await
            .expect("SMEMBERS");
        let [cid] = members.as_slice() else {
            panic!("precondition: the parked join is registered; members {members:?}");
        };
        let removed: i64 = tv
            .client()
            .pool
            .del(tv.key_schema().session_meta(session_id))
            .await
            .expect("DEL session meta");
        assert_eq!(removed, 1, "fixture: the session existed");

        gate.release();
        tokio::time::timeout(SETTLE, stalled.entered.notified())
            .await
            .expect("step 6's recovery pass did not reach the checkpoint fetch");
        (join, client, cid.clone())
    }

    /// `JoinSession` computes its recovery-pass budget from its presence knobs
    /// (`config::session_recovery_pass_budget`): with no shutdown, a step-6
    /// pass into a fetch that never returns ends the join `UNAVAILABLE
    /// "session recovery did not finish within 999 ms; retry"` — the budget
    /// those knobs give — logged at WARN, after its teardown (member, info
    /// hash and lease gone; one `PeerLeft`). A handler that passed another
    /// budget (the checkpoint fetch budget's 60 s, say) would still be waiting
    /// when the join read's deadline runs out.
    #[tokio::test]
    async fn step_six_recovery_pass_is_bounded_by_the_budget_the_presence_knobs_give() {
        let tv = TestValkey::setup().await;
        let (svc, stalled, seam) = stalled_service(&tv, true);
        let (session_id, _) = create_session(&tv, &svc, "cause-step-six-budget", false).await;
        let mut tap = PresenceTap::open(&tv, &session_id).await;
        let capture = ThreadCapture::start();

        let (mut join, _client, cid) =
            into_a_stalled_step_six(&tv, &svc, &stalled, &seam, &session_id).await;

        let (delivered, status) = read_to_end(&mut join, "a step-6 pass past its budget").await;
        let status = status.expect("an elapsed recovery pass must not end the join OK");
        assert_eq!(status.code(), Code::Unavailable, "{status:?}");
        assert_eq!(status.message(), PASS_ELAPSED_MESSAGE);
        assert!(
            delivered.is_empty(),
            "the join stops before SessionConnected: {delivered:?}"
        );
        drain_tracker(&svc).await;
        assert_join_torn_down(&tv, &session_id, &cid).await;
        assert_eq!(
            tap.lefts_for(&tv, &cid).await,
            1,
            "peers see exactly one PeerLeft, teardown's"
        );

        let captured = capture.snapshot();
        let line = only_failure_of(&captured, "JoinSession");
        assert_common_fields(line, "WARN", "JoinSession", "stream", "UNAVAILABLE");
        assert_field(line, "status_message", PASS_ELAPSED_MESSAGE.into());
        drop(capture);
        tap.close().await;
        purge_keys(&tv).await;
    }

    /// A graceful shutdown that begins while a join's SR-4 step 6 is inside a slow
    /// recovery pass ends the join at once: `UNAVAILABLE "server shutting down"`,
    /// logged at INFO as a shutdown, with its teardown run (member, info hash and
    /// lease gone; one `PeerLeft`). Step 6 runs with the connection registered and
    /// the lease keeper not yet started, so a join held there past the shutdown
    /// budget would never tear down, and its presence row would linger until the
    /// sweeper reaped the lapsed lease.
    ///
    /// The join is parked after its registration (the registration seam), the
    /// session's metadata is flushed, and the join is released into step 6, whose
    /// re-check finds the session missing and runs the recovery pass into a fetch
    /// that never returns. Shutdown is fired once the fetch has started. At the
    /// default presence knobs the pass's own budget is 7.5 s, so a join that did
    /// not race shutdown would end on that budget instead — `UNAVAILABLE` with
    /// another message, at WARN.
    #[tokio::test]
    async fn shutdown_during_a_slow_step_six_recovery_pass_ends_the_join_unavailable_after_teardown(
    ) {
        let tv = TestValkey::setup().await;
        let (svc, stalled, seam) = stalled_service(&tv, false);
        let (session_id, _) = create_session(&tv, &svc, "cause-step-six", false).await;
        let mut tap = PresenceTap::open(&tv, &session_id).await;
        let capture = ThreadCapture::start();

        let (mut join, _client, cid) =
            into_a_stalled_step_six(&tv, &svc, &stalled, &seam, &session_id).await;
        let cid = cid.as_str();
        svc.shutdown.cancel();

        let (delivered, status) = read_to_end(&mut join, "shutdown during step 6").await;
        let status = status.expect("a shutdown must not end the join OK");
        assert_eq!(status.code(), Code::Unavailable, "{status:?}");
        assert_eq!(
            status.message(),
            "server shutting down",
            "the join must end on the shutdown, not wait out the recovery pass's budget"
        );
        assert!(
            delivered.is_empty(),
            "the join stops before SessionConnected: {delivered:?}"
        );
        drain_tracker(&svc).await;
        assert_join_torn_down(&tv, &session_id, cid).await;
        assert_eq!(
            tap.lefts_for(&tv, cid).await,
            1,
            "peers see exactly one PeerLeft, teardown's"
        );

        let captured = capture.snapshot();
        let line = only_failure_of(&captured, "JoinSession");
        assert_common_fields(line, "INFO", "JoinSession", "stream", "UNAVAILABLE");
        assert_field(line, "status_message", "server shutting down".into());
        drop(capture);
        tap.close().await;
        purge_keys(&tv).await;
    }
}
