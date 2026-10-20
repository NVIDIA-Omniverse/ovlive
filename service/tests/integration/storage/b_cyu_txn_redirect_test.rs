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

//! b.cyu — a fred `Transaction` a cluster node redirects is transient at every
//! txn-pool call site, and the `CreateLayer` materialization's lost-reply
//! replay answers `UNAVAILABLE`.
//!
//! The txn pool allows no redirect, so a `MOVED` / `-ASK` to a fred
//! `Transaction` (`multi()` … `exec()`) fails it with fred's own `Unknown` +
//! "Max redirections exceeded", which before b.cyu the shared classifier
//! called permanent: `DeleteSession` and `CreateLayer` answered a slot
//! migration `INTERNAL`. Each test here stages the redirect with a
//! [`ScriptReplyProxy`] in front of the standalone test store, so fred's own
//! transaction router reads the reply and builds the error, and the service's
//! classification runs as in production. Each asserts the proxy matched the
//! frames it staged (the vacuity guard).
//!
//! | Test | What it pins |
//! |---|---|
//! | [`delete_session_whose_exec_is_redirected_answers_unavailable_and_keeps_the_session`] | (a) the `EXEC` answered `-ASK` (Valkey 9.x redirects the `EXEC`): `UNAVAILABLE`, the session's metadata kept, a retry deletes it |
//! | [`delete_session_whose_queued_marker_set_is_redirected_answers_unavailable_and_keeps_the_session`] | (b) the deletion marker's queued `SET` answered `-ASK`, the `EXEC` `EXECABORT` (Valkey ≤ 8.x and Redis 7.x check each queued command): the same |
//! | [`create_layer_whose_materialization_exec_is_redirected_answers_unavailable_and_writes_nothing`] | (c) the materialization's `EXEC` answered `-ASK`: plain `UNAVAILABLE`, no header, stream or mapping |
//! | [`create_layer_whose_landed_materialization_replay_is_redirected_answers_unavailable_and_a_resend_creates_it`] and its two variants | (c2) the first `EXEC` lands and its reply is lost, fred's one replay is redirected: plain `UNAVAILABLE` with no refusal detail, an unmapped header and no record left behind; a re-send creates the layer (`APPLIED` with the `operation_id`, `UNSPECIFIED` without; an overwrite leaves the old object mapped and not superseded until then) |
//! | [`create_layer_whose_landed_materialization_replay_reaches_the_store_answers_unavailable`] | (c3) the same lost reply, the replay reaching the store: its `XADD 1-0` is refused "equal or smaller" and the call answers `UNAVAILABLE` (decision 7), not `INTERNAL`; a re-send creates the layer |
//! | [`the_session_checkpoint_applier_grades_a_redirected_exec_transient_as_the_classifier_does`] | the applier half of the classifier/applier agreement (`checkpoint::session`'s unit test holds the classifier half): session recovery whose Step A `EXEC` is redirected ends `Transient`, and the re-drive recovers |
//!
//! Before b.cyu, (a), (b), (c) and (c2) answer `INTERNAL` (the classifier did
//! not match "Max redirections exceeded") and (c3) answers `INTERNAL` (the
//! materialization mapped the refused `XADD 1-0` with the general mapper);
//! the applier test fails on its classifier assertion.
//!
//! What a proxy cannot show is that a real cluster answers these shapes, and
//! the routing re-sync after a redirect, a no-op on a standalone store. Every
//! call site gets the re-sync by construction: its `EXEC` goes through
//! `ValkeyClient::exec_txn`, the only path a fred `Transaction`'s `EXEC`
//! takes, which `txn_exec_routing_guard_test` pins. The re-sync's run-time
//! effect is pinned only on a cluster, by the cluster-gated
//! `b_cyu_slot_migration_test`, for `DeleteSession` (T-C1, T-C2) and the
//! `CreateLayer` materialization (T-C3); the session checkpoint applier's
//! Step A and Step B have no behavioural test of it.
//!
//! # Wait budget
//!
//! Every call is a handful of round trips against a healthy store, plus, in
//! (c2) / (c3), one fred reconnect (backoff from 100 ms); each is bounded by
//! [`SETTLE`]. Healthy, every test takes well under a second.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fred::interfaces::{HashesInterface, KeysInterface};
use tonic::Code;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    CheckpointError, CheckpointMeta, CheckpointStore, SessionCheckpoint, SessionRecoveryHook,
    SessionRecoveryOutcome,
};
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::grpc::layer::handle_create_layer;
use live_edit::grpc::proto::{
    create_layer_request, CreateLayerHeader, CreateLayerRequest, OperationOutcome,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::storage::client::is_transient_fred_error;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::{delta_to_ompf_bytes, delta_with_set_field_inline};
use crate::common::refusal_detail::assert_no_refusal_detail;
use crate::grpc::helpers::{
    build_service, make_create_data, make_request, streaming_create, AllowAllPermissions,
};
use crate::storage::snapshot_read_support::{
    test_valkey_addr, ConnProxyRule, ProxyAction, ScriptReplyProxy,
};

/// Upper bound on one call under test. A deadline, not a speed claim.
const SETTLE: Duration = Duration::from_secs(10);

const POOL_SIZE: u32 = 4;
const CREATED_BY: &str = "bcyu-redirect-user";
const LAYER: &str = "bcyu-redirect-layer";
const OPERATION_ID: &str = "6f1c2d3e-4a5b-4c6d-8e7f-9a0b1c2d3e4f";

/// The redirect a cluster node answers a command naming a key of a slot it is
/// migrating away. The target is never contacted: the txn pool's redirection
/// budget is zero, so fred gives up on the first redirect.
const ASK_REPLY: &str = "ASK 1234 127.0.0.1:7001";

/// What fred's transaction router fails a redirected transaction with
/// (`vendor/fred-10.1.0/src/router/transactions.rs:65-67`).
const REDIRECTED: &str = "Max redirections exceeded";

/// The start of decision 7's answer to a materialization whose `XADD 1-0` was
/// refused as equal or smaller (`LayerStore::write_object`).
const XADD_REFUSED_ANSWER: &str =
    "EXEC object write: the stream entry 1-0 was refused as equal or smaller";

fn is_command(args: &[Vec<u8>], name: &str) -> bool {
    args.first()
        .is_some_and(|a| a.eq_ignore_ascii_case(name.as_bytes()))
}

fn names(args: &[Vec<u8>], key: &str) -> bool {
    args[1..].iter().any(|a| a.as_slice() == key.as_bytes())
}

/// A `TestValkey` whose every connection goes through `proxy`.
async fn through(proxy: &ScriptReplyProxy) -> TestValkey {
    TestValkey::setup_with_url(proxy.url(), POOL_SIZE).await
}

/// A rule redirecting the first `EXEC` of a transaction that queued
/// `queued(args)` on the same connection, once: the proxy sends `DISCARD` in
/// the `EXEC`'s place (nothing queued runs) and answers [`ASK_REPLY`].
/// Inactive while `armed` holds `None`.
fn redirect_exec_after(
    armed: Arc<Mutex<Option<String>>>,
    queued: fn(&[Vec<u8>], &str) -> bool,
) -> ConnProxyRule {
    let queued_on = AtomicUsize::new(usize::MAX);
    let fired = AtomicBool::new(false);
    Arc::new(move |conn: usize, args: &[Vec<u8>]| {
        if fired.load(Ordering::SeqCst) {
            return None;
        }
        let key = armed.lock().unwrap().clone()?;
        if queued(args, &key) {
            queued_on.store(conn, Ordering::SeqCst);
            return None;
        }
        (is_command(args, "EXEC")
            && queued_on.load(Ordering::SeqCst) == conn
            && !fired.swap(true, Ordering::SeqCst))
        .then(|| ProxyAction::DiscardReplying(ASK_REPLY.to_owned()))
    })
}

/// `Aborted` carrying [`REDIRECTED`], which reaches the client as
/// `UNAVAILABLE`.
fn assert_aborted_redirect(answer: Result<(), ServiceError>, what: &str) {
    match answer {
        Err(ServiceError::Aborted(msg)) => {
            assert!(
                msg.contains(REDIRECTED),
                "{what}: the answer must carry fred's {REDIRECTED:?}; got {msg:?}"
            );
            assert_eq!(
                tonic::Status::from(ServiceError::Aborted(msg)).code(),
                Code::Unavailable,
                "{what}: an Aborted DeleteSession reaches the client as UNAVAILABLE"
            );
        }
        other => panic!(
            "{what}: a redirected transaction must answer Aborted (UNAVAILABLE on the wire), \
             never Internal — before b.cyu it answered INTERNAL; got {other:?}"
        ),
    }
}

async fn session_meta_len(tv: &TestValkey, session_id: &str) -> usize {
    let meta: std::collections::HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().session_meta(session_id))
        .await
        .expect("HGETALL session meta must succeed");
    meta.len()
}

async fn exists(tv: &TestValkey, key: &str) -> bool {
    let n: u64 = tv
        .client()
        .pool
        .exists(key)
        .await
        .expect("EXISTS must succeed");
    n > 0
}

/// The session's metadata is still there and no deletion marker was written:
/// the redirected transaction ran nothing. Then a retry, with the proxy
/// inert, deletes the session — `DeleteSession` is idempotent, which is what
/// makes the transient answer safe to retry.
async fn assert_kept_then_deleted_by_a_retry(
    tv: &TestValkey,
    sessions: &SessionStore,
    session_id: &str,
) {
    assert!(
        session_meta_len(tv, session_id).await > 0,
        "the redirected transaction ran nothing: the session's metadata is kept"
    );
    assert!(
        !exists(tv, &tv.key_schema().session_deleted_marker(session_id)).await,
        "the redirected transaction ran nothing: no deletion marker"
    );
    tokio::time::timeout(SETTLE, sessions.delete(session_id))
        .await
        .expect("the retry did not settle within its budget")
        .expect("a retry, once the slot is served, deletes the session");
    assert_eq!(session_meta_len(tv, session_id).await, 0);
    assert!(exists(tv, &tv.key_schema().session_deleted_marker(session_id)).await);
}

// ═════════════════════════════════════════════════════════════════════════════
// DeleteSession
// ═════════════════════════════════════════════════════════════════════════════

/// (a) `DeleteSession`'s `EXEC` is answered `-ASK`, as Valkey 9.x answers an
/// `EXEC` whose queued keys are absent from a migrating slot (its cluster
/// check runs at `EXEC`, after discarding the transaction; the proxy sends
/// `DISCARD` in its place). fred fails the transaction "Max redirections
/// exceeded"; the call answers `Aborted` → `UNAVAILABLE`, nothing ran, and a
/// retry deletes the session.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_whose_exec_is_redirected_answers_unavailable_and_keeps_the_session() {
    let meta_key: Arc<Mutex<Option<String>>> = Arc::default();
    let proxy = ScriptReplyProxy::start_rewriting_replies(
        test_valkey_addr(),
        redirect_exec_after(Arc::clone(&meta_key), |args, key| {
            is_command(args, "DEL") && names(args, key)
        }),
    )
    .await;
    let tv = through(&proxy).await;
    let sessions = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = sessions
        .create("bcyu-delete-exec", "/bcyu-delete-exec", CREATED_BY)
        .await
        .expect("session creation must succeed");
    *meta_key.lock().unwrap() = Some(tv.key_schema().session_meta(&session_id));

    let answer = tokio::time::timeout(SETTLE, sessions.delete(&session_id))
        .await
        .expect("the delete did not settle within its budget");

    assert_eq!(
        proxy.hits(),
        1,
        "precondition: the DeleteSession EXEC was redirected once; the call answered {answer:?}"
    );
    assert_aborted_redirect(answer, "DeleteSession, EXEC redirected");
    assert_kept_then_deleted_by_a_retry(&tv, &sessions, &session_id).await;
}

/// (b) The deletion marker's queued `SET` is answered `-ASK`, as Valkey ≤ 8.x
/// and Redis 7.x answer a queued command naming a key absent from a
/// migrating slot; the store flags the transaction, so its `EXEC` answers
/// `EXECABORT` and nothing runs (the proxy sends a command the store refuses
/// before queueing in the `SET`'s place). fred reads the redirect in the
/// queued command's position and fails the transaction the same way: the
/// call answers `UNAVAILABLE`, nothing ran, and a retry deletes the session.
/// CI's cluster runs Valkey 9.0.3, so this is the only test of the queue-time
/// shape.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_whose_queued_marker_set_is_redirected_answers_unavailable_and_keeps_the_session(
) {
    let marker_key: Arc<Mutex<Option<String>>> = Arc::default();
    let rule: ConnProxyRule = {
        let marker_key = Arc::clone(&marker_key);
        let fired = AtomicBool::new(false);
        Arc::new(move |_conn: usize, args: &[Vec<u8>]| {
            let key = marker_key.lock().unwrap().clone()?;
            (is_command(args, "SET")
                && args.get(1).is_some_and(|a| a.as_slice() == key.as_bytes())
                && !fired.swap(true, Ordering::SeqCst))
            .then(|| ProxyAction::RefuseReplying(ASK_REPLY.to_owned()))
        })
    };
    let proxy = ScriptReplyProxy::start_rewriting_replies(test_valkey_addr(), rule).await;
    let tv = through(&proxy).await;
    let sessions = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = sessions
        .create("bcyu-delete-queued", "/bcyu-delete-queued", CREATED_BY)
        .await
        .expect("session creation must succeed");
    *marker_key.lock().unwrap() = Some(tv.key_schema().session_deleted_marker(&session_id));

    let answer = tokio::time::timeout(SETTLE, sessions.delete(&session_id))
        .await
        .expect("the delete did not settle within its budget");

    assert_eq!(
        proxy.hits(),
        1,
        "precondition: the queued marker SET was redirected once; the call answered {answer:?}"
    );
    assert_aborted_redirect(answer, "DeleteSession, queued SET redirected");
    assert_kept_then_deleted_by_a_retry(&tv, &sessions, &session_id).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// CreateLayer's materialization
// ═════════════════════════════════════════════════════════════════════════════

/// A rule over the `CreateLayer` materialization's `EXEC`s — each one a
/// connection sent after queueing the new object's `XADD … 1-0` under the
/// armed key prefix: the `n`-th (from 1) gets `actions[n - 1]` (`None` passes
/// it through), and every later one passes. `seen` counts them. Inactive
/// while `armed` holds `None`, so the fixture's own setup is untouched.
fn materialization_execs(
    armed: Arc<Mutex<Option<String>>>,
    actions: Vec<Option<ProxyAction>>,
    seen: Arc<AtomicUsize>,
) -> ConnProxyRule {
    let xadd_on: Mutex<HashSet<usize>> = Mutex::default();
    Arc::new(move |conn: usize, args: &[Vec<u8>]| {
        let prefix = armed.lock().unwrap().clone()?;
        if is_command(args, "XADD")
            && args
                .get(1)
                .is_some_and(|k| k.starts_with(prefix.as_bytes()))
            && args.iter().any(|a| a.as_slice() == b"1-0")
        {
            xadd_on.lock().unwrap().insert(conn);
            return None;
        }
        if is_command(args, "EXEC") && xadd_on.lock().unwrap().remove(&conn) {
            let n = seen.fetch_add(1, Ordering::SeqCst);
            return actions.get(n).cloned().flatten();
        }
        None
    })
}

/// The last `object_id` the store allocated: the object the call under test
/// materialized (or tried to).
async fn last_allocated(tv: &TestValkey) -> u64 {
    let raw: Option<String> = tv
        .client()
        .pool
        .get(tv.key_schema().counter_object_id())
        .await
        .expect("GET object-id counter must succeed");
    raw.expect("an object id was allocated")
        .parse()
        .expect("the counter is decimal")
}

async fn mapping(tv: &TestValkey, session_id: &str) -> Option<u64> {
    let raw: Option<String> = tv
        .client()
        .pool
        .get(tv.key_schema().layer_mapping(session_id, LAYER))
        .await
        .expect("GET layer mapping must succeed");
    raw.map(|v| v.parse().expect("the mapping holds a decimal object id"))
}

async fn new_session(tv: &TestValkey, name: &str) -> String {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(name, &format!("/{name}"), CREATED_BY)
        .await
        .expect("session creation must succeed")
}

/// (c) The materialization's `EXEC` is answered `-ASK` (the proxy discards
/// the transaction in its place). fred fails it "Max redirections exceeded",
/// which `write_object` maps to a plain `Unavailable`: nothing was written —
/// no header, no stream — and the layer is not mapped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_whose_materialization_exec_is_redirected_answers_unavailable_and_writes_nothing(
) {
    let armed: Arc<Mutex<Option<String>>> = Arc::default();
    let seen = Arc::new(AtomicUsize::new(0));
    let proxy = ScriptReplyProxy::start_rewriting_replies(
        test_valkey_addr(),
        materialization_execs(
            Arc::clone(&armed),
            vec![Some(ProxyAction::DiscardReplying(ASK_REPLY.to_owned()))],
            Arc::clone(&seen),
        ),
    )
    .await;
    let tv = through(&proxy).await;
    let session_id = new_session(&tv, "bcyu-create-exec").await;
    let store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    *armed.lock().unwrap() = Some(tv.prefix().to_owned());

    let answer = tokio::time::timeout(
        SETTLE,
        store.create_layer(
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
        ),
    )
    .await
    .expect("the create did not settle within its budget");

    assert_eq!(
        (proxy.hits(), seen.load(Ordering::SeqCst)),
        (1, 1),
        "precondition: the one materialization EXEC was redirected; the call answered {answer:?}"
    );
    match &answer {
        Err(ServiceError::Unavailable(msg)) => assert!(
            msg.starts_with("EXEC object write") && msg.contains(REDIRECTED),
            "a redirected materialization answers the plain mapper's Unavailable naming the \
             op and fred's {REDIRECTED:?}; got {msg:?}"
        ),
        other => panic!(
            "a redirected materialization must answer Unavailable, never Internal — before b.cyu \
             it answered INTERNAL; got {other:?}"
        ),
    }
    let oid = last_allocated(&tv).await;
    assert!(
        !exists(&tv, &tv.key_schema().obj_header(&oid.to_string())).await,
        "the redirected EXEC ran nothing: no header"
    );
    assert!(
        !exists(&tv, &tv.key_schema().layer_stream(oid)).await,
        "the redirected EXEC ran nothing: no stream"
    );
    assert_eq!(mapping(&tv, &session_id).await, None, "no layer was made");
}

/// What the scenario of (c2) creates over.
#[derive(Debug, Clone, Copy)]
enum LostReplyCase {
    /// A new layer, the call carrying an `operation_id`.
    NewLayer,
    /// A new layer, the call carrying none.
    AnonymousNewLayer,
    /// An existing layer (an overwrite), the call carrying an `operation_id`.
    Overwrite,
}

impl LostReplyCase {
    fn operation_id(self) -> &'static str {
        match self {
            Self::AnonymousNewLayer => "",
            Self::NewLayer | Self::Overwrite => OPERATION_ID,
        }
    }
}

fn create_header(session_id: &str, operation_id: &str) -> CreateLayerRequest {
    CreateLayerRequest {
        message_type: Some(create_layer_request::MessageType::Header(
            CreateLayerHeader {
                session_id: session_id.to_owned(),
                layer_name: LAYER.to_owned(),
                operation_id: operation_id.to_owned(),
            },
        )),
    }
}

/// One `CreateLayer` through the handler, answering `(object_id, outcome)`.
async fn create_through_handler(
    svc: &LiveEditService,
    session_id: &str,
    operation_id: &str,
    value: u8,
) -> Result<(u64, OperationOutcome), tonic::Status> {
    let payload = delta_to_ompf_bytes(
        &delta_with_set_field_inline(1, "color", vec![value], 0),
        &[],
    );
    let response = tokio::time::timeout(
        SETTLE,
        handle_create_layer(
            svc,
            make_request(streaming_create(vec![
                create_header(session_id, operation_id),
                make_create_data(payload),
            ])),
        ),
    )
    .await
    .expect("the CreateLayer call did not settle within its budget")?
    .into_inner();
    let outcome = OperationOutcome::try_from(response.operation_outcome).unwrap_or_else(|_| {
        panic!(
            "operation_outcome {} is not an OperationOutcome value",
            response.operation_outcome
        )
    });
    Ok((response.object_id, outcome))
}

/// The scenario of (c2) and (c3): the materialization's first `EXEC` lands
/// and its reply is lost (the proxy closes the connection once the reply has
/// arrived), so fred replays the whole transaction once on a new connection,
/// and the replay is handled by `replay` (`None`: it reaches the store).
/// Returns the failed call's status, the object the first send landed, the
/// layer's object before the call (an overwrite's), and the service and
/// fixture for the re-send.
async fn lost_reply_then(
    case: LostReplyCase,
    replay: Option<ProxyAction>,
    session_name: &str,
) -> (
    tonic::Status,
    u64,
    Option<u64>,
    String,
    LiveEditService,
    TestValkey,
    ScriptReplyProxy,
    Arc<AtomicUsize>,
) {
    let armed: Arc<Mutex<Option<String>>> = Arc::default();
    let seen = Arc::new(AtomicUsize::new(0));
    let proxy = ScriptReplyProxy::start_rewriting_replies(
        test_valkey_addr(),
        materialization_execs(
            Arc::clone(&armed),
            vec![Some(ProxyAction::CloseAfterReply), replay],
            Arc::clone(&seen),
        ),
    )
    .await;
    let tv = through(&proxy).await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = new_session(&tv, session_name).await;
    let old = match case {
        LostReplyCase::Overwrite => {
            let (old, _) = create_through_handler(&svc, &session_id, "", 1)
                .await
                .expect("the layer to overwrite must be created");
            Some(old)
        }
        _ => None,
    };
    *armed.lock().unwrap() = Some(tv.prefix().to_owned());

    let status = create_through_handler(&svc, &session_id, case.operation_id(), 2)
        .await
        .expect_err("a create whose materialization reply was lost must fail");
    *armed.lock().unwrap() = None;
    let landed = last_allocated(&tv).await;
    (status, landed, old, session_id, svc, tv, proxy, seen)
}

/// What the lost-reply scenario leaves, whatever its replay met: the first
/// send's object is complete and unmapped, the operation is not recorded, and
/// the layer still maps to what it mapped to before the call — an
/// overwrite's old object, not marked superseded. Then a re-send creates the
/// layer under a new object.
async fn assert_orphan_left_then_a_resend_creates_the_layer(
    case: LostReplyCase,
    landed: u64,
    old: Option<u64>,
    session_id: &str,
    svc: &LiveEditService,
    tv: &TestValkey,
) {
    assert!(
        exists(tv, &tv.key_schema().obj_header(&landed.to_string())).await,
        "{case:?}: the first send landed, so its object's header exists (an unmapped orphan)"
    );
    assert_eq!(
        mapping(tv, session_id).await,
        old,
        "{case:?}: no layer was made; the mapping is what it was before the call"
    );
    assert!(
        !exists(
            tv,
            &tv.key_schema()
                .create_layer_operation(session_id, OPERATION_ID)
        )
        .await,
        "{case:?}: the operation was not recorded"
    );
    if let Some(old) = old {
        let superseded: Option<String> = tv
            .client()
            .pool
            .hget(tv.key_schema().obj_header(&old.to_string()), "superseded")
            .await
            .expect("HGET superseded must succeed");
        assert_eq!(
            superseded, None,
            "{case:?}: the failed overwrite stopped before marking the old object superseded"
        );
    }

    let (created, outcome) = create_through_handler(svc, session_id, case.operation_id(), 2)
        .await
        .unwrap_or_else(|e| panic!("{case:?}: the re-send must create the layer; got {e:?}"));
    let expected_outcome = match case {
        LostReplyCase::AnonymousNewLayer => OperationOutcome::Unspecified,
        LostReplyCase::NewLayer | LostReplyCase::Overwrite => OperationOutcome::Applied,
    };
    assert_eq!(
        outcome, expected_outcome,
        "{case:?}: the re-send is a fresh create, answered as one"
    );
    assert_ne!(created, landed, "{case:?}: the orphan is never mapped");
    assert_ne!(
        Some(created),
        old,
        "{case:?}: the re-send made a new object"
    );
    assert_eq!(
        mapping(tv, session_id).await,
        Some(created),
        "{case:?}: the layer maps to the re-send's object"
    );
    let names = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .list_layer_names(session_id)
        .await
        .expect("listing the session's layers must succeed");
    assert_eq!(
        names,
        vec![LAYER.to_owned()],
        "{case:?}: the layer is listed once"
    );
}

/// (c2), one case: the replay is redirected (the proxy discards it in the
/// `EXEC`'s place and answers [`ASK_REPLY`]), so fred fails it "Max
/// redirections exceeded" — a redirect past the zero budget ran nothing in
/// that attempt, but the first send landed. The call answers a plain
/// `UNAVAILABLE` with no refusal detail (the commit marker is set, so nothing
/// claims "nothing applied").
async fn lost_reply_then_redirected_replay(case: LostReplyCase, session_name: &str) {
    let (status, landed, old, session_id, svc, tv, proxy, seen) = lost_reply_then(
        case,
        Some(ProxyAction::DiscardReplying(ASK_REPLY.to_owned())),
        session_name,
    )
    .await;

    assert_eq!(
        (proxy.hits(), seen.load(Ordering::SeqCst)),
        (2, 2),
        "{case:?} precondition: the first EXEC's reply was lost and fred's one replay was \
         redirected; the call answered {status:?}"
    );
    assert_eq!(
        status.code(),
        Code::Unavailable,
        "{case:?}: a redirected replay must answer UNAVAILABLE, never INTERNAL — before b.cyu it \
         answered INTERNAL; got {status:?}"
    );
    assert!(
        status.message().contains(REDIRECTED),
        "{case:?}: the answer carries fred's {REDIRECTED:?}; got {:?}",
        status.message()
    );
    assert_no_refusal_detail(&status);
    assert_orphan_left_then_a_resend_creates_the_layer(case, landed, old, &session_id, &svc, &tv)
        .await;
}

/// (c2) A new layer, with an `operation_id`: the re-send carrying the same
/// id finds no record, creates the layer under a new object and answers
/// `APPLIED`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_whose_landed_materialization_replay_is_redirected_answers_unavailable_and_a_resend_creates_it(
) {
    lost_reply_then_redirected_replay(LostReplyCase::NewLayer, "bcyu-c2-op").await;
}

/// (c2) A new layer, without an `operation_id`: the re-send creates it and
/// answers `UNSPECIFIED`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_without_an_operation_whose_landed_materialization_replay_is_redirected_answers_unavailable(
) {
    lost_reply_then_redirected_replay(LostReplyCase::AnonymousNewLayer, "bcyu-c2-noop").await;
}

/// (c2) An overwrite: the failed call leaves the old object mapped and not
/// superseded; the re-send replaces it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_overwrite_whose_landed_materialization_replay_is_redirected_answers_unavailable(
) {
    lost_reply_then_redirected_replay(LostReplyCase::Overwrite, "bcyu-c2-over").await;
}

/// (c3) The replay reaches the store: every queued write runs again, and its
/// `XADD 1-0` is refused as equal or smaller than the entry the first send
/// added. `write_object` recognises that refusal and answers a plain
/// `UNAVAILABLE` (decision 7: an earlier send may have landed, and no layer
/// was made), before the general mapper, which called it `INTERNAL` before
/// b.cyu. A re-send creates the layer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_whose_landed_materialization_replay_reaches_the_store_answers_unavailable() {
    let case = LostReplyCase::NewLayer;
    let (status, landed, old, session_id, svc, tv, proxy, seen) =
        lost_reply_then(case, None, "bcyu-c3").await;

    assert_eq!(
        (proxy.hits(), seen.load(Ordering::SeqCst)),
        (1, 2),
        "precondition: the first EXEC's reply was lost and fred's one replay reached the store; \
         the call answered {status:?}"
    );
    assert_eq!(
        status.code(),
        Code::Unavailable,
        "a replay whose XADD 1-0 is refused must answer UNAVAILABLE, never INTERNAL; got \
         {status:?}"
    );
    assert!(
        status.message().starts_with(XADD_REFUSED_ANSWER),
        "the answer is decision 7's; got {:?}",
        status.message()
    );
    assert_no_refusal_detail(&status);
    assert_orphan_left_then_a_resend_creates_the_layer(case, landed, old, &session_id, &svc, &tv)
        .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The session checkpoint applier
// ═════════════════════════════════════════════════════════════════════════════

/// The applier half of the classifier/applier agreement
/// (`checkpoint::session`'s
/// `the_storage_classifier_agrees_with_the_applier_that_the_fred_transaction_router_shapes_are_transient`
/// holds the classifier half). Session recovery applies a checkpoint whose
/// Step A `EXEC` (the path index's `SADD`s) is redirected: fred fails it "Max
/// redirections exceeded", the applier grades it `Transient` by its blanket
/// wrap, and the shared classifier grades the same shape transient too — the
/// two agree. The re-drive, with the proxy inert, recovers the session.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_session_checkpoint_applier_grades_a_redirected_exec_transient_as_the_classifier_does()
{
    let path_index_key: Arc<Mutex<Option<String>>> = Arc::default();
    let proxy = ScriptReplyProxy::start_rewriting_replies(
        test_valkey_addr(),
        redirect_exec_after(Arc::clone(&path_index_key), |args, key| {
            is_command(args, "SADD") && args.get(1).is_some_and(|a| a.as_slice() == key.as_bytes())
        }),
    )
    .await;
    let tv = through(&proxy).await;
    let session_id = format!("bcyu-applier-{}", uuid::Uuid::new_v4());
    let base_path = format!("/bcyu-applier/{session_id}");
    let mut checkpoint = SessionCheckpoint {
        schema_version: live_edit::checkpoint::session_payload::SESSION_CHECKPOINT_SCHEMA_VERSION,
        session_id: session_id.clone(),
        name: "bcyu-applier".to_owned(),
        base_path: base_path.clone(),
        created_by: CREATED_BY.to_owned(),
        created_at: 1_700_000_000_000_000,
        layer_mappings: Default::default(),
        layers_index: Default::default(),
        path_name_members: Default::default(),
        path_index_session_id_member: session_id.clone(),
    };
    checkpoint.add_path_name_member("bcyu-applier", &session_id);
    let store = Arc::new(InMemoryCheckpointStore::new());
    store
        .put_session_checkpoint(
            &session_id,
            checkpoint.to_bytes().expect("the checkpoint encodes"),
            CheckpointMeta::default(),
        )
        .await
        .expect("the in-memory store takes the checkpoint");
    let mut config = crate::common::test_config();
    config.checkpoint_enabled = true;
    let hook = SessionRecoveryHook::new(
        store,
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::new(config),
        Arc::new(crate::common::metrics_scrape_only("bcyu-applier")),
    );
    *path_index_key.lock().unwrap() = Some(tv.key_schema().path_index(&base_path));

    let first = tokio::time::timeout(SETTLE, hook.try_recover_session(&session_id))
        .await
        .expect("the recovery did not settle within its budget");

    assert_eq!(
        proxy.hits(),
        1,
        "precondition: the applier's Step A EXEC was redirected once; recovery answered \
         {first:?}"
    );
    match &first {
        SessionRecoveryOutcome::Error(CheckpointError::Transient { source }) => assert!(
            source.to_string().contains(REDIRECTED),
            "the applier's Transient carries fred's {REDIRECTED:?}; got {source}"
        ),
        other => panic!("the applier grades a redirected EXEC Transient; got {other:?}"),
    }
    assert!(
        is_transient_fred_error(&fred::error::Error::new(
            fred::error::ErrorKind::Unknown,
            REDIRECTED
        )),
        "the shared classifier must grade the shape the applier just met transient, as the \
         applier does — before b.cyu it called it permanent"
    );
    assert_eq!(
        session_meta_len(&tv, &session_id).await,
        0,
        "Step A ran nothing and Step B never ran"
    );

    let second = tokio::time::timeout(SETTLE, hook.try_recover_session(&session_id))
        .await
        .expect("the re-drive did not settle within its budget");
    assert!(
        matches!(second, SessionRecoveryOutcome::Recovered),
        "the re-drive recovers the session; got {second:?}"
    );
    assert!(session_meta_len(&tv, &session_id).await > 0);
}
