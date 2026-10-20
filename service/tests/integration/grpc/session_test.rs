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

// Integration tests for CreateSession and ListSessions handlers.
//
// Tests call `handle_create_session` and `handle_list_sessions` directly with
// a LiveEditService backed by real Valkey (TestValkey fixture) and mock auth.

#[cfg(feature = "test-fault-injection")]
use std::collections::VecDeque;
use std::sync::Arc;
#[cfg(feature = "test-fault-injection")]
use std::sync::Mutex;

use dashmap::DashMap;
#[cfg(feature = "test-fault-injection")]
use tokio_util::sync::CancellationToken;
use tonic::Request;

use live_edit::auth::userinfo::UserinfoClient;
use live_edit::error::ServiceError;
use live_edit::grpc::proto::{CreateSessionRequest, ListSessionsRequest};
use live_edit::grpc::session::{
    acquire_session_cancel, handle_create_session, handle_list_sessions, SessionCancel,
    SessionCancelCause,
};
use live_edit::grpc::session_cache::SessionBasePathCache;
use live_edit::grpc::BearerToken;
use live_edit::storage::session::SessionStore;
#[cfg(feature = "test-fault-injection")]
use live_edit::storage::test_support::release_all;
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;

// ── Request builders ────────────────────────────────────────────────────────

fn create_session_request(path: &str, name: &str) -> Request<CreateSessionRequest> {
    let mut request = Request::new(CreateSessionRequest {
        path: path.to_owned(),
        name: name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

fn list_sessions_request(path: &str) -> Request<ListSessionsRequest> {
    let mut request = Request::new(ListSessionsRequest {
        path: path.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

// ═══════════════════════════════════════════════════════════════════════════════
// CreateSession tests
// ═══════════════════════════════════════════════════════════════════════════════

// ── 1. Happy path ───────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_happy_path_returns_session_id() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let response = handle_create_session(&svc, create_session_request("/stage", "my-session"))
        .await
        .expect("happy path must succeed");

    let resp = response.into_inner();
    assert!(!resp.session_id.is_empty(), "session_id must be non-empty");
}

// ── 2. Empty path → INVALID_ARGUMENT ────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_empty_path_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let status = handle_create_session(&svc, create_session_request("", "my-session"))
        .await
        .expect_err("empty path must fail");

    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "empty path must return INVALID_ARGUMENT, got: {}",
        status.code()
    );
}

// ── 3. Auth denied → PERMISSION_DENIED ──────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_auth_denied_returns_permission_denied() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(DenyAllPermissions));

    let status = handle_create_session(&svc, create_session_request("/stage", "denied-session"))
        .await
        .expect_err("denied auth must fail");

    assert_eq!(
        status.code(),
        tonic::Code::PermissionDenied,
        "denied auth must return PERMISSION_DENIED, got: {}",
        status.code()
    );
}

// ── 3a. Auth unauthenticated (expired/invalid token) → UNAUTHENTICATED ──────
// Regression: previously 401 from storage was conflated with 403 and returned
// PERMISSION_DENIED, which broke token-refresh flows in standard gRPC clients.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_auth_unauthenticated_returns_unauthenticated() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(UnauthenticatedPermissions));

    let status = handle_create_session(&svc, create_session_request("/stage", "unauth-session"))
        .await
        .expect_err("unauthenticated auth must fail");

    assert_eq!(
        status.code(),
        tonic::Code::Unauthenticated,
        "unauthenticated auth must return UNAUTHENTICATED, got: {}",
        status.code()
    );
}

// ── 3b. Auth path-not-found → NOT_FOUND with the actual path in the message ─
// Regression: previously the message was a bare "base path not found" which
// hid which input was rejected.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_path_not_found_includes_path_in_message() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(PathNotFoundPermissions));

    let status = handle_create_session(
        &svc,
        create_session_request("/missing/stage", "no-path-session"),
    )
    .await
    .expect_err("path-not-found auth must fail");

    assert_eq!(status.code(), tonic::Code::NotFound);
    assert!(
        status.message().contains("/missing/stage"),
        "message must contain the rejected path, got: {:?}",
        status.message()
    );
}

// ── 4. Session creates a CancellationToken entry ────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_registers_cancellation_token() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let response = handle_create_session(&svc, create_session_request("/stage", "token-session"))
        .await
        .expect("session creation must succeed");

    let session_id = response.into_inner().session_id;

    assert!(
        svc.session_tokens.contains_key(&session_id),
        "session_tokens map must contain the new session"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// ListSessions tests
// ═══════════════════════════════════════════════════════════════════════════════

// ── 5. Happy path: returns created sessions ─────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_happy_path_returns_created_sessions() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // Create two sessions on the same path.
    let base_path = "/stage/list-test";
    handle_create_session(&svc, create_session_request(base_path, "session-a"))
        .await
        .expect("session-a creation must succeed");
    handle_create_session(&svc, create_session_request(base_path, "session-b"))
        .await
        .expect("session-b creation must succeed");

    let response = handle_list_sessions(&svc, list_sessions_request(base_path))
        .await
        .expect("list must succeed");

    let sessions = response.into_inner().sessions;
    assert_eq!(
        sessions.len(),
        2,
        "must return exactly 2 sessions, got: {}",
        sessions.len()
    );

    let names: Vec<&str> = sessions.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"session-a"), "must contain session-a");
    assert!(names.contains(&"session-b"), "must contain session-b");
}

// ── 6. Empty path → INVALID_ARGUMENT ────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_empty_path_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let status = handle_list_sessions(&svc, list_sessions_request(""))
        .await
        .expect_err("empty path must fail");

    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "empty path must return INVALID_ARGUMENT, got: {}",
        status.code()
    );
}

// ── 7. Auth denied → PERMISSION_DENIED ──────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_auth_denied_returns_permission_denied() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(DenyAllPermissions));

    let status = handle_list_sessions(&svc, list_sessions_request("/stage"))
        .await
        .expect_err("denied auth must fail");

    assert_eq!(
        status.code(),
        tonic::Code::PermissionDenied,
        "denied auth must return PERMISSION_DENIED, got: {}",
        status.code()
    );
}

// ── 8. No sessions for path → empty list ────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_no_sessions_returns_empty_list() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let response = handle_list_sessions(&svc, list_sessions_request("/nonexistent/path"))
        .await
        .expect("list on empty path must succeed");

    let sessions = response.into_inner().sessions;
    assert!(
        sessions.is_empty(),
        "must return empty list for path with no sessions, got: {} sessions",
        sessions.len()
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// acquire_session_cancel: the token is cancelled only when the session is gone
// ═══════════════════════════════════════════════════════════════════════════════
//
// Cancelling a session token ends every stream sharing it — by the cancel's
// cause: an explicit deletion ends a `SubscribeReadLayer` `NOT_FOUND` (not
// retried) and a `JoinSession` `OK`; a session found missing from the live
// store with no deletion marker ends both `UNAVAILABLE`, which the client
// retries so the session can be recovered. Either way the streams end, so
// `acquire_session_cancel` removes and cancels the entry only on a `NotFound`
// that survives its one recovery pass, and on nothing else. A transient lookup
// error, or a caller whose future is dropped at an `.await` (a tonic deadline,
// a client disconnect), is no evidence the session is gone: the entry and its
// token are left alone, whoever inserted them, because another caller on the
// same session may already share the token. (An entry left for a session that
// is in fact gone is reaped by the `session_tokens` sweeper, which also
// cancels on `NotFound` only.)
//
// These tests pin:
//
//   1. A inserts, B shares the token, then A's lookup fails transiently → B's
//      token is not cancelled and the entry stays.
//   2. The same, with A's future dropped at its lookup.
//   3. A transient error against an entry another caller inserted leaves it.
//   4. Happy path → the entry remains, uncancelled.
//   5. `NotFound` → the entry is removed and its token cancelled — including a
//      token another caller already shares.
//   6. `NotFound` with the deletion marker unreadable → the entry is still
//      removed and cancelled, with the retryable cause (`AbsentFromStore`).
//
// The cause the marker picks, and the recovery pass itself, are pinned by
// `session_cancel_cause_test.rs` and by `session_cache.rs`'s
// `recovery_tests`.
//
// 1 and 2 need A's lookup to be in flight while B shares the token, so A's
// lookup is held behind a stalled data pool (`TestValkey::stall_data_pool`),
// while B looks the session up through a second replica's client that is not
// stalled. Under the stall, A's lookup fails at the fixture's command timeout
// with a real, transient error. 6 uses the same stall for the marker read.

/// Replica B's `SessionStore` over its own `ValkeyClient` and `tv`'s keyspace,
/// so B's lookup is not held by a stall on `tv`'s data pool.
#[cfg(feature = "test-fault-injection")]
async fn unstalled_session_store(tv: &TestValkey) -> SessionStore {
    let config = live_edit::config::Config {
        valkey_connect_timeout_secs: 30,
        ..crate::common::test_config()
    };
    let client = live_edit::storage::test_support::connect_test_client(&config)
        .await
        .expect("replica B's ValkeyClient must connect");
    SessionStore::new(client, tv.key_schema().clone())
}

/// Replica A's lookup, in flight behind a stalled data pool, with the entry it
/// inserted shared by B: the `session_tokens` map, the token B holds, A's
/// running call, and the stalls to release.
#[cfg(feature = "test-fault-injection")]
struct SharedInFlight {
    session_id: String,
    session_tokens: Arc<DashMap<String, SessionCancel>>,
    b_token: SessionCancel,
    a_call: tokio::task::JoinHandle<Result<SessionCancel, ServiceError>>,
    stalls: Vec<live_edit::storage::test_support::ConnStall>,
}

#[cfg(feature = "test-fault-injection")]
async fn a_inserts_and_b_shares_while_a_is_in_flight(tv: &TestValkey) -> SharedInFlight {
    /// Bound on waiting for A's insert. Unreachable on the healthy path.
    const SETTLE: std::time::Duration = std::time::Duration::from_secs(10);

    let store_a = Arc::new(SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let session_id = store_a
        .create("shared-token-session", "/stage", "test-user")
        .await
        .expect("create must succeed");
    let store_b = unstalled_session_store(tv).await;
    let session_tokens: Arc<DashMap<String, SessionCancel>> = Arc::new(DashMap::new());

    let stalls = tv.stall_data_pool().await;
    let a_call = {
        let session_tokens = Arc::clone(&session_tokens);
        let store_a = Arc::clone(&store_a);
        let session_id = session_id.clone();
        tokio::spawn(async move {
            acquire_session_cancel(
                &session_tokens,
                &store_a,
                &SessionBasePathCache::new(),
                std::time::Duration::from_secs(10),
                &session_id,
            )
            .await
        })
    };
    let deadline = tokio::time::Instant::now() + SETTLE;
    while !session_tokens.contains_key(&session_id) {
        assert!(
            tokio::time::Instant::now() < deadline,
            "A never inserted its entry"
        );
        tokio::task::yield_now().await;
    }
    assert!(
        !a_call.is_finished(),
        "fixture self-check: A's lookup must still be in flight behind the stall"
    );

    let b_token = acquire_session_cancel(
        &session_tokens,
        &store_b,
        &SessionBasePathCache::new(),
        std::time::Duration::from_secs(10),
        &session_id,
    )
    .await
    .expect("B's lookup, on an unstalled client, must succeed");

    SharedInFlight {
        session_id,
        session_tokens,
        b_token,
        a_call,
        stalls,
    }
}

/// Test 1: A's lookup fails transiently after B shares its token. The error
/// is surfaced to A, and B's token is neither cancelled nor unmapped.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acquire_session_cancel_transient_error_leaves_a_token_another_caller_shares() {
    let tv = TestValkey::setup().await;
    let shared = a_inserts_and_b_shares_while_a_is_in_flight(&tv).await;

    let a_result = tokio::time::timeout(std::time::Duration::from_secs(20), shared.a_call)
        .await
        .expect("A's lookup must fail at its command timeout")
        .expect("A's call must not panic");
    release_all(shared.stalls).await;

    assert!(
        matches!(&a_result, Err(e) if !matches!(e, ServiceError::NotFound(_))),
        "precondition: A's lookup failed transiently, not NotFound; got {a_result:?}"
    );
    assert!(
        !shared.b_token.is_cancelled(),
        "A's transient failure cancelled the token B shares — every stream on B's session \
         would end as if the session were gone"
    );
    let mapped = shared
        .session_tokens
        .get(&shared.session_id)
        .expect("the entry must stay in the map after a transient failure")
        .clone();
    mapped.cancel();
    assert!(
        shared.b_token.is_cancelled(),
        "the entry left in the map must be the very token B holds"
    );
}

/// Test 2: A's future is dropped at its lookup after B shares its token. B's
/// token is neither cancelled nor unmapped.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acquire_session_cancel_dropped_future_leaves_a_token_another_caller_shares() {
    let tv = TestValkey::setup().await;
    let shared = a_inserts_and_b_shares_while_a_is_in_flight(&tv).await;

    shared.a_call.abort();
    let joined = tokio::time::timeout(std::time::Duration::from_secs(10), shared.a_call)
        .await
        .expect("an aborted call resolves promptly");
    release_all(shared.stalls).await;

    assert!(
        joined.is_err_and(|e| e.is_cancelled()),
        "fixture self-check: A's call was dropped mid-lookup, not completed"
    );
    assert!(
        !shared.b_token.is_cancelled(),
        "dropping A's lookup cancelled the token B shares"
    );
    assert!(
        shared.session_tokens.contains_key(&shared.session_id),
        "dropping A's lookup removed the entry B's token is registered under"
    );
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acquire_session_cancel_preexisting_entry_not_removed_on_error() {
    // Another caller already owns a `session_tokens` entry for this
    // session_id. When the injected Internal error fires, our call must NOT
    // remove that entry or cancel its token — otherwise we'd end someone
    // else's in-flight streams as if the session were gone.
    let tv = TestValkey::setup().await;

    let fault_queue: Arc<Mutex<VecDeque<ServiceError>>> =
        Arc::new(Mutex::new(VecDeque::from(vec![ServiceError::Internal(
            "simulated transient Valkey blip".to_owned(),
        )])));

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_get_base_path_fault_injector_for_test(fault_queue.clone());

    let session_tokens: DashMap<String, SessionCancel> = DashMap::new();
    let pre_existing_token = CancellationToken::new();
    let sid = "shared-session-id";
    session_tokens.insert(sid.to_owned(), pre_existing_token.clone().into());

    let result = acquire_session_cancel(
        &session_tokens,
        &session_store,
        &SessionBasePathCache::new(),
        std::time::Duration::from_secs(10),
        sid,
    )
    .await;
    assert!(
        matches!(result, Err(ServiceError::Internal(_))),
        "expected Internal, got: {result:?}"
    );

    // The pre-existing entry must still be there: a failed lookup is no
    // evidence the session is gone.
    assert!(
        session_tokens.contains_key(sid),
        "pre-existing entry must not be removed by our failed call"
    );
    assert!(
        !pre_existing_token.is_cancelled(),
        "pre-existing token must not be cancelled by our failed call; \
         cancellation is the responsibility of the owner"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acquire_session_cancel_happy_path_keeps_entry() {
    // Baseline: a healthy session means `get_base_path` succeeds and the
    // entry persists, uncancelled, for the lifetime of the streaming RPC.
    let tv = TestValkey::setup().await;
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let sid = session_store
        .create("live-session", "/stage", "test-user")
        .await
        .expect("create must succeed");

    let session_tokens: DashMap<String, SessionCancel> = DashMap::new();

    let token = acquire_session_cancel(
        &session_tokens,
        &session_store,
        &SessionBasePathCache::new(),
        std::time::Duration::from_secs(10),
        &sid,
    )
    .await
    .expect("happy path must succeed");

    assert!(
        session_tokens.contains_key(&sid),
        "entry must persist after Ok return"
    );
    assert!(
        !token.is_cancelled(),
        "returned token must not be cancelled on happy path"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acquire_session_cancel_not_found_still_removes_entry() {
    // `NotFound` is the one outcome that means the session is gone, and the
    // entry this call inserted is removed.
    let tv = TestValkey::setup().await;
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());

    let session_tokens: DashMap<String, SessionCancel> = DashMap::new();
    let sid = "never-existed";

    let result = acquire_session_cancel(
        &session_tokens,
        &session_store,
        &SessionBasePathCache::new(),
        std::time::Duration::from_secs(10),
        sid,
    )
    .await;
    assert!(
        matches!(result, Err(ServiceError::NotFound(_))),
        "expected NotFound, got: {result:?}"
    );

    assert!(
        !session_tokens.contains_key(sid),
        "NotFound branch must still remove the inserted entry"
    );
}

/// Test 5: `NotFound` removes and cancels the entry **whoever inserted it** —
/// a token another caller already shares included. Every stream sharing it is
/// serving a session that is missing from the live store. The session carries
/// no deletion marker, so the cancel's cause is `AbsentFromStore`: the streams
/// end retryably, and their retries find out whether a checkpoint can restore
/// the session.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acquire_session_cancel_not_found_removes_and_cancels_a_token_another_caller_shares() {
    let tv = TestValkey::setup().await;
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_tokens: DashMap<String, SessionCancel> = DashMap::new();
    let sid = "deleted-while-shared";
    let shared = SessionCancel::new();
    session_tokens.insert(sid.to_owned(), shared.clone());

    let result = acquire_session_cancel(
        &session_tokens,
        &session_store,
        &SessionBasePathCache::new(),
        std::time::Duration::from_secs(10),
        sid,
    )
    .await;
    assert!(
        matches!(result, Err(ServiceError::NotFound(_))),
        "expected NotFound, got: {result:?}"
    );
    assert!(
        !session_tokens.contains_key(sid),
        "NotFound must remove the entry even though another caller inserted it"
    );
    assert!(
        shared.is_cancelled(),
        "NotFound must cancel the token other streams share: the session is gone"
    );
    assert_eq!(
        shared.cause(),
        SessionCancelCause::AbsentFromStore,
        "an unmarked session missing from the store is cancelled with the retryable cause, not \
         as a deletion"
    );
}

/// Test 6: the deletion-marker read that picks the cause fails, and the cancel
/// happens anyway, with the retryable cause.
///
/// The session was deleted (`SessionStore::delete` wrote the marker), so a
/// successful read would say `Deleted`. The marker read is held behind a
/// stalled data pool and fails at the fixture's command timeout; the store's
/// lookups are answered `NotFound` by the fault injector, so the stall reaches
/// only the marker read. `AbsentFromStore` on a failed read is the documented
/// fallback: a retry costs a deleted session's client one refused setup, while
/// a deletion ending on a restorable session would lose it.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acquire_session_cancel_marker_read_failure_still_cancels_as_absent_from_store() {
    let tv = TestValkey::setup().await;
    let writer = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let sid = writer
        .create("marker-read-fails", "/stage", "test-user")
        .await
        .expect("create must succeed");
    writer.delete(&sid).await.expect("delete must succeed");
    assert!(
        writer
            .deleted_marker_exists(&sid)
            .await
            .expect("marker read must succeed before the stall"),
        "precondition: the deleted session carries the deletion marker"
    );

    // The re-check and the recovery pass's lookup (no hook) each answer
    // NotFound without touching Valkey.
    let fault_queue: Arc<Mutex<VecDeque<ServiceError>>> =
        Arc::new(Mutex::new(VecDeque::from(vec![
            ServiceError::NotFound(format!("session {sid} not found")),
            ServiceError::NotFound(format!("session {sid} not found")),
        ])));
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_get_base_path_fault_injector_for_test(Arc::clone(&fault_queue));
    let session_tokens: DashMap<String, SessionCancel> = DashMap::new();
    let shared = SessionCancel::new();
    session_tokens.insert(sid.clone(), shared.clone());

    let stalls = tv.stall_data_pool().await;
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(20),
        acquire_session_cancel(
            &session_tokens,
            &session_store,
            &SessionBasePathCache::new(),
            std::time::Duration::from_secs(10),
            &sid,
        ),
    )
    .await
    .expect("the marker read must fail at its command timeout, not hang");
    release_all(stalls).await;

    assert!(
        fault_queue.lock().unwrap().is_empty(),
        "fixture self-check: both lookups were answered by the injector"
    );
    assert!(
        matches!(result, Err(ServiceError::NotFound(_))),
        "expected NotFound, got: {result:?}"
    );
    assert!(
        !session_tokens.contains_key(&sid),
        "a failed marker read must not keep the entry: the session is missing either way"
    );
    assert!(
        shared.is_cancelled(),
        "a failed marker read must not skip the cancel"
    );
    assert_eq!(
        shared.cause(),
        SessionCancelCause::AbsentFromStore,
        "an unreadable marker falls back to the retryable cause"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SR-2.x: CreateSession through a real JwtUserinfoClient (b.g3z t1 Task 4)
// ═══════════════════════════════════════════════════════════════════════════════
//
// The JWT happy-path and opaque-bearer tests pin SR-5.4 (the userinfo-not-found
// test between them reaches the same claim chain through the fallback
// decorator): CreateSession must drive the full pipeline through
// a real `JwtUserinfoClient` for both the happy path (chosen claim flows through
// to `created_by_user_name`) and the opaque-bearer path (verbatim SR-2.4 error
// surfaces at the `AuthError` boundary; the wire stays security-redacted). The
// trust-delegation ordering invariant (permissions before userinfo) is implicitly
// exercised by `AllowAllPermissions` — the opaque-bearer rejection cannot fire
// until permissions has accepted.
//
// The verbatim SR-2.4 / SR-2.5 strings live in `super::helpers` as shared
// `pub const`s (`SR_2_4_OPAQUE_TOKEN_MSG`, `SR_2_5_MISSING_CLAIM_MSG`) — drift
// here is a Task 2 contract violation, and lifting them to one place means a
// reword surfaces as a single fix rather than silent no-ops in sibling tests.

// ── JWT happy path: name claim flows through to created_by_user_name ────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_jwt_happy_path_name_claim_flows_to_created_by_user_name() {
    let tv = TestValkey::setup().await;
    let userinfo = Arc::new(live_edit::auth::JwtUserinfoClient::new());
    let svc = build_service_with_userinfo(&tv, Arc::new(AllowAllPermissions), userinfo);

    let base_path = "/stage/jwt-happy";
    let session_name = "ada-session";
    let jwt = make_jwt(&serde_json::json!({"name": "Ada Lovelace"}));

    // CreateSession with a JWT bearer token carrying `name = "Ada Lovelace"`.
    let mut create_req = Request::new(CreateSessionRequest {
        path: base_path.to_owned(),
        name: session_name.to_owned(),
    });
    create_req
        .extensions_mut()
        .insert(BearerToken::new(jwt.as_str()));

    handle_create_session(&svc, create_req)
        .await
        .expect("create_session must succeed for a well-formed JWT");

    // ListSessions reads `created_by_user_name` back out of the session record.
    let mut list_req = Request::new(ListSessionsRequest {
        path: base_path.to_owned(),
    });
    list_req
        .extensions_mut()
        .insert(BearerToken::new(jwt.as_str()));

    let response = handle_list_sessions(&svc, list_req)
        .await
        .expect("list_sessions must succeed");

    let sessions = response.into_inner().sessions;
    assert_eq!(sessions.len(), 1, "must return exactly one session");
    assert_eq!(
        sessions[0].created_by_user_name, "Ada Lovelace",
        "JWT `name` claim must flow through to `created_by_user_name`, got: {:?}",
        sessions[0].created_by_user_name
    );
}

// ── Userinfo "no such user": the creator's name comes from the JWT claims ───
//
// With the userinfo service and authorization both on, a caller the userinfo
// directory does not hold (an app identity, say) used to fail CreateSession
// `INTERNAL`. The userinfo client `build_userinfo_client` builds for that
// deployment answers from the bearer's JWT claims instead; the claim label
// must reach `created_by_user_name`, for an app-only token (` (app)` label)
// and for a user token with a `name` claim alike.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_userinfo_not_found_uses_jwt_claim_label_for_created_by_user_name() {
    let tv = TestValkey::setup().await;
    let (userinfo, _userinfo_server) = userinfo_answering_not_found().await;
    let svc = build_service_with_userinfo(&tv, Arc::new(AllowAllPermissions), userinfo);

    let base_path = "/stage/userinfo-not-found";
    let app_jwt = unlisted_app_jwt("Asset Agent");
    let user_jwt = make_jwt(&serde_json::json!({
        "name": "Ada Lovelace",
        "oid": "99999999-8888-7777-6666-555555555555",
        "sub": "ada-pairwise-sub",
    }));

    for (session_name, jwt) in [("app-session", &app_jwt), ("user-session", &user_jwt)] {
        let mut create_req = Request::new(CreateSessionRequest {
            path: base_path.to_owned(),
            name: session_name.to_owned(),
        });
        create_req
            .extensions_mut()
            .insert(BearerToken::new(jwt.as_str()));
        handle_create_session(&svc, create_req)
            .await
            .unwrap_or_else(|status| {
                panic!("{session_name}: create_session must succeed for an authorized caller the userinfo directory does not hold, got {status:?}")
            });
    }

    let mut list_req = Request::new(ListSessionsRequest {
        path: base_path.to_owned(),
    });
    list_req
        .extensions_mut()
        .insert(BearerToken::new(app_jwt.as_str()));
    let sessions = handle_list_sessions(&svc, list_req)
        .await
        .expect("list_sessions must succeed")
        .into_inner()
        .sessions;

    let created_by = |name: &str| -> String {
        sessions
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("session {name} must be listed; got {sessions:?}"))
            .created_by_user_name
            .clone()
    };
    assert_eq!(created_by("app-session"), "Asset Agent (app)");
    assert_eq!(created_by("user-session"), "Ada Lovelace");
}

// ── Opaque-bearer rejection: redacted wire + verbatim SR-2.4 at AuthError ───

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_opaque_bearer_returns_internal_with_sr_2_4_message() {
    // Opaque-bearer rejection: a non-JWT bearer string must surface as
    // `INTERNAL` at the CreateSession wire boundary.
    //
    // Status-message contract — important. SR-2.4 pins the verbatim wording
    // on the `AuthError::InvalidResponse(msg)` boundary that the production
    // `JwtUserinfoClient::get_current_user` returns; that string flows
    // through `ServiceError::Internal(format!("auth backend: {e}"))` and
    // lands in the operator's `tracing::error!(error = %msg, ...)` log. The
    // `ServiceError → tonic::Status` impl deliberately redacts the wire
    // message to a generic `"internal error"` (`src/error.rs` line ~206) so
    // backend detail does not leak to clients. This is a security contract,
    // not a bug — see the `ServiceError::Internal` arm and its
    // `internal_status_does_not_leak_message` regression test.
    //
    // The verbatim-wording check therefore lives in two places:
    //   1. `src/auth/userinfo_jwt.rs::tests` pins the `AuthError` boundary.
    //   2. Below we pin `SR_2_4_OPAQUE_TOKEN_MSG` as a `const` referenced
    //      from this file, so a Task 2 reword surfaces as a compile-time
    //      drift signal even though the wire never sees the string.
    //
    // What we CAN assert from the integration-test position: the redacted
    // wire shape (`INTERNAL` + `"internal error"`). Both halves of the
    // contract are exercised; the verbatim-wire-string check the Subtask
    // body asks for is structurally impossible past the security redaction
    // and is left to the unit-level pin (mirrors the sibling reconciliation
    // in `create_layer_test.rs`).
    let tv = TestValkey::setup().await;
    let userinfo = Arc::new(live_edit::auth::JwtUserinfoClient::new());
    let svc = build_service_with_userinfo(&tv, Arc::new(AllowAllPermissions), userinfo);

    let mut request = Request::new(CreateSessionRequest {
        path: "/stage/opaque".to_owned(),
        name: "opaque-session".to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("opaque-token-not-a-jwt"));

    let status = handle_create_session(&svc, request)
        .await
        .expect_err("opaque bearer token must be rejected by JwtUserinfoClient");

    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "opaque bearer must surface as INTERNAL, got: {} — msg: {}",
        status.code(),
        status.message(),
    );
    assert_eq!(
        status.message(),
        "internal error",
        "wire-side message must remain the security-redacted placeholder; \
         backend detail (incl. SR-2.4 wording {SR_2_4_OPAQUE_TOKEN_MSG:?}) \
         must never reach the client. Got: {:?}",
        status.message(),
    );

    // Cross-file pin: also exercise the production decoder directly with
    // the same opaque bearer to confirm the underlying `AuthError`
    // carries the verbatim SR-2.4 wording. This is what makes
    // `SR_2_4_OPAQUE_TOKEN_MSG` load-bearing — a reword in
    // `JwtUserinfoClient::get_current_user` surfaces here even though the
    // gRPC wire path redacts the string.
    let decoder = live_edit::auth::JwtUserinfoClient::new();
    let auth_err = decoder
        .get_current_user("opaque-token-not-a-jwt")
        .await
        .expect_err("opaque bearer must return AuthError at the decoder boundary");
    match auth_err {
        live_edit::auth::AuthError::InvalidResponse(msg) => {
            assert_eq!(
                msg, SR_2_4_OPAQUE_TOKEN_MSG,
                "JwtUserinfoClient must return the verbatim SR-2.4 wording at the \
                 AuthError boundary — Task 2 contract. Got: {msg:?}",
            );
        }
        other => panic!("expected AuthError::InvalidResponse with SR-2.4 wording, got {other:?}"),
    }
}

// ── Missing-claim rejection: redacted wire + verbatim SR-2.5 at AuthError ───

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_jwt_missing_display_name_claim_returns_internal_with_sr25_message() {
    // Missing-display-name-claim rejection (PRD AC-4 / SR-2.5): a
    // structurally-valid 3-segment JWT whose payload omits every
    // fallback-chain claim (`name` / `preferred_username` / `sub` / `oid`)
    // must surface as `INTERNAL` at the CreateSession wire boundary.
    //
    // Status-message contract — important. SR-2.5 pins the verbatim wording
    // on the `AuthError::InvalidResponse(msg)` boundary that the production
    // `JwtUserinfoClient::get_current_user` returns; that string flows
    // through `ServiceError::Internal(format!("auth backend: {e}"))` and
    // lands in the operator's `tracing::error!(error = %msg, ...)` log. The
    // `ServiceError → tonic::Status` impl deliberately redacts the wire
    // message to a generic `"internal error"` (`src/error.rs` line ~206) so
    // backend detail does not leak to clients — same security contract that
    // governs the sibling SR-2.4 opaque-bearer test above (b.ohw).
    //
    // The verbatim-wording check therefore lives in two places:
    //   1. `src/auth/userinfo_jwt.rs::tests` pins the `AuthError` boundary
    //      at the decoder.
    //   2. Below we pin `SR_2_5_MISSING_CLAIM_MSG` as a `const` referenced
    //      from this file, so a Task 2 reword surfaces as a test failure
    //      even though the wire never sees the string. Mirrors the
    //      reconciliation pattern the sibling Subtasks
    //      `t3.g3z.zq.yw.ss / .h3 / .yo` already shipped for SR-2.4.
    //
    // `AllowAllPermissions` ensures the permissions check (which runs first)
    // does not short-circuit — the missing-claim rejection cannot fire
    // until permissions has accepted, which is the trust-delegation
    // ordering invariant under test.
    let tv = TestValkey::setup().await;
    let userinfo = Arc::new(live_edit::auth::JwtUserinfoClient::new());
    let svc = build_service_with_userinfo(&tv, Arc::new(AllowAllPermissions), userinfo);

    // Payload contains only `iat`, a field NOT in the fallback chain. The
    // JWT is structurally valid (3 segments, base64url-decodable, parses as
    // JSON), passes every JWT-shape check, then falls through every rung of
    // `name → preferred_username → sub → oid` and emits SR-2.5 at the
    // decoder.
    let jwt = make_jwt(&serde_json::json!({"iat": 1700000000}));

    let mut request = Request::new(CreateSessionRequest {
        path: "/stage/missing-claim".to_owned(),
        name: "missing-claim-session".to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new(jwt.as_str()));

    let status = handle_create_session(&svc, request)
        .await
        .expect_err("JWT with no display-name claim must be rejected by JwtUserinfoClient");

    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "missing-claim JWT must surface as INTERNAL, got: {} — msg: {}",
        status.code(),
        status.message(),
    );
    assert_eq!(
        status.message(),
        "internal error",
        "wire-side message must remain the security-redacted placeholder; \
         backend detail (incl. SR-2.5 wording {SR_2_5_MISSING_CLAIM_MSG:?}) \
         must never reach the client. Got: {:?}",
        status.message(),
    );

    // Cross-file pin: also exercise the production decoder directly with
    // the same missing-claim JWT to confirm the underlying `AuthError`
    // carries the verbatim SR-2.5 wording. This is what makes
    // `SR_2_5_MISSING_CLAIM_MSG` load-bearing — a reword in
    // `JwtUserinfoClient::get_current_user` surfaces here even though the
    // gRPC wire path redacts the string.
    let decoder = live_edit::auth::JwtUserinfoClient::new();
    let auth_err = decoder
        .get_current_user(jwt.as_str())
        .await
        .expect_err("missing-claim JWT must return AuthError at the decoder boundary");
    match auth_err {
        live_edit::auth::AuthError::InvalidResponse(msg) => {
            assert_eq!(
                msg, SR_2_5_MISSING_CLAIM_MSG,
                "JwtUserinfoClient must return the verbatim SR-2.5 wording at the \
                 AuthError boundary — Task 2 contract. Got: {msg:?}",
            );
        }
        other => panic!("expected AuthError::InvalidResponse with SR-2.5 wording, got {other:?}"),
    }
}

// ── 9. Session fields are populated correctly ───────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_response_contains_expected_fields() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let base_path = "/stage/fields-test";
    handle_create_session(&svc, create_session_request(base_path, "field-session"))
        .await
        .expect("session creation must succeed");

    let response = handle_list_sessions(&svc, list_sessions_request(base_path))
        .await
        .expect("list must succeed");

    let sessions = response.into_inner().sessions;
    assert_eq!(sessions.len(), 1);

    let session = &sessions[0];
    assert!(
        !session.session_id.is_empty(),
        "session_id must be non-empty"
    );
    assert_eq!(session.name, "field-session");
    assert!(
        !session.created_by_user_name.is_empty(),
        "created_by_user_name must be non-empty"
    );
    assert!(session.created_at > 0, "created_at must be non-zero");
}

// ═══════════════════════════════════════════════════════════════════════════════
// b.uoc — base-path canonicalization across CreateSession / ListSessions
// ═══════════════════════════════════════════════════════════════════════════════
//
// The bug: `CreateSession` and `ListSessions` used the caller's `path` string
// byte-for-byte as the identity of a session's base path — hex-encoded verbatim
// into `le:path:{hex(base_path)}` and `le:path:{hex(base_path)}:names`, stored
// verbatim in `session_meta.base_path`, and forwarded verbatim to the
// permission probe. Kit sends `.../Jdoe@example.com/...` (literal `@`); every
// ordinary URL encoder — `encodeURIComponent`, Python's `quote`, Postman's
// "URL encode" — sends `.../Jdoe%40example.com/...`. Two RFC 3986-equivalent
// spellings therefore keyed two unrelated Valkey key families, so a Postman
// caller's `ListSessions` returned an **empty list** for a path Kit had live
// sessions on — no error, indistinguishable from "no sessions". Observed in dev
// on 2026-09-08.
//
// User-email folders are the standard path convention in this deployment, so
// `@` is on essentially every real path. These tests drive the two wire-taking
// RPCs end-to-end against real Valkey and assert the two spellings are one
// identity everywhere it matters: the path index, the composite-name
// uniqueness index, the stored `session_meta.base_path`, and the permission
// probe.

/// The two spellings used throughout this section. Same storage object, and
/// after canonicalization the same string.
const AT_SPELLING: &str = "/main/Playground/Jdoe@example.com";
const PCT40_SPELLING: &str = "/main/Playground/Jdoe%40example.com";
/// What both must canonicalize to — the storage API's own strict form.
const CANONICAL: &str = "/main/Playground/Jdoe%40example.com";

// ── Cross-spelling visibility (the reported symptom, both directions) ────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_percent_40_spelling_finds_session_created_with_at_spelling() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // Kit's spelling on the way in...
    handle_create_session(&svc, create_session_request(AT_SPELLING, "kit-session"))
        .await
        .expect("create with the `@` spelling must succeed");

    // ...an ordinary URL encoder's spelling on the way out.
    let sessions = handle_list_sessions(&svc, list_sessions_request(PCT40_SPELLING))
        .await
        .expect("list with the `%40` spelling must succeed")
        .into_inner()
        .sessions;

    let names: Vec<&str> = sessions.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["kit-session"],
        "a session created under {AT_SPELLING:?} must be visible when listed \
         under the equivalent {PCT40_SPELLING:?}; pre-b.uoc this returned an \
         empty list with no error"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_at_spelling_finds_session_created_with_percent_40_spelling() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    handle_create_session(
        &svc,
        create_session_request(PCT40_SPELLING, "postman-session"),
    )
    .await
    .expect("create with the `%40` spelling must succeed");

    let sessions = handle_list_sessions(&svc, list_sessions_request(AT_SPELLING))
        .await
        .expect("list with the `@` spelling must succeed")
        .into_inner()
        .sessions;

    let names: Vec<&str> = sessions.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["postman-session"],
        "the converse direction must hold too — canonicalization is symmetric, \
         not a one-way accommodation of Kit's encoder"
    );
}

/// Sessions created under either spelling land in one set, not two — the
/// property a caller actually observes when Kit and a script share a file.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_returns_sessions_created_under_both_spellings_as_one_set() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    handle_create_session(&svc, create_session_request(AT_SPELLING, "from-kit"))
        .await
        .expect("create with the `@` spelling must succeed");
    handle_create_session(&svc, create_session_request(PCT40_SPELLING, "from-script"))
        .await
        .expect("create with the `%40` spelling must succeed");

    for spelling in [AT_SPELLING, PCT40_SPELLING] {
        let sessions = handle_list_sessions(&svc, list_sessions_request(spelling))
            .await
            .unwrap_or_else(|e| panic!("list under {spelling:?} must succeed, got {e:?}"))
            .into_inner()
            .sessions;

        let mut names: Vec<&str> = sessions.iter().map(|s| s.name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            vec!["from-kit", "from-script"],
            "both sessions must be visible under {spelling:?} — one path, one \
             session set, whichever spelling asks"
        );
    }
}

// ── Composite-name uniqueness holds across spellings ────────────────────────

/// The SDD calls `le:path:{hex(base_path)}:names` the authoritative
/// composite-name uniqueness index. Pre-b.uoc the same `(path, name)` pair
/// could be created twice under two spellings, defeating it outright.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_duplicate_name_across_spellings_returns_already_exists() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    handle_create_session(&svc, create_session_request(AT_SPELLING, "dup-name"))
        .await
        .expect("the first create must succeed");

    let status = handle_create_session(&svc, create_session_request(PCT40_SPELLING, "dup-name"))
        .await
        .expect_err("the same name under an equivalent spelling must be rejected");

    assert_eq!(
        status.code(),
        tonic::Code::AlreadyExists,
        "duplicate-name detection must span RFC 3986-equivalent spellings, \
         got {}: {}",
        status.code(),
        status.message()
    );
}

// ── The stored spelling is the canonical one ────────────────────────────────

/// `session_meta.base_path` is what every *other* RPC reads back (via
/// `SessionBasePathCache`) and what the session checkpoint persists, so the
/// canonical form has to be what lands there — canonicalizing only the lookup
/// would leave the stored value carrying caller spelling forever.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_stores_the_canonical_base_path_whichever_spelling_arrived() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());

    for (spelling, name) in [(AT_SPELLING, "stored-a"), (PCT40_SPELLING, "stored-b")] {
        let session_id = handle_create_session(&svc, create_session_request(spelling, name))
            .await
            .unwrap_or_else(|e| panic!("create under {spelling:?} must succeed, got {e:?}"))
            .into_inner()
            .session_id;

        let stored = session_store
            .get_base_path(&session_id)
            .await
            .expect("the created session's base path must be readable");

        assert_eq!(
            stored, CANONICAL,
            "the base path stored for a session created under {spelling:?} must \
             be the canonical form"
        );
    }
}

// ── The permission probe sees the canonical path ────────────────────────────

/// The canonical form — not the caller's spelling — must reach the permissions
/// service. It is the very form that service stores and matches policies in
/// (`encode_resource_url`), so the decision is identical either way; sending
/// one form is what keeps live-edit's auth-cache keys and its logs consistent
/// with its Valkey keys.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_probes_permission_with_the_canonical_path() {
    let tv = TestValkey::setup().await;
    let permissions = Arc::new(RecordingPermissions::allowing());
    let svc = build_service(&tv, permissions.clone());

    handle_create_session(&svc, create_session_request(AT_SPELLING, "probe-create"))
        .await
        .expect("create must succeed");

    let calls = permissions.calls();
    assert_eq!(
        calls,
        vec![(CANONICAL.to_owned(), live_edit::auth::Permission::Write)],
        "CreateSession must probe write permission exactly once, on the \
         canonical path — not on the caller's {AT_SPELLING:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_probes_permission_with_the_canonical_path() {
    let tv = TestValkey::setup().await;
    let permissions = Arc::new(RecordingPermissions::allowing());
    let svc = build_service(&tv, permissions.clone());

    handle_list_sessions(&svc, list_sessions_request(AT_SPELLING))
        .await
        .expect("list must succeed");

    let calls = permissions.calls();
    assert_eq!(
        calls,
        vec![(CANONICAL.to_owned(), live_edit::auth::Permission::Read)],
        "ListSessions must probe read permission exactly once, on the canonical \
         path — not on the caller's {AT_SPELLING:?}"
    );
}

// ── Rejections reach the wire as INVALID_ARGUMENT ───────────────────────────

/// A `..` segment is rejected by the canonicalizer, so it must surface as
/// `INVALID_ARGUMENT` on both wire-taking RPCs rather than reaching the key
/// schema. `..%2Fkey` is the interesting spelling: an encoded slash would
/// smuggle the traversal past any per-segment check that ran before decoding.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_dot_segment_path_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    for path in [
        "/main/../secret",
        "/main/..%2Fsecret",
        "/main/%2E%2E%2Fsecret",
    ] {
        let status = handle_create_session(&svc, create_session_request(path, "traversal"))
            .await
            .err()
            .unwrap_or_else(|| panic!("{path:?} must be rejected by CreateSession"));

        assert_eq!(
            status.code(),
            tonic::Code::InvalidArgument,
            "{path:?} must return INVALID_ARGUMENT, got {}: {}",
            status.code(),
            status.message()
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_dot_segment_path_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    for path in [
        "/main/../secret",
        "/main/..%2Fsecret",
        "/main/%2E%2E%2Fsecret",
    ] {
        let status = handle_list_sessions(&svc, list_sessions_request(path))
            .await
            .err()
            .unwrap_or_else(|| panic!("{path:?} must be rejected by ListSessions"));

        assert_eq!(
            status.code(),
            tonic::Code::InvalidArgument,
            "{path:?} must return INVALID_ARGUMENT, got {}: {}",
            status.code(),
            status.message()
        );
    }
}

/// Invalid UTF-8 in a percent-escape is likewise a wire-level
/// `INVALID_ARGUMENT`, not a 500 and not a silently mangled key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_invalid_utf8_escape_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let status = handle_create_session(&svc, create_session_request("/main/%FF", "bad-utf8"))
        .await
        .expect_err("an escape that does not decode as UTF-8 must be rejected");

    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "got {}: {}",
        status.code(),
        status.message()
    );
}

// ── Spellings that are NOT equivalent stay distinct ─────────────────────────

/// The counter-invariant to every convergence test above: canonicalization
/// must not *over*-merge. Two paths that decode differently are two identities,
/// and a session created under one must not appear under the other.
///
/// The near-miss spellings carry the weight here. A wholly different user
/// (`someone.else@…`) would stay distinct under almost any broken rule; an
/// empty segment and a trailing slash are the two shapes the canonicalizer has
/// a live opportunity to erase, because both are *accepted* — they are legal,
/// distinct blob names, deliberately neither collapsed nor trimmed — and both
/// differ from the base spelling by a single character.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_does_not_merge_genuinely_distinct_paths() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    handle_create_session(&svc, create_session_request(AT_SPELLING, "only-here"))
        .await
        .expect("create must succeed");

    for other in [
        // Same folder name but a different user — decodes differently.
        "/main/Playground/someone.else@example.com",
        // One empty segment: `a//b` is a legal, distinct blob name.
        "/main/Playground//Jdoe@example.com",
        // One trailing slash, likewise not trimmed.
        "/main/Playground/Jdoe@example.com/",
    ] {
        let sessions = handle_list_sessions(&svc, list_sessions_request(other))
            .await
            .unwrap_or_else(|e| panic!("list under {other:?} must succeed, got {e:?}"))
            .into_inner()
            .sessions;

        assert!(
            sessions.is_empty(),
            "{other:?} is a distinct path from {AT_SPELLING:?} and must not pick \
             up its sessions, got: {:?}",
            sessions.iter().map(|s| s.name.as_str()).collect::<Vec<_>>()
        );
    }
}

// ── Scheme / host case folds across the wire ────────────────────────────────
//
// `canonicalize_base_path` folds the scheme of every scheme and the host of a
// WHATWG special scheme only, matching what `Url::parse` → `to_string()` does
// upstream. The unit tests in `src/domain/base_path.rs` pin the string
// transform; these two drive it end-to-end through both wire-taking RPCs, so
// the fold is proven to reach the path index, the stored base path, and the
// permission probe rather than just the canonicalizer's return value.

/// A `https` URL differing only in scheme and host case is ONE identity
/// upstream, so a session created under one spelling must be listable under
/// the other — and the permissions service must see the folded form, since
/// that is the spelling it stores its own policies under.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_lowercase_url_finds_session_created_with_mixed_case_url() {
    const MIXED_CASE_URL: &str = "HTTPS://Host.Example.COM/Stage/Jdoe@example.com";
    const LOWERCASE_URL: &str = "https://host.example.com/Stage/Jdoe%40example.com";

    let tv = TestValkey::setup().await;
    let permissions = Arc::new(RecordingPermissions::allowing());
    let svc = build_service(&tv, permissions.clone());

    handle_create_session(&svc, create_session_request(MIXED_CASE_URL, "cross-case"))
        .await
        .expect("create with the mixed-case URL must succeed");

    let sessions = handle_list_sessions(&svc, list_sessions_request(LOWERCASE_URL))
        .await
        .expect("list with the lowercase URL must succeed")
        .into_inner()
        .sessions;

    let names: Vec<&str> = sessions.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["cross-case"],
        "a session created under {MIXED_CASE_URL:?} must be visible when listed \
         under the case-equivalent {LOWERCASE_URL:?}"
    );

    assert_eq!(
        permissions.calls(),
        vec![
            (LOWERCASE_URL.to_owned(), live_edit::auth::Permission::Write),
            (LOWERCASE_URL.to_owned(), live_edit::auth::Permission::Read),
        ],
        "both probes must carry the folded, canonical spelling — not the \
         caller's {MIXED_CASE_URL:?}"
    );
}

/// The counter-invariant, and the more dangerous direction. Under a
/// non-special scheme upstream's opaque-host parser preserves host case, so
/// `omniverse://Server/x` and `omniverse://server/x` are TWO identities.
/// Merging them here would hand one caller the session set of a path the
/// authorization probe cleared under a different spelling.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_keeps_non_special_scheme_host_case_as_distinct_paths() {
    const UPPER_HOST_URL: &str = "omniverse://Server/Stage/Jdoe@example.com";
    const LOWER_HOST_URL: &str = "omniverse://server/Stage/Jdoe@example.com";

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    handle_create_session(
        &svc,
        create_session_request(UPPER_HOST_URL, "upper-host-only"),
    )
    .await
    .expect("create under the upper-case host must succeed");

    let sessions = handle_list_sessions(&svc, list_sessions_request(LOWER_HOST_URL))
        .await
        .expect("list must succeed")
        .into_inner()
        .sessions;

    assert!(
        sessions.is_empty(),
        "{LOWER_HOST_URL:?} is a different upstream identity from \
         {UPPER_HOST_URL:?} — folding a non-special scheme's host would merge \
         two session sets, got: {:?}",
        sessions.iter().map(|s| s.name.as_str()).collect::<Vec<_>>()
    );

    // ...and the session is still there under its own spelling, so the empty
    // list above is a genuine miss rather than a create that never landed.
    let own = handle_list_sessions(&svc, list_sessions_request(UPPER_HOST_URL))
        .await
        .expect("list must succeed")
        .into_inner()
        .sessions;
    let names: Vec<&str> = own.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["upper-host-only"]);
}
