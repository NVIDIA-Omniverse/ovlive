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

// Fault-injection integration tests for `SessionStore` (b.91j).
//
// Gated behind `--features test-fault-injection`; the seams these tests drive
// (`with_delete_exec_fault_injector_for_test`,
// `with_create_fault_injector_for_test`) compile out of release builds.
//
// Covers:
//   * C-3: `delete` must surface `Aborted` (retriable) when the EXEC reply
//     arrives as `Value::Null` or as a `fred::error::Error::is_canceled()`.
//     Pre-fix both modes mapped to `Internal` (UNKNOWN/INTERNAL gRPC status)
//     and clients wouldn't retry, leaving orphan session_meta / layers_index.
//   * H-stor-3: `create`'s retry loop must honor the `with_retry_max_attempts`
//     override — N attempts, then `Aborted("... N attempts exhausted ...")`.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use fred::interfaces::{HashesInterface, SetsInterface};

use live_edit::error::ServiceError;
use live_edit::storage::session::{DeleteExecFault, SessionStore};
use live_edit::storage::test_support::TestValkey;

fn store(tv: &TestValkey) -> SessionStore {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
}

// ═══════════════════════════════════════════════════════════════════════════════
// C-3: delete EXEC null / canceled → Aborted with retryable marker
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_delete_connection_drop_mid_exec_surfaces_aborted() {
    // Primary C-3 regression: the `is_canceled()` branch of the EXEC match.
    // Pre-fix, a fred `Canceled` error (connection drop / oneshot-channel
    // close) was surfaced as `ServiceError::Internal` — clients got
    // UNKNOWN/INTERNAL and would not retry. The fix adds
    //
    //     Err(e) if e.is_canceled() => Err(Aborted("transaction canceled
    //         (connection drop), retry_on_valkey_drop"))
    //
    // which this test drives deterministically via the fault-injection seam.
    let tv = TestValkey::setup().await;

    let fault_queue: Arc<Mutex<VecDeque<DeleteExecFault>>> =
        Arc::new(Mutex::new(VecDeque::from(vec![
            DeleteExecFault::ReturnCanceled,
        ])));

    let ss = store(&tv).with_delete_exec_fault_injector_for_test(fault_queue.clone());

    // Create a real session so `delete` gets past the HGETALL short-circuit
    // and reaches the EXEC boundary where the fault fires.
    let sid = ss
        .create("canceled-exec-session", "/stage", "alice")
        .await
        .expect("create must succeed");

    let err = ss.delete(&sid).await.expect_err("delete must fail");

    match err {
        ServiceError::Aborted(msg) => {
            // Must include the retry marker so middleware / clients that
            // key on the message substring can find it.
            assert!(
                msg.contains("retry_on_valkey_drop"),
                "Aborted message must include retry_on_valkey_drop marker, got: {msg}"
            );
            assert!(
                msg.contains("canceled") || msg.contains("connection drop"),
                "message must mention the underlying cause, got: {msg}"
            );
        }
        other => panic!(
            "expected Aborted (which reaches the client as retriable gRPC UNAVAILABLE); \
             pre-fix this returned Internal, got: {other:?}"
        ),
    }

    // Queue was drained — exactly one EXEC was substituted.
    assert!(
        fault_queue.lock().unwrap().is_empty(),
        "fault queue must be drained after a single EXEC substitution"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_delete_exec_null_reply_surfaces_aborted() {
    // Companion to the canceled-arm test: the pre-b.91j code decoded EXEC
    // as `()` via `trx.exec::<()>(true)`, which coerces nil replies to
    // `Ok(())` — the caller silently believed the delete landed. The fix
    // decodes as `fred::types::Value` and branches on `is_null()`, treating
    // it as retriable. This test drives the null branch deterministically.
    let tv = TestValkey::setup().await;

    let fault_queue: Arc<Mutex<VecDeque<DeleteExecFault>>> =
        Arc::new(Mutex::new(VecDeque::from(vec![
            DeleteExecFault::ReturnNull,
        ])));

    let ss = store(&tv).with_delete_exec_fault_injector_for_test(fault_queue.clone());

    let sid = ss
        .create("null-exec-session", "/stage", "bob")
        .await
        .expect("create must succeed");

    let err = ss.delete(&sid).await.expect_err("delete must fail");

    match err {
        ServiceError::Aborted(msg) => {
            assert!(
                msg.contains("retry_on_valkey_drop"),
                "null-arm Aborted must also carry retry marker, got: {msg}"
            );
            assert!(
                msg.contains("null") || msg.contains("reply null"),
                "message must mention null reply, got: {msg}"
            );
        }
        other => panic!("expected Aborted on null EXEC reply, got: {other:?}"),
    }
}

/// The deletion marker is queued inside `delete`'s MULTI with the metadata
/// DEL, not written beside it: when the EXEC never runs (the injected
/// canceled and null replies both drop the queued transaction unsent), no
/// marker is left behind, and the session's metadata is still there.
///
/// A marker written outside the transaction would outlive a failed delete and
/// make recovery refuse a session that was never deleted — the case this pins.
/// The retry the client makes after the `Aborted` then writes both.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_delete_aborted_exec_leaves_no_deletion_marker() {
    use fred::interfaces::KeysInterface;

    let tv = TestValkey::setup().await;
    for fault in [DeleteExecFault::ReturnCanceled, DeleteExecFault::ReturnNull] {
        let fault_queue: Arc<Mutex<VecDeque<DeleteExecFault>>> =
            Arc::new(Mutex::new(VecDeque::from(vec![fault])));
        let ss = store(&tv).with_delete_exec_fault_injector_for_test(fault_queue);
        let sid = ss
            .create(&format!("marker-atomic-{fault:?}"), "/stage", "alice")
            .await
            .expect("create must succeed");

        let err = ss.delete(&sid).await.expect_err("delete must fail");
        assert!(
            matches!(err, ServiceError::Aborted(_)),
            "{fault:?}: expected Aborted, got {err:?}"
        );

        let marker: bool = tv
            .client()
            .pool
            .exists(tv.key_schema().session_deleted_marker(&sid))
            .await
            .expect("EXISTS must succeed");
        assert!(
            !marker,
            "{fault:?}: a delete whose transaction never executed must not leave the deletion \
             marker — it would make session recovery refuse a session that still exists"
        );
        let meta: HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(tv.key_schema().session_meta(&sid))
            .await
            .expect("HGETALL must succeed");
        assert!(
            !meta.is_empty(),
            "{fault:?}: fixture self-check: the metadata DEL in the same transaction did not run \
             either"
        );

        // The client's retry lands both halves together.
        ss.delete(&sid)
            .await
            .expect("the retried delete must succeed");
        let marker_after_retry: bool = tv
            .client()
            .pool
            .exists(tv.key_schema().session_deleted_marker(&sid))
            .await
            .expect("EXISTS must succeed");
        assert!(
            marker_after_retry,
            "{fault:?}: the retry must write the marker"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// H-stor-3: create retry budget is bounded by configured max_attempts
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_retry_under_watch_contention_does_not_exceed_configured_max() {
    // Pre-fix `SessionStore::create` hard-coded `MAX_TXN_RETRIES = 5` with
    // no jitter. The fix extracts the retry loop into
    // `retry_aborted_with_jitter` and accepts `LIVE_EDIT_SESSION_RETRY_MAX_ATTEMPTS`.
    // This test verifies the bound is honored: with `max_attempts = 3`
    // and a contention source that always returns `Aborted` (simulated via
    // the fault-injection seam so we don't depend on real concurrent
    // writers hitting WATCH), exactly 3 attempts run and the final error
    // names the exhaustion count.
    let tv = TestValkey::setup().await;

    const MAX: u32 = 3;

    // Queue 4 Aborted errors: the retry helper should consume the first 3
    // (one per attempt) and bail with an exhaustion error. The 4th stays
    // in the queue, confirming the loop did not run a 4th attempt.
    let fault_queue: Arc<Mutex<VecDeque<ServiceError>>> =
        Arc::new(Mutex::new(VecDeque::from(vec![
            ServiceError::Aborted("attempt-1-watch-conflict".to_owned()),
            ServiceError::Aborted("attempt-2-watch-conflict".to_owned()),
            ServiceError::Aborted("attempt-3-watch-conflict".to_owned()),
            ServiceError::Aborted("attempt-4-must-not-fire".to_owned()),
        ])));

    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_retry_max_attempts(MAX)
        .with_retry_initial_delay_ms(1)
        .with_create_fault_injector_for_test(fault_queue.clone());

    let err = ss
        .create("budget-cap", "/stage", "carol")
        .await
        .expect_err("create must fail once budget exhausts");

    // The helper wraps the last Aborted with the attempt count. The
    // "3 attempts exhausted" substring is the load-bearing assertion —
    // operators and log-alerting key on this shape.
    match &err {
        ServiceError::Aborted(msg) => {
            let expected_needle = format!("{MAX} attempts exhausted");
            assert!(
                msg.contains(&expected_needle),
                "message must contain '{expected_needle}', got: {msg}"
            );
            // Operation label must prefix the message so mixed retry loops
            // (update_layer vs create_session) stay distinguishable.
            assert!(
                msg.contains("create_session"),
                "message must name the op, got: {msg}"
            );
            // Last-cause chaining must preserve the terminal fault so
            // operators can diagnose without turning on debug logging.
            assert!(
                msg.contains("attempt-3-watch-conflict"),
                "message must name the final cause, got: {msg}"
            );
        }
        other => panic!("expected Aborted (exhaustion), got: {other:?}"),
    }

    // Queue has exactly one entry left (the "must-not-fire" guard) —
    // proving the loop stopped at the budget instead of continuing.
    let remaining = fault_queue.lock().unwrap().len();
    assert_eq!(
        remaining, 1,
        "expected exactly 1 unfired injection after {MAX} attempts; got {remaining} remaining"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_retry_budget_one_attempt_then_abort() {
    // Edge: with `max_attempts = 1` the loop makes exactly one attempt
    // and surfaces the Aborted without retrying. Pins the lower bound of
    // the new bounded-retry contract — a follow-up knob lowering the cap
    // to 1 must not accidentally still try 2.
    let tv = TestValkey::setup().await;

    let fault_queue: Arc<Mutex<VecDeque<ServiceError>>> =
        Arc::new(Mutex::new(VecDeque::from(vec![
            ServiceError::Aborted("first-and-only".to_owned()),
            ServiceError::Aborted("must-not-fire".to_owned()),
        ])));

    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_retry_max_attempts(1)
        .with_retry_initial_delay_ms(1)
        .with_create_fault_injector_for_test(fault_queue.clone());

    let err = ss
        .create("single-attempt", "/stage", "dan")
        .await
        .expect_err("create must fail");

    assert!(
        matches!(err, ServiceError::Aborted(ref m) if m.contains("1 attempts exhausted")),
        "expected Aborted with exhaustion at 1 attempt, got: {err:?}"
    );

    // Queue has one entry remaining.
    assert_eq!(
        fault_queue.lock().unwrap().len(),
        1,
        "loop must have made exactly 1 attempt"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_retry_non_aborted_error_bypasses_retry_loop() {
    // Regression guard: only `Aborted` is retriable. A non-Aborted error
    // (e.g. `Internal`) must short-circuit the retry helper — otherwise
    // the loop would burn the whole budget on a permanent failure.
    let tv = TestValkey::setup().await;

    let fault_queue: Arc<Mutex<VecDeque<ServiceError>>> =
        Arc::new(Mutex::new(VecDeque::from(vec![
            ServiceError::Internal("permanent-failure".to_owned()),
            ServiceError::Aborted("must-not-fire".to_owned()),
        ])));

    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_retry_max_attempts(32)
        .with_retry_initial_delay_ms(1)
        .with_create_fault_injector_for_test(fault_queue.clone());

    let err = ss
        .create("non-retriable", "/stage", "eve")
        .await
        .expect_err("create must fail");

    assert!(
        matches!(err, ServiceError::Internal(ref m) if m.contains("permanent-failure")),
        "non-Aborted must propagate unchanged, got: {err:?}"
    );

    assert_eq!(
        fault_queue.lock().unwrap().len(),
        1,
        "retry loop must have made exactly 1 attempt on non-Aborted error"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_create_eventually_succeeds_within_budget() {
    // Companion to the exhaustion test: the retry loop must succeed if
    // the injected Aborteds run out before the budget does. Injects 2
    // Aborteds with max_attempts=5 and empty-queue afterwards; the 3rd
    // attempt sees an empty queue, runs normally, and commits. Verifies
    // the retry helper doesn't leak state across attempts.
    let tv = TestValkey::setup().await;

    let fault_queue: Arc<Mutex<VecDeque<ServiceError>>> =
        Arc::new(Mutex::new(VecDeque::from(vec![
            ServiceError::Aborted("attempt-1".to_owned()),
            ServiceError::Aborted("attempt-2".to_owned()),
        ])));

    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_retry_max_attempts(5)
        .with_retry_initial_delay_ms(1)
        .with_create_fault_injector_for_test(fault_queue.clone());

    let sid = ss
        .create("eventual-success", "/stage", "frank")
        .await
        .expect("create must succeed after 2 simulated conflicts");

    // Session is fully committed — path_names_index contains the composite
    // member, session_meta has the name.
    let members: Vec<String> = tv
        .client()
        .pool
        .smembers(tv.key_schema().path_names_index("/stage"))
        .await
        .expect("SMEMBERS must succeed");
    assert_eq!(
        members.len(),
        1,
        "exactly one name-set member must exist after successful retry"
    );

    let meta: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().session_meta(&sid))
        .await
        .expect("HGETALL must succeed");
    assert_eq!(
        meta.get("name").map(String::as_str),
        Some("eventual-success"),
        "session_meta must reflect the committed name"
    );
}
