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

// Integration tests for the `session_tokens` crash-recovery sweep
// (b.j5t item 5 / b.f36 A6).
//
// The sweep periodically probes `le:{session_id}:sess` via
// `SessionStore::get_base_path` for every entry in the in-memory
// `session_tokens` DashMap, removing + cancelling any whose owning session
// is absent. Without this sweep, sessions whose clients crash (or simply
// never call `DeleteSession`) leave entries in `session_tokens` until
// the process exits — an unbounded slow leak on long-running pods. Each
// cancel records its cause from the session's deletion marker: `Deleted`
// when `DeleteSession` marked it, `AbsentFromStore` otherwise (including
// when the marker cannot be read).
//
// The sweeper also runs a *requested* pass, `revalidate_deleted_sessions`,
// when the session-deleted relay may have missed notices. That pass is
// deleted-only: it cancels, as `Deleted`, only entries whose session is
// missing *and* marked, and leaves an unmarked-missing entry, or one whose
// probe or marker read fails, to the periodic pass; the failures count as
// `deferred` in its `SweepOutcome`. The sweeper's schedule for those passes —
// a lag request at once, `REVALIDATION_MIN_GAP` between passes, a random delay
// before a reconnect's — is checked here on a paused clock. The requests
// themselves, from a relay lag and a pub/sub reconnect, are driven in
// `pubsub_relay_lag_test.rs`.
//
// These tests run against a real Valkey instance via `TestValkey`.

use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use futures::FutureExt;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use live_edit::grpc::session::{
    revalidate_deleted_sessions, spawn_session_tokens_sweeper, sweep_session_tokens, SessionCancel,
    SessionCancelCause, SessionRevalidation, SweepOutcome, REVALIDATION_MIN_GAP,
    REVALIDATION_RECONNECT_JITTER_MAX,
};
use live_edit::grpc::session_cache::SessionBasePathCache;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

/// A live session that exists in Valkey survives the sweep; a dead entry
/// that has no `session_meta` is removed and its token cancelled.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_removes_absent_session_keeps_live_session() {
    let tv = TestValkey::setup().await;
    let store = Arc::new(SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));

    // Create a real session via SessionStore so its meta lives in Valkey.
    let live_session_id = store
        .create("live-session", "/p", "tester")
        .await
        .expect("create live session");

    // Build the session_tokens map with one live entry + one orphan entry.
    let session_tokens: Arc<DashMap<String, SessionCancel>> = Arc::new(DashMap::new());
    let live_token = CancellationToken::new();
    let orphan_token = CancellationToken::new();
    session_tokens.insert(live_session_id.clone(), live_token.clone().into());
    session_tokens.insert(
        "orphan-session-id-never-existed".to_string(),
        orphan_token.clone().into(),
    );

    let base_path_cache = SessionBasePathCache::new();

    // Run a single sweep.
    let removed = sweep_session_tokens(&session_tokens, &store, &base_path_cache).await;

    assert_eq!(removed, 1, "exactly one orphan entry must be removed");
    assert!(
        session_tokens.contains_key(&live_session_id),
        "live session entry must survive the sweep"
    );
    assert!(
        !session_tokens.contains_key("orphan-session-id-never-existed"),
        "orphan entry must be removed"
    );
    assert!(
        !live_token.is_cancelled(),
        "live session's token must NOT be cancelled"
    );
    assert!(
        orphan_token.is_cancelled(),
        "orphan session's token MUST be cancelled so any subscribers exit promptly"
    );
}

/// The spawned sweeper task runs the sweep at its configured interval and
/// exits cleanly when its `CancellationToken` is fired.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawned_sweeper_runs_periodically_and_exits_on_cancel() {
    let tv = TestValkey::setup().await;
    let store = Arc::new(SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));

    let session_tokens: Arc<DashMap<String, SessionCancel>> = Arc::new(DashMap::new());
    let orphan_token = CancellationToken::new();
    session_tokens.insert("orphan-1".to_string(), orphan_token.clone().into());

    let base_path_cache = SessionBasePathCache::new();
    let cancel = CancellationToken::new();

    // 50ms interval — short enough for the test to complete fast, long
    // enough that one tick after spawn is unambiguous.
    let handle = spawn_session_tokens_sweeper(
        Arc::clone(&session_tokens),
        Arc::clone(&store),
        base_path_cache,
        Duration::from_millis(50),
        SessionRevalidation::new(),
        cancel.clone(),
    );

    // Wait for the sweep to fire by awaiting the side-effect it produces:
    // `sweep_session_tokens` removes the map entry and then cancels its token
    // on the way out, so the cancellation token is an event-driven signal
    // that the sweep has actually run end-to-end.
    //
    // This replaces the pre-fix 5 s clock-based poll, which flaked on
    // loaded CI runners when the 50 ms tokio interval slipped past the
    // budget. The 30 s timeout below is fallback only — on a healthy
    // runtime this resolves the instant the sweeper's first tick fires
    // (typically <150 ms), and only trips if the spawn machinery is
    // genuinely broken (interval not firing, select arm dead, sweep
    // hanging on Valkey).
    tokio::time::timeout(Duration::from_secs(30), orphan_token.cancelled())
        .await
        .expect(
            "background sweeper did not cancel the orphan token within 30 s — \
             either the spawn loop is broken or the runtime is starved",
        );

    // Ordering invariant: `sweep_session_tokens` does `map.remove()` before
    // cancelling the token. Once `cancelled()` returns, the map MUST be empty;
    // this assertion guards that ordering against future regressions.
    assert_eq!(
        session_tokens.len(),
        0,
        "map entry must be removed before the orphan token is cancelled"
    );

    // Fire shutdown and ensure the task exits.
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(2), handle)
        .await
        .expect("sweeper must exit within shutdown budget")
        .expect("sweeper task must not panic");
}

/// Delete `session_id`'s metadata hash directly, as a keyspace loss would:
/// the session is gone from the live store and, unlike a `DeleteSession`,
/// leaves no deletion marker behind.
async fn flush_session_meta(tv: &TestValkey, session_id: &str) {
    use fred::interfaces::KeysInterface;
    let removed: i64 = tv
        .client()
        .pool
        .del(tv.key_schema().session_meta(session_id))
        .await
        .expect("DEL session meta must succeed");
    assert_eq!(removed, 1, "fixture: the session's metadata existed");
}

/// The sweep picks each cancel's cause from the session's deletion marker: a
/// session `SessionStore::delete` removed (marked) is cancelled as a deletion,
/// one lost from the keyspace (no marker) as absent from the store. The cause
/// is what makes a `SubscribeReadLayer` end `NOT_FOUND` for the first and a
/// retryable `UNAVAILABLE` for the second (`session_cancel_cause_test.rs`
/// drives those endings through real streams).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_picks_deleted_for_a_marked_session_and_absent_for_an_unmarked_one() {
    let tv = TestValkey::setup().await;
    let store = Arc::new(SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let deleted_id = store
        .create("sweep-deleted", "/p", "tester")
        .await
        .expect("create the deleted session");
    store
        .delete(&deleted_id)
        .await
        .expect("delete must succeed");
    let flushed_id = store
        .create("sweep-flushed", "/p", "tester")
        .await
        .expect("create the flushed session");
    flush_session_meta(&tv, &flushed_id).await;

    let session_tokens: Arc<DashMap<String, SessionCancel>> = Arc::new(DashMap::new());
    let deleted_token = SessionCancel::new();
    let flushed_token = SessionCancel::new();
    session_tokens.insert(deleted_id.clone(), deleted_token.clone());
    session_tokens.insert(flushed_id.clone(), flushed_token.clone());

    let removed = sweep_session_tokens(&session_tokens, &store, &SessionBasePathCache::new()).await;

    assert_eq!(removed, 2, "both absent sessions' entries must be removed");
    assert!(session_tokens.is_empty());
    assert!(deleted_token.is_cancelled() && flushed_token.is_cancelled());
    assert_eq!(
        deleted_token.cause(),
        SessionCancelCause::Deleted,
        "a session carrying DeleteSession's marker is cancelled as a deletion, even though this \
         replica never saw the le:session:deleted fan-out"
    );
    assert_eq!(
        flushed_token.cause(),
        SessionCancelCause::AbsentFromStore,
        "an unmarked session missing from the store is cancelled retryably, so a checkpoint can \
         restore it on the retry"
    );
}

/// The sweep's marker read fails, and the sweep cancels anyway, with the
/// retryable cause: a failed read is no licence to leave a stale entry, and no
/// evidence of a deletion either.
///
/// The session was deleted, so a successful read would say `Deleted`. The
/// store's lookup is answered `NotFound` by the fault injector, and the marker
/// read — the only command that reaches Valkey — is held behind a stalled data
/// pool until it fails at the fixture's command timeout.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_marker_read_failure_still_cancels_as_absent_from_store() {
    use live_edit::error::ServiceError;
    use live_edit::storage::test_support::release_all;

    let tv = TestValkey::setup().await;
    let writer = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let sid = writer
        .create("sweep-marker-fails", "/p", "tester")
        .await
        .expect("create must succeed");
    writer.delete(&sid).await.expect("delete must succeed");

    let injector_queue = Arc::new(std::sync::Mutex::new(std::collections::VecDeque::from(
        vec![ServiceError::NotFound(format!("session {sid} not found"))],
    )));
    let store = Arc::new(
        SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .with_get_base_path_fault_injector_for_test(Arc::clone(&injector_queue)),
    );
    let session_tokens: Arc<DashMap<String, SessionCancel>> = Arc::new(DashMap::new());
    let token = SessionCancel::new();
    session_tokens.insert(sid.clone(), token.clone());

    let stalls = tv.stall_data_pool().await;
    let removed = tokio::time::timeout(
        Duration::from_secs(20),
        sweep_session_tokens(&session_tokens, &store, &SessionBasePathCache::new()),
    )
    .await
    .expect("the marker read must fail at its command timeout, not hang");
    release_all(stalls).await;

    assert!(
        injector_queue.lock().unwrap().is_empty(),
        "fixture self-check: the lookup was answered by the injector"
    );
    assert_eq!(removed, 1, "a failed marker read must not skip the removal");
    assert!(!session_tokens.contains_key(&sid));
    assert!(token.is_cancelled(), "nor the cancel");
    assert_eq!(
        token.cause(),
        SessionCancelCause::AbsentFromStore,
        "an unreadable marker falls back to the retryable cause"
    );
}

/// Transient Valkey errors (returned via the
/// `get_base_path_fault_injector_for_test` seam) must NOT cause the sweep
/// to mass-evict — only an unambiguous `NotFound` removes an entry. This
/// guards against a regression where a Valkey blip surfaces as e.g.
/// `Unavailable` and the sweeper interprets that as "session absent"
/// and cancels every live session's streams.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_transient_valkey_error_leaves_entry_alone() {
    use live_edit::error::ServiceError;

    let tv = TestValkey::setup().await;
    // Build a store with one queued transient error so the first probe in
    // the sweep returns Unavailable.
    let injector_queue = Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new()));
    injector_queue
        .lock()
        .unwrap()
        .push_back(ServiceError::Unavailable("valkey blip".to_owned()));
    let store = Arc::new(
        SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .with_get_base_path_fault_injector_for_test(Arc::clone(&injector_queue)),
    );

    let session_tokens: Arc<DashMap<String, SessionCancel>> = Arc::new(DashMap::new());
    let token = CancellationToken::new();
    session_tokens.insert("sid-1".to_string(), token.clone().into());

    let base_path_cache = SessionBasePathCache::new();
    let removed = sweep_session_tokens(&session_tokens, &store, &base_path_cache).await;

    assert_eq!(
        removed, 0,
        "transient Valkey error must NOT cause mass-eviction"
    );
    assert!(session_tokens.contains_key("sid-1"));
    assert!(
        !token.is_cancelled(),
        "transient probe failures must not cancel live tokens"
    );
}

// ── The requested, deleted-only pass ─────────────────────────────────────────

/// A requested pass cancels only the session `DeleteSession` marked, as a
/// deletion; the live session and the one missing without a marker (a keyspace
/// loss) keep their entries and their tokens. The periodic pass that follows
/// then cancels the unmarked one as absent from the store, so the requested
/// pass delays it to the periodic cadence and never loses it.
///
/// The scope matters because a requested pass runs on every replica at about
/// the same moment after a store failover; cancelling unmarked sessions there
/// would make the whole fleet end every session of a store that lost its
/// keyspace at once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn requested_pass_cancels_only_the_marked_session_and_leaves_the_rest_to_the_periodic_pass() {
    let tv = TestValkey::setup().await;
    let store = Arc::new(SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let live_id = store
        .create("requested-live", "/p", "tester")
        .await
        .expect("create the live session");
    let marked_id = store
        .create("requested-marked", "/p", "tester")
        .await
        .expect("create the marked session");
    store.delete(&marked_id).await.expect("delete must succeed");
    let unmarked_id = store
        .create("requested-unmarked", "/p", "tester")
        .await
        .expect("create the unmarked session");
    flush_session_meta(&tv, &unmarked_id).await;

    let session_tokens: Arc<DashMap<String, SessionCancel>> = Arc::new(DashMap::new());
    let live_token = SessionCancel::new();
    let marked_token = SessionCancel::new();
    let unmarked_token = SessionCancel::new();
    session_tokens.insert(live_id.clone(), live_token.clone());
    session_tokens.insert(marked_id.clone(), marked_token.clone());
    session_tokens.insert(unmarked_id.clone(), unmarked_token.clone());
    let base_path_cache = SessionBasePathCache::new();

    let outcome = revalidate_deleted_sessions(&session_tokens, &store, &base_path_cache).await;

    assert_eq!(
        outcome,
        SweepOutcome {
            probed: 3,
            cancelled: 1,
            deferred: 0,
        },
        "a requested pass probes every entry and cancels only the marked session"
    );
    assert!(
        marked_token.is_cancelled() && !session_tokens.contains_key(&marked_id),
        "the marked session's entry is removed and its token cancelled"
    );
    assert_eq!(
        marked_token.cause(),
        SessionCancelCause::Deleted,
        "a marked session is cancelled as a deletion"
    );
    assert!(
        !unmarked_token.is_cancelled() && session_tokens.contains_key(&unmarked_id),
        "a session missing without a marker is left to the periodic pass"
    );
    assert!(
        !live_token.is_cancelled() && session_tokens.contains_key(&live_id),
        "a live session is left alone"
    );

    let removed = sweep_session_tokens(&session_tokens, &store, &base_path_cache).await;

    assert_eq!(
        removed, 1,
        "the periodic pass cancels the unmarked session the requested pass left"
    );
    assert!(unmarked_token.is_cancelled() && !session_tokens.contains_key(&unmarked_id));
    assert_eq!(
        unmarked_token.cause(),
        SessionCancelCause::AbsentFromStore,
        "the periodic pass cancels an unmarked session retryably"
    );
    assert!(
        !live_token.is_cancelled() && session_tokens.contains_key(&live_id),
        "a live session survives the periodic pass too"
    );
}

/// A requested pass whose probe fails leaves the entry, counts it as
/// `deferred`, and cancels nothing, even for a session `DeleteSession` marked:
/// a store blip must not end streams, and the periodic pass retries the probe.
///
/// The probe is answered `Unavailable` by the fault injector; the session is
/// really deleted, so a probe that reached the store would cancel it.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn requested_pass_defers_an_entry_whose_probe_fails() {
    use live_edit::error::ServiceError;

    let tv = TestValkey::setup().await;
    let writer = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let sid = writer
        .create("requested-probe-fails", "/p", "tester")
        .await
        .expect("create must succeed");
    writer.delete(&sid).await.expect("delete must succeed");

    let injector_queue = Arc::new(std::sync::Mutex::new(std::collections::VecDeque::from(
        vec![ServiceError::Unavailable("valkey blip".to_owned())],
    )));
    let store = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_get_base_path_fault_injector_for_test(Arc::clone(&injector_queue));
    let session_tokens: Arc<DashMap<String, SessionCancel>> = Arc::new(DashMap::new());
    let token = SessionCancel::new();
    session_tokens.insert(sid.clone(), token.clone());

    let outcome =
        revalidate_deleted_sessions(&session_tokens, &store, &SessionBasePathCache::new()).await;

    assert!(
        injector_queue.lock().unwrap().is_empty(),
        "fixture self-check: the probe was answered by the injector"
    );
    assert_eq!(
        outcome,
        SweepOutcome {
            probed: 1,
            cancelled: 0,
            deferred: 1,
        },
        "a failed probe is deferred, not cancelled"
    );
    assert!(session_tokens.contains_key(&sid));
    assert!(
        !token.is_cancelled(),
        "a failed probe must not cancel the session's streams"
    );
}

/// A requested pass that finds a session missing but cannot read its deletion
/// marker leaves the entry and counts it as `deferred`: the pass cancels only
/// what it knows was deleted. The periodic pass that follows reads the marker
/// and cancels the session as a deletion. (The periodic pass alone, with the
/// same failed read, cancels retryably instead:
/// `sweep_marker_read_failure_still_cancels_as_absent_from_store`.)
///
/// The store's lookup is answered `NotFound` by the fault injector, and the
/// marker read — the only command that reaches Valkey — is held behind a
/// stalled data pool until it fails at the fixture's command timeout.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn requested_pass_defers_a_missing_session_whose_marker_read_fails() {
    use live_edit::error::ServiceError;
    use live_edit::storage::test_support::release_all;

    let tv = TestValkey::setup().await;
    let writer = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let sid = writer
        .create("requested-marker-fails", "/p", "tester")
        .await
        .expect("create must succeed");
    writer.delete(&sid).await.expect("delete must succeed");

    let injector_queue = Arc::new(std::sync::Mutex::new(std::collections::VecDeque::from(
        vec![ServiceError::NotFound(format!("session {sid} not found"))],
    )));
    let store = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_get_base_path_fault_injector_for_test(Arc::clone(&injector_queue));
    let session_tokens: Arc<DashMap<String, SessionCancel>> = Arc::new(DashMap::new());
    let token = SessionCancel::new();
    session_tokens.insert(sid.clone(), token.clone());
    let base_path_cache = SessionBasePathCache::new();

    let stalls = tv.stall_data_pool().await;
    let outcome = tokio::time::timeout(
        Duration::from_secs(20),
        revalidate_deleted_sessions(&session_tokens, &store, &base_path_cache),
    )
    .await
    .expect("the marker read must fail at its command timeout, not hang");
    release_all(stalls).await;

    assert!(
        injector_queue.lock().unwrap().is_empty(),
        "fixture self-check: the lookup was answered by the injector"
    );
    assert_eq!(
        outcome,
        SweepOutcome {
            probed: 1,
            cancelled: 0,
            deferred: 1,
        },
        "a missing session whose marker cannot be read is deferred, not cancelled"
    );
    assert!(session_tokens.contains_key(&sid));
    assert!(
        !token.is_cancelled(),
        "an unreadable marker must not cancel the session's streams in a requested pass"
    );

    let removed = sweep_session_tokens(&session_tokens, &store, &base_path_cache).await;

    assert_eq!(removed, 1, "the periodic pass cancels the deferred entry");
    assert!(token.is_cancelled() && !session_tokens.contains_key(&sid));
    assert_eq!(
        token.cause(),
        SessionCancelCause::Deleted,
        "with the marker readable again, the periodic pass cancels it as a deletion"
    );
}

// ── Scheduling the requested pass ────────────────────────────────────────────

/// Yields enough times for the sweeper, woken by a request or a timer, to run
/// until it waits again. A yield never moves a paused clock, so the clock
/// moves only through the test's `advance` calls.
async fn let_the_sweeper_run() {
    for _ in 0..16 {
        tokio::task::yield_now().await;
    }
}

/// Whether a requested pass is still pending on `revalidation`, that is, not
/// yet taken by the sweeper.
fn pending(revalidation: &SessionRevalidation) -> bool {
    revalidation.requested().now_or_never().is_some()
}

/// The sweeper's schedule for requested passes, on a paused clock that only
/// the test moves. A lag request is taken at once; a second request waits out
/// `REVALIDATION_MIN_GAP` after the first pass and is kept, not lost, while it
/// waits; a reconnect request on a production handle waits a random delay of
/// at most `REVALIDATION_RECONNECT_JITTER_MAX` before its pass; and cancelling
/// the sweeper during that delay ends it without waiting for the delay.
///
/// The `session_tokens` map is empty, so a pass reads nothing from the store;
/// the sweeper runs on a paused current-thread runtime of its own, with its
/// periodic pass an hour away. Each step checks the clock has moved exactly as
/// far as the test advanced it: a runtime that went idle would move it to the
/// sweeper's next timer, and that would let a pass run, or the sweeper exit,
/// at a time the test did not choose.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn requested_pass_schedule_takes_a_lag_at_once_keeps_the_gap_jitters_a_reconnect_and_cancels_in_the_delay(
) {
    let tv = TestValkey::setup().await;
    let store = Arc::new(SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let session_tokens: Arc<DashMap<String, SessionCancel>> = Arc::new(DashMap::new());
    let paused = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .start_paused(true)
        .build()
        .expect("building the paused runtime must succeed");

    tokio::task::spawn_blocking(move || {
        paused.block_on(async move {
            let revalidation = SessionRevalidation::new();
            let cancel = CancellationToken::new();
            let handle = spawn_session_tokens_sweeper(
                session_tokens,
                store,
                SessionBasePathCache::new(),
                Duration::from_secs(3600),
                revalidation.clone(),
                cancel.clone(),
            );
            let start = Instant::now();
            let mut advanced = Duration::ZERO;
            let_the_sweeper_run().await;

            // A lag request: taken with no delay.
            revalidation.request();
            assert!(pending(&revalidation), "fixture: the request is pending");
            tokio::time::advance(Duration::ZERO).await;
            let_the_sweeper_run().await;
            assert!(
                !pending(&revalidation),
                "a lag request's pass starts at once, with no delay"
            );
            assert_eq!(start.elapsed(), advanced, "the clock moved by itself");

            // A second request, inside the gap after that pass: kept until
            // the gap ends, then taken.
            revalidation.request();
            let short_of_the_gap = REVALIDATION_MIN_GAP - Duration::from_millis(1);
            tokio::time::advance(short_of_the_gap).await;
            advanced += short_of_the_gap;
            let_the_sweeper_run().await;
            assert!(
                pending(&revalidation),
                "a request made inside the gap must wait for the gap to end"
            );
            assert_eq!(start.elapsed(), advanced, "the clock moved by itself");
            tokio::time::advance(Duration::from_millis(2)).await;
            advanced += Duration::from_millis(2);
            let_the_sweeper_run().await;
            assert!(
                !pending(&revalidation),
                "a request kept through the gap is taken once the gap ends"
            );
            assert_eq!(start.elapsed(), advanced, "the clock moved by itself");

            // Past that pass's gap, a reconnect request waits its random delay.
            // (A millisecond past it: the timer wheel rounds a deadline up to
            // the next millisecond.)
            let past_the_gap = REVALIDATION_MIN_GAP + Duration::from_millis(1);
            tokio::time::advance(past_the_gap).await;
            advanced += past_the_gap;
            let_the_sweeper_run().await;
            revalidation.request_after_reconnect();
            let_the_sweeper_run().await;
            assert!(
                pending(&revalidation),
                "a reconnect request's pass waits its random delay"
            );
            assert_eq!(start.elapsed(), advanced, "the clock moved by itself");
            let past_the_jitter = REVALIDATION_RECONNECT_JITTER_MAX + Duration::from_millis(1);
            tokio::time::advance(past_the_jitter).await;
            advanced += past_the_jitter;
            let_the_sweeper_run().await;
            assert!(
                !pending(&revalidation),
                "a reconnect request's pass starts within REVALIDATION_RECONNECT_JITTER_MAX"
            );
            assert_eq!(start.elapsed(), advanced, "the clock moved by itself");

            // Another reconnect request; once the gap ends the sweeper is in
            // the delay, and cancelling it there ends it with the clock still.
            revalidation.request_after_reconnect();
            tokio::time::advance(past_the_gap).await;
            advanced += past_the_gap;
            let_the_sweeper_run().await;
            assert!(
                pending(&revalidation),
                "fixture: the sweeper is in the reconnect delay, its pass not yet started"
            );
            assert!(!handle.is_finished(), "fixture: the sweeper is running");
            cancel.cancel();
            let_the_sweeper_run().await;
            assert!(
                handle.is_finished(),
                "a sweeper cancelled during the reconnect delay must exit without waiting it out"
            );
            assert_eq!(start.elapsed(), advanced, "the clock moved by itself");
            handle.await.expect("the sweeper task must not panic");
        });
    })
    .await
    .expect("the paused-clock scenario must not panic");
}
