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

// Regression tests for b.w29 — presence sweeper crash-recovery + shutdown
// PeerLeft cleanup.
//
// The Engineer's diff splits the fix across four items:
//
//   Item 1 — Fleet-wide `le:presence:active_sessions` set enumerated by the
//            sweeper, so sessions whose only member lived on a crashed
//            replica still get reaped. Pre-fix the sweeper iterated the
//            local `LocalPresenceRouter::session_ids()`, which is empty on
//            a replica that never serviced a JoinSession for the dead
//            session.
//   Item 2 — Leader-gated sweep against `le:presence:sweeper:leader`.
//            Pre-fix every replica swept every tick, multiplying fleet-wide
//            presence load by N. The lock is intentionally distinct from
//            `le:gc:leader` so the GC + presence sweeps can land on
//            different replicas.
//   Item 3 — TTL/refresh slack: defaults raised to `TTL=60 / refresh=15`,
//            new env vars `LIVE_EDIT_PRESENCE_LEASE_TTL_SECS` and
//            `LIVE_EDIT_PRESENCE_REFRESH_INTERVAL_SECS`, validated with
//            `refresh × 2 ≤ ttl`. The unit-level config check lives in
//            `src/config.rs` tests; this file does not duplicate it.
//   Item 4 — Shutdown awaits every in-flight JoinSession task, and its
//            presence teardown, via a shared `TaskTracker` before Valkey
//            `quit()`, so teardown's PeerLeft reaches peers via pub/sub.
//
// All tests in this file talk to a live Valkey (the project convention is
// panic-on-no-Valkey via `TestValkey::setup()`; CI provisions one). The
// previous Engineer-stalled containerised verification cycle is documented
// at the test-run boundary; runtime verification happens in CI.

use std::sync::Arc;
use std::time::{Duration, Instant};

use fred::interfaces::{
    ClientLike, HashesInterface, KeysInterface, PubsubInterface, SetsInterface,
};
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use live_edit::presence::{
    spawn_presence_subscriber, spawn_presence_sweeper, PresenceEvent, SweeperBudget,
};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;

// The JoinSession stream helpers (`open_join`, `expect_session_connected`), the
// external presence-channel observers (`spawn_external_channel_subscriber` with
// `wait_for_peer_left`, and `Tap`) and `poll_until` live in `super::helpers`.

// ── AC1 — Crash-replica reap: PeerLeft within sweeper_interval + lease_ttl ──

/// **AC1 (b.w29 item 1).** Simulate a replica that crashed mid-JoinSession:
/// register the presence rows AND `le:presence:active_sessions` membership
/// in Valkey, but never SET the lease key (analogous to a lease that
/// already expired between crash and sweep). A separate "surviving"
/// replica's sweeper enumerates `presence:active_sessions`, finds the
/// stale entry, and publishes PeerLeft via Valkey pub/sub. An independent
/// subscriber must receive that PeerLeft within the worst-case budget
/// (`sweeper_interval` + `lease_ttl`).
///
/// Pre-fix the sweeper iterated `router.session_ids()` (a local-only
/// view): a crashed replica's session was never enumerated and PeerLeft
/// never fired. This test, run against the pre-fix sweeper, would
/// timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweeper_reaps_crashed_replica_session_publishes_peer_left() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema().clone();

    let session_id = "crash-recovery-session";
    let stale_connection_id = "conn-crashed-A";

    // ── Simulate crashed replica A's pre-crash state ────────────────────
    // SADD presence members + presence:active_sessions but DO NOT SET the
    // lease key — modelling the post-crash state after the lease has
    // already expired. The sweeper's "lease missing ⇒ evict" rule must
    // fire.
    let members_key = ks.presence_members(session_id);
    let user_key = ks.presence_user(session_id, stale_connection_id);
    let active_sessions_key = ks.presence_active_sessions();

    tv.client()
        .pool
        .next()
        .sadd::<(), _, _>(&members_key, stale_connection_id)
        .await
        .expect("SADD presence members must succeed");
    tv.client()
        .pool
        .next()
        .hset::<(), _, _>(
            &user_key,
            [
                ("user_name", "Crashed User"),
                ("user_application", "test-app"),
            ],
        )
        .await
        .expect("HSET presence user must succeed");
    tv.client()
        .pool
        .next()
        .sadd::<(), _, _>(&active_sessions_key, session_id)
        .await
        .expect("SADD presence:active_sessions must succeed");

    // ── Subscribe a fresh external observer to the presence channel ─────
    // We do this BEFORE the sweep so we cannot miss the PeerLeft publish.
    let (mut events_rx, observer_handle, observer_subscriber) =
        spawn_external_channel_subscriber(&tv, session_id).await;

    // ── Start replica B's sweeper with a short interval ─────────────────
    let cancel = CancellationToken::new();
    let sweep_interval = Duration::from_millis(100);
    let sweeper_handle = spawn_presence_sweeper(
        tv.client().clone(),
        ks.clone(),
        crate::common::test_leader("replica-B"),
        SweeperBudget::production(),
        Arc::new(crate::common::metrics_scrape_only("replica-B")),
        sweep_interval,
        cancel.clone(),
    );

    // ── Assert PeerLeft within budget ──────────────────────────────────
    // Worst case: one full sweep interval before the first tick + lease
    // already missing → reap immediately. Add a generous 2s margin for
    // CI flake — the assertion is "within the budget", not "exactly at".
    let observation_deadline = sweep_interval + Duration::from_secs(2);
    let got = wait_for_peer_left(&mut events_rx, stale_connection_id, observation_deadline).await;
    assert!(
        got,
        "AC1: presence sweeper must publish PeerLeft for the crashed connection \
         within sweep_interval ({:?}) + slack; before b.w29 the sweeper iterated \
         router.session_ids() (empty on surviving replicas) and the stale row \
         was never reaped",
        sweep_interval,
    );

    // ── Sanity: the stale presence row is gone from Valkey ──────────────
    let still_member: bool = tv
        .client()
        .pool
        .next()
        .sismember(&members_key, stale_connection_id)
        .await
        .expect("SISMEMBER must succeed");
    assert!(
        !still_member,
        "sweeper must SREM the stale member after evicting it"
    );

    // ── Teardown ────────────────────────────────────────────────────────
    cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), sweeper_handle).await;
    let _ = observer_subscriber
        .unsubscribe(ks.presence_events_channel(session_id))
        .await;
    let _ = observer_subscriber.quit().await;
    let _ = tokio::time::timeout(Duration::from_secs(2), observer_handle).await;
}

// ── AC2 — Leader-gated sweep: only one replica runs per tick ────────────────

/// `live_edit_presence_sweep_duration_seconds{outcome="completed"}` passes
/// recorded on `metrics`; 0 before the first.
fn completed_passes(metrics: &live_edit::observability::metrics::Metrics) -> f64 {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    sample_value(
        &render(metrics),
        "live_edit_presence_sweep_duration_seconds_count",
        &[("outcome", "completed")],
    )
    .unwrap_or(0.0)
}

/// Whether `metrics` has published `live_edit_presence_sweep_oldest_stale_age_seconds`:
/// a sweeper does so on every tick that completed a pass or lost the leader
/// probe.
fn has_ticked(metrics: &live_edit::observability::metrics::Metrics) -> bool {
    use live_edit::observability::metrics::test_util::{has_metric, render};
    has_metric(
        &render(metrics),
        "live_edit_presence_sweep_oldest_stale_age_seconds",
    )
}

/// **AC2 (b.w29 item 2).** Two presence sweepers race against the same
/// Valkey. Only one holds `le:presence:sweeper:leader` at a time, so only that
/// replica's ticks run a pass. The stale connection is removed, and its
/// PeerLeft published, exactly once — by the one sweep script execution that
/// removed it — and never again: neither the leader's later passes, which find
/// nothing left to do, nor the other replica's ticks, which lose the leader
/// probe, publish another. Pre-b.w29 every replica swept every tick, and the
/// same stale connection produced one PeerLeft publish per replica per sweep
/// window.
///
/// The lock is distinct from `le:gc:leader` — verified by asserting the two
/// key names differ.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_sweepers_racing_remove_and_announce_a_stale_member_once() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema().clone();

    let session_id = "two-sweeper-session";
    let stale_connection_id = "conn-crashed-X";

    // Seed a stale presence row: a member with no lease and no info hash.
    let members_key = ks.presence_members(session_id);
    let active_sessions_key = ks.presence_active_sessions();
    tv.client()
        .pool
        .next()
        .sadd::<(), _, _>(&members_key, stale_connection_id)
        .await
        .expect("SADD members must succeed");
    tv.client()
        .pool
        .next()
        .sadd::<(), _, _>(&active_sessions_key, session_id)
        .await
        .expect("SADD active_sessions must succeed");

    // Observe the pub/sub channel for the entire test window so we can
    // count PeerLeft publishes.
    let mut tap = Tap::open(&tv, session_id).await;

    // Spawn two sweepers with distinct leader identities and a short interval.
    let cancel = CancellationToken::new();
    let sweep_interval = Duration::from_millis(100);

    let leader_a = crate::common::test_leader("replica-A");
    let leader_b = crate::common::test_leader("replica-B");
    let metrics_a = Arc::new(crate::common::metrics_scrape_only("replica-A"));
    let metrics_b = Arc::new(crate::common::metrics_scrape_only("replica-B"));
    let handle_a = spawn_presence_sweeper(
        tv.client().clone(),
        ks.clone(),
        leader_a.clone(),
        SweeperBudget::production(),
        Arc::clone(&metrics_a),
        sweep_interval,
        cancel.clone(),
    );
    let handle_b = spawn_presence_sweeper(
        tv.client().clone(),
        ks.clone(),
        leader_b.clone(),
        SweeperBudget::production(),
        Arc::clone(&metrics_b),
        sweep_interval,
        cancel.clone(),
    );

    // Wait for the first PeerLeft to land (proves at least one replica
    // got the lock and swept).
    let until_first = tap
        .until("the stale member's PeerLeft", |e| {
            is_left(e, stale_connection_id)
        })
        .await;
    assert!(
        until_first
            .iter()
            .all(|e| matches!(e, PresenceEvent::PeerLeft { .. })),
        "only the sweep's PeerLeft may be published; got {until_first:?}"
    );

    // Inspect the leader lock — only one replica is recorded. The value is
    // the winner's `LeaderIdentity::lock_value()` (`label:instance`): b.w29
    // reuses LEADER_ACQUIRE_OR_REFRESH_SCRIPT, which stores the lock value
    // the acquiring process passed.
    let lock_holder: Option<String> = tv
        .client()
        .pool
        .next()
        .get(&ks.presence_sweeper_leader_lock())
        .await
        .expect("GET presence sweeper lock must succeed");
    let holder = lock_holder.expect("presence sweeper lock must be held after first tick");
    assert!(
        holder == leader_a.lock_value() || holder == leader_b.lock_value(),
        "lock holder must be exactly one of the two replicas' lock values, got {holder}"
    );

    // The presence-sweeper lock must be DISTINCT from the GC lock — both
    // can coexist (b.w29: deliberately separate so the two maintenance
    // loops land on different replicas in steady state). The GC lock may
    // be unset in this test (no GC task spawned), but the *key names*
    // are distinct strings.
    assert_ne!(
        ks.presence_sweeper_leader_lock(),
        ks.gc_leader_lock(),
        "AC2: presence-sweeper lock key must be distinct from gc lock key"
    );

    // After the removal the stale row is gone from the member set and the
    // emptied session from the index, so later ticks find nothing to do. Let
    // both sweepers tick on: until the leader has completed three more passes
    // and each replica has ticked at least once (the non-leader's tick loses
    // the probe), so both the leader's empty passes and the non-leader's skip
    // branch have run.
    let (a, b) = (&metrics_a, &metrics_b);
    poll_until(
        "three more passes, and a tick of each replica",
        PRESENCE_WAIT,
        move || async move {
            completed_passes(a) + completed_passes(b) >= 4.0 && has_ticked(a) && has_ticked(b)
        },
    )
    .await;

    // Everything published since the first PeerLeft, proven complete by a
    // fence. A real regression would be further publishes for the same
    // connection from either replica.
    let later = tap.drain(&tv).await;
    let extra_peer_lefts = later
        .iter()
        .filter(|e| is_left(e, stale_connection_id))
        .count();
    assert_eq!(
        extra_peer_lefts, 0,
        "AC2 leader-gate: once removed, the stale connection {stale_connection_id} must never \
         be announced again, by either replica; got {later:?}"
    );

    // Teardown.
    cancel.cancel();
    for h in [handle_a, handle_b] {
        let _ = tokio::time::timeout(Duration::from_secs(5), h).await;
    }
    tap.close().await;
}

// ── AC4 — Shutdown drains every join's teardown; peers see PeerLeft ─────────

/// **AC4 (b.w29 item 4).** Drive a single JoinSession into a real service,
/// then run the shutdown sequence's two stream steps as `main.rs` does:
///
///   1. Cancel the service shutdown token (`GracefulShutdown::cancel`),
///      which ends every enrolled stream at once — what
///      `end_streams_after_drain`'s paced ending reaches, batch by batch,
///      once the readiness drain is over, and what `main.rs` does outright
///      once the server has returned. Shutdown never cancels a per-session
///      token: that token means the session is gone.
///   2. `drain_stream_tasks`, which closes the `presence_cleanup_tracker`
///      (and the subscription tracker) and waits for it under the per-task
///      shutdown budget.
///
/// A separate external pub/sub subscriber observes the presence-events
/// channel for that session and MUST receive PeerLeft once the drain has
/// returned — i.e. before "Valkey quit would be issued" in production. The
/// join itself ends `UNAVAILABLE "server shutting down"`, the ending a client
/// retries (where the retry lands is the ingress's choice). Pre-b.w29 the
/// shutdown dropped these
/// `run_join_session` futures mid-cleanup and PeerLeft never fired.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shutdown_tracker_drain_waits_for_join_teardown_so_peer_left_lands() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // Install the real presence subscriber so the router learns about
    // pub/sub events (production wiring from main.rs).
    let presence_subscriber_cancel = CancellationToken::new();
    let presence_subscriber_handle = spawn_presence_subscriber(
        tv.client().subscriber.clone(),
        Arc::clone(&svc.presence_router),
        tv.key_schema().clone(),
        Arc::clone(&svc.metrics),
        presence_subscriber_cancel.clone(),
    );

    // Create the session + populate session_tokens the same way
    // CreateSession would.
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("shutdown-cleanup-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    let session_token = CancellationToken::new();
    svc.session_tokens
        .insert(session_id.clone(), session_token.clone().into());

    // Subscribe a fully external observer to the presence channel BEFORE
    // we start the JoinSession — so we can assert PeerLeft lands during
    // the shutdown drain, not from a missed-then-replayed event.
    let (mut events_rx, observer_handle, observer_subscriber) =
        spawn_external_channel_subscriber(&tv, &session_id).await;

    // ── Start a JoinSession ─────────────────────────────────────────────
    let (mut resp_stream, _req_tx) = open_join(&svc, &session_id, "shutdown-test-app").await;

    // Wait for SessionConnected so we know the join registered: its register
    // script wrote the lease, info hash and membership and published
    // PeerJoined.
    let connection_id = expect_session_connected(&mut resp_stream, Duration::from_secs(5)).await;

    // Drain the PeerJoined the join's register published — we only care
    // about PeerLeft for the assertion below. The external observer
    // also sees the PeerJoined; consume it so the queue starts empty.
    let mut drained_joined = false;
    while let Ok(Some(ev)) =
        tokio::time::timeout(Duration::from_millis(200), events_rx.recv()).await
    {
        if let PresenceEvent::PeerJoined(u) = &ev {
            if u.connection_id == connection_id {
                drained_joined = true;
            }
        }
    }
    assert!(
        drained_joined,
        "precondition: external observer must see our PeerJoined (connection_id={connection_id})"
    );

    // ── The shutdown sequence's stream steps ───────────────────────────
    // Step 1: cancel the service shutdown token, which ends the join as its
    // batch of the paced ending (`end_streams_after_drain`) does in
    // production. Inside `run_bidi_loop` the shutdown arm fires, and the join
    // task runs its teardown — whose unregister script publishes PeerLeft —
    // then ends the stream UNAVAILABLE.
    svc.shutdown.cancel();

    // Step 2: drain the stream tasks as `main.rs` does, under a per-task
    // budget standing in for `LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS`.
    let shutdown_task_budget = Duration::from_secs(5);
    let wait_start = Instant::now();
    live_edit::grpc::transport::drain_stream_tasks(
        &svc.presence_cleanup_tracker,
        &svc.subscription_tracker,
        shutdown_task_budget,
    )
    .await;
    let drain_elapsed = wait_start.elapsed();

    assert!(
        svc.presence_cleanup_tracker.is_empty(),
        "AC4: shutdown drain must complete within shutdown_task_budget \
         ({shutdown_task_budget:?}); pre-b.w29 the futures were dropped \
         instead of awaited and the drain would have completed in zero \
         time but with no PeerLeft published. Elapsed: {drain_elapsed:?}, \
         {} task(s) left",
        svc.presence_cleanup_tracker.len()
    );
    assert!(
        !session_token.is_cancelled(),
        "shutdown must not cancel the session token: the session still exists"
    );

    // ── PeerLeft was published before the drain returned, so the external
    //    subscriber receives it once the drain has returned ─────────────────
    //
    // The join task's teardown stops the lease keeper, then runs the
    // unregister script (which removes the member, info hash and lease and
    // publishes PeerLeft), then removes the now-empty session from
    // active_sessions. By the time `tracker.wait()` returns the PUBLISH has
    // executed; its delivery to the external observer may still be in
    // flight, so the observer is given 2 s to receive it.
    let saw_peer_left =
        wait_for_peer_left(&mut events_rx, &connection_id, Duration::from_secs(2)).await;
    assert!(
        saw_peer_left,
        "AC4: external pub/sub observer must receive PeerLeft for our \
         connection_id ({connection_id}) — pre-b.w29 the shutdown sequence \
         dropped run_join_session futures mid-cleanup, so peers saw \
         phantom members until the lease TTL expired",
    );

    // The join ended retryably: a client rejoins, wherever the ingress routes
    // it.
    let ending = tokio::time::timeout(Duration::from_secs(2), async {
        while let Some(item) = resp_stream.next().await {
            if let Err(status) = item {
                return Some(status);
            }
        }
        None
    })
    .await
    .expect("the join stream must end once its task has drained")
    .expect(
        "the join ended OK on shutdown, which the Kit client never retries — its presence feed \
         would stop for good",
    );
    assert_eq!(ending.code(), tonic::Code::Unavailable, "got {ending:?}");
    assert_eq!(ending.message(), "server shutting down");

    presence_subscriber_cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), presence_subscriber_handle).await;
    let _ = observer_subscriber
        .unsubscribe(tv.key_schema().presence_events_channel(&session_id))
        .await;
    let _ = observer_subscriber.quit().await;
    let _ = tokio::time::timeout(Duration::from_secs(2), observer_handle).await;
}
