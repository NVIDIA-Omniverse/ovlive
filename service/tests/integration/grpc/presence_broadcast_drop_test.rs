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

// Regression tests for b.k2t item 2: when a JoinSession subscriber's local
// `PresenceReceiver` falls behind the per-session fan-out capacity,
// `PresenceRecvError::Lagged { dropped: n, .. }` is observed, and the
// `live_edit_presence_broadcast_dropped_total` counter must advance by `n`
// (not by 1).
//
// The wiring lives in `src/grpc/presence.rs` — the `Err(Lagged(n))` arm of
// the JoinSession bidi loop calls
// `metrics.inc_presence_broadcast_dropped(svc_label, n)`. Engineer's unit
// tests in metrics.rs cover the CounterVec itself (registration, inc_by N,
// zero safety). These tests complement them by pinning the observation
// mechanism: given the real `LocalPresenceRouter` with a small capacity
// flooded past that capacity, `PresenceReceiver::recv()` actually
// returns `Lagged(n)` with the expected drop count, so the arm will
// actually fire in production.

use live_edit::presence::{ConnectedUser, LocalPresenceRouter, PresenceEvent, PresenceRecvError};

fn make_event(id: usize) -> PresenceEvent {
    PresenceEvent::PeerJoined(ConnectedUser {
        connection_id: format!("conn-{id}"),
        user_name: format!("user-{id}"),
        user_application: "test-app".to_string(),
    })
}

/// Flooding a small-capacity channel causes the next `recv()` to return
/// `Lagged(n)` — this is the exact observation the JoinSession loop relies
/// on to detect drops. If the fan-out's lag reporting changed, the wiring
/// would silently stop counting drops.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn overflowing_presence_channel_yields_lagged_n_on_next_recv() {
    let router = LocalPresenceRouter::with_capacity(4);
    let (mut rx, _first) = router.subscribe("sess-overflow");

    // Send 10 events into a capacity-4 channel. Oldest 6 are dropped.
    for i in 0..10 {
        router.broadcast("sess-overflow", make_event(i));
    }

    let first_recv = rx.recv().await;
    match first_recv {
        Err(PresenceRecvError::Lagged { dropped: n, .. }) => {
            // The fan-out reports the exact number of skipped events.
            // With capacity=4 and 10 sends, 6 were dropped.
            assert_eq!(n, 6, "expected Lagged(6) with capacity=4 and 10 sends, got Lagged({n})");
        }
        other => panic!(
            "expected Err(Lagged(_)) — the JoinSession drop-counter wiring depends on this; got {other:?}"
        ),
    }
}

/// End-to-end wiring: simulate the exact Lagged-arm behaviour from
/// `src/grpc/presence.rs` by calling `inc_presence_broadcast_dropped` with
/// the `n` pulled out of a real `PresenceRecvError::Lagged`, and assert the
/// counter advances by that `n`. This pins the contract that the counter
/// is incremented by the number of *dropped events*, not the number of
/// *Lagged observations*.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lagged_arm_increments_counter_by_dropped_event_count() {
    let metrics = crate::common::metrics_scrape_only("test-pod");

    let router = LocalPresenceRouter::with_capacity(2);
    let (mut rx, _first) = router.subscribe("sess-wiring");

    for i in 0..7 {
        router.broadcast("sess-wiring", make_event(i));
    }

    // Drain receiver once, observe Lagged, simulate the production Lagged arm.
    let n = match rx.recv().await {
        Err(PresenceRecvError::Lagged { dropped: n, .. }) => n,
        other => panic!("expected Lagged, got {other:?}"),
    };
    // capacity=2, 7 sends → 5 dropped.
    assert_eq!(n, 5, "precondition: expected 5 drops, got {n}");

    // This is the literal call from src/grpc/presence.rs.
    metrics.inc_presence_broadcast_dropped("test-pod", n);

    let val = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_presence_broadcast_dropped_total",
        &[("service", "test-pod")],
    )
    .unwrap_or(0.0) as u64;
    assert_eq!(
        val as u64, 5,
        "counter must advance by the number of dropped events reported by Lagged, not by 1"
    );
}

/// Every scrape must include the drop counter at zero even when nothing
/// has dropped yet — otherwise the metric would appear/disappear in
/// dashboards and alert silences would mis-match on bootstrap. Pre-
/// initialization is a contract with operations; this test pins it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_counter_visible_on_registry_gather_before_any_drops() {
    let metrics = crate::common::metrics_scrape_only("test-pod");

    // The pre-initialised `service="test-pod"` cell must render at zero on the
    // very first scrape (before any drop), so dashboards / alert silences see
    // a continuous series from boot.
    let value = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_presence_broadcast_dropped_total",
        &[("service", "test-pod")],
    )
    .expect("drop counter must be pre-initialised for the service label on first scrape");
    assert_eq!(
        value, 0.0,
        "drop counter must start at zero for the pre-initialised service label"
    );
}

/// The same contract for the resync counter: a stream ended because it lagged
/// past a join or leave is rare, so an alert on it must see a continuous
/// zero series from boot rather than a metric that first appears with the
/// first resync.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resync_counter_visible_on_registry_gather_before_any_resync() {
    let metrics = crate::common::metrics_scrape_only("test-pod");

    let value = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_presence_resyncs_total",
        &[("service", "test-pod")],
    )
    .expect("resync counter must be pre-initialised for the service label on first scrape");
    assert_eq!(
        value, 0.0,
        "resync counter must start at zero for the pre-initialised service label"
    );
}
