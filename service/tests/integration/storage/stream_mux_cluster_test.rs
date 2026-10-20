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

//! Stream multiplexer in Valkey Cluster: readers are bound to one hash slot at
//! a time, so every XREAD (wake key included) stays inside one slot.
//!
//! Self-gates on `LIVE_EDIT_TEST_CLUSTER_URL`
//! (`bash scripts/dev-valkey.sh cluster-up` →
//! `LIVE_EDIT_TEST_CLUSTER_URL=redis-cluster://127.0.0.1:7000`).

use fred::interfaces::ClientLike;

use live_edit::storage::stream_mux::router::Routing;
use live_edit::storage::test_support::TestValkey;
use live_edit::storage::{consume_layer_stream, LayerStreamConsumer, StreamCursor};

use super::mux_test_helpers::{await_mux, await_readers_parked, recv_seqs, xadd_update};

async fn cluster(readers: u32) -> Option<TestValkey> {
    let Ok(url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set");
        return None;
    };
    let tv = TestValkey::setup_with_url_and_readers(url, 4, readers).await;
    assert!(tv.client().pool.next().is_clustered());
    assert_eq!(
        tv.client().stream_mux.snapshot().routing,
        Routing::SlotAffinity
    );
    Some(tv)
}

fn slot_of(tv: &TestValkey, object_id: u64) -> u16 {
    fred::util::redis_keyslot(tv.key_schema().layer_stream(object_id).as_bytes())
}

/// `n` object ids whose streams sit in pairwise different hash slots.
fn distinct_slots(tv: &TestValkey, n: usize) -> Vec<u64> {
    let mut out: Vec<u64> = Vec::new();
    for oid in 1u64.. {
        if out.iter().all(|&o| slot_of(tv, o) != slot_of(tv, oid)) {
            out.push(oid);
            if out.len() == n {
                return out;
            }
        }
    }
    unreachable!()
}

async fn consume(tv: &TestValkey, object_id: u64, from: &str) -> LayerStreamConsumer {
    consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From(from.to_owned()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("a reader is free for this slot")
}

async fn settle_reader(tv: &TestValkey) {
    await_readers_parked(tv.client()).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribers_of_one_layer_share_one_slot_bound_reader() {
    let Some(tv) = cluster(2).await else { return };
    let consumers: Vec<_> =
        futures::future::join_all((0..10).map(|_| consume(&tv, 42, "0-0"))).await;
    let snap = await_mux(tv.client(), "bound", |s| s.subscriptions() == 10).await;
    assert_eq!(snap.active_readers(), 1);
    let bound = snap.readers.iter().find(|r| r.subscriptions == 10).unwrap();
    assert_eq!(bound.slot, Some(slot_of(&tv, 42)));

    xadd_update(&tv, 42, 1).await;
    for c in &consumers {
        assert_eq!(recv_seqs(c, 1).await, [1]);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_objects_in_one_slot_are_read_by_one_xread() {
    let Some(tv) = cluster(2).await else { return };
    let a = 7u64;
    let b = (a + 1..)
        .find(|&o| slot_of(&tv, o) == slot_of(&tv, a))
        .unwrap();
    let ca = consume(&tv, a, "0-0").await;
    let cb = consume(&tv, b, "0-0").await;
    await_mux(tv.client(), "one reader, two keys", |s| {
        s.active_readers() == 1 && s.readers.iter().any(|r| r.keys == 2)
    })
    .await;
    xadd_update(&tv, a, 1).await;
    xadd_update(&tv, b, 1).await;
    assert_eq!(recv_seqs(&ca, 1).await, [1]);
    assert_eq!(recv_seqs(&cb, 1).await, [1]);
    assert!(ca.take_terminal_exit().is_none() && cb.take_terminal_exit().is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rebind_to_another_slot_wakes_the_stale_read_promptly() {
    let Some(tv) = cluster(1).await else { return };
    let ids = distinct_slots(&tv, 2);
    let first = consume(&tv, ids[0], "0-0").await;
    settle_reader(&tv).await;
    drop(first);
    await_mux(tv.client(), "unbound", |s| s.subscriptions() == 0).await;

    let second = consume(&tv, ids[1], "0-0").await;
    xadd_update(&tv, ids[1], 1).await;
    // `recv_seqs` gives up at `SETTLE` (10 s), inside the 30 s BLOCK: a
    // reader left blocked on the old slot would miss it.
    assert_eq!(recv_seqs(&second, 1).await, [1]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_late_joiner_on_a_blocked_slot_is_woken() {
    let Some(tv) = cluster(1).await else { return };
    let a = consume(&tv, 9, "0-0").await;
    xadd_update(&tv, 9, 1).await;
    assert_eq!(recv_seqs(&a, 1).await, [1]);
    settle_reader(&tv).await;
    // Within `SETTLE`, so inside the 30 s BLOCK: only a wake re-reads from
    // the rewound cursor.
    let b = consume(&tv, 9, "0-0").await;
    assert_eq!(recv_seqs(&b, 1).await, [1]);
}

/// Past the reader count, a subscription on another slot waits until its
/// deadline and is refused `Saturated` (counted once on
/// `live_edit_stream_mux_bind_refusals_total`); a subscription already parked
/// in that wait is admitted as soon as a reader frees.
///
/// The parked subscription is polled once by hand before the reader frees, so
/// it is provably inside its wait when the release lands — the release is what
/// admits it, not its arriving late — and its deadline is `SETTLE` away, so it
/// cannot refuse itself first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn distinct_slots_beyond_the_reader_count_are_refused_until_one_frees() {
    use std::time::Duration;
    let Some(tv) = cluster(2).await else { return };
    let metrics = super::mux_test_helpers::install_metrics(&tv).await;
    let refusals = || {
        super::mux_test_helpers::read_series(
            &metrics,
            "live_edit_stream_mux_bind_refusals_total",
            &[],
        )
    };
    let ids = distinct_slots(&tv, 3);
    let x = consume(&tv, ids[0], "0-0").await;
    let y = consume(&tv, ids[1], "0-0").await;
    let refused = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        ids[2],
        StreamCursor::From("0-0".into()),
        tokio::time::Instant::now() + Duration::from_millis(200),
    )
    .await;
    assert!(
        matches!(refused, Err(live_edit::storage::ConsumeError::Saturated)),
        "a third slot with two readers must be refused"
    );
    assert_eq!(refusals(), 1);

    let cursor = StreamCursor::From("0-0".into());
    let mut waiting = std::pin::pin!(tv.client().stream_mux.consume(
        tv.key_schema(),
        ids[2],
        &cursor,
        tokio::time::Instant::now() + super::mux_test_helpers::SETTLE,
    ));
    assert!(
        futures::poll!(waiting.as_mut()).is_pending(),
        "with both readers bound elsewhere the subscription must park in its bind wait"
    );
    drop(x);
    let z = tokio::time::timeout(super::mux_test_helpers::SETTLE, waiting)
        .await
        .expect("the freed reader must wake the parked subscription")
        .expect("a freed reader admits the waiting subscription");
    assert_eq!(refusals(), 1, "the admitted wait is not a refusal");
    xadd_update(&tv, ids[1], 1).await;
    xadd_update(&tv, ids[2], 1).await;
    assert_eq!(recv_seqs(&y, 1).await, [1]);
    assert_eq!(recv_seqs(&z, 1).await, [1]);
}

/// A reservation binds its reader to the key's hash slot exactly as a
/// registered stream does, with no stream registered: a layer in another slot
/// is refused `Saturated` while it is held. Dropping it — here with the task
/// that held it, aborted while reserved — unbinds the slot, and the other
/// layer then takes the reader.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_reservation_binds_its_readers_slot_and_an_aborted_holder_unbinds_it() {
    use std::sync::Arc;
    use std::time::Duration;
    let Some(tv) = cluster(1).await else { return };
    let tv = Arc::new(tv);
    let ids = distinct_slots(&tv, 2);

    let (reserved_tx, reserved_rx) = tokio::sync::oneshot::channel();
    let holder = tokio::spawn({
        let tv = Arc::clone(&tv);
        let object_id = ids[0];
        async move {
            let _held = live_edit::storage::reserve_layer_stream(
                tv.client(),
                tv.key_schema(),
                object_id,
                tokio::time::Instant::now() + super::mux_test_helpers::SETTLE,
            )
            .await
            .expect("the only reader is free");
            reserved_tx
                .send(())
                .expect("the test awaits the reservation");
            std::future::pending::<()>().await;
        }
    });
    tokio::time::timeout(super::mux_test_helpers::SETTLE, reserved_rx)
        .await
        .expect("the holder reserves")
        .expect("the holder reports its reservation");
    let snap = tv.client().stream_mux.snapshot();
    assert_eq!(snap.subscriptions(), 1);
    assert_eq!(
        snap.readers[0].slot,
        Some(slot_of(&tv, ids[0])),
        "the reservation binds the reader to its key's slot"
    );
    assert_eq!(snap.readers[0].keys, 0, "and registers no stream");

    let refused = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        ids[1],
        StreamCursor::From("0-0".into()),
        tokio::time::Instant::now() + Duration::from_millis(200),
    )
    .await;
    assert!(
        matches!(refused, Err(live_edit::storage::ConsumeError::Saturated)),
        "a layer in another slot is refused while the reservation holds the only reader"
    );

    holder.abort();
    let joined = tokio::time::timeout(super::mux_test_helpers::SETTLE, holder)
        .await
        .expect("the aborted holder finishes");
    assert!(
        joined
            .as_ref()
            .is_err_and(tokio::task::JoinError::is_cancelled),
        "the holder was aborted while reserved: {joined:?}"
    );
    let snap = tv.client().stream_mux.snapshot();
    assert_eq!(snap.subscriptions(), 0);
    assert_eq!(
        snap.readers[0].slot, None,
        "the aborted holder's reservation unbound the reader's slot"
    );

    let other = consume(&tv, ids[1], "0-0").await;
    assert_eq!(
        tv.client().stream_mux.snapshot().readers[0].slot,
        Some(slot_of(&tv, ids[1]))
    );
    xadd_update(&tv, ids[1], 1).await;
    assert_eq!(recv_seqs(&other, 1).await, [1]);
}

/// A subscription parked in its bind wait (every reader bound to another slot)
/// is released `ShuttingDown` when the multiplexer shuts down, not left to
/// wait out its deadline and be reported `Saturated`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_subscription_waiting_for_a_reader_is_refused_shutting_down_when_shutdown_fires() {
    use std::time::Duration;
    let Some(tv) = cluster(1).await else { return };
    let ids = distinct_slots(&tv, 2);
    let _bound = consume(&tv, ids[0], "0-0").await;

    let cursor = StreamCursor::From("0-0".into());
    let mut waiting = std::pin::pin!(tv.client().stream_mux.consume(
        tv.key_schema(),
        ids[1],
        &cursor,
        tokio::time::Instant::now() + Duration::from_secs(60),
    ));
    assert!(
        futures::poll!(waiting.as_mut()).is_pending(),
        "with the only reader bound elsewhere the subscription must park in its bind wait"
    );
    tv.client()
        .stream_mux
        .shutdown(Duration::from_secs(1))
        .await;
    let outcome = tokio::time::timeout(super::mux_test_helpers::SETTLE, waiting)
        .await
        .expect("shutdown must wake the parked subscription, not leave it the 60 s bind wait");
    assert!(
        matches!(outcome, Err(live_edit::storage::ConsumeError::ShuttingDown)),
        "a parked subscription is refused `ShuttingDown` at shutdown"
    );
}
