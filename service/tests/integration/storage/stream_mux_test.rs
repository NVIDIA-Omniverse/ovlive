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

//! Stream multiplexer against a real Valkey: fan-out, multi-key XREAD, wake-ups
//! for new keys and rewinding cursors, the wake key's TTL, key isolation,
//! shutdown, the multiplexer's metric series, and a reader reservation that
//! registers its stream later (`reserve` then `ReservedStream::register`).

use std::time::{Duration, Instant};

use fred::interfaces::{ClientInterface, KeysInterface, StreamsInterface};
use fred::types::{ClientUnblockFlag, Value};

use live_edit::observability::metrics::{
    CONSUMER_PHASE_PERMANENT, STREAM_EXIT_ERROR, STREAM_EXIT_SHUTDOWN,
};
use live_edit::storage::notifications::LayerNotification;
use live_edit::storage::stream_mux::router::Routing;
use live_edit::storage::stream_mux::WAKE_KEY_TTL_MS;
use live_edit::storage::test_support::{connect_test_client, TestValkey};
use live_edit::storage::{
    consume_layer_stream, reserve_layer_stream, ConsumeError, LayerStreamConsumer, ReservedStream,
    StreamCursor,
};

use super::mux_test_helpers::{
    assert_quiet, await_mux, await_readers_parked, await_series, forced_reconnect_row,
    forced_reconnects_only_on, install_metrics, read_series, read_unblocked_retries, recv_seqs,
    recv_within, xadd_update, SETTLE,
};

const QUIET: Duration = Duration::from_millis(300);

async fn consume(tv: &TestValkey, object_id: u64, from: &str) -> LayerStreamConsumer {
    consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From(from.to_owned()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("registration is in-memory and must succeed")
}

async fn settle_reader(tv: &TestValkey) {
    await_readers_parked(tv.client()).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_xread_fans_a_stream_out_to_every_subscriber() {
    let tv = TestValkey::setup_with_readers(1).await;
    let consumers: Vec<_> =
        futures::future::join_all((0..50).map(|_| consume(&tv, 7, "0-0"))).await;
    let snap = await_mux(tv.client(), "all registered", |s| s.subscriptions() == 50).await;
    assert_eq!(snap.routing, Routing::Hash);
    assert_eq!(snap.active_readers(), 1);

    for seq in 1..=20 {
        xadd_update(&tv, 7, seq).await;
    }
    let expected: Vec<u64> = (1..=20).collect();
    for c in &consumers {
        assert_eq!(recv_seqs(c, 20).await, expected);
    }
    let snap = tv.client().stream_mux.snapshot();
    assert_eq!(
        snap.readers[0].keys, 1,
        "50 subscribers of one stream share one key"
    );
    assert!(
        snap.xreads() < 25,
        "20 entries fanned out to 50 subscribers must take ~one XREAD per entry batch, not one \
         per subscriber; saw {}",
        snap.xreads()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_reader_serves_many_streams_in_one_xread() {
    let tv = TestValkey::setup_with_readers(1).await;
    let mut consumers = Vec::new();
    for oid in 100..130 {
        consumers.push((oid, consume(&tv, oid, "0-0").await));
    }
    await_mux(tv.client(), "one XREAD over all 30 keys", |s| {
        s.readers[0].keys == 30
    })
    .await;
    for (oid, _) in &consumers {
        xadd_update(&tv, *oid, 1).await;
    }
    for (oid, c) in &consumers {
        match recv_within(c, SETTLE).await {
            Some((id, LayerNotification::Update { object_id, .. })) => {
                assert_eq!((id.as_str(), object_id), ("1-0", *oid));
            }
            other => panic!("object {oid}: {other:?}"),
        }
        assert_quiet(c, Duration::from_millis(20), "no cross-stream delivery").await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_new_stream_is_read_promptly_while_the_reader_is_blocked() {
    let tv = TestValkey::setup_with_readers(1).await;
    let _a = consume(&tv, 1, "0-0").await;
    settle_reader(&tv).await;
    let wakes_before = tv.client().stream_mux.snapshot().readers[0].wakes;

    let b = consume(&tv, 2, "0-0").await;
    xadd_update(&tv, 2, 1).await;
    // `recv_seqs` gives up at `SETTLE` (10 s), well inside the 30 s BLOCK: the
    // delivery can only have come from a wake.
    assert_eq!(recv_seqs(&b, 1).await, [1]);
    assert!(tv.client().stream_mux.snapshot().readers[0].wakes > wakes_before);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_late_joiner_behind_the_reader_gets_its_backlog_without_duplicating_others() {
    let tv = TestValkey::setup_with_readers(1).await;
    let a = consume(&tv, 3, "0-0").await;
    for seq in 1..=5 {
        xadd_update(&tv, 3, seq).await;
    }
    assert_eq!(recv_seqs(&a, 5).await, [1, 2, 3, 4, 5]);
    settle_reader(&tv).await;

    // Within `SETTLE`, so inside the 30 s BLOCK: only a wake re-reads from
    // the rewound cursor that fast.
    let b = consume(&tv, 3, "2-0").await;
    assert_eq!(recv_seqs(&b, 3).await, [3, 4, 5]);
    assert_quiet(&a, QUIET, "the rewind re-read must not re-deliver to `a`").await;

    xadd_update(&tv, 3, 6).await;
    assert_eq!(recv_seqs(&a, 1).await, [6]);
    assert_eq!(recv_seqs(&b, 1).await, [6]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_joiner_ahead_of_the_stream_skips_older_entries() {
    let tv = TestValkey::setup_with_readers(1).await;
    let a = consume(&tv, 4, "0-0").await;
    let b = consume(&tv, 4, "3-0").await;
    for seq in 1..=5 {
        xadd_update(&tv, 4, seq).await;
    }
    assert_eq!(recv_seqs(&a, 5).await, [1, 2, 3, 4, 5]);
    assert_eq!(recv_seqs(&b, 2).await, [4, 5]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unsubscribing_drops_the_stream_from_the_read() {
    let tv = TestValkey::setup_with_readers(1).await;
    let x = consume(&tv, 5, "0-0").await;
    let y = consume(&tv, 6, "0-0").await;
    await_mux(tv.client(), "both keys read", |s| s.readers[0].keys == 2).await;

    y.unsubscribe().await;
    xadd_update(&tv, 5, 1).await;
    assert_eq!(recv_seqs(&x, 1).await, [1]);
    let snap = await_mux(tv.client(), "key set shrinks", |s| s.readers[0].keys == 1).await;
    assert_eq!(snap.subscriptions(), 1);

    drop(x);
    await_mux(tv.client(), "no subscriptions", |s| s.subscriptions() == 0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deleted_sentinels_and_binary_relay_ompf_are_delivered() {
    let tv = TestValkey::setup_with_readers(1).await;
    let c = consume(&tv, 8, "0-0").await;
    let key = tv.key_schema().layer_stream(8);
    let ompf: &[u8] = &[0x00, 0xff, 0xfe, 0x80];
    let _: String = tv
        .client()
        .pool
        .xadd(
            &key,
            false,
            None,
            "1-0",
            vec![
                ("payload", Value::from("update|8|1|w|10|20")),
                ("ompf", Value::Bytes(bytes::Bytes::from_static(ompf))),
            ],
        )
        .await
        .unwrap();
    let _: String = tv
        .client()
        .pool
        .xadd(&key, false, None, "*", vec![("payload", "deleted")])
        .await
        .unwrap();

    match recv_within(&c, SETTLE).await {
        Some((
            _,
            LayerNotification::Update {
                relay_ompf: Some(b),
                ..
            },
        )) => assert_eq!(&b[..], ompf),
        other => panic!("{other:?}"),
    }
    match recv_within(&c, SETTLE).await {
        Some((_, LayerNotification::Deleted)) => {}
        other => panic!("{other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_key_that_is_not_a_stream_fails_only_its_own_subscribers() {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    let healthy = consume(&tv, 11, "0-0").await;
    let poisoned_key = tv.key_schema().layer_stream(12);
    let _: () = tv
        .client()
        .pool
        .set(&poisoned_key, "not a stream", None, None, false)
        .await
        .unwrap();
    let poisoned = consume(&tv, 12, "0-0").await;

    assert!(
        recv_within(&poisoned, SETTLE).await.is_none(),
        "the subscription on the non-stream key must end"
    );
    let exit = poisoned
        .take_terminal_exit()
        .expect("an ended subscription carries its reason");
    assert_eq!(exit.reason, STREAM_EXIT_ERROR);

    xadd_update(&tv, 11, 1).await;
    assert_eq!(
        recv_seqs(&healthy, 1).await,
        [1],
        "the healthy key keeps flowing"
    );
    let poison = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_stream_mux_poison_keys_total",
        &[],
    );
    assert_eq!(poison, Some(1.0));
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_PERMANENT, 0),
        "a failure one key explains is that key's alone: the reader must not heal its \
         connection for it"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_client_unblock_on_a_reader_is_replanned_without_loss() {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    let c = consume(&tv, 13, "0-0").await;
    settle_reader(&tv).await;

    let readers = tv.client().stream_mux.reader_clients_for_test();
    let reader = &readers[0];
    let id = *reader
        .connection_ids()
        .values()
        .next()
        .expect("fred caches CLIENT ID on connect");
    let deadline = Instant::now() + SETTLE;
    loop {
        let n: i64 = tv
            .client()
            .pool
            .next()
            .client_unblock(id, Some(ClientUnblockFlag::Error))
            .await
            .unwrap();
        if n == 1 {
            break;
        }
        assert!(Instant::now() < deadline, "the reader never blocked");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    xadd_update(&tv, 13, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);
    assert!(read_unblocked_retries(&metrics) >= 1);
    assert!(c.take_terminal_exit().is_none());
}

/// Every wake key under the test prefix, from a full `SCAN` iteration — not
/// one page, which on a large shared keyspace can miss the key.
async fn wake_keys(tv: &TestValkey) -> Vec<String> {
    use futures::StreamExt;
    let pattern = format!("{}*mux:wake*", tv.prefix());
    let mut scan = tv
        .client()
        .pool
        .next()
        .scan_buffered(pattern, Some(1_000), None);
    let mut keys = Vec::new();
    while let Some(key) = scan.next().await {
        let key = key.expect("SCAN must succeed");
        keys.push(key.as_str().expect("wake keys are UTF-8").to_owned());
    }
    keys
}

/// The wake key never exists without a TTL: the wake script appends and sets
/// the TTL in one atomic step. Checked on every wake key the test's readers
/// created, after more than one wake landed on it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_wake_key_carries_the_prefix_and_always_has_a_ttl() {
    let tv = TestValkey::setup_with_readers(1).await;
    let _a = consume(&tv, 21, "0-0").await;
    settle_reader(&tv).await;
    let _b = consume(&tv, 22, "0-0").await;
    await_mux(tv.client(), "first wake sent", |s| s.readers[0].wakes >= 1).await;
    settle_reader(&tv).await;
    let _c = consume(&tv, 23, "0-0").await;
    await_mux(tv.client(), "second wake sent", |s| s.readers[0].wakes >= 2).await;

    let deadline = Instant::now() + SETTLE;
    let keys = loop {
        let keys = wake_keys(&tv).await;
        if !keys.is_empty() {
            break keys;
        }
        assert!(
            Instant::now() < deadline,
            "no wake key under the test prefix"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    for key in keys {
        let ttl: i64 = tv.client().pool.pttl(&key).await.unwrap();
        assert!(
            ttl > 0 && ttl <= WAKE_KEY_TTL_MS,
            "wake key {key} must carry a TTL of at most WAKE_KEY_TTL_MS ({WAKE_KEY_TTL_MS} ms); \
             PTTL {ttl}"
        );
    }
}

/// `quit` ends a live subscription `shutdown` — `UNAVAILABLE "server shutting
/// down"` with the shutdown retry hint — without waiting out the reader's 30 s
/// BLOCK: the multiplexer's shutdown wakes the blocked read, which its DEBUG
/// line reports, and the multiplexer then refuses new subscriptions.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn quit_ends_live_subscriptions_unavailable_without_waiting_out_block() {
    let config = crate::common::test_config();
    let client = connect_test_client(&config).await.unwrap();
    let schema =
        live_edit::storage::keys::KeySchema::new(&format!("test_{}:", uuid::Uuid::new_v4()));
    let c = consume_layer_stream(
        &client,
        &schema,
        1,
        StreamCursor::From("0-0".into()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .unwrap();
    // The read is in flight once it is counted, and nothing makes it return.
    await_mux(&client, "reader parked", |s| s.xreads() >= 1).await;

    let ((), captured) = crate::grpc::log_capture::capture_debug(client.quit()).await;
    let woke = captured.with_message("woke blocked stream reads before the reader pool QUIT");
    assert_eq!(woke.len(), 1, "full capture:\n{}", captured.text);
    assert_eq!(
        woke[0]["fields"]["woken"], 1,
        "the reader's blocked read must have been woken; full capture:\n{}",
        captured.text
    );
    assert!(recv_within(&c, SETTLE).await.is_none());
    let exit = c
        .take_terminal_exit()
        .expect("shutdown is not a clean close");
    assert_eq!(exit.reason, STREAM_EXIT_SHUTDOWN);
    // The multiplexer's shutdown ending is the shared shutdown status:
    // `UNAVAILABLE "server shutting down"` with the `SERVER_SHUTTING_DOWN`
    // retry hint.
    crate::common::refusal_detail::mux_ending_message(exit.error, STREAM_EXIT_SHUTDOWN);
    let refused = consume_layer_stream(
        &client,
        &schema,
        2,
        StreamCursor::From("0-0".into()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await;
    assert!(
        matches!(refused, Err(ConsumeError::ShuttingDown)),
        "a subscription after shutdown is refused `ShuttingDown`"
    );
}

#[tokio::test]
async fn an_invalid_cursor_is_refused() {
    let tv = TestValkey::setup_with_readers(1).await;
    let err = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        1,
        StreamCursor::From("$".into()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .err()
    .expect("`$` is not a durable cursor");
    assert!(
        matches!(&err, ConsumeError::InvalidCursor(c) if c == "$"),
        "{err}"
    );
}

/// The stream-multiplexer series follow the multiplexer: `subscriptions` and
/// `streams` rise and fall with registrations and the streams the reader
/// reads, `xreads_total` counts reads, and a join the in-flight read does not
/// cover counts a landed wake. (That each series exports at zero before its
/// first event is
/// `subscribe_admission_test::admission_and_stream_exit_label_values_export_at_zero_before_any_event`.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_stream_mux_series_follow_subscriptions_reads_and_wakes() {
    const SUBSCRIPTIONS: &str = "live_edit_stream_mux_subscriptions";
    const STREAMS: &str = "live_edit_stream_mux_streams";
    const XREADS: &str = "live_edit_stream_mux_xreads_total";
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;

    let a = consume(&tv, 31, "0-0").await;
    assert_eq!(read_series(&metrics, SUBSCRIPTIONS, &[]), 1);
    await_series(&metrics, STREAMS, &[], 1).await;
    settle_reader(&tv).await;
    let xreads = read_series(&metrics, XREADS, &[]);
    assert!(xreads >= 1, "the reader's reads are counted");
    assert_eq!(xreads as u64, tv.client().stream_mux.snapshot().xreads());

    // Two subscriptions of one new stream: one more stream, and a wake.
    let b = consume(&tv, 32, "0-0").await;
    let b2 = consume(&tv, 32, "0-0").await;
    assert_eq!(read_series(&metrics, SUBSCRIPTIONS, &[]), 3);
    await_series(&metrics, STREAMS, &[], 2).await;
    await_series(
        &metrics,
        "live_edit_stream_mux_wakes_total",
        &[("outcome", "ok")],
        1,
    )
    .await;

    drop(b);
    assert_eq!(read_series(&metrics, SUBSCRIPTIONS, &[]), 2);
    drop(a);
    await_series(&metrics, STREAMS, &[], 1).await;
    drop(b2);
    assert_eq!(read_series(&metrics, SUBSCRIPTIONS, &[]), 0);
    await_series(&metrics, STREAMS, &[], 0).await;
}

// ── Reserve, then register ──────────────────────────────────────────────────

async fn reserve(tv: &TestValkey, object_id: u64) -> ReservedStream {
    reserve_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        tokio::time::Instant::now() + SETTLE,
    )
    .await
    .expect("standalone routing always has a reader")
}

/// A reservation holds its reader — it counts on
/// `live_edit_stream_mux_subscriptions` and in the snapshot — but the reader
/// neither reads nor queues anything for it: no stream is counted, no reader
/// connection is in use and no XREAD is issued, however many entries land.
/// Registering starts the stream at the cursor it is given, so entries at or
/// below it are never delivered, and those after it all are.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_reservation_holds_a_reader_and_reads_nothing_until_registered_at_its_cursor() {
    const SUBSCRIPTIONS: &str = "live_edit_stream_mux_subscriptions";
    const STREAMS: &str = "live_edit_stream_mux_streams";
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;

    let reserved = reserve(&tv, 51).await;
    for seq in 1..=3 {
        xadd_update(&tv, 51, seq).await;
    }
    let snap = tv.client().stream_mux.snapshot();
    assert_eq!(snap.subscriptions(), 1, "the reservation holds the reader");
    assert_eq!(read_series(&metrics, SUBSCRIPTIONS, &[]), 1);
    assert_eq!(
        read_series(&metrics, STREAMS, &[]),
        0,
        "no stream is registered"
    );
    assert_eq!(
        read_series(
            &metrics,
            "live_edit_valkey_pool_in_use",
            &[("pool", "subscriber_pool")]
        ),
        0,
        "no reader connection is in use for a reservation"
    );
    assert_eq!(
        (snap.readers[0].keys, snap.xreads()),
        (0, 0),
        "the reader has nothing to read: a reservation names no key"
    );

    let consumer = reserved
        .register(&StreamCursor::From("2-0".into()))
        .expect("registration is in-memory and must succeed");
    assert_eq!(
        recv_seqs(&consumer, 1).await,
        [3],
        "the stream starts strictly after its registration cursor"
    );
    await_series(&metrics, STREAMS, &[], 1).await;
    xadd_update(&tv, 51, 4).await;
    assert_eq!(recv_seqs(&consumer, 1).await, [4]);
    assert_quiet(
        &consumer,
        QUIET,
        "nothing at or below the cursor comes later",
    )
    .await;
    assert_eq!(
        read_series(&metrics, SUBSCRIPTIONS, &[]),
        1,
        "one subscription throughout"
    );
    drop(consumer);
    assert_eq!(read_series(&metrics, SUBSCRIPTIONS, &[]), 0);
    await_series(&metrics, STREAMS, &[], 0).await;
}

/// A reservation is released however it goes: dropped, or dropped with the
/// task that held it when that task is aborted while reserved. Either way the
/// reader's reference, the snapshot count and the subscriptions gauge return
/// to zero. (The cluster twin, where the release also unbinds the reader's
/// hash slot, is in `stream_mux_cluster_test.rs`.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_reservation_is_released_when_dropped_or_when_its_task_is_aborted() {
    const SUBSCRIPTIONS: &str = "live_edit_stream_mux_subscriptions";
    let tv = std::sync::Arc::new(TestValkey::setup_with_readers(1).await);
    let metrics = install_metrics(&tv).await;

    let reserved = reserve(&tv, 52).await;
    assert_eq!(read_series(&metrics, SUBSCRIPTIONS, &[]), 1);
    drop(reserved);
    assert_eq!(tv.client().stream_mux.snapshot().subscriptions(), 0);
    assert_eq!(read_series(&metrics, SUBSCRIPTIONS, &[]), 0);

    let (reserved_tx, reserved_rx) = tokio::sync::oneshot::channel();
    let task = tokio::spawn({
        let tv = std::sync::Arc::clone(&tv);
        async move {
            let _held = reserve(&tv, 53).await;
            reserved_tx
                .send(())
                .expect("the test awaits the reservation");
            std::future::pending::<()>().await;
        }
    });
    tokio::time::timeout(SETTLE, reserved_rx)
        .await
        .expect("the task reserves")
        .expect("the task reports its reservation");
    assert_eq!(tv.client().stream_mux.snapshot().subscriptions(), 1);
    assert_eq!(read_series(&metrics, SUBSCRIPTIONS, &[]), 1);

    task.abort();
    let joined = tokio::time::timeout(SETTLE, task)
        .await
        .expect("the aborted task finishes");
    assert!(
        joined
            .as_ref()
            .is_err_and(tokio::task::JoinError::is_cancelled),
        "the task was aborted while reserved: {joined:?}"
    );
    assert_eq!(
        tv.client().stream_mux.snapshot().subscriptions(),
        0,
        "the aborted task's reservation released its reader"
    );
    assert_eq!(read_series(&metrics, SUBSCRIPTIONS, &[]), 0);
}

/// A reservation that registers after the multiplexer shut down is refused
/// `ShuttingDown` — the ending the gRPC layer maps to `shutdown` — and the
/// failed registration releases its reader.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_registration_after_shutdown_is_refused_shutting_down_and_releases_its_reader() {
    let config = crate::common::test_config();
    let client = connect_test_client(&config).await.unwrap();
    let schema =
        live_edit::storage::keys::KeySchema::new(&format!("test_{}:", uuid::Uuid::new_v4()));
    let reserved = reserve_layer_stream(&client, &schema, 54, tokio::time::Instant::now() + SETTLE)
        .await
        .expect("a reader is free");
    assert_eq!(client.stream_mux.snapshot().subscriptions(), 1);

    client.stream_mux.shutdown(Duration::from_secs(1)).await;
    let refused = reserved.register(&StreamCursor::From("0-0".into()));
    assert!(
        matches!(refused, Err(ConsumeError::ShuttingDown)),
        "registering on a closed multiplexer is refused ShuttingDown"
    );
    assert_eq!(
        client.stream_mux.snapshot().subscriptions(),
        0,
        "the refused registration released the reader"
    );
    client.quit().await;
}
