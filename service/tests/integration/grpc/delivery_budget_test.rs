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

//! b.ntm — the live-delivery budget `D_live` and the discard at server-driven
//! endings, end to end through `handle_subscribe_read_layer` and the stream
//! multiplexer (SDD §12.6):
//!
//! - **The victim** (T6): with the replica's relay bytes held for undelivered
//!   updates past the budget (sized down by `set_budget_bytes_for_test`), the
//!   subscription holding the most of them — one that stopped reading — is
//!   ended `delivery_budget`, `UNAVAILABLE`, and catches up on its retry,
//!   while a sibling on the same layer keeps receiving every commit in order:
//!   the stream reader never paused, and no second subscription was ended.
//! - **The discard** (T9): the ended subscription's queued updates are
//!   dropped at its ending, so a client that resumes reading gets only the
//!   status; so are an `inner_watchdog` ending's, with its catch-up content
//!   queued ahead of them.
//! - **The excusal** (T7 a): one relay larger than the whole budget reaches a
//!   caught-up subscriber, byte-identical, with small edits continuing.
//! - **Release** (T8, T15): the ledger is back to zero — no copy, no queued
//!   entry, no hold, nothing pending — once the subscriptions are gone, and
//!   its usage gauges read what it holds.
//!
//! The ledger's rules — the charges, the dedupe, the victim choice and its
//! ties, the lone rule, the overshoot, every release path — are pinned on
//! their own in `storage::stream_mux::delivery_ledger`'s unit tests.
//!
//! **Wait budget.** Every wait is a deadline, not a measurement: `SETTLE`
//! (10 s) per awaited message, park or polled condition, `DRAIN` (15 s) per
//! drain — a catch-up to its ack, a run of deliveries (each message's own
//! `SETTLE` inside it), a stream to its end. Each test's worst case, every
//! wait at its full budget, against nextest's 120 s kill:
//!
//! | Test | Waits | Worst case |
//! |---|---|---|
//! | the victim's | 5 `DRAIN` (two catch-ups, the sibling's deliveries, the victim's end, the retry's catch-up) + 2 `SETTLE` (its ending, the ledger released) | 95 s |
//! | the excusal's | 2 `DRAIN` (the catch-up, the four deliveries) + 1 `SETTLE` (the ledger released) | 40 s |
//! | the `inner_watchdog` discard's | 5 `SETTLE` (the park, the cap, the release, the ending, the ledger released) + 1 `DRAIN` (the stream's end) | 65 s |
//!
//! The healthy path takes about a second.

use std::sync::Arc;
use std::time::Duration;

use futures::{Stream, StreamExt};
use tonic::{Code, Request, Status};

use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{
    subscribe_read_layer_response, SubscribeReadLayerHeader, SubscribeReadLayerRequest,
    SubscribeReadLayerResponse,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::STREAM_EXIT_DELIVERY_BUDGET;
use live_edit::storage::session::SessionStore;
use live_edit::storage::stream_mux::delivery_ledger::DELIVERY_BUDGET_ENDING_MSG;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandCreateSection, OwnedDelta, StructCommand};

use super::helpers::*;
use crate::common::owned_delta::{delta_to_ompf_bytes, delta_with_set_field_inline};

/// The bound on each awaited message, park or polled condition.
const SETTLE: Duration = Duration::from_secs(10);
/// The bound on each drain: a catch-up to its ack, a run of deliveries, a
/// stream to its end.
const DRAIN: Duration = Duration::from_secs(15);

/// The section every field here lives in.
const SECTION_ID: u64 = 7;

type SubStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Send>>;

/// A service over `tv`, its client's metrics installed on the service's
/// handle, so the multiplexer's and the ledger's series render with the
/// handler's.
async fn service(tv: &TestValkey) -> LiveEditService {
    let svc = build_service(tv, Arc::new(AllowAllPermissions));
    tv.client()
        .install_observability_hooks(Arc::clone(&svc.metrics))
        .await;
    svc
}

/// A session and a layer holding [`SECTION_ID`]: `(session_id, object_id)`,
/// at sequence 1.
async fn layer(tv: &TestValkey, svc: &LiveEditService, layer: &str) -> (String, u64) {
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(&format!("{layer}-session"), "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    let create = OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: SECTION_ID,
            section_name: "Asset".to_owned(),
            section_type: 1,
        })],
        ..Default::default()
    };
    let stream = streaming_create(vec![
        make_create_header(&session_id, layer),
        make_create_data(delta_to_ompf_bytes(&create, &[])),
    ]);
    let object_id = Box::pin(handle_create_layer(svc, make_request(stream)))
        .await
        .expect("the setup CreateLayer must succeed")
        .into_inner()
        .object_id;
    (session_id, object_id)
}

/// The OMPF of commit `n`: one inline field of `len` bytes, distinct per `n`.
fn commit_ompf(n: u64, len: usize) -> Vec<u8> {
    let value: Vec<u8> = (0..len).map(|i| (i as u64 ^ n) as u8).collect();
    delta_to_ompf_bytes(
        &delta_with_set_field_inline(SECTION_ID, "blob", value, n),
        &[],
    )
}

/// One `UpdateLayer` of `ompf`: its relay is these bytes.
async fn commit(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    object_id: u64,
    ompf: Vec<u8>,
) {
    let stream = streaming_update(vec![
        make_update_header(session_id, layer, object_id),
        make_update_data(ompf),
    ]);
    Box::pin(handle_update_layer(svc, make_request(stream)))
        .await
        .expect("the UpdateLayer must succeed");
}

async fn subscribe(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    have: (u64, u64),
    client: &str,
) -> SubStream {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer.to_owned(),
        object_id: have.0,
        sequence_num: have.1,
        client_id: client.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    Box::pin(handle_subscribe_read_layer(svc, request))
        .await
        .expect("the subscription must be admitted")
        .into_inner()
}

/// One delivery: a header and its body.
struct Delivery {
    header: SubscribeReadLayerHeader,
    body: Vec<u8>,
}

/// The next delivery on `stream`, or its ending status: a header and, for
/// content, the chunks that make up its `total_size`.
async fn next_delivery(stream: &mut SubStream) -> Result<Delivery, Status> {
    let header = match tokio::time::timeout(SETTLE, stream.next())
        .await
        .unwrap_or_else(|_| panic!("a delivery did not arrive within {SETTLE:?}"))
        .expect("the stream ended without a status")?
        .response_type
    {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => h,
        other => panic!("a header, not {other:?}"),
    };
    let mut body = Vec::new();
    while (body.len() as u64) < header.total_size {
        match tokio::time::timeout(SETTLE, stream.next())
            .await
            .unwrap_or_else(|_| panic!("a chunk did not arrive within {SETTLE:?}"))
            .expect("the stream ended mid-body")?
            .response_type
        {
            Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                body.extend_from_slice(&c.data)
            }
            other => panic!("a chunk, not {other:?}"),
        }
    }
    Ok(Delivery { header, body })
}

/// Drain a catch-up to its ack: the ack's header.
async fn drain_catch_up(stream: &mut SubStream) -> SubscribeReadLayerHeader {
    tokio::time::timeout(DRAIN, async {
        loop {
            let delivery = next_delivery(stream)
                .await
                .unwrap_or_else(|s| panic!("a catch-up message, not {s:?}"));
            if delivery.header.total_size == 0 {
                return delivery.header;
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the catch-up did not finish within {DRAIN:?}"))
}

/// Everything `stream` yields to its end: its messages and its status.
async fn drain_to_end(stream: &mut SubStream) -> (Vec<SubscribeReadLayerResponse>, Option<Status>) {
    tokio::time::timeout(DRAIN, async {
        let mut messages = Vec::new();
        let mut status = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(m) => messages.push(m),
                Err(s) => status = Some(s),
            }
        }
        (messages, status)
    })
    .await
    .unwrap_or_else(|_| panic!("the stream did not end within {DRAIN:?}"))
}

fn exits(svc: &LiveEditService, reason: &str) -> u64 {
    sample_value(
        &render(&svc.metrics),
        "live_edit_subscriber_stream_exits_total",
        &[("reason", reason)],
    )
    .expect("every exit reason is seeded") as u64
}

fn series(svc: &LiveEditService, name: &str) -> u64 {
    sample_value(&render(&svc.metrics), name, &[]).unwrap_or_else(|| panic!("{name} renders"))
        as u64
}

async fn await_condition<F: FnMut() -> bool>(mut condition: F, what: &str) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    while !condition() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what} did not hold within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// The ledger back to zero: no relay copy, no queued entry, no hold, nothing
/// pending; and its usage gauges reading so.
async fn assert_ledger_released(svc: &LiveEditService, what: &str) {
    let ledger = svc.valkey.stream_mux.delivery_ledger();
    await_condition(
        || {
            ledger.charged_bytes() == 0
                && ledger.exempt_bytes() == 0
                && ledger.pending_release_bytes() == 0
                && ledger.holds() == 0
        },
        what,
    )
    .await;
    assert_eq!(
        (
            series(svc, "live_edit_subscriber_delivery_charged_bytes"),
            series(svc, "live_edit_subscriber_delivery_exempt_bytes"),
        ),
        (0, 0),
        "{what}: the usage gauges read the ledger"
    );
}

/// The relay each commit carries: 64 KiB, so a dozen commits a stalled
/// subscriber holds pass [`VICTIM_BUDGET`] long before its 32 queue slots
/// fill (two messages a commit), and no send of it ever parks.
const RELAY_BYTES: usize = 64 * 1024;

/// The live-delivery budget the victim's test runs at.
const VICTIM_BUDGET: u64 = 512 * 1024;

/// Commits the writer makes while the victim does not read.
const COMMITS: u64 = 24;

/// T6 (fails before b.ntm, which had no live-delivery budget) and T9 for
/// `delivery_budget`: a subscriber that stops reading after its catch-up
/// holds every commit's relay copy; once it holds more than the budget, it
/// is ended `delivery_budget`, `UNAVAILABLE` with the ending's message, and a
/// client resuming its read gets only that status — the updates queued for
/// it were dropped at the ending. Meanwhile its sibling on the same layer,
/// reading as it goes, receives every one of the commits in order: the
/// stream reader never paused for the budget, and no cursor passed content
/// it had not delivered. The victim's retry from its last acked position
/// catches up to the layer's sequence in one step. Exactly one subscription
/// was ended for the budget, and no enforcement pass found itself over with
/// nothing to end. With every subscription gone the ledger is back to zero.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_largest_holder_of_undelivered_relays_is_ended_delivery_budget_while_its_sibling_keeps_delivering(
) {
    let tv = TestValkey::setup().await;
    let svc = service(&tv).await;
    svc.valkey
        .stream_mux
        .delivery_ledger()
        .set_budget_bytes_for_test(VICTIM_BUDGET);
    let layer_name = "delivery-budget-victim";
    let (session_id, object_id) = layer(&tv, &svc, layer_name).await;

    let mut victim = subscribe(&svc, &session_id, layer_name, (0, 0), "victim").await;
    let victim_ack = drain_catch_up(&mut victim).await;
    let mut sibling = subscribe(&svc, &session_id, layer_name, (0, 0), "sibling").await;
    let sibling_ack = drain_catch_up(&mut sibling).await;
    assert_eq!((victim_ack.sequence_num, sibling_ack.sequence_num), (1, 1));

    let reading = tokio::spawn(async move {
        let mut seen = Vec::new();
        while seen.len() < COMMITS as usize {
            let delivery = next_delivery(&mut sibling)
                .await
                .unwrap_or_else(|s| panic!("the sibling keeps delivering, not {s:?}"));
            seen.push(delivery.header.sequence_num);
        }
        (sibling, seen)
    });
    for n in 1..=COMMITS {
        commit(
            &svc,
            &session_id,
            layer_name,
            object_id,
            commit_ompf(n, RELAY_BYTES),
        )
        .await;
    }
    let (sibling, seen) = tokio::time::timeout(DRAIN, reading)
        .await
        .expect("the sibling read every commit")
        .expect("the sibling's reader");
    assert_eq!(
        seen,
        (2..=COMMITS + 1).collect::<Vec<_>>(),
        "the sibling received every commit, in order"
    );
    await_condition(
        || exits(&svc, STREAM_EXIT_DELIVERY_BUDGET) == 1,
        "the stalled subscriber is ended delivery_budget",
    )
    .await;

    let (messages, status) = drain_to_end(&mut victim).await;
    assert!(
        messages.is_empty(),
        "the delivery_budget ending dropped the {} updates queued for the victim; a resuming \
         client gets the status next",
        messages.len()
    );
    let status = status.expect("the victim's ending status");
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert_eq!(status.message(), DELIVERY_BUDGET_ENDING_MSG);

    let mut retried = subscribe(
        &svc,
        &session_id,
        layer_name,
        (object_id, victim_ack.sequence_num),
        "victim",
    )
    .await;
    let caught_up = drain_catch_up(&mut retried).await;
    assert_eq!(
        caught_up.sequence_num,
        COMMITS + 1,
        "the retry catches up to the layer's sequence"
    );
    assert_eq!(
        exits(&svc, STREAM_EXIT_DELIVERY_BUDGET),
        1,
        "exactly one subscription ended for the budget"
    );
    assert_eq!(
        series(
            &svc,
            "live_edit_subscriber_delivery_budget_overshoots_total"
        ),
        0,
        "every pass over the budget found the victim, or counted it as pending release"
    );

    drop((sibling, retried, victim));
    assert_ledger_released(&svc, "every subscription gone").await;
}

/// T7 (a): one relay larger than the whole live-delivery budget reaches a
/// caught-up subscriber, byte-identical to the writer's OMPF, with small
/// edits continuing on the layer behind it, and nothing is ended for the
/// budget: the layer's largest pending update is excused, and a subscriber
/// holding at most two is never a victim.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_relay_larger_than_the_budget_reaches_a_caught_up_subscriber_with_edits_behind_it() {
    let tv = TestValkey::setup().await;
    let svc = service(&tv).await;
    svc.valkey
        .stream_mux
        .delivery_ledger()
        .set_budget_bytes_for_test(64 * 1024);
    let layer_name = "delivery-budget-large";
    let (session_id, object_id) = layer(&tv, &svc, layer_name).await;
    let mut reader = subscribe(&svc, &session_id, layer_name, (0, 0), "reader").await;
    drain_catch_up(&mut reader).await;

    let large = commit_ompf(1, 2 << 20);
    commit(&svc, &session_id, layer_name, object_id, large.clone()).await;
    for n in 2..=4 {
        commit(
            &svc,
            &session_id,
            layer_name,
            object_id,
            commit_ompf(n, 100),
        )
        .await;
    }
    tokio::time::timeout(DRAIN, async {
        let first = next_delivery(&mut reader).await.expect("the large relay");
        assert_eq!(first.header.sequence_num, 2);
        assert!(
            first.body == large,
            "the relay is the writer's OMPF, byte-identical"
        );
        for seq in 3..=5 {
            let next = next_delivery(&mut reader).await.expect("a small edit");
            assert_eq!(next.header.sequence_num, seq);
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the four deliveries did not arrive within {DRAIN:?}"));
    assert_eq!(exits(&svc, STREAM_EXIT_DELIVERY_BUDGET), 0);
    drop(reader);
    assert_ledger_released(&svc, "the reader gone").await;
}

/// T9 for `inner_watchdog`: a subscription whose stream registration is
/// killed by its backlog hard cap while its catch-up's content sits in its
/// response queue unread ends `inner_watchdog`, and a client resuming its
/// read gets only the status: the catch-up content and the live updates
/// queued behind it were dropped at the ending. The catch-up is held after
/// its first content message by the chunk seam while the cap is overrun.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_inner_watchdog_ending_drops_the_queued_catch_up_and_updates_and_leaves_only_its_status()
{
    use live_edit::grpc::subscribe::CatchUpChunkSeam;
    use live_edit::observability::metrics::STREAM_EXIT_INNER_WATCHDOG;
    use live_edit::storage::stream_mux::StreamMuxKnobs;

    const TINY_HARD_CAP: usize = 8;
    let tv = TestValkey::setup().await;
    let mut svc = service(&tv).await;
    let seam = Arc::new(CatchUpChunkSeam::new());
    svc.catch_up_chunk_seam_for_test = Some(Arc::clone(&seam));
    let knobs = svc.valkey.stream_mux.knobs();
    svc.valkey.stream_mux.set_knobs_for_test(StreamMuxKnobs {
        channel_capacity: TINY_HARD_CAP / 2,
        backlog_hard_cap: TINY_HARD_CAP,
        send_watchdog: Duration::from_secs(600),
        ..knobs
    });
    let layer_name = "delivery-budget-watchdog";
    let (session_id, object_id) = layer(&tv, &svc, layer_name).await;

    let mut stream = subscribe(&svc, &session_id, layer_name, (0, 0), "watchdog").await;
    tokio::time::timeout(SETTLE, seam.wait_until_parked())
        .await
        .expect("the catch-up parks after its first content message");
    for n in 1..=(TINY_HARD_CAP as u64 + 4) {
        commit(
            &svc,
            &session_id,
            layer_name,
            object_id,
            commit_ompf(n, 100),
        )
        .await;
    }
    await_condition(
        || {
            sample_value(
                &render(&svc.metrics),
                "live_edit_stream_mux_overflow_kills_total",
                &[("bound", "entries")],
            )
            .unwrap_or(0.0)
                == 1.0
        },
        "the registration's backlog hits the hard cap",
    )
    .await;
    tokio::time::timeout(SETTLE, seam.release())
        .await
        .expect("the parked catch-up accepts its release");
    await_condition(
        || exits(&svc, STREAM_EXIT_INNER_WATCHDOG) == 1,
        "the subscription ends inner_watchdog",
    )
    .await;

    let (messages, status) = drain_to_end(&mut stream).await;
    assert!(
        messages.is_empty(),
        "the inner_watchdog ending dropped the {} queued message(s) — the catch-up's content and \
         ack and the live updates behind them — before its status",
        messages.len()
    );
    let status = status.expect("the ending's status");
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert!(
        status
            .message()
            .contains("fell behind its delivery watchdog"),
        "{status:?}"
    );
    drop(stream);
    assert_ledger_released(&svc, "the ended subscription gone").await;
}
