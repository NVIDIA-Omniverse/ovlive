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

//! b.zt3 F-C — fred's command-queue counter can no longer hold a frame back.
//!
//! fred writes a command without flushing it when its command-queue counter
//! (`ClientInner::counters.cmd_buffer_len`, incremented by each sender and
//! decremented by the router as it takes the command) says another command
//! follows, and relies on that next write to flush it. A counter that drifted
//! above the channel's real length — the hypothesis for a client whose socket
//! looked dead while the store answered it — holds every write back: the
//! frame sits in fred's write buffer, and nothing arrives to flush it. The
//! vendored fred (1) resyncs the counter to the channel's length at every
//! receive, logging the drift (F-C2), and (2) flushes every connection holding
//! fed frames before its router waits on an empty channel (F-C1).
//!
//! The counter is a public field of fred's `ClientInner`, so these tests set it
//! directly — the drift a leaked count would leave — on one data member of a
//! client they own:
//!
//! | Test | Design item | What it pins |
//! |---|---|---|
//! | [`a_get_at_idle_with_the_counter_drifted_to_one_is_answered_and_still_is_after_a_reconnect`] | T6 | with the counter forced to 1 at idle, a `GET` is answered (and `PING` too), and again after a forced reconnect: the counter hypothesis's second line of evidence, against stock fred where the `GET` times out |
//! | [`a_drifted_counter_is_reset_at_the_next_receive_after_a_router_restart_and_logged_once`] | T7 | after `end_router_for_test` and the restart, a drifted counter equals the channel's length (0 at idle) once the next command is taken; a drift of exactly one either way (+1, −1) logs at DEBUG only (R4-D), and a larger one the drift WARN, once in its interval |
//! | [`a_frame_fed_at_idle_is_sent_without_another_command_arriving`] | T18 | with the counter held high by a second thread throughout, so every write is only fed, each `GET` is still sent (and answered) with no further command behind it |
//!
//! T18 is the deterministic form of F-C1 the design asks for, as far as an
//! integration test can make it: a spinning thread keeps the counter high, so
//! the resync of F-C2 (which runs just before the write decision) is
//! overwritten before the decision reads it, and only F-C1's flush can send
//! the frame. A `GET` left fed waits out fred's 5 s command timeout and fails
//! `Timeout`.
//!
//! # Wait budget
//!
//! The client's connect (30 s), its commands at fred's 5 s command timeout
//! (T6 six, T7 six plus a 10 s restart poll, T18 four), the forced reconnect
//! (5 s) and the closing `quit()` (11.5 s, the sum of its internal bounds: it
//! has no outer one): at most 30 + 30 + 10 + 11.5 = 81.5 s (T7), under the
//! 120 s hang-killer.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fred::clients::Client;
use fred::interfaces::{ClientLike, KeysInterface};
use tokio::time::Instant;

use live_edit::observability::metrics::VALKEY_POOL_DATA;
use live_edit::storage::client::{ValkeyClient, POOL_LABEL_DATA};
use live_edit::storage::test_support::connect_test_client;

use super::b_zt3_support::{await_until, bridge_fred_log, messages_containing, ROUTER_RESTARTS};
use super::mux_test_helpers::read_series;
use crate::grpc::log_capture::ThreadCapture;

/// Bound on the router restart after `end_router_for_test`.
const SETTLE: Duration = Duration::from_secs(10);

/// The value the counter is held at in T18: far above any real queue.
const HELD_HIGH: usize = 1_000_000;

/// The start of fred's drift WARN (a drift of two or more).
const DRIFT_LINE: &str = "Command-queue counter drift";

/// The start of fred's DEBUG line for a drift of exactly one.
const OFF_BY_ONE_LINE: &str = "Command-queue counter off by one";

/// A client of its own, straight to the store, with one data member; its key
/// holds `"v"`. (A client of its own: the drift is staged on its counter, and
/// under the old code it would leave the client stuck.)
async fn own_client(key: &str) -> ValkeyClient {
    let mut config = crate::common::test_config();
    config.valkey_pool_size = 1;
    let client = connect_test_client(&config)
        .await
        .expect("the client must connect");
    let _: () = client
        .pool
        .set(key, "v", None, None, false)
        .await
        .expect("the key must be written");
    client
}

fn key() -> String {
    format!("zt3:counter:{}", uuid::Uuid::new_v4())
}

fn counter(member: &Client) -> &std::sync::atomic::AtomicUsize {
    &member.inner().counters.cmd_buffer_len
}

/// T6: the counter forced to 1 with nothing queued — a sender's count the
/// router never took off — would, on stock fred, leave the next `GET` fed and
/// unsent (it times out while a `PING`, which fred always flushes, is
/// answered), and a reconnect would not cure it: the counter is per client.
/// Here the `GET` is answered, and so is one after a forced reconnect with the
/// counter forced again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_get_at_idle_with_the_counter_drifted_to_one_is_answered_and_still_is_after_a_reconnect()
{
    let key = key();
    let client = own_client(&key).await;
    let member = client.pool.member(0).clone();

    for phase in ["before", "after"] {
        if phase == "after" {
            member
                .force_reconnection()
                .await
                .expect("the forced reconnect must complete");
        }
        counter(&member).store(1, Ordering::SeqCst);
        let got: Option<String> = member.get(&key).await.unwrap_or_else(|e| {
            panic!("{phase} the reconnect, a GET at idle with the counter at 1 must be answered: {e:?}")
        });
        assert_eq!(got.as_deref(), Some("v"));
        let pong: String = member.ping(None).await.expect("PING must be answered");
        assert_eq!(pong, "PONG");
    }
    let _: () = member.del(&key).await.expect("the key must be deleted");
    client.quit().await;
}

/// T7: after the member's router was ended (`end_router_for_test`) and
/// restarted, a counter drifted from the channel's length is reset at the
/// next receive: once that command is answered the counter reads the
/// channel's length, 0 at idle.
///
/// - A drift of exactly one either way (b.zt3 rev 4, R4-D) — usually a send in
///   flight as the counter was read — is reset and logged at DEBUG only, never
///   as the WARN: +1 is the counter stored at 1 at idle (as T6 does), −1 the
///   counter stored at `usize::MAX`, which the sender's increment wraps to 0
///   (fred's increment is a wrapping `fetch_add`).
/// - A larger drift logs one WARN naming it; a second one inside the same
///   interval is reset as well but not logged again.
///
/// Current-thread runtime, so the thread-local capture sees fred's router
/// task, and nothing runs between a staged counter and the next send.
#[tokio::test]
async fn a_drifted_counter_is_reset_at_the_next_receive_after_a_router_restart_and_logged_once() {
    bridge_fred_log();
    let capture = ThreadCapture::start_debug();
    let key = key();
    let client = own_client(&key).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("zt3"));
    client
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    let member = client.pool.member(0).clone();

    assert!(client.end_router_for_test(POOL_LABEL_DATA, 0));
    await_until(
        "the ended router must be restarted",
        Instant::now() + SETTLE,
        || read_series(&metrics, ROUTER_RESTARTS, &[("pool", VALKEY_POOL_DATA)]) == 1,
    )
    .await;
    // The counter value each drift is staged with at idle, and the drift the
    // next receive reads (the sender's increment comes in between).
    for (stored, drift) in [(1, "+1"), (usize::MAX, "-1"), (7, "+7"), (5, "+5")] {
        counter(&member).store(stored, Ordering::SeqCst);
        let got: Option<String> = member
            .get(&key)
            .await
            .unwrap_or_else(|e| panic!("the GET after a drift of {drift} must be answered: {e:?}"));
        assert_eq!(got.as_deref(), Some("v"));
        assert_eq!(
            counter(&member).load(Ordering::SeqCst),
            0,
            "after a drift of {drift}, the counter must read the channel's length (empty) once \
             the next command was taken"
        );
    }

    let captured = capture.snapshot();
    let off_by_one: Vec<&serde_json::Value> = captured
        .events
        .iter()
        .filter(|e| {
            e["fields"]["message"]
                .as_str()
                .is_some_and(|m| m.contains(OFF_BY_ONE_LINE))
        })
        .collect();
    for drift in ["(+1)", "(-1)"] {
        let lines: Vec<&&serde_json::Value> = off_by_one
            .iter()
            .filter(|e| {
                e["fields"]["message"]
                    .as_str()
                    .is_some_and(|m| m.contains(drift))
            })
            .collect();
        assert_eq!(
            lines.len(),
            1,
            "one off-by-one line for the drift {drift}; full capture:\n{}",
            captured.text
        );
        assert_eq!(lines[0]["level"], "DEBUG", "{}", lines[0]);
    }
    assert_eq!(
        off_by_one.len(),
        2,
        "no other off-by-one drift; full capture:\n{}",
        captured.text
    );
    let drifts = messages_containing(&captured, DRIFT_LINE);
    assert_eq!(
        drifts.len(),
        1,
        "one drift WARN per interval; full capture:\n{}",
        captured.text
    );
    assert!(
        drifts[0].contains("(+7)"),
        "the WARN names the first drift larger than one, not an off-by-one: {}",
        drifts[0]
    );
    let _: () = member.del(&key).await.expect("the key must be deleted");
    client.quit().await;
}

/// T18: a second thread holds the counter far above the channel's length for
/// the whole test, so every `GET` this member writes is fed without a flush
/// and no command follows it. Each is still sent and answered: before the
/// router waits on its empty channel it flushes the fed frame.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_frame_fed_at_idle_is_sent_without_another_command_arriving() {
    let key = key();
    let client = own_client(&key).await;
    let member = client.pool.member(0).clone();
    let stop = Arc::new(AtomicBool::new(false));
    let holder = {
        let (counter, stop) = (
            Arc::clone(&member.inner().counters.cmd_buffer_len),
            Arc::clone(&stop),
        );
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                counter.store(HELD_HIGH, Ordering::Relaxed);
                std::hint::spin_loop();
            }
        })
    };

    let mut replies = Vec::new();
    for _ in 0..3 {
        replies.push(member.get::<Option<String>, _>(&key).await);
    }
    stop.store(true, Ordering::Relaxed);
    holder.join().expect("the counter holder must stop");
    counter(&member).store(0, Ordering::SeqCst);

    for (i, reply) in replies.into_iter().enumerate() {
        assert_eq!(
            reply
                .unwrap_or_else(|e| panic!(
                    "GET {i}, fed at idle, must be sent and answered: {e:?}"
                ))
                .as_deref(),
            Some("v")
        );
    }
    let _: () = member.del(&key).await.expect("the key must be deleted");
    client.quit().await;
}
