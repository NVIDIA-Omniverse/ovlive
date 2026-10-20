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

// Regression test: a reader must survive an XREAD BLOCK timeout with no new
// entries.
//
// Valkey's reply to a blocking XREAD that hits its BLOCK timeout with no new
// entries is the RESP Null value, not an empty array/map, and fred's
// `into_xread_response` errors on Null. A reader that treated that as a failure
// would end every idle subscription after one BLOCK period (30 s in
// production), visible to clients as a silent stream close.
//
// A 500 ms BLOCK (via the test knobs) makes one idle cycle cheap; the code path
// is the same as a 30 s idle.

use fred::interfaces::StreamsInterface;

use live_edit::storage::notifications::{consume_layer_stream, StreamCursor};
use live_edit::storage::stream_mux::StreamMuxKnobs;
use live_edit::storage::test_support::TestValkey;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn xread_block_consumer_survives_idle_timeout() {
    let tv = TestValkey::setup_with_readers(1).await;
    let object_id: u64 = 77_777;

    const BLOCK_MS: u64 = 500;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        block_ms: BLOCK_MS,
        ..tv.client().stream_mux.knobs()
    });

    let consumer = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From("0-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consume_layer_stream must succeed");

    // Two idle BLOCK timeouts with no writes: the reader re-issues its XREAD
    // after each, so a third read proves it survived both empty replies.
    let snap = super::mux_test_helpers::await_mux(
        tv.client(),
        "the reader re-issues its XREAD after each idle BLOCK timeout",
        |s| s.xreads() >= 3,
    )
    .await;
    assert!(
        consumer.take_terminal_exit().is_none(),
        "an idle BLOCK timeout must not end the subscription"
    );
    assert_eq!(
        snap.subscriptions(),
        1,
        "the idle subscription must still be registered"
    );

    let stream_key = tv.key_schema().layer_stream(object_id);
    let payload = "update|77777|1|probe|0|1";
    let id: String = tv
        .client()
        .pool
        .xadd(stream_key, false, None, "1-0", vec![("payload", payload)])
        .await
        .expect("XADD must succeed");
    assert_eq!(id, "1-0");

    // A deadline, not a measurement: it only has to catch a dead reader.
    let delivery = tokio::time::timeout(super::mux_test_helpers::SETTLE, consumer.recv()).await;
    match delivery {
        Ok(Some((got_id, _notif))) => assert_eq!(got_id, "1-0"),
        Ok(None) => panic!("the stream ended after idle BLOCK timeouts — the silent-close bug"),
        Err(_) => panic!(
            "no delivery within {:?} after XADD — the reader is dead or stalled",
            super::mux_test_helpers::SETTLE
        ),
    }

    consumer.unsubscribe().await;
}
