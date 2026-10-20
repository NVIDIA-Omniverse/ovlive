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

//! b.zt3 — one owed GC decrement takes one reference, even when its reply is
//! lost with its connection.
//!
//! fred re-sends a command that is in flight when its connection is lost (on
//! any reconnect, and since b.zt3 also after a fast unresponsive verdict about
//! 2 s into a stall). A `gc_decrement_and_collect` that ran but whose reply
//! was lost would then run a second time: with `ref_count` 2 the second
//! decrement reaches 0 and deletes the value a layer still references.
//!
//! The test stages exactly that through a [`ScriptReplyProxy`] in front of the
//! real store. A value with `ref_count` 2 owes one decrement, and the GC sweep
//! runs over a client connected through the proxy. The proxy forwards the
//! decrement and holds the store's reply ([`ProxyAction::HoldReplies`]); once
//! the reply has reached the proxy (the decrement has run), the test sends a
//! `GET` of a marker key on the same connection — the maintenance pool has one
//! member and the store is one node, so it is the one connection the sweep
//! uses — and the proxy closes that connection in the `GET`'s place
//! ([`ProxyAction::Close`]). The held reply is never delivered.
//!
//! It asserts the invariant only: afterwards `ref_count` is 1 and the data is
//! intact. How the sweep gets there — whether the call fails, whether the hash
//! stays claimed, how many attempts go out — is left open, so the test holds
//! for any at-most-once mechanism.
//!
//! The decrement keeps fred's default write attempts, so it is re-sent on the
//! rebuilt connection, and the re-send must find the member in its replay
//! record and answer `replayed` (b.yuo part H). Without that check — and on
//! the code before b.zt3, whose script had no record — the re-send decrements
//! again: `ref_count` reaches 0 and the data is deleted.
//!
//! # Wait budget
//!
//! Every wait at its full bound: the `TestValkey` fixture's connect (30 s);
//! the four setup writes — metadata, data, the owed release, the script load
//! — at fred's 5 s command timeout (20 s); the proxied client's connect
//! (30 s); the sweep beside the close — the sweep's commands at 5 s each,
//! one per script call (the first pass's selection, claim, decrement,
//! completion and audit page; the second's selection and the replay record's
//! unmark and retire; the third's selection: 45 s), against the held reply
//! reaching the proxy (`ReplyGate::reply_held`, 20 s) and the closing `GET`
//! (5 s), so 45 s; the settling `GET` (5 s); the assertions' two reads
//! (10 s); the closing `quit()` (11.5 s, the sum of its internal bounds: it
//! has no outer one); and the key purge (10 s):
//! 30 + 20 + 30 + 45 + 5 + 10 + 11.5 + 10 = 161.5 s. That is under the 180 s kill
//! deadline `.config/nextest.toml` sets for the `b_zt3_` modules (60 s × 3),
//! in place of the global 120 s hang-killer, which kills a test silently. On
//! the healthy path it runs in under a second.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use fred::interfaces::{HashesInterface, KeysInterface};

use live_edit::storage::external_values::{
    process_pending_gc, ExtRelease, GcPolicy, ValkeyExternalValueStore,
    GC_DECREMENT_AND_COLLECT_SCRIPT,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::test_support::TestValkey;

use super::snapshot_read_support::{
    client_through, test_valkey_addr, ProxyAction, ProxyRule, ReplyGate, ScriptReplyProxy,
};

/// The value's content hash.
const HASH: [u8; 32] = [0x5A; 32];

/// The retiring object whose release owes the one decrement.
const RETIRING_OBJECT: u64 = 7101;

/// The value's bytes.
const DATA: &[u8] = b"zt3-referenced-value";

/// A proxy rule that holds the replies on the connection carrying the first
/// `EVALSHA` naming `meta_key` (the decrement), and closes the connection
/// carrying the first `GET` of `close_marker`.
fn hold_decrement_then_close(meta_key: String, close_marker: String, gate: ReplyGate) -> ProxyRule {
    let held = AtomicBool::new(false);
    let closed = AtomicBool::new(false);
    Arc::new(move |args: &[Vec<u8>]| {
        let name = args.first()?.to_ascii_uppercase();
        if name == b"EVALSHA" && args.iter().any(|a| a == meta_key.as_bytes()) {
            return (!held.swap(true, Ordering::SeqCst))
                .then(|| ProxyAction::HoldReplies(gate.clone()));
        }
        if name == b"GET" && args.get(1).is_some_and(|k| k == close_marker.as_bytes()) {
            return (!closed.swap(true, Ordering::SeqCst)).then_some(ProxyAction::Close);
        }
        None
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_gc_decrement_whose_reply_is_lost_with_its_connection_takes_one_reference_and_keeps_the_data(
) {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema().clone();
    let hash_hex = sha256_to_hex(&HASH);
    let meta_key = keys.ext_meta(&hash_hex);
    let data_key = keys.ext_data(&hash_hex);
    let close_marker = format!("{}zt3:close-marker", tv.prefix());
    let _: () = tv
        .client()
        .pool
        .hset(&meta_key, [("ref_count", "2")])
        .await
        .expect("the value's metadata must be written");
    let _: () = tv
        .client()
        .pool
        .set(&data_key, DATA.to_vec(), None, None, false)
        .await
        .expect("the value's data must be written");
    ValkeyExternalValueStore::new(tv.client().clone(), keys.clone())
        .schedule_release(&[ExtRelease::retirement(HASH, RETIRING_OBJECT)])
        .await
        .expect("the one decrement must be owed");
    // Loaded up front, so the first decrement `EVALSHA` runs the script: a
    // held `NOSCRIPT` reply would stage nothing.
    GC_DECREMENT_AND_COLLECT_SCRIPT
        .load(tv.client().pool.next())
        .await
        .expect("the decrement script must load");
    let gate = ReplyGate::new();
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        hold_decrement_then_close(meta_key.clone(), close_marker.clone(), gate.clone()),
    )
    .await;
    let client = client_through(&proxy).await;
    assert_eq!(
        client.maintenance_pool.size(),
        1,
        "the closing GET must share the sweep's one maintenance connection"
    );
    let maintenance = client.maintenance_pool.next().clone();
    let store = ValkeyExternalValueStore::new(client.clone(), keys.clone());

    // `IMMEDIATE`: the release was enqueued just now, under the production
    // policy's age floor, and must be claimed in this sweep.
    let sweep = process_pending_gc(&store, None, GcPolicy::IMMEDIATE);
    let (_stats, _closing_get) = tokio::join!(sweep, async {
        gate.reply_held().await;
        // Its outcome depends on the maintenance pool's attempt budget, which
        // this test does not pin: only the close matters.
        maintenance.get::<Option<String>, _>(&close_marker).await
    });
    // Settle: a round trip behind anything still queued on the rebuilt
    // maintenance connection, so a late re-send has landed before the reads.
    let _: Option<String> = maintenance
        .get(&close_marker)
        .await
        .expect("the maintenance connection must be rebuilt");

    assert_eq!(
        proxy.hits(),
        2,
        "the decrement's reply must have been held and its connection closed"
    );
    let ref_count: Option<i64> = tv
        .client()
        .pool
        .hget(&meta_key, "ref_count")
        .await
        .expect("the metadata must be readable");
    assert_eq!(
        ref_count,
        Some(1),
        "the one owed decrement must take exactly one reference (2 -> 1), not be applied twice"
    );
    let data: Option<Vec<u8>> = tv
        .client()
        .pool
        .get(&data_key)
        .await
        .expect("the data key must be readable");
    assert_eq!(
        data.as_deref(),
        Some(DATA),
        "a value still referenced once must keep its data"
    );
    client.quit().await;
}
