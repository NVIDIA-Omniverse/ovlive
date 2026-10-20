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

//! b.zt3 T-RESP2 — on a RESP2 data connection, a reply shaped like a pubsub
//! message is the reply to its own command.
//!
//! Stock fred classifies any reply shaped like a pubsub frame (a three-element
//! array starting `"message"`, a three-element array starting `"subscribe"`,
//! …) as a pubsub message on every connection and drops it, and turns a nested
//! one into a `Push` frame. The command that asked for it never gets its
//! reply, and every later command on the connection gets the reply of the one
//! before it. The vendored fred classifies such a frame as pubsub only on a
//! connection that has sent a subscription command.
//!
//! Each data test runs its commands on one data-pool member, so they share one
//! connection, and checks that the command after the pubsub-shaped reply gets
//! its own reply. The service's pub/sub subscriber, which did subscribe, still
//! receives a real `PUBLISH`.
//!
//! | Test | What it pins |
//! |---|---|
//! | [`an_hmget_whose_values_read_message_ch_x_is_answered_and_the_next_command_gets_its_own_reply`] | `HMGET` returning `message`, `ch`, `x` |
//! | [`a_script_returning_a_subscribe_shaped_array_is_answered_and_the_next_command_gets_its_own_reply`] | an `EVAL` returning `{'subscribe','x',1}` |
//! | [`a_nested_message_shaped_array_stays_an_array`] | an `EVAL` returning `{{'message','a','b'}}` |
//! | [`the_pubsub_subscriber_still_receives_a_real_publish`] | the subscriber keeps stock classification |
//!
//! # Wait budget
//!
//! The `TestValkey` fixture's connect (30 s), its writes and commands at
//! fred's 5 s command timeout (at most five), the subscriber's subscribe and
//! the delivery (5 s each) and the key purge (10 s): at most 30 + 25 + 10 + 10
//! = 75 s, under the 120 s hang-killer.

use std::time::Duration;

use fred::clients::Client;
use fred::interfaces::{
    ClientLike, EventInterface, HashesInterface, KeysInterface, LuaInterface, PubsubInterface,
};
use fred::types::{RespVersion, Value};

use live_edit::storage::test_support::TestValkey;

/// Bound on the subscriber's delivery of the published message.
const DELIVERY: Duration = Duration::from_secs(5);

/// The value of the key every test reads after its pubsub-shaped reply.
const OWN: &str = "zt3-own-reply";

/// Data member 0's client, after the fixture self-check that it speaks RESP2:
/// over RESP3 a real pubsub message is a `Push` frame, so stock fred would
/// pass every data test here and none would pin anything.
fn resp2_member(tv: &TestValkey) -> &Client {
    let member = tv.client().pool.member(0);
    assert_eq!(
        member.protocol_version(),
        RespVersion::RESP2,
        "fixture self-check: the data connection is RESP2"
    );
    member
}

/// A key holding [`OWN`].
async fn own_reply_key(tv: &TestValkey) -> String {
    let key = format!("{}zt3:own", tv.prefix());
    let _: () = tv
        .client()
        .pool
        .set(&key, OWN, None, None, false)
        .await
        .expect("the key must be written");
    key
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_hmget_whose_values_read_message_ch_x_is_answered_and_the_next_command_gets_its_own_reply(
) {
    let tv = TestValkey::setup().await;
    let own = own_reply_key(&tv).await;
    let hash = format!("{}zt3:shaped", tv.prefix());
    let member = resp2_member(&tv);
    let _: () = member
        .hset(&hash, [("f1", "message"), ("f2", "ch"), ("f3", "x")])
        .await
        .expect("the hash must be written");

    let shaped: Vec<String> = member
        .hmget(&hash, vec!["f1", "f2", "f3"])
        .await
        .expect("the HMGET must be answered with its own reply");
    let next: String = member
        .get(&own)
        .await
        .expect("the next GET must be answered");

    assert_eq!(shaped, ["message", "ch", "x"]);
    assert_eq!(next, OWN, "the command after it must get its own reply");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_script_returning_a_subscribe_shaped_array_is_answered_and_the_next_command_gets_its_own_reply(
) {
    let tv = TestValkey::setup().await;
    let own = own_reply_key(&tv).await;
    let member = resp2_member(&tv);

    let shaped: Value = member
        .eval(
            "return {'subscribe','x',1}",
            Vec::<String>::new(),
            Vec::<String>::new(),
        )
        .await
        .expect("the EVAL must be answered with its own reply");
    let next: String = member
        .get(&own)
        .await
        .expect("the next GET must be answered");

    assert_eq!(
        shaped,
        Value::Array(vec![
            Value::from("subscribe"),
            Value::from("x"),
            Value::Integer(1)
        ])
    );
    assert_eq!(next, OWN, "the command after it must get its own reply");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_nested_message_shaped_array_stays_an_array() {
    let tv = TestValkey::setup().await;
    let own = own_reply_key(&tv).await;
    let member = resp2_member(&tv);

    let nested: Value = member
        .eval(
            "return {{'message','a','b'}}",
            Vec::<String>::new(),
            Vec::<String>::new(),
        )
        .await
        .expect("the EVAL must be answered with its own reply");
    let next: String = member
        .get(&own)
        .await
        .expect("the next GET must be answered");

    assert_eq!(
        nested,
        Value::Array(vec![Value::Array(vec![
            Value::from("message"),
            Value::from("a"),
            Value::from("b")
        ])])
    );
    assert_eq!(next, OWN, "the command after it must get its own reply");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_pubsub_subscriber_still_receives_a_real_publish() {
    let tv = TestValkey::setup().await;
    let channel = format!("{}zt3:channel", tv.prefix());
    let subscriber = &tv.client().subscriber;
    let mut messages = subscriber.message_rx();
    subscriber
        .subscribe(&channel)
        .await
        .expect("the subscriber must subscribe");

    let receivers: i64 = tv
        .client()
        .pool
        .next()
        .publish(&channel, "zt3-published")
        .await
        .expect("the PUBLISH must be answered");
    let message = tokio::time::timeout(DELIVERY, async {
        loop {
            let message = messages
                .recv()
                .await
                .expect("the message stream stays open");
            if message.channel == channel {
                return message;
            }
        }
    })
    .await
    .expect("the subscriber must receive the published message");

    assert_eq!(receivers, 1, "the subscriber is the channel's one receiver");
    assert_eq!(message.value.as_str().as_deref(), Some("zt3-published"));
    subscriber
        .unsubscribe(&channel)
        .await
        .expect("the subscriber must unsubscribe");
}
