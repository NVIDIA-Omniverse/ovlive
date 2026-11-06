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

//! b.myh — a reply is a cluster redirect only when it is an error frame.
//!
//! # The contract (SDD §15.3, vendored fred change 10)
//!
//! On a clustered connection, fred takes a reply for a `MOVED` / `ASK`
//! redirect only when it is a `SimpleError` or `BlobError` frame whose text
//! redis-protocol 6.0.0's rule accepts (a case-sensitive `MOVED` or `ASK`
//! prefix and exactly three space-separated tokens). Any other frame is
//! returned to its command as data, whatever it reads: no cluster sync, no
//! re-send, no `ASKING` and no connection to the host it names. The mechanism
//! is the vendored fred's `live_edit::redirect::is_redirection`, called from
//! `clustered::process_response_frame` (`vendor/fred-10.1.0.SOURCE.md` § 10).
//!
//! Stock fred 10.1.0 asked redis-protocol's text rule of every frame, data
//! included. A data reply that read like a redirect was lost: a `MOVED`-shaped
//! one re-synced the cluster and re-sent the command, and an `ASK`-shaped one,
//! with the vendored fred's `-ASK` follow (change 8), opened a connection to
//! the host the data named, with the client's store credentials.
//!
//! # The facade
//!
//! `cluster_facade_support`'s [`AskFacade`]: node A owns the slot of every key
//! here, and answers a trapped key with an `ASK` or `MOVED` text naming B or
//! the [`SilentNode`](crate::storage::cluster_facade_support::SilentNode)
//! (a listener that accepts and never answers), in one of three shapes: the
//! store's own reply forwarded (a stored value, a bulk string, `BlobString`),
//! a status reply (`SimpleString`), or an error reply (`SimpleError`). The
//! data member speaks RESP2 (checked), where those three arrive as
//! `BlobString`, `SimpleString` and `SimpleError` (`liveness::resp2_as_data`).
//!
//! Every data command runs at fred's default write budget (three attempts): a
//! one-attempt command answered `-ASK` fails "Max attempts reached." on stock
//! fred without connecting anywhere, so it would not tell a data reply taken
//! for a redirect from one returned as data.
//!
//! | Test | What it pins | Failed, in place, with |
//! |---|---|---|
//! | [`is_redirection_is_true_only_for_error_frames_and_keeps_redis_protocols_text_rule_on_them`] | `SimpleError` / `BlobError` holding canonical `ASK` and `MOVED` texts are redirects; `SimpleString`, `BlobString`, `VerbatimString`, `BigNumber`, `ChunkedString` holding them are not; on error frames, non-canonical texts answer what redis-protocol's own rule answers; a non-UTF-8 `BlobError` is not | the predicate reduced to stock's text-only rule: every data-frame row; forced to `false`: every canonical error-frame row and the `ASKING 1 …` rows |
//! | [`a_get_of_a_stored_value_reading_like_an_ask_redirect_returns_it_as_data_and_connects_nowhere`] | a stored `ASK <slot> 127.0.0.1:<silent>` is returned by the `GET`; the silent node accepted nothing; A received the `GET` once; no router restart | line 322 restored (and the predicate reduced to the text rule): `IO` "Failed connecting to ASK target 127.0.0.1:<silent>: Timeout Error …", at the 2 s cap |
//! | [`a_get_answered_by_a_status_reply_reading_like_an_ask_redirect_returns_it_as_data_and_connects_nowhere`] | the same text as A's status reply: the same | the same |
//! | [`a_get_of_a_stored_value_reading_like_a_moved_redirect_returns_it_as_data_and_is_sent_once`] | a stored `MOVED <slot> 127.0.0.1:<B>` is returned by the `GET`; A received the `GET` once | line 322 restored (and the predicate reduced to the text rule): not `Ok` ("Too many failed write attempts.") |
//! | [`an_ask_error_reply_is_still_followed_to_the_node_it_names`] | the control: the `ASK` text as A's error reply is followed; the silent node accepted a connection and the `GET` fails `IO` naming it | the predicate forced to `false`: the silent node accepted nothing |
//! | [`a_moved_error_reply_still_resyncs_the_cluster_and_resends_the_command`] | the control: the `MOVED` text as A's error reply is not answered as a value; A received the `GET` at least twice; the facade's `CLUSTER SLOTS` count rose | the predicate forced to `false`: A received the `GET` once |
//!
//! # Shown to fail
//!
//! Each perturbation was made in place, the module run, and the file restored
//! byte-identical:
//!
//! - `vendor/fred-10.1.0/src/router/clustered.rs:322` back to stock's
//!   `if frame.is_redirection() {`: the three data cases fail, as the table
//!   says; the controls and the pure table pass.
//! - `live_edit::redirect::is_redirection` forced to `false`: both controls and
//!   the pure table fail; the data cases pass.
//! - `live_edit::redirect::is_redirection` reduced to stock's text-only rule
//!   (`frame.is_redirection()`): the pure table and the three data cases fail;
//!   the controls pass.
//!
//! # Not staged
//!
//! - RESP3 data frames on a live connection (`VerbatimString`, `BigNumber`,
//!   `ChunkedString`, a RESP3 `BlobError`): the service's connections speak
//!   RESP2; the pure table covers those frame types.
//! - A data reply inside a fred `Transaction`: the transaction router decides
//!   by type already (`protocol::utils::frame_to_error`), unchanged by the fix.
//!
//! # Wait budget
//!
//! The facade client's connect under `CONNECT_BUDGET` (30 s), the direct
//! fixture's `SET` at fred's 5 s command timeout, then the one `GET` under
//! `SETTLE` (10 s): 45 s per facade test, inside nextest's 120 s
//! hang-killer. The healthy path takes well under a second, except the `ASK`
//! control, which waits out the 2 s connect cap by construction (and, on stock
//! fred, so do the two `ASK` data cases).

use fred::bytes::Bytes;
use fred::bytes_utils::Str;
use fred::clients::Client;
use fred::interfaces::{ClientLike, KeysInterface};
use fred::live_edit::redirect::is_redirection;
use fred::types::{Resp3Frame, RespVersion};
use redis_protocol::resp3::types::{Resp3Frame as _, VerbatimStringFormat};

use live_edit::storage::client::is_transient_fred_error;
use live_edit::storage::test_support::TestValkey;

use crate::storage::cluster_facade_support::{
    answered, assert_no_router_restart, hooked, key_on, settle, store_directly, value, AskFacade,
    Redirect, Target, TrapReply,
};

// ═════════════════════════════════════════════════════════════════════════════
// The pure table
// ═════════════════════════════════════════════════════════════════════════════

/// The redirect texts a cluster node sends, in their canonical form.
const CANONICAL: [&str; 2] = ["ASK 1234 127.0.0.1:6379", "MOVED 1234 127.0.0.1:6379"];

/// Texts redis-protocol's rule judges by its own letter: four tokens, the
/// wrong case, a bare `MOVED`, a prefixed error code, and `ASKING` (which
/// starts with `ASK` and has three tokens).
const NON_CANONICAL: [&str; 5] = [
    "ASK 1234 127.0.0.1:6379 extra",
    "ask 1234 127.0.0.1:6379",
    "MOVED",
    "ERR MOVED 1234 127.0.0.1:6379",
    "ASKING 1 127.0.0.1:6379",
];

/// `text` in each error frame type.
fn error_frames(text: &str) -> [(&'static str, Resp3Frame); 2] {
    [
        (
            "SimpleError",
            Resp3Frame::SimpleError {
                data: Str::from(text.to_owned()),
                attributes: None,
            },
        ),
        (
            "BlobError",
            Resp3Frame::BlobError {
                data: Bytes::from(text.to_owned()),
                attributes: None,
            },
        ),
    ]
}

/// `text` in each other string-carrying frame type.
fn data_frames(text: &str) -> [(&'static str, Resp3Frame); 5] {
    let bytes = || Bytes::from(text.to_owned());
    [
        (
            "SimpleString",
            Resp3Frame::SimpleString {
                data: bytes(),
                attributes: None,
            },
        ),
        (
            "BlobString",
            Resp3Frame::BlobString {
                data: bytes(),
                attributes: None,
            },
        ),
        (
            "VerbatimString",
            Resp3Frame::VerbatimString {
                data: bytes(),
                format: VerbatimStringFormat::Text,
                attributes: None,
            },
        ),
        (
            "BigNumber",
            Resp3Frame::BigNumber {
                data: bytes(),
                attributes: None,
            },
        ),
        ("ChunkedString", Resp3Frame::ChunkedString(bytes())),
    ]
}

/// `is_redirection` over every string-carrying RESP3 frame type: an error
/// frame holding a canonical `ASK` or `MOVED` text is a redirect, every other
/// frame type holding it is not; on error frames, every non-canonical text
/// gets the answer redis-protocol's own rule (`Resp3Frame::is_redirection`)
/// gives, so the error-frame rule is stock's, unchanged; and a `BlobError`
/// whose bytes read like a redirect but are not UTF-8 is not one.
#[test]
fn is_redirection_is_true_only_for_error_frames_and_keeps_redis_protocols_text_rule_on_them() {
    let mut wrong = Vec::new();
    for text in CANONICAL {
        for (kind, frame) in error_frames(text) {
            if !is_redirection(&frame) {
                wrong.push(format!("{kind} {text:?}: not a redirect, but it is one"));
            }
        }
        for (kind, frame) in data_frames(text) {
            assert!(
                frame.is_redirection(),
                "precondition: redis-protocol's text rule accepts {kind} {text:?}, so this row \
                 tells the type rule from it"
            );
            if is_redirection(&frame) {
                wrong.push(format!("{kind} {text:?}: a redirect, but it is data"));
            }
        }
    }
    let mut stock_answers = Vec::new();
    for text in NON_CANONICAL {
        for (kind, frame) in error_frames(text) {
            let stock = frame.is_redirection();
            stock_answers.push(stock);
            if is_redirection(&frame) != stock {
                wrong.push(format!(
                    "{kind} {text:?}: answered {}, where redis-protocol's rule answers {stock}",
                    !stock
                ));
            }
        }
    }
    assert!(
        stock_answers.contains(&true) && stock_answers.contains(&false),
        "precondition: redis-protocol's rule accepts some of the non-canonical texts and refuses \
         others, so a constant answer cannot match them all; got {stock_answers:?}"
    );
    let mut not_utf8 = b"ASK 1234 127.0.0.1:6379".to_vec();
    not_utf8.push(0xff);
    let frame = Resp3Frame::BlobError {
        data: Bytes::from(not_utf8),
        attributes: None,
    };
    if is_redirection(&frame) {
        wrong.push("a non-UTF-8 BlobError: a redirect, but it has no text".to_owned());
    }
    assert!(
        wrong.is_empty(),
        "a reply is a redirect only when it is an error frame that redis-protocol's text rule \
         accepts; these rows broke it: {wrong:#?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Through the facade
// ═════════════════════════════════════════════════════════════════════════════

/// Data member 0's client, after the fixture self-check that it speaks RESP2,
/// the protocol the service's connections speak, where a bulk string, a status
/// reply and an error reply arrive as `BlobString`, `SimpleString` and
/// `SimpleError`.
fn resp2_member(tv: &TestValkey) -> &Client {
    let member = tv.client().pool.member(0);
    assert_eq!(
        member.protocol_version(),
        RespVersion::RESP2,
        "fixture self-check: the data connection is RESP2"
    );
    member
}

/// The `GET`s naming `key` that A received.
fn gets_at_a(facade: &AskFacade, key: &str) -> usize {
    facade
        .a_log()
        .iter()
        .filter(|f| f.is("GET") && f.names(key))
        .count()
}

/// A default-budget `GET` on a key A owns, which A answers, in the shape
/// `reply` says, with `ASK <slot> 127.0.0.1:<silent>` (stored at the key
/// first when `reply` forwards to the store). It is returned to the `GET` as
/// data: the silent node accepted no connection, A received the `GET` once,
/// and no router was restarted.
async fn assert_ask_shaped_reply_returned_as_data(reply: TrapReply) {
    let facade = AskFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let direct = TestValkey::setup().await;
    let metrics = hooked(&tv).await;
    let key = key_on(&tv, "myh-ask-data", true);
    let text = facade.redirect_text(&key, Redirect::Ask, Target::Silent);
    if reply == TrapReply::Forward {
        store_directly(&direct, &key, &text).await;
    }
    let member = resp2_member(&tv);
    facade.trap_as(&key, Target::Silent, Redirect::Ask, reply);

    let got = settle(
        "the GET of an ASK-shaped value",
        member.get::<Option<String>, _>(&key),
    )
    .await;
    facade.disarm();

    assert_eq!(
        answered(got),
        value(&text),
        "a {reply:?} reply reading {text:?} is data and must be returned to the GET"
    );
    assert_eq!(
        facade.silent.accepted(),
        0,
        "no connection may be opened to the host a data reply names"
    );
    assert_eq!(
        gets_at_a(&facade, &key),
        1,
        "A received the GET once: nothing was re-sent"
    );
    assert_no_router_restart(&metrics);
}

/// A stored value reading `ASK <slot> 127.0.0.1:<silent>` (a bulk string,
/// `BlobString`) is returned by a default-budget `GET` as data, with no
/// connection to the host it names.
///
/// On stock fred the `GET` follows it as an `-ASK`: it opens a connection to
/// the silent node, which the 2 s cap cuts, and fails `IO` "Failed connecting
/// to ASK target 127.0.0.1:<silent>".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_get_of_a_stored_value_reading_like_an_ask_redirect_returns_it_as_data_and_connects_nowhere(
) {
    assert_ask_shaped_reply_returned_as_data(TrapReply::Forward).await;
}

/// A's status reply (`+ASK <slot> 127.0.0.1:<silent>`, `SimpleString`) to a
/// default-budget `GET` is returned as data, with no connection to the host
/// it names.
///
/// On stock fred it fails as the stored value's case does.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_get_answered_by_a_status_reply_reading_like_an_ask_redirect_returns_it_as_data_and_connects_nowhere(
) {
    assert_ask_shaped_reply_returned_as_data(TrapReply::StatusReply).await;
}

/// A stored value reading `MOVED <slot> 127.0.0.1:<B>` is returned by a
/// default-budget `GET` as data: A received the `GET` once.
///
/// On stock fred the `GET` takes it for a `-MOVED`: it re-syncs the cluster
/// and re-sends the command to the slot's owner, A, which answers the same
/// value again, until the command's budget runs out, so the `GET` is not
/// answered `Ok`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_get_of_a_stored_value_reading_like_a_moved_redirect_returns_it_as_data_and_is_sent_once()
{
    let facade = AskFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let direct = TestValkey::setup().await;
    let key = key_on(&tv, "myh-moved-data", true);
    let text = facade.redirect_text(&key, Redirect::Moved, Target::B);
    store_directly(&direct, &key, &text).await;
    let member = resp2_member(&tv);
    facade.trap_as(&key, Target::B, Redirect::Moved, TrapReply::Forward);

    let got = settle(
        "the GET of a MOVED-shaped value",
        member.get::<Option<String>, _>(&key),
    )
    .await;
    facade.disarm();

    assert_eq!(
        answered(got),
        value(&text),
        "a stored value reading {text:?} is data and must be returned to the GET"
    );
    assert_eq!(
        gets_at_a(&facade, &key),
        1,
        "A received the GET once: nothing was re-sent"
    );
}

/// The control for the `ASK` cases: the same text as A's error reply
/// (`-ASK <slot> 127.0.0.1:<silent>`) is still a redirect, followed to the
/// node it names. The silent node accepted a connection, and the `GET` fails
/// `IO` "Failed connecting to ASK target …" naming it, transiently.
///
/// Fails if error frames are no longer taken for redirects: the `GET` is
/// then answered the `-ASK` as a plain error, and nothing connects.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_ask_error_reply_is_still_followed_to_the_node_it_names() {
    let facade = AskFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let key = key_on(&tv, "myh-ask-error", true);
    let member = resp2_member(&tv);
    facade.trap_as(&key, Target::Silent, Redirect::Ask, TrapReply::ErrorFrame);

    let got = settle(
        "the GET answered -ASK",
        member.get::<Option<String>, _>(&key),
    )
    .await;
    facade.disarm();

    assert!(
        facade.silent.accepted() >= 1,
        "the -ASK error reply must be followed: a connection is opened to the node it names"
    );
    let silent = format!("127.0.0.1:{}", facade.silent.port);
    let err = got.expect_err("the node the -ASK names never answers, so the GET cannot be served");
    assert!(
        matches!(err.kind(), fred::error::ErrorKind::IO)
            && err.details().starts_with("Failed connecting to ASK target")
            && err.details().contains(&silent),
        "the GET fails as a connect to its ASK target cut short, naming it; got {err:?}"
    );
    assert!(
        is_transient_fred_error(&err),
        "…and transiently, so its caller retries it"
    );
}

/// The control for the `MOVED` case: the same text as A's error reply
/// (`-MOVED <slot> 127.0.0.1:<B>`) is still a redirect. The `GET` is not
/// answered the key's value, A received it at least twice (it was re-sent),
/// and the facade answered more `CLUSTER SLOTS` (the cluster was re-synced).
///
/// Asserts that the count rose, not by how much: how many syncs a redirect
/// runs is not this contract. B, the node the `-MOVED` names, does not own
/// the slot in the facade's table, so the re-send goes back to A.
///
/// Fails if error frames are no longer taken for redirects: the `GET` is
/// then answered the `-MOVED` as a plain error, sent once, with no sync.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_moved_error_reply_still_resyncs_the_cluster_and_resends_the_command() {
    let facade = AskFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let direct = TestValkey::setup().await;
    let key = key_on(&tv, "myh-moved-error", true);
    store_directly(&direct, &key, "myh-moved-error-value").await;
    let member = resp2_member(&tv);
    let syncs_before = facade.cluster_slots();
    facade.trap_as(&key, Target::B, Redirect::Moved, TrapReply::ErrorFrame);

    let got = settle(
        "the GET answered -MOVED",
        member.get::<Option<String>, _>(&key),
    )
    .await;
    facade.disarm();

    assert_ne!(
        answered(got),
        value("myh-moved-error-value"),
        "A answers every send -MOVED, so the GET cannot be answered the stored value"
    );
    assert!(
        gets_at_a(&facade, &key) >= 2,
        "the -MOVED error reply must be followed: the GET is re-sent; A received it {} times",
        gets_at_a(&facade, &key)
    );
    assert!(
        facade.cluster_slots() > syncs_before,
        "the -MOVED error reply must re-sync the cluster: CLUSTER SLOTS answered {} times before \
         and {} after",
        syncs_before,
        facade.cluster_slots()
    );
}
