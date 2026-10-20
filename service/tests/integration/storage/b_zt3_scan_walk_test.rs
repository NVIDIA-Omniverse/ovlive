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

//! b.zt3 postcomp-1 T-W0 — the premise of "a walk is complete only on its
//! terminal page" (SDD §11.3.2), on a real fred client: when a client's
//! router ends with a `SCAN` page in flight, fred ends the page stream with
//! **no error**, exactly as it ends a complete walk; and
//! `orphan_sweep::complete_pages` turns that same end into one error.
//!
//! Each half runs the same sequence on a fred client of its own, outside any
//! supervision, connected through a [`ScriptReplyProxy`] in front of the real
//! store: take page 1 (not the last), make the proxy stall every `SCAN` from
//! then on, drop page 1 so fred sends page 2 (which the proxy holds, so it is
//! in flight), then abort the client's router through its connect handle.
//! The proxy's `hits()` shows page 2 was in flight before the abort. Only this
//! test's clients send `SCAN` through its proxy.
//!
//! What the sweep, the reconciler and the checkpoint loop do with the error is
//! pinned in their own modules (`orphan_sweep`, `ext_reconcile`,
//! `checkpointer` unit tests), through a seam that ends a walk the same way.
//!
//! # Wait budget
//!
//! The fixture's connect (30 s), each client's connect ([`CONNECT_BUDGET`],
//! 30 s each), the seeding writes (5 s each at fred's command timeout, but
//! pipelined into one round of [`SEEDED_KEYS`] concurrent commands), each page
//! wait and each polled condition ([`SETTLE`]), and `TestValkey`'s purge
//! (10 s): 30 + 5 + 2 × (30 + 10 + 10 + 10) + 10 = 165 s, under the 180 s kill
//! deadline `.config/nextest.toml` sets for the `b_zt3_` modules. On the
//! healthy path the test runs in well under a second.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fred::clients::Client;
use fred::error::ErrorKind;
use fred::interfaces::{ClientLike, KeysInterface};
use fred::prelude::Builder;
use fred::types::config::Config as FredConfig;
use fred::types::scan::{ScanResult, Scanner};
use fred::types::ConnectHandle;
use futures::{Stream, StreamExt};
use tokio::time::Instant;

use live_edit::storage::orphan_sweep::complete_pages;
use live_edit::storage::test_support::TestValkey;

use super::b_zt3_support::{await_until, CONNECT_BUDGET};
use super::snapshot_read_support::{test_valkey_addr, ProxyAction, ProxyRule, ScriptReplyProxy};

/// Bound on each page wait and each polled condition.
const SETTLE: Duration = Duration::from_secs(10);

/// The `COUNT` hint of every page.
const PAGE: u32 = 10;

/// Keys seeded under the test's prefix: well above [`PAGE`], so page 1 is
/// not the walk's last whatever else the database holds.
const SEEDED_KEYS: usize = 200;

/// A proxy that stalls every `SCAN` while `stall` is set: the command is not
/// forwarded and its connection answers nothing more.
async fn scan_stalling_proxy(stall: &Arc<AtomicBool>) -> ScriptReplyProxy {
    let rule: ProxyRule = {
        let stall = Arc::clone(stall);
        Arc::new(move |args: &[Vec<u8>]| {
            (stall.load(Ordering::SeqCst) && args.first()?.eq_ignore_ascii_case(b"SCAN"))
                .then_some(ProxyAction::Stall)
        })
    };
    ScriptReplyProxy::start(test_valkey_addr(), rule).await
}

/// A fred client of its own through `proxy`, outside any supervision, and
/// its router's connect handle.
async fn unsupervised_client(proxy: &ScriptReplyProxy) -> (Client, ConnectHandle) {
    let config = FredConfig::from_url(&proxy.url()).expect("the proxy URL must parse");
    let client = Builder::from_config(config)
        .build()
        .expect("the client must build");
    let router = client.connect();
    tokio::time::timeout(CONNECT_BUDGET, client.wait_for_connect())
        .await
        .expect("the client must connect within the budget")
        .expect("the client must connect");
    (client, router)
}

/// Take page 1 off `walk` (asserting it is not the last), stall `SCAN`, drop
/// page 1 so page 2 is sent and held, wait until the proxy holds it (its
/// `hits()` reaching `hits_after`), then abort `router`.
async fn abort_with_page_2_in_flight<S>(
    walk: &mut std::pin::Pin<Box<S>>,
    stall: &AtomicBool,
    proxy: &ScriptReplyProxy,
    hits_after: usize,
    router: ConnectHandle,
) where
    S: Stream<Item = Result<ScanResult, fred::error::Error>>,
{
    let first = tokio::time::timeout(SETTLE, walk.next())
        .await
        .expect("page 1 must arrive")
        .expect("the walk must yield page 1")
        .expect("page 1 must be a page");
    assert!(
        first.has_more(),
        "precondition: page 1 is not the walk's last"
    );
    stall.store(true, Ordering::SeqCst);
    drop(first);
    await_until(
        "page 2's SCAN must reach the proxy and be held",
        Instant::now() + SETTLE,
        || proxy.hits() == hits_after,
    )
    .await;
    router.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_scan_stream_whose_router_is_aborted_with_a_page_in_flight_ends_silently_and_complete_pages_ends_it_with_an_error(
) {
    let tv = TestValkey::setup().await;
    let pattern = format!("{}zt3:scan-walk:*", tv.prefix());
    let writes = (0..SEEDED_KEYS).map(|i| {
        let pool = tv.client().pool.clone();
        let key = format!("{}zt3:scan-walk:{i}", tv.prefix());
        async move {
            let _: () = pool
                .set(&key, "v", None, None, false)
                .await
                .expect("seeding a key must succeed");
        }
    });
    futures::future::join_all(writes).await;
    let stall = Arc::new(AtomicBool::new(false));
    let proxy = scan_stalling_proxy(&stall).await;

    // ── The premise: fred's own stream ends with no error ───────────────
    let (raw_client, raw_router) = unsupervised_client(&proxy).await;
    let mut raw = Box::pin(raw_client.scan(pattern.clone(), Some(PAGE), None));
    abort_with_page_2_in_flight(&mut raw, &stall, &proxy, 1, raw_router).await;
    let end = tokio::time::timeout(SETTLE, raw.next())
        .await
        .expect("the stream must end once the router is gone, not hang");
    assert!(
        end.is_none(),
        "fred ends a SCAN stream whose page was dropped unanswered with no error, as it ends a \
         complete walk: {:?}",
        end.map(|item| item.map(|_| "a page"))
    );

    // ── The same end through `complete_pages`: one error ────────────────
    stall.store(false, Ordering::SeqCst);
    let (client, router) = unsupervised_client(&proxy).await;
    let mut walk = Box::pin(complete_pages(
        client.scan(pattern.clone(), Some(PAGE), None),
        1,
        &pattern,
    ));
    abort_with_page_2_in_flight(&mut walk, &stall, &proxy, 2, router).await;
    let rest: Vec<Result<ScanResult, fred::error::Error>> =
        tokio::time::timeout(SETTLE, walk.collect())
            .await
            .expect("the walk must end once the router is gone, not hang");
    let errors: Vec<&fred::error::Error> =
        rest.iter().filter_map(|item| item.as_ref().err()).collect();
    assert_eq!(
        (rest.len(), errors.len()),
        (1, 1),
        "the walk ends with exactly one item, the error"
    );
    assert_eq!(*errors[0].kind(), ErrorKind::Canceled, "{:?}", errors[0]);
    assert!(
        errors[0].details().contains(&format!(
            "SCAN walk `{pattern}` ended after 0 of 1 terminal page(s)"
        )),
        "{:?}",
        errors[0]
    );
}
