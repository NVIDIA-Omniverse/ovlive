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

//! b.zt3 — a pooled connection that goes silent while its shard keeps
//! answering other connections is closed by the fast unresponsive verdict
//! (its in-flight commands re-sent on the rebuilt connection), and a data
//! pool member whose client keeps failing that way is replaced by a fresh
//! client.
//!
//! Every test connects a `ValkeyClient` with a two-member data pool through a
//! [`ScriptReplyProxy`] in front of the real store. The proxy stalls the
//! connection that carries a `GET` of a marker key, once per marker
//! ([`stall_marker_gets`]): it forwards nothing more on that connection, so the
//! `GET` and everything behind it never get a reply, as on a connection a
//! managed frontend stopped answering without a FIN or RST. The marker is a
//! key only the test reads, through one chosen member, so the stall lands on
//! that member's connection and nowhere else; `hits()` shows it did. The other
//! member keeps reading every 200 ms (`b_zt3_support::PeerReads`): the
//! shard answering another connection is the evidence the fast verdict needs
//! (a standalone store is one node owning every slot). The production
//! unresponsive bounds apply throughout; only the replacement's retirement
//! grace is shortened ([`RETIRE_GRACE`]).
//!
//! | Test | Design item | What it pins |
//! |---|---|---|
//! | [`a_pooled_connection_silent_while_its_shard_answers_others_gets_a_fast_verdict_and_its_get_is_answered`] | CORE-1 | the fast verdict closes the silent connection within fast bound + scan interval + 1 s, and the stalled `GET` is answered with its value on the rebuilt connection inside the 5 s command timeout |
//! | [`a_one_attempt_get_on_a_silent_pooled_connection_fails_with_the_fast_verdict_instead_of_timing_out`] | CORE-1 | a command with no write attempt left fails with the fast verdict's error, not the command timeout |
//! | [`a_data_member_with_two_fast_verdicts_is_replaced_by_a_fresh_client_and_the_old_one_is_retired_unrestarted`] | CORE-2 | a new client id; the old client ended, not counted as a router restart; hooks and supervision moved to the fresh client; `replaced` 1, `fast` 2; the member serves; the readers' routing view follows member 0; `quit()` leaves no member watch |
//! | [`a_client_borrowed_from_a_member_before_its_replacement_fails_at_once_once_retired`] | CORE-2 | a `GET` and an `SSCAN` page sent through a `&Client` taken before the replacement fail at once after the retirement, instead of hanging |
//!
//! On fred 10.1.0 as published there is no fast verdict: the silent connection
//! is closed only at the 45 s slow bound, so the stalled `GET` times out at
//! 5 s and no verdict arrives within the bound; and there is no member
//! replacement, so the member keeps its client id.
//!
//! The fast-verdict bound in the first test, and the bound on a send through
//! a retired client in the last, are deadlines a hang would blow, not latency
//! measurements: the first is the guarantee b.zt3 exists for (a verdict that
//! leaves room for the re-send inside the 5 s command timeout), and the last
//! separates "fails at once" from fred's 5 s command timeout and from a scan
//! stream that never ends.
//!
//! # Wait budget
//!
//! Every test's worst case, every wait at its full bound — the `TestValkey`
//! fixture's connect (30 s) and the proxied client's (30 s),
//! fred's 5 s command timeout per command (the fixture's marker and set
//! writes included), the polled conditions ([`SETTLE`] each), the bounded
//! sends through a retired client ([`RETIRED_SEND_BOUND`] each), the closing
//! `quit()` (11.5 s, the sum of its internal bounds: it has no outer one), and
//! `TestValkey`'s key purge (a `SCAN` page and an
//! `UNLINK`, 5 s each) — is under the 180 s kill deadline `.config/nextest.toml`
//! sets for this module (60 s × 3), in place of the global 120 s hang-killer,
//! which kills a test silently. On the healthy path each runs in under six
//! seconds.
//!
//! | Test | Worst case |
//! |---|---|
//! | `a_pooled_connection_…_its_get_is_answered` | 30 + 5 + 30 + 5 + quit 11.5 + purge 10 = 91.5 s |
//! | `a_one_attempt_get_…_instead_of_timing_out` | 30 + 5 + 30 + 5 + quit 11.5 + purge 10 = 91.5 s |
//! | `a_data_member_with_two_fast_verdicts_…` | 30 + 5 × 2 + 30 + 5 × 2 + 15 × 2 + 5 + quit 11.5 + purge 10 = 136.5 s |
//! | `a_client_borrowed_…_fails_at_once_once_retired` | 30 + 5 × 2 + 5 + 30 + 5 × 2 + 15 × 2 + 1 × 2 + 5 + quit 11.5 + purge 10 = 143.5 s |

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use fred::clients::Client;
use fred::error::ErrorKind;
use fred::interfaces::{ClientLike, EventInterface, KeysInterface, SetsInterface};
use fred::live_edit::liveness::{verdict_of, Verdict};
use fred::types::config::Options;
use fred::types::ClientState;
use futures::StreamExt;
use tokio::time::Instant;

use live_edit::observability::metrics::{
    Metrics, VALKEY_MEMBER_REPLACEMENT_REPLACED, VALKEY_POOL_DATA, VALKEY_UNRESPONSIVE_VERDICT_FAST,
};
use live_edit::storage::client::{LivenessOverrides, ValkeyClient, POOL_LABEL_DATA};
use live_edit::storage::member_replacement::{
    ReplaceKnobs, MEMBER_REPLACE_AFTER_FAST_VERDICTS, MEMBER_REPLACE_COOLDOWN,
    MEMBER_REPLACE_WINDOW,
};
use live_edit::storage::test_support::{
    TestValkey, UNRESPONSIVE_CHECK_INTERVAL, UNRESPONSIVE_FAST_TIMEOUT,
};

use super::b_zt3_support::{
    await_until, proxied_client, stall_marker_gets, PeerReads, MEMBER_REPLACEMENTS,
    ROUTER_RESTARTS, UNRESPONSIVE,
};
use super::mux_test_helpers::read_series;
use super::snapshot_read_support::{test_valkey_addr, ScriptReplyProxy};

/// Bound on each polled condition: a replacement taking over its member, and
/// its retirement of the old client finishing (the [`RETIRE_GRACE`], then a
/// `QUIT` the rebuilt connection answers at once).
const SETTLE: Duration = Duration::from_secs(15);

/// The latest a fast verdict may close the silent connection, from the send
/// of the stalled command: the fast bound, one scan interval, and 1 s of slack
/// (3.5 s at the production values) — which leaves the re-send room inside the
/// 5 s command timeout.
const FAST_VERDICT_BOUND: Duration = UNRESPONSIVE_FAST_TIMEOUT
    .saturating_add(UNRESPONSIVE_CHECK_INTERVAL)
    .saturating_add(Duration::from_secs(1));

/// The replacement's retirement grace, shortened from the production command
/// timeout + 1 s so the retirement finishes inside [`SETTLE`].
const RETIRE_GRACE: Duration = Duration::from_secs(1);

/// The bound on a send through a retired client: well above anything an
/// in-process refusal takes, well below fred's 5 s command timeout.
const RETIRED_SEND_BOUND: Duration = Duration::from_secs(1);

/// The value every marker key holds, so a rescued `GET` is seen returning it.
const MARKER_VALUE: &str = "zt3-marker-value";

/// The data pool member the proxy silences.
const SILENT: usize = 0;

/// The data pool member that keeps reading meanwhile.
const PEER: usize = 1;

/// A store, a proxy in front of it that stalls the first `GET` of each marker,
/// and a client with a two-member data pool connected through the proxy.
struct Fixture {
    client: ValkeyClient,
    proxy: ScriptReplyProxy,
    /// Marker keys, each holding [`MARKER_VALUE`].
    markers: Vec<String>,
    /// A key only the peer member reads.
    peer_key: String,
    tv: TestValkey,
}

impl Fixture {
    async fn start(markers: usize) -> Self {
        let tv = TestValkey::setup().await;
        let markers: Vec<String> = (0..markers)
            .map(|i| format!("{}zt3:stalled-get:{i}", tv.prefix()))
            .collect();
        for marker in &markers {
            let _: () = tv
                .client()
                .pool
                .set(marker, MARKER_VALUE, None, None, false)
                .await
                .expect("the marker key must be written");
        }
        let proxy = ScriptReplyProxy::start(test_valkey_addr(), stall_marker_gets(&markers)).await;
        let client = proxied_client(&proxy, 2, LivenessOverrides::default()).await;
        let peer_key = format!("{}zt3:peer", tv.prefix());
        Self {
            client,
            proxy,
            markers,
            peer_key,
            tv,
        }
    }

    /// Start member [`PEER`] reading every 200 ms.
    fn peer_reads(&self) -> PeerReads {
        PeerReads::start(self.client.pool.member(PEER).clone(), self.peer_key.clone())
    }
}

/// Hooks with a fresh metrics registry, and the replacement trigger at its
/// production values with the retirement grace shortened to [`RETIRE_GRACE`].
async fn arm_replacement(client: &ValkeyClient) -> Arc<Metrics> {
    let metrics = Arc::new(crate::common::metrics_scrape_only("zt3"));
    client
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    client.set_member_replacement_knobs_for_test(ReplaceKnobs {
        after_fast_verdicts: MEMBER_REPLACE_AFTER_FAST_VERDICTS,
        window: MEMBER_REPLACE_WINDOW,
        cooldown: MEMBER_REPLACE_COOLDOWN,
        grace: RETIRE_GRACE,
    });
    metrics
}

/// Drive member [`SILENT`] to its second fast verdict — a `GET` of each of the
/// fixture's two markers through it, each stalled once, while member [`PEER`]
/// reads — and wait until the member names a fresh client and the
/// replacement's retirement of the old one has finished. Returns the fresh
/// client's id.
async fn replace_through_two_stalls(fx: &Fixture) -> String {
    let client = &fx.client;
    let old_id = ClientLike::id(client.pool.member(SILENT)).to_owned();
    let _peer = fx.peer_reads();
    for marker in &fx.markers {
        let value: Option<String> =
            client
                .pool
                .member(SILENT)
                .get(marker)
                .await
                .unwrap_or_else(|e| {
                    panic!("the stalled GET of {marker} must be re-sent and answered: {e:?}")
                });
        assert_eq!(value.as_deref(), Some(MARKER_VALUE));
    }
    assert_eq!(
        fx.proxy.hits(),
        2,
        "each marker GET must have been stalled once, on member {SILENT}'s connection"
    );
    await_until(
        &format!("member {SILENT} must be pointed at a fresh client after its second fast verdict"),
        Instant::now() + SETTLE,
        || ClientLike::id(client.pool.member(SILENT)) != old_id,
    )
    .await;
    // The replacement frees its pool's slot at the swap; its task runs on
    // through the old client's retirement (grace, hooks disarmed, `QUIT`,
    // release), so the retirement has finished once no replacement task is
    // left.
    await_until(
        "the replacement's retirement of the old client must finish",
        Instant::now() + SETTLE,
        || client.member_replacement_tasks_for_test() == 0,
    )
    .await;
    ClientLike::id(client.pool.member(SILENT)).to_owned()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pooled_connection_silent_while_its_shard_answers_others_gets_a_fast_verdict_and_its_get_is_answered(
) {
    let fx = Fixture::start(1).await;
    let _peer = fx.peer_reads();
    let silent = fx.client.pool.member(SILENT);
    let mut errors = silent.error_rx();

    let sent = Instant::now();
    let (reply, verdict) = tokio::join!(
        silent.get::<Option<String>, _>(&fx.markers[0]),
        tokio::time::timeout_at(sent + FAST_VERDICT_BOUND, errors.recv()),
    );

    let (error, _server) = verdict
        .unwrap_or_else(|_| {
            panic!("no verdict closed the silent connection within {FAST_VERDICT_BOUND:?}")
        })
        .expect("the member's error stream must stay open");
    assert_eq!(
        verdict_of(&error),
        Some(Verdict::Fast),
        "the silent connection must be closed by the fast verdict: {error:?}"
    );
    assert_eq!(
        reply
            .unwrap_or_else(|e| panic!(
                "the stalled GET must be re-sent and answered within the command timeout: {e:?}"
            ))
            .as_deref(),
        Some(MARKER_VALUE),
    );
    assert_eq!(fx.proxy.hits(), 1, "the marker GET must have been stalled");
    fx.client.quit().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_one_attempt_get_on_a_silent_pooled_connection_fails_with_the_fast_verdict_instead_of_timing_out(
) {
    let fx = Fixture::start(1).await;
    let _peer = fx.peer_reads();
    let one_attempt = fx.client.pool.member(SILENT).with_options(&Options {
        max_attempts: Some(1),
        ..Default::default()
    });

    let error = one_attempt
        .get::<Option<String>, _>(&fx.markers[0])
        .await
        .expect_err("a GET with no attempt left must fail when its connection is closed");

    // The command timeout (5 s) would fail it with a `Timeout` instead.
    assert_eq!(
        verdict_of(&error),
        Some(Verdict::Fast),
        "the GET must fail with the fast verdict's error: {error:?}"
    );
    assert_eq!(fx.proxy.hits(), 1, "the marker GET must have been stalled");
    fx.client.quit().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_data_member_with_two_fast_verdicts_is_replaced_by_a_fresh_client_and_the_old_one_is_retired_unrestarted(
) {
    let fx = Fixture::start(2).await;
    let client = &fx.client;
    let metrics = arm_replacement(client).await;
    let old: &Client = client.pool.member(SILENT);
    let old_id = ClientLike::id(old).to_owned();
    assert_eq!(client.stream_mux.routing_client_id_for_test(), old_id);
    assert!(client.armed_hook_client_ids_for_test().contains(&old_id));
    let data_restarts = || read_series(&metrics, ROUTER_RESTARTS, &[("pool", VALKEY_POOL_DATA)]);
    let restarts_before = data_restarts();

    let fresh_id = replace_through_two_stalls(&fx).await;

    assert_ne!(fresh_id, old_id);
    assert_eq!(
        old.state(),
        ClientState::Disconnected,
        "the retired client's router must have ended"
    );
    assert_eq!(
        data_restarts(),
        restarts_before,
        "ending the retired client's router must not count as a router restart"
    );
    assert_eq!(
        client.member_watches_for_test(POOL_LABEL_DATA).get(&SILENT),
        Some(&fresh_id),
        "the member's supervision must follow the fresh client"
    );
    let hooked = client.armed_hook_client_ids_for_test();
    assert!(
        hooked.contains(&fresh_id),
        "hooks must be armed on the fresh client: {hooked:?}"
    );
    assert!(
        !hooked.contains(&old_id),
        "hooks must be gone from the retired client: {hooked:?}"
    );
    assert_eq!(
        read_series(
            &metrics,
            MEMBER_REPLACEMENTS,
            &[
                ("pool", VALKEY_POOL_DATA),
                ("outcome", VALKEY_MEMBER_REPLACEMENT_REPLACED)
            ],
        ),
        1
    );
    assert_eq!(
        read_series(
            &metrics,
            UNRESPONSIVE,
            &[
                ("pool", VALKEY_POOL_DATA),
                ("verdict", VALKEY_UNRESPONSIVE_VERDICT_FAST)
            ],
        ),
        2
    );
    let value: Option<String> = client
        .pool
        .member(SILENT)
        .get(&fx.markers[0])
        .await
        .expect("the member must serve on its fresh client");
    assert_eq!(value.as_deref(), Some(MARKER_VALUE));
    assert_eq!(
        client.stream_mux.routing_client_id_for_test(),
        fresh_id,
        "the readers' routing view must follow data member 0 to its fresh client"
    );

    client.quit().await;
    assert_eq!(
        client.member_watches_for_test(POOL_LABEL_DATA),
        HashMap::new(),
        "quit() must leave no data member watch"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_client_borrowed_from_a_member_before_its_replacement_fails_at_once_once_retired() {
    let fx = Fixture::start(2).await;
    let client = &fx.client;
    let _metrics = arm_replacement(client).await;
    let set_key = format!("{}zt3:set", fx.tv.prefix());
    let _: () = fx
        .tv
        .client()
        .pool
        .sadd(&set_key, vec!["a", "b", "c"])
        .await
        .expect("the set must be written");
    let old: &Client = client.pool.member(SILENT);

    replace_through_two_stalls(&fx).await;

    let get = tokio::time::timeout(
        RETIRED_SEND_BOUND,
        old.get::<Option<String>, _>(&fx.markers[0]),
    )
    .await
    .unwrap_or_else(|_| {
        panic!("a GET through the retired client hung past {RETIRED_SEND_BOUND:?}")
    });
    let error = get.expect_err("a GET through the retired client must fail");
    assert_ne!(
        *error.kind(),
        ErrorKind::Timeout,
        "the GET must be refused, not left to the command timeout: {error:?}"
    );
    let mut pages = Box::pin(old.sscan(&set_key, "*", Some(10)));
    let page = tokio::time::timeout(RETIRED_SEND_BOUND, pages.next())
        .await
        .unwrap_or_else(|_| {
            panic!("an SSCAN page through the retired client hung past {RETIRED_SEND_BOUND:?}")
        });
    assert!(
        matches!(page, Some(Err(_))),
        "an SSCAN through the retired client must end in an error, not a page or a silent end: {:?}",
        page.map(|p| p.map(|_| ()))
    );
    // The same walk through the member's current client is served: the
    // refusal is the retirement's, not the key's.
    let mut fresh_pages = Box::pin(client.pool.member(SILENT).sscan(&set_key, "*", Some(10)));
    assert!(
        matches!(fresh_pages.next().await, Some(Ok(_))),
        "the member's fresh client must serve the SSCAN"
    );
    client.quit().await;
}
