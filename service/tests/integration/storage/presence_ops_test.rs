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

// Presence store operations (`storage::presence`, SDD SR-1.0–SR-1.7, SR-3.6,
// SR-3.8): the register / refresh / unregister scripts and their wrappers, the
// teardown marker `le:{sid}:presence:{cid}:gone`, the register deadline
// `not_after_ms` and the store-clock read `store_time_ms` it is built from, the
// `lease_expires_at_ms` info-hash field, the sweep's per-chunk removal
// `sweep_member_batch` and its `SWEEP_MEMBER_BATCH` bound, the paged
// member-info read `read_member_infos` and its `REPLAY_PAGE` bound, the
// index-removal helper `remove_session_from_index`, and the test-only eviction
// helper `evict_connection_for_test`.
//
// A sweep test makes a member stale by deleting its lease after registering it
// (`register_stale`), as the lease TTL running out would; its
// `lease_expires_at_ms` is then still in the future, so its removal delay is
// clamped to zero unless the test plants another stamp.
//
// The `register` helper passes the deadline a real caller computes (a fresh
// `store_time_ms` plus one lease TTL), so every test that registers through it
// uses the production deadline. The deadline tests call
// `register_with_deadline` with a chosen deadline. The ttl-guard and timeout
// tests call `register_connection` with `u64::MAX` (a deadline that never
// passes), because the deadline plays no part in what they check. The
// member-info read tests seed through `register_member`, which also calls the
// real `register_connection` with a production deadline, but gives every
// connection its own user name and application so a read that returns another
// connection's entry, or the right entries out of order, cannot pass.
//
// ── How events are observed ──────────────────────────────────────────────────
//
// Every test that cares about announcements opens an `EventTap`: a subscriber
// on the session's `presence_events_channel`, on the fixture's own pub/sub
// connection, which none of the functions under test uses (they PUBLISH from
// inside their scripts on the data pool). The one exception is the sweep's
// cluster variant, whose tap (`open_tap_on_another_node`) owns a subscriber
// connected directly to a node that does not own `{sid}`. "Exactly these
// payloads, and nothing else" is proven with a fence rather than a sleep:
// after the call under test returns, the tap PUBLISHes a unique fence message
// on the same channel and collects everything that arrives before it. A
// single node delivers one
// channel's messages to one subscriber in the order it executed the PUBLISHes,
// and the script's PUBLISH ran before its reply reached us, so anything the
// script published is guaranteed to be in the collected list. On a cluster
// that ordering holds only when the fence is published on the node that ran
// the script: one node's PUBLISHes reach a subscriber on another node in the
// order it executed them (the SR-8 spike,
// `docs-internal/validation/presence-publish-in-eval-20260925/report.md`). The
// sweep's cluster variant pins its fence to the `{sid}` owner and drains. The
// other cluster variants publish no fence and never drain: the presence-script
// variant waits for each expected event with `next()`, and the member-info
// read variant opens no tap.
//
// Payloads are built with `PresenceEvent::encode`, as the real callers will.
// The wrappers treat them as opaque strings, so every event assertion compares
// the received payload byte-for-byte with the one passed in.
//
// ── Waits ────────────────────────────────────────────────────────────────────
//
// Every wait is bounded by a named budget and returns as soon as its condition
// holds. Per-test worst cases against nextest's 120 s hang-killer:
//
//   * event tests: at most three fenced drains, each held to one
//     `EVENT_BUDGET` (10 s) for the whole drain → 30 s;
//   * timeout tests: `stall_data_pool` parks the fixture's four data-pool
//     connections one after another, waiting up to `CONN_STALL_WAIT` (10 s)
//     for each to block → 40 s; then one wrapper call under `SCENARIO_BUDGET`
//     (30 s; the healthy failing path is one 5 s command timeout) → 30 s; then
//     the release settles each stall under the same 10 s bound → 40 s. Total
//     110 s, under the 120 s hang-killer with 10 s to spare. The healthy path
//     spends about 5 s;
//   * sends-nothing tests (`read_member_infos` with an empty or over-long
//     slice; `sweep_member_batch` with an empty or over-long chunk, or a
//     payload count that does not match it): the same stall, one call under
//     `SCENARIO_BUDGET` and release as the timeout tests → 110 s. The healthy
//     path sends nothing and returns in milliseconds; a call that did send
//     would fail its 5 s command timeout;
//   * one-pipeline test: the same stall and release (40 s each), plus the page
//     reaching the stalled server and the released read settling, each held
//     to `PIPELINE_BUDGET` (10 s) → 100 s. The healthy path spends
//     milliseconds; a read that sent one id at a time returns after one 5 s
//     command timeout, which ends the wait at once;
//   * cluster variants: two event waits or two fenced drains → 20 s; the
//     member-info read variant waits on nothing beyond its own commands.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fred::clients::SubscriberClient;
use fred::interfaces::{
    ClientLike, ClusterInterface, EventInterface, HashesInterface, KeysInterface, PubsubInterface,
    SetsInterface,
};
use fred::types::config::{Config, Options, Server, ServerConfig};
use fred::types::{Builder, ClusterHash, CustomCommand, Expiration, RespVersion, Value};
use tokio::sync::mpsc;

use live_edit::observability::metrics::Metrics;
use live_edit::presence::{ConnectedUser, PresenceEvent};
use live_edit::storage::client::{FredErrOrigin, POOL_LABEL_DATA};
use live_edit::storage::presence::{
    evict_connection_for_test, read_member_infos, refresh_connection, register_connection,
    remove_session_from_index, store_time_ms, sweep_member_batch, unregister_connection,
    RefreshOutcome, RegisterOutcome, SweepBatch, TestEviction, Unregistered, REPLAY_PAGE,
    SWEEP_MEMBER_BATCH,
};
use live_edit::storage::test_support::{
    command_timeout_row, command_timeouts_only_on, release_all, ConnStall, TestValkey,
};

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

/// The lease TTL the tests register with (the production default). Long enough
/// that no key the tests inspect can expire mid-test.
const LEASE_TTL_SECS: u64 = 60;

/// A TTL well below [`LEASE_TTL_SECS`], planted with `EXPIRE` so a later read
/// can tell "the script reset the TTL" (back above this) from "the script left
/// it alone" (still at or below this).
const SHORTENED_TTL_SECS: i64 = 5;

/// The lowest `TTL` a key the call under test just set to [`LEASE_TTL_SECS`]
/// may read back. The five seconds of slack absorb test-runner load between
/// the write and the `TTL` read. Far above [`SHORTENED_TTL_SECS`], so a TTL
/// that was left alone, or set to some other value, cannot pass.
const FRESH_TTL_FLOOR_SECS: i64 = LEASE_TTL_SECS as i64 - 5;

/// The error text the connection scripts' own `ttl_secs` guard raises. Valkey
/// would reject `SET … EX 0` by itself too, so only this text proves the
/// script's guard (which runs before any write) is what refused the call.
const TTL_GUARD_MESSAGE: &str = "ttl_secs must be a positive integer";

/// A stale `lease_expires_at_ms` planted before a call, so a later read can
/// tell a rewrite from an untouched field.
const STALE_EXPIRES_AT: &str = "1";

/// Sanity floor for a millisecond epoch stamp (2020-09-13). Catches a stamp
/// written in seconds, which the TIME bracket alone would also catch but with
/// a far less readable failure.
const PLAUSIBLE_MS_FLOOR: u64 = 1_600_000_000_000;

/// Bound on one event wait (a fenced drain, or one expected event).
const EVENT_BUDGET: Duration = Duration::from_secs(10);

/// Bound on one wrapper call against a stalled data pool. The failing path is
/// one 5 s command timeout; the rest is headroom for runner load.
const SCENARIO_BUDGET: Duration = Duration::from_secs(30);

/// Bound on each of the one-pipeline test's two waits: the whole page reaching
/// the stalled server (milliseconds when healthy), and the read settling once
/// released (its failing path is one 5 s command timeout).
const PIPELINE_BUDGET: Duration = Duration::from_secs(10);

/// Interval between `tot-net-in` probes while waiting for the page to arrive.
const NET_IN_POLL: Duration = Duration::from_millis(5);

const USER_NAME: &str = "presence-ops-user";
const USER_APPLICATION: &str = "presence-ops-app";

// ═════════════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════════════

fn origin(tv: &TestValkey) -> FredErrOrigin<'_> {
    FredErrOrigin::client(tv.client(), POOL_LABEL_DATA)
}

fn joined_payload(cid: &str) -> String {
    PresenceEvent::PeerJoined(ConnectedUser {
        connection_id: cid.to_owned(),
        user_name: USER_NAME.to_owned(),
        user_application: USER_APPLICATION.to_owned(),
    })
    .encode()
}

fn left_payload(cid: &str) -> String {
    PresenceEvent::PeerLeft {
        connection_id: cid.to_owned(),
    }
    .encode()
}

/// Register with the deadline a real caller uses: a fresh [`store_time_ms`]
/// for the session plus one lease TTL. That deadline is still in the future
/// when the script runs, so this registers normally unless the marker exists.
async fn register(tv: &TestValkey, sid: &str, cid: &str) -> RegisterOutcome {
    let deadline = store_now_ms(tv, sid).await + LEASE_TTL_SECS * 1000;
    register_with_deadline(tv, sid, cid, deadline)
        .await
        .expect("register_connection must succeed")
}

async fn register_with_deadline(
    tv: &TestValkey,
    sid: &str,
    cid: &str,
    not_after_ms: u64,
) -> Result<RegisterOutcome, fred::error::Error> {
    register_connection(
        tv.client(),
        origin(tv),
        tv.key_schema(),
        sid,
        cid,
        USER_NAME,
        USER_APPLICATION,
        LEASE_TTL_SECS,
        not_after_ms,
        &joined_payload(cid),
    )
    .await
}

async fn store_now_ms(tv: &TestValkey, sid: &str) -> u64 {
    store_time_ms(tv.client(), origin(tv), tv.key_schema(), sid)
        .await
        .expect("store_time_ms must succeed")
}

/// Assert `result` is an error whose text carries the script guard's `message`.
fn assert_guard_error<T: std::fmt::Debug>(
    result: Result<T, fred::error::Error>,
    what: &str,
    message: &str,
) {
    match result {
        Ok(v) => panic!("{what} must be refused by the script's guard; got Ok({v:?})"),
        Err(e) => assert!(
            e.to_string().contains(message),
            "{what} must be refused by the script's own guard ({message:?}); got {e:?}"
        ),
    }
}

async fn refresh(tv: &TestValkey, sid: &str, cid: &str) -> RefreshOutcome {
    refresh_connection(
        tv.client(),
        origin(tv),
        tv.key_schema(),
        sid,
        cid,
        LEASE_TTL_SECS,
    )
    .await
    .expect("refresh_connection must succeed")
}

async fn unregister(tv: &TestValkey, sid: &str, cid: &str) -> Unregistered {
    unregister_connection(
        tv.client(),
        origin(tv),
        tv.key_schema(),
        sid,
        cid,
        LEASE_TTL_SECS,
        &left_payload(cid),
    )
    .await
    .expect("unregister_connection must succeed")
}

async fn evict(tv: &TestValkey, sid: &str, cid: &str) -> TestEviction {
    evict_connection_for_test(
        tv.client(),
        origin(tv),
        tv.key_schema(),
        sid,
        cid,
        &left_payload(cid),
    )
    .await
    .expect("evict_connection_for_test must succeed")
}

async fn is_member(tv: &TestValkey, sid: &str, cid: &str) -> bool {
    tv.client()
        .pool
        .sismember(tv.key_schema().presence_members(sid), cid)
        .await
        .expect("SISMEMBER must succeed")
}

async fn exists(tv: &TestValkey, key: &str) -> bool {
    tv.client()
        .pool
        .exists(key)
        .await
        .expect("EXISTS must succeed")
}

/// `TTL` in seconds: `-2` when the key is absent, `-1` when it has no TTL.
async fn ttl(tv: &TestValkey, key: &str) -> i64 {
    tv.client().pool.ttl(key).await.expect("TTL must succeed")
}

async fn shorten_ttl(tv: &TestValkey, key: &str) {
    let applied: bool = tv
        .client()
        .pool
        .expire(key, SHORTENED_TTL_SECS, None)
        .await
        .expect("EXPIRE must succeed");
    assert!(applied, "EXPIRE must find {key} to shorten");
}

async fn info(tv: &TestValkey, sid: &str, cid: &str) -> HashMap<String, String> {
    tv.client()
        .pool
        .hgetall(tv.key_schema().presence_user(sid, cid))
        .await
        .expect("HGETALL info must succeed")
}

async fn plant_stale_expires_at(tv: &TestValkey, sid: &str, cid: &str) {
    plant_expires_at(tv, sid, cid, STALE_EXPIRES_AT).await;
}

/// Overwrite `cid`'s `lease_expires_at_ms` with `raw`, verbatim.
async fn plant_expires_at(tv: &TestValkey, sid: &str, cid: &str, raw: &str) {
    let _: i64 = tv
        .client()
        .pool
        .hset(
            tv.key_schema().presence_user(sid, cid),
            ("lease_expires_at_ms", raw),
        )
        .await
        .expect("HSET lease_expires_at_ms must succeed");
}

/// Arm the teardown marker directly, as `presence_unregister.lua` would.
async fn plant_gone(tv: &TestValkey, sid: &str, cid: &str) {
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().presence_gone(sid, cid),
            "1",
            Some(Expiration::EX(LEASE_TTL_SECS as i64)),
            None,
            false,
        )
        .await
        .expect("SET gone marker must succeed");
}

async fn index_contains(tv: &TestValkey, sid: &str) -> bool {
    tv.client()
        .pool
        .sismember(tv.key_schema().presence_active_sessions(), sid)
        .await
        .expect("SISMEMBER index must succeed")
}

async fn index_add(tv: &TestValkey, sid: &str) {
    let _: i64 = tv
        .client()
        .pool
        .sadd(tv.key_schema().presence_active_sessions(), sid)
        .await
        .expect("SADD index must succeed");
}

/// Valkey's own clock in milliseconds, read with a plain unpinned `TIME`.
/// On the single-node fixture this is the same clock the scripts read, and it
/// is independent of [`store_time_ms`], which the round-trip test checks
/// against it.
async fn valkey_now_ms(tv: &TestValkey) -> u64 {
    let reply: Vec<String> = tv
        .client()
        .pool
        .custom(
            CustomCommand::new_static("TIME", ClusterHash::FirstKey, false),
            Vec::<Value>::new(),
        )
        .await
        .expect("TIME must succeed");
    let secs: u64 = reply[0].parse().expect("TIME seconds must parse");
    let usecs: u64 = reply[1].parse().expect("TIME microseconds must parse");
    secs * 1000 + usecs / 1000
}

/// Assert `raw` is a plain decimal millisecond stamp equal to Valkey's clock
/// plus the lease TTL at some instant within `[before_ms, after_ms]`.
fn assert_expires_at_bracketed(raw: &str, before_ms: u64, after_ms: u64) {
    assert!(
        !raw.is_empty() && raw.bytes().all(|b| b.is_ascii_digit()),
        "lease_expires_at_ms must be a plain decimal integer (no exponent, sign or \
         decimal point); got {raw:?}"
    );
    let stamp: u64 = raw.parse().expect("lease_expires_at_ms must parse as u64");
    assert!(
        stamp > PLAUSIBLE_MS_FLOOR,
        "lease_expires_at_ms must be epoch MILLISECONDS; got {stamp}"
    );
    let ttl_ms = LEASE_TTL_SECS * 1000;
    assert!(
        (before_ms + ttl_ms..=after_ms + ttl_ms).contains(&stamp),
        "lease_expires_at_ms must be the store's TIME + ttl during the call: expected within \
         [{}, {}], got {stamp}",
        before_ms + ttl_ms,
        after_ms + ttl_ms
    );
}

/// A subscriber on one session's presence-events channel. See the module
/// header for why the fence proves "nothing else was published".
struct EventTap {
    channel: String,
    rx: mpsc::UnboundedReceiver<String>,
    /// The node [`EventTap::drain`] publishes its fence on: `None` lets the
    /// pool pick (single node), `Some` pins it to the `{sid}` owner (cluster).
    fence_node: Option<Server>,
    /// A dedicated subscriber connection the tap owns, kept alive with it.
    _own_subscriber: Option<SubscriberClient>,
    _task: tokio::task::JoinHandle<()>,
}

impl EventTap {
    /// A tap on the fixture's own pub/sub connection.
    async fn open(tv: &TestValkey, sid: &str) -> Self {
        let channel = tv.key_schema().presence_events_channel(sid);
        Self::open_on(&tv.client().subscriber, channel, None, None).await
    }

    async fn open_on(
        subscriber: &SubscriberClient,
        channel: String,
        fence_node: Option<Server>,
        own_subscriber: Option<SubscriberClient>,
    ) -> Self {
        // Taken before SUBSCRIBE so no message can slip between the two.
        let mut messages = subscriber.message_rx();
        subscriber
            .subscribe(&channel)
            .await
            .expect("SUBSCRIBE on the presence-events channel must succeed");

        let (tx, rx) = mpsc::unbounded_channel();
        let wanted = channel.clone();
        let task = tokio::spawn(async move {
            while let Ok(msg) = messages.recv().await {
                if msg.channel.as_bytes() != wanted.as_bytes() {
                    continue;
                }
                let payload: String = msg
                    .value
                    .convert()
                    .expect("a presence-events payload must be a string");
                if tx.send(payload).is_err() {
                    break;
                }
            }
        });
        EventTap {
            channel,
            rx,
            fence_node,
            _own_subscriber: own_subscriber,
            _task: task,
        }
    }

    /// The next payload on the channel, within [`EVENT_BUDGET`].
    async fn next(&mut self) -> String {
        tokio::time::timeout(EVENT_BUDGET, self.rx.recv())
            .await
            .expect("an expected presence event did not arrive within the budget")
            .expect("the tap's forwarding task must stay alive")
    }

    /// Every payload published on the channel since the last drain, within one
    /// [`EVENT_BUDGET`] for the whole drain however many payloads precede the
    /// fence. On a cluster only with a `fence_node` (see the module header).
    async fn drain(&mut self, tv: &TestValkey) -> Vec<String> {
        static FENCES: AtomicU64 = AtomicU64::new(0);
        let fence = format!("fence|{}", FENCES.fetch_add(1, Ordering::Relaxed));
        let fence_options = Options {
            cluster_node: self.fence_node.clone(),
            ..Default::default()
        };
        let _: i64 = tv
            .client()
            .pool
            .next()
            .with_options(&fence_options)
            .publish(&self.channel, fence.as_str())
            .await
            .expect("PUBLISH of the fence must succeed");
        let rx = &mut self.rx;
        tokio::time::timeout(EVENT_BUDGET, async move {
            let mut seen = Vec::new();
            loop {
                let payload = rx
                    .recv()
                    .await
                    .expect("the tap's forwarding task must stay alive");
                if payload == fence {
                    return seen;
                }
                seen.push(payload);
            }
        })
        .await
        .expect("the published fence did not arrive within the budget")
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Register
// ═════════════════════════════════════════════════════════════════════════════

/// A first register writes the lease, the whole info hash and the membership,
/// and publishes exactly one `joined`, carrying the caller's payload verbatim.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_first_call_writes_lease_info_membership_and_publishes_joined_once() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid) = ("sess-register", "conn-1");
    let mut tap = EventTap::open(&tv, sid).await;

    let before = valkey_now_ms(&tv).await;
    let outcome = register(&tv, sid, cid).await;
    let after = valkey_now_ms(&tv).await;

    assert_eq!(outcome, RegisterOutcome::Registered { added: true });
    assert_eq!(
        tap.drain(&tv).await,
        vec![joined_payload(cid)],
        "a register that added the member publishes its joined payload exactly once"
    );
    assert!(is_member(&tv, sid, cid).await, "the member must be added");
    let lease_ttl = ttl(&tv, &ks.presence_lease(sid, cid)).await;
    assert!(
        (FRESH_TTL_FLOOR_SECS..=LEASE_TTL_SECS as i64).contains(&lease_ttl),
        "the lease must be set to the {LEASE_TTL_SECS}s lease TTL; got {lease_ttl}"
    );
    let info = info(&tv, sid, cid).await;
    assert_eq!(info.get("user_name").map(String::as_str), Some(USER_NAME));
    assert_eq!(
        info.get("user_application").map(String::as_str),
        Some(USER_APPLICATION)
    );
    assert_expires_at_bracketed(
        info.get("lease_expires_at_ms")
            .expect("register must write lease_expires_at_ms"),
        before,
        after,
    );
    assert_eq!(
        info.len(),
        3,
        "the info hash carries exactly user_name, user_application and lease_expires_at_ms; \
         got {info:?}"
    );
}

/// A second register for a connection that is still a member reports
/// `added: false` and publishes nothing, but still rewrites the lease to the
/// full lease TTL.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_existing_member_returns_not_added_resets_lease_and_publishes_nothing() {
    let tv = TestValkey::setup().await;
    let (sid, cid) = ("sess-register-twice", "conn-1");
    let lease_key = tv.key_schema().presence_lease(sid, cid);
    let mut tap = EventTap::open(&tv, sid).await;
    register(&tv, sid, cid).await;
    tap.drain(&tv).await;
    shorten_ttl(&tv, &lease_key).await;

    let outcome = register(&tv, sid, cid).await;

    assert_eq!(outcome, RegisterOutcome::Registered { added: false });
    assert_eq!(
        tap.drain(&tv).await,
        Vec::<String>::new(),
        "a register that did not add the member must not publish"
    );
    assert!(is_member(&tv, sid, cid).await);
    let lease_ttl = ttl(&tv, &lease_key).await;
    assert!(
        (FRESH_TTL_FLOOR_SECS..=LEASE_TTL_SECS as i64).contains(&lease_ttl),
        "a repeat register must reset the lease to {LEASE_TTL_SECS}s, not leave the shortened \
         {SHORTENED_TTL_SECS}s; got {lease_ttl}"
    );
}

/// The info-repair mode: a member whose info hash was lost is re-registered,
/// which rewrites the whole hash without announcing a second join.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_member_with_lost_info_rewrites_info_and_publishes_nothing() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid) = ("sess-register-repair", "conn-1");
    let mut tap = EventTap::open(&tv, sid).await;
    register(&tv, sid, cid).await;
    tap.drain(&tv).await;
    let _: i64 = tv
        .client()
        .pool
        .del(ks.presence_user(sid, cid))
        .await
        .expect("DEL info must succeed");

    let before = valkey_now_ms(&tv).await;
    let outcome = register(&tv, sid, cid).await;
    let after = valkey_now_ms(&tv).await;

    assert_eq!(outcome, RegisterOutcome::Registered { added: false });
    assert_eq!(tap.drain(&tv).await, Vec::<String>::new());
    let info = info(&tv, sid, cid).await;
    assert_eq!(info.get("user_name").map(String::as_str), Some(USER_NAME));
    assert_eq!(
        info.get("user_application").map(String::as_str),
        Some(USER_APPLICATION)
    );
    assert_expires_at_bracketed(
        info.get("lease_expires_at_ms")
            .expect("the repair must write lease_expires_at_ms"),
        before,
        after,
    );
}

/// The self-rejoin mode: a connection evicted by a sweep is re-registered,
/// which re-adds it and announces it again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_after_eviction_readds_member_and_publishes_joined() {
    let tv = TestValkey::setup().await;
    let (sid, cid) = ("sess-register-rejoin", "conn-1");
    let mut tap = EventTap::open(&tv, sid).await;
    register(&tv, sid, cid).await;
    evict(&tv, sid, cid).await;
    tap.drain(&tv).await;

    let outcome = register(&tv, sid, cid).await;

    assert_eq!(outcome, RegisterOutcome::Registered { added: true });
    assert_eq!(tap.drain(&tv).await, vec![joined_payload(cid)]);
    assert!(is_member(&tv, sid, cid).await);
}

/// While the teardown marker exists, register writes and publishes nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_with_teardown_marker_returns_gone_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid) = ("sess-register-gone", "conn-1");
    let mut tap = EventTap::open(&tv, sid).await;
    plant_gone(&tv, sid, cid).await;

    let outcome = register(&tv, sid, cid).await;

    assert_eq!(outcome, RegisterOutcome::Gone);
    assert_eq!(tap.drain(&tv).await, Vec::<String>::new());
    assert!(!is_member(&tv, sid, cid).await, "no member may be written");
    assert!(
        !exists(&tv, &ks.presence_user(sid, cid)).await,
        "no info hash may be written"
    );
    assert!(
        !exists(&tv, &ks.presence_lease(sid, cid)).await,
        "no lease may be written"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Register deadline (`not_after_ms`) and `store_time_ms`
// ═════════════════════════════════════════════════════════════════════════════

/// `store_time_ms` reads the store's clock: its value falls between two plain
/// `TIME` reads taken either side of it, in milliseconds.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn store_time_ms_returns_store_clock_in_milliseconds() {
    let tv = TestValkey::setup().await;

    let before = valkey_now_ms(&tv).await;
    let read = store_now_ms(&tv, "sess-store-time").await;
    let after = valkey_now_ms(&tv).await;

    assert!(
        read > PLAUSIBLE_MS_FLOOR,
        "store_time_ms must be epoch MILLISECONDS; got {read}"
    );
    assert!(
        (before..=after).contains(&read),
        "store_time_ms must be the store's TIME: expected within [{before}, {after}], got {read}"
    );
}

/// With no teardown marker, a register whose deadline is 1 ms before a fresh
/// store-time read is `Expired` (not `Gone`: no marker exists before or after)
/// and writes and publishes nothing. The script's own `TIME` is read after
/// that store-time read, so it is already past the deadline; no sleep needed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_past_deadline_without_marker_returns_expired_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid) = ("sess-register-expired", "conn-1");
    let mut tap = EventTap::open(&tv, sid).await;
    let deadline = store_now_ms(&tv, sid).await - 1;

    let outcome = register_with_deadline(&tv, sid, cid, deadline)
        .await
        .expect("register_connection must succeed");

    assert_eq!(outcome, RegisterOutcome::Expired);
    assert_eq!(tap.drain(&tv).await, Vec::<String>::new());
    assert!(!is_member(&tv, sid, cid).await, "no member may be written");
    assert!(
        !exists(&tv, &ks.presence_user(sid, cid)).await,
        "no info hash may be written"
    );
    assert!(
        !exists(&tv, &ks.presence_lease(sid, cid)).await,
        "no lease may be written"
    );
    assert!(
        !exists(&tv, &ks.presence_gone(sid, cid)).await,
        "the refusal came from the deadline, so no marker was involved or written"
    );
}

/// An `Expired` register for a connection that is still a member rewrites
/// nothing: the lease keeps its shortened TTL and `lease_expires_at_ms` keeps
/// its stale value. This pins the deadline check ahead of the `{1, 0}` path's
/// writes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_past_deadline_on_existing_member_returns_expired_and_rewrites_nothing() {
    let tv = TestValkey::setup().await;
    let (sid, cid) = ("sess-register-expired-member", "conn-1");
    let lease_key = tv.key_schema().presence_lease(sid, cid);
    let mut tap = EventTap::open(&tv, sid).await;
    register(&tv, sid, cid).await;
    tap.drain(&tv).await;
    shorten_ttl(&tv, &lease_key).await;
    plant_stale_expires_at(&tv, sid, cid).await;
    let deadline = store_now_ms(&tv, sid).await - 1;

    let outcome = register_with_deadline(&tv, sid, cid, deadline)
        .await
        .expect("register_connection must succeed");

    assert_eq!(outcome, RegisterOutcome::Expired);
    assert_eq!(tap.drain(&tv).await, Vec::<String>::new());
    assert!(
        is_member(&tv, sid, cid).await,
        "the membership is untouched"
    );
    let lease_ttl = ttl(&tv, &lease_key).await;
    assert!(
        (1..=SHORTENED_TTL_SECS).contains(&lease_ttl),
        "an Expired register must not renew the lease; got TTL {lease_ttl}"
    );
    assert_eq!(
        info(&tv, sid, cid)
            .await
            .get("lease_expires_at_ms")
            .map(String::as_str),
        Some(STALE_EXPIRES_AT),
        "an Expired register must not rewrite lease_expires_at_ms"
    );
}

/// A register whose deadline is one lease TTL after a fresh store-time read
/// (what a real caller passes) registers normally.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_future_deadline_registers_and_publishes_joined() {
    let tv = TestValkey::setup().await;
    let (sid, cid) = ("sess-register-deadline-ok", "conn-1");
    let mut tap = EventTap::open(&tv, sid).await;
    let deadline = store_now_ms(&tv, sid).await + LEASE_TTL_SECS * 1000;

    let outcome = register_with_deadline(&tv, sid, cid, deadline)
        .await
        .expect("register_connection must succeed");

    assert_eq!(outcome, RegisterOutcome::Registered { added: true });
    assert_eq!(tap.drain(&tv).await, vec![joined_payload(cid)]);
    assert!(is_member(&tv, sid, cid).await);
}

/// With both a teardown marker and a past deadline, the marker is checked
/// first and the outcome is `Gone`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_with_marker_and_past_deadline_returns_gone() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid) = ("sess-register-gone-expired", "conn-1");
    let mut tap = EventTap::open(&tv, sid).await;
    plant_gone(&tv, sid, cid).await;
    let deadline = store_now_ms(&tv, sid).await - 1;

    let outcome = register_with_deadline(&tv, sid, cid, deadline)
        .await
        .expect("register_connection must succeed");

    assert_eq!(outcome, RegisterOutcome::Gone);
    assert_eq!(tap.drain(&tv).await, Vec::<String>::new());
    assert!(!is_member(&tv, sid, cid).await);
    assert!(!exists(&tv, &ks.presence_user(sid, cid)).await);
    assert!(!exists(&tv, &ks.presence_lease(sid, cid)).await);
}

// ═════════════════════════════════════════════════════════════════════════════
// Refresh
// ═════════════════════════════════════════════════════════════════════════════

/// A fully registered connection's refresh resets the lease TTL (SET EX
/// semantics, not a preserved TTL) and rewrites `lease_expires_at_ms` from the
/// store's clock, without publishing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn refresh_registered_connection_resets_lease_ttl_and_advances_expires_at() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid) = ("sess-refresh", "conn-1");
    let lease_key = ks.presence_lease(sid, cid);
    let mut tap = EventTap::open(&tv, sid).await;
    register(&tv, sid, cid).await;
    tap.drain(&tv).await;
    shorten_ttl(&tv, &lease_key).await;
    plant_stale_expires_at(&tv, sid, cid).await;

    let before = valkey_now_ms(&tv).await;
    let outcome = refresh(&tv, sid, cid).await;
    let after = valkey_now_ms(&tv).await;

    assert_eq!(outcome, RefreshOutcome::Renewed);
    let lease_ttl = ttl(&tv, &lease_key).await;
    assert!(
        (FRESH_TTL_FLOOR_SECS..=LEASE_TTL_SECS as i64).contains(&lease_ttl),
        "refresh must reset the lease TTL to {LEASE_TTL_SECS}s, not preserve the shortened \
         {SHORTENED_TTL_SECS}s; got {lease_ttl}"
    );
    assert_expires_at_bracketed(
        info(&tv, sid, cid)
            .await
            .get("lease_expires_at_ms")
            .expect("refresh must keep lease_expires_at_ms"),
        before,
        after,
    );
    assert_eq!(
        tap.drain(&tv).await,
        Vec::<String>::new(),
        "refresh never publishes"
    );
}

/// A member with an intact info hash whose lease key alone is missing gets the
/// lease back: the renewal is a `SET … EX`, so it heals a lost lease rather
/// than no-opping against it (the b.k2t regression intent).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn refresh_member_with_missing_lease_recreates_lease() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid) = ("sess-refresh-heal", "conn-1");
    let lease_key = ks.presence_lease(sid, cid);
    register(&tv, sid, cid).await;
    let _: i64 = tv
        .client()
        .pool
        .del(&lease_key)
        .await
        .expect("DEL lease must succeed");

    let outcome = refresh(&tv, sid, cid).await;

    assert_eq!(outcome, RefreshOutcome::Renewed);
    let lease_ttl = ttl(&tv, &lease_key).await;
    assert!(
        (FRESH_TTL_FLOOR_SECS..=LEASE_TTL_SECS as i64).contains(&lease_ttl),
        "refresh must re-create the lease with the {LEASE_TTL_SECS}s lease TTL; got {lease_ttl}"
    );
}

/// The realistic post-sweep state: the member and its info hash are gone but
/// the lease survives (a sweep does not touch it). Refresh is `Evicted`, not
/// `InfoMissing`, because membership is checked before the info hash, and it
/// leaves the surviving lease alone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn refresh_after_sweep_eviction_returns_evicted_and_leaves_lease_alone() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid) = ("sess-refresh-after-evict", "conn-1");
    let lease_key = ks.presence_lease(sid, cid);
    register(&tv, sid, cid).await;
    evict(&tv, sid, cid).await;
    shorten_ttl(&tv, &lease_key).await;

    let outcome = refresh(&tv, sid, cid).await;

    assert_eq!(outcome, RefreshOutcome::Evicted);
    let lease_ttl = ttl(&tv, &lease_key).await;
    assert!(
        (1..=SHORTENED_TTL_SECS).contains(&lease_ttl),
        "an Evicted refresh must not renew the surviving lease; got TTL {lease_ttl}"
    );
    assert!(
        !exists(&tv, &ks.presence_user(sid, cid)).await,
        "an Evicted refresh must not create the info hash"
    );
}

/// A connection that is not a member is `Evicted`, and no lease is created for
/// it — even when an info hash with a `user_name` exists, so the membership
/// check (not the info check) is what refused it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn refresh_non_member_returns_evicted_and_creates_no_lease() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid) = ("sess-refresh-evicted", "conn-1");
    let info_key = ks.presence_user(sid, cid);
    let _: i64 = tv
        .client()
        .pool
        .hset(&info_key, ("user_name", USER_NAME))
        .await
        .expect("HSET info must succeed");

    let outcome = refresh(&tv, sid, cid).await;

    assert_eq!(outcome, RefreshOutcome::Evicted);
    assert!(
        !exists(&tv, &ks.presence_lease(sid, cid)).await,
        "refresh must not create a lease for a non-member"
    );
    assert_eq!(
        info(&tv, sid, cid).await,
        HashMap::from([("user_name".to_owned(), USER_NAME.to_owned())]),
        "refresh must not write lease_expires_at_ms for a non-member"
    );
}

/// A member whose info hash is gone is `InfoMissing`: refresh neither
/// re-creates the hash nor renews the lease.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn refresh_member_with_deleted_info_returns_info_missing_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid) = ("sess-refresh-no-info", "conn-1");
    let lease_key = ks.presence_lease(sid, cid);
    let info_key = ks.presence_user(sid, cid);
    register(&tv, sid, cid).await;
    let _: i64 = tv
        .client()
        .pool
        .del(&info_key)
        .await
        .expect("DEL info must succeed");
    shorten_ttl(&tv, &lease_key).await;

    let outcome = refresh(&tv, sid, cid).await;

    assert_eq!(outcome, RefreshOutcome::InfoMissing);
    assert!(
        !exists(&tv, &info_key).await,
        "refresh must never create the info hash"
    );
    let lease_ttl = ttl(&tv, &lease_key).await;
    assert!(
        (1..=SHORTENED_TTL_SECS).contains(&lease_ttl),
        "an InfoMissing refresh must not renew the lease; got TTL {lease_ttl}"
    );
}

/// A member whose info hash lacks `user_name` is `InfoMissing`, and the hash
/// is left exactly as it was.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn refresh_member_without_user_name_returns_info_missing_and_leaves_info_unchanged() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid) = ("sess-refresh-no-name", "conn-1");
    let lease_key = ks.presence_lease(sid, cid);
    register(&tv, sid, cid).await;
    let _: i64 = tv
        .client()
        .pool
        .hdel(ks.presence_user(sid, cid), "user_name")
        .await
        .expect("HDEL user_name must succeed");
    shorten_ttl(&tv, &lease_key).await;
    let info_before = info(&tv, sid, cid).await;

    let outcome = refresh(&tv, sid, cid).await;

    assert_eq!(outcome, RefreshOutcome::InfoMissing);
    assert_eq!(
        info(&tv, sid, cid).await,
        info_before,
        "an InfoMissing refresh must leave the info hash unchanged"
    );
    let lease_ttl = ttl(&tv, &lease_key).await;
    assert!(
        (1..=SHORTENED_TTL_SECS).contains(&lease_ttl),
        "an InfoMissing refresh must not renew the lease; got TTL {lease_ttl}"
    );
}

/// While the teardown marker exists, refresh is `Gone` — even for a connection
/// that is otherwise fully registered — and writes nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn refresh_with_teardown_marker_returns_gone_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid) = ("sess-refresh-gone", "conn-1");
    let lease_key = ks.presence_lease(sid, cid);
    register(&tv, sid, cid).await;
    shorten_ttl(&tv, &lease_key).await;
    plant_stale_expires_at(&tv, sid, cid).await;
    plant_gone(&tv, sid, cid).await;

    let outcome = refresh(&tv, sid, cid).await;

    assert_eq!(outcome, RefreshOutcome::Gone);
    let lease_ttl = ttl(&tv, &lease_key).await;
    assert!(
        (1..=SHORTENED_TTL_SECS).contains(&lease_ttl),
        "a Gone refresh must not renew the lease; got TTL {lease_ttl}"
    );
    assert_eq!(
        info(&tv, sid, cid)
            .await
            .get("lease_expires_at_ms")
            .map(String::as_str),
        Some(STALE_EXPIRES_AT),
        "a Gone refresh must not rewrite lease_expires_at_ms"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Unregister
// ═════════════════════════════════════════════════════════════════════════════

/// Unregister removes the member, its info hash and lease, arms the teardown
/// marker for one lease TTL, reports the remaining count, and publishes one
/// `left`. A sibling connection is untouched.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unregister_member_removes_state_arms_marker_and_publishes_left_once() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid, sibling) = ("sess-unregister", "conn-1", "conn-2");
    let mut tap = EventTap::open(&tv, sid).await;
    register(&tv, sid, cid).await;
    register(&tv, sid, sibling).await;
    tap.drain(&tv).await;

    let outcome = unregister(&tv, sid, cid).await;

    assert_eq!(
        outcome,
        Unregistered {
            removed: true,
            remaining: 1
        }
    );
    assert_eq!(tap.drain(&tv).await, vec![left_payload(cid)]);
    assert!(
        !is_member(&tv, sid, cid).await,
        "the member must be removed"
    );
    assert!(
        !exists(&tv, &ks.presence_user(sid, cid)).await,
        "the info hash must be deleted"
    );
    assert!(
        !exists(&tv, &ks.presence_lease(sid, cid)).await,
        "the lease must be deleted"
    );
    let marker_ttl = ttl(&tv, &ks.presence_gone(sid, cid)).await;
    assert!(
        (FRESH_TTL_FLOOR_SECS..=LEASE_TTL_SECS as i64).contains(&marker_ttl),
        "the teardown marker must live one lease TTL ({LEASE_TTL_SECS}s); got {marker_ttl}"
    );
    assert!(
        is_member(&tv, sid, sibling).await
            && exists(&tv, &ks.presence_user(sid, sibling)).await
            && exists(&tv, &ks.presence_lease(sid, sibling)).await,
        "unregister must not touch another connection's state"
    );
}

/// Unregistering the last member reports `remaining: 0`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unregister_last_member_reports_zero_remaining() {
    let tv = TestValkey::setup().await;
    let (sid, cid) = ("sess-unregister-last", "conn-1");
    register(&tv, sid, cid).await;

    let outcome = unregister(&tv, sid, cid).await;

    assert_eq!(
        outcome,
        Unregistered {
            removed: true,
            remaining: 0
        }
    );
}

/// A repeated unregister removes and publishes nothing and does not fail, but
/// re-arms the marker (restarting its TTL).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unregister_repeated_call_removes_nothing_publishes_nothing_and_rearms_marker() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid, sibling) = ("sess-unregister-twice", "conn-1", "conn-2");
    let gone_key = ks.presence_gone(sid, cid);
    let mut tap = EventTap::open(&tv, sid).await;
    register(&tv, sid, cid).await;
    register(&tv, sid, sibling).await;
    unregister(&tv, sid, cid).await;
    tap.drain(&tv).await;
    shorten_ttl(&tv, &gone_key).await;

    let outcome = unregister(&tv, sid, cid).await;

    assert_eq!(
        outcome,
        Unregistered {
            removed: false,
            remaining: 1
        }
    );
    assert_eq!(
        tap.drain(&tv).await,
        Vec::<String>::new(),
        "an unregister that removed nothing must not publish"
    );
    let marker_ttl = ttl(&tv, &gone_key).await;
    assert!(
        (FRESH_TTL_FLOOR_SECS..=LEASE_TTL_SECS as i64).contains(&marker_ttl),
        "a repeated unregister must re-arm the marker to {LEASE_TTL_SECS}s; got {marker_ttl}"
    );
}

/// Unregister after a sweep already evicted the connection: the sweep
/// announced the removal, so unregister removes and publishes nothing. It
/// still deletes whatever the connection left behind, unconditionally: the
/// lease the sweep leaves, and an info hash (planted here after the eviction)
/// — and it arms the marker.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unregister_after_sweep_eviction_publishes_nothing_and_clears_leftovers() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid) = ("sess-unregister-after-evict", "conn-1");
    let mut tap = EventTap::open(&tv, sid).await;
    register(&tv, sid, cid).await;
    evict(&tv, sid, cid).await;
    tap.drain(&tv).await;
    assert!(
        exists(&tv, &ks.presence_lease(sid, cid)).await,
        "precondition: the eviction leaves the lease behind"
    );
    let _: i64 = tv
        .client()
        .pool
        .hset(ks.presence_user(sid, cid), ("user_name", USER_NAME))
        .await
        .expect("HSET info must succeed");

    let outcome = unregister(&tv, sid, cid).await;

    assert_eq!(
        outcome,
        Unregistered {
            removed: false,
            remaining: 0
        }
    );
    assert_eq!(
        tap.drain(&tv).await,
        Vec::<String>::new(),
        "the sweep already announced this removal; unregister must not announce it again"
    );
    assert!(
        !exists(&tv, &ks.presence_lease(sid, cid)).await,
        "unregister must delete the lease the sweep left"
    );
    assert!(
        !exists(&tv, &ks.presence_user(sid, cid)).await,
        "unregister must delete the info hash even for a non-member"
    );
    let marker_ttl = ttl(&tv, &ks.presence_gone(sid, cid)).await;
    assert!(
        (FRESH_TTL_FLOOR_SECS..=LEASE_TTL_SECS as i64).contains(&marker_ttl),
        "unregister must arm the marker for {LEASE_TTL_SECS}s; got {marker_ttl}"
    );
}

/// The marker unregister arms fences both late writers: a register and a
/// refresh that reach the store after teardown are `Gone` and resurrect
/// nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unregister_marker_fences_late_register_and_refresh() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid) = ("sess-unregister-fence", "conn-1");
    let mut tap = EventTap::open(&tv, sid).await;
    register(&tv, sid, cid).await;
    unregister(&tv, sid, cid).await;
    tap.drain(&tv).await;

    let registered = register(&tv, sid, cid).await;
    let refreshed = refresh(&tv, sid, cid).await;

    assert_eq!(registered, RegisterOutcome::Gone);
    assert_eq!(refreshed, RefreshOutcome::Gone);
    assert_eq!(tap.drain(&tv).await, Vec::<String>::new());
    assert!(!is_member(&tv, sid, cid).await);
    assert!(!exists(&tv, &ks.presence_user(sid, cid)).await);
    assert!(!exists(&tv, &ks.presence_lease(sid, cid)).await);
}

// ═════════════════════════════════════════════════════════════════════════════
// Script argument validation
// ═════════════════════════════════════════════════════════════════════════════

/// A zero lease TTL is rejected by each connection script's own guard before
/// it writes anything: no marker is armed and the existing registration is
/// untouched. The error text is checked because Valkey would also reject a
/// `SET … EX 0` on its own, so a failure alone does not prove the guard ran.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connection_scripts_reject_zero_ttl_before_writing() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid, fresh) = ("sess-zero-ttl", "conn-1", "conn-2");
    register(&tv, sid, cid).await;
    let info_before = info(&tv, sid, cid).await;

    let registered = register_connection(
        tv.client(),
        origin(&tv),
        ks,
        sid,
        fresh,
        USER_NAME,
        USER_APPLICATION,
        0,
        u64::MAX,
        &joined_payload(fresh),
    )
    .await;
    let refreshed = refresh_connection(tv.client(), origin(&tv), ks, sid, cid, 0).await;
    let unregistered = unregister_connection(
        tv.client(),
        origin(&tv),
        ks,
        sid,
        cid,
        0,
        &left_payload(cid),
    )
    .await;

    assert_guard_error(registered, "register with ttl 0", TTL_GUARD_MESSAGE);
    assert_guard_error(refreshed, "refresh with ttl 0", TTL_GUARD_MESSAGE);
    assert_guard_error(unregistered, "unregister with ttl 0", TTL_GUARD_MESSAGE);
    assert!(
        !is_member(&tv, sid, fresh).await,
        "the rejected register wrote no member"
    );
    assert!(!exists(&tv, &ks.presence_user(sid, fresh)).await);
    assert!(!exists(&tv, &ks.presence_lease(sid, fresh)).await);
    assert!(
        is_member(&tv, sid, cid).await,
        "the rejected unregister removed nothing"
    );
    assert_eq!(info(&tv, sid, cid).await, info_before);
    assert!(
        !exists(&tv, &ks.presence_gone(sid, cid)).await,
        "the rejected unregister armed no marker"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Index removal (SR-3.8)
// ═════════════════════════════════════════════════════════════════════════════

/// A session that still has members ends indexed, and the helper says so.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remove_session_from_index_with_members_keeps_session_indexed() {
    let tv = TestValkey::setup().await;
    let (sid, cid) = ("sess-index-live", "conn-1");
    register(&tv, sid, cid).await;
    index_add(&tv, sid).await;

    let indexed = remove_session_from_index(tv.client(), origin(&tv), tv.key_schema(), sid)
        .await
        .expect("remove_session_from_index must succeed");

    assert!(indexed, "the helper must report the session re-added");
    assert!(
        index_contains(&tv, sid).await,
        "a session with members must end indexed"
    );
}

/// A session with an empty member set ends unindexed, and other sessions'
/// entries are left alone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remove_session_from_index_without_members_unindexes_only_that_session() {
    let tv = TestValkey::setup().await;
    let (sid, other) = ("sess-index-empty", "sess-index-other");
    index_add(&tv, sid).await;
    index_add(&tv, other).await;

    let indexed = remove_session_from_index(tv.client(), origin(&tv), tv.key_schema(), sid)
        .await
        .expect("remove_session_from_index must succeed");

    assert!(!indexed, "the helper must report the session left out");
    assert!(
        !index_contains(&tv, sid).await,
        "a session with no members must end unindexed"
    );
    assert!(
        index_contains(&tv, other).await,
        "another session's index entry must survive"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Test-only eviction helper
// ═════════════════════════════════════════════════════════════════════════════

/// The eviction helper removes the member and its info hash and publishes one
/// `left`, touching neither the lease nor the teardown marker. A second call
/// removes and publishes nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn evict_connection_for_test_removes_member_and_info_and_publishes_left_once() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid, sibling) = ("sess-evict", "conn-1", "conn-2");
    let mut tap = EventTap::open(&tv, sid).await;
    register(&tv, sid, cid).await;
    register(&tv, sid, sibling).await;
    tap.drain(&tv).await;

    let first = evict(&tv, sid, cid).await;

    assert_eq!(
        first,
        TestEviction {
            removed: true,
            remaining: 1
        }
    );
    assert_eq!(tap.drain(&tv).await, vec![left_payload(cid)]);
    assert!(!is_member(&tv, sid, cid).await);
    assert!(!exists(&tv, &ks.presence_user(sid, cid)).await);
    assert!(
        exists(&tv, &ks.presence_lease(sid, cid)).await,
        "the eviction helper must not touch the lease"
    );
    assert!(
        !exists(&tv, &ks.presence_gone(sid, cid)).await,
        "the eviction helper must not arm the teardown marker"
    );

    let second = evict(&tv, sid, cid).await;

    assert_eq!(
        second,
        TestEviction {
            removed: false,
            remaining: 1
        }
    );
    assert_eq!(tap.drain(&tv).await, Vec::<String>::new());
}

/// For a non-member the eviction helper changes nothing: in particular it does
/// not delete an info hash that exists without a membership.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn evict_connection_for_test_non_member_leaves_info_hash() {
    let tv = TestValkey::setup().await;
    let (sid, cid) = ("sess-evict-non-member", "conn-1");
    let _: i64 = tv
        .client()
        .pool
        .hset(
            tv.key_schema().presence_user(sid, cid),
            ("user_name", USER_NAME),
        )
        .await
        .expect("HSET info must succeed");

    let outcome = evict(&tv, sid, cid).await;

    assert_eq!(
        outcome,
        TestEviction {
            removed: false,
            remaining: 0
        }
    );
    assert_eq!(
        info(&tv, sid, cid).await,
        HashMap::from([("user_name".to_owned(), USER_NAME.to_owned())]),
        "the info hash of a non-member must be left alone"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Sweep batch (SR-1.4, SR-1.6, SR-3.6)
// ═════════════════════════════════════════════════════════════════════════════

/// The `left` payloads a sweep passes for `cids`: one per id, in order.
fn left_payloads(cids: &[String]) -> Vec<String> {
    cids.iter().map(|cid| left_payload(cid)).collect()
}

async fn sweep(tv: &TestValkey, sid: &str, cids: &[String]) -> SweepBatch {
    sweep_member_batch(
        tv.client(),
        origin(tv),
        tv.key_schema(),
        sid,
        cids,
        &left_payloads(cids),
    )
    .await
    .expect("sweep_member_batch must succeed")
}

/// Expire `cid`'s lease now, as its TTL running out would: the connection
/// stays a member with its info hash but has no lease, which is what a sweep
/// calls stale.
async fn expire_lease(tv: &TestValkey, sid: &str, cid: &str) {
    let deleted: i64 = tv
        .client()
        .pool
        .del(tv.key_schema().presence_lease(sid, cid))
        .await
        .expect("DEL lease must succeed");
    assert_eq!(deleted, 1, "precondition: {cid} had a lease to expire");
}

/// Register `cid` afresh, then expire its lease. Its `lease_expires_at_ms` is
/// still one lease TTL ahead, so a sweep reports its delay as zero.
async fn register_stale(tv: &TestValkey, sid: &str, cid: &str) {
    assert_eq!(
        register(tv, sid, cid).await,
        RegisterOutcome::Registered { added: true },
        "seeding must register {cid} afresh"
    );
    expire_lease(tv, sid, cid).await;
}

/// The removal delay of a member whose lease was gone before its recorded
/// expiry: clamped to zero.
const ZERO_DELAY: Option<Duration> = Some(Duration::ZERO);

/// The chunk [`seed_mixed_chunk`] builds and what a sweep of it must do.
struct MixedChunk {
    /// The chunk: the stale, live and non-member ids below, interleaved.
    cids: Vec<String>,
    /// The members with no lease, in chunk order.
    stale: [&'static str; 2],
    /// The members with a lease.
    live: [&'static str; 2],
    /// Not a member; has an info hash and no lease.
    non_member: &'static str,
}

impl MixedChunk {
    /// What a first sweep of the chunk returns.
    fn first_sweep(&self) -> SweepBatch {
        SweepBatch {
            removed: self
                .stale
                .iter()
                .map(|cid| ((*cid).to_owned(), ZERO_DELAY))
                .collect(),
            remaining: self.live.len() as u64,
        }
    }

    fn lefts(&self) -> Vec<String> {
        self.stale.iter().map(|cid| left_payload(cid)).collect()
    }
}

/// Seed one session with two stale members, two live ones, and a non-member
/// that has an info hash but no lease (so only the membership check can keep
/// a sweep off it). Returns the chunk with the stale ids out of registration
/// order, so a sweep that reported in any order but the chunk's cannot pass.
async fn seed_mixed_chunk(tv: &TestValkey, sid: &str) -> MixedChunk {
    let chunk = MixedChunk {
        cids: owned(&[
            "conn-live-1",
            "conn-stale-b",
            "conn-not-member",
            "conn-stale-a",
            "conn-live-2",
        ]),
        stale: ["conn-stale-b", "conn-stale-a"],
        live: ["conn-live-1", "conn-live-2"],
        non_member: "conn-not-member",
    };
    register_stale(tv, sid, "conn-stale-a").await;
    register_stale(tv, sid, "conn-stale-b").await;
    for cid in chunk.live {
        register(tv, sid, cid).await;
    }
    let _: i64 = tv
        .client()
        .pool
        .hset(
            tv.key_schema().presence_user(sid, chunk.non_member),
            ("user_name", USER_NAME),
        )
        .await
        .expect("HSET the non-member's info hash must succeed");
    chunk
}

/// One chunk holding stale members, live members and a non-member: only the
/// stale members are removed (member and info hash), each is reported in chunk
/// order and announced once, and the live members and the non-member are left
/// as they were. A stale member's lease is not recreated.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_member_batch_mixed_chunk_removes_and_announces_only_stale_members() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let sid = "sess-sweep-mixed";
    let mut tap = EventTap::open(&tv, sid).await;
    let chunk = seed_mixed_chunk(&tv, sid).await;
    tap.drain(&tv).await;
    let live_infos_before = [
        info(&tv, sid, chunk.live[0]).await,
        info(&tv, sid, chunk.live[1]).await,
    ];

    let batch = sweep(&tv, sid, &chunk.cids).await;

    assert_eq!(
        batch,
        chunk.first_sweep(),
        "exactly the stale members are removed, in chunk order, each with its delay \
         (zero: the lease went before its recorded expiry); the live ones remain"
    );
    assert_eq!(
        tap.drain(&tv).await,
        chunk.lefts(),
        "each removed member is announced exactly once, in chunk order, and no one else is"
    );
    for cid in chunk.stale {
        assert!(!is_member(&tv, sid, cid).await, "{cid} must be removed");
        assert!(
            !exists(&tv, &ks.presence_user(sid, cid)).await,
            "{cid}'s info hash must be deleted"
        );
        assert!(
            !exists(&tv, &ks.presence_lease(sid, cid)).await,
            "the sweep checks leases and never writes one; {cid} must have none"
        );
    }
    for (cid, info_before) in chunk.live.iter().zip(live_infos_before) {
        assert!(
            is_member(&tv, sid, cid).await,
            "{cid} is live and must stay"
        );
        assert_eq!(
            info(&tv, sid, cid).await,
            info_before,
            "{cid}'s info hash must be untouched"
        );
        assert!(
            exists(&tv, &ks.presence_lease(sid, cid)).await,
            "{cid}'s lease must be untouched"
        );
    }
    assert!(!is_member(&tv, sid, chunk.non_member).await);
    assert_eq!(
        info(&tv, sid, chunk.non_member).await,
        HashMap::from([("user_name".to_owned(), USER_NAME.to_owned())]),
        "a non-member's info hash must be left alone"
    );
}

/// With `lease_expires_at_ms` planted far in the past, the reported delay is
/// the store's clock at the call minus the planted stamp: at least the planted
/// age read just before the call, at most that read just after.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_member_batch_reports_delay_since_planted_lease_expiry() {
    let tv = TestValkey::setup().await;
    let (sid, cid) = ("sess-sweep-delay", "conn-1");
    register_stale(&tv, sid, cid).await;
    plant_stale_expires_at(&tv, sid, cid).await;
    let planted: u64 = STALE_EXPIRES_AT
        .parse()
        .expect("the planted stamp is an integer");

    let before = valkey_now_ms(&tv).await;
    let batch = sweep(&tv, sid, &owned(&[cid])).await;
    let after = valkey_now_ms(&tv).await;

    let [(removed, delay)] = batch.removed.as_slice() else {
        panic!("exactly {cid} must be removed; got {batch:?}");
    };
    assert_eq!(removed, cid);
    let delay_ms = u64::try_from(
        delay
            .expect("a numeric lease_expires_at_ms must yield a delay")
            .as_millis(),
    )
    .expect("the delay fits u64 milliseconds");
    assert!(
        (before - planted..=after - planted).contains(&delay_ms),
        "the delay must be the store's TIME minus the planted {planted} ms: expected within \
         [{}, {}], got {delay_ms}",
        before - planted,
        after - planted
    );
    assert_eq!(batch.remaining, 0);
}

/// Without a `lease_expires_at_ms` field the member is still removed and
/// announced, with no delay.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_member_batch_without_expiry_field_reports_no_delay() {
    let tv = TestValkey::setup().await;
    let (sid, cid) = ("sess-sweep-no-field", "conn-1");
    let mut tap = EventTap::open(&tv, sid).await;
    register_stale(&tv, sid, cid).await;
    hdel_field(&tv, sid, cid, "lease_expires_at_ms").await;
    tap.drain(&tv).await;

    let batch = sweep(&tv, sid, &owned(&[cid])).await;

    assert_eq!(
        batch,
        SweepBatch {
            removed: vec![(cid.to_owned(), None)],
            remaining: 0
        }
    );
    assert_eq!(tap.drain(&tv).await, vec![left_payload(cid)]);
}

/// A corrupted `lease_expires_at_ms` never yields a made-up delay: a negative,
/// `-inf`, `nan` or non-numeric value reads as absent (`None`), and `inf` or a
/// far-future stamp clamps to zero. Each member is still removed and announced.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_member_batch_unusable_expiry_reports_no_delay_and_future_expiry_reports_zero() {
    let tv = TestValkey::setup().await;
    let sid = "sess-sweep-delay-guard";
    let planted: [(&str, &str, Option<Duration>); 6] = [
        ("conn-negative", "-5", None),
        ("conn-negative-inf", "-inf", None),
        ("conn-nan", "nan", None),
        ("conn-not-numeric", "soon", None),
        ("conn-inf", "inf", ZERO_DELAY),
        ("conn-far-future", "99999999999999", ZERO_DELAY),
    ];
    let mut tap = EventTap::open(&tv, sid).await;
    for (cid, raw, _) in planted {
        register_stale(&tv, sid, cid).await;
        plant_expires_at(&tv, sid, cid, raw).await;
    }
    tap.drain(&tv).await;
    let cids: Vec<String> = planted
        .iter()
        .map(|(cid, _, _)| (*cid).to_owned())
        .collect();

    let batch = sweep(&tv, sid, &cids).await;

    assert_eq!(
        batch,
        SweepBatch {
            removed: planted
                .iter()
                .map(|(cid, _, delay)| ((*cid).to_owned(), *delay))
                .collect(),
            remaining: 0
        },
        "planted lease_expires_at_ms values {:?}",
        planted.map(|(cid, raw, _)| (cid, raw))
    );
    assert_eq!(tap.drain(&tv).await, left_payloads(&cids));
}

/// Re-running a chunk removes and announces nothing, and still reports the
/// members that remain.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_member_batch_repeated_chunk_removes_and_announces_nothing() {
    let tv = TestValkey::setup().await;
    let sid = "sess-sweep-repeat";
    let mut tap = EventTap::open(&tv, sid).await;
    let chunk = seed_mixed_chunk(&tv, sid).await;
    assert_eq!(sweep(&tv, sid, &chunk.cids).await, chunk.first_sweep());
    tap.drain(&tv).await;

    let again = sweep(&tv, sid, &chunk.cids).await;

    assert_eq!(
        again,
        SweepBatch {
            removed: vec![],
            remaining: chunk.live.len() as u64
        },
        "a repeat removes nothing and reports the live members left"
    );
    assert_eq!(
        tap.drain(&tv).await,
        Vec::<String>::new(),
        "the first sweep announced every removal; a repeat must announce none"
    );
}

/// A chunk that names a stale member twice (an `SSCAN` page may repeat an
/// entry) removes, reports and announces it once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_member_batch_duplicate_cid_in_chunk_is_removed_and_announced_once() {
    let tv = TestValkey::setup().await;
    let (sid, cid) = ("sess-sweep-duplicate", "conn-1");
    let mut tap = EventTap::open(&tv, sid).await;
    register_stale(&tv, sid, cid).await;
    tap.drain(&tv).await;

    let batch = sweep(&tv, sid, &owned(&[cid, cid])).await;

    assert_eq!(
        batch,
        SweepBatch {
            removed: vec![(cid.to_owned(), ZERO_DELAY)],
            remaining: 0
        }
    );
    assert_eq!(tap.drain(&tv).await, vec![left_payload(cid)]);
}

/// A member whose lease is renewed after the caller read it as stale and
/// before the script runs is kept: the script re-checks the lease itself. The
/// renewal here is the keeper's own refresh, which recreates a missing lease.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_member_batch_lease_renewed_before_the_call_keeps_the_member() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, cid) = ("sess-sweep-race", "conn-1");
    let mut tap = EventTap::open(&tv, sid).await;
    register_stale(&tv, sid, cid).await;
    tap.drain(&tv).await;
    // The caller's read: a member with no lease.
    assert!(is_member(&tv, sid, cid).await && !exists(&tv, &ks.presence_lease(sid, cid)).await);
    assert_eq!(refresh(&tv, sid, cid).await, RefreshOutcome::Renewed);
    let info_after_refresh = info(&tv, sid, cid).await;

    let batch = sweep(&tv, sid, &owned(&[cid])).await;

    assert_eq!(
        batch,
        SweepBatch {
            removed: vec![],
            remaining: 1
        },
        "a member whose lease is back must not be removed"
    );
    assert_eq!(tap.drain(&tv).await, Vec::<String>::new());
    assert!(is_member(&tv, sid, cid).await);
    assert_eq!(
        info(&tv, sid, cid).await,
        info_after_refresh,
        "the kept member's info hash must be untouched"
    );
}

/// A chunk of exactly [`SWEEP_MEMBER_BATCH`] stale members is accepted whole:
/// every member is removed and announced, in chunk order.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_member_batch_full_chunk_removes_and_announces_every_stale_member() {
    let tv = TestValkey::setup().await;
    let sid = "sess-sweep-full";
    let mut tap = EventTap::open(&tv, sid).await;
    let registered: Vec<String> = (0..SWEEP_MEMBER_BATCH)
        .map(|i| format!("conn-{i:03}"))
        .collect();
    for cid in &registered {
        register_stale(&tv, sid, cid).await;
    }
    tap.drain(&tv).await;
    let cids: Vec<String> = registered.into_iter().rev().collect();

    let batch = sweep(&tv, sid, &cids).await;

    assert_eq!(
        batch,
        SweepBatch {
            removed: cids.iter().map(|cid| (cid.clone(), ZERO_DELAY)).collect(),
            remaining: 0
        },
        "a full chunk of {SWEEP_MEMBER_BATCH} stale members is removed whole, in chunk order"
    );
    assert_eq!(tap.drain(&tv).await, left_payloads(&cids));
}

/// Call [`sweep_member_batch`] with every data-pool connection stalled and
/// assert it was refused with `InvalidArgument` without sending a command.
async fn assert_sweep_refused_without_sending(
    cids: &[String],
    left_payloads: &[String],
    what: &str,
) {
    let (tv, metrics) = metered_fixture().await;
    let call = sweep_member_batch(
        tv.client(),
        origin(&tv),
        tv.key_schema(),
        "sess-sweep-refused",
        cids,
        left_payloads,
    );

    let result = run_sending_nothing(&tv, &metrics, what, call).await;

    let err = result.expect_err("the call must be refused");
    assert_eq!(
        *err.kind(),
        fred::error::ErrorKind::InvalidArgument,
        "{what} is refused by the wrapper's argument check; got {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_member_batch_empty_chunk_is_rejected_without_sending() {
    assert_sweep_refused_without_sending(&[], &[], "an empty sweep_member_batch").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_member_batch_over_one_batch_is_rejected_without_sending() {
    let cids: Vec<String> = (0..=SWEEP_MEMBER_BATCH)
        .map(|i| format!("conn-{i:03}"))
        .collect();
    assert_sweep_refused_without_sending(
        &cids,
        &left_payloads(&cids),
        "a sweep_member_batch of SWEEP_MEMBER_BATCH + 1 ids",
    )
    .await;
}

/// One payload short: zipping would silently drop the last id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_member_batch_fewer_payloads_than_cids_is_rejected_without_sending() {
    let cids = owned(&["conn-1", "conn-2"]);
    let payloads = left_payloads(&cids[..1]);
    assert_sweep_refused_without_sending(
        &cids,
        &payloads,
        "a sweep_member_batch with fewer left payloads than ids",
    )
    .await;
}

/// One payload over: zipping would silently ignore it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_member_batch_more_payloads_than_cids_is_rejected_without_sending() {
    let cids = owned(&["conn-1", "conn-2"]);
    let payloads = left_payloads(&owned(&["conn-1", "conn-2", "conn-3"]));
    assert_sweep_refused_without_sending(
        &cids,
        &payloads,
        "a sweep_member_batch with more left payloads than ids",
    )
    .await;
}

/// The sweep neither reads nor writes the session index or the teardown
/// marker, and touches nothing of another session, even one with the same
/// connection ids, also stale. Removing the session's last member leaves it
/// indexed (the caller runs the index removal), and a marker on a swept member
/// keeps its TTL (it is not re-armed) and no marker is armed for the other.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_member_batch_leaves_index_markers_and_other_sessions_alone() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (sid, other) = ("sess-sweep-scope", "sess-sweep-scope-other");
    let (marked, unmarked) = ("conn-marked", "conn-unmarked");
    let mut other_tap = EventTap::open(&tv, other).await;
    for session in [sid, other] {
        register_stale(&tv, session, marked).await;
        register_stale(&tv, session, unmarked).await;
        index_add(&tv, session).await;
    }
    plant_gone(&tv, sid, marked).await;
    shorten_ttl(&tv, &ks.presence_gone(sid, marked)).await;
    other_tap.drain(&tv).await;
    let other_infos_before = [
        info(&tv, other, marked).await,
        info(&tv, other, unmarked).await,
    ];

    let batch = sweep(&tv, sid, &owned(&[marked, unmarked])).await;

    assert_eq!(
        batch,
        SweepBatch {
            removed: vec![
                (marked.to_owned(), ZERO_DELAY),
                (unmarked.to_owned(), ZERO_DELAY)
            ],
            remaining: 0
        }
    );
    assert!(
        index_contains(&tv, sid).await,
        "the sweep must not unindex the session, even at zero members"
    );
    assert!(index_contains(&tv, other).await);
    let marker_ttl = ttl(&tv, &ks.presence_gone(sid, marked)).await;
    assert!(
        (1..=SHORTENED_TTL_SECS).contains(&marker_ttl),
        "the existing marker must be neither deleted nor re-armed (TTL still at most \
         {SHORTENED_TTL_SECS}s); got {marker_ttl}"
    );
    assert!(
        !exists(&tv, &ks.presence_gone(sid, unmarked)).await,
        "the sweep must not arm a teardown marker"
    );
    for (cid, info_before) in [marked, unmarked].into_iter().zip(other_infos_before) {
        assert!(
            is_member(&tv, other, cid).await,
            "another session's {cid} must stay a member"
        );
        assert_eq!(
            info(&tv, other, cid).await,
            info_before,
            "another session's {cid} info hash must be untouched"
        );
        assert!(!exists(&tv, &ks.presence_gone(other, cid)).await);
    }
    assert_eq!(
        other_tap.drain(&tv).await,
        Vec::<String>::new(),
        "nothing may be announced on another session's channel"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Paged member-info read (SR-1.5, SR-1.6)
// ═════════════════════════════════════════════════════════════════════════════

/// A connection's decoded member info as [`read_member_infos`] returns it.
type MemberInfo = Option<(String, String)>;

/// The user name and application [`register_member`] registers `cid` with:
/// distinct per connection, so every entry of a read names the connection it
/// came from.
fn member_identity(cid: &str) -> (String, String) {
    (format!("user-of-{cid}"), format!("app-of-{cid}"))
}

/// The entry a well-formed read returns for a connection [`register_member`]
/// registered.
fn decoded(cid: &str) -> MemberInfo {
    Some(member_identity(cid))
}

/// The register deadline a real caller computes: a fresh [`store_time_ms`] for
/// the session plus one lease TTL. Read once per seeding; it stays in the
/// future for the whole of it.
async fn production_deadline(tv: &TestValkey, sid: &str) -> u64 {
    store_now_ms(tv, sid).await + LEASE_TTL_SECS * 1000
}

/// Register `cid` through the real [`register_connection`] with the identity
/// [`member_identity`] gives it.
async fn register_member(tv: &TestValkey, sid: &str, cid: &str, not_after_ms: u64) {
    let (user_name, user_application) = member_identity(cid);
    let payload = PresenceEvent::PeerJoined(ConnectedUser {
        connection_id: cid.to_owned(),
        user_name: user_name.clone(),
        user_application: user_application.clone(),
    })
    .encode();
    let outcome = register_connection(
        tv.client(),
        origin(tv),
        tv.key_schema(),
        sid,
        cid,
        &user_name,
        &user_application,
        LEASE_TTL_SECS,
        not_after_ms,
        &payload,
    )
    .await
    .expect("register_connection must succeed");
    assert_eq!(
        outcome,
        RegisterOutcome::Registered { added: true },
        "seeding must register {cid} afresh"
    );
}

async fn read_infos(
    tv: &TestValkey,
    sid: &str,
    cids: &[String],
) -> Result<Vec<MemberInfo>, fred::error::Error> {
    read_member_infos(tv.client(), origin(tv), tv.key_schema(), sid, cids).await
}

fn owned(cids: &[&str]) -> Vec<String> {
    cids.iter().map(|cid| (*cid).to_owned()).collect()
}

/// Register [`REPLAY_PAGE`] members and return their ids in the reverse of
/// registration order, so a read that answered in registration (or sorted)
/// order rather than input order cannot pass.
async fn seed_full_page(tv: &TestValkey, sid: &str) -> Vec<String> {
    let deadline = production_deadline(tv, sid).await;
    let cids: Vec<String> = (0..REPLAY_PAGE).map(|i| format!("conn-{i:03}")).collect();
    for cid in &cids {
        register_member(tv, sid, cid, deadline).await;
    }
    cids.into_iter().rev().collect()
}

fn assert_full_page_decoded(cids: &[String], infos: Vec<MemberInfo>) {
    assert_eq!(cids.len(), REPLAY_PAGE, "precondition: a full page of ids");
    let expected: Vec<MemberInfo> = cids.iter().map(|cid| decoded(cid)).collect();
    assert_eq!(
        infos, expected,
        "a full page reads one entry per id, in input order, each with its own connection's \
         user name and application"
    );
}

async fn hdel_field(tv: &TestValkey, sid: &str, cid: &str, field: &str) {
    let removed: i64 = tv
        .client()
        .pool
        .hdel(tv.key_schema().presence_user(sid, cid), field)
        .await
        .expect("HDEL must succeed");
    assert_eq!(removed, 1, "precondition: {cid}'s info hash had {field}");
}

/// `tot-net-in` of a stalled connection; it must still be open.
async fn stalled_net_in(stall: &ConnStall) -> u64 {
    stall
        .net_in()
        .await
        .expect("the stalled connection must still be open")
}

/// Run `call` with every data-pool connection stalled and assert it sent no
/// command: no stalled connection's `tot-net-in` moved, which counts bytes the
/// server has read even while they queue behind the stall, and no command
/// timeout was counted. A command the call did send would also have stopped it
/// returning until its 5 s command timeout, as a `Timeout` error. Returns the
/// call's own result for the caller to check.
async fn run_sending_nothing<T>(
    tv: &TestValkey,
    metrics: &Metrics,
    what: &str,
    call: impl std::future::Future<Output = Result<T, fred::error::Error>>,
) -> Result<T, fred::error::Error> {
    let stalls = tv.stall_data_pool().await;
    let mut net_in_before = Vec::new();
    for stall in &stalls {
        net_in_before.push(stalled_net_in(stall).await);
    }

    let outcome = tokio::time::timeout(SCENARIO_BUDGET, call).await;

    let mut net_in_after = Vec::new();
    for stall in &stalls {
        net_in_after.push(stalled_net_in(stall).await);
    }
    release_all(stalls).await;
    let result = outcome
        .unwrap_or_else(|_| panic!("{what} did not settle within the budget; it sent a command"));
    assert_eq!(
        net_in_after, net_in_before,
        "{what} must send no command: every data-pool connection's tot-net-in must be unchanged"
    );
    assert_eq!(
        command_timeout_row(metrics),
        command_timeouts_only_on("data", 0),
        "{what} must reach no command timeout"
    );
    result
}

/// The bytes one `HGETALL <key>` occupies on the wire: a two-element RESP array
/// of bulk strings, `*2\r\n$7\r\nHGETALL\r\n$<len>\r\n<key>\r\n`.
fn hgetall_wire_len(key: &str) -> u64 {
    let key_bulk = format!("${}\r\n{key}\r\n", key.len());
    ("*2\r\n".len() + "$7\r\nHGETALL\r\n".len() + key_bulk.len()) as u64
}

/// The page goes out as one pipeline: with every data-pool connection stalled,
/// all of its `HGETALL`s reach the server while the first is still unanswered.
/// A read that awaited each id's reply before sending the next would send one
/// command, wait out its timeout behind the stall, and return before the rest
/// were sent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_member_infos_sends_whole_page_as_one_pipeline() {
    let tv = TestValkey::setup().await;
    let sid = "sess-read-one-pipeline";
    let cids = owned(&["conn-1", "conn-2", "conn-3", "conn-4", "conn-5"]);
    let page_wire_len: u64 = cids
        .iter()
        .map(|cid| hgetall_wire_len(&tv.key_schema().presence_user(sid, cid)))
        .sum();
    let stalls = tv.stall_data_pool().await;
    let mut baseline = 0;
    for stall in &stalls {
        baseline += stalled_net_in(stall).await;
    }

    let call = read_infos(&tv, sid, &cids);
    tokio::pin!(call);
    let received_all = async {
        let deadline = tokio::time::Instant::now() + PIPELINE_BUDGET;
        loop {
            let mut net_in = 0;
            for stall in &stalls {
                net_in += stalled_net_in(stall).await;
            }
            let received = net_in - baseline;
            if received >= page_wire_len || tokio::time::Instant::now() >= deadline {
                return received;
            }
            tokio::time::sleep(NET_IN_POLL).await;
        }
    };
    let watched = tokio::select! {
        received = received_all => Ok(received),
        returned = &mut call => Err(returned),
    };
    release_all(stalls).await;

    let received = watched.unwrap_or_else(|returned| {
        panic!(
            "read_member_infos returned {returned:?} with every connection stalled, before the \
             server received the whole page; it did not send the page as one pipeline"
        )
    });
    assert!(
        received >= page_wire_len,
        "the server must receive all {} HGETALLs ({page_wire_len} bytes) while the first is still \
         unanswered; it received {received} bytes within {PIPELINE_BUDGET:?}",
        cids.len()
    );
    let infos = tokio::time::timeout(PIPELINE_BUDGET, call)
        .await
        .expect("the released read did not settle within the budget")
        .expect("the released read must succeed");
    assert_eq!(
        infos,
        vec![None; cids.len()],
        "no id was registered, so every entry is None"
    );
}

/// A full page of [`REPLAY_PAGE`] members registered through the real
/// register script reads back whole, in input order.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_member_infos_full_page_returns_every_entry_in_input_order() {
    let tv = TestValkey::setup().await;
    let sid = "sess-read-full-page";
    let cids = seed_full_page(&tv, sid).await;

    let infos = read_infos(&tv, sid, &cids)
        .await
        .expect("read_member_infos must succeed");

    assert_full_page_decoded(&cids, infos);
}

/// Each malformed hash is `None` at its own position and the rest of the page
/// still decodes: an empty hash (no info key at all), a hash with `user_name`
/// deleted, and a hash whose `user_application` is not UTF-8 (its `user_name`
/// is intact, so only the value check can make it `None`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_member_infos_mixed_page_returns_none_in_place_for_each_malformed_hash() {
    let tv = TestValkey::setup().await;
    let sid = "sess-read-mixed";
    let good = ["conn-good-0", "conn-good-1", "conn-good-2", "conn-good-3"];
    let (empty, no_name, not_utf8) = ("conn-empty", "conn-no-name", "conn-not-utf8");
    let deadline = production_deadline(&tv, sid).await;
    for cid in good.iter().chain(&[no_name, not_utf8]) {
        register_member(&tv, sid, cid, deadline).await;
    }
    // `empty` is never registered: HGETALL reads its absent key as an empty hash.
    hdel_field(&tv, sid, no_name, "user_name").await;
    let _: i64 = tv
        .client()
        .pool
        .hset(
            tv.key_schema().presence_user(sid, not_utf8),
            (
                "user_application",
                Value::Bytes(bytes::Bytes::from_static(&[0xff, 0xfe])),
            ),
        )
        .await
        .expect("HSET of a non-UTF-8 user_application must succeed");
    let cids = owned(&[good[0], empty, good[1], no_name, good[2], not_utf8, good[3]]);

    let infos = read_infos(&tv, sid, &cids)
        .await
        .expect("a malformed hash must not fail the page");

    assert_eq!(
        infos,
        vec![
            decoded(good[0]),
            None,
            decoded(good[1]),
            None,
            decoded(good[2]),
            None,
            decoded(good[3]),
        ],
        "the empty, nameless and non-UTF-8 hashes are None in place; the rest decode"
    );
}

/// A hash without `user_application` is not malformed: it decodes with an
/// empty application, as the replay it replaces did.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_member_infos_hash_without_user_application_decodes_with_empty_application() {
    let tv = TestValkey::setup().await;
    let (sid, cid) = ("sess-read-no-app", "conn-no-app");
    register_member(&tv, sid, cid, production_deadline(&tv, sid).await).await;
    hdel_field(&tv, sid, cid, "user_application").await;

    let infos = read_infos(&tv, sid, &owned(&[cid]))
        .await
        .expect("read_member_infos must succeed");

    assert_eq!(
        infos,
        vec![Some((member_identity(cid).0, String::new()))],
        "a missing user_application reads as an empty string, not None"
    );
}

/// A one-id page returns exactly one decoded entry. fred wraps a pipeline's
/// replies in an array even for one command; were the lone `HGETALL` reply
/// taken as the reply list, its fields would be miscounted as entries.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_member_infos_single_cid_returns_one_decoded_entry() {
    let tv = TestValkey::setup().await;
    let (sid, cid) = ("sess-read-single", "conn-1");
    register_member(&tv, sid, cid, production_deadline(&tv, sid).await).await;

    let infos = read_infos(&tv, sid, &owned(&[cid]))
        .await
        .expect("read_member_infos must succeed");

    assert_eq!(infos, vec![decoded(cid)]);
}

/// An empty slice returns an empty `Vec` without a round trip.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_member_infos_empty_slice_returns_empty_and_sends_nothing() {
    let (tv, metrics) = metered_fixture().await;
    let call = read_infos(&tv, "sess-read-empty", &[]);

    let infos = run_sending_nothing(&tv, &metrics, "an empty read_member_infos", call)
        .await
        .expect("an empty read must succeed");

    assert_eq!(infos, Vec::<MemberInfo>::new());
}

/// A slice one longer than [`REPLAY_PAGE`] is a caller bug: it is refused with
/// `InvalidArgument` before any command is sent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_member_infos_over_one_page_is_rejected_without_sending() {
    let (tv, metrics) = metered_fixture().await;
    let cids: Vec<String> = (0..=REPLAY_PAGE).map(|i| format!("conn-{i:03}")).collect();
    let call = read_infos(&tv, "sess-read-oversize", &cids);

    let result = run_sending_nothing(&tv, &metrics, "an over-long read_member_infos", call).await;

    let err = result.expect_err("a slice longer than REPLAY_PAGE must be refused");
    assert_eq!(
        *err.kind(),
        fred::error::ErrorKind::InvalidArgument,
        "the refusal is the length check's; got {err:?}"
    );
}

/// An info key that is not a hash breaks the key schema, so its `WRONGTYPE`
/// reply fails the whole page rather than reading as `None`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_member_infos_info_key_of_wrong_type_fails_the_whole_page() {
    let tv = TestValkey::setup().await;
    let (sid, good, wrong) = ("sess-read-wrongtype", "conn-good", "conn-wrongtype");
    register_member(&tv, sid, good, production_deadline(&tv, sid).await).await;
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().presence_user(sid, wrong),
            "not-a-hash",
            None,
            None,
            false,
        )
        .await
        .expect("SET of a string at the info key must succeed");

    let result = read_infos(&tv, sid, &owned(&[good, wrong])).await;

    let err = result.expect_err("a WRONGTYPE reply must fail the page");
    assert!(
        err.to_string().contains("WRONGTYPE"),
        "the page fails with the store's WRONGTYPE reply; got {err:?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Timeout counting (SR-1.7)
// ═════════════════════════════════════════════════════════════════════════════

/// A fixture with a metrics registry installed on the client, so every
/// `FredErrOrigin::client` count lands on the returned `Metrics`.
async fn metered_fixture() -> (TestValkey, Arc<Metrics>) {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;
    (tv, metrics)
}

/// Run `call` against a stalled data pool and assert it fails with fred's
/// `Timeout`, counted exactly once, on the data pool.
async fn assert_one_data_pool_timeout<T: std::fmt::Debug>(
    tv: &TestValkey,
    metrics: &Metrics,
    what: &str,
    call: impl std::future::Future<Output = Result<T, fred::error::Error>>,
) {
    let stalls = tv.stall_data_pool().await;
    let outcome = tokio::time::timeout(SCENARIO_BUDGET, call).await;
    release_all(stalls).await;

    let err = outcome
        .unwrap_or_else(|_| panic!("the stalled {what} did not settle within the budget"))
        .expect_err("a call against a stalled data pool must fail");
    assert_eq!(
        *err.kind(),
        fred::error::ErrorKind::Timeout,
        "the stalled {what} must fail with fred's command timeout; got {err:?}"
    );
    assert_eq!(
        command_timeout_row(metrics),
        command_timeouts_only_on("data", 1),
        "the timed-out {what} counts exactly once, on the data pool"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_connection_timeout_counts_once_on_data_pool() {
    let (tv, metrics) = metered_fixture().await;
    let payload = joined_payload("conn-1");
    let call = register_connection(
        tv.client(),
        origin(&tv),
        tv.key_schema(),
        "sess-timeout",
        "conn-1",
        USER_NAME,
        USER_APPLICATION,
        LEASE_TTL_SECS,
        u64::MAX,
        &payload,
    );
    assert_one_data_pool_timeout(&tv, &metrics, "register_connection", call).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn refresh_connection_timeout_counts_once_on_data_pool() {
    let (tv, metrics) = metered_fixture().await;
    let call = refresh_connection(
        tv.client(),
        origin(&tv),
        tv.key_schema(),
        "sess-timeout",
        "conn-1",
        LEASE_TTL_SECS,
    );
    assert_one_data_pool_timeout(&tv, &metrics, "refresh_connection", call).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unregister_connection_timeout_counts_once_on_data_pool() {
    let (tv, metrics) = metered_fixture().await;
    let payload = left_payload("conn-1");
    let call = unregister_connection(
        tv.client(),
        origin(&tv),
        tv.key_schema(),
        "sess-timeout",
        "conn-1",
        LEASE_TTL_SECS,
        &payload,
    );
    assert_one_data_pool_timeout(&tv, &metrics, "unregister_connection", call).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remove_session_from_index_timeout_counts_once_on_data_pool() {
    let (tv, metrics) = metered_fixture().await;
    let call = remove_session_from_index(tv.client(), origin(&tv), tv.key_schema(), "sess-timeout");
    assert_one_data_pool_timeout(&tv, &metrics, "remove_session_from_index", call).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn evict_connection_for_test_timeout_counts_once_on_data_pool() {
    let (tv, metrics) = metered_fixture().await;
    let payload = left_payload("conn-1");
    let call = evict_connection_for_test(
        tv.client(),
        origin(&tv),
        tv.key_schema(),
        "sess-timeout",
        "conn-1",
        &payload,
    );
    assert_one_data_pool_timeout(&tv, &metrics, "evict_connection_for_test", call).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn store_time_ms_timeout_counts_once_on_data_pool() {
    let (tv, metrics) = metered_fixture().await;
    let call = store_time_ms(tv.client(), origin(&tv), tv.key_schema(), "sess-timeout");
    assert_one_data_pool_timeout(&tv, &metrics, "store_time_ms", call).await;
}

/// Several ids, so a count per timed-out `HGETALL` rather than one per call
/// would read above one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_member_infos_timeout_counts_once_on_data_pool() {
    let (tv, metrics) = metered_fixture().await;
    let cids = owned(&["conn-1", "conn-2", "conn-3"]);
    let call = read_infos(&tv, "sess-timeout", &cids);
    assert_one_data_pool_timeout(&tv, &metrics, "read_member_infos", call).await;
}

/// Several ids, so a count per id rather than one per call would read above
/// one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_member_batch_timeout_counts_once_on_data_pool() {
    let (tv, metrics) = metered_fixture().await;
    let cids = owned(&["conn-1", "conn-2", "conn-3"]);
    let payloads = left_payloads(&cids);
    let call = sweep_member_batch(
        tv.client(),
        origin(&tv),
        tv.key_schema(),
        "sess-timeout",
        &cids,
        &payloads,
    );
    assert_one_data_pool_timeout(&tv, &metrics, "sweep_member_batch", call).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Cluster variant
// ═════════════════════════════════════════════════════════════════════════════

/// Against a multi-shard cluster, register and unregister publish from inside
/// their scripts on the node owning `{sid}`, and a subscriber reached through
/// the cluster receives both announcements. The register's deadline comes from
/// `store_time_ms` read on the `{sid}` node, as a real caller's does.
/// Self-gates on `LIVE_EDIT_TEST_CLUSTER_URL` and prints a `SKIP:` line when it
/// is unset.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn presence_scripts_cluster_publish_joined_and_left_to_external_subscriber() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };
    let tv = TestValkey::setup_with_url(cluster_url, 4).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment (redis-cluster://...)"
    );
    let ks = tv.key_schema();
    let (sid, cid) = ("sess-cluster", "conn-1");
    let mut tap = EventTap::open(&tv, sid).await;

    let registered = register(&tv, sid, cid).await;
    assert_eq!(registered, RegisterOutcome::Registered { added: true });
    assert_eq!(tap.next().await, joined_payload(cid));

    let unregistered = unregister(&tv, sid, cid).await;
    assert_eq!(
        unregistered,
        Unregistered {
            removed: true,
            remaining: 0
        }
    );
    assert_eq!(tap.next().await, left_payload(cid));
    assert!(
        exists(&tv, &ks.presence_gone(sid, cid)).await,
        "the teardown marker must be armed on the cluster too"
    );
}

/// Against a multi-shard cluster, a full page of [`REPLAY_PAGE`] members reads
/// back whole and in input order: every key carries the `{sid}` tag, so the
/// one pipeline goes to the node owning that slot. Self-gates on
/// `LIVE_EDIT_TEST_CLUSTER_URL` and prints a `SKIP:` line when it is unset.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_member_infos_cluster_full_page_returns_every_entry_in_input_order() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };
    let tv = TestValkey::setup_with_url(cluster_url, 4).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment (redis-cluster://...)"
    );
    let sid = "sess-cluster-read";
    let cids = seed_full_page(&tv, sid).await;

    let infos = read_infos(&tv, sid, &cids)
        .await
        .expect("read_member_infos must succeed on the cluster");

    assert_full_page_decoded(&cids, infos);
}

/// The primary owning `key`'s slot, from the client's cached `CLUSTER SLOTS`
/// view: the node fred sends a command on `key` to.
fn slot_owner(tv: &TestValkey, key: &str) -> Server {
    let ClusterHash::Custom(slot) = ClusterHash::from(key) else {
        panic!("ClusterHash::from(&str) must yield a slot for {key}");
    };
    tv.client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view")
        .get_server(slot)
        .unwrap_or_else(|| panic!("no primary owns slot {slot} in the cached routing table"))
        .clone()
}

/// `CLUSTER MYID` of the node `client` sends its commands to.
async fn cluster_myid<C: ClusterInterface>(client: &C, what: &str) -> String {
    client
        .cluster_myid()
        .await
        .unwrap_or_else(|e| panic!("CLUSTER MYID on {what} must succeed: {e}"))
}

/// A tap on `sid`'s channel through a subscriber connected directly to a
/// primary that does not own `{sid}`, with its fence published on the owner.
/// `CLUSTER MYID` on both connections proves they reach different nodes.
async fn open_tap_on_another_node(tv: &TestValkey, cluster_url: &str, sid: &str) -> EventTap {
    let owner = slot_owner(tv, &tv.key_schema().presence_members(sid));
    let other = tv
        .client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view")
        .unique_primary_nodes()
        .into_iter()
        .find(|node| *node != owner)
        .unwrap_or_else(|| panic!("a multi-shard cluster must have a primary besides {owner}"));

    // The cluster URL's credentials and TLS settings, aimed at one node.
    let mut config = Config::from_url(cluster_url)
        .expect("LIVE_EDIT_TEST_CLUSTER_URL must parse as a fred config");
    config.server = ServerConfig::Centralized {
        server: other.clone(),
    };
    config.version = RespVersion::RESP2;
    let subscriber = Builder::from_config(config)
        .build_subscriber_client()
        .expect("subscriber client build must succeed");
    subscriber
        .init()
        .await
        .unwrap_or_else(|e| panic!("the subscriber must connect to {other}: {e}"));

    let owner_options = Options {
        cluster_node: Some(owner.clone()),
        ..Default::default()
    };
    let owner_id = cluster_myid(
        &tv.client().pool.next().with_options(&owner_options),
        "the session's slot owner",
    )
    .await;
    let subscriber_id = cluster_myid(&subscriber, "the subscriber's node").await;
    assert_ne!(
        subscriber_id, owner_id,
        "the subscriber must sit on a node other than the {{sid}} owner {owner}"
    );

    let channel = tv.key_schema().presence_events_channel(sid);
    EventTap::open_on(&subscriber.clone(), channel, Some(owner), Some(subscriber)).await
}

/// Against a multi-shard cluster, a mixed chunk's sweep runs on the node
/// owning `{sid}`, and a subscriber connected to a different node receives
/// exactly one `left` per removed member, in chunk order, and nothing else.
/// Self-gates on `LIVE_EDIT_TEST_CLUSTER_URL` and prints a `SKIP:` line when it
/// is unset.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_member_batch_cluster_mixed_chunk_announces_each_removal_once_on_another_node() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };
    let tv = TestValkey::setup_with_url(cluster_url.clone(), 4).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment (redis-cluster://...)"
    );
    let sid = "sess-cluster-sweep";
    let mut tap = open_tap_on_another_node(&tv, &cluster_url, sid).await;
    let chunk = seed_mixed_chunk(&tv, sid).await;
    tap.drain(&tv).await;

    let batch = sweep(&tv, sid, &chunk.cids).await;

    assert_eq!(batch, chunk.first_sweep());
    assert_eq!(
        tap.drain(&tv).await,
        chunk.lefts(),
        "a subscriber on another node receives each removal's left exactly once, in chunk \
         order, and nothing else"
    );
    for cid in chunk.live {
        assert!(
            is_member(&tv, sid, cid).await,
            "{cid} is live and must stay"
        );
    }
}
