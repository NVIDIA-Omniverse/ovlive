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

//! The per-object snapshot fence's storage layer against a live store (b.y3m
//! Epic 3 Task 2; Spec SDD SR-5.1–SR-5.4, SR-5.7, SR-11.4): the six fence and
//! intent scripts through their `storage::snapshot_fence` wrappers, the keys'
//! inventories, and the Lua copies of the cost model.
//!
//! # Time without timing assertions
//!
//! Every limit is long (a lease is about 2.2 s), and no test sleeps through
//! one. A test moves time by **staging** the fence or intent key's fields
//! relative to the store's own clock ([`Fx::now`], a script reading `TIME`
//! exactly as the fence scripts do): an expired lease is a deadline in the
//! past, an epoch outside its join window is an `epoch_start` a minute ago.
//! A value the script derives from its own `now` is checked against the store
//! clock read just before and just after the call ([`bracketed`]), never
//! against a wall-clock duration.
//!
//! "Writes nothing" is checked on the key's whole image ([`KeyImage`]: every
//! field plus the absolute expiry from `PEXPIRETIME`), after first giving the
//! key an expiry no script would set ([`Fx::mark_expiry`]), so a stray
//! `PEXPIRE` shows up as well as a stray `HSET` / `HDEL`.
//!
//! # What lives here
//!
//! | Section | Pins |
//! |---|---|
//! | 1. Acquire | `NOT_FOUND`; `started_epoch` at the header `seq`; `joined_epoch` under the stored size; `RETRY_AFTER` `epoch` / `gap` (the shared instant) / `intent` (only until `gap_until + intent_ttl(L)`); the crashed-holder gap; every wait writes nothing; `PEXPIRE` |
//! | 2. Renew | extends a live lease, raises never lowers; `LOST` after expiry or a wipe, no revival, nothing written |
//! | 3. Release | `VERIFIED` and the gap; `BROKEN` (seq moved, lease expired, field gone); the epoch stays open for a live joiner; the late release; nothing written after cleanup or a restore wipe; the guard's drop, and `ValkeyClient::quit` waiting for that drop's release before it closes the pools — bounded: on a stalled data pool it gives up with a WARN and closes them anyway |
//! | 4. Probe | absent / live / closed / no stored size, read-only; the nine-element reply and its epoch marker (`epoch_start`, `gap_until`, reader count) after a grant, a join, a release under a live joiner, the epoch's last release, a late release (readers alone) and a lapse (nothing moves) |
//! | 5. Intents | refused while fenced; `intent_ttl` of the stored size; the queue floor for a small layer; renew / `LOST`; clear and the guard's drop, and `ValkeyClient::quit` waiting for that drop's `HDEL` before it closes the pools; a crashed writer's intent; `NOT_FOUND` with no header, writing nothing |
//! | 6. Lua and Rust agree | the four Lua `timing` copies are one text, and each limit a script stores or returns equals `FenceCoefficients::timing` at eight sizes and a lowered ceiling |
//! | 7. Malformed input | argument errors raised before any write; a wrong-typed fence key fails acquire; the probe and registration answer `WrongType` (fence first) with the commit paths' exact refusal |
//! | 8. Measurement event | one `snapshot fence lease` DEBUG event per release on its own target: the grant's object, kind (`started_epoch` / `joined_epoch`) and cut, the outcome, `via` `release` / `drop`, ordered brackets that hold the store's own times, no token; a `BROKEN` release logs `broken` and a `NOT_FOUND` release `not_found`; nothing for an acquire that grants nothing; a failed release logs `error` and its guard's re-release a second event with the same acquire bracket |
//!
//! Elsewhere: the writer protocol (commits waiting on a fence, intents, the
//! fence-wait cap, both commit backends' refusal of a wrong-typed fence or
//! intent key) in `snapshot_fence_writer_test.rs`; the formulas against the
//! Task 10 record in
//! `src/storage/snapshot_fence/cost.rs`; the scripts on every shard of a
//! cluster in `snapshot_fence_cluster_test.rs`; cleanup in `b_9fn_tests.rs`
//! and `orphan_sweep_test.rs`; the restore residue wipe in
//! `src/checkpoint/residue.rs`; the snapshot reads leaving both keys alone in
//! `snapshot_read_atomic_test.rs` and `snapshot_read_unfenced_test.rs`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{HashesInterface, KeysInterface};
use fred::types::scripts::Script;
use fred::types::Value;

use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::snapshot_fence::writer::IntentRenewal;
use live_edit::storage::snapshot_fence::{
    self as fence, ms_to_duration, AcquireVerdict, FenceLease, FenceMarker, FenceProbe,
    FenceTiming, GrantKind, IntentRenewVerdict, IntentVerdict, LayerSize, ReleaseVerdict,
    RenewVerdict, RetryReason, WriterIntent,
};
use live_edit::storage::test_support::{connect_test_client, TestValkey};

use crate::storage::snapshot_read_support::client_with_limits;

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

/// The object every single-object test fences.
const OID: u64 = 4_242;

/// The header `seq` [`Fx::new`] seeds: every grant's cut.
const CUT: u64 = 41;

/// 4× the shipped ceiling, the report's checkpoint reference layer. Its
/// `hold`, `reader_wait_max` and `key_linger` differ from [`OTHER_SIZE`]'s, so
/// a size mix-up between the two shows in those. On the wired cost model (Task
/// 10's `F`, the re-fitted `S`) both sit on the queue floor for `lease` and
/// `intent_ttl` (2,162.9 ms), and at the floor-derived value for `epoch`
/// (that floor plus `join_window`), so a test that must tell two sizes' intent windows apart uses
/// [`ABOVE_FLOOR_SIZE`] or [`LONG_INTENT_SIZE`] as the stored size instead.
const SIZE: LayerSize = LayerSize {
    items: 150_216,
    bytes: 1_799_752,
};

/// 2× the shipped ceiling: the estimate a second caller brings, and the size a
/// release measures, wherever a test needs one distinguishable from [`SIZE`].
const OTHER_SIZE: LayerSize = LayerSize {
    items: 75_108,
    bytes: 899_876,
};

/// An 8,000-item layer on Kitchen_set's ray: below `S`'s fitted range
/// (`S_MIN_ITEMS`), so `S` charges it the structural term of an
/// `S_MIN_ITEMS`-item layer. Its `1.5·S` is under the queue floor, so its
/// intent lasts the floor.
const SMALL_SIZE: LayerSize = LayerSize {
    items: 8_000,
    bytes: 95_848,
};

/// The report's 200k-entry layer: its `intent_ttl` is `1.5·S`, above the
/// 2,162.9 ms queue floor, and its `hold`
/// (about 4.0 s) passes the lease floor, so `epoch` follows `hold`. The
/// stored size of a test that needs an intent window distinguishable from a
/// floored estimate's.
const ABOVE_FLOOR_SIZE: LayerSize = LayerSize {
    items: 1_053_196,
    bytes: 12_618_442,
};

/// A 3.5M-item layer at the fitted bytes per item: its `intent_ttl` is
/// `1.5·S`, well over the 15 s that
/// `live_intent_delays_a_new_epoch_only_until_the_gap_plus_intent_ttl` checks
/// as its precondition, on the wired cost model. Tests that stage a time
/// inside the intent window (`gap_until + intent_ttl(L)`) record this size on
/// the fence, so the window outlasts any pause of the test itself by a wide
/// margin.
const LONG_INTENT_SIZE: LayerSize = LayerSize {
    items: 3_500_000,
    bytes: 41_933_837,
};

/// `ceil(4·(rtt99 + Q))` at the shipped ceiling on AMR: the 2,162.9 ms floor of
/// `lease(L)` and `intent_ttl(L)` (report § SR-11.4, amendment 5), in the whole
/// milliseconds the scripts store.
const QUEUE_FLOOR_MS: u64 = 2_163;

/// The tolerance on an absolute key expiry derived from a script's `PEXPIRE`.
/// The script computes its `PEXPIRE` argument from `TIME`, and the server adds
/// it to its own cached command time; the two can differ by a millisecond of
/// truncation. A wrong `key_linger` is off by seconds.
const CLOCK_SLACK_MS: i64 = 2;

/// The expiry [`Fx::mark_expiry`] gives a key before a call that must write
/// nothing: one hour, which no fence script ever sets (their expiries are
/// computed from the deadlines and `key_linger(L)`).
const MARK_EXPIRY_MS: i64 = 3_600_000;

/// How far ahead of the store's clock a staged deadline is "far": past every
/// `poll` and lease a test's sizes produce.
const FAR_MS: i64 = 600_000;

/// Bound on waiting for a dropped guard's spawned clean-up to land. The spawn
/// is one script call; this is unreachable on a healthy store and well inside
/// nextest's 120 s hang-killer.
const DROP_POLL_BUDGET: Duration = Duration::from_secs(10);

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// The store clock, as the fence scripts read it: `TIME` in whole ms. Keyed on
/// the object's fence key so it runs on the object's node.
static CLOCK_SCRIPT: std::sync::LazyLock<Script> = std::sync::LazyLock::new(|| {
    Script::from_lua(
        "local t = redis.call('TIME') \
         return tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)",
    )
});

/// One object's fence, intent and header keys on a fresh `TestValkey`.
struct Fx {
    tv: TestValkey,
    /// The client the wrappers run on: the fixture's own, or one with other
    /// atomic-read limits ([`Fx::using`]).
    client: ValkeyClient,
    oid: u64,
}

/// Every field of a key plus its absolute expiry (`PEXPIRETIME`: `-1` none,
/// `-2` no key). Equal images mean nothing was written.
#[derive(Debug, PartialEq, Eq)]
struct KeyImage {
    fields: BTreeMap<String, String>,
    expires_at: i64,
}

impl Fx {
    /// A fresh store with [`OID`]'s header at `seq` = [`CUT`].
    async fn new() -> Self {
        let fx = Self::without_header().await;
        fx.hset(&fx.header_key(), &[("seq", CUT.to_string())]).await;
        fx
    }

    /// A fresh store with no key under [`OID`].
    async fn without_header() -> Self {
        let tv = TestValkey::setup().await;
        let client = tv.client().clone();
        Fx {
            tv,
            client,
            oid: OID,
        }
    }

    /// The same store and keys, with the wrappers running on `client`.
    fn using(mut self, client: ValkeyClient) -> Self {
        self.client = client;
        self
    }

    fn keys(&self) -> &KeySchema {
        self.tv.key_schema()
    }

    fn fence_key(&self) -> String {
        self.keys().obj_snapshot_fence(&self.oid.to_string())
    }

    fn intent_key(&self) -> String {
        self.keys().obj_snapshot_intent(&self.oid.to_string())
    }

    fn header_key(&self) -> String {
        self.keys().obj_header(&self.oid.to_string())
    }

    /// Every limit for `size`, from the coefficients the wrappers pass.
    fn timing(&self, size: LayerSize) -> FenceTiming {
        fence::coefficients(&self.client).timing(size)
    }

    async fn acquire(&self, estimate: LayerSize) -> AcquireVerdict {
        fence::acquire_fence(&self.client, self.keys(), self.oid, estimate)
            .await
            .expect("acquire_fence must answer")
    }

    async fn grant(&self, estimate: LayerSize) -> Box<FenceLease> {
        granted(self.acquire(estimate).await)
    }

    async fn register(&self, estimate: LayerSize) -> IntentVerdict {
        fence::register_intent(&self.client, self.keys(), self.oid, estimate)
            .await
            .expect("register_intent must answer")
    }

    async fn probe(&self) -> FenceProbe {
        fence::probe_fence(&self.client, self.keys(), self.oid)
            .await
            .expect("probe_fence must answer")
    }

    /// The store's clock in whole ms, read the way the fence scripts read it.
    async fn now(&self) -> i64 {
        CLOCK_SCRIPT
            .evalsha_with_reload::<i64, _, _>(
                self.tv.client().pool.next(),
                vec![self.fence_key()],
                Vec::<String>::new(),
            )
            .await
            .expect("TIME must answer")
    }

    async fn fields(&self, key: &str) -> BTreeMap<String, String> {
        let raw: HashMap<String, String> = self
            .tv
            .client()
            .pool
            .hgetall(key)
            .await
            .expect("HGETALL must succeed");
        raw.into_iter().collect()
    }

    async fn hset(&self, key: &str, fields: &[(&str, String)]) {
        let pairs: Vec<(String, String)> = fields
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect();
        let _: () = self
            .tv
            .client()
            .pool
            .hset(key, pairs)
            .await
            .expect("staging HSET must succeed");
    }

    async fn expires_at(&self, key: &str) -> i64 {
        self.tv
            .client()
            .pool
            .pexpire_time(key)
            .await
            .expect("PEXPIRETIME must succeed")
    }

    async fn pexpire(&self, key: &str, ms: i64) {
        let set: i64 = self
            .tv
            .client()
            .pool
            .pexpire(key, ms, None)
            .await
            .expect("staging PEXPIRE must succeed");
        assert_eq!(set, 1, "staging: {key} must exist to take an expiry");
    }

    /// Give `key` an expiry no fence script sets, so [`KeyImage`] also catches
    /// a stray `PEXPIRE`.
    async fn mark_expiry(&self, key: &str) {
        self.pexpire(key, MARK_EXPIRY_MS).await;
    }

    async fn image(&self, key: &str) -> KeyImage {
        KeyImage {
            fields: self.fields(key).await,
            expires_at: self.expires_at(key).await,
        }
    }

    async fn exists(&self, key: &str) -> bool {
        let n: i64 = self
            .tv
            .client()
            .pool
            .exists(key)
            .await
            .expect("EXISTS must succeed");
        n == 1
    }

    async fn unlink(&self, keys: Vec<String>) {
        let _: i64 = self
            .tv
            .client()
            .pool
            .unlink(keys)
            .await
            .expect("UNLINK must succeed");
    }
}

/// Run `op` between two readings of the store clock: `(before, output, after)`.
/// A value the script computed from its own `now` lies between what it gives
/// for `before` and for `after`.
async fn bracketed<T>(fx: &Fx, op: impl std::future::Future<Output = T>) -> (i64, T, i64) {
    let before = fx.now().await;
    let out = op.await;
    let after = fx.now().await;
    (before, out, after)
}

fn assert_between(value: i64, low: i64, high: i64, what: &str) {
    assert!(
        (low..=high).contains(&value),
        "{what}: {value} must lie in [{low}, {high}]"
    );
}

fn assert_expiry(got: i64, want: i64, what: &str) {
    assert!(
        (got - want).abs() <= CLOCK_SLACK_MS,
        "{what}: the key must expire at {want} (± {CLOCK_SLACK_MS} ms); PEXPIRETIME is {got} \
         ({} ms off)",
        got - want
    );
}

/// A limit in the whole ms the scripts store and return.
fn whole_ms(ms: f64) -> i64 {
    ms.ceil() as i64
}

fn ms(value: i64) -> String {
    value.to_string()
}

fn int(fields: &BTreeMap<String, String>, name: &str) -> i64 {
    fields
        .get(name)
        .unwrap_or_else(|| panic!("{name} must be present; fields: {fields:?}"))
        .parse()
        .unwrap_or_else(|e| panic!("{name} must be an integer: {e}; fields: {fields:?}"))
}

/// The fields under `prefix` (`r:` readers, `w:` writer intents), with their
/// deadlines.
fn deadlines(fields: &BTreeMap<String, String>, prefix: &str) -> BTreeMap<String, i64> {
    fields
        .iter()
        .filter(|(k, _)| k.starts_with(prefix))
        .map(|(k, v)| (k.clone(), v.parse().expect("a deadline is an integer")))
        .collect()
}

/// The one reader field of a single-reader fence.
fn only_reader(fields: &BTreeMap<String, String>) -> (String, i64) {
    let readers = deadlines(fields, "r:");
    assert_eq!(readers.len(), 1, "exactly one reader expected: {fields:?}");
    readers.into_iter().next().expect("one reader")
}

/// The reader field `after` has and `before` did not.
fn new_reader(
    before: &BTreeMap<String, String>,
    after: &BTreeMap<String, String>,
) -> (String, i64) {
    let mut fresh: Vec<(String, i64)> = deadlines(after, "r:")
        .into_iter()
        .filter(|(k, _)| !before.contains_key(k))
        .collect();
    assert_eq!(fresh.len(), 1, "exactly one new reader expected: {after:?}");
    fresh.pop().expect("one new reader")
}

fn granted(verdict: AcquireVerdict) -> Box<FenceLease> {
    match verdict {
        AcquireVerdict::Granted(lease) => lease,
        other => panic!("acquire must grant; got {other:?}"),
    }
}

/// `(hint ms, reason, size)` of a `RETRY_AFTER`.
fn retry_after(verdict: AcquireVerdict) -> (i64, RetryReason, LayerSize) {
    match verdict {
        AcquireVerdict::RetryAfter { hint, reason, size } => {
            (hint.as_millis() as i64, reason, size)
        }
        other => panic!("acquire must answer RETRY_AFTER; got {other:?}"),
    }
}

fn registered(verdict: IntentVerdict) -> Box<WriterIntent> {
    match verdict {
        IntentVerdict::Registered(intent) => intent,
        other => panic!("register_intent must register; got {other:?}"),
    }
}

fn assert_size_fields(fields: &BTreeMap<String, String>, size: LayerSize, what: &str) {
    assert_eq!(
        (fields.get("size_e"), fields.get("size_b")),
        (Some(&size.items.to_string()), Some(&size.bytes.to_string())),
        "{what}: the stored size must be {size:?}; fields: {fields:?}"
    );
    match size.bytes_per_item() {
        Some(bpi) => {
            let stored: f64 = fields
                .get("bpi")
                .unwrap_or_else(|| panic!("{what}: bpi must be stored; fields: {fields:?}"))
                .parse()
                .expect("bpi must be a number");
            assert_eq!(
                stored.to_bits(),
                bpi.to_bits(),
                "{what}: bpi must be exactly size_b / size_e ({bpi}); stored {stored}"
            );
        }
        None => assert!(
            !fields.contains_key("bpi"),
            "{what}: an empty layer has no bytes per item; fields: {fields:?}"
        ),
    }
}

/// Poll `done` until it holds, within [`DROP_POLL_BUDGET`].
async fn eventually<F, Fut>(what: &str, mut done: F)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let wait = async {
        while !done().await {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(DROP_POLL_BUDGET, wait)
        .await
        .unwrap_or_else(|_| panic!("{what} did not happen within {DROP_POLL_BUDGET:?}"));
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. Acquire (SR-5.2)
// ═════════════════════════════════════════════════════════════════════════════

/// No header, or a header without a numeric `seq`: `NOT_FOUND`, and no fence
/// key is created.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acquire_without_a_numeric_header_seq_answers_not_found_and_writes_nothing() {
    let fx = Fx::without_header().await;
    for seq in [None, Some("not-a-number")] {
        if let Some(seq) = seq {
            fx.hset(&fx.header_key(), &[("seq", seq.to_owned())]).await;
        }
        let verdict = fx.acquire(SIZE).await;
        assert!(
            matches!(verdict, AcquireVerdict::NotFound),
            "header seq {seq:?}: acquire must answer NOT_FOUND; got {verdict:?}"
        );
        assert!(
            !fx.exists(&fx.fence_key()).await,
            "header seq {seq:?}: a refused acquire must not create the fence key"
        );
    }
}

/// A grant on a fresh key starts an epoch at the header `seq`: one lease of
/// `lease(L)` from the epoch's start, the estimate stored as the layer's size,
/// and the key expiring `key_linger(L)` after the lease.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acquire_on_a_fresh_fence_starts_an_epoch_at_the_header_seq() {
    let fx = Fx::new().await;
    let t = fx.timing(SIZE);

    let lease = fx.grant(SIZE).await;

    assert_eq!(lease.kind(), GrantKind::StartedEpoch);
    assert_eq!(lease.cut(), CUT, "the cut is the header seq at the grant");
    assert_eq!(lease.size(), SIZE, "a fresh key's size is the estimate");
    assert_eq!(lease.timing(), t, "the lease's limits are the size's");
    assert_eq!(
        lease.renew_interval(),
        ms_to_duration(t.lease_ms / 4.0),
        "renew every lease(L)/4"
    );
    let fields = fx.fields(&fx.fence_key()).await;
    let (_, deadline) = only_reader(&fields);
    let start = int(&fields, "epoch_start");
    assert_eq!(
        deadline - start,
        whole_ms(t.lease_ms),
        "the lease runs lease(L) from the epoch's start; fields: {fields:?}"
    );
    assert_size_fields(&fields, SIZE, "fresh grant");
    assert!(
        !fields.contains_key("gap_until"),
        "a first epoch has no gap: {fields:?}"
    );
    assert_expiry(
        fx.expires_at(&fx.fence_key()).await,
        deadline + whole_ms(t.key_linger_ms),
        "grant: latest deadline + key_linger(L)",
    );
}

/// A second reader inside the join window joins the live epoch: same cut, the
/// stored size rather than its own estimate, the epoch's start untouched.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acquire_inside_the_join_window_joins_the_epoch_under_the_stored_size() {
    let fx = Fx::new().await;
    let _first = fx.grant(SIZE).await;
    let before = fx.fields(&fx.fence_key()).await;

    let second = fx.grant(OTHER_SIZE).await;

    assert_eq!(second.kind(), GrantKind::JoinedEpoch);
    assert_eq!(second.cut(), CUT);
    assert_eq!(
        second.size(),
        SIZE,
        "the stored size wins over the joiner's estimate, so every reader of the epoch derives \
         the same deadlines"
    );
    let after = fx.fields(&fx.fence_key()).await;
    assert_eq!(
        after.get("epoch_start"),
        before.get("epoch_start"),
        "joining must not restart the epoch"
    );
    assert_size_fields(&after, SIZE, "after a join");
    let readers = deadlines(&after, "r:");
    assert_eq!(readers.len(), 2, "both readers hold leases: {after:?}");
    let latest = *readers.values().max().expect("two readers");
    assert_expiry(
        fx.expires_at(&fx.fence_key()).await,
        latest + whole_ms(fx.timing(SIZE).key_linger_ms),
        "join: latest deadline + key_linger(L)",
    );
}

/// A reader arriving after a live epoch's join window waits `epoch`, for
/// `min(latest live deadline − now, poll)`, and writes nothing.
///
/// The epoch is staged to have started one millisecond more than
/// `join_window` ago, so the reader is outside the window by the smallest
/// margin the store's clock resolves: a join threshold any longer than
/// `join_window` (`lease`, `epoch`, …) would let it join. Time passing
/// between the staging and the call only moves it further outside.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acquire_outside_the_join_window_waits_for_the_epoch_and_writes_nothing() {
    let fx = Fx::new().await;
    let _holder = fx.grant(SIZE).await;
    let (holder_field, deadline) = only_reader(&fx.fields(&fx.fence_key()).await);
    let t = fx.timing(SIZE);
    let poll = whole_ms(t.poll_ms);
    let s = fx.now().await;
    fx.hset(
        &fx.fence_key(),
        &[("epoch_start", ms(s - whole_ms(t.join_window_ms) - 1))],
    )
    .await;
    fx.mark_expiry(&fx.fence_key()).await;
    let image = fx.image(&fx.fence_key()).await;

    let (t0, verdict, t1) = bracketed(&fx, fx.acquire(OTHER_SIZE)).await;

    let (hint, reason, size) = retry_after(verdict);
    assert_eq!(reason, RetryReason::Epoch);
    assert_eq!(size, SIZE, "the wait is sized from the stored size");
    assert_between(
        hint,
        poll.min(deadline - t1),
        poll.min(deadline - t0),
        "hint = min(latest live deadline − now, poll)",
    );
    assert_eq!(
        fx.image(&fx.fence_key()).await,
        image,
        "a waiting reader must write nothing"
    );

    // The live deadline caps the hint when it is nearer than `poll`: staged
    // just under `poll` ahead, so the call has most of a second to land
    // before the lease would lapse.
    let s = fx.now().await;
    let near = s + poll - 100;
    fx.hset(&fx.fence_key(), &[(holder_field.as_str(), ms(near))])
        .await;
    let image = fx.image(&fx.fence_key()).await;
    let (t0, verdict, t1) = bracketed(&fx, fx.acquire(OTHER_SIZE)).await;
    let (hint, reason, _) = retry_after(verdict);
    assert_eq!(reason, RetryReason::Epoch);
    assert_between(
        hint,
        near - t1,
        near - t0,
        "hint = latest live deadline − now",
    );
    assert_eq!(fx.image(&fx.fence_key()).await, image);
}

/// Readers arriving in the writers' gap wait `gap` until `gap_until` exactly —
/// every waiting reader wakes at the same instant — and write nothing; at that
/// instant the next reader starts an epoch and the passed gap is cleared.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acquire_in_the_writers_gap_waits_until_the_shared_instant_and_writes_nothing() {
    let fx = Fx::new().await;
    let lease = fx.grant(SIZE).await;
    assert_eq!(
        lease.release(Some(SIZE)).await.unwrap(),
        ReleaseVerdict::Verified
    );
    let gap_until = fx.now().await + 60_000;
    fx.hset(&fx.fence_key(), &[("gap_until", ms(gap_until))])
        .await;
    fx.mark_expiry(&fx.fence_key()).await;
    let image = fx.image(&fx.fence_key()).await;

    for reader in ["first", "second"] {
        let (t0, verdict, t1) = bracketed(&fx, fx.acquire(OTHER_SIZE)).await;
        let (hint, reason, size) = retry_after(verdict);
        assert_eq!(reason, RetryReason::Gap, "{reader} reader");
        assert_eq!(size, SIZE, "{reader} reader: the stored size");
        assert_between(
            hint,
            gap_until - t1,
            gap_until - t0,
            &format!("{reader} reader: hint = gap_until − now, so it wakes at gap_until"),
        );
    }
    assert_eq!(
        fx.image(&fx.fence_key()).await,
        image,
        "readers waiting out the gap must write nothing"
    );

    let passed = fx.now().await - 1;
    fx.hset(&fx.fence_key(), &[("gap_until", ms(passed))]).await;
    let next = fx.grant(OTHER_SIZE).await;
    assert_eq!(next.kind(), GrantKind::StartedEpoch);
    assert!(
        !fx.fields(&fx.fence_key()).await.contains_key("gap_until"),
        "the grant clears a gap that has passed"
    );
}

/// A holder that crashed without releasing leaves `epoch_start` set and an
/// expired lease. Its epoch still owes writers a gap — `ended + max(min_gap,
/// ended − epoch_start)` from its last deadline — which readers wait out
/// (writing nothing, the dead lease included); once it has passed, the next
/// grant prunes the dead lease and starts a new epoch.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acquire_after_a_crashed_holder_waits_out_the_gap_its_epoch_owed() {
    let fx = Fx::new().await;
    let _crashed = fx.grant(SIZE).await;
    let (crashed_field, _) = only_reader(&fx.fields(&fx.fence_key()).await);
    let s = fx.now().await;
    // A 4 s epoch whose lease ran out 1 s ago, never released.
    fx.hset(
        &fx.fence_key(),
        &[
            (crashed_field.as_str(), ms(s - 1_000)),
            ("epoch_start", ms(s - 5_000)),
        ],
    )
    .await;
    fx.mark_expiry(&fx.fence_key()).await;
    let image = fx.image(&fx.fence_key()).await;
    let owed_gap = (s - 1_000) + 4_000;

    let (t0, verdict, t1) = bracketed(&fx, fx.acquire(OTHER_SIZE)).await;

    let (hint, reason, _) = retry_after(verdict);
    assert_eq!(
        reason,
        RetryReason::Gap,
        "a crash must not skip the writers' gap"
    );
    assert_between(hint, owed_gap - t1, owed_gap - t0, "hint = owed gap − now");
    assert_eq!(
        fx.image(&fx.fence_key()).await,
        image,
        "a waiting reader writes nothing, not even the dead lease's removal"
    );

    // Long ago: a 1 s epoch that ended 20 s ago owed a gap that has passed.
    fx.hset(
        &fx.fence_key(),
        &[
            (crashed_field.as_str(), ms(s - 20_000)),
            ("epoch_start", ms(s - 21_000)),
        ],
    )
    .await;
    let next = fx.grant(OTHER_SIZE).await;
    assert_eq!(next.kind(), GrantKind::StartedEpoch);
    let fields = fx.fields(&fx.fence_key()).await;
    assert!(
        !fields.contains_key(&crashed_field),
        "the grant prunes the dead lease: {fields:?}"
    );
    assert!(
        int(&fields, "epoch_start") >= s,
        "the grant replaces the stale epoch start: {fields:?}"
    );
}

/// After the gap, a live writer intent holds new epochs off (`intent`, for
/// `min(latest intent deadline − now, poll)`, writing nothing) — but only
/// until `gap_until + intent_ttl(L)`, after which readers start an epoch
/// whatever the intents say.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_intent_delays_a_new_epoch_only_until_the_gap_plus_intent_ttl() {
    let fx = Fx::new().await;
    let lease = fx.grant(SIZE).await;
    assert_eq!(
        lease.release(Some(LONG_INTENT_SIZE)).await.unwrap(),
        ReleaseVerdict::Verified
    );
    let t = fx.timing(LONG_INTENT_SIZE);
    let s = fx.now().await;
    fx.hset(&fx.fence_key(), &[("gap_until", ms(s - 100))])
        .await;
    fx.hset(&fx.intent_key(), &[("w:writer", ms(s + FAR_MS))])
        .await;
    for key in [fx.fence_key(), fx.intent_key()] {
        fx.mark_expiry(&key).await;
    }
    let images = (
        fx.image(&fx.fence_key()).await,
        fx.image(&fx.intent_key()).await,
    );

    let (hint, reason, size) = retry_after(fx.acquire(OTHER_SIZE).await);
    assert_eq!(reason, RetryReason::Intent);
    assert_eq!(size, LONG_INTENT_SIZE, "the stored size");
    assert_eq!(
        hint,
        whole_ms(t.poll_ms),
        "a distant intent deadline: the reader polls"
    );
    assert_eq!(
        (
            fx.image(&fx.fence_key()).await,
            fx.image(&fx.intent_key()).await
        ),
        images,
        "a reader waiting on an intent writes nothing to either key"
    );

    // A near intent deadline caps the hint: staged just under `poll` ahead, so
    // the call has most of a second to land before the intent would lapse.
    let near = fx.now().await + whole_ms(t.poll_ms) - 100;
    fx.hset(&fx.intent_key(), &[("w:writer", ms(near))]).await;
    let (t0, verdict, t1) = bracketed(&fx, fx.acquire(OTHER_SIZE)).await;
    let (hint, reason, _) = retry_after(verdict);
    assert_eq!(reason, RetryReason::Intent);
    assert_between(hint, near - t1, near - t0, "hint = intent deadline − now");

    // The window is the stored size's: 5 s after the gap is past the window
    // the caller's estimate would give, but well inside the stored size's
    // (over 15 s, checked here), so the intent still holds the reader off.
    let estimate_window = whole_ms(fx.timing(OTHER_SIZE).intent_ttl_ms);
    let stored_window = whole_ms(t.intent_ttl_ms);
    assert!(
        estimate_window < 5_000 && 5_000 + 10_000 < stored_window,
        "precondition: 5 s past the gap must be outside the estimate's window ({estimate_window} \
         ms) and well inside the stored size's ({stored_window} ms)"
    );
    let s = fx.now().await;
    fx.hset(&fx.intent_key(), &[("w:writer", ms(s + FAR_MS))])
        .await;
    fx.hset(&fx.fence_key(), &[("gap_until", ms(s - 5_000))])
        .await;
    let (_, reason, _) = retry_after(fx.acquire(OTHER_SIZE).await);
    assert_eq!(
        reason,
        RetryReason::Intent,
        "intents are honoured for intent_ttl of the stored size, not the caller's estimate"
    );

    // Past gap_until + intent_ttl(L), by the smallest margin the store's clock
    // resolves, a live intent no longer holds readers off. Time passing between
    // the staging and the call only moves the reader further past it.
    let s = fx.now().await;
    fx.hset(&fx.fence_key(), &[("gap_until", ms(s - stored_window - 1))])
        .await;
    let next = fx.grant(OTHER_SIZE).await;
    assert_eq!(
        next.kind(),
        GrantKind::StartedEpoch,
        "readers honour intents only until gap_until + intent_ttl(L), so writers cannot starve \
         them"
    );
}

/// With no gap recorded there is no writers' window to extend: a first epoch
/// starts whatever intents are live.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acquire_with_no_recorded_gap_ignores_live_intents() {
    let fx = Fx::new().await;
    let s = fx.now().await;
    fx.hset(&fx.intent_key(), &[("w:writer", ms(s + FAR_MS))])
        .await;

    let lease = fx.grant(SIZE).await;

    assert_eq!(lease.kind(), GrantKind::StartedEpoch);
}

/// An intent that has lapsed holds nobody off, inside the window or not.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acquire_inside_the_intent_window_ignores_lapsed_intents() {
    let fx = Fx::new().await;
    let lease = fx.grant(SIZE).await;
    assert_eq!(
        lease.release(Some(LONG_INTENT_SIZE)).await.unwrap(),
        ReleaseVerdict::Verified
    );
    let s = fx.now().await;
    fx.hset(&fx.fence_key(), &[("gap_until", ms(s - 100))])
        .await;
    fx.hset(&fx.intent_key(), &[("w:writer", ms(s - 1))]).await;

    let next = fx.grant(SIZE).await;

    assert_eq!(next.kind(), GrantKind::StartedEpoch);
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. Renew (SR-5.3)
// ═════════════════════════════════════════════════════════════════════════════

/// Renewing a live lease moves its deadline to `now + lease(L)` and raises the
/// key's expiry to that deadline + `key_linger(L)`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn renew_extends_a_live_lease_and_raises_the_key_expiry() {
    let fx = Fx::new().await;
    let lease = fx.grant(SIZE).await;
    let t = fx.timing(SIZE);
    let (field, _) = only_reader(&fx.fields(&fx.fence_key()).await);
    let s = fx.now().await;
    fx.hset(&fx.fence_key(), &[(field.as_str(), ms(s + 2_000))])
        .await;
    fx.pexpire(&fx.fence_key(), 3_000).await;

    let (t0, verdict, t1) = bracketed(&fx, lease.renew()).await;

    assert_eq!(verdict.unwrap(), RenewVerdict::Renewed);
    let renewed = int(&fx.fields(&fx.fence_key()).await, &field);
    let lease_ms = whole_ms(t.lease_ms);
    assert_between(
        renewed,
        t0 + lease_ms,
        t1 + lease_ms,
        "renewed deadline = now + lease(L)",
    );
    assert_expiry(
        fx.expires_at(&fx.fence_key()).await,
        renewed + whole_ms(t.key_linger_ms),
        "renew: deadline + key_linger(L)",
    );
}

/// A renewal never shortens: a lease already further out than `now +
/// lease(L)`, and a key expiry already later than the renewal's, stay as they
/// are.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn renew_never_shortens_a_lease_or_the_key_expiry() {
    let fx = Fx::new().await;
    let lease = fx.grant(SIZE).await;
    let (field, _) = only_reader(&fx.fields(&fx.fence_key()).await);
    let s = fx.now().await;
    fx.hset(&fx.fence_key(), &[(field.as_str(), ms(s + FAR_MS))])
        .await;
    fx.mark_expiry(&fx.fence_key()).await;
    let image = fx.image(&fx.fence_key()).await;

    assert_eq!(lease.renew().await.unwrap(), RenewVerdict::Renewed);

    assert_eq!(fx.image(&fx.fence_key()).await, image);
}

/// A lease whose deadline has passed answers `LOST` and is not revived — the
/// field, the rest of the key and its expiry stay as they were — and its
/// release answers `BROKEN`. A lease whose key was wiped answers `LOST` too,
/// and no key is created.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn renew_after_the_lease_expired_or_was_wiped_answers_lost_and_writes_nothing() {
    let fx = Fx::new().await;
    let expired = fx.grant(SIZE).await;
    let before = fx.fields(&fx.fence_key()).await;
    let wiped = fx.grant(SIZE).await;
    let (expired_field, _) = only_reader(&before);
    let s = fx.now().await;
    fx.hset(&fx.fence_key(), &[(expired_field.as_str(), ms(s - 1))])
        .await;
    fx.mark_expiry(&fx.fence_key()).await;
    let image = fx.image(&fx.fence_key()).await;

    assert_eq!(
        expired.renew().await.unwrap(),
        RenewVerdict::Lost,
        "a lease whose deadline has passed is lost"
    );
    assert_eq!(
        fx.image(&fx.fence_key()).await,
        image,
        "an expired lease must not be revived: writers judge expiry on the same clock and may \
         already have resumed"
    );
    assert_eq!(
        expired.release(None).await.unwrap(),
        ReleaseVerdict::Broken,
        "a lost lease cannot prove its read"
    );

    fx.unlink(vec![fx.fence_key()]).await;
    assert_eq!(wiped.renew().await.unwrap(), RenewVerdict::Lost);
    assert!(
        !fx.exists(&fx.fence_key()).await,
        "a renew of a wiped lease must not recreate the key"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. Release (SR-5.3)
// ═════════════════════════════════════════════════════════════════════════════

/// The last live reader's release at an unmoved header is `VERIFIED` and ends
/// the epoch: the lease and `epoch_start` go, `gap_until` = `now + max(min_gap,
/// now − epoch_start)`, the measured size is recorded, and the key expires
/// `key_linger(measured)` after the gap.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_at_an_unmoved_header_is_verified_and_opens_a_gap_as_long_as_the_epoch() {
    let fx = Fx::new().await;
    let lease = fx.grant(SIZE).await;
    let epoch_start = fx.now().await - 5_000;
    fx.hset(&fx.fence_key(), &[("epoch_start", ms(epoch_start))])
        .await;

    let (t0, verdict, t1) = bracketed(&fx, lease.release(Some(OTHER_SIZE))).await;

    assert_eq!(verdict.unwrap(), ReleaseVerdict::Verified);
    let fields = fx.fields(&fx.fence_key()).await;
    assert!(
        deadlines(&fields, "r:").is_empty(),
        "lease given up: {fields:?}"
    );
    assert!(
        !fields.contains_key("epoch_start"),
        "the epoch ended: {fields:?}"
    );
    let gap_until = int(&fields, "gap_until");
    assert_between(
        gap_until,
        t0 + (t0 - epoch_start),
        t1 + (t1 - epoch_start),
        "gap_until = now + (now − epoch_start) for an epoch longer than min_gap",
    );
    assert_size_fields(&fields, OTHER_SIZE, "the measured size is recorded");
    assert_expiry(
        fx.expires_at(&fx.fence_key()).await,
        gap_until + whole_ms(fx.timing(OTHER_SIZE).key_linger_ms),
        "release: gap_until + key_linger(measured)",
    );
}

/// An epoch shorter than `min_gap` still owes writers `min_gap`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_of_a_short_epoch_still_opens_at_least_min_gap() {
    let fx = Fx::new().await;
    let lease = fx.grant(SIZE).await;
    let epoch_start = int(&fx.fields(&fx.fence_key()).await, "epoch_start");
    let min_gap = whole_ms(fx.timing(SIZE).min_gap_ms);

    let (t0, verdict, t1) = bracketed(&fx, lease.release(None)).await;

    assert_eq!(verdict.unwrap(), ReleaseVerdict::Verified);
    let gap_until = int(&fx.fields(&fx.fence_key()).await, "gap_until");
    // gap − now = max(min_gap, ceil(now − epoch_start)), increasing in now.
    assert_between(
        gap_until,
        t0 + min_gap.max(t0 - epoch_start),
        t1 + min_gap.max(t1 - epoch_start + 1),
        "gap_until = now + max(min_gap, now − epoch_start)",
    );
    assert!(gap_until >= t0 + min_gap, "never shorter than min_gap");
}

/// A release is `BROKEN` when its read cannot be proven — the header `seq`
/// moved past the cut (a restore or other non-commit mutator), or its lease had
/// expired — and still gives the lease up and ends the epoch: the gap is owed
/// to writers either way.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_is_broken_when_the_seq_moved_or_the_lease_expired_and_still_ends_the_epoch() {
    for cause in ["seq moved", "lease expired"] {
        let fx = Fx::new().await;
        let lease = fx.grant(SIZE).await;
        let (field, _) = only_reader(&fx.fields(&fx.fence_key()).await);
        match cause {
            "seq moved" => {
                fx.hset(&fx.header_key(), &[("seq", (CUT + 1).to_string())])
                    .await
            }
            _ => {
                let s = fx.now().await;
                fx.hset(&fx.fence_key(), &[(field.as_str(), ms(s - 1))])
                    .await
            }
        }

        let verdict = lease.release(None).await.unwrap();

        assert_eq!(verdict, ReleaseVerdict::Broken, "{cause}");
        let fields = fx.fields(&fx.fence_key()).await;
        assert!(
            !fields.contains_key(&field) && !fields.contains_key("epoch_start"),
            "{cause}: the lease is given up and the epoch ended: {fields:?}"
        );
        let gap_until = int(&fields, "gap_until");
        assert_size_fields(&fields, SIZE, &format!("{cause}: the granted size"));
        assert_expiry(
            fx.expires_at(&fx.fence_key()).await,
            gap_until + whole_ms(fx.timing(SIZE).key_linger_ms),
            &format!("{cause}: gap_until + key_linger(L)"),
        );
    }
}

/// While a joiner's lease is live the epoch goes on for it: an earlier release
/// sets no gap and records no size, and the key expires after the joiner's
/// lease. The joiner's own release then ends the epoch.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_while_another_reader_is_live_keeps_the_epoch_open() {
    let fx = Fx::new().await;
    let first = fx.grant(SIZE).await;
    let before = fx.fields(&fx.fence_key()).await;
    let joiner = fx.grant(SIZE).await;
    let (joiner_field, joiner_deadline) = new_reader(&before, &fx.fields(&fx.fence_key()).await);

    assert_eq!(
        first.release(Some(OTHER_SIZE)).await.unwrap(),
        ReleaseVerdict::Verified
    );

    let fields = fx.fields(&fx.fence_key()).await;
    assert_eq!(
        deadlines(&fields, "r:").into_keys().collect::<Vec<_>>(),
        vec![joiner_field],
        "only the joiner's lease remains"
    );
    assert!(
        fields.contains_key("epoch_start") && !fields.contains_key("gap_until"),
        "the epoch is still open: {fields:?}"
    );
    assert_size_fields(&fields, SIZE, "only the epoch's last reader records a size");
    assert_expiry(
        fx.expires_at(&fx.fence_key()).await,
        joiner_deadline + whole_ms(fx.timing(SIZE).key_linger_ms),
        "the joiner's deadline + key_linger(L)",
    );

    assert_eq!(
        joiner.release(Some(OTHER_SIZE)).await.unwrap(),
        ReleaseVerdict::Verified
    );
    let fields = fx.fields(&fx.fence_key()).await;
    assert!(
        !fields.contains_key("epoch_start") && fields.contains_key("gap_until"),
        "the last reader ends the epoch: {fields:?}"
    );
    assert_size_fields(&fields, OTHER_SIZE, "the last reader records its size");
}

/// The late release: reader A closes a 3 s epoch while joiner B's lease has
/// already expired, so A's release is the epoch's last. B's later release
/// answers `BROKEN` and removes only its own field: `gap_until` (which covers
/// the 3 s epoch), the stored size and `bpi` stay A's, and the key's expiry is
/// unchanged. Recomputing the gap there, with no epoch start left to measure
/// from, would shrink it below the epoch's length.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn late_release_after_the_epoch_closed_removes_only_its_own_field() {
    const LATE_SIZE: LayerSize = LayerSize {
        items: 1_000,
        bytes: 7,
    };
    let fx = Fx::new().await;
    let a = fx.grant(SIZE).await;
    let before = fx.fields(&fx.fence_key()).await;
    let b = fx.grant(SIZE).await;
    let (b_field, _) = new_reader(&before, &fx.fields(&fx.fence_key()).await);
    let s = fx.now().await;
    fx.hset(
        &fx.fence_key(),
        &[
            (b_field.as_str(), ms(s - 5)),
            ("epoch_start", ms(s - 3_000)),
        ],
    )
    .await;

    assert_eq!(
        a.release(Some(OTHER_SIZE)).await.unwrap(),
        ReleaseVerdict::Verified
    );
    let after_a = fx.fields(&fx.fence_key()).await;
    assert!(
        int(&after_a, "gap_until") >= s + 3_000,
        "A's release closes the 3 s epoch with a gap at least as long: {after_a:?}"
    );
    assert!(
        after_a.contains_key(&b_field),
        "precondition: B's dead lease is still there"
    );
    let expiry_after_a = fx.expires_at(&fx.fence_key()).await;

    assert_eq!(
        b.release(Some(LATE_SIZE)).await.unwrap(),
        ReleaseVerdict::Broken,
        "B's lease had expired"
    );

    let mut expected = after_a;
    expected.remove(&b_field);
    assert_eq!(
        fx.fields(&fx.fence_key()).await,
        expected,
        "the late release removes only its own field: gap_until, size_e, size_b and bpi stay A's"
    );
    assert_expiry(
        fx.expires_at(&fx.fence_key()).await,
        expiry_after_a,
        "the late release leaves the key's expiry where A's release put it",
    );
}

/// Release after `DeleteLayer` retired the object answers `NOT_FOUND` and
/// creates no fence key: retirement `UNLINK`s the fence with the object's other
/// keys, and the release finds no header.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_after_delete_layer_retired_the_object_answers_not_found_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let layer = "fence-retired";
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(layer, &format!("/{layer}"), "fence-user")
        .await
        .expect("session creation must succeed");
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        ..Default::default()
    };
    let oid = ls
        .create_layer(
            &session_id,
            layer,
            "fence-user",
            &state,
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("create_layer must succeed");
    let client = tv.client().clone();
    let fx = Fx { tv, client, oid };
    let lease = fx.grant(SIZE).await;
    assert!(
        fx.exists(&fx.fence_key()).await,
        "precondition: the grant wrote the fence"
    );

    ls.delete_layer(&session_id, layer, Arc::new(MockExternalValueStore::new()))
        .await
        .expect("delete_layer must succeed");
    assert!(
        !fx.exists(&fx.fence_key()).await,
        "retirement UNLINKs the fence key with the object"
    );

    assert_eq!(lease.release(None).await.unwrap(), ReleaseVerdict::NotFound);
    assert!(
        !fx.exists(&fx.fence_key()).await,
        "a release after retirement must not recreate the fence key"
    );
}

/// Release after a restore wipe — the fence and intent keys `UNLINK`ed with the
/// object's other keys, then the header written back by the restore — answers
/// `BROKEN` and writes nothing: the reader's own field is gone, and the script
/// stops there (release step 2), before any write.
///
/// Three pre-wipe readers release in turn:
/// - with no fence key at all, the release creates none;
/// - once a fresh reader holds a new epoch on the restored object, a release
///   leaves that epoch's key byte-identical — without step 2 it would re-expire
///   the key from the new reader's deadline;
/// - and once the new reader's lease has expired, a release still leaves the
///   key byte-identical — without step 2 it would close the new epoch, writing
///   `gap_until` and its own size and removing `epoch_start`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_after_a_restore_wipe_is_broken_and_writes_nothing() {
    let fx = Fx::new().await;
    let before_wipe = fx.grant(SIZE).await;
    let before_new_epoch = fx.grant(SIZE).await;
    let after_restored_reader_expired = fx.grant(SIZE).await;
    fx.unlink(vec![fx.header_key(), fx.fence_key(), fx.intent_key()])
        .await;
    fx.hset(&fx.header_key(), &[("seq", "7".to_owned())]).await;

    assert_eq!(
        before_wipe.release(None).await.unwrap(),
        ReleaseVerdict::Broken
    );
    assert!(
        !fx.exists(&fx.fence_key()).await,
        "a release on the restored object must not recreate the wiped fence key"
    );

    let restored_reader = fx.grant(OTHER_SIZE).await;
    assert_eq!(restored_reader.kind(), GrantKind::StartedEpoch);
    assert_eq!(
        restored_reader.cut(),
        7,
        "the new epoch is on the restored object"
    );
    fx.mark_expiry(&fx.fence_key()).await;
    let image = fx.image(&fx.fence_key()).await;
    assert_eq!(
        before_new_epoch.release(Some(SIZE)).await.unwrap(),
        ReleaseVerdict::Broken
    );
    assert_eq!(
        fx.image(&fx.fence_key()).await,
        image,
        "a pre-wipe release must leave the restored object's live epoch untouched"
    );

    let (new_field, _) = only_reader(&fx.fields(&fx.fence_key()).await);
    let s = fx.now().await;
    fx.hset(&fx.fence_key(), &[(new_field.as_str(), ms(s - 1))])
        .await;
    let image = fx.image(&fx.fence_key()).await;
    assert!(
        image.fields.contains_key("epoch_start"),
        "precondition: the new epoch is still open, so a release that went on could close it"
    );
    assert_eq!(
        after_restored_reader_expired
            .release(Some(SIZE))
            .await
            .unwrap(),
        ReleaseVerdict::Broken
    );
    assert_eq!(
        fx.image(&fx.fence_key()).await,
        image,
        "a pre-wipe release must not close the restored object's epoch"
    );
}

/// A lease dropped without `release` — a cancelled read, a panic, an early `?`
/// — is released in the background: its field goes and the epoch ends.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_an_unreleased_lease_releases_it_in_the_background() {
    let fx = Fx::new().await;
    let lease = fx.grant(SIZE).await;
    let (field, _) = only_reader(&fx.fields(&fx.fence_key()).await);

    drop(lease);

    let fx = &fx;
    eventually("the dropped lease's release", move || {
        let field = field.clone();
        async move {
            let fields = fx.fields(&fx.fence_key()).await;
            !fields.contains_key(&field) && fields.contains_key("gap_until")
        }
    })
    .await;
}

/// A client of its own, on the fixture's store, for a test that quits it: the
/// fixture's own client must stay open for its checks and its clean-up.
async fn own_client() -> ValkeyClient {
    connect_test_client(&crate::common::test_config())
        .await
        .expect("a client of the test's own must connect")
}

/// `ValkeyClient::quit` waits for a dropped lease's background release before
/// it closes the pools (post-completion review F12), so a fenced read that
/// graceful shutdown drops gives its lease back rather than leaving the
/// layer's writers paused until it lapses. On a current-thread runtime the
/// release the drop spawns cannot run before the test's next await, so it is
/// still in flight — tracked on the client — when `quit` starts. Once `quit`
/// returns nothing is in flight, and the store shows a release, not a lapse:
/// the reader field is gone and the epoch's gap opened (a lapse removes
/// nothing).
#[tokio::test(flavor = "current_thread")]
async fn quit_waits_for_a_dropped_leases_release_before_closing_the_pools() {
    let fx = Fx::new().await.using(own_client().await);
    let lease = fx.grant(SIZE).await;
    let (field, _) = only_reader(&fx.fields(&fx.fence_key()).await);

    drop(lease);
    assert_eq!(
        fx.client.fence_guard_cleanups_inflight(),
        1,
        "the dropped lease's release is spawned tracked on the client, not yet run"
    );
    fx.client.quit().await;

    assert_eq!(
        fx.client.fence_guard_cleanups_inflight(),
        0,
        "quit waited for the release"
    );
    let fields = fx.fields(&fx.fence_key()).await;
    assert!(
        !fields.contains_key(&field) && fields.contains_key("gap_until"),
        "the release reached the store before the pools closed — the reader field removed and \
         the epoch's gap opened; fields: {fields:?}"
    );
}

/// Hang guard on a `quit` whose fence-guard cleanup drain runs out: the drain
/// gives up after 2 s and each pool's `QUIT` after 2 s more, so a healthy run
/// returns in about 4 s. A deadline, never a timing claim.
const QUIT_HANG_GUARD: Duration = Duration::from_secs(30);

/// `ValkeyClient::quit`'s fence-guard cleanup drain is bounded (post-completion
/// test review F4): with every data-pool connection stalled, a dropped lease's
/// release cannot reach the store, so `quit` gives up on it, logs the WARN
/// naming how many cleanups it left, and closes the pools anyway, the release
/// still in flight (the lease then lapses at its own deadline). The client's
/// command timeout is raised far past the test, so the stalled release cannot
/// end by timing out first. The stalls are released once `quit` returned.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn quit_gives_up_on_a_stalled_dropped_lease_release_with_a_warn_and_closes_the_pools() {
    let mut config = crate::common::test_config();
    config.valkey_command_timeout_ms = 600_000;
    let client = connect_test_client(&config)
        .await
        .expect("a client of the test's own must connect");
    let fx = Fx::new().await.using(client);
    let lease = fx.grant(SIZE).await;
    let observer = fx.tv.client().pool.next().clone();
    let stalls = live_edit::storage::test_support::stall_pool(
        &fx.client.pool,
        &observer,
        &format!("{}conn-stall:quit-data:", fx.tv.prefix()),
    )
    .await;

    drop(lease);
    assert_eq!(
        fx.client.fence_guard_cleanups_inflight(),
        1,
        "the dropped lease's release is tracked, and cannot finish on a stalled pool"
    );
    let (quit, captured) = crate::grpc::log_capture::capture_debug(tokio::time::timeout(
        QUIT_HANG_GUARD,
        fx.client.quit(),
    ))
    .await;

    quit.unwrap_or_else(|_| {
        panic!("quit returned within {QUIT_HANG_GUARD:?}: its cleanup drain is bounded")
    });
    let warn = captured
        .events
        .iter()
        .find(|e| {
            e["fields"]["message"].as_str().is_some_and(|m| {
                m.starts_with("timed out waiting for dropped snapshot-fence leases")
            })
        })
        .unwrap_or_else(|| {
            panic!(
                "quit logs that it gave up on the cleanup:\n{}",
                captured.text
            )
        });
    assert_eq!(warn["level"], "WARN", "{warn}");
    assert_eq!(warn["fields"]["inflight_remaining"], 1, "{warn}");
    assert_eq!(
        fx.client.fence_guard_cleanups_inflight(),
        1,
        "the pools closed with the release still in flight"
    );
    live_edit::storage::test_support::release_all(stalls).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. Probe (SR-5.4)
// ═════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn probe_of_an_absent_fence_reports_absent() {
    let fx = Fx::new().await;

    assert_eq!(fx.probe().await, FenceProbe::Absent);
    assert!(
        !fx.exists(&fx.fence_key()).await,
        "the probe writes nothing"
    );
}

/// A live fence reports the latest lease's remaining time on the store's clock
/// and the stored size and `bpi`, and the probe writes nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn probe_of_a_live_fence_reports_the_remaining_lease_and_the_stored_size() {
    let fx = Fx::new().await;
    let _lease = fx.grant(SIZE).await;
    let (_, deadline) = only_reader(&fx.fields(&fx.fence_key()).await);
    fx.mark_expiry(&fx.fence_key()).await;
    let image = fx.image(&fx.fence_key()).await;

    let (t0, probe, t1) = bracketed(&fx, fx.probe()).await;

    let FenceProbe::Present {
        live_retry_after: Some(retry_after),
        stored_size,
        stored_bpi,
        ..
    } = probe
    else {
        panic!("a live fence must probe Present and live; got {probe:?}");
    };
    assert_between(
        retry_after.as_millis() as i64,
        deadline - t1,
        deadline - t0,
        "retry_after = latest live deadline − now",
    );
    assert_eq!(stored_size, Some(SIZE));
    assert_eq!(stored_bpi, SIZE.bytes_per_item());
    assert_eq!(
        fx.image(&fx.fence_key()).await,
        image,
        "the probe writes nothing"
    );
}

/// After the epoch the key lingers: present, not live, with the measured size.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn probe_after_the_epoch_reports_present_and_not_live() {
    let fx = Fx::new().await;
    let lease = fx.grant(SIZE).await;
    assert_eq!(
        lease.release(Some(OTHER_SIZE)).await.unwrap(),
        ReleaseVerdict::Verified
    );

    let probe = fx.probe().await;
    assert!(
        matches!(
            probe,
            FenceProbe::Present {
                live_retry_after: None,
                stored_size: Some(OTHER_SIZE),
                stored_bpi,
                ..
            } if stored_bpi == OTHER_SIZE.bytes_per_item()
        ),
        "got {probe:?}"
    );
}

/// A key with a live reader but no stored size reports neither size nor `bpi`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn probe_of_a_fence_without_a_stored_size_reports_none() {
    let fx = Fx::new().await;
    let s = fx.now().await;
    fx.hset(&fx.fence_key(), &[("r:staged", ms(s + FAR_MS))])
        .await;

    let probe = fx.probe().await;

    assert!(
        matches!(
            probe,
            FenceProbe::Present {
                live_retry_after: Some(_),
                stored_size: None,
                stored_bpi: None,
                ..
            }
        ),
        "got {probe:?}"
    );
}

// ── The epoch marker (design revision 5: the conflict classifier's input) ──
//
// The probe's last three reply elements — `epoch_start`, `gap_until` and the
// number of `r:*` fields — are what the slow path compares between an
// attempt's own probe and its conflict re-probe: every fence write after a
// probe that found no live lease must move one of them (or the key's
// existence). Each test reads the marker against the key's own fields after
// one such write.

/// The marker of a present probe.
fn marker_of(probe: &FenceProbe) -> FenceMarker {
    match probe {
        FenceProbe::Present { marker, .. } => *marker,
        other => panic!("the fence key must probe Present; got {other:?}"),
    }
}

/// The marker the fence key's own fields spell: `epoch_start` and
/// `gap_until` as stored (absent → `None`), and the count of `r:*` fields.
fn marker_from_fields(fields: &BTreeMap<String, String>) -> FenceMarker {
    let optional = |name: &str| {
        fields.get(name).map(|v| {
            v.parse::<u64>()
                .unwrap_or_else(|e| panic!("{name} must be an integer: {e}; fields: {fields:?}"))
        })
    };
    FenceMarker {
        epoch_start: optional("epoch_start"),
        gap_until: optional("gap_until"),
        readers: deadlines(fields, "r:").len() as u64,
    }
}

/// The probe script's raw reply for an absent fence key: nine elements, the
/// marker's three `-1`, `-1`, `0` among them — not the earlier script's six.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn probe_reply_of_an_absent_key_is_nine_elements_with_an_empty_marker() {
    let fx = Fx::new().await;
    let script = Script::from_lua(include_str!(
        "../../../src/storage/scripts/snapshot_fence_probe.lua"
    ));
    let reply = script
        .evalsha_with_reload::<Value, _, _>(
            fx.tv.client().pool.next(),
            vec![fx.fence_key(), fx.intent_key()],
            Vec::<String>::new(),
        )
        .await
        .expect("the probe script must answer");
    let rendered: Vec<String> = reply
        .into_array()
        .into_iter()
        .map(|v| match v {
            Value::Integer(n) => n.to_string(),
            other => format!(
                "{:?}",
                other.as_string().unwrap_or_else(|| format!("{other:?}"))
            ),
        })
        .collect();
    assert_eq!(
        rendered,
        ["0", "0", "0", "-1", "-1", "\"\"", "-1", "-1", "0"],
        "{{exists, live, retry_after_ms, size_e, size_b, bpi, epoch_start, gap_until, readers}}"
    );
}

/// A grant starts an epoch: the marker carries its `epoch_start` and one
/// reader; a join adds a reader under the same `epoch_start`; a release while
/// the joiner is live drops a reader and keeps the epoch; the last release
/// ends it — `epoch_start` cleared, a `gap_until`, no reader. Each step moves
/// the marker, and each marker is the key's own fields.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn probe_marker_moves_with_every_grant_join_and_release() {
    let fx = Fx::new().await;
    let mut seen: Vec<(&str, FenceProbe)> = vec![("no key", fx.probe().await)];
    assert_eq!(seen[0].1, FenceProbe::Absent);

    let first = fx.grant(SIZE).await;
    let after_grant = fx.probe().await;
    let fields = fx.fields(&fx.fence_key()).await;
    assert_eq!(marker_of(&after_grant), marker_from_fields(&fields));
    assert_eq!(
        marker_of(&after_grant),
        FenceMarker {
            epoch_start: Some(int(&fields, "epoch_start") as u64),
            gap_until: None,
            readers: 1,
        },
        "a fresh epoch: its start, no gap, one reader"
    );
    seen.push(("grant", after_grant));

    let joiner = fx.grant(SIZE).await;
    let after_join = fx.probe().await;
    assert_eq!(
        marker_of(&after_join),
        FenceMarker {
            readers: 2,
            ..marker_of(&seen[1].1)
        },
        "a join: the same epoch_start, two readers"
    );
    seen.push(("join", after_join));

    assert_eq!(first.release(None).await.unwrap(), ReleaseVerdict::Verified);
    let after_first_release = fx.probe().await;
    assert_eq!(
        marker_of(&after_first_release),
        FenceMarker {
            readers: 1,
            ..marker_of(&seen[1].1)
        },
        "a release while the joiner is live: the epoch goes on, one reader fewer"
    );
    seen.push(("release under a live joiner", after_first_release));

    assert_eq!(
        joiner.release(None).await.unwrap(),
        ReleaseVerdict::Verified
    );
    let after_epoch = fx.probe().await;
    let fields = fx.fields(&fx.fence_key()).await;
    assert_eq!(marker_of(&after_epoch), marker_from_fields(&fields));
    assert_eq!(
        marker_of(&after_epoch),
        FenceMarker {
            epoch_start: None,
            gap_until: Some(int(&fields, "gap_until") as u64),
            readers: 0,
        },
        "the last release ends the epoch: no epoch_start, a gap, no reader"
    );
    seen.push(("the epoch's last release", after_epoch));

    for pair in seen.windows(2) {
        let [(before, earlier), (after, later)] = pair else {
            unreachable!("windows(2)")
        };
        assert!(
            later.changed_since(earlier),
            "{after} after {before} must change the marker: {earlier:?} then {later:?}"
        );
    }
}

/// The late release — the epoch already closed by another reader, this
/// reader's lease expired — removes only its own field (section 3): of the
/// marker, only `readers` moves, and that alone is a change.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn probe_marker_after_a_late_release_moves_only_readers() {
    let fx = Fx::new().await;
    let a = fx.grant(SIZE).await;
    let before = fx.fields(&fx.fence_key()).await;
    let b = fx.grant(SIZE).await;
    let (b_field, _) = new_reader(&before, &fx.fields(&fx.fence_key()).await);
    let s = fx.now().await;
    fx.hset(&fx.fence_key(), &[(b_field.as_str(), ms(s - 5))])
        .await;
    assert_eq!(a.release(None).await.unwrap(), ReleaseVerdict::Verified);
    let closed = fx.probe().await;
    assert_eq!(
        marker_of(&closed).readers,
        1,
        "precondition: B's dead field outlives the epoch A closed"
    );

    assert_eq!(b.release(None).await.unwrap(), ReleaseVerdict::Broken);
    let after_late = fx.probe().await;

    assert_eq!(
        marker_of(&after_late),
        FenceMarker {
            readers: 0,
            ..marker_of(&closed)
        },
        "the late release moves readers alone"
    );
    assert_eq!(
        marker_of(&after_late),
        marker_from_fields(&fx.fields(&fx.fence_key()).await)
    );
    assert!(after_late.changed_since(&closed));
}

/// A lease that lapses unreleased writes nothing: the probe stops reporting
/// it live, and the marker — `epoch_start` kept, the dead reader still
/// counted — is the one the live probe saw. (The classifier's live-lease
/// rule, not the marker, covers a lease live at the re-probe.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn probe_marker_after_a_lease_lapses_keeps_the_epoch_start() {
    let fx = Fx::new().await;
    let _lease = fx.grant(SIZE).await;
    let live = fx.probe().await;
    assert!(
        matches!(
            live,
            FenceProbe::Present {
                live_retry_after: Some(_),
                ..
            }
        ),
        "precondition: the lease is live; got {live:?}"
    );
    let (field, _) = only_reader(&fx.fields(&fx.fence_key()).await);
    let s = fx.now().await;
    fx.hset(&fx.fence_key(), &[(field.as_str(), ms(s - 1))])
        .await;

    let lapsed = fx.probe().await;

    assert!(
        matches!(
            lapsed,
            FenceProbe::Present {
                live_retry_after: None,
                ..
            }
        ),
        "a lapsed lease is not live; got {lapsed:?}"
    );
    let marker = marker_of(&lapsed);
    assert!(
        marker.epoch_start.is_some() && marker.readers == 1,
        "the lapse writes nothing: epoch_start kept, the dead reader counted; got {marker:?}"
    );
    assert_eq!(marker, marker_of(&live));
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. Writer intents (SR-5.4)
// ═════════════════════════════════════════════════════════════════════════════

/// While a reader holds the fence an intent is refused `FENCED` with the
/// latest lease's remaining time, and nothing is written to either key. A
/// reader whose lease has expired refuses nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_intent_while_a_reader_is_live_is_fenced_and_writes_nothing() {
    let fx = Fx::new().await;
    let _lease = fx.grant(SIZE).await;
    let (field, deadline) = only_reader(&fx.fields(&fx.fence_key()).await);
    fx.mark_expiry(&fx.fence_key()).await;
    let image = fx.image(&fx.fence_key()).await;

    let (t0, verdict, t1) = bracketed(&fx, fx.register(SIZE)).await;

    let IntentVerdict::Fenced { retry_after } = verdict else {
        panic!("a live reader must refuse the intent; got {verdict:?}");
    };
    assert_between(
        retry_after.as_millis() as i64,
        deadline - t1,
        deadline - t0,
        "retry_after = latest live deadline − now",
    );
    assert!(
        !fx.exists(&fx.intent_key()).await,
        "a refused intent must not be written"
    );
    assert_eq!(
        fx.image(&fx.fence_key()).await,
        image,
        "intent registration never writes the fence key (writers WATCH it)"
    );

    let s = fx.now().await;
    fx.hset(&fx.fence_key(), &[(field.as_str(), ms(s - 1))])
        .await;
    registered(fx.register(SIZE).await);
}

/// A registered intent lasts `intent_ttl(L)` of the fence key's stored size
/// (not the caller's estimate), prunes lapsed intents, keeps live ones, and
/// expires the key at its latest deadline.
///
/// The stored size's `intent_ttl` is above the queue floor and the estimate's
/// sits on it, so a script that used the estimate would register a shorter
/// intent and fail the first assertion.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_intent_lasts_intent_ttl_of_the_stored_size_and_prunes_lapsed_intents() {
    let fx = Fx::new().await;
    let lease = fx.grant(SIZE).await;
    assert_eq!(
        lease.release(Some(ABOVE_FLOOR_SIZE)).await.unwrap(),
        ReleaseVerdict::Verified
    );
    let ttl = whole_ms(fx.timing(ABOVE_FLOOR_SIZE).intent_ttl_ms);
    let estimate_ttl = whole_ms(fx.timing(OTHER_SIZE).intent_ttl_ms);
    assert!(
        estimate_ttl == QUEUE_FLOOR_MS as i64 && ttl > estimate_ttl,
        "precondition: the estimate's intent_ttl ({estimate_ttl} ms) must sit on the queue floor \
         and the stored size's ({ttl} ms) above it"
    );
    let s = fx.now().await;
    let other_deadline = s + FAR_MS;
    fx.hset(
        &fx.intent_key(),
        &[("w:lapsed", ms(s - 1)), ("w:other", ms(other_deadline))],
    )
    .await;
    let before = fx.fields(&fx.intent_key()).await;

    let (t0, verdict, t1) = bracketed(&fx, fx.register(OTHER_SIZE)).await;

    let intent = registered(verdict);
    assert_eq!(
        intent.ttl(),
        Duration::from_millis(ttl as u64),
        "intent_ttl(L) of the stored size, not the estimate"
    );
    assert_eq!(
        intent.renew_interval(),
        ms_to_duration(ttl as f64 / 4.0),
        "renew every intent_ttl(L)/4"
    );
    let after = fx.fields(&fx.intent_key()).await;
    assert!(
        !after.contains_key("w:lapsed"),
        "lapsed intents are pruned: {after:?}"
    );
    assert_eq!(
        after.get("w:other"),
        Some(&ms(other_deadline)),
        "live intents stay"
    );
    let own: Vec<i64> = deadlines(&after, "w:")
        .into_iter()
        .filter(|(k, _)| !before.contains_key(k))
        .map(|(_, d)| d)
        .collect();
    assert_eq!(own.len(), 1, "one new intent: {after:?}");
    assert_between(own[0], t0 + ttl, t1 + ttl, "deadline = now + intent_ttl(L)");
    assert_expiry(
        fx.expires_at(&fx.intent_key()).await,
        other_deadline,
        "the intent key expires at its latest deadline",
    );
}

/// A small layer's intent lasts the queue floor `4·(rtt99 + Q)` — 2,163 ms —
/// not its shorter `1.5·S`, so it outlives the data-pool queue its renewal
/// waits behind.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_intent_on_a_small_layer_lasts_the_queue_floor() {
    let fx = Fx::new().await;
    let small_s = fx.timing(SMALL_SIZE).structural_attempt_ms;
    assert!(
        whole_ms(1.5 * small_s) < QUEUE_FLOOR_MS as i64,
        "precondition: the 8,000-item layer's 1.5·S ({} ms) must be under the queue floor \
         ({QUEUE_FLOOR_MS} ms), or a floored intent_ttl is indistinguishable from 1.5·S",
        1.5 * small_s
    );

    let intent = registered(fx.register(SMALL_SIZE).await);

    assert_eq!(
        intent.ttl(),
        Duration::from_millis(QUEUE_FLOOR_MS),
        "an 8,000-item layer's intent lasts the queue floor, not its 1.5·S"
    );
    let (_, deadline) = deadlines(&fx.fields(&fx.intent_key()).await, "w:")
        .into_iter()
        .next()
        .expect("the intent is registered");
    assert_expiry(
        fx.expires_at(&fx.intent_key()).await,
        deadline,
        "the intent key expires with the intent",
    );
}

/// Renewing a live intent moves its deadline to `now + intent_ttl` and raises
/// the key's expiry to it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn renew_intent_extends_a_live_intent_and_raises_the_key_expiry() {
    let fx = Fx::new().await;
    let intent = registered(fx.register(SIZE).await);
    let ttl = intent.ttl().as_millis() as i64;
    let (field, _) = deadlines(&fx.fields(&fx.intent_key()).await, "w:")
        .into_iter()
        .next()
        .expect("the intent is registered");
    // Stage a still-live deadline and a key expiry both well under
    // `intent_ttl` ahead (the script only ever raises an expiry, so a staged
    // expiry at or above the renewed deadline would hide the raise).
    let staged_deadline = ttl / 2;
    let staged_expiry = ttl / 2 + 200;
    assert!(
        staged_expiry + 500 < ttl,
        "precondition: the staged expiry ({staged_expiry} ms) must sit well under intent_ttl \
         ({ttl} ms)"
    );
    let s = fx.now().await;
    fx.hset(
        &fx.intent_key(),
        &[(field.as_str(), ms(s + staged_deadline))],
    )
    .await;
    fx.pexpire(&fx.intent_key(), staged_expiry).await;

    let (t0, verdict, t1) = bracketed(&fx, intent.renew()).await;

    assert_eq!(verdict.unwrap(), IntentRenewVerdict::Renewed);
    let renewed = int(&fx.fields(&fx.intent_key()).await, &field);
    assert_between(renewed, t0 + ttl, t1 + ttl, "deadline = now + intent_ttl");
    assert_expiry(
        fx.expires_at(&fx.intent_key()).await,
        renewed,
        "the key expires with the renewed intent",
    );
}

/// An intent that has lapsed is not revived (`LOST`, nothing written); one
/// whose key was removed answers `LOST` and creates nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn renew_intent_after_it_lapsed_or_was_removed_answers_lost_and_writes_nothing() {
    let fx = Fx::new().await;
    let intent = registered(fx.register(SIZE).await);
    let (field, _) = deadlines(&fx.fields(&fx.intent_key()).await, "w:")
        .into_iter()
        .next()
        .expect("the intent is registered");
    let s = fx.now().await;
    fx.hset(&fx.intent_key(), &[(field.as_str(), ms(s - 1))])
        .await;
    fx.mark_expiry(&fx.intent_key()).await;
    let image = fx.image(&fx.intent_key()).await;

    assert_eq!(intent.renew().await.unwrap(), IntentRenewVerdict::Lost);
    assert_eq!(
        fx.image(&fx.intent_key()).await,
        image,
        "a lapsed intent must not be revived: readers may already have started an epoch"
    );

    fx.unlink(vec![fx.intent_key()]).await;
    assert_eq!(intent.renew().await.unwrap(), IntentRenewVerdict::Lost);
    assert!(!fx.exists(&fx.intent_key()).await, "nothing recreated");
}

/// `clear` removes the attempt's own intent and nobody else's.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn clear_intent_removes_only_its_own_field() {
    let fx = Fx::new().await;
    let s = fx.now().await;
    fx.hset(&fx.intent_key(), &[("w:other", ms(s + FAR_MS))])
        .await;
    let intent = registered(fx.register(SIZE).await);

    intent.clear().await.expect("clear must succeed");

    assert_eq!(
        deadlines(&fx.fields(&fx.intent_key()).await, "w:")
            .into_keys()
            .collect::<Vec<_>>(),
        vec!["w:other".to_owned()]
    );
}

/// An intent dropped without `clear` is removed by the guard's spawned `HDEL`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_an_uncleared_intent_removes_it_in_the_background() {
    let fx = Fx::new().await;
    let intent = registered(fx.register(SIZE).await);
    assert!(
        fx.exists(&fx.intent_key()).await,
        "precondition: registered"
    );

    drop(intent);

    let fx = &fx;
    eventually("the dropped intent's HDEL", move || async move {
        !fx.exists(&fx.intent_key()).await
    })
    .await;
}

/// `ValkeyClient::quit` waits for a dropped intent's background `HDEL` before
/// it closes the pools (post-completion review F12): a stranded intent would
/// hold other replicas' fenced readers of the layer off until it lapsed. As
/// for the lease above, a current-thread runtime keeps the spawned `HDEL`
/// in flight until `quit` starts.
#[tokio::test(flavor = "current_thread")]
async fn quit_waits_for_a_dropped_intents_removal_before_closing_the_pools() {
    let fx = Fx::new().await.using(own_client().await);
    let intent = registered(fx.register(SIZE).await);
    assert!(
        fx.exists(&fx.intent_key()).await,
        "precondition: registered"
    );

    drop(intent);
    assert_eq!(
        fx.client.fence_guard_cleanups_inflight(),
        1,
        "the dropped intent's HDEL is spawned tracked on the client, not yet run"
    );
    fx.client.quit().await;

    assert_eq!(
        fx.client.fence_guard_cleanups_inflight(),
        0,
        "quit waited for the HDEL"
    );
    assert!(
        !fx.exists(&fx.intent_key()).await,
        "the HDEL reached the store before the pools closed"
    );
}

/// A WATCH-path attempt's intent is cleared by its renewal task, not by the
/// guard's own drop: dropping the `IntentRenewal` (the attempt's future
/// dropped mid-commit, as graceful shutdown drops a handler) ends the task,
/// which then clears the intent. That task is tracked on the client from its
/// start, so `ValkeyClient::quit` waits for its `HDEL` too. The in-flight
/// counts pin that tracking: the task is counted from `start`, and on a
/// current-thread runtime it is still counted when the handle is dropped,
/// since it cannot run between the drop and `quit`'s first await. The store
/// check alone would not pin the wait: `quit` yields before its drain, so
/// the task may clear the intent before the drain begins.
#[tokio::test(flavor = "current_thread")]
async fn quit_waits_for_a_dropped_intent_renewals_removal_before_closing_the_pools() {
    let fx = Fx::new().await.using(own_client().await);
    let renewal = IntentRenewal::start(registered(fx.register(SIZE).await));
    assert!(
        fx.exists(&fx.intent_key()).await,
        "precondition: registered"
    );
    assert_eq!(
        fx.client.fence_guard_cleanups_inflight(),
        1,
        "the renewal task, which clears the intent when it ends, is tracked on the client"
    );

    drop(renewal);
    assert_eq!(
        fx.client.fence_guard_cleanups_inflight(),
        1,
        "the dropped renewal's task, which now clears the intent, is still tracked"
    );
    fx.client.quit().await;

    assert_eq!(
        fx.client.fence_guard_cleanups_inflight(),
        0,
        "quit waited for the renewal task's clear"
    );
    assert!(
        !fx.exists(&fx.intent_key()).await,
        "the HDEL reached the store before the pools closed"
    );
}

/// A crashed writer runs no clean-up, so its intent stays until its own
/// deadline: `intent_ttl(L)` after it registered, which is also when its key
/// expires. Readers wait on it until then, and not after.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_crashed_writers_intent_holds_readers_off_for_at_most_intent_ttl() {
    let fx = Fx::new().await;
    let lease = fx.grant(SIZE).await;
    assert_eq!(
        lease.release(Some(LONG_INTENT_SIZE)).await.unwrap(),
        ReleaseVerdict::Verified
    );
    let s = fx.now().await;
    fx.hset(&fx.fence_key(), &[("gap_until", ms(s - 1))]).await;

    let (t0, verdict, t1) = bracketed(&fx, fx.register(SIZE)).await;
    let intent = registered(verdict);
    let ttl = intent.ttl().as_millis() as i64;
    assert_eq!(
        ttl,
        whole_ms(fx.timing(LONG_INTENT_SIZE).intent_ttl_ms),
        "intent_ttl(L) of the stored size"
    );
    // A crash runs no destructor: no spawned HDEL, no renewal.
    std::mem::forget(intent);

    let (field, deadline) = deadlines(&fx.fields(&fx.intent_key()).await, "w:")
        .into_iter()
        .next()
        .expect("the intent is registered");
    assert_between(
        deadline,
        t0 + ttl,
        t1 + ttl,
        "the intent lasts intent_ttl(L)",
    );
    assert_expiry(
        fx.expires_at(&fx.intent_key()).await,
        deadline,
        "the crashed writer's key goes at the intent's deadline",
    );
    let (_, reason, _) = retry_after(fx.acquire(SIZE).await);
    assert_eq!(
        reason,
        RetryReason::Intent,
        "until its deadline it holds readers off"
    );

    // Its deadline passes.
    let lapsed = fx.now().await - 1;
    fx.hset(&fx.intent_key(), &[(field.as_str(), ms(lapsed))])
        .await;
    assert_eq!(
        fx.grant(SIZE).await.kind(),
        GrantKind::StartedEpoch,
        "after its deadline it holds nobody off"
    );
}

/// A removed object takes no intent (decider ruling of 2026-10-03, Spec SDD
/// SR-5.4): with no header the registration answers `NOT_FOUND` and writes
/// nothing — no intent key is created, an existing one (a lingering live
/// intent of another writer) is left exactly as it was, and the fence key is
/// untouched. The header check comes first, so a live reader on the fence does
/// not turn it into `FENCED`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_intent_without_a_header_answers_not_found_and_writes_nothing() {
    let fx = Fx::without_header().await;

    assert!(
        matches!(fx.register(SIZE).await, IntentVerdict::NotFound),
        "no header: NOT_FOUND"
    );
    assert!(
        !fx.exists(&fx.intent_key()).await,
        "a NOT_FOUND registration must not create the intent key"
    );

    // A wipe that raced a writer: another writer's intent and a live reader
    // are still there, the header is gone.
    let s = fx.now().await;
    fx.hset(&fx.intent_key(), &[("w:other", ms(s + FAR_MS))])
        .await;
    fx.hset(&fx.fence_key(), &[("r:staged", ms(s + FAR_MS))])
        .await;
    for key in [fx.intent_key(), fx.fence_key()] {
        fx.mark_expiry(&key).await;
    }
    let images = (
        fx.image(&fx.intent_key()).await,
        fx.image(&fx.fence_key()).await,
    );

    let verdict = fx.register(SIZE).await;

    assert!(
        matches!(verdict, IntentVerdict::NotFound),
        "the header check runs before the fence check: NOT_FOUND, not FENCED; got {verdict:?}"
    );
    assert_eq!(
        (
            fx.image(&fx.intent_key()).await,
            fx.image(&fx.fence_key()).await
        ),
        images,
        "NOT_FOUND writes nothing to either key"
    );
}

/// "Absent" means no header key at all: a header without a numeric `seq` is a
/// present (corrupt) object, which the commit path refuses on its own terms,
/// so the registration still registers.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_intent_on_a_header_without_a_numeric_seq_still_registers() {
    let fx = Fx::without_header().await;
    fx.hset(&fx.header_key(), &[("seq", "not-a-number".to_owned())])
        .await;

    let intent = registered(fx.register(SIZE).await);

    assert_eq!(
        deadlines(&fx.fields(&fx.intent_key()).await, "w:").len(),
        1,
        "the intent is written"
    );
    intent.clear().await.expect("clear must succeed");
}

// ═════════════════════════════════════════════════════════════════════════════
// 6. Lua and Rust compute the same limits (SR-5.7)
// ═════════════════════════════════════════════════════════════════════════════

const SCRIPTS_WITH_TIMING: [(&str, &str); 4] = [
    (
        "snapshot_fence_acquire.lua",
        include_str!("../../../src/storage/scripts/snapshot_fence_acquire.lua"),
    ),
    (
        "snapshot_fence_renew.lua",
        include_str!("../../../src/storage/scripts/snapshot_fence_renew.lua"),
    ),
    (
        "snapshot_fence_release.lua",
        include_str!("../../../src/storage/scripts/snapshot_fence_release.lua"),
    ),
    (
        "snapshot_intent_register.lua",
        include_str!("../../../src/storage/scripts/snapshot_intent_register.lua"),
    ),
];

/// `text` from the first line starting with `start` through the first line
/// after it that is exactly `end`.
fn block<'a>(script: &str, text: &'a str, start: &str, end: &str) -> &'a str {
    let from = text
        .find(start)
        .unwrap_or_else(|| panic!("{script}: no `{start}`"));
    let rest = &text[from..];
    let len = rest
        .lines()
        .scan(0, |offset, line| {
            *offset += line.len() + 1;
            Some((*offset, line))
        })
        .find(|(_, line)| *line == end)
        .map(|(offset, _)| offset)
        .unwrap_or_else(|| panic!("{script}: `{start}` has no closing `{end}`"));
    &rest[..len]
}

/// The four scripts that compute limits carry one `timing` function and one
/// coefficient order, character for character — so the behavioural parity
/// below, checked through each script's own outputs, cannot be undone by
/// editing one copy.
fn timing_block<'a>(script: &str, text: &'a str) -> &'a str {
    block(script, text, "local function timing(e, b)", "end")
}

fn coefficient_order<'a>(script: &str, text: &'a str) -> &'a str {
    block(script, text, "local COEFFICIENTS = ", "local c = {}")
}

#[test]
fn lua_timing_copies_and_coefficient_orders_are_identical() {
    let (first_name, first) = SCRIPTS_WITH_TIMING[0];
    for (name, text) in &SCRIPTS_WITH_TIMING[1..] {
        assert_eq!(
            timing_block(name, text),
            timing_block(first_name, first),
            "{name}'s timing function must be {first_name}'s, character for character"
        );
        assert_eq!(
            coefficient_order(name, text),
            coefficient_order(first_name, first),
            "{name}'s coefficient order must be {first_name}'s"
        );
    }
    assert!(
        timing_block(first_name, first).contains("math.max(4 * rtt_q, 1.5 *"),
        "the timing function must carry the floored intent_ttl"
    );
}

/// The edge of the structural term's clamp: `S_MIN_ITEMS` items, read from
/// the constant the cost model and the fence scripts share, so the size
/// follows a re-fit of `S`; its bytes on the same Kitchen_set ray as the
/// other parity sizes ([`SIZE`]'s bytes per item).
fn s_min_edge_size() -> LayerSize {
    let items = fence::cost::S_MIN_ITEMS;
    LayerSize {
        items: items as u64,
        bytes: (items * SIZE.bytes as f64 / SIZE.items as f64).round() as u64,
    }
}

/// The sizes the parity check runs at, besides the clamp's edge
/// ([`s_min_edge_size`]): empty; below `S`'s range (the clamp, and the
/// `intent_ttl` floor); the shipped ceiling; 4× (the checkpoint reference)
/// and 10× the ceiling; 200k entries (`epoch` follows `hold`);
/// and 30M items, where `hold/4` passes the lease floor so `lease` follows
/// `hold` too. Which of them take `intent_ttl` = `1.5·S` above the queue floor
/// follows the wired `S`; the test checks that at least one does, so both of
/// `intent_ttl`'s arms are compared.
const PARITY_SIZES: [LayerSize; 7] = [
    LayerSize { items: 0, bytes: 0 },
    SMALL_SIZE,
    LayerSize {
        items: 37_554,
        bytes: 449_938,
    },
    SIZE,
    LayerSize {
        items: 375_540,
        bytes: 4_499_380,
    },
    ABOVE_FLOOR_SIZE,
    LayerSize {
        items: 30_000_000,
        bytes: 359_432_884,
    },
];

/// Every limit each script stores or returns for `size` on `fx`'s client
/// equals `FenceCoefficients::timing`'s, in whole ms: acquire's `lease` and
/// `key_linger`, renew's `lease` and `key_linger`, release's `min_gap` and
/// `key_linger`, and intent registration's `intent_ttl`.
async fn assert_scripts_compute_timing(fx: &Fx, size: LayerSize) {
    let t = fx.timing(size);
    let lease_ms = whole_ms(t.lease_ms);
    let linger_ms = whole_ms(t.key_linger_ms);
    let what = |step: &str| format!("{size:?}: {step}");

    // Acquire.
    let lease = fx.grant(size).await;
    let fields = fx.fields(&fx.fence_key()).await;
    let (field, deadline) = only_reader(&fields);
    assert_eq!(
        deadline - int(&fields, "epoch_start"),
        lease_ms,
        "{}",
        what("acquire's lease(L)")
    );
    assert_size_fields(&fields, size, &what("acquire's stored size"));
    assert_expiry(
        fx.expires_at(&fx.fence_key()).await,
        deadline + linger_ms,
        &what("acquire's key_linger(L)"),
    );

    // Renew.
    let s = fx.now().await;
    fx.hset(&fx.fence_key(), &[(field.as_str(), ms(s + 2_000))])
        .await;
    fx.pexpire(&fx.fence_key(), 3_000).await;
    let (t0, verdict, t1) = bracketed(fx, lease.renew()).await;
    assert_eq!(verdict.unwrap(), RenewVerdict::Renewed, "{}", what("renew"));
    let renewed = int(&fx.fields(&fx.fence_key()).await, &field);
    assert_between(
        renewed,
        t0 + lease_ms,
        t1 + lease_ms,
        &what("renew's lease(L)"),
    );
    assert_expiry(
        fx.expires_at(&fx.fence_key()).await,
        renewed + linger_ms,
        &what("renew's key_linger(L)"),
    );

    // Release. An epoch start ahead of the store's clock gives a negative
    // measured length, so the gap is `min_gap` alone.
    fx.hset(&fx.fence_key(), &[("epoch_start", ms(s + FAR_MS))])
        .await;
    let (t0, verdict, t1) = bracketed(fx, lease.release(Some(size))).await;
    assert_eq!(
        verdict.unwrap(),
        ReleaseVerdict::Verified,
        "{}",
        what("release")
    );
    let gap_until = int(&fx.fields(&fx.fence_key()).await, "gap_until");
    let min_gap = whole_ms(t.min_gap_ms);
    assert_between(
        gap_until,
        t0 + min_gap,
        t1 + min_gap,
        &what("release's min_gap"),
    );
    assert_expiry(
        fx.expires_at(&fx.fence_key()).await,
        gap_until + linger_ms,
        &what("release's key_linger(L)"),
    );

    // Intent registration, under the size the release stored.
    let intent = registered(fx.register(LayerSize { items: 1, bytes: 1 }).await);
    assert_eq!(
        intent.ttl(),
        Duration::from_millis(whole_ms(t.intent_ttl_ms) as u64),
        "{}",
        what("register's intent_ttl(L)")
    );
    intent.clear().await.expect("clear must succeed");
}

/// The Lua copies of the cost model compute what `FenceCoefficients::timing`
/// computes, at every parity size, on the shipped ceiling.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fence_scripts_compute_every_limit_exactly_as_cost_timing_does() {
    let edge = s_min_edge_size();
    assert_eq!(
        edge.items as f64,
        fence::cost::S_MIN_ITEMS,
        "precondition: the clamp-edge parity size must sit at S_MIN_ITEMS, a whole item count"
    );
    let sizes: Vec<LayerSize> = std::iter::once(edge).chain(PARITY_SIZES).collect();
    let fx = Fx::new().await;
    let floor = fx.timing(LayerSize { items: 0, bytes: 0 }).intent_ttl_ms;
    assert!(
        sizes
            .iter()
            .any(|size| fx.timing(*size).intent_ttl_ms > floor),
        "precondition: at least one parity size must take intent_ttl = 1.5·S above the queue \
         floor ({floor} ms), or the scripts' 1.5·S arm is never compared"
    );
    for size in sizes {
        assert_scripts_compute_timing(&Fx::new().await, size).await;
    }
}

/// A lowered atomic-read ceiling reaches the scripts through the coefficient
/// block: `Q`, and with it `lease`, `min_gap` and the floored `intent_ttl`,
/// come out shorter, and the scripts still agree with `timing`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fence_scripts_follow_a_lowered_atomic_read_ceiling() {
    let shipped = Fx::new().await;
    let lowered_client = client_with_limits(10_000, u64::MAX / 2).await;
    let fx = shipped.using(lowered_client);
    let lowered = fx.timing(SMALL_SIZE);
    let at_shipped = fence::coefficients(fx.tv.client()).timing(SMALL_SIZE);
    assert!(
        whole_ms(lowered.lease_ms) < whole_ms(at_shipped.lease_ms),
        "precondition: the lowered ceiling must change the whole-ms lease ({} vs {})",
        lowered.lease_ms,
        at_shipped.lease_ms
    );

    assert_scripts_compute_timing(&fx, SMALL_SIZE).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 7. Malformed input
// ═════════════════════════════════════════════════════════════════════════════

/// Every script rejects a malformed argument with an error reply before it
/// reads or writes anything: neither key changes. (The wrappers never send one;
/// the scripts guard their own contract.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fence_scripts_reject_malformed_arguments_and_write_nothing() {
    let fx = Fx::new().await;
    let s = fx.now().await;
    fx.hset(
        &fx.fence_key(),
        &[("r:staged", ms(s - 1)), ("epoch_start", ms(s - 5_000))],
    )
    .await;
    fx.hset(&fx.intent_key(), &[("w:staged", ms(s + FAR_MS))])
        .await;
    for key in [fx.fence_key(), fx.intent_key()] {
        fx.mark_expiry(&key).await;
    }
    let images = (
        fx.image(&fx.fence_key()).await,
        fx.image(&fx.intent_key()).await,
    );
    let coefficients: Vec<String> = fence::coefficients(&fx.client).script_argv().to_vec();
    let with_coefficients = |head: &[&str], tail: &[String]| -> Vec<String> {
        head.iter()
            .map(|s| (*s).to_owned())
            .chain(tail.iter().cloned())
            .collect()
    };
    let mut nan_f0 = coefficients.clone();
    nan_f0[0] = "nan".to_owned();
    let mut inf_rtt = coefficients.clone();
    inf_rtt[6] = "inf".to_owned();
    let short = coefficients[..8].to_vec();

    let acquire = SCRIPTS_WITH_TIMING[0].1;
    let renew = SCRIPTS_WITH_TIMING[1].1;
    let release = SCRIPTS_WITH_TIMING[2].1;
    let register = SCRIPTS_WITH_TIMING[3].1;
    let intent_renew = include_str!("../../../src/storage/scripts/snapshot_intent_renew.lua");
    let acquire_keys = vec![fx.fence_key(), fx.intent_key(), fx.header_key()];
    let cases: Vec<(&str, Vec<String>, Vec<String>, &str)> = vec![
        (
            acquire,
            acquire_keys.clone(),
            with_coefficients(&["", "1", "1"], &coefficients),
            "snapshot_fence_acquire: empty token",
        ),
        (
            acquire,
            acquire_keys.clone(),
            with_coefficients(&["t", "-1", "1"], &coefficients),
            "snapshot_fence_acquire: est_e",
        ),
        (
            acquire,
            acquire_keys.clone(),
            with_coefficients(&["t", "1", "1.5"], &coefficients),
            "snapshot_fence_acquire: est_b",
        ),
        (
            acquire,
            acquire_keys.clone(),
            with_coefficients(&["t", "1", "1"], &nan_f0),
            "snapshot_fence_acquire: coefficient f0",
        ),
        (
            acquire,
            acquire_keys,
            with_coefficients(&["t", "1", "1"], &short),
            "snapshot_fence_acquire: coefficient s_min_items",
        ),
        (
            renew,
            vec![fx.fence_key()],
            with_coefficients(&["staged", "x", "1"], &coefficients),
            "snapshot_fence_renew: est_e",
        ),
        (
            release,
            vec![fx.fence_key(), fx.header_key()],
            with_coefficients(&["staged", "41a", "1", "1"], &coefficients),
            "snapshot_fence_release: cut",
        ),
        (
            release,
            vec![fx.fence_key(), fx.header_key()],
            with_coefficients(&["staged", "41", "1", "-2"], &coefficients),
            "snapshot_fence_release: measured_b",
        ),
        (
            register,
            vec![fx.intent_key(), fx.fence_key(), fx.header_key()],
            with_coefficients(&["t", "1", "1"], &inf_rtt),
            "snapshot_intent_register: coefficient rtt99",
        ),
        (
            intent_renew,
            vec![fx.intent_key()],
            vec!["staged".to_owned(), "0".to_owned()],
            "snapshot_intent_renew: ttl_ms",
        ),
        (
            intent_renew,
            vec![fx.intent_key()],
            vec![String::new(), "5".to_owned()],
            "snapshot_intent_renew: empty token",
        ),
    ];

    for (source, keys, argv, expected) in cases {
        let err = Script::from_lua(source)
            .evalsha_with_reload::<Value, _, _>(fx.tv.client().pool.next(), keys, argv)
            .await
            .expect_err(expected);
        assert!(
            err.details().contains(expected),
            "the error reply must name the argument ({expected:?}); got {err:?}"
        );
        assert_eq!(
            (
                fx.image(&fx.fence_key()).await,
                fx.image(&fx.intent_key()).await
            ),
            images,
            "{expected}: an argument error is raised before any write"
        );
    }
}

/// A fence key that is not a hash fails the acquire with a permanent error
/// (`Internal`) and is left as it was.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acquire_on_a_wrong_typed_fence_key_is_internal_and_leaves_it() {
    let fx = Fx::new().await;
    let _: () = fx
        .tv
        .client()
        .pool
        .set(fx.fence_key(), "not-a-hash", None, None, false)
        .await
        .expect("planting a string must succeed");

    let err = fence::acquire_fence(&fx.client, fx.keys(), fx.oid, SIZE)
        .await
        .expect_err("a wrong-typed fence key must fail the acquire");

    assert!(
        matches!(err, ServiceError::Internal(_)),
        "keyspace corruption is permanent, not retriable; got {err:?}"
    );
    let still: String = fx.tv.client().pool.get(fx.fence_key()).await.unwrap();
    assert_eq!(still, "not-a-hash");
}

/// The refusal both commit backends give a wrong-typed fence or intent key,
/// byte for byte: `update_layer_fast.lua` step 1d's sentence, which the probe
/// and the register script repeat (SDD §7.5 parity).
fn wrongtype_detail(key: &str) -> String {
    format!("WRONGTYPE_KEY: {key} holds a string, expected hash")
}

/// Plant a string on `key`.
async fn plant_string(fx: &Fx, key: &str) {
    let _: () = fx
        .tv
        .client()
        .pool
        .set(key, "not-a-hash", None, None, false)
        .await
        .expect("planting a string must succeed");
}

/// The probe type-checks both keys, fence first: a string on either answers
/// `WrongType` carrying the update script's exact refusal (not an error, so
/// the WATCH path can refuse its commit in step 1d's position), and the
/// probe leaves the key as it found it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn probe_of_a_wrong_typed_fence_or_intent_key_answers_wrong_type_fence_first() {
    let fx = Fx::new().await;

    for key in [fx.fence_key(), fx.intent_key()] {
        plant_string(&fx, &key).await;
        assert_eq!(
            fx.probe().await,
            FenceProbe::WrongType {
                detail: wrongtype_detail(&key)
            },
            "{key}"
        );
        let still: String = fx.tv.client().pool.get(&key).await.unwrap();
        assert_eq!(still, "not-a-hash", "{key}: the probe writes nothing");
        fx.unlink(vec![key]).await;
    }

    plant_string(&fx, &fx.fence_key()).await;
    plant_string(&fx, &fx.intent_key()).await;
    assert_eq!(
        fx.probe().await,
        FenceProbe::WrongType {
            detail: wrongtype_detail(&fx.fence_key())
        },
        "both wrong: the fence is named, as the fast path's step 1d names it"
    );
}

/// Registration type-checks both keys after the header, fence first: a
/// string on either answers `WrongType` with the same refusal and writes
/// nothing; with no header the answer is still `NOT_FOUND`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_intent_on_a_wrong_typed_key_answers_wrong_type_and_writes_nothing() {
    let fx = Fx::new().await;

    for key in [fx.fence_key(), fx.intent_key()] {
        plant_string(&fx, &key).await;
        let verdict = fx.register(SIZE).await;
        match verdict {
            IntentVerdict::WrongType { detail } => assert_eq!(detail, wrongtype_detail(&key)),
            other => panic!("{key}: a wrong-typed key must answer WrongType; got {other:?}"),
        }
        let still: String = fx.tv.client().pool.get(&key).await.unwrap();
        assert_eq!(still, "not-a-hash", "{key}: nothing written");
        fx.unlink(vec![key]).await;
    }
    assert!(
        !fx.exists(&fx.intent_key()).await && !fx.exists(&fx.fence_key()).await,
        "no refusal wrote either key"
    );

    plant_string(&fx, &fx.fence_key()).await;
    plant_string(&fx, &fx.intent_key()).await;
    match fx.register(SIZE).await {
        IntentVerdict::WrongType { detail } => assert_eq!(
            detail,
            wrongtype_detail(&fx.fence_key()),
            "both wrong: the fence is named first"
        ),
        other => panic!("both wrong: WrongType expected; got {other:?}"),
    }

    fx.unlink(vec![fx.header_key()]).await;
    assert!(
        matches!(fx.register(SIZE).await, IntentVerdict::NotFound),
        "the header check precedes the type check"
    );
}

// The commit paths' refusal of a wrong-typed fence or intent key — the same
// `WRONGTYPE_KEY` on the Lua fast path and the txn backend, field-only and
// structural — is pinned in `snapshot_fence_writer_test.rs`.

// ═════════════════════════════════════════════════════════════════════════════
// 8. The measurement event (`snapshot fence lease`, design revision 3)
// ═════════════════════════════════════════════════════════════════════════════
//
// One DEBUG event per lease release, on its own target, carrying the grant's
// object, kind and cut, the release's outcome and path, and the replica's
// wall-clock brackets around the acquire and release `EVALSHA`s. The
// validation tooling unions these events across replicas to judge the
// writers' share per epoch–gap cycle, so the event's presence, its fields and
// the brackets' meaning are the contract here (test-writing guide § What NOT
// to Test, the carve-out for a line that is the deliverable).
//
// The bracket checks compare the replica's clock with the store's `TIME`.
// The test process and the store read one kernel `CLOCK_REALTIME` (the
// sandbox's sister container, CI's service container), so a script's `now`,
// floored to whole ms, lies between the floors of the bracket around the call
// that ran it — exactly, with no tolerance beyond the scripts' whole
// milliseconds.

/// The measurement event's message and target (`mod.rs`'s
/// `LEASE_TRACE_TARGET`), spelled out: the tooling greps for both, so a
/// rename must fail here.
const LEASE_EVENT_MSG: &str = "snapshot fence lease";
const LEASE_EVENT_TARGET: &str = "live_edit::snapshot_fence_lease";

/// A `u64` field of a captured event.
fn event_u64(event: &serde_json::Value, field: &str) -> u64 {
    event["fields"][field]
        .as_u64()
        .unwrap_or_else(|| panic!("`{field}` must be a non-negative integer; event: {event}"))
}

/// The four brackets of a lease event, in µs: acquire sent / replied, release
/// sent / replied.
fn brackets(event: &serde_json::Value) -> [u64; 4] {
    [
        event_u64(event, "acquire_sent_unix_us"),
        event_u64(event, "acquire_replied_unix_us"),
        event_u64(event, "release_sent_unix_us"),
        event_u64(event, "release_replied_unix_us"),
    ]
}

/// Assert `event` is a lease event on its own target with these grant and
/// release fields, and that its brackets are ordered.
fn assert_lease_event(
    event: &serde_json::Value,
    kind: GrantKind,
    cut: u64,
    outcome: &str,
    via: &str,
) {
    assert_eq!(event["level"], "DEBUG", "the event is DEBUG: {event}");
    assert_eq!(
        event["target"], LEASE_EVENT_TARGET,
        "the event sits on its own target: {event}"
    );
    assert_eq!(event_u64(event, "object_id"), OID, "object_id: {event}");
    assert_eq!(event["fields"]["kind"], kind.as_str(), "kind: {event}");
    assert_eq!(event_u64(event, "cut"), cut, "cut: {event}");
    assert_eq!(event["fields"]["outcome"], outcome, "outcome: {event}");
    assert_eq!(event["fields"]["via"], via, "via: {event}");
    let b = brackets(event);
    assert!(
        b.windows(2).all(|w| w[0] <= w[1]),
        "acquire sent ≤ acquire replied ≤ release sent ≤ release replied: {b:?}; event: {event}"
    );
}

/// The token of the single reader on `fx`'s fence (its `r:<token>` field).
async fn only_reader_token(fx: &Fx) -> String {
    let (field, _) = only_reader(&fx.fields(&fx.fence_key()).await);
    field
        .strip_prefix("r:")
        .expect("a reader field starts with r:")
        .to_owned()
}

/// An explicit release logs exactly one event, `via` `release`, with the
/// grant's object, kind and cut and a `verified` outcome. Its acquire bracket
/// holds the store time the grant stamped as `epoch_start`, its release
/// bracket the store time the release ran at (read back from `gap_until`
/// minus the gap the coefficients give), both to the scripts' whole ms; and
/// the line carries no trace of the lease token.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_logs_one_lease_event_with_the_grant_and_store_bracketing_brackets() {
    let fx = Fx::new().await;
    let lease = fx.grant(SIZE).await;
    let token = only_reader_token(&fx).await;
    let epoch_start = int(&fx.fields(&fx.fence_key()).await, "epoch_start");
    // An epoch start ahead of the store's clock gives a negative measured
    // length, so the release's gap is `min_gap` alone and its store time is
    // `gap_until − ceil(min_gap)`.
    let s = fx.now().await;
    fx.hset(&fx.fence_key(), &[("epoch_start", ms(s + FAR_MS))])
        .await;

    let (verdict, captured) =
        crate::grpc::log_capture::capture_debug(lease.release(Some(SIZE))).await;

    assert_eq!(verdict.unwrap(), ReleaseVerdict::Verified);
    let event = captured.only(LEASE_EVENT_MSG);
    assert_lease_event(event, GrantKind::StartedEpoch, CUT, "verified", "release");
    let [acquire_sent, acquire_replied, release_sent, release_replied] = brackets(event);
    assert_between(
        epoch_start,
        (acquire_sent / 1_000) as i64,
        (acquire_replied / 1_000) as i64,
        "the grant's store time (epoch_start) lies inside the acquire bracket",
    );
    let gap_until = int(&fx.fields(&fx.fence_key()).await, "gap_until");
    let released_at = gap_until - whole_ms(fx.timing(SIZE).min_gap_ms);
    assert_between(
        released_at,
        (release_sent / 1_000) as i64,
        (release_replied / 1_000) as i64,
        "the release's store time (gap_until − min_gap) lies inside the release bracket",
    );
    captured.assert_absent_everywhere(&token);
}

/// A second lease granted inside the join window logs `kind` `joined_epoch`
/// when it is released (and only its own event).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_joined_lease_logs_kind_joined_epoch() {
    let fx = Fx::new().await;
    let first = fx.grant(SIZE).await;
    let joiner = fx.grant(OTHER_SIZE).await;
    assert_eq!(joiner.kind(), GrantKind::JoinedEpoch, "precondition");

    let (verdict, captured) =
        crate::grpc::log_capture::capture_debug(joiner.release(Some(SIZE))).await;

    assert_eq!(verdict.unwrap(), ReleaseVerdict::Verified);
    assert_lease_event(
        captured.only(LEASE_EVENT_MSG),
        GrantKind::JoinedEpoch,
        CUT,
        "verified",
        "release",
    );
    assert_eq!(
        first.release(Some(SIZE)).await.unwrap(),
        ReleaseVerdict::Verified
    );
}

/// A `BROKEN` release (the header `seq` moved past the cut) logs `outcome`
/// `broken`, and a `NOT_FOUND` release (the header deleted under the lease)
/// logs `not_found`, each `via` `release` with the grant's fields: the
/// tooling's writers'-share coverage reads this word, so neither verdict may
/// be reported as `verified` or as the other.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_broken_or_not_found_release_logs_its_own_outcome() {
    let fx = Fx::new().await;
    let lease = fx.grant(SIZE).await;
    fx.hset(&fx.header_key(), &[("seq", (CUT + 1).to_string())])
        .await;

    let (verdict, captured) =
        crate::grpc::log_capture::capture_debug(lease.release(Some(SIZE))).await;

    assert_eq!(verdict.unwrap(), ReleaseVerdict::Broken, "precondition");
    assert_lease_event(
        captured.only(LEASE_EVENT_MSG),
        GrantKind::StartedEpoch,
        CUT,
        "broken",
        "release",
    );

    let fx = Fx::new().await;
    let lease = fx.grant(SIZE).await;
    fx.unlink(vec![fx.header_key()]).await;

    let (verdict, captured) =
        crate::grpc::log_capture::capture_debug(lease.release(Some(SIZE))).await;

    assert_eq!(verdict.unwrap(), ReleaseVerdict::NotFound, "precondition");
    assert_lease_event(
        captured.only(LEASE_EVENT_MSG),
        GrantKind::StartedEpoch,
        CUT,
        "not_found",
        "release",
    );
}

/// An acquire that grants nothing — `RETRY_AFTER` or `NOT_FOUND` — logs no
/// lease event: only a release writes one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_acquire_that_grants_nothing_logs_no_lease_event() {
    let fx = Fx::new().await;
    let holder = fx.grant(SIZE).await;
    let s = fx.now().await;
    fx.hset(
        &fx.fence_key(),
        &[(
            "epoch_start",
            ms(s - whole_ms(fx.timing(SIZE).join_window_ms) - 1),
        )],
    )
    .await;

    let (verdict, captured) = crate::grpc::log_capture::capture_debug(fx.acquire(SIZE)).await;

    let (_, reason, _) = retry_after(verdict);
    assert_eq!(reason, RetryReason::Epoch, "precondition: a wait, no grant");
    assert!(
        captured.with_message(LEASE_EVENT_MSG).is_empty(),
        "RETRY_AFTER logs no lease event:\n{}",
        captured.text
    );

    let absent = Fx::without_header().await;
    let (verdict, captured) = crate::grpc::log_capture::capture_debug(absent.acquire(SIZE)).await;
    assert!(
        matches!(verdict, AcquireVerdict::NotFound),
        "precondition: NOT_FOUND; got {verdict:?}"
    );
    assert!(
        captured.with_message(LEASE_EVENT_MSG).is_empty(),
        "NOT_FOUND logs no lease event:\n{}",
        captured.text
    );
    assert_eq!(
        holder.release(Some(SIZE)).await.unwrap(),
        ReleaseVerdict::Verified
    );
}

/// Wait, within [`DROP_POLL_BUDGET`], until `capture` holds `n` lease events.
async fn lease_events_eventually(
    capture: &crate::grpc::log_capture::ThreadCapture,
    n: usize,
    what: &str,
) -> crate::grpc::log_capture::Captured {
    let wait = async {
        loop {
            let captured = capture.snapshot();
            if captured.with_message(LEASE_EVENT_MSG).len() >= n {
                return captured;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(DROP_POLL_BUDGET, wait)
        .await
        .unwrap_or_else(|_| {
            panic!("{what}: {n} lease event(s) did not appear within {DROP_POLL_BUDGET:?}")
        })
}

/// A lease dropped unreleased logs exactly one event, `via` `drop`, from the
/// guard's spawned release. Current-thread runtime: the spawned task is
/// polled on the thread holding the capture (test-writing guide, a line from
/// a spawned task).
#[tokio::test]
async fn a_lease_dropped_unreleased_logs_one_event_via_drop() {
    let capture = crate::grpc::log_capture::ThreadCapture::start_debug();
    let fx = Fx::new().await;
    let lease = fx.grant(SIZE).await;
    let token = only_reader_token(&fx).await;

    drop(lease);

    let captured = lease_events_eventually(&capture, 1, "the dropped lease's release").await;
    let events = captured.with_message(LEASE_EVENT_MSG);
    assert_eq!(
        events.len(),
        1,
        "one event for one drop:\n{}",
        captured.text
    );
    assert_lease_event(events[0], GrantKind::StartedEpoch, CUT, "verified", "drop");
    captured.assert_absent_everywhere(&token);
    crate::grpc::log_capture::purge_keys(&fx.tv).await;
}

/// A release whose call fails logs `outcome` `error`, `via` `release`; the
/// guard stays armed, and its spawned re-release logs a second event, `via`
/// `drop`, with the same acquire bracket (the pair the tooling keys a lease
/// on, keeping the last event's outcome). A string planted on the fence key
/// makes both release scripts fail with `WRONGTYPE` — no fault seam needed.
#[tokio::test]
async fn a_failed_release_logs_error_and_its_guard_re_release_logs_via_drop_with_the_same_pair() {
    let capture = crate::grpc::log_capture::ThreadCapture::start_debug();
    let fx = Fx::new().await;
    let lease = fx.grant(SIZE).await;
    plant_string(&fx, &fx.fence_key()).await;

    let verdict = lease.release(Some(SIZE)).await;

    assert!(
        verdict.is_err(),
        "a release on a wrong-typed fence key must fail; got {verdict:?}"
    );
    let captured =
        lease_events_eventually(&capture, 2, "the failed release and its re-release").await;
    let events = captured.with_message(LEASE_EVENT_MSG);
    assert_eq!(events.len(), 2, "two events:\n{}", captured.text);
    assert_lease_event(events[0], GrantKind::StartedEpoch, CUT, "error", "release");
    assert_lease_event(events[1], GrantKind::StartedEpoch, CUT, "error", "drop");
    assert_eq!(
        brackets(events[0])[..2],
        brackets(events[1])[..2],
        "both events carry the grant's acquire bracket"
    );
    crate::grpc::log_capture::purge_keys(&fx.tv).await;
}
