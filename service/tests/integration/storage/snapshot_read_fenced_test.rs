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

//! The fenced snapshot read —
//! `storage::snapshot_fence::read::read_object_snapshot_fenced` (b.y3m Epic 3
//! Task 5; Spec SDD SR-5.3, SR-5.2's acquire, SR-7.4, SR-8.1–SR-8.3, SR-1.5).
//!
//! The read takes a lease on the object's fence, runs the bounded fan-out
//! under it with a renewal timer beside it, and proves the pass in the script
//! that releases the lease: `VERIFIED` serves the state at the cut (the `seq`
//! acquire read), `BROKEN` retries, up to `FENCE_READ_ATTEMPTS` = 3. What is
//! pinned here:
//!
//! | Area | What it proves |
//! |---|---|
//! | Equality with the atomic read | on a created object and after structural and fast-path commits, marked and unmarked; a marked object with a list its index does not name; an unmarked object with a partial index; at a one-command, one-byte window (every entry larger than `FANOUT_WINDOW_BYTES`) — the fenced read returns the atomic read's state, header timestamps and tallies, on `path = Fenced`, with `cut` = the grant's `seq` |
//! | Corrupt index | an `obj:secs` member with no hash is reported, as on the atomic path |
//! | Round 1 in pages | indices that each take several pages (12,000 sections, 30,000 fields) read as the atomic read does, every round-1 window within `FANOUT_INDEX_WINDOW_MEMBERS` (the `fan-out window` event) |
//! | `NotFound` at acquire | an absent header, or one without a numeric `seq`: `NotFound`, no fence key, no hook event past `before_attempt`; one `not_found` tick, one duration and one reader-wait observation per read, no hold, no acquire tick |
//! | Commits wait | a commit started while the lease is held is refused (the writer records one fence wait), lands nothing during the read, and commits after the release; the read is at the grant's `seq` and excludes it |
//! | The proof | a pass that read another header `seq` is broken even when the release verifies; a header moved after the pass makes the release answer `BROKEN`; both retry and serve the next attempt's state |
//! | Restore / cleanup under the read | a restore wipe the pass missed (release `BROKEN`) and a header the pass missed but the release found (`VERIFIED`) are broken reads, retried; a `DeleteLayer` under the read, and a header removed after a complete pass, are `NotFound` after one attempt; a failed release ends the read with the release's classified error, whether the pass found no header or completed (its state never served) |
//! | Attempts | three broken reads are `Unavailable`, never an unverified state |
//! | The hook | the event order across a broken attempt and its retry, with the sizes each event carries; each `after_release`'s outcome — `Verified` exactly once, on the attempt whose snapshot is served, `Broken` on every retried attempt (one whose release answered `VERIFIED` included), `Ended` on a read's erroring last attempt; `before_attempt` and `on_retry_after` errors end the read as they are; `CappedFenceWait` sleeps a short `RETRY_AFTER`, refuses one past `reader_wait_max(L)`, and (on tokio's paused clock) refuses the attempt after a broken read once past the cap for that attempt's size; a read inside a live epoch's join window joins it |
//! | Metrics | `live_edit_snapshot_read_total{path="fenced"}` per attempt (`fence_broken`, `ok`, `not_found`), the duration and size histograms, `live_edit_snapshot_fence_hold_seconds` and `live_edit_snapshot_fence_reader_wait_seconds` once per attempt, `live_edit_snapshot_fence_acquire_total{result}` per acquire reply |
//! | Cancellation | dropping the read in its hook's `on_grant` or `before_release`, at its acquire or at its release leaves no live lease: a held lease is released in the background (its field gone, the epoch closed), a lease whose grant the read never saw lapses |
//! | Seams (`test-fault-injection`) | a window held past the lease keeps it (the timer renewed it); a renewal that fails is retried at the next period, not taken as lost; stalled renewals past the lease make the release `BROKEN` and the read retries; a `LOST` renewal stops the fan-out mid-window, but a `LOST` during `on_grant` or `before_release` lets that hook event run to its end; cancellation mid-window and during a renewal releases the lease and stops the timer; a `next_read()` error is consumed by the fenced fan-out's first window; a fan-out that failed with a release that found no header is `NotFound`, not the fan-out's error |
//!
//! Elsewhere: the fence scripts themselves in `snapshot_fence_test.rs`; the
//! writers' side of the fence in `snapshot_fence_writer_test.rs`; the windows'
//! mechanics (an over-target reply sharing a window sized from a smaller
//! average, an over-target entry completing) inline in
//! `src/storage/diff_read.rs`; the random-history property in
//! `tests/proptests.rs`; its production consumer, the catch-up's shared
//! fenced-read flight, in `snapshot_flights_test.rs` and the `grpc::subscribe_*`
//! tests, and its checkpoint consumer, the assembler's fenced read, in
//! `checkpoint::assembler_test`. Only the catch-up and checkpoint consumers'
//! fenced `snapshot_read_total` cells are zero-seeded (the seeding itself is
//! pinned in `observability_metrics_test.rs`), so an absent cell here — a
//! fenced read for another consumer — reads as zero.
//!
//! # Time without timing assertions
//!
//! Nothing here asserts how long anything took. Where a test needs time to
//! pass — a window held past the lease, a lease lapsing — it compares
//! deadlines the fence key stores with the store's own clock (a script reading
//! `TIME` as the fence scripts do), never a wall-clock duration. The real
//! waits are the read's own: a lease (~2 s at the test client's ceiling) where
//! a test makes one lapse, the writer's fence sleep (~1 s), the gaps the
//! fence scripts impose between epochs (the epoch's own length), two renewal
//! periods (~1 s) where a test checks that no renewal follows, three where a
//! test waits for the renewal after two failed ones, and the slow-window
//! test's window, 1.5 of the ~8 s lease its estimate is given (~12 s). The
//! `CappedFenceWait` cap test needs ~29 s to pass and takes it on tokio's
//! paused clock, not the wall clock.
//!
//! A renewal is a store round trip, and on a loaded store it can land late —
//! up to the client's 5 s command timeout. So the slow-window test, whose
//! subject is renewals keeping a lease live, gives its lease a `0.75·lease`
//! margin over that timeout; the failed-renewals test, whose subject is the
//! timer calling again, stages its lease's deadline far ahead, out of the
//! picture; and the mid-window cancellation test opens its absence window
//! only once every connection a renewal could be queued on has drained.
//!
//! # Hang budget
//!
//! Every test body runs under [`TEST_BUDGET`] (90 s), so none can reach
//! nextest's 120 s hang-killer, which prints nothing. Inside it each wait has
//! its own budget, so a stuck wait panics with its own message: one fenced
//! read [`READ_BUDGET`] (30 s; the slowest healthy read, the slow window's,
//! takes about 1.5 of its ~8 s lease, ~12 s), a writer's commit
//! [`CALL_BUDGET`] (20 s), a dropped guard's background release or a hook's
//! signal [`SIGNAL_BUDGET`] (10 s), a lease lapsing [`LAPSE_BUDGET`] (15 s),
//! a `MONITOR` start or drain `MONITOR_DRAIN_BUDGET` (20 s). A wait a hook
//! makes inside a read is nested in that read's budget. Single round trips
//! are bounded by the client's 5 s command timeout and are not tabulated.
//! Every wait summed at its full budget:
//!
//! | Test | Waits in series | Sum | Worst case |
//! |---|---|---|---|
//! | equality after commits (marked, unmarked) | 6 reads | 180 s | 90 s |
//! | one-command, one-byte windows; not-found at acquire | 2 reads | 60 s | 60 s |
//! | indices in several pages | tap + 1 read + 1 drain | 70 s | 70 s |
//! | every other equality test, corrupt index, the proof, restore / cleanup (the failed releases included), attempts, hook order and errors, `CappedFenceWait` through a read, joined epoch, metrics | 1 read | 30 s | 30 s |
//! | `CappedFenceWait` on the paused clock | none (~29 s of virtual time) | 0 s | 0 s |
//! | commits wait | tap + 1 read (one drain inside it) + 1 call | 70 s | 70 s |
//! | cancel in `on_grant` / `before_release` | 1 signal + 1 join + 1 release poll | 30 s | 30 s |
//! | cancel at acquire | warm-up read + tap + 1 drain + 1 join + 1 lapse poll | 95 s | 90 s |
//! | cancel at release | warm-up read + tap + 1 signal + 1 drain + 1 join + 1 release poll | 100 s | 90 s |
//! | seams: slow window, stalled renewals, `LOST`, failed renewals, `LOST` during a hook event (their lapse polls, renewal polls and renewal periods nested in the read) | 1 read | 30 s | 30 s |
//! | seams: cancel mid-window | warm-up read + 2 taps + 2 drains + 1 join + 1 release poll + two renewal periods (~1 s) | 131 s | 90 s |
//! | seams: cancel during a renewal | warm-up read + tap + 1 drain + 1 join + 1 release poll | 90 s | 90 s |
//! | seams: `next_read()` error; failed fan-out with a release that found no header | 1 read | 30 s | 30 s |
//!
//! The healthy path takes a few seconds.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::future::Future;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use fred::interfaces::{HashesInterface, KeysInterface, ListInterface, SetsInterface};
use fred::types::scripts::Script;
use futures::future::BoxFuture;

use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::diff_read::{AtomicSnapshot, SnapshotReadConsumer, SnapshotReadPath};
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::lua::{UPDATE_SCRIPT_KEY_FENCE, UPDATE_SCRIPT_KEY_HEADER};
use live_edit::storage::snapshot_fence::read::{
    read_object_snapshot_fenced, CappedFenceWait, FenceAttempt, FenceAttemptOutcome, FenceGrant,
    FenceReleased, FenceRetry, FencedReadHook,
};
use live_edit::storage::snapshot_fence::{
    self as fence, AcquireVerdict, GrantKind, LayerSize, ReleaseVerdict, RetryReason,
    FENCE_READ_ATTEMPTS,
};
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::delta_with_set_field_inline;
use crate::storage::snapshot_read_atomic_test::{
    commit, create, create_cmd, create_session, every_shape_state, indexed_parents, layer_store,
    mock_ext_store, structural, unmark, CUBE, ROOT, SIBLING, WORLD,
};
use crate::storage::snapshot_read_support::{
    client_through, client_with_limits, error_reply_script, monitor_fence, read_atomic,
    read_too_large, test_valkey_addr, view, MonitorTap, MonitoredCommand, ProxyAction, ProxyRule,
    ScriptReplyProxy,
};
use crate::storage::snapshot_read_unfenced_test::{
    add_paged_bulk, assert_round_one_windows_within_the_bound, commit_steps, index_pages,
    PAGED_FIELDS, PAGED_SECTIONS,
};

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

/// The reading client's item ceiling: far below every fixture object here, so
/// each is above the ceiling and the fenced read is the path a consumer would
/// take. `FANOUT_WINDOW` is then ⌊16 ÷ 5.266⌋ = 3 commands.
const LOW_ITEMS: u64 = 16;

/// The reading client's byte ceiling, and so the fan-out's byte target.
const LOW_BYTES: u64 = 256;

/// Bound on one fenced read. See the module's hang budget.
const READ_BUDGET: Duration = Duration::from_secs(30);

/// Bound on one `update_layer` call (a fenced writer sleeps about one `poll`,
/// ~1.1 s, per refusal).
const CALL_BUDGET: Duration = Duration::from_secs(20);

/// Bound on a dropped lease's background release landing, and on a hook's
/// signal arriving. One script call each; unreachable on a healthy store.
const SIGNAL_BUDGET: Duration = Duration::from_secs(10);

/// Bound on a lease nobody releases lapsing: one lease (~2.2 s) plus slack.
const LAPSE_BUDGET: Duration = Duration::from_secs(15);

/// The one deadline over a whole test. See the module's hang budget.
const TEST_BUDGET: Duration = Duration::from_secs(90);

/// How far ahead of the store's clock a staged gap is "far": past every
/// `reader_wait_max(L)` of the sizes here.
const FAR_MS: i64 = 600_000;

/// The caller's estimate for a read on a client connected through a
/// [`ScriptReplyProxy`]: that client has the default ceilings, so no
/// `TOO_LARGE` answer gives one. A small layer's size.
const PROXIED_ESTIMATE: LayerSize = LayerSize {
    items: 1_000,
    bytes: 12_000,
};

/// Run a test's body under [`TEST_BUDGET`], boxed so the body's future lives
/// on the heap rather than in the test thread's stack frame.
fn within_test_budget<'a>(
    body: impl Future<Output = ()> + 'a,
) -> std::pin::Pin<Box<dyn Future<Output = ()> + 'a>> {
    let body = Box::pin(body);
    Box::pin(async move {
        tokio::time::timeout(TEST_BUDGET, body)
            .await
            .unwrap_or_else(|_| {
                panic!("the test did not finish within TEST_BUDGET ({TEST_BUDGET:?})")
            });
    })
}

/// The store clock, as the fence scripts read it: `TIME` in whole ms.
static CLOCK_SCRIPT: std::sync::LazyLock<Script> = std::sync::LazyLock::new(|| {
    Script::from_lua(
        "local t = redis.call('TIME') \
         return tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)",
    )
});

/// The fence scripts' SHAs, for recognising their `EVALSHA`s in a `MONITOR`
/// feed and on a proxied connection.
fn script_sha(text: &str) -> String {
    let script = Script::from_lua(text);
    let sha: &str = script.sha1();
    sha.to_owned()
}

fn acquire_sha() -> String {
    script_sha(include_str!(
        "../../../src/storage/scripts/snapshot_fence_acquire.lua"
    ))
}

fn release_sha() -> String {
    script_sha(include_str!(
        "../../../src/storage/scripts/snapshot_fence_release.lua"
    ))
}

#[cfg(feature = "test-fault-injection")]
fn renew_sha() -> String {
    script_sha(include_str!(
        "../../../src/storage/scripts/snapshot_fence_renew.lua"
    ))
}

/// Whether a proxied command frame is an `EVALSHA` of the script `sha`.
fn is_evalsha_of(args: &[Vec<u8>], sha: &str) -> bool {
    args.first().map(|a| a.to_ascii_uppercase()) == Some(b"EVALSHA".to_vec())
        && args.get(1).map(|a| a.as_slice()) == Some(sha.as_bytes())
}

// ═════════════════════════════════════════════════════════════════════════════
// The recording hook
// ═════════════════════════════════════════════════════════════════════════════

/// One event the read told its hook of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HookEvent {
    BeforeAttempt(FenceAttempt),
    RetryAfter(FenceRetry),
    Grant(FenceGrant),
    BeforeRelease,
    AfterRelease(FenceReleased),
}

impl HookEvent {
    fn name(&self) -> &'static str {
        match self {
            HookEvent::BeforeAttempt(_) => "before_attempt",
            HookEvent::RetryAfter(_) => "on_retry_after",
            HookEvent::Grant(_) => "on_grant",
            HookEvent::BeforeRelease => "before_release",
            HookEvent::AfterRelease(_) => "after_release",
        }
    }
}

type EventLog = Arc<StdMutex<Vec<HookEvent>>>;

/// What the hook does on an event, given the attempt it belongs to: a staged
/// mutation of the store, a signal to the test, or an error to refuse with.
/// An error from an infallible event (`on_grant`, `before_release`,
/// `after_release`) fails the test.
type Reaction =
    Box<dyn FnMut(u32, HookEvent) -> BoxFuture<'static, Result<(), ServiceError>> + Send>;

fn react<F, Fut>(mut f: F) -> Reaction
where
    F: FnMut(u32, HookEvent) -> Fut + Send + 'static,
    Fut: Future<Output = Result<(), ServiceError>> + Send + 'static,
{
    Box::new(
        move |attempt, event| -> BoxFuture<'static, Result<(), ServiceError>> {
            Box::pin(f(attempt, event))
        },
    )
}

fn no_reaction() -> Reaction {
    react(|_, _| async { Ok(()) })
}

/// A [`FencedReadHook`] that logs every event, runs its [`Reaction`] on it,
/// and on `RETRY_AFTER` sleeps the hint (as `CappedFenceWait` does, with no
/// cap). It holds no capacity.
struct TestHook {
    log: EventLog,
    attempt: u32,
    react: Reaction,
}

impl TestHook {
    fn new(react: Reaction) -> (Self, EventLog) {
        let log = EventLog::default();
        (
            TestHook {
                log: Arc::clone(&log),
                attempt: 0,
                react,
            },
            log,
        )
    }

    async fn event(&mut self, event: HookEvent) -> Result<(), ServiceError> {
        if let HookEvent::BeforeAttempt(a) = event {
            self.attempt = a.attempt;
        }
        self.log.lock().expect("event log").push(event);
        (self.react)(self.attempt, event).await
    }
}

impl FencedReadHook for TestHook {
    async fn before_attempt(&mut self, attempt: FenceAttempt) -> Result<(), ServiceError> {
        self.event(HookEvent::BeforeAttempt(attempt)).await
    }

    async fn on_retry_after(&mut self, retry: FenceRetry) -> Result<(), ServiceError> {
        self.event(HookEvent::RetryAfter(retry)).await?;
        tokio::time::sleep(retry.hint).await;
        Ok(())
    }

    async fn on_grant(&mut self, grant: FenceGrant) {
        self.event(HookEvent::Grant(grant))
            .await
            .expect("an on_grant reaction must not fail");
    }

    async fn before_release(&mut self) {
        self.event(HookEvent::BeforeRelease)
            .await
            .expect("a before_release reaction must not fail");
    }

    async fn after_release(&mut self, released: FenceReleased) {
        self.event(HookEvent::AfterRelease(released))
            .await
            .expect("an after_release reaction must not fail");
    }
}

fn events(log: &EventLog) -> Vec<HookEvent> {
    log.lock().expect("event log").clone()
}

/// The event names, `RETRY_AFTER`s left out (how many a retry meets depends
/// on how much of the writers' gap is left when it calls).
fn names_without_retries(log: &EventLog) -> Vec<&'static str> {
    events(log)
        .iter()
        .filter(|e| !matches!(e, HookEvent::RetryAfter(_)))
        .map(HookEvent::name)
        .collect()
}

fn grants(log: &EventLog) -> Vec<FenceGrant> {
    events(log)
        .into_iter()
        .filter_map(|e| match e {
            HookEvent::Grant(g) => Some(g),
            _ => None,
        })
        .collect()
}

fn releases(log: &EventLog) -> Vec<FenceReleased> {
    events(log)
        .into_iter()
        .filter_map(|e| match e {
            HookEvent::AfterRelease(r) => Some(r),
            _ => None,
        })
        .collect()
}

/// How each attempt that got a lease ended, as its `after_release` was told.
fn outcomes(log: &EventLog) -> Vec<FenceAttemptOutcome> {
    releases(log).into_iter().map(|r| r.outcome).collect()
}

/// Assert `fenced` was served by the read's `attempts`-th attempt and only
/// that one: every earlier attempt's `after_release` heard `Broken`, the last
/// one `Verified` — the read's only `Verified`, the event a hook's
/// once-per-read work keys on — and the served snapshot is that attempt's: its
/// cut is that attempt's grant, its measured size the one that release was
/// handed.
fn assert_served_by_the_last_attempt(
    log: &EventLog,
    fenced: &AtomicSnapshot,
    attempts: usize,
    what: &str,
) {
    let mut expected = vec![FenceAttemptOutcome::Broken; attempts - 1];
    expected.push(FenceAttemptOutcome::Verified);
    assert_eq!(
        outcomes(log),
        expected,
        "{what}: Verified exactly once, on the attempt whose snapshot the read returns"
    );
    assert_eq!(
        grants(log).last().map(|g| g.cut),
        Some(fenced.cut),
        "{what}: the served snapshot is at the Verified attempt's cut"
    );
    assert_eq!(
        releases(log).last().and_then(|r| r.measured),
        Some(fenced.measured_size()),
        "{what}: the Verified attempt's release was handed the served snapshot's size"
    );
}

/// The event sequence of `n` attempts that each got a lease.
fn held_attempts(n: usize) -> Vec<&'static str> {
    [
        "before_attempt",
        "on_grant",
        "before_release",
        "after_release",
    ]
    .into_iter()
    .cycle()
    .take(4 * n)
    .collect()
}

/// A one-shot signal from a reaction to the test.
#[derive(Clone, Default)]
struct Signal(Arc<tokio::sync::Notify>);

impl Signal {
    fn fire(&self) {
        self.0.notify_one();
    }

    async fn wait(&self, what: &str) {
        tokio::time::timeout(SIGNAL_BUDGET, self.0.notified())
            .await
            .unwrap_or_else(|_| panic!("{what} did not happen within {SIGNAL_BUDGET:?}"));
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Staging: what a non-commit mutator does to the object under the read
// ═════════════════════════════════════════════════════════════════════════════

/// The object's header, fence and intent keys on the fixture's own client,
/// for reactions to change under the read. Cheap to clone into a reaction.
#[derive(Clone)]
struct Stage {
    client: ValkeyClient,
    header: String,
    fence: String,
    intent: String,
}

impl Stage {
    async fn set_seq(&self, seq: u64) {
        let _: () = self
            .client
            .pool
            .hset(&self.header, [("seq", seq.to_string())])
            .await
            .expect("staging HSET seq must succeed");
    }

    async fn incr_seq(&self) {
        let _: i64 = self
            .client
            .pool
            .hincrby(&self.header, "seq", 1)
            .await
            .expect("staging HINCRBY seq must succeed");
    }

    /// Remove the header, returning what it held.
    async fn take_header(&self) -> HashMap<String, String> {
        let header: HashMap<String, String> = self
            .client
            .pool
            .hgetall(&self.header)
            .await
            .expect("staging HGETALL must succeed");
        assert!(!header.is_empty(), "staging: the header must exist to take");
        let _: i64 = self
            .client
            .pool
            .del(&self.header)
            .await
            .expect("staging DEL must succeed");
        header
    }

    async fn put_header(&self, header: HashMap<String, String>) {
        let _: () = self
            .client
            .pool
            .hset(&self.header, header)
            .await
            .expect("staging HSET header must succeed");
    }

    /// A restore's wipe of the object's fence-protocol keys and header (the
    /// restore residue wipe `UNLINK`s them with the object's other keys); the
    /// header's content is returned for the restore to write back.
    async fn wipe(&self) -> HashMap<String, String> {
        let header: HashMap<String, String> = self
            .client
            .pool
            .hgetall(&self.header)
            .await
            .expect("staging HGETALL must succeed");
        let _: i64 = self
            .client
            .pool
            .unlink(vec![
                self.header.clone(),
                self.fence.clone(),
                self.intent.clone(),
            ])
            .await
            .expect("staging UNLINK must succeed");
        header
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// One object on a fresh store, read by a client whose atomic-read ceiling it
/// is above, with a metrics registry on that client.
struct Rx {
    tv: TestValkey,
    /// The low-ceiling client every fenced read here runs on.
    reader: ValkeyClient,
    metrics: Arc<Metrics>,
    ls: LayerStore,
    session_id: String,
    layer: String,
    oid: u64,
}

impl Rx {
    /// [`every_shape_state`] created as `layer`, marked.
    async fn new(layer: &str) -> Self {
        let tv = TestValkey::setup().await;
        let reader = client_with_limits(LOW_ITEMS, LOW_BYTES).await;
        Self::with_reader(tv, reader, layer).await
    }

    async fn with_reader(tv: TestValkey, reader: ValkeyClient, layer: &str) -> Self {
        let metrics = Arc::new(crate::common::metrics_scrape_only("fenced-read"));
        reader
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let ls = layer_store(&tv);
        let session_id = create_session(&tv, layer).await;
        let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
        Rx {
            tv,
            reader,
            metrics,
            ls,
            session_id,
            layer: layer.to_owned(),
            oid,
        }
    }

    fn keys(&self) -> &KeySchema {
        self.tv.key_schema()
    }

    fn oid_str(&self) -> String {
        self.oid.to_string()
    }

    fn header_key(&self) -> String {
        self.keys().obj_header(&self.oid_str())
    }

    fn fence_key(&self) -> String {
        self.keys().obj_snapshot_fence(&self.oid_str())
    }

    fn intent_key(&self) -> String {
        self.keys().obj_snapshot_intent(&self.oid_str())
    }

    fn stage(&self) -> Stage {
        Stage {
            client: self.tv.client().clone(),
            header: self.header_key(),
            fence: self.fence_key(),
            intent: self.intent_key(),
        }
    }

    /// The caller's size estimate, as a consumer derives it (Spec SDD
    /// SR-11.1): from the atomic read's `TOO_LARGE` answer on the reading
    /// client. Also the precondition that the object is above its ceiling.
    async fn estimate(&self) -> LayerSize {
        let too_large = read_too_large(&self.reader, self.keys(), self.oid).await;
        LayerSize::from_too_large(&too_large, None)
    }

    /// The fenced read of this object by `consumer` on the reading client,
    /// bounded by [`READ_BUDGET`].
    async fn read_as<H: FencedReadHook>(
        &self,
        consumer: SnapshotReadConsumer,
        estimate: LayerSize,
        hook: &mut H,
    ) -> Result<AtomicSnapshot, ServiceError> {
        tokio::time::timeout(
            READ_BUDGET,
            read_object_snapshot_fenced(
                &self.reader,
                self.keys(),
                self.oid,
                consumer,
                estimate,
                hook,
            ),
        )
        .await
        .unwrap_or_else(|_| panic!("the fenced read did not finish within {READ_BUDGET:?}"))
    }

    async fn read<H: FencedReadHook>(
        &self,
        estimate: LayerSize,
        hook: &mut H,
    ) -> Result<AtomicSnapshot, ServiceError> {
        self.read_as(SnapshotReadConsumer::Catchup, estimate, hook)
            .await
    }

    /// The fenced read as a background task, for cancelling it.
    fn spawn_read(
        &self,
        estimate: LayerSize,
        mut hook: TestHook,
    ) -> tokio::task::JoinHandle<Result<AtomicSnapshot, ServiceError>> {
        let (client, keys, oid) = (self.reader.clone(), self.keys().clone(), self.oid);
        tokio::spawn(async move {
            read_object_snapshot_fenced(
                &client,
                &keys,
                oid,
                SnapshotReadConsumer::Catchup,
                estimate,
                &mut hook,
            )
            .await
        })
    }

    /// The atomic read through the fixture's own client (default ceilings):
    /// the oracle.
    async fn atomic(&self) -> AtomicSnapshot {
        read_atomic(self.tv.client(), self.keys(), self.oid).await
    }

    async fn now(&self) -> i64 {
        store_now(self.tv.client(), &self.fence_key()).await
    }

    async fn fields(&self, key: &str) -> BTreeMap<String, String> {
        key_fields(self.tv.client(), key).await
    }

    async fn seq(&self) -> u64 {
        self.fields(&self.header_key())
            .await
            .get("seq")
            .expect("the header has a seq")
            .parse()
            .expect("seq is numeric")
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

    /// The reader fields (`r:<token>` → deadline) of the fence key.
    async fn readers(&self) -> BTreeMap<String, i64> {
        readers_of(&self.fields(&self.fence_key()).await)
    }

    fn metric(&self, name: &str, labels: &[(&str, &str)]) -> f64 {
        sample_value(&render(&self.metrics), name, labels).unwrap_or(0.0)
    }

    /// `live_edit_snapshot_read_total{consumer, path="fenced", outcome}`;
    /// absent cells read as zero: only the `catchup` and `checkpoint`
    /// consumers' fenced cells are seeded, and a test here may read the fenced
    /// path for another consumer.
    fn reads(&self, consumer: &str, outcome: &str) -> u64 {
        self.metric(
            "live_edit_snapshot_read_total",
            &[
                ("consumer", consumer),
                ("path", "fenced"),
                ("outcome", outcome),
            ],
        ) as u64
    }

    fn acquires(&self, result: &str) -> u64 {
        self.metric(
            "live_edit_snapshot_fence_acquire_total",
            &[("result", result)],
        ) as u64
    }

    fn histogram_count(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        self.metric(&format!("{name}_count"), labels) as u64
    }

    /// Poll until the fence key holds no `r:` field: a held lease was
    /// released — a lapsed one leaves its field in place until the next grant
    /// prunes it — within [`SIGNAL_BUDGET`].
    async fn until_released(&self, what: &str) {
        let wait = async {
            while !self.readers().await.is_empty() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        };
        tokio::time::timeout(SIGNAL_BUDGET, wait)
            .await
            .unwrap_or_else(|_| {
                panic!("{what}: the lease was not released within {SIGNAL_BUDGET:?}")
            });
    }

    /// Poll until no reader field is live on the store's clock, within
    /// [`LAPSE_BUDGET`].
    async fn until_no_live_lease(&self, what: &str) {
        let wait = async {
            loop {
                let now = self.now().await;
                if self
                    .readers()
                    .await
                    .values()
                    .all(|deadline| *deadline <= now)
                {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        };
        tokio::time::timeout(LAPSE_BUDGET, wait)
            .await
            .unwrap_or_else(|_| panic!("{what}: a lease was still live after {LAPSE_BUDGET:?}"));
    }

    /// Load the fence scripts on the store before a `MONITOR` tap starts, so
    /// no `EVALSHA` it records is a `NOSCRIPT` miss: one fenced read of a
    /// second layer, and one renewal of a lease on it.
    async fn warm_scripts(&self) {
        let other = create(
            &self.ls,
            &self.session_id,
            &format!("{}-warm", self.layer),
            &every_shape_state(),
        )
        .await;
        let estimate = self.estimate().await;
        let mut wait = CappedFenceWait::new(&self.reader, other, SnapshotReadConsumer::Catchup);
        tokio::time::timeout(
            READ_BUDGET,
            read_object_snapshot_fenced(
                &self.reader,
                self.keys(),
                other,
                SnapshotReadConsumer::Catchup,
                estimate,
                &mut wait,
            ),
        )
        .await
        .expect("the warm-up read must finish within its budget")
        .expect("the warm-up read must succeed");
        // The epoch just ended, so acquire answers RETRY_AFTER gap first: a
        // writers' gap as long as that epoch, a few ms.
        loop {
            match fence::acquire_fence(&self.reader, self.keys(), other, estimate)
                .await
                .expect("the warm-up acquire must answer")
            {
                AcquireVerdict::Granted(lease) => {
                    lease
                        .renew()
                        .await
                        .expect("the warm-up renewal must answer");
                    lease
                        .release(None)
                        .await
                        .expect("the warm-up release must answer");
                    return;
                }
                AcquireVerdict::RetryAfter { hint, .. } => tokio::time::sleep(hint).await,
                AcquireVerdict::NotFound => panic!("the warm-up object must exist"),
            }
        }
    }
}

async fn store_now(client: &ValkeyClient, routing_key: &str) -> i64 {
    CLOCK_SCRIPT
        .evalsha_with_reload::<i64, _, _>(
            client.pool.next(),
            vec![routing_key.to_owned()],
            Vec::<String>::new(),
        )
        .await
        .expect("TIME must answer")
}

async fn key_fields(client: &ValkeyClient, key: &str) -> BTreeMap<String, String> {
    let raw: HashMap<String, String> = client
        .pool
        .hgetall(key)
        .await
        .expect("HGETALL must succeed");
    raw.into_iter().collect()
}

fn readers_of(fields: &BTreeMap<String, String>) -> BTreeMap<String, i64> {
    fields
        .iter()
        .filter(|(k, _)| k.starts_with("r:"))
        .map(|(k, v)| (k.clone(), v.parse().expect("a deadline is an integer")))
        .collect()
}

/// A verified fenced read: `Ok`, on the fenced path.
fn verified(result: Result<AtomicSnapshot, ServiceError>, what: &str) -> AtomicSnapshot {
    let snapshot =
        result.unwrap_or_else(|e| panic!("{what}: the fenced read must verify; got {e:?}"));
    assert_eq!(
        snapshot.path,
        SnapshotReadPath::Fenced,
        "{what}: the fenced read reports path = Fenced"
    );
    snapshot
}

/// Assert `fenced` is exactly `atomic`'s read of the same quiescent state.
fn assert_same_read(fenced: &AtomicSnapshot, atomic: &AtomicSnapshot, what: &str) {
    assert_eq!(
        view(&fenced.state),
        view(&atomic.state),
        "{what}: on a quiescent object the fenced pass must return exactly the atomic read's \
         state — the same keys under the same children rule"
    );
    assert_eq!(
        (fenced.cut, fenced.last_server_ts, fenced.last_client_ts),
        (atomic.cut, atomic.last_server_ts, atomic.last_client_ts),
        "{what}: the cut and the header timestamps are the header's"
    );
    assert_eq!(
        (fenced.item_count, fenced.reply_bytes, fenced.entry_count),
        (atomic.item_count, atomic.reply_bytes, atomic.entry_count),
        "{what}: the tallies are counted as the atomic read counts them — the fence key records \
         them as the layer's size for its next epoch"
    );
    assert!(
        fenced.corrupt_section_index_members.is_empty(),
        "{what}: a healthy object has no corrupt index"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. Equality with the atomic read (SR-1.2, SR-5.3)
// ═════════════════════════════════════════════════════════════════════════════

/// Read `rx`'s object fenced and atomically and assert they agree; also that
/// the read's cut is its grant's, that a clean read is one attempt with the
/// four events in order, and that the release verified, the attempt ended
/// `Verified`, and the release was handed the read's measured size. Returns
/// the fenced read.
async fn assert_fenced_equals_atomic(rx: &Rx, what: &str) -> AtomicSnapshot {
    let atomic = rx.atomic().await;
    let estimate = rx.estimate().await;
    let (mut hook, log) = TestHook::new(no_reaction());
    let fenced = verified(rx.read(estimate, &mut hook).await, what);

    assert_same_read(&fenced, &atomic, what);
    assert_eq!(
        names_without_retries(&log),
        held_attempts(1),
        "{what}: a read nothing disturbs is one attempt"
    );
    assert_eq!(
        grants(&log)[0].cut,
        fenced.cut,
        "{what}: the cut is the seq acquire read"
    );
    assert_eq!(
        releases(&log),
        vec![FenceReleased {
            verdict: Some(ReleaseVerdict::Verified),
            outcome: FenceAttemptOutcome::Verified,
            measured: Some(fenced.measured_size()),
        }],
        "{what}: the release verified the read, the attempt is the served one, and the release \
         was handed the read's measured size"
    );
    assert!(
        rx.readers().await.is_empty(),
        "{what}: the released lease leaves no reader field"
    );
    fenced
}

async fn equality_on_a_created_object(marked: bool) {
    let rx = Rx::new("srf-eq-create").await;
    if !marked {
        unmark(&rx.tv, rx.oid).await;
    }
    assert_fenced_equals_atomic(
        &rx,
        if marked {
            "created, marked"
        } else {
            "created, unmarked"
        },
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_equals_the_atomic_read_on_a_created_marked_object() {
    within_test_budget(equality_on_a_created_object(true)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_equals_the_atomic_read_on_a_created_unmarked_object() {
    within_test_budget(equality_on_a_created_object(false)).await;
}

async fn equality_after_commits(marked: bool) {
    let rx = Rx::new("srf-eq-commits").await;
    if !marked {
        unmark(&rx.tv, rx.oid).await;
    }
    for (step, delta) in commit_steps() {
        commit(&rx.ls, &rx.session_id, &rx.layer, rx.oid, delta).await;
        assert_fenced_equals_atomic(
            &rx,
            &format!(
                "after {step} ({})",
                if marked { "marked" } else { "unmarked" }
            ),
        )
        .await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_equals_the_atomic_read_after_structural_and_fast_path_commits_marked() {
    within_test_budget(equality_after_commits(true)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_equals_the_atomic_read_after_structural_and_fast_path_commits_unmarked() {
    within_test_budget(equality_after_commits(false)).await;
}

/// The marked children rule (SR-1.2): a list under a live section that a
/// marked object's `obj:chs` does not name is not part of the object, so the
/// fenced pass, like the atomic read, must leave it out.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_of_a_marked_object_reads_only_the_lists_its_children_index_names() {
    within_test_budget(async {
        let rx = Rx::new("srf-marked-stray").await;
        let _: i64 = rx
            .tv
            .client()
            .pool
            .rpush(
                rx.keys().obj_children(&rx.oid_str(), &SIBLING.to_string()),
                vec![CUBE.to_string()],
            )
            .await
            .expect("RPUSH of the stray list must succeed");
        assert!(
            !indexed_parents(&rx.tv, rx.oid).await.contains(&SIBLING),
            "fixture precondition: the stray list must be unindexed"
        );

        let fenced = assert_fenced_equals_atomic(&rx, "marked, stray unindexed list").await;
        assert!(
            !fenced.state.children.contains_key(&SIBLING),
            "on a marked object the pass takes its lists from `obj:chs`"
        );
    })
    .await;
}

/// The unmarked-partial-index regression (Spec SDD SR-1.7) on the fenced
/// pass: an unmarked object's `obj:chs` may name only some of its lists, so
/// every list under the root and the live sections must be read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_of_an_unmarked_object_with_a_partial_children_index_returns_every_list() {
    within_test_budget(async {
        let rx = Rx::new("srf-unmarked-partial").await;
        unmark(&rx.tv, rx.oid).await;
        let _: i64 = rx
            .tv
            .client()
            .pool
            .del(rx.keys().obj_children_index(&rx.oid_str()))
            .await
            .expect("DEL obj:chs must succeed");
        commit(
            &rx.ls,
            &rx.session_id,
            &rx.layer,
            rx.oid,
            structural(vec![create_cmd(CUBE, 5, "New")]),
        )
        .await;
        assert_eq!(
            indexed_parents(&rx.tv, rx.oid).await,
            BTreeSet::from([CUBE]),
            "fixture precondition: `obj:chs` names only the list the one SetChildren wrote"
        );

        let fenced = assert_fenced_equals_atomic(&rx, "unmarked, partial obj:chs").await;
        assert_eq!(
            fenced
                .state
                .children
                .keys()
                .copied()
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([ROOT, WORLD, CUBE]),
            "an unmarked object's lists must all be read, not only the indexed ones"
        );
    })
    .await;
}

/// The fan-out at its smallest under the fence: a one-item ceiling makes
/// `FANOUT_WINDOW` one command and a one-byte ceiling makes
/// `FANOUT_WINDOW_BYTES` one byte, so every entry is larger than the byte
/// target. The read still completes — a window never admits fewer than one
/// command — and equals the atomic read, marked and unmarked.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_completes_when_every_entry_is_larger_than_the_window_byte_target() {
    within_test_budget(async {
        for marked in [true, false] {
            let tv = TestValkey::setup().await;
            let narrow = client_with_limits(1, 1).await;
            let rx = Rx::with_reader(tv, narrow, "srf-narrow").await;
            if !marked {
                unmark(&rx.tv, rx.oid).await;
            }
            for (_, delta) in commit_steps() {
                commit(&rx.ls, &rx.session_id, &rx.layer, rx.oid, delta).await;
            }
            let fenced = assert_fenced_equals_atomic(
                &rx,
                &format!(
                    "one-command, one-byte windows ({})",
                    if marked { "marked" } else { "unmarked" }
                ),
            )
            .await;
            assert!(
                fenced.entry_count > 1 && fenced.reply_bytes > fenced.entry_count,
                "fixture precondition: several entries, each over the one-byte target"
            );
        }
    })
    .await;
}

/// An `obj:secs` member whose section hash is gone is a corrupt index under
/// the fence, where no commit can land between the pass's rounds: the pass
/// reports it, as the atomic read does (Spec SDD SR-4.1), for the checkpoint
/// assembler to refuse the layer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_reports_an_indexed_section_without_a_hash_as_a_corrupt_index() {
    within_test_budget(async {
        const GHOST: u64 = 99;
        let rx = Rx::new("srf-corrupt").await;
        let _: i64 = rx
            .tv
            .client()
            .pool
            .sadd(
                rx.keys().obj_sections_index(&rx.oid_str()),
                GHOST.to_string(),
            )
            .await
            .expect("SADD of the ghost member must succeed");
        let atomic = rx.atomic().await;
        assert_eq!(
            atomic.corrupt_section_index_members,
            vec![GHOST.to_string()],
            "fixture precondition: the atomic read reports the ghost member"
        );

        let estimate = rx.estimate().await;
        let fenced = verified(
            rx.read_as(
                SnapshotReadConsumer::Checkpoint,
                estimate,
                &mut TestHook::new(no_reaction()).0,
            )
            .await,
            "ghost section member",
        );
        assert_eq!(
            fenced.corrupt_section_index_members,
            vec![GHOST.to_string()],
            "the fenced pass must name the member whose hash is gone, exactly as the atomic read \
             does"
        );
        assert_eq!(view(&fenced.state), view(&atomic.state));
    })
    .await;
}

/// Round 1 in pages under the fence (Spec SDD SR-11.3, SR-11.6 (i)): a layer
/// whose sections set (12,000 members) and fields sorted set (30,000) each
/// take more than one page reads exactly as the atomic read does (run on a
/// client whose ceilings it is under), every round-1 window reads at most
/// `FANOUT_INDEX_WINDOW_MEMBERS` index members, and the store saw several
/// `SSCAN`s of the sections index and several `ZRANGE`s of the fields index.
/// The read runs at the default ceilings, which the layer is far above.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_of_indices_that_take_several_pages_equals_the_atomic_read() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let reader = tv.client().clone();
        let rx = Rx::with_reader(tv, reader, "srf-paged").await;
        add_paged_bulk(&rx.tv, rx.oid, PAGED_SECTIONS, PAGED_FIELDS).await;
        let unbounded = client_with_limits(u64::MAX / 4, u64::MAX / 4).await;
        let atomic = read_atomic(&unbounded, rx.keys(), rx.oid).await;
        let estimate = rx.estimate().await;
        let (mut hook, _) = TestHook::new(no_reaction());
        let mut tap = MonitorTap::start(&test_valkey_addr()).await;

        let (result, captured) =
            crate::grpc::log_capture::capture_debug(rx.read(estimate, &mut hook)).await;

        let fence = monitor_fence(rx.tv.prefix());
        let _: i64 = rx.tv.client().pool.exists(&fence).await.unwrap();
        let seen = tap.drain_until(&fence).await;
        let fenced = verified(result, "indices in several pages");
        let pages = assert_round_one_windows_within_the_bound(&captured, "fenced");
        assert!(
            pages.len() >= 2,
            "42,000 index members take more than one window: {pages:?}"
        );
        let (sscans, zranges) = index_pages(
            &seen,
            &rx.keys().obj_sections_index(&rx.oid_str()),
            &rx.keys().obj_fields_index(&rx.oid_str()),
        );
        assert!(
            sscans >= 2 && zranges >= 2,
            "each index is past its share of one window, so each took more than one page \
             ({sscans} SSCANs of the sections index, {zranges} ZRANGEs of the fields index)"
        );
        assert_same_read(&fenced, &atomic, "indices in several pages");
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. NotFound at acquire (SR-1.5, SR-5.2)
// ═════════════════════════════════════════════════════════════════════════════

/// An absent header (the first pass), and a header without a numeric `seq`
/// (the second), are `NotFound` at acquire: no fence key is created, the hook
/// hears only `before_attempt`, and each read records one `not_found` tick
/// with its acquire round trip as its duration, one reader-wait observation,
/// no hold, and no acquire tick (`NOT_FOUND` is not an acquire result). The
/// counts are cumulative, so after pass `n` each per-read count is `n`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_without_a_numeric_header_seq_is_not_found_at_acquire_and_writes_nothing() {
    within_test_budget(async {
        let rx = Rx::new("srf-not-found").await;
        let estimate = rx.estimate().await;
        let stage = rx.stage();
        let consumer = [("consumer", "catchup")];
        for (n, seq) in [None, Some("not-a-number")].into_iter().enumerate() {
            let reads = n as u64 + 1;
            stage.take_header().await;
            if let Some(seq) = seq {
                stage
                    .put_header(HashMap::from([("seq".to_owned(), seq.to_owned())]))
                    .await;
            }
            let (mut hook, log) = TestHook::new(no_reaction());

            let err = rx.read(estimate, &mut hook).await.err();

            assert!(
                matches!(err, Some(ServiceError::NotFound(_))),
                "seq {seq:?}: no live header is NotFound; got {err:?}"
            );
            assert_eq!(
                names_without_retries(&log),
                vec!["before_attempt"],
                "seq {seq:?}"
            );
            assert!(
                !rx.exists(&rx.fence_key()).await,
                "seq {seq:?}: an acquire that grants nothing writes nothing"
            );
            assert_eq!(
                (
                    rx.reads("catchup", "not_found"),
                    rx.histogram_count(
                        "live_edit_snapshot_read_duration_seconds",
                        &[("consumer", "catchup"), ("path", "fenced")]
                    ),
                    rx.histogram_count("live_edit_snapshot_fence_reader_wait_seconds", &consumer),
                    rx.histogram_count("live_edit_snapshot_fence_hold_seconds", &consumer),
                ),
                (reads, reads, reads, 0),
                "seq {seq:?}: (not_found reads, read durations, reader waits, holds)"
            );
            for result in live_edit::observability::metrics::SNAPSHOT_FENCE_ACQUIRE_RESULTS {
                assert_eq!(rx.acquires(result), 0, "seq {seq:?}: {result}");
            }
            if seq.is_none() {
                stage
                    .put_header(HashMap::from([("seq".to_owned(), "1".to_owned())]))
                    .await;
            }
        }
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. Commits wait during the read (SR-5.3, SR-5.4)
// ═════════════════════════════════════════════════════════════════════════════

/// A fast-path `EVALSHA` on `rx`'s object: the update script's shared `KEYS`
/// prefix puts the header at `UPDATE_SCRIPT_KEY_HEADER` and the fence at
/// `UPDATE_SCRIPT_KEY_FENCE` (`args[0]` is the command, `[1]` the SHA, `[2]`
/// the key count).
fn is_fast_path_eval(header: String, fence: String) -> impl Fn(&MonitoredCommand) -> bool {
    move |c| {
        c.name() == "EVALSHA"
            && c.args.get(2 + UPDATE_SCRIPT_KEY_HEADER) == Some(&header)
            && c.args.get(2 + UPDATE_SCRIPT_KEY_FENCE) == Some(&fence)
    }
}

/// A background `update_layer` call: its `(seq, server_ts)` or its error.
type WriterTask = tokio::task::JoinHandle<Result<(u64, u64), ServiceError>>;

/// The writer's `live_edit_update_layer_fence_wait_seconds` count: one
/// observation per `update_layer` call that met a fence.
fn fence_waits(rx: &Rx) -> u64 {
    rx.histogram_count(
        "live_edit_update_layer_fence_wait_seconds",
        &[("backend", "lua")],
    )
}

/// A commit started once the lease is granted is refused — the store's own
/// feed shows its script ran, the writer records that it met the fence, and
/// the header has not moved when the read is about to release — and commits
/// after the release. The read is proven at the grant's `seq`: its state is
/// the state before the commit, and the commit lands at the next `seq`.
///
/// One fast-path commit lands before the tap starts, so the fast-path script
/// is loaded and the `EVALSHA` the tap waits for is the writer's real attempt,
/// not a `NOSCRIPT` miss that never read the fence.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn commit_during_a_fenced_read_waits_and_is_excluded_from_the_read_at_the_grants_seq() {
    within_test_budget(async {
        let rx = Rx::new("srf-commit-waits").await;
        let writer_store = layer_store(&rx.tv).with_metrics(Arc::clone(&rx.metrics));
        commit(
            &writer_store,
            &rx.session_id,
            &rx.layer,
            rx.oid,
            delta_with_set_field_inline(WORLD, "warm", vec![1], 0),
        )
        .await;
        assert_eq!(
            fence_waits(&rx),
            0,
            "fixture precondition: the warm-up commit met no fence"
        );
        let estimate = rx.estimate().await;
        let before = rx.atomic().await;
        let seq_before = rx.seq().await;
        let tap = Arc::new(tokio::sync::Mutex::new(
            MonitorTap::start(&test_valkey_addr()).await,
        ));
        let writer: Arc<StdMutex<Option<WriterTask>>> = Arc::default();
        let seq_at_release = Arc::new(StdMutex::new(None::<u64>));

        let (store, session_id, layer, oid) = (
            writer_store.clone(),
            rx.session_id.clone(),
            rx.layer.clone(),
            rx.oid,
        );
        let (stage, slot, at_release, feed) = (
            rx.stage(),
            Arc::clone(&writer),
            Arc::clone(&seq_at_release),
            Arc::clone(&tap),
        );
        // On the first attempt only: a read whose first attempt let the commit
        // through would retry, and must not meet a second writer there.
        let (mut hook, log) = TestHook::new(react(move |attempt, event| {
            let (store, session_id, layer) = (store.clone(), session_id.clone(), layer.clone());
            let (stage, slot, at_release, feed) = (
                stage.clone(),
                Arc::clone(&slot),
                Arc::clone(&at_release),
                Arc::clone(&feed),
            );
            async move {
                match (attempt, event) {
                    (1, HookEvent::Grant(_)) => {
                        let handle = tokio::spawn(async move {
                            store
                                .update_layer_relayed(
                                    &session_id,
                                    &layer,
                                    oid,
                                    Some(delta_with_set_field_inline(WORLD, "late", vec![7], 0)),
                                    "srf-writer",
                                    1_000,
                                )
                                .await
                        });
                        *slot.lock().unwrap() = Some(handle);
                        feed.lock()
                            .await
                            .drain_until_nth(
                                1,
                                "the writer's fast-path EVALSHA",
                                is_fast_path_eval(stage.header.clone(), stage.fence.clone()),
                            )
                            .await;
                    }
                    (1, HookEvent::BeforeRelease) => {
                        let seq: String = stage
                            .client
                            .pool
                            .hget(&stage.header, "seq")
                            .await
                            .expect("HGET seq must succeed");
                        *at_release.lock().unwrap() = seq.parse().ok();
                    }
                    _ => {}
                }
                Ok(())
            }
        }));

        let fenced = verified(rx.read(estimate, &mut hook).await, "commit during the read");

        assert_eq!(
            *seq_at_release.lock().unwrap(),
            Some(seq_before),
            "the writer's attempt ran while the lease was held and must have landed nothing \
             before the release"
        );
        assert_eq!(names_without_retries(&log), held_attempts(1));
        assert_eq!(
            (grants(&log)[0].cut, fenced.cut),
            (seq_before, seq_before),
            "the read is proven at the seq acquire read, before the commit"
        );
        assert_same_read(&fenced, &before, "commit during the read");

        let handle = writer
            .lock()
            .unwrap()
            .take()
            .expect("the writer was started");
        let (seq, _) = tokio::time::timeout(CALL_BUDGET, handle)
            .await
            .expect("the writer must settle within its budget")
            .expect("the writer task must not panic")
            .unwrap_or_else(|e| panic!("the writer must commit after the release; got {e:?}"));
        assert_eq!(seq, seq_before + 1, "the commit lands after the read's cut");
        assert_eq!(rx.atomic().await.cut, seq_before + 1);
        assert_eq!(
            fence_waits(&rx),
            1,
            "the writer's call met the read's live fence and waited it out"
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. The proof (SR-5.3)
// ═════════════════════════════════════════════════════════════════════════════

/// A non-commit mutator (the fence does not stop it) moves the header `seq`
/// after the grant and moves it back before the release: the pass read the
/// moved header, the release finds the lease live and the header at the cut
/// and answers `VERIFIED` — but the pass did not read the cut's state, so the
/// attempt is broken and retried, and the second attempt is served. The hook
/// hears `Broken` for the first attempt, whatever its release answered: a hook
/// keying its once-per-read work on the verdict would run it twice.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_pass_that_read_another_seq_is_broken_even_when_the_release_verifies() {
    within_test_budget(async {
        let rx = Rx::new("srf-pass-seq").await;
        let estimate = rx.estimate().await;
        let atomic = rx.atomic().await;
        let cut = atomic.cut;
        let stage = rx.stage();
        let (mut hook, log) = TestHook::new(react(move |attempt, event| {
            let stage = stage.clone();
            async move {
                match (attempt, event) {
                    (1, HookEvent::Grant(_)) => stage.set_seq(cut + 1).await,
                    (1, HookEvent::BeforeRelease) => stage.set_seq(cut).await,
                    _ => {}
                }
                Ok(())
            }
        }));

        let fenced = verified(
            rx.read(estimate, &mut hook).await,
            "seq moved under the pass",
        );

        assert_eq!(
            releases(&log)[0].verdict,
            Some(ReleaseVerdict::Verified),
            "fixture precondition: the first release found the lease live and the header at the \
             cut"
        );
        assert_eq!(
            names_without_retries(&log),
            held_attempts(2),
            "the first pass read seq {} rather than the cut {cut}: never served, retried",
            cut + 1
        );
        assert_served_by_the_last_attempt(&log, &fenced, 2, "seq moved under the pass");
        assert_same_read(&fenced, &atomic, "the retry");
        assert_eq!(
            (
                rx.reads("catchup", "fence_broken"),
                rx.reads("catchup", "ok")
            ),
            (1, 1)
        );
    })
    .await;
}

/// The release's own proof: the header moves after the pass read it, before
/// the release. The pass read the cut, so only the release's `seq` check can
/// catch it: it answers `BROKEN`, and the retry is served at the new `seq`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_whose_header_moved_before_its_release_is_broken_and_retried_at_the_new_seq() {
    within_test_budget(async {
        let rx = Rx::new("srf-release-seq").await;
        let estimate = rx.estimate().await;
        let cut = rx.seq().await;
        let stage = rx.stage();
        let (mut hook, log) = TestHook::new(react(move |attempt, event| {
            let stage = stage.clone();
            async move {
                if (attempt, event) == (1, HookEvent::BeforeRelease) {
                    stage.set_seq(cut + 1).await;
                }
                Ok(())
            }
        }));

        let fenced = verified(
            rx.read(estimate, &mut hook).await,
            "seq moved before release",
        );

        assert_eq!(
            releases(&log)[0].verdict,
            Some(ReleaseVerdict::Broken),
            "the release must catch a header that moved after the pass read it"
        );
        assert_eq!(names_without_retries(&log), held_attempts(2));
        assert_eq!(
            (grants(&log)[1].cut, fenced.cut),
            (cut + 1, cut + 1),
            "the retry is proven at the seq its own acquire read"
        );
        assert_served_by_the_last_attempt(&log, &fenced, 2, "seq moved before release");
        assert_eq!(rx.reads("catchup", "fence_broken"), 1);
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. Restore and cleanup under the read (SR-5.3, amended 2026-10-03)
// ═════════════════════════════════════════════════════════════════════════════
//
// A pass that finds no header is `NotFound` only when the release confirms it.
// A release that finds a header again — `BROKEN` or `VERIFIED` — means a
// restore ran under the read: a broken read, retried. A release that finds no
// header ends the read `NotFound` whatever the pass found (a pass that failed
// otherwise: `seams`). A release that fails proves nothing, so the read ends
// with the release's error whether its pass found no header or completed.

/// A restore wipe during the read: the header, fence and intent keys
/// `UNLINK`ed after the grant (the pass finds no header), the header written
/// back before the release (which finds no lease field and answers `BROKEN`).
/// The read is retried on the restored object and served; nothing about it is
/// `NotFound`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_across_a_restore_wipe_the_release_finds_broken_is_retried_not_not_found() {
    within_test_budget(async {
        let rx = Rx::new("srf-restore-broken").await;
        let estimate = rx.estimate().await;
        let atomic = rx.atomic().await;
        let stage = rx.stage();
        let saved = Arc::new(StdMutex::new(HashMap::new()));
        let (mut hook, log) = TestHook::new(react(move |attempt, event| {
            let (stage, saved) = (stage.clone(), Arc::clone(&saved));
            async move {
                match (attempt, event) {
                    (1, HookEvent::Grant(_)) => *saved.lock().unwrap() = stage.wipe().await,
                    (1, HookEvent::BeforeRelease) => {
                        let header = saved.lock().unwrap().clone();
                        stage.put_header(header).await;
                    }
                    _ => {}
                }
                Ok(())
            }
        }));

        let result = rx.read(estimate, &mut hook).await;

        assert_eq!(
            releases(&log)[0],
            FenceReleased {
                verdict: Some(ReleaseVerdict::Broken),
                outcome: FenceAttemptOutcome::Broken,
                measured: None,
            },
            "the pass found no header and the release found the header back without the lease: \
             a broken attempt"
        );
        let fenced = verified(result, "restore wipe under the read");
        assert_eq!(names_without_retries(&log), held_attempts(2));
        assert_served_by_the_last_attempt(&log, &fenced, 2, "restore wipe under the read");
        assert_same_read(&fenced, &atomic, "the retry on the restored object");
        assert_eq!(
            (
                rx.reads("catchup", "fence_broken"),
                rx.reads("catchup", "not_found"),
                rx.reads("catchup", "ok"),
            ),
            (1, 0, 1),
            "the broken attempt is counted fence_broken, never not_found"
        );
    })
    .await;
}

/// The header taken away after the grant and put back, unchanged, before the
/// release: the pass found no header, the release finds the lease live and
/// the header at the cut and answers `VERIFIED`. The object exists, so this is
/// a broken read, retried — not `NotFound`, and not served.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_whose_pass_missed_a_header_the_release_verifies_is_retried_not_not_found() {
    within_test_budget(async {
        let rx = Rx::new("srf-restore-verified").await;
        let estimate = rx.estimate().await;
        let atomic = rx.atomic().await;
        let stage = rx.stage();
        let saved = Arc::new(StdMutex::new(HashMap::new()));
        let (mut hook, log) = TestHook::new(react(move |attempt, event| {
            let (stage, saved) = (stage.clone(), Arc::clone(&saved));
            async move {
                match (attempt, event) {
                    (1, HookEvent::Grant(_)) => *saved.lock().unwrap() = stage.take_header().await,
                    (1, HookEvent::BeforeRelease) => {
                        let header = saved.lock().unwrap().clone();
                        stage.put_header(header).await;
                    }
                    _ => {}
                }
                Ok(())
            }
        }));

        let result = rx.read(estimate, &mut hook).await;

        assert_eq!(
            releases(&log)[0],
            FenceReleased {
                verdict: Some(ReleaseVerdict::Verified),
                outcome: FenceAttemptOutcome::Broken,
                measured: None,
            },
            "the pass found no header and the release verified: a broken attempt, not the \
             served one"
        );
        let fenced = verified(result, "header missed by the pass");
        assert_eq!(names_without_retries(&log), held_attempts(2));
        assert_served_by_the_last_attempt(&log, &fenced, 2, "header missed by the pass");
        assert_same_read(&fenced, &atomic, "the retry");
        assert_eq!(
            (
                rx.reads("catchup", "fence_broken"),
                rx.reads("catchup", "not_found")
            ),
            (1, 0)
        );
    })
    .await;
}

/// `DeleteLayer` retires the object under the read: the pass finds no live
/// header and the release, finding none either, answers `NOT_FOUND`. The read
/// is `NotFound` after that one attempt — not retried, not counted broken.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_of_an_object_deleted_under_it_is_not_found_after_one_attempt() {
    within_test_budget(async {
        let rx = Rx::new("srf-cleanup").await;
        let estimate = rx.estimate().await;
        let (ls, session_id, layer) = (rx.ls.clone(), rx.session_id.clone(), rx.layer.clone());
        let (mut hook, log) = TestHook::new(react(move |attempt, event| {
            let (ls, session_id, layer) = (ls.clone(), session_id.clone(), layer.clone());
            async move {
                if let (1, HookEvent::Grant(_)) = (attempt, event) {
                    ls.delete_layer(&session_id, &layer, mock_ext_store())
                        .await
                        .expect("DeleteLayer must succeed under the read");
                }
                Ok(())
            }
        }));

        let err = rx.read(estimate, &mut hook).await.err();

        assert!(
            matches!(err, Some(ServiceError::NotFound(_))),
            "a pass and a release that both find no header: NotFound; got {err:?}"
        );
        assert_eq!(
            releases(&log),
            vec![FenceReleased {
                verdict: Some(ReleaseVerdict::NotFound),
                outcome: FenceAttemptOutcome::Ended,
                measured: None,
            }],
            "one attempt, whose release confirmed the header gone: the read ended there"
        );
        assert_eq!(
            names_without_retries(&log),
            held_attempts(1),
            "a confirmed NotFound is not retried"
        );
        assert_eq!(
            (
                rx.reads("catchup", "not_found"),
                rx.reads("catchup", "fence_broken")
            ),
            (1, 0)
        );
    })
    .await;
}

/// The header removed after the pass completed, before the release: the
/// release finds no header and answers `NOT_FOUND`, and the read is `NotFound`
/// after that one attempt — the pass's complete state is never served, and the
/// attempt is neither retried nor counted broken. The release was still handed
/// the size the pass measured.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_whose_release_finds_no_header_after_a_complete_pass_is_not_found_unserved() {
    within_test_budget(async {
        let rx = Rx::new("srf-gone-at-release").await;
        let estimate = rx.estimate().await;
        let measured = rx.atomic().await.measured_size();
        let stage = rx.stage();
        let (mut hook, log) = TestHook::new(react(move |attempt, event| {
            let stage = stage.clone();
            async move {
                if (attempt, event) == (1, HookEvent::BeforeRelease) {
                    stage.take_header().await;
                }
                Ok(())
            }
        }));

        let err = rx.read(estimate, &mut hook).await.err();

        assert!(
            matches!(err, Some(ServiceError::NotFound(_))),
            "a release that finds no header ends the read NotFound, whatever the pass read; got \
             {err:?}"
        );
        assert_eq!(
            releases(&log),
            vec![FenceReleased {
                verdict: Some(ReleaseVerdict::NotFound),
                outcome: FenceAttemptOutcome::Ended,
                measured: Some(measured),
            }],
            "one attempt: the pass completed, the release found the header gone"
        );
        assert_eq!(
            names_without_retries(&log),
            held_attempts(1),
            "a release's NotFound is not retried"
        );
        assert_eq!(
            (
                rx.reads("catchup", "not_found"),
                rx.reads("catchup", "fence_broken"),
                rx.reads("catchup", "ok"),
            ),
            (1, 0, 0)
        );
    })
    .await;
}

/// The read through a proxy that answers every release `EVALSHA` with an error
/// reply (every other command reaches the store), with the header taken away
/// after the grant when `pass_finds_no_header`. Either way the release call
/// fails, and the read ends with the release's classified error — a permanent
/// one, `Internal`, naming the release script — after one attempt: never
/// served, never `NotFound`, never retried.
async fn release_fails(pass_finds_no_header: bool) {
    let sha = release_sha();
    let rule: ProxyRule = Arc::new(move |args: &[Vec<u8>]| {
        is_evalsha_of(args, &sha)
            .then(|| ProxyAction::Eval(error_reply_script("ERR injected release failure")))
    });
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let tv = TestValkey::setup().await;
    let rx = Rx::with_reader(tv, client_through(&proxy).await, "srf-release-fails").await;
    let estimate = PROXIED_ESTIMATE;
    let measured = rx.atomic().await.measured_size();
    let stage = rx.stage();
    let (mut hook, log) = TestHook::new(react(move |attempt, event| {
        let stage = stage.clone();
        async move {
            if pass_finds_no_header && (attempt, event.name()) == (1, "on_grant") {
                stage.take_header().await;
            }
            Ok(())
        }
    }));

    let result = rx.read(estimate, &mut hook).await;

    assert!(proxy.hits() >= 1, "the proxy must have failed the release");
    assert_eq!(
        releases(&log),
        vec![FenceReleased {
            verdict: None,
            outcome: FenceAttemptOutcome::Ended,
            measured: (!pass_finds_no_header).then_some(measured),
        }],
        "one attempt, whose release call failed"
    );
    let err = result.err();
    assert!(
        matches!(&err, Some(ServiceError::Internal(m)) if m.contains("snapshot_fence_release")),
        "the read ends with the release's classified error — never the pass's state, never the \
         pass's NotFound; got {err:?}"
    );
    assert_eq!(names_without_retries(&log), held_attempts(1));
    assert_eq!(
        (
            rx.reads("catchup", "permanent_error"),
            rx.reads("catchup", "not_found"),
            rx.reads("catchup", "ok"),
        ),
        (1, 0, 0)
    );
}

/// A pass that found no header, with a release call that failed: the read
/// cannot tell whether the object is gone, so it ends with the release's
/// error, not `NotFound`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_whose_pass_found_no_header_and_whose_release_failed_ends_with_the_release_error(
) {
    within_test_budget(release_fails(true)).await;
}

/// A pass that completed, with a release call that failed: nothing proved the
/// pass, so its state is not served; the read ends with the release's error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_whose_pass_completed_and_whose_release_failed_ends_with_the_release_error() {
    within_test_budget(release_fails(false)).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 6. Attempts (SR-5.3: never an unverified state)
// ═════════════════════════════════════════════════════════════════════════════

/// Every attempt is broken (a mutator moves the header after each grant):
/// after `FENCE_READ_ATTEMPTS` the read is `Unavailable` — retryable — and no
/// attempt's unverified state is ever returned.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_broken_on_every_attempt_is_unavailable_after_three_attempts() {
    within_test_budget(async {
        let rx = Rx::new("srf-exhausted").await;
        let estimate = rx.estimate().await;
        let stage = rx.stage();
        let (mut hook, log) = TestHook::new(react(move |_, event| {
            let stage = stage.clone();
            async move {
                if let HookEvent::Grant(_) = event {
                    stage.incr_seq().await;
                }
                Ok(())
            }
        }));

        let err = rx.read(estimate, &mut hook).await.err();

        assert_eq!(FENCE_READ_ATTEMPTS, 3);
        assert!(
            matches!(err, Some(ServiceError::Unavailable(_))),
            "three broken reads are a retryable Unavailable, never a served state; got {err:?}"
        );
        assert_eq!(
            names_without_retries(&log),
            held_attempts(3),
            "exactly FENCE_READ_ATTEMPTS attempts"
        );
        assert!(
            releases(&log)
                .iter()
                .all(|r| r.verdict == Some(ReleaseVerdict::Broken)),
            "every release saw the moved header: {:?}",
            releases(&log)
        );
        assert_eq!(
            outcomes(&log),
            vec![FenceAttemptOutcome::Broken; 3],
            "no attempt is ever Verified: the read served nothing"
        );
        assert_eq!(
            (
                rx.reads("catchup", "fence_broken"),
                rx.reads("catchup", "ok")
            ),
            (3, 0)
        );
        assert!(rx.readers().await.is_empty(), "every lease was released");
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 7. The hook (SR-5.6's acquire cycle)
// ═════════════════════════════════════════════════════════════════════════════

/// The events of a broken attempt and its retry, in order, with what each
/// carries: `before_attempt` (attempt 1, the caller's estimate) → `on_grant`
/// (the cut) → `before_release` → `after_release` (`BROKEN`, outcome
/// `Broken`, the measured size) → `before_attempt` (attempt 2, now sized from
/// what the broken read measured; it follows a broken read) → any
/// `RETRY_AFTER`s of the writers' gap the broken release opened (under the
/// size it recorded) → `on_grant` → `before_release` → `after_release`
/// (`VERIFIED`, outcome `Verified`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_tells_its_hook_each_step_in_order_across_a_broken_attempt() {
    within_test_budget(async {
        let rx = Rx::new("srf-hook-order").await;
        let estimate = rx.estimate().await;
        let cut = rx.seq().await;
        let measured = rx.atomic().await.measured_size();
        assert_ne!(
            measured, estimate,
            "fixture precondition: the estimate and the measured size differ, so a mix-up shows"
        );
        let stage = rx.stage();
        let (mut hook, log) = TestHook::new(react(move |attempt, event| {
            let stage = stage.clone();
            async move {
                if (attempt, event) == (1, HookEvent::BeforeRelease) {
                    stage.set_seq(cut + 1).await;
                }
                Ok(())
            }
        }));

        verified(rx.read(estimate, &mut hook).await, "hook order");

        let all = events(&log);
        let (first, rest) = all.split_at(4);
        assert_eq!(
            first,
            [
                HookEvent::BeforeAttempt(FenceAttempt {
                    attempt: 1,
                    size: estimate
                }),
                HookEvent::Grant(FenceGrant {
                    cut,
                    kind: GrantKind::StartedEpoch,
                    size: estimate,
                }),
                HookEvent::BeforeRelease,
                HookEvent::AfterRelease(FenceReleased {
                    verdict: Some(ReleaseVerdict::Broken),
                    outcome: FenceAttemptOutcome::Broken,
                    measured: Some(measured),
                }),
            ],
            "attempt 1: before_attempt → on_grant → before_release → after_release"
        );
        let second = FenceAttempt {
            attempt: 2,
            size: measured,
        };
        assert!(second.follows_broken_read());
        assert_eq!(
            rest[0],
            HookEvent::BeforeAttempt(second),
            "BROKEN → before_attempt"
        );
        let retries = rest[1..]
            .iter()
            .take_while(|e| matches!(e, HookEvent::RetryAfter(_)))
            .count();
        for event in &rest[1..1 + retries] {
            let HookEvent::RetryAfter(retry) = event else {
                unreachable!()
            };
            assert_eq!(
                (retry.reason, retry.size),
                (RetryReason::Gap, measured),
                "a retry waits only for the writers' gap, under the size the release recorded"
            );
        }
        assert_eq!(
            &rest[1 + retries..],
            [
                HookEvent::Grant(FenceGrant {
                    cut: cut + 1,
                    kind: GrantKind::StartedEpoch,
                    size: measured,
                }),
                HookEvent::BeforeRelease,
                HookEvent::AfterRelease(FenceReleased {
                    verdict: Some(ReleaseVerdict::Verified),
                    outcome: FenceAttemptOutcome::Verified,
                    measured: Some(measured),
                }),
            ],
            "attempt 2: on_grant → before_release → after_release"
        );
    })
    .await;
}

/// A `before_attempt` refusal ends the read with exactly that error before any
/// acquire: no fence key, no read counted, no wait observed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_ends_with_the_hooks_before_attempt_error_before_acquiring() {
    within_test_budget(async {
        let rx = Rx::new("srf-hook-refuses").await;
        let estimate = rx.estimate().await;
        let (mut hook, log) = TestHook::new(react(|_, event| async move {
            match event {
                HookEvent::BeforeAttempt(_) => Err(ServiceError::Unavailable(
                    "test hook: no capacity".to_owned(),
                )),
                _ => Ok(()),
            }
        }));

        let err = rx.read(estimate, &mut hook).await.err();

        assert!(
            matches!(&err, Some(ServiceError::Unavailable(m)) if m == "test hook: no capacity"),
            "the hook's own error, as it is; got {err:?}"
        );
        assert_eq!(names_without_retries(&log), vec!["before_attempt"]);
        assert!(!rx.exists(&rx.fence_key()).await, "nothing was acquired");
        for outcome in [
            "ok",
            "not_found",
            "fence_broken",
            "transient_error",
            "permanent_error",
        ] {
            assert_eq!(
                rx.reads("catchup", outcome),
                0,
                "an attempt the hook ended is not a read"
            );
        }
        assert_eq!(
            rx.histogram_count(
                "live_edit_snapshot_fence_reader_wait_seconds",
                &[("consumer", "catchup")]
            ),
            0,
            "no acquire was called, so no wait is observed"
        );
    })
    .await;
}

/// Stage a writers' gap on `rx`'s fence ending `ahead_ms` after the store's
/// clock, with no reader live: acquire answers `RETRY_AFTER gap` until then.
async fn stage_gap(rx: &Rx, ahead_ms: i64, size: LayerSize) {
    let until = rx.now().await + ahead_ms;
    let _: () = rx
        .tv
        .client()
        .pool
        .hset(
            rx.fence_key(),
            [
                ("gap_until", until.to_string()),
                ("size_e", size.items.to_string()),
                ("size_b", size.bytes.to_string()),
            ],
        )
        .await
        .expect("staging the gap must succeed");
    let _: i64 = rx
        .tv
        .client()
        .pool
        .pexpire(rx.fence_key(), FAR_MS * 2, None)
        .await
        .expect("staging the key expiry must succeed");
}

/// An `on_retry_after` refusal ends the read with exactly that error: the
/// `RETRY_AFTER` is counted, the wait observed once, no read counted, and the
/// fence key keeps the gap it had (a wait writes nothing).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_ends_with_the_hooks_retry_after_error() {
    within_test_budget(async {
        let rx = Rx::new("srf-hook-retry-refuses").await;
        let estimate = rx.estimate().await;
        stage_gap(&rx, FAR_MS, estimate).await;
        let staged = rx.fields(&rx.fence_key()).await;
        let (mut hook, log) = TestHook::new(react(|_, event| async move {
            match event {
                HookEvent::RetryAfter(_) => {
                    Err(ServiceError::Unavailable("test hook: wait cap".to_owned()))
                }
                _ => Ok(()),
            }
        }));

        let err = rx.read(estimate, &mut hook).await.err();

        assert!(
            matches!(&err, Some(ServiceError::Unavailable(m)) if m == "test hook: wait cap"),
            "got {err:?}"
        );
        let events = events(&log);
        assert!(
            matches!(
                events.as_slice(),
                [
                    HookEvent::BeforeAttempt(_),
                    HookEvent::RetryAfter(FenceRetry {
                        reason: RetryReason::Gap,
                        ..
                    })
                ]
            ),
            "before_attempt, then the one RETRY_AFTER the hook refused: {events:?}"
        );
        assert_eq!(
            rx.fields(&rx.fence_key()).await,
            staged,
            "a wait writes nothing"
        );
        assert_eq!(rx.acquires("retry_after_gap"), 1);
        assert_eq!(
            rx.histogram_count(
                "live_edit_snapshot_fence_reader_wait_seconds",
                &[("consumer", "catchup")]
            ),
            1,
            "the wait the hook ended is observed once"
        );
        for outcome in [
            "ok",
            "not_found",
            "fence_broken",
            "transient_error",
            "permanent_error",
        ] {
            assert_eq!(rx.reads("catchup", outcome), 0);
        }
    })
    .await;
}

/// `CappedFenceWait` sleeps a `RETRY_AFTER` inside its cap and the read then
/// goes on: here a writers' gap a few hundred ms long.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn capped_fence_wait_sleeps_a_short_retry_after_and_the_read_goes_on() {
    within_test_budget(async {
        let rx = Rx::new("srf-capped-sleeps").await;
        let estimate = rx.estimate().await;
        stage_gap(&rx, 300, estimate).await;
        let mut wait = CappedFenceWait::new(&rx.reader, rx.oid, SnapshotReadConsumer::Checkpoint);

        let fenced = verified(
            rx.read_as(SnapshotReadConsumer::Checkpoint, estimate, &mut wait)
                .await,
            "short gap",
        );

        assert_same_read(&fenced, &rx.atomic().await, "after the gap");
        assert!(rx.acquires("retry_after_gap") >= 1, "the read met the gap");
        assert_eq!(rx.acquires("started_epoch"), 1);
        assert_eq!(rx.reads("checkpoint", "ok"), 1);
    })
    .await;
}

/// `CappedFenceWait` refuses a `RETRY_AFTER` whose sleep would pass
/// `reader_wait_max(L)` at once — without sleeping it — with a retryable
/// `Unavailable`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn capped_fence_wait_refuses_a_retry_after_past_reader_wait_max() {
    within_test_budget(async {
        let rx = Rx::new("srf-capped-refuses").await;
        let estimate = rx.estimate().await;
        let wait = CappedFenceWait::new(&rx.reader, rx.oid, SnapshotReadConsumer::Checkpoint);
        assert!(
            wait.cap(estimate) < Duration::from_millis(FAR_MS as u64),
            "fixture precondition: the staged gap is past the cap"
        );
        stage_gap(&rx, FAR_MS, estimate).await;
        let mut wait = wait;

        let err = rx
            .read_as(SnapshotReadConsumer::Checkpoint, estimate, &mut wait)
            .await
            .err();

        assert!(
            matches!(&err, Some(ServiceError::Unavailable(m)) if m.contains("reader_wait_max")),
            "a wait past the cap is refused, retryable; got {err:?}"
        );
        assert_eq!(
            rx.acquires("retry_after_gap"),
            1,
            "refused at the first RETRY_AFTER"
        );
    })
    .await;
}

/// `CappedFenceWait::before_attempt` refuses the attempt after a broken read
/// once the read has waited past `reader_wait_max(L)`, with `L` the size that
/// attempt carries — and allows it up to the cap. Driven on the hook directly
/// under tokio's paused clock: through a read, getting past the cap (~29 s at
/// a small layer) would take a cap-length real wait. The client only supplies
/// the fence coefficients; it is connected before the clock is paused.
#[tokio::test(flavor = "current_thread")]
async fn capped_fence_wait_refuses_an_attempt_after_a_broken_read_once_past_reader_wait_max() {
    within_test_budget(async {
        let client = client_with_limits(LOW_ITEMS, LOW_BYTES).await;
        let small = LayerSize {
            items: 1_000,
            bytes: 12_000,
        };
        let large = LayerSize {
            items: 1_000_000,
            bytes: 12_000_000,
        };
        let mut wait = CappedFenceWait::new(&client, 1, SnapshotReadConsumer::Checkpoint);
        let (cap, large_cap) = (wait.cap(small), wait.cap(large));
        assert!(
            large_cap > cap + Duration::from_secs(1),
            "fixture precondition: a larger layer has a longer cap ({cap:?} vs {large_cap:?})"
        );
        let attempt = |attempt, size| FenceAttempt { attempt, size };
        tokio::time::pause();

        wait.before_attempt(attempt(1, small))
            .await
            .expect("the first attempt starts the clock and is never refused");
        tokio::time::advance(cap - Duration::from_millis(1)).await;
        wait.before_attempt(attempt(2, small))
            .await
            .expect("a retry within reader_wait_max(L) goes ahead");
        tokio::time::advance(Duration::from_millis(2)).await;
        wait.before_attempt(attempt(3, large)).await.expect(
            "past the small layer's cap but within the large one's: L is the attempt's size",
        );
        let err = wait.before_attempt(attempt(3, small)).await.err();

        assert!(
            matches!(&err, Some(ServiceError::Unavailable(m)) if m.contains("reader_wait_max")),
            "a retry after a broken read past reader_wait_max(L) is refused, retryable; got {err:?}"
        );
    })
    .await;
}

/// A read inside a live epoch's join window joins it: granted at once, with
/// the epoch's cut, and counted `joined_epoch`; the other reader's lease is
/// left live.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_inside_a_live_epochs_join_window_joins_it() {
    within_test_budget(async {
        let rx = Rx::new("srf-joins").await;
        let estimate = rx.estimate().await;
        let other = match fence::acquire_fence(&rx.reader, rx.keys(), rx.oid, estimate)
            .await
            .expect("the other reader's acquire must answer")
        {
            AcquireVerdict::Granted(lease) => lease,
            other => panic!("the other reader must be granted; got {other:?}"),
        };
        let (mut hook, log) = TestHook::new(no_reaction());

        let fenced = verified(rx.read(estimate, &mut hook).await, "joined epoch");

        assert_eq!(
            grants(&log),
            vec![FenceGrant {
                cut: other.cut(),
                kind: GrantKind::JoinedEpoch,
                size: estimate,
            }]
        );
        assert_eq!(fenced.cut, other.cut());
        assert_eq!(
            (rx.acquires("joined_epoch"), rx.acquires("started_epoch")),
            (1, 0)
        );
        assert_eq!(
            rx.readers().await.len(),
            1,
            "only the other reader's field is left"
        );
        assert_eq!(
            other.release(None).await.expect("release must answer"),
            ReleaseVerdict::Verified
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 8. Metrics (SR-8.1, SR-8.2, SR-8.3)
// ═════════════════════════════════════════════════════════════════════════════

/// A read broken once and served on its retry, for the checkpoint consumer:
/// one `snapshot_read_total{path="fenced"}` tick per attempt (`fence_broken`,
/// then `ok`), the duration histogram once per attempt, the size histograms
/// once with the served read's tallies, the hold and reader-wait histograms
/// once per attempt, and one acquire tick per acquire reply. Nothing on the
/// atomic path, and nothing under another consumer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_records_its_metrics_once_per_attempt_under_its_consumer() {
    within_test_budget(async {
        let rx = Rx::new("srf-metrics").await;
        let estimate = rx.estimate().await;
        let atomic_before = rx.metric(
            "live_edit_snapshot_read_total",
            &[
                ("consumer", "checkpoint"),
                ("path", "atomic"),
                ("outcome", "ok"),
            ],
        );
        let cut = rx.seq().await;
        let stage = rx.stage();
        let (mut hook, log) = TestHook::new(react(move |attempt, event| {
            let stage = stage.clone();
            async move {
                if (attempt, event) == (1, HookEvent::BeforeRelease) {
                    stage.set_seq(cut + 1).await;
                }
                Ok(())
            }
        }));

        let fenced = verified(
            rx.read_as(SnapshotReadConsumer::Checkpoint, estimate, &mut hook)
                .await,
            "metrics",
        );

        let retries = events(&log)
            .iter()
            .filter(|e| matches!(e, HookEvent::RetryAfter(_)))
            .count() as u64;
        assert_eq!(
            (
                rx.reads("checkpoint", "fence_broken"),
                rx.reads("checkpoint", "ok"),
                rx.reads("catchup", "ok"),
            ),
            (1, 1, 0),
            "one tick per attempt, under the reading consumer"
        );
        let consumer = [("consumer", "checkpoint")];
        assert_eq!(
            rx.histogram_count(
                "live_edit_snapshot_read_duration_seconds",
                &[("consumer", "checkpoint"), ("path", "fenced")]
            ),
            2,
            "the duration is recorded for every attempt"
        );
        assert_eq!(
            (
                rx.histogram_count("live_edit_snapshot_read_reply_bytes", &consumer),
                rx.metric("live_edit_snapshot_read_reply_bytes_sum", &consumer),
                rx.metric("live_edit_snapshot_read_entries_sum", &consumer),
            ),
            (1, fenced.reply_bytes as f64, fenced.entry_count as f64),
            "the size histograms record the served read only"
        );
        assert_eq!(
            (
                rx.histogram_count("live_edit_snapshot_fence_hold_seconds", &consumer),
                rx.histogram_count("live_edit_snapshot_fence_reader_wait_seconds", &consumer),
            ),
            (2, 2),
            "each attempt held one lease and waited for it once"
        );
        assert_eq!(
            (
                rx.acquires("started_epoch"),
                rx.acquires("retry_after_gap"),
                rx.acquires("joined_epoch"),
                rx.acquires("retry_after_epoch"),
                rx.acquires("retry_after_intent"),
            ),
            (2, retries, 0, 0, 0),
            "one acquire tick per reply: two grants, and the gap's RETRY_AFTERs the hook heard"
        );
        assert_eq!(
            rx.metric(
                "live_edit_snapshot_read_total",
                &[
                    ("consumer", "checkpoint"),
                    ("path", "atomic"),
                    ("outcome", "ok")
                ],
            ),
            atomic_before,
            "the fenced read never runs the atomic path"
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 9. Cancellation (SR-7.4)
// ═════════════════════════════════════════════════════════════════════════════
//
// The read is a spawned task, aborted at the step under test. A lease the read
// held is released in the background by its dropped guard: its field leaves the
// fence key and the epoch closes (a gap recorded, no `epoch_start` left). A
// lapsed lease would leave its field in place until the next grant prunes it,
// so a missing field proves the release ran.

/// Abort `read` and assert it was cancelled rather than finished.
async fn abort(read: tokio::task::JoinHandle<Result<AtomicSnapshot, ServiceError>>) {
    read.abort();
    let joined = tokio::time::timeout(SIGNAL_BUDGET, read)
        .await
        .expect("the aborted read must settle within its budget");
    assert!(
        matches!(&joined, Err(e) if e.is_cancelled()),
        "the read must have been cancelled mid-way, not finished: {:?}",
        joined.map(|r| r.map(|s| s.cut))
    );
}

/// The key image a lease released in the background leaves: no reader field,
/// the epoch closed.
async fn assert_released_in_the_background(rx: &Rx, what: &str) {
    rx.until_released(what).await;
    let fields = rx.fields(&rx.fence_key()).await;
    assert!(
        fields.contains_key("gap_until") && !fields.contains_key("epoch_start"),
        "{what}: the background release closed the epoch: {fields:?}"
    );
}

/// A reaction that fires `signal` on the event `at` names and then never
/// returns, so the read can be cancelled inside that event.
fn park_at(at: &'static str, signal: Signal) -> Reaction {
    react(move |_, event| {
        let signal = signal.clone();
        async move {
            if event.name() == at {
                signal.fire();
                futures::future::pending::<()>().await;
            }
            Ok(())
        }
    })
}

async fn cancelled_inside_the_hook(at: &'static str) {
    let rx = Rx::new("srf-cancel-hook").await;
    let estimate = rx.estimate().await;
    let signal = Signal::default();
    let (hook, _log) = TestHook::new(park_at(at, signal.clone()));

    let read = rx.spawn_read(estimate, hook);
    signal.wait(at).await;
    assert_eq!(
        rx.readers().await.len(),
        1,
        "precondition: the lease is held"
    );
    abort(read).await;

    assert_released_in_the_background(&rx, at).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_cancelled_in_on_grant_releases_its_lease_in_the_background() {
    within_test_budget(cancelled_inside_the_hook("on_grant")).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_cancelled_in_before_release_releases_its_lease_in_the_background() {
    within_test_budget(cancelled_inside_the_hook("before_release")).await;
}

/// A fence script's `EVALSHA` on `fence_key`, by its SHA.
fn is_script_on(sha: String, fence_key: String) -> impl Fn(&MonitoredCommand) -> bool {
    move |c| {
        !c.issued_by_script()
            && c.name() == "EVALSHA"
            && c.args.get(1) == Some(&sha)
            && c.args.get(3) == Some(&fence_key)
    }
}

/// Cancelled once the store has its acquire: whether or not the read saw the
/// grant, no lease is live after `lease(L)` — a seen grant's guard released
/// it, an unseen one lapses at its deadline.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_cancelled_at_its_acquire_leaves_no_live_lease_after_the_lease() {
    within_test_budget(async {
        let rx = Rx::new("srf-cancel-acquire").await;
        rx.warm_scripts().await;
        let estimate = rx.estimate().await;
        let mut tap = MonitorTap::start(&test_valkey_addr()).await;
        let (hook, _log) = TestHook::new(no_reaction());

        let read = rx.spawn_read(estimate, hook);
        tap.drain_until_nth(
            1,
            "the read's acquire EVALSHA",
            is_script_on(acquire_sha(), rx.fence_key()),
        )
        .await;
        read.abort();
        let _ = tokio::time::timeout(SIGNAL_BUDGET, read)
            .await
            .expect("the aborted read must settle within its budget");

        assert!(
            rx.exists(&rx.fence_key()).await,
            "precondition: the acquire granted (it wrote the fence key)"
        );
        rx.until_no_live_lease("cancelled at acquire").await;
    })
    .await;
}

/// Cancelled once the store has its release: the release ran, and a dropped
/// release future's guard releases again harmlessly; either way no field is
/// left.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_cancelled_at_its_release_leaves_no_lease() {
    within_test_budget(async {
        let rx = Rx::new("srf-cancel-release").await;
        rx.warm_scripts().await;
        let estimate = rx.estimate().await;
        let mut tap = MonitorTap::start(&test_valkey_addr()).await;
        let signal = Signal::default();
        let fired = signal.clone();
        let (hook, _log) = TestHook::new(react(move |_, event| {
            let fired = fired.clone();
            async move {
                if event == HookEvent::BeforeRelease {
                    fired.fire();
                }
                Ok(())
            }
        }));

        let read = rx.spawn_read(estimate, hook);
        signal.wait("before_release").await;
        tap.drain_until_nth(
            1,
            "the read's release EVALSHA",
            is_script_on(release_sha(), rx.fence_key()),
        )
        .await;
        read.abort();
        let _ = tokio::time::timeout(SIGNAL_BUDGET, read)
            .await
            .expect("the aborted read must settle within its budget");

        rx.until_released("cancelled at release").await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 10. Seams: the renewal timer, cancellation mid-read (test-fault-injection)
// ═════════════════════════════════════════════════════════════════════════════

/// The fenced fan-out's test-only levers: a delay holding one command back
/// before it is issued, an error in place of its first command, and stalled
/// renewals. Compiled out — not skipped — without `test-fault-injection`.
#[cfg(feature = "test-fault-injection")]
mod seams {
    use super::*;

    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use fred::interfaces::ClientLike;
    use live_edit::storage::diff_read::{
        ObjectSnapshotReadFaultInjector, ObjectSnapshotReadFaultTarget, ObjectSnapshotReader,
    };
    use live_edit::storage::snapshot_fence::ms_to_duration;

    use crate::storage::snapshot_read_support::monitor_fence;

    /// The per-command timeout `client`'s data pool runs with (fred's
    /// `PerformanceConfig::default_command_timeout`, set from
    /// `valkey_command_timeout_ms` at connect): the longest one renewal on
    /// that client can take to reach the store and answer.
    fn command_timeout(client: &ValkeyClient) -> Duration {
        client.pool.next().perf_config().default_command_timeout
    }

    /// An estimate whose `lease(L)` on the reading client (about 8.1 s, its
    /// `hold(L)/4`) gives a renewal sent at `lease/4` `0.75·lease` (about
    /// 6 s) to land before the deadline it extends: more than the reader's
    /// [`command_timeout`] (5 s on the fixture).
    /// Far larger than the object: a fresh object's grant, and so its lease,
    /// is sized from the estimate, while the fan-out reads the object as it
    /// is.
    const SLOW_WINDOW_ESTIMATE: LayerSize = LayerSize {
        items: 8_500_000,
        bytes: 102_000_000,
    };

    /// `lease(L)` at `size`, from the reading client's coefficients.
    fn lease(rx: &Rx, size: LayerSize) -> Duration {
        ms_to_duration(fence::coefficients(&rx.reader).timing(size).lease_ms)
    }

    fn injector(rx: &Rx) -> Arc<ObjectSnapshotReadFaultInjector> {
        let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
        rx.reader
            .install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));
        injector
    }

    /// Stage the one live reader field's deadline [`FAR_MS`] ahead of the
    /// store's clock, and the fence key's expiry with it: a lease that stays
    /// live whatever its renewals do (a renewal never moves a deadline back),
    /// until it is released.
    async fn hold_the_lease(stage: &Stage) {
        let readers = readers_of(&key_fields(&stage.client, &stage.fence).await);
        assert_eq!(readers.len(), 1, "one reader holds the fence: {readers:?}");
        let field = readers.into_keys().next().expect("one reader field");
        let far = store_now(&stage.client, &stage.fence).await + FAR_MS;
        let _: () = stage
            .client
            .pool
            .hset(&stage.fence, (field, far.to_string()))
            .await
            .expect("staging the lease deadline must succeed");
        let _: i64 = stage
            .client
            .pool
            .pexpire(&stage.fence, FAR_MS, None)
            .await
            .expect("staging the key expiry must succeed");
    }

    /// The one reader field's deadline, read on grant, and the store clock and
    /// that field's deadline read before release.
    #[derive(Debug, Default, Clone, Copy)]
    struct Deadlines {
        granted: Option<i64>,
        now_at_release: Option<i64>,
        at_release: Option<i64>,
    }

    /// A reaction recording [`Deadlines`] on attempt 1.
    fn record_deadlines(rx: &Rx, into: Arc<StdMutex<Deadlines>>) -> Reaction {
        let stage = rx.stage();
        react(move |attempt, event| {
            let (stage, into) = (stage.clone(), Arc::clone(&into));
            async move {
                if attempt == 1 {
                    let deadline = || async {
                        readers_of(&key_fields(&stage.client, &stage.fence).await)
                            .into_values()
                            .next()
                    };
                    match event {
                        HookEvent::Grant(_) => into.lock().unwrap().granted = deadline().await,
                        HookEvent::BeforeRelease => {
                            let now = store_now(&stage.client, &stage.fence).await;
                            let at = deadline().await;
                            let mut d = into.lock().unwrap();
                            d.now_at_release = Some(now);
                            d.at_release = at;
                        }
                        _ => {}
                    }
                }
                Ok(())
            }
        })
    }

    /// A window held back for longer than the lease does not lose it: the
    /// renewal timer, running beside the fan-out, extends the lease while the
    /// window waits, so the release verifies on the first attempt. The store
    /// clock at the release is past the deadline the grant set — the window
    /// did outlast the lease as granted — and the lease's deadline then is
    /// later than that.
    ///
    /// The read brings [`SLOW_WINDOW_ESTIMATE`], whose lease leaves each
    /// renewal (sent every `lease/4`) more than the client's command timeout
    /// to land before the deadline it extends, so the lease is lost only if
    /// the timer stops, never because a loaded store answered a renewal late.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fenced_read_keeps_its_lease_across_a_window_held_longer_than_the_lease() {
        within_test_budget(async {
            let rx = Rx::new("srf-slow-window").await;
            let estimate = SLOW_WINDOW_ESTIMATE;
            let lease = lease(&rx, estimate);
            let command_timeout = command_timeout(&rx.reader);
            assert!(
                lease.mul_f64(0.75) > command_timeout,
                "fixture precondition: a renewal sent at lease/4 has more than the reader's \
                 command timeout ({command_timeout:?}) to land before the deadline it extends \
                 (lease {lease:?})"
            );
            injector(&rx).push_delay_at(ObjectSnapshotReadFaultTarget::next_read(), lease * 3 / 2);
            let deadlines = Arc::new(StdMutex::new(Deadlines::default()));
            let (mut hook, log) = TestHook::new(record_deadlines(&rx, Arc::clone(&deadlines)));

            let fenced = verified(rx.read(estimate, &mut hook).await, "slow window");

            assert_eq!(
                grants(&log).first().map(|g| g.size),
                Some(estimate),
                "fixture precondition: the grant's lease is sized from the estimate"
            );
            let d = *deadlines.lock().unwrap();
            let (granted, now, at_release) = (
                d.granted.expect("the grant's deadline"),
                d.now_at_release.expect("the clock at release"),
                d.at_release.expect("the deadline at release"),
            );
            assert!(
                now > granted,
                "fixture precondition: the held window outlasted the lease as granted (now {now}, \
                 granted deadline {granted})"
            );
            assert!(
                at_release > now,
                "the renewal timer kept the lease live through the window: deadline {at_release} \
                 at release, now {now}"
            );
            assert_eq!(
                names_without_retries(&log),
                held_attempts(1),
                "the lease held, so the first attempt was served"
            );
            assert_same_read(&fenced, &rx.atomic().await, "slow window");
            assert_eq!(rx.reads("catchup", "fence_broken"), 0);
        })
        .await;
    }

    /// Renewals stalled while a window is held past the lease: the lease lapses,
    /// the pass still completes, and the release answers `BROKEN` — the read is
    /// retried (renewals resumed) and served on its second attempt.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fenced_read_whose_renewals_stall_past_the_lease_is_broken_and_retried() {
        within_test_budget(async {
            let rx = Rx::new("srf-stalled").await;
            let estimate = rx.estimate().await;
            let seam = injector(&rx);
            seam.stall_fence_renewals(true).push_delay_at(
                ObjectSnapshotReadFaultTarget::next_read(),
                lease(&rx, estimate) * 3 / 2,
            );
            let deadlines = Arc::new(StdMutex::new(Deadlines::default()));
            let mut record = record_deadlines(&rx, Arc::clone(&deadlines));
            let resume = Arc::clone(&seam);
            let (mut hook, log) = TestHook::new(Box::new(move |attempt, event| {
                if attempt == 2 {
                    resume.stall_fence_renewals(false);
                }
                record(attempt, event)
            }));

            let fenced = verified(rx.read(estimate, &mut hook).await, "stalled renewals");

            let d = *deadlines.lock().unwrap();
            assert_eq!(
                d.at_release, d.granted,
                "fixture precondition: no renewal ran on attempt 1"
            );
            assert!(
                d.now_at_release > d.granted,
                "fixture precondition: the lease had lapsed by the release"
            );
            let first = releases(&log)[0];
            assert_eq!(
                first.verdict,
                Some(ReleaseVerdict::Broken),
                "a lapsed lease's release is BROKEN, though the pass completed"
            );
            assert!(first.measured.is_some(), "the pass completed");
            assert_eq!(names_without_retries(&log), held_attempts(2));
            assert_served_by_the_last_attempt(&log, &fenced, 2, "stalled renewals");
            assert_same_read(&fenced, &rx.atomic().await, "the retry");
            assert_eq!(
                (
                    rx.reads("catchup", "fence_broken"),
                    rx.reads("catchup", "ok")
                ),
                (1, 1)
            );
        })
        .await;
    }

    /// A renewal that answers `LOST` stops the fan-out: the first window is held
    /// back far longer than the read's budget, renewals stall until the lease
    /// has lapsed on the store's clock and then resume, and the next renewal's
    /// `LOST` drops the held window — the release is `BROKEN` with no measured
    /// size (the pass never completed) and the retry is served, all well within
    /// the read's budget, which the held window alone would overrun.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fenced_read_stops_its_fan_out_when_a_renewal_answers_lost() {
        within_test_budget(async {
            const HELD: Duration = Duration::from_secs(45);
            assert!(HELD > READ_BUDGET);
            let rx = Rx::new("srf-lost").await;
            let estimate = rx.estimate().await;
            let seam = injector(&rx);
            seam.stall_fence_renewals(true)
                .push_delay_at(ObjectSnapshotReadFaultTarget::next_read(), HELD);
            let stage = rx.stage();
            let resume = Arc::clone(&seam);
            // On attempt 1's grant, resume renewals once the lease's deadline
            // has passed on the store's clock: the next renewal then finds it
            // expired and answers LOST.
            let (mut hook, log) = TestHook::new(react(move |attempt, event| {
                let (stage, resume) = (stage.clone(), Arc::clone(&resume));
                async move {
                    if let (1, HookEvent::Grant(_)) = (attempt, event) {
                        let deadline = readers_of(&key_fields(&stage.client, &stage.fence).await)
                            .into_values()
                            .next()
                            .expect("the granted lease's deadline");
                        tokio::spawn(async move {
                            while store_now(&stage.client, &stage.fence).await <= deadline {
                                tokio::time::sleep(Duration::from_millis(50)).await;
                            }
                            resume.stall_fence_renewals(false);
                        });
                    }
                    Ok(())
                }
            }));

            let fenced = verified(rx.read(estimate, &mut hook).await, "LOST renewal");

            assert_eq!(
                releases(&log)[0],
                FenceReleased {
                    verdict: Some(ReleaseVerdict::Broken),
                    outcome: FenceAttemptOutcome::Broken,
                    measured: None,
                },
                "the LOST renewal stopped the pass before it completed"
            );
            assert_eq!(names_without_retries(&log), held_attempts(2));
            assert_served_by_the_last_attempt(&log, &fenced, 2, "LOST renewal");
            assert_eq!(
                seam.delays_remaining(),
                0,
                "the held window was the first attempt's"
            );
            assert_same_read(&fenced, &rx.atomic().await, "the retry");
            assert_eq!(rx.reads("catchup", "fence_broken"), 1);
        })
        .await;
    }

    /// Cancelled with a window in flight (its second command held back): the
    /// lease is released in the background, and the renewal timer is gone with
    /// the read — no renewal of this fence reaches the store afterwards, over
    /// two renewal periods watched only once every data-pool connection has
    /// drained what it held at the abort.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fenced_read_cancelled_mid_window_releases_its_lease_and_stops_renewing() {
        within_test_budget(async {
            let rx = Rx::new("srf-cancel-window").await;
            rx.warm_scripts().await;
            let estimate = rx.estimate().await;
            injector(&rx).push_delay_at(
                ObjectSnapshotReadFaultTarget::nth_command(ObjectSnapshotReader::FencedFanout, 1),
                Duration::from_secs(60),
            );
            let mut tap = MonitorTap::start(&test_valkey_addr()).await;
            let (hook, _log) = TestHook::new(no_reaction());
            let header = rx.header_key();

            let read = rx.spawn_read(estimate, hook);
            tap.drain_until_nth(1, "the window's header HGETALL", move |c| {
                !c.issued_by_script() && c.name() == "HGETALL" && c.args.get(1) == Some(&header)
            })
            .await;
            abort(read).await;

            assert_released_in_the_background(&rx, "cancelled mid-window").await;
            // A renewal the timer sent before the abort may still be queued on
            // one of the reader's data-pool connections. Each connection
            // answers in order, so once every one of them has answered a
            // PING, any such renewal has run, before the quiet tap opens.
            for (i, connection) in rx.reader.pool.clients().iter().enumerate() {
                let _: String = connection
                    .ping(None)
                    .await
                    .unwrap_or_else(|e| panic!("data-pool connection {i} must answer PING: {e}"));
            }
            // An absence check needs a window to be absent over: two renewal
            // periods, after which a live timer would have renewed.
            let period = ms_to_duration(
                fence::coefficients(&rx.reader)
                    .timing(estimate)
                    .renew_interval_ms(),
            );
            let mut quiet = MonitorTap::start(&test_valkey_addr()).await;
            tokio::time::sleep(period * 2).await;
            let marker = monitor_fence(rx.tv.prefix());
            let _: i64 = rx.tv.client().pool.exists(&marker).await.unwrap();
            let seen = quiet.drain_until(&marker).await;
            let renewals: Vec<_> = seen
                .iter()
                .filter(|c| is_script_on(renew_sha(), rx.fence_key())(c))
                .collect();
            assert!(
                renewals.is_empty(),
                "the cancelled read's renewal timer must be gone: {renewals:?}"
            );
        })
        .await;
    }

    /// Cancelled while a renewal is in flight: the lease is released in the
    /// background.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fenced_read_cancelled_during_a_renewal_releases_its_lease() {
        within_test_budget(async {
            let rx = Rx::new("srf-cancel-renew").await;
            rx.warm_scripts().await;
            let estimate = rx.estimate().await;
            injector(&rx).push_delay_at(
                ObjectSnapshotReadFaultTarget::next_read(),
                Duration::from_secs(60),
            );
            let mut tap = MonitorTap::start(&test_valkey_addr()).await;
            let (hook, _log) = TestHook::new(no_reaction());

            let read = rx.spawn_read(estimate, hook);
            tap.drain_until_nth(
                1,
                "the read's first renewal EVALSHA",
                is_script_on(renew_sha(), rx.fence_key()),
            )
            .await;
            abort(read).await;

            assert_released_in_the_background(&rx, "cancelled during a renewal").await;
        })
        .await;
    }

    /// An entry queued with `next_read()` answers in place of the fenced
    /// fan-out's first command (the header `HGETALL` on a fresh read): the pass
    /// fails with that command's classification — transient, `Aborted` — the
    /// read ends with it after one attempt, the lease is released, and the
    /// entry is consumed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn next_read_fault_is_consumed_by_the_fenced_fan_outs_first_window() {
        within_test_budget(async {
            let rx = Rx::new("srf-next-read").await;
            let estimate = rx.estimate().await;
            let seam = injector(&rx);
            seam.push(FredError::new(
                FredErrorKind::Timeout,
                "injected fenced timeout",
            ));
            let (mut hook, log) = TestHook::new(no_reaction());

            let err = rx.read(estimate, &mut hook).await.err();

            assert_eq!(seam.remaining(), 0, "the fenced fan-out consumed the entry");
            assert!(
                matches!(&err, Some(ServiceError::Aborted(m)) if m.contains("HGETALL obj header")),
                "the pass's transient failure ends the read; got {err:?}"
            );
            assert_eq!(names_without_retries(&log), held_attempts(1));
            assert_eq!(
                releases(&log),
                vec![FenceReleased {
                    verdict: Some(ReleaseVerdict::Verified),
                    outcome: FenceAttemptOutcome::Ended,
                    measured: None,
                }],
                "the lease itself was fine, but the failed pass ended the read: a VERIFIED \
                 release, never the served attempt"
            );
            assert!(rx.readers().await.is_empty(), "the lease was released");
            assert_eq!(rx.reads("catchup", "transient_error"), 1);
        })
        .await;
    }

    /// The fan-out fails with an error other than `NotFound` — an injected
    /// timeout on its first command, transient, `Aborted` — and the header is
    /// gone by the release, which answers `NOT_FOUND`: the read ends
    /// `NotFound` after that one attempt and counts `not_found`, never the
    /// fan-out's error (Spec SDD SR-5.3: a release that finds no header ends
    /// the read `NotFound` whatever the fan-out answered).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fenced_read_whose_fan_out_failed_and_whose_release_found_no_header_is_not_found() {
        within_test_budget(async {
            let rx = Rx::new("srf-failed-then-gone").await;
            let estimate = rx.estimate().await;
            let seam = injector(&rx);
            seam.push(FredError::new(
                FredErrorKind::Timeout,
                "injected fenced timeout",
            ));
            let stage = rx.stage();
            let (mut hook, log) = TestHook::new(react(move |attempt, event| {
                let stage = stage.clone();
                async move {
                    if (attempt, event) == (1, HookEvent::BeforeRelease) {
                        stage.take_header().await;
                    }
                    Ok(())
                }
            }));

            let err = rx.read(estimate, &mut hook).await.err();

            assert_eq!(
                seam.remaining(),
                0,
                "fixture precondition: the fan-out's first command failed"
            );
            assert!(
                matches!(err, Some(ServiceError::NotFound(_))),
                "a release that finds no header ends the read NotFound, not with the fan-out's \
                 error; got {err:?}"
            );
            assert_eq!(
                releases(&log),
                vec![FenceReleased {
                    verdict: Some(ReleaseVerdict::NotFound),
                    outcome: FenceAttemptOutcome::Ended,
                    measured: None,
                }],
                "one attempt: the fan-out failed, the release found the header gone"
            );
            assert_eq!(names_without_retries(&log), held_attempts(1));
            assert_eq!(
                (
                    rx.reads("catchup", "not_found"),
                    rx.reads("catchup", "transient_error"),
                    rx.reads("catchup", "fence_broken"),
                ),
                (1, 0, 0),
                "the attempt is counted not_found, not as the fan-out's transient error"
            );
        })
        .await;
    }

    /// Load the renewal script on the store, so each renewal a proxy sees is
    /// one `EVALSHA` — no `NOSCRIPT` miss and reload to count twice.
    async fn load_renew_script(rx: &Rx) {
        Script::from_lua(include_str!(
            "../../../src/storage/scripts/snapshot_fence_renew.lua"
        ))
        .load(rx.tv.client().pool.next())
        .await
        .expect("SCRIPT LOAD of the renewal script must succeed");
    }

    /// A renewal call that fails — an error, not a `LOST` answer — is retried
    /// at the next period, not taken as a lost lease. A proxy fails the read's
    /// first `FAILED_RENEWALS` renewals. The hook's `on_grant`, during which
    /// the renewal timer runs, first stages the granted lease's deadline far
    /// ahead — so the lease's life is not what is under test, and no renewal
    /// landing late on a loaded store can break the attempt — and then waits
    /// for the proxy to see a renewal after the failed ones. A timer that took
    /// a failure as a lost lease would have stopped, and that wait would end
    /// unmet. The first attempt is served.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fenced_read_retries_a_failed_renewal_at_the_next_period_instead_of_taking_the_lease_as_lost(
    ) {
        /// How many of the read's renewals the proxy fails.
        const FAILED_RENEWALS: usize = 2;
        within_test_budget(async {
            let sha = renew_sha();
            let renewals = Arc::new(AtomicUsize::new(0));
            let counted = Arc::clone(&renewals);
            let rule: ProxyRule = Arc::new(move |args: &[Vec<u8>]| {
                if !is_evalsha_of(args, &sha) {
                    return None;
                }
                (counted.fetch_add(1, Ordering::SeqCst) < FAILED_RENEWALS)
                    .then(|| ProxyAction::Eval(error_reply_script("ERR injected renewal failure")))
            });
            let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
            let tv = TestValkey::setup().await;
            let rx = Rx::with_reader(tv, client_through(&proxy).await, "srf-renew-fails").await;
            load_renew_script(&rx).await;
            let estimate = PROXIED_ESTIMATE;
            let renewed_after_failures = Arc::new(AtomicBool::new(false));
            let (stage, seen, renewed) = (
                rx.stage(),
                Arc::clone(&renewals),
                Arc::clone(&renewed_after_failures),
            );
            let (mut hook, log) = TestHook::new(react(move |attempt, event| {
                let (stage, seen, renewed) =
                    (stage.clone(), Arc::clone(&seen), Arc::clone(&renewed));
                async move {
                    if let (1, HookEvent::Grant(_)) = (attempt, event) {
                        hold_the_lease(&stage).await;
                        let next_renewal = async {
                            while seen.load(Ordering::SeqCst) <= FAILED_RENEWALS {
                                tokio::time::sleep(Duration::from_millis(10)).await;
                            }
                        };
                        renewed.store(
                            tokio::time::timeout(SIGNAL_BUDGET, next_renewal)
                                .await
                                .is_ok(),
                            Ordering::SeqCst,
                        );
                    }
                    Ok(())
                }
            }));

            let fenced = verified(rx.read(estimate, &mut hook).await, "failed renewals");

            assert!(
                renewed_after_failures.load(Ordering::SeqCst),
                "the timer must call again after its renewals failed: no renewal followed the \
                 failed ones within {SIGNAL_BUDGET:?} (renewals seen: {}, failed: {})",
                renewals.load(Ordering::SeqCst),
                proxy.hits()
            );
            assert_eq!(
                proxy.hits(),
                FAILED_RENEWALS,
                "the timer called again after each failed renewal: the proxy failed exactly the \
                 read's first {FAILED_RENEWALS}"
            );
            assert_eq!(
                names_without_retries(&log),
                held_attempts(1),
                "a failed renewal neither stopped the pass nor lost the lease: the first attempt \
                 was served"
            );
            assert_served_by_the_last_attempt(&log, &fenced, 1, "failed renewals");
            assert_eq!(rx.reads("catchup", "fence_broken"), 0);
        })
        .await;
    }

    /// The lease is lost while the hook is inside `at` (`on_grant` or
    /// `before_release`) on attempt 1, and the event still runs to its end.
    ///
    /// Renewals start stalled. The hook, parked in `at`, waits for the lease
    /// to lapse on the store's clock, lets renewals resume, waits for the
    /// next renewal to reach the store (counted at a proxy) — it answers
    /// `LOST` — and stays parked two more renewal periods. No renewal follows,
    /// so the timer has stopped: the read took the `LOST` while the hook was
    /// parked. The hook then records that it finished. The attempt is broken
    /// (the lapsed lease's release answers `BROKEN`; the pass did not run when
    /// the `LOST` came during `on_grant`, and completed before
    /// `before_release`), and the retry is served.
    async fn lease_lost_during_a_hook_event(at: &'static str) {
        let sha = renew_sha();
        let renewals = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&renewals);
        let rule: ProxyRule = Arc::new(move |args: &[Vec<u8>]| {
            if is_evalsha_of(args, &sha) {
                counted.fetch_add(1, Ordering::SeqCst);
            }
            None
        });
        let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
        let tv = TestValkey::setup().await;
        let rx = Rx::with_reader(tv, client_through(&proxy).await, "srf-lost-in-hook").await;
        load_renew_script(&rx).await;
        let estimate = PROXIED_ESTIMATE;
        let measured = rx.atomic().await.measured_size();
        let period = lease(&rx, estimate) / 4;
        let seam = injector(&rx);
        seam.stall_fence_renewals(true);
        let finished = Arc::new(AtomicBool::new(false));
        let parked_renewals = Arc::new(StdMutex::new(None::<(usize, usize)>));
        let (stage, done, seen) = (
            rx.stage(),
            Arc::clone(&finished),
            Arc::clone(&parked_renewals),
        );
        let (mut hook, log) = TestHook::new(react(move |attempt, event| {
            let (stage, seam, renewals, done, seen) = (
                stage.clone(),
                Arc::clone(&seam),
                Arc::clone(&renewals),
                Arc::clone(&done),
                Arc::clone(&seen),
            );
            async move {
                if attempt != 1 || event.name() != at {
                    return Ok(());
                }
                let deadline = readers_of(&key_fields(&stage.client, &stage.fence).await)
                    .into_values()
                    .next()
                    .expect("the held lease's deadline");
                let lapse = async {
                    while store_now(&stage.client, &stage.fence).await <= deadline {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                };
                tokio::time::timeout(LAPSE_BUDGET, lapse)
                    .await
                    .unwrap_or_else(|_| {
                        panic!("{at}: the lease did not lapse in {LAPSE_BUDGET:?}")
                    });
                let before = renewals.load(Ordering::SeqCst);
                seam.stall_fence_renewals(false);
                let next_renewal = async {
                    while renewals.load(Ordering::SeqCst) == before {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                };
                tokio::time::timeout(SIGNAL_BUDGET, next_renewal)
                    .await
                    .unwrap_or_else(|_| {
                        panic!("{at}: no renewal reached the store in {SIGNAL_BUDGET:?}")
                    });
                // An absence check needs a window to be absent over: two
                // renewal periods, after which a live timer would have renewed.
                tokio::time::sleep(period * 2).await;
                *seen.lock().unwrap() = Some((before, renewals.load(Ordering::SeqCst)));
                done.store(true, Ordering::SeqCst);
                Ok(())
            }
        }));

        let fenced = verified(rx.read(estimate, &mut hook).await, at);

        assert!(
            finished.load(Ordering::SeqCst),
            "{at}: the hook event must run to its end although the lease was lost during it"
        );
        let (before, after) = parked_renewals
            .lock()
            .unwrap()
            .expect("a finished hook recorded its renewal counts");
        assert_eq!(
            after,
            before + 1,
            "{at}: fixture precondition: one renewal after the lapse, then none — it answered \
             LOST and the timer stopped while the hook was parked"
        );
        assert_eq!(
            releases(&log)[0],
            FenceReleased {
                verdict: Some(ReleaseVerdict::Broken),
                outcome: FenceAttemptOutcome::Broken,
                measured: (at == "before_release").then_some(measured),
            },
            "{at}: the lost lease broke the attempt; the pass ran only if it came before {at}"
        );
        assert_eq!(names_without_retries(&log), held_attempts(2));
        assert_served_by_the_last_attempt(&log, &fenced, 2, at);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fenced_read_runs_on_grant_to_its_end_when_the_lease_is_lost_during_it() {
        within_test_budget(lease_lost_during_a_hook_event("on_grant")).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fenced_read_runs_before_release_to_its_end_when_the_lease_is_lost_during_it() {
        within_test_budget(lease_lost_during_a_hook_event("before_release")).await;
    }
}
