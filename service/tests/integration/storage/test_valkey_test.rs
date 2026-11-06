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

// Integration tests for the TestValkey fixture: prefix isolation, uniqueness,
// drop cleanup, cross-prefix safety, parallel safety, and the drop's wait for
// its cleanup (it never swallows another future's wake-up, hands a runtime
// worker's core off, skips on a current-thread runtime, ends with a runtime
// shutting down, and fails a passing test on an unanswered cleanup step), and
// the bridge that wait runs through (`block_on_spawned`'s three outcomes, and
// `rpush_blocking` re-raising a failed push's own panic), and what the drop
// does with each outcome of its cleanup (`fixture_drop_action`).
//
// Requires a running Valkey instance, except for the three `block_on_spawned`
// tests and the `fixture_drop_action` tests. Address is read from
// TEST_VALKEY_URL, defaulting to redis://127.0.0.1:6379. Tests that connect a
// verification client first (`raw_client`) fail on an unreachable store when
// TEST_VALKEY_URL is set and skip only when it is unset outside CI; the rest
// fail at `TestValkey::setup`.

use std::collections::HashSet;
use std::panic::AssertUnwindSafe;
use std::time::Duration;

use fred::interfaces::{ClientLike, KeysInterface};
use futures::{FutureExt, StreamExt};
use tokio::time::Instant;

use live_edit::storage::client::ValkeyClient;
use live_edit::storage::test_support::{
    block_on_spawned, connect_test_client_or_skip, fixture_drop_action, release_all,
    rpush_blocking, stall_pool, FixtureDropAction, SpawnedTaskEnded, TestValkey,
};

/// Connect a one-connection client (not a `TestValkey`, so no prefix and no
/// cleanup) for verification queries on its `pool`, through
/// [`connect_test_client_or_skip`]: `None` only when `TEST_VALKEY_URL` is
/// unset outside CI and no store answered.
async fn raw_client() -> Option<ValkeyClient> {
    let mut cfg = crate::common::test_config();
    cfg.valkey_pool_size = 1;
    cfg.subscriber_pool_size = 1;
    connect_test_client_or_skip(&cfg).await
}

/// Count keys matching `pattern` via SCAN on the given pool.
async fn count_keys(pool: &fred::clients::Pool, pattern: &str) -> usize {
    let client = pool.next();
    let mut stream = client.scan_buffered(pattern, Some(100), None);
    let mut count = 0;
    while let Some(Ok(_)) = stream.next().await {
        count += 1;
    }
    count
}

// ── 1. Prefix isolation ──────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_valkey_two_fixtures_same_key_name_see_different_values() {
    let fixture_a: TestValkey = TestValkey::setup().await;
    let fixture_b: TestValkey = TestValkey::setup().await;

    let key_name = "shared_name";
    let key_a = format!("{}{key_name}", fixture_a.prefix());
    let key_b = format!("{}{key_name}", fixture_b.prefix());

    fixture_a
        .client()
        .pool
        .set::<(), _, _>(&key_a, "value_a", None, None, false)
        .await
        .expect("SET on fixture A must succeed");

    fixture_b
        .client()
        .pool
        .set::<(), _, _>(&key_b, "value_b", None, None, false)
        .await
        .expect("SET on fixture B must succeed");

    let got_a: String = fixture_a
        .client()
        .pool
        .get(&key_a)
        .await
        .expect("GET on fixture A must succeed");

    let got_b: String = fixture_b
        .client()
        .pool
        .get(&key_b)
        .await
        .expect("GET on fixture B must succeed");

    assert_eq!(got_a, "value_a", "fixture A must see its own value");
    assert_eq!(got_b, "value_b", "fixture B must see its own value");

    // The important point: same logical name, different actual keys.
    assert_ne!(
        key_a, key_b,
        "actual Valkey keys must differ due to different prefixes"
    );
}

// ── 2. Prefix uniqueness ─────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_valkey_100_instances_all_have_distinct_prefixes() {
    // Build 100 fixtures with bounded concurrency. Sequential setup runs
    // ~180 ms per instance (each `TestValkey::setup` spins up a fresh
    // ValkeyClient with its three pools); a fully-parallel `join_all`
    // floods Valkey with ~1200 simultaneous TCP handshakes and degrades
    // under suite contention. Chunks of 8 shave the single-test
    // wall-clock from ~17 s to ~13 s when run in isolation; under full
    // suite load every test on the shared Valkey is competing, so the
    // suite-wide win is small — the bounded form just avoids making
    // things worse than serial.
    use futures::stream::{FuturesUnordered, StreamExt};
    const CONCURRENCY: usize = 8;
    let mut in_flight = FuturesUnordered::new();
    let mut fixtures: Vec<TestValkey> = Vec::with_capacity(100);
    let mut to_spawn = 100usize;
    while to_spawn > 0 || !in_flight.is_empty() {
        while in_flight.len() < CONCURRENCY && to_spawn > 0 {
            in_flight.push(TestValkey::setup());
            to_spawn -= 1;
        }
        if let Some(f) = in_flight.next().await {
            fixtures.push(f);
        }
    }

    let prefixes: HashSet<String> = fixtures.iter().map(|f| f.prefix().to_string()).collect();

    assert_eq!(
        prefixes.len(),
        100,
        "all 100 TestValkey instances must have distinct prefixes"
    );
    // Keep `fixtures` alive past the assertion: each fixture's `Drop`
    // cleans Valkey keys, so we don't want to start dropping mid-assert.
    drop(fixtures);
}

// ── 3. Drop cleanup ─────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_valkey_drop_removes_all_prefixed_keys() {
    let Some(raw) = raw_client().await else {
        return;
    };

    let prefix;
    {
        let fixture: TestValkey = TestValkey::setup().await;
        prefix = fixture.prefix().to_string();

        // Write 10 keys under this fixture's prefix.
        for i in 0..10 {
            let key = format!("{prefix}key_{i}");
            fixture
                .client()
                .pool
                .set::<(), _, _>(&key, format!("val_{i}"), None, None, false)
                .await
                .expect("SET must succeed");
        }

        // Verify keys exist before drop.
        let before = count_keys(&raw.pool, &format!("{prefix}*")).await;
        assert_eq!(before, 10, "10 keys must exist before drop");

        // fixture is dropped here at end of scope, triggering cleanup
    }

    // After drop, verify zero keys remain with this prefix.
    let after = count_keys(&raw.pool, &format!("{prefix}*")).await;
    assert_eq!(
        after, 0,
        "all keys with the dropped fixture's prefix must be cleaned up"
    );
}

// ── 4. Cleanup does not affect other prefixes ────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_valkey_drop_one_fixture_preserves_other_fixture_keys() {
    let Some(raw) = raw_client().await else {
        return;
    };

    let fixture_survivor: TestValkey = TestValkey::setup().await;
    let survivor_prefix = fixture_survivor.prefix().to_string();

    // Write keys on the survivor fixture.
    for i in 0..5 {
        let key = format!("{survivor_prefix}kept_{i}");
        fixture_survivor
            .client()
            .pool
            .set::<(), _, _>(&key, format!("val_{i}"), None, None, false)
            .await
            .expect("SET on survivor must succeed");
    }

    {
        let fixture_doomed: TestValkey = TestValkey::setup().await;
        let doomed_prefix = fixture_doomed.prefix().to_string();

        // Write keys on the doomed fixture.
        for i in 0..5 {
            let key = format!("{doomed_prefix}doomed_{i}");
            fixture_doomed
                .client()
                .pool
                .set::<(), _, _>(&key, format!("val_{i}"), None, None, false)
                .await
                .expect("SET on doomed must succeed");
        }

        // fixture_doomed is dropped here
    }

    // Survivor's keys must still be present.
    let remaining = count_keys(&raw.pool, &format!("{survivor_prefix}*")).await;
    assert_eq!(
        remaining, 5,
        "survivor fixture's keys must not be affected by dropping another fixture"
    );
}

// ── 5. Parallel safety ──────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_valkey_parallel_tasks_no_cross_contamination() {
    // Spawn 8 workers, each as its own tokio task so they run in parallel on
    // the runtime's 4 workers, then join! on all handles.
    let h0 = tokio::spawn(parallel_worker(0));
    let h1 = tokio::spawn(parallel_worker(1));
    let h2 = tokio::spawn(parallel_worker(2));
    let h3 = tokio::spawn(parallel_worker(3));
    let h4 = tokio::spawn(parallel_worker(4));
    let h5 = tokio::spawn(parallel_worker(5));
    let h6 = tokio::spawn(parallel_worker(6));
    let h7 = tokio::spawn(parallel_worker(7));

    let (r0, r1, r2, r3, r4, r5, r6, r7) = tokio::join!(h0, h1, h2, h3, h4, h5, h6, h7);

    assert!(r0.expect("task 0 panicked"), "worker 0 must succeed");
    assert!(r1.expect("task 1 panicked"), "worker 1 must succeed");
    assert!(r2.expect("task 2 panicked"), "worker 2 must succeed");
    assert!(r3.expect("task 3 panicked"), "worker 3 must succeed");
    assert!(r4.expect("task 4 panicked"), "worker 4 must succeed");
    assert!(r5.expect("task 5 panicked"), "worker 5 must succeed");
    assert!(r6.expect("task 6 panicked"), "worker 6 must succeed");
    assert!(r7.expect("task 7 panicked"), "worker 7 must succeed");
}

/// A single parallel worker: creates its own TestValkey, writes keys with
/// a worker-specific value, reads them back, and verifies correctness.
async fn parallel_worker(worker_id: u32) -> bool {
    let fixture: TestValkey = TestValkey::setup().await;
    let prefix = fixture.prefix().to_string();
    let expected_value = format!("worker_{worker_id}");

    // Write 5 keys per worker.
    for i in 0..5 {
        let key = format!("{prefix}item_{i}");
        if fixture
            .client()
            .pool
            .set::<(), _, _>(&key, &expected_value, None, None, false)
            .await
            .is_err()
        {
            return false;
        }
    }

    // Read back and verify each key holds this worker's value (not another's).
    for i in 0..5 {
        let key = format!("{prefix}item_{i}");
        let got: Result<String, _> = fixture.client().pool.get(&key).await;
        match got {
            Ok(val) if val == expected_value => {}
            _ => return false,
        }
    }

    true
}

// ── 6. The drop's wait for its cleanup ──────────────────────────────────────
//
// `TestValkey`'s Drop runs its key cleanup and its quit as one task spawned
// on the test's runtime, and waits for that task on a std channel inside
// `block_in_place` (`test_support::block_on_spawned`), never in a nested
// `Handle::block_on`. The scenarios that could hang run on a thread and a
// runtime of their own ([`on_own_thread`]), so a drop that never returns fails
// its test at `SCENARIO_DEADLINE` instead of hanging it until nextest's kill,
// which prints nothing.

/// Bound on one scenario thread: unreachable on a healthy store (each scenario
/// takes a few seconds), and well inside nextest's 120 s hang-killer.
const SCENARIO_DEADLINE: Duration = Duration::from_secs(60);

/// Bound on each other wait of a scenario (a spawned task's join, a ready
/// signal): unreachable on a healthy store.
const STEP_BUDGET: Duration = Duration::from_secs(20);

/// How long the sibling future beside a drop sleeps: due while the drop is
/// held in its cleanup.
const SIBLING_SLEEP: Duration = Duration::from_millis(50);

/// How long after its spawn [`spawn_releaser`] releases a held cleanup: ten
/// times [`SIBLING_SLEEP`], so the sibling's sleep comes due inside the drop.
const DROP_HOLD: Duration = Duration::from_millis(500);

/// Rounds of a scenario per test: each round is one more chance for a wake-up
/// to be lost or a core to stay put.
const ROUNDS: usize = 3;

/// Run `scenario` on a thread of its own, waiting for it for at most
/// [`SCENARIO_DEADLINE`] and re-raising its panic. A scenario that hangs fails
/// the test naming `name`; its thread is left parked until the test process
/// exits.
fn on_own_thread(name: &str, scenario: impl FnOnce() + Send + 'static) {
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name(name.to_owned())
        .spawn(move || {
            let _ = done_tx.send(std::panic::catch_unwind(AssertUnwindSafe(scenario)));
        })
        .expect("the scenario thread must start");
    match done_rx.recv_timeout(SCENARIO_DEADLINE) {
        Ok(Ok(())) => {}
        Ok(Err(payload)) => std::panic::resume_unwind(payload),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => panic!(
            "{name}: the scenario did not finish within {SCENARIO_DEADLINE:?}: a fixture's drop \
             or the runtime's shutdown never returned"
        ),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            unreachable!("the scenario thread reports before it ends")
        }
    }
}

fn multi_thread_runtime(workers: usize) -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_all()
        .build()
        .expect("the scenario's runtime must build")
}

async fn key_exists(raw: &fred::clients::Pool, key: &str) -> bool {
    let n: u64 = raw.exists(key).await.expect("EXISTS must succeed");
    n == 1
}

/// A fixture whose drop is held in its cleanup until the returned stalls are
/// released: a one-connection data pool (the cleanup's `SCAN` and `UNLINK`
/// both go through it), one key under its prefix for the cleanup to remove
/// (returned), and that connection stalled. The stall is observed and
/// released from `raw`, which the fixture's own quit does not close.
async fn fixture_held_in_its_cleanup(
    raw: &fred::clients::Pool,
) -> (
    TestValkey,
    String,
    Vec<live_edit::storage::test_support::ConnStall>,
) {
    let fixture = TestValkey::setup_with_pool_size(1).await;
    let key = format!("{}held", fixture.prefix());
    fixture
        .client()
        .pool
        .set::<(), _, _>(&key, "v", None, None, false)
        .await
        .expect("SET must succeed");
    let stalls = stall_pool(
        &fixture.client().pool,
        raw.next(),
        &format!("{}conn-stall:", fixture.prefix()),
    )
    .await;
    (fixture, key, stalls)
}

/// Release `stalls` [`DROP_HOLD`] after the spawn. Reports the thread that ran
/// the task just before the release, and when the release landed.
fn spawn_releaser(
    stalls: Vec<live_edit::storage::test_support::ConnStall>,
) -> tokio::task::JoinHandle<(std::thread::ThreadId, Instant)> {
    tokio::spawn(async move {
        tokio::time::sleep(DROP_HOLD).await;
        let thread = std::thread::current().id();
        release_all(stalls).await;
        (thread, Instant::now())
    })
}

/// What a future of [`drop_beside_a_sibling_timer`]'s `join_all` reports.
enum Ended {
    /// The sibling's sleep, due at `due`, fired and the sibling was polled
    /// again.
    Sibling { due: Instant },
    /// The fixture's drop ran from `started` to `returned`.
    Dropped { started: Instant, returned: Instant },
}

/// One round: a fixture dropped inside a `futures::future::join_all`, on the
/// runtime's `block_on` thread (as a `#[tokio::test]` body's `join_all` runs),
/// beside a sibling whose sleep comes due while the drop is held in its
/// cleanup. The sibling is first: `join_all` polls in order, so its sleep is
/// registered before the drop starts (`tokio::join!` rotates where it starts).
///
/// A drop that parks on the thread's tokio parker — a nested `block_on` —
/// consumes the sleep's wake-up, and the `block_on` then parks for good: the
/// round never returns, and [`on_own_thread`]'s deadline fails the test.
async fn drop_beside_a_sibling_timer(raw: &fred::clients::Pool, round: usize) {
    let (fixture, key, stalls) = fixture_held_in_its_cleanup(raw).await;
    let releaser = spawn_releaser(stalls);
    let sibling = async {
        let sleep = tokio::time::sleep(SIBLING_SLEEP);
        let due = sleep.deadline();
        sleep.await;
        Ended::Sibling { due }
    };
    let dropper = async move {
        let started = Instant::now();
        drop(fixture);
        Ended::Dropped {
            started,
            returned: Instant::now(),
        }
    };

    let ended = futures::future::join_all([sibling.boxed_local(), dropper.boxed_local()]).await;

    let [Ended::Sibling { due }, Ended::Dropped { started, returned }] = ended[..] else {
        unreachable!("join_all keeps its futures' order")
    };
    assert!(
        started < due && due < returned,
        "round {round}: fixture: the drop must start before the sibling's sleep is due and return \
         after it, so the wake-up arrives during the drop; started {started:?}, due {due:?}, \
         returned {returned:?}"
    );
    assert!(
        !key_exists(raw, &key).await,
        "round {round}: the drop waited for its cleanup, so the key under its prefix is gone"
    );
    tokio::time::timeout(STEP_BUDGET, releaser)
        .await
        .expect("the releaser must finish")
        .expect("the releaser must not panic");
}

/// A fixture dropped inside a `join_all` on the runtime's `block_on` thread,
/// beside a sibling whose timer fires during the drop: the drop still waits
/// for its cleanup, and the sibling's wake-up still reaches the `block_on`, so
/// the `join_all` returns. The regression: a nested `block_on` in the drop
/// consumed that wake-up and the test hung until nextest killed it.
#[test]
fn test_valkey_drop_inside_join_all_beside_a_due_timer_loses_no_wake_up() {
    on_own_thread("drop-beside-a-due-timer", || {
        multi_thread_runtime(2).block_on(async {
            let Some(raw) = raw_client().await else {
                return;
            };
            for round in 0..ROUNDS {
                drop_beside_a_sibling_timer(&raw.pool, round).await;
            }
        });
    });
}

/// A fixture dropped by a spawned task on a runtime with ONE worker: the
/// drop's wait hands the worker's core off (`block_in_place`), so the
/// releaser task, spawned on that worker just before the drop, runs on
/// another thread while the drop is in progress, and the cleanup completes.
/// A wait that kept the core would leave nothing to run either task, and the
/// drop would never return.
#[test]
fn test_valkey_drop_on_the_only_runtime_worker_hands_its_core_off_and_cleans_up() {
    on_own_thread("drop-on-the-only-worker", || {
        multi_thread_runtime(1).block_on(async {
            let Some(raw) = raw_client().await else {
                return;
            };
            for round in 0..ROUNDS {
                let (fixture, key, stalls) = fixture_held_in_its_cleanup(&raw.pool).await;
                let dropped = tokio::spawn(async move {
                    let releaser = spawn_releaser(stalls);
                    let dropping_thread = std::thread::current().id();
                    let started = Instant::now();
                    drop(fixture);
                    (releaser, dropping_thread, started, Instant::now())
                });
                let (releaser, dropping_thread, started, returned) =
                    tokio::time::timeout(STEP_BUDGET, dropped)
                        .await
                        .expect("the dropping task must finish")
                        .expect("the dropping task must not panic");
                let (releasing_thread, released) = tokio::time::timeout(STEP_BUDGET, releaser)
                    .await
                    .expect("the releaser must finish")
                    .expect("the releaser must not panic");

                assert!(
                    started < released && released < returned,
                    "round {round}: fixture: the release lands while the drop is in progress; \
                     started {started:?}, released {released:?}, returned {returned:?}"
                );
                assert_ne!(
                    releasing_thread, dropping_thread,
                    "round {round}: with the only worker in the drop, the releaser ran on another \
                     thread: the drop's wait handed the worker's core off"
                );
                assert!(
                    !key_exists(&raw.pool, &key).await,
                    "round {round}: the drop's cleanup ran to the end, so the key is gone"
                );
            }
        });
    });
}

/// On a current-thread runtime the drop skips its cleanup and quit
/// (`block_in_place` panics there), so a fixture dropped inside a `join_all`
/// beside a sleeping sibling returns at once, nothing panics, and the
/// `join_all` completes. The fixture writes no key: nothing would remove it.
#[test]
fn test_valkey_drop_on_a_current_thread_runtime_skips_its_cleanup_without_panicking() {
    on_own_thread("drop-on-a-current-thread-runtime", || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("the scenario's runtime must build");
        rt.block_on(async {
            if raw_client().await.is_none() {
                return;
            }
            let fixture = TestValkey::setup().await;
            let sibling = async { tokio::time::sleep(SIBLING_SLEEP).await };
            let dropper = async move { drop(fixture) };
            futures::future::join_all([sibling.boxed_local(), dropper.boxed_local()]).await;
        });
    });
}

/// Holds a fixture, and reports on `dropped` once the fixture's drop has
/// returned.
struct ReportsItsFixtureDropped {
    fixture: Option<TestValkey>,
    dropped: std::sync::mpsc::Sender<()>,
}

impl Drop for ReportsItsFixtureDropped {
    fn drop(&mut self) {
        drop(self.fixture.take());
        let _ = self.dropped.send(());
    }
}

/// A fixture owned by a spawned task that is still pending when its runtime is
/// dropped: the shutdown drops the task, and with it the fixture, whose drop
/// returns (its cleanup task is refused or cancelled by the shutdown, which
/// ends the wait), so the runtime's drop completes. A wait that held a sender
/// of its own would never end, and neither would the runtime's drop. The
/// fixture writes no key: the shutdown leaves nothing to remove it.
#[test]
fn test_valkey_owned_by_a_pending_task_drops_with_its_runtime() {
    on_own_thread("fixture-in-a-pending-task", || {
        let rt = multi_thread_runtime(2);
        let (dropped_tx, dropped_rx) = std::sync::mpsc::channel();
        let store_answered = rt.block_on(async move {
            if raw_client().await.is_none() {
                return false;
            }
            let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
            tokio::spawn(async move {
                let _held = ReportsItsFixtureDropped {
                    fixture: Some(TestValkey::setup().await),
                    dropped: dropped_tx,
                };
                let _ = ready_tx.send(());
                std::future::pending::<()>().await;
            });
            tokio::time::timeout(STEP_BUDGET, ready_rx)
                .await
                .expect("the task must set its fixture up")
                .expect("the task must report its fixture set up");
            true
        });
        if !store_answered {
            return;
        }

        drop(rt);

        dropped_rx.try_recv().expect(
            "the runtime's shutdown dropped the pending task, and the fixture's drop returned",
        );
    });
}

/// A fixture whose data pool was quit: the drop's first cleanup step, a
/// `SCAN` on that pool, gets no answer within the step budget.
async fn fixture_with_its_data_pool_quit() -> (TestValkey, String) {
    let fixture = TestValkey::setup().await;
    let pattern = format!("{}*", fixture.prefix());
    tokio::time::timeout(STEP_BUDGET, fixture.client().pool.quit())
        .await
        .expect("the data pool's quit must finish")
        .expect("the data pool's quit must succeed");
    (fixture, pattern)
}

/// An unanswered cleanup step fails an otherwise passing test: the drop
/// panics, naming the step.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_valkey_drop_with_an_unanswered_cleanup_step_panics_naming_the_step() {
    let (fixture, pattern) = fixture_with_its_data_pool_quit().await;

    let payload = std::panic::catch_unwind(AssertUnwindSafe(|| drop(fixture)))
        .expect_err("the drop must fail the test: a pooled client can no longer answer");

    let message = payload
        .downcast_ref::<String>()
        .expect("the drop's panic carries a formatted message");
    assert_eq!(
        message,
        &format!(
            "TestValkey: key cleanup for `{pattern}` stopped — a SCAN step got no answer within \
             5s; leaving those keys behind — a pooled client can no longer answer"
        ),
    );
}

/// The same unanswered step in a test that is already failing only reports
/// it: the thread's panic stays the test's own, and the process does not
/// abort (a panic in a drop during an unwind would abort it, failing the whole
/// test process).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_valkey_drop_with_an_unanswered_cleanup_step_while_unwinding_keeps_the_original_panic()
{
    let (fixture, _) = fixture_with_its_data_pool_quit().await;

    let payload = std::panic::catch_unwind(AssertUnwindSafe(move || {
        let _fixture = fixture;
        panic!("the test's own failure");
    }))
    .expect_err("the closure panics");

    assert_eq!(
        payload.downcast_ref::<&str>(),
        Some(&"the test's own failure"),
        "the unwind's payload is the test's own panic, not the drop's"
    );
}

// ── 7. The bridge the drop waits through ────────────────────────────────────
//
// `block_on_spawned` reports one of three outcomes, and the drop and
// `rpush_blocking` each map them. The first three tests need no Valkey.

/// A future that completes hands its output back.
#[test]
fn block_on_spawned_of_a_future_that_completes_returns_its_output() {
    let rt = multi_thread_runtime(2);

    let outcome =
        rt.block_on(async { block_on_spawned(&tokio::runtime::Handle::current(), async { 7 }) });

    assert!(
        matches!(outcome, Ok(7)),
        "the future's output comes back; got {outcome:?}"
    );
}

/// A future that panics comes back as `Panicked` with its own payload, caught
/// in the spawned task. Uncaught, the task would end without reporting and
/// read as `NeverReported`, a runtime shutting down, which the drop only
/// prints and `rpush_blocking` misreports.
#[test]
fn block_on_spawned_of_a_future_that_panics_returns_its_panic_payload() {
    let rt = multi_thread_runtime(2);

    let outcome = rt.block_on(async {
        block_on_spawned(&tokio::runtime::Handle::current(), async {
            panic!("boom");
        })
    });

    match outcome {
        Err(SpawnedTaskEnded::Panicked(payload)) => assert_eq!(
            payload.downcast_ref::<&str>(),
            Some(&"boom"),
            "the payload is the future's own panic"
        ),
        other => panic!("a panicking future comes back as Panicked; got {other:?}"),
    }
}

/// Sets its flag when dropped.
struct SetsOnDrop(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Drop for SetsOnDrop {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

/// On the handle of a runtime already dropped, called from a thread outside
/// any runtime (where `block_in_place` runs its closure in place), the spawn
/// is refused: the future is dropped without being polled, its task never
/// reports, and the outcome is `NeverReported`.
#[test]
fn block_on_spawned_on_a_dropped_runtime_returns_never_reported() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let rt = multi_thread_runtime(2);
    let handle = rt.handle().clone();
    drop(rt);
    let (polled, dropped) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    );
    let guard = SetsOnDrop(Arc::clone(&dropped));

    let outcome = block_on_spawned(&handle, {
        let polled = Arc::clone(&polled);
        async move {
            let _guard = guard;
            polled.store(true, Ordering::SeqCst);
        }
    });

    assert!(
        matches!(outcome, Err(SpawnedTaskEnded::NeverReported)),
        "a task the runtime refuses never reports; got {outcome:?}"
    );
    assert!(
        dropped.load(Ordering::SeqCst) && !polled.load(Ordering::SeqCst),
        "the future was dropped without being polled"
    );
}

/// A push that fails inside `rpush_blocking` re-raises the push's own panic
/// (its `expect`), never the "runtime is shutting down" one. The push fails
/// at once: the key holds a string, so the store answers `WRONGTYPE`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rpush_blocking_whose_push_fails_re_raises_the_pushs_own_panic() {
    let fixture = TestValkey::setup().await;
    let key = format!("{}not-a-list", fixture.prefix());
    fixture
        .client()
        .pool
        .set::<(), _, _>(&key, "a string", None, None, false)
        .await
        .expect("SET must succeed");
    let client = fixture.client().clone();

    // On a runtime worker, as a blocking test sink calls it.
    let pushed = tokio::spawn(async move {
        std::panic::catch_unwind(AssertUnwindSafe(|| rpush_blocking(&client, &key, "v")))
    });
    let payload = tokio::time::timeout(STEP_BUDGET, pushed)
        .await
        .expect("the push must finish")
        .expect("the push's panic is caught inside the task")
        .expect_err("a push onto a string fails");

    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .expect("the re-raised panic carries a message");
    assert!(
        message.starts_with("RPUSH from a blocking test sink must succeed")
            && message.contains("WRONGTYPE"),
        "the re-raised panic is the push's own; got {message:?}"
    );
}

// ── 8. What the drop does with its cleanup's outcome ────────────────────────
//
// `fixture_drop_action` maps the cleanup task's outcome, and whether the
// dropping thread is already unwinding, to what `TestValkey`'s drop does:
// report lines on stderr, panic, or re-raise the task's own panic. These
// tests need no Valkey and no runtime, and pin each leg's exact text.

/// The key pattern the messages name.
const PATTERN: &str = "test_p:*";

/// A cleanup step that stopped, as `cleanup_keys` names one.
const STEP: &str = "a SCAN step got no answer within 5s";

/// The line reporting a quit that ran past its 10 s budget.
fn quit_line() -> String {
    format!(
        "TestValkey: the fixture client's quit for `{PATTERN}` did not finish within 10s; its \
         connections close without a `QUIT`"
    )
}

/// The message of a cleanup that stopped at [`STEP`].
fn stopped_message() -> String {
    format!(
        "TestValkey: key cleanup for `{PATTERN}` stopped — {STEP}; leaving those keys behind — \
         a pooled client can no longer answer"
    )
}

/// The line reporting a cleanup task that panicked with `message`.
fn panicked_line(message: &str) -> String {
    format!(
        "TestValkey: the drop-time cleanup or quit for `{PATTERN}` panicked — {message}; keys \
         under that prefix may be left behind"
    )
}

/// The line reporting a cleanup task the runtime's shutdown dropped.
fn never_reported_line() -> String {
    format!(
        "TestValkey: the drop-time cleanup or quit for `{PATTERN}` did not run to the end — the \
         test's runtime is shutting down; keys under that prefix may be left behind"
    )
}

/// The lines of a `Report` action; any other action fails, naming `context`.
fn reported(action: FixtureDropAction, context: &str) -> Vec<String> {
    match action {
        FixtureDropAction::Report(lines) => lines,
        other => panic!("{context}: the drop only reports; got {other:?}"),
    }
}

/// A clean drop, the cleanup finished and the quit within its budget,
/// reports nothing, whether or not the thread is unwinding.
#[test]
fn fixture_drop_action_of_a_clean_drop_reports_nothing() {
    for panicking in [false, true] {
        let action = fixture_drop_action(Ok((Ok(()), true)), PATTERN, panicking);

        assert_eq!(
            reported(action, &format!("panicking={panicking}")),
            Vec::<String>::new(),
            "panicking={panicking}: nothing to report"
        );
    }
}

/// A quit that ran past its budget, after a finished cleanup, is only
/// reported: the test does not fail on it.
#[test]
fn fixture_drop_action_of_a_quit_out_of_time_reports_the_quit_line() {
    for panicking in [false, true] {
        let action = fixture_drop_action(Ok((Ok(()), false)), PATTERN, panicking);

        assert_eq!(
            reported(action, &format!("panicking={panicking}")),
            vec![quit_line()],
            "panicking={panicking}"
        );
    }
}

/// A stopped cleanup fails a passing test: a panic with the stopped
/// message, after the quit line when the quit ran out of time too.
#[test]
fn fixture_drop_action_of_a_stopped_cleanup_in_a_passing_test_panics_naming_the_step() {
    for (quit_finished, expected_report) in [(true, vec![]), (false, vec![quit_line()])] {
        let action = fixture_drop_action(Ok((Err(STEP.to_owned()), quit_finished)), PATTERN, false);

        match action {
            FixtureDropAction::Panic { report, message } => {
                assert_eq!(
                    report, expected_report,
                    "quit_finished={quit_finished}: the lines before the panic"
                );
                assert_eq!(
                    message,
                    stopped_message(),
                    "quit_finished={quit_finished}: the panic names the step"
                );
            }
            other => panic!("quit_finished={quit_finished}: the drop panics; got {other:?}"),
        }
    }
}

/// A stopped cleanup in a test already unwinding is only reported, so the
/// test's own panic survives: the quit line first when the quit ran out of
/// time, then the stopped message.
#[test]
fn fixture_drop_action_of_a_stopped_cleanup_while_unwinding_reports_it() {
    for (quit_finished, expected) in [
        (true, vec![stopped_message()]),
        (false, vec![quit_line(), stopped_message()]),
    ] {
        let action = fixture_drop_action(Ok((Err(STEP.to_owned()), quit_finished)), PATTERN, true);

        assert_eq!(
            reported(action, &format!("quit_finished={quit_finished}")),
            expected,
            "quit_finished={quit_finished}"
        );
    }
}

/// A cleanup task that panicked, in a passing test, re-raises that task's own
/// payload, the very box it came in.
#[test]
fn fixture_drop_action_of_a_panicked_cleanup_in_a_passing_test_re_raises_its_payload() {
    let payload: Box<dyn std::any::Any + Send> = Box::new("boom");
    let address = &*payload as *const (dyn std::any::Any + Send) as *const ();

    let action = fixture_drop_action(Err(SpawnedTaskEnded::Panicked(payload)), PATTERN, false);

    match action {
        FixtureDropAction::ResumeUnwind(raised) => {
            assert_eq!(
                &*raised as *const (dyn std::any::Any + Send) as *const (), address,
                "the payload passes through unchanged"
            );
            assert_eq!(raised.downcast_ref::<&str>(), Some(&"boom"));
        }
        other => panic!("the drop re-raises the task's panic; got {other:?}"),
    }
}

/// A cleanup task that panicked, in a test already unwinding, is only
/// reported, with the panic's message: a `&str` or `String` payload's text,
/// or a stand-in for any other payload.
#[test]
fn fixture_drop_action_of_a_panicked_cleanup_while_unwinding_reports_its_message() {
    let payloads: [(Box<dyn std::any::Any + Send>, &str); 3] = [
        (Box::new("boom"), "boom"),
        (Box::new(String::from("formatted boom")), "formatted boom"),
        (Box::new(42_u32), "a panic with a non-string payload"),
    ];
    for (payload, message) in payloads {
        let action = fixture_drop_action(Err(SpawnedTaskEnded::Panicked(payload)), PATTERN, true);

        assert_eq!(
            reported(action, message),
            vec![panicked_line(message)],
            "payload reported as {message:?}"
        );
    }
}

/// A cleanup task the runtime's shutdown dropped unreported never fails the
/// test: it is reported, whether or not the thread is unwinding.
#[test]
fn fixture_drop_action_of_a_never_reported_cleanup_reports_it() {
    for panicking in [false, true] {
        let action = fixture_drop_action(Err(SpawnedTaskEnded::NeverReported), PATTERN, panicking);

        assert_eq!(
            reported(action, &format!("panicking={panicking}")),
            vec![never_reported_line()],
            "panicking={panicking}"
        );
    }
}
