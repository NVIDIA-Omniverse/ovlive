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

// Regression test for b.jov — `WatchGuard` must UNWATCH on Drop so a
// guard that is dropped mid-transaction (panic, `tokio::time::timeout` drop,
// early `?` before exec()) does not leak WATCH state onto the pooled
// connection. Without the fix, the next consumer of that connection
// inherits the stale watch set: their EXEC returns nil (WATCH-canceled)
// because the *previous* caller watched a key that was later modified —
// even though the current caller watched nothing or only unrelated keys.
//
// Test strategy (single test, deterministic):
//
//   1. Build a `TestValkey` with `pool_size = 1` so `txn_pool` holds exactly
//      ONE underlying connection. With the fred 10.1.0 `ExclusivePool`
//      round-robin (`counter % clients.len()` — see fred's `pool.rs::acquire`)
//      every `acquire().await` necessarily returns a guard wrapping that
//      single connection. This is the ONLY reliable way to prove two
//      sequential acquires hit the same Valkey session — without it the
//      bug is invisible under multi-slot pools.
//
//   2. Acquire guard A, wrap in `WatchGuard::new`, call `.watch(key_X)`,
//      call `mark_watched()`, drop the `WatchGuard` without `exec()` /
//      `unwatch()`. The `WatchGuard` Drop impl is responsible for issuing
//      UNWATCH on the held connection before the mutex is released.
//
//   3. Externally modify `key_X` via the SEPARATE data pool (shares the
//      Valkey server but NOT the txn_pool's connection). On Valkey, WATCH
//      trips when the watched key is modified by ANY connection after
//      WATCH was issued and before EXEC. If the stale WATCH from guard A
//      is still present on the connection, step 5's EXEC will see it
//      and return nil.
//
//   4. Acquire guard B on the same `txn_pool` (pool_size=1 guarantees the
//      same connection). Issue a brand-new `MULTI { SET key_Y }` / `EXEC`
//      on an UNRELATED key (no WATCH by guard B).
//
//   5. Assert `EXEC` returns a non-null value (transaction committed and
//      `key_Y` now holds the written value). Decode as `fred::types::Value`
//      so a WATCH-canceled nil reply cannot be silently coerced to
//      `Ok(())` — the exact mistake `src/storage/layer.rs` calls out when
//      it decodes EXEC as `Value` (not `()`). If guard A leaked WATCH,
//      `SET key_X` from step 3 trips it and EXEC is `Value::Null`, which
//      this test rejects.
//
// Why this test is deterministic under Option 1 (async-spawn UNWATCH):
//   The engineer's Drop impl does NOT release the `OwnedMutexGuard` back
//   to the pool before UNWATCH completes — the spawned task owns the
//   guard and only drops it after `unwatch().await` returns. So step 4's
//   `acquire().await` CANNOT succeed until UNWATCH has landed on the
//   wire. No barrier, sleep, or scheduler hint needed.
//
// Why NOT a negative assertion inside this test: the team-lead's spec
// calls for one integration test that verifies the "unrelated transaction
// is not spuriously cancelled" contract. The pre-fix failing case is
// verified manually by the test author (Test Writer's responsibility per
// workflow): temporarily neuter the Drop impl in `watch_guard.rs` and
// re-run — the single assertion below must flip from pass to fail. That
// keeps the production test minimal and aligned with the ticket's ask.

use std::ops::Deref;

use fred::interfaces::{KeysInterface, TransactionInterface};
use fred::types::Value as FredValue;

use live_edit::storage::test_support::TestValkey;
use live_edit::storage::watch_guard::WatchGuard;

/// b.jov — dropping a `WatchGuard` mid-WATCH (no explicit `unwatch()` /
/// `exec()`) must not leak the watch set onto the pooled connection. A
/// subsequent acquirer running an UNRELATED `MULTI {SET key_Y} EXEC` on the
/// same connection must see its EXEC succeed even when the previously-
/// watched `key_X` is modified between the drop and the new EXEC.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_guard_drop_without_exec_does_not_leak_watch_to_next_acquirer() {
    // pool_size = 1 — the `ExclusivePool` has exactly one slot, so guards A
    // and B provably share one connection. Anything larger and the second
    // `acquire()` might return a different slot (no stale WATCH on it) and
    // the test would pass vacuously even against the buggy pre-fix code.
    let tv = TestValkey::setup_with_pool_size(1).await;
    let keys = tv.key_schema();
    let client = tv.client();

    // Use the KeySchema's `obj_header` helper to get two keys whose names
    // are comfortably namespaced under the per-test prefix. They do not
    // share the same hash tag, but that's fine — we only issue single-key
    // commands and single-key MULTI/EXEC, never a cross-slot transaction.
    let key_x = keys.obj_header("watchleak-X");
    let key_y = keys.obj_header("watchleak-Y");

    // Seed `key_x` with an initial value so there's something to WATCH and
    // to later modify.
    let _: () = client
        .pool
        .set(&key_x, "initial", None, None, false)
        .await
        .expect("seed SET key_x must succeed");

    // ── Step 1: acquire guard A, WATCH key_x, drop without exec/unwatch. ──
    //
    // This simulates a `tokio::time::timeout`-dropped future mid-WATCH: the
    // caller had started a transaction, issued WATCH, and then their future
    // was dropped before they could call exec() or unwatch().
    {
        let raw_guard_a = client.acquire_txn().await;
        let mut guard_a = WatchGuard::new(raw_guard_a, client.watch_unwatch_tracker());
        guard_a
            .deref()
            .watch::<&str>(&key_x)
            .await
            .expect("WATCH key_x on guard A must succeed");
        guard_a.mark_watched(key_x.as_str(), &[]);
        // Drop guard_a WITHOUT calling exec() or unwatch(). The `WatchGuard`
        // Drop impl must UNWATCH before releasing the mutex; if it does not,
        // the `SET key_x` below trips the stale WATCH and the later EXEC on
        // guard B returns nil.
    } // <- WatchGuard dropped here; fix must UNWATCH before releasing the slot.

    // ── Step 2: externally modify key_x on a different connection. ──
    //
    // Under the bug, the stale WATCH on the txn_pool's single connection
    // is armed against `key_x`. This SET (issued on the shared data pool,
    // NOT the txn_pool's exclusive slot) trips that WATCH, which will
    // cancel the next EXEC issued on the txn_pool connection — even
    // though that next caller never watched `key_x` themselves.
    let _: () = client
        .pool
        .set(&key_x, "modified-externally", None, None, false)
        .await
        .expect("external SET key_x must succeed");

    // ── Step 3: acquire guard B (same slot, same connection). ──
    //
    // With pool_size=1 and the Option 1 fix holding the mutex across the
    // spawned UNWATCH, this `acquire().await` can only return once guard
    // A's Drop has finished issuing UNWATCH. No extra synchronization is
    // needed.
    let raw_guard_b = client.acquire_txn().await;
    // guard B is NOT expected to issue WATCH — the whole point of this
    // test is that a caller that didn't WATCH should not be cancelled by
    // a previous caller's abandoned WATCH. We still wrap it in the
    // production type so we exercise the real ownership story.
    let guard_b = WatchGuard::new(raw_guard_b, client.watch_unwatch_tracker());

    // ── Step 4: MULTI { SET key_y } EXEC. ──
    //
    // Queue an unrelated single-key SET inside a MULTI/EXEC transaction on
    // guard B's (same underlying) connection.
    let trx = guard_b.deref().multi();
    let _: () = trx
        .set(&key_y, "committed-by-guard-b", None, None, false)
        .await
        .expect("MULTI queue SET key_y must succeed");

    // Decode as `fred::types::Value` (NOT `()`): fred's `FromValue for ()`
    // silently accepts `Value::Null`, which would mask the exact regression
    // we're catching. See the commentary in
    // `src/storage/layer.rs::update_layer_once_watch` — same nil-EXEC decoding
    // lesson as b.8q1. Here we want the raw value so we can assert it is
    // NOT null.
    let exec_reply: FredValue = trx
        .exec(true)
        .await
        .expect("EXEC on guard B must not error at the fred layer");

    assert!(
        !exec_reply.is_null(),
        "EXEC on guard B's unrelated `MULTI {{ SET key_y }}` returned nil \
         (Value::Null) — this indicates guard A's abandoned WATCH on key_x \
         leaked onto the pooled connection and was tripped by the external \
         SET key_x. The WatchGuard Drop impl must issue UNWATCH before \
         releasing the connection so downstream acquirers inherit a clean \
         session. Got reply: {exec_reply:?}"
    );

    // Belt-and-suspenders — confirm key_y actually holds the committed value
    // (nil EXEC would have discarded the SET; a non-nil EXEC from a buggy
    // decode layer would hypothetically let the committed value still be
    // absent). Read via a fresh command on the data pool so we don't
    // accidentally observe queued-but-not-committed state.
    let stored: Option<String> = client
        .pool
        .get(&key_y)
        .await
        .expect("GET key_y must succeed");
    assert_eq!(
        stored.as_deref(),
        Some("committed-by-guard-b"),
        "key_y must hold the value written by guard B's MULTI/EXEC; \
         a missing or wrong value indicates the transaction was silently \
         discarded by a stale WATCH leaked from guard A's Drop"
    );
}

/// b.f4f — the `WatchGuard` spawned-UNWATCH arm returns its **semaphore permit**
/// as well as its slot.
///
/// This is the one shape in the codebase where a `txn_pool` checkout outlives
/// `Drop::drop`'s return: with `has_watched = true`, `WatchGuard::drop` moves the
/// whole `CheckedOutClient` into a spawned body that issues UNWATCH and only then
/// lets it go (`src/storage/watch_guard.rs`). Every other release happens inline,
/// so a test that covered only the inline arm would discriminate nothing about
/// this one — and this is precisely the arm where a permit can go missing without
/// anything failing loudly: the pool would simply be one slot smaller for the
/// life of the process, and the symptom would surface later as unexplained
/// checkout latency under load.
///
/// **What this pins, and what it deliberately does not.** It pins the permit
/// accounting: a second acquirer provably parks on the permit queue while the
/// guard is alive (`waiters() == 1` — a queue that did not exist before b.f4f),
/// and every permit is back once the tracker has drained. It does NOT re-pin
/// "the next borrower inherits a clean session": `acquire_txn_for`'s own checkout
/// probe issues `DISCARD` + `UNWATCH` before handing the slot over
/// (`probe_txn_slot`), so a borrower's clean session is guaranteed by the probe
/// whatever `Drop` did. The sibling test above owns that property through the
/// pre-probe era's shape; conflating the two here would produce an assertion
/// that cannot fail.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn watch_guard_spawned_unwatch_returns_its_permit_as_well_as_its_slot() {
    use std::time::Duration;

    /// Unreachable on the healthy path; a deadline, not a measurement.
    const SETTLE_DEADLINE: Duration = Duration::from_secs(10);

    // One txn slot, so the permit count is 1 and a missing permit is the
    // difference between a working pool and a dead one.
    let tv = TestValkey::setup_with_pool_size(1).await;
    let keys = tv.key_schema();
    let client = tv.client();
    let key_x = keys.obj_header("f4f-permit-X");
    let key_y = keys.obj_header("f4f-permit-Y");

    let _: () = client
        .pool
        .set(&key_x, "initial", None, None, false)
        .await
        .expect("seed SET key_x must succeed");

    assert_eq!(
        client.txn_pool.available_permits(),
        1,
        "fixture self-check: a one-slot txn pool starts with exactly one permit"
    );

    // ── A holds the slot with a WATCH armed, so its Drop takes the spawned arm.
    let mut guard_a = live_edit::storage::watch_guard::WatchGuard::new(
        client.acquire_txn_for(&key_x).await,
        client.watch_unwatch_tracker(),
    );
    guard_a
        .deref()
        .watch::<&str>(&key_x)
        .await
        .expect("WATCH key_x must succeed");
    guard_a.mark_watched(key_x.as_str(), &[]);

    // ── B parks. On the permit queue, which is the b.f4f change: pre-fix it
    //    would have parked inside `ExclusivePool::acquire` on one selected
    //    mutex, invisible to any instrument.
    let waiter_client = client.clone();
    let waiter_key_x = key_x.clone();
    let waiter_key_y = key_y.clone();
    let waiter = tokio::spawn(async move {
        let checkout = waiter_client.acquire_txn_for(&waiter_key_x).await;
        let trx = checkout.deref().multi();
        let _: () = trx
            .set(&waiter_key_y, "committed-by-waiter", None, None, false)
            .await
            .expect("MULTI queue SET key_y must succeed");
        let reply: FredValue = trx
            .exec(true)
            .await
            .expect("EXEC must not error at the fred layer");
        reply
    });

    let deadline = std::time::Instant::now() + SETTLE_DEADLINE;
    while client.txn_pool.waiters() == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "the second acquirer never registered on the txn permit queue. Either it was handed \
             a slot while the WatchGuard still holds one — permits and slots have diverged — or \
             it is parked somewhere the checkout does not account for"
        );
        tokio::task::yield_now().await;
    }
    assert_eq!(
        client.txn_pool.available_permits(),
        0,
        "the live WatchGuard must still hold the pool's only permit while a caller waits on it"
    );

    // ── Drop A. The checkout — slot, gauge token and permit together — moves
    //    into the spawned UNWATCH body and is released only when that body ends.
    drop(guard_a);

    let reply = tokio::time::timeout(SETTLE_DEADLINE, waiter)
        .await
        .expect(
            "the parked acquirer never got the slot after the WatchGuard was dropped. The \
             spawned UNWATCH body owns the permit, so a body that never completes — or one that \
             leaked the permit instead of dropping it — strands every waiter on this pool",
        )
        .expect("the waiter task must not panic");
    assert!(
        !reply.is_null(),
        "the waiter's unrelated MULTI/EXEC was discarded: {reply:?}"
    );

    // ── Everything is back. This is the assertion the spawned arm exists to
    //    risk: one permit, no in-flight UNWATCH, and the pool still usable.
    let deadline = std::time::Instant::now() + SETTLE_DEADLINE;
    while client.watch_unwatch_inflight() > 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "the spawned UNWATCH task never finished"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        client.txn_pool.available_permits(),
        1,
        "a permit went missing across the WatchGuard's spawned UNWATCH. The pool is now \
         permanently one slot smaller than its slot count, which surfaces only as unexplained \
         checkout latency under load — never as an error"
    );
    assert_eq!(
        client.txn_pool.waiters(),
        0,
        "the waiter gauge did not return to zero, so a `WaiterSlot` guard leaked"
    );

    // …and the pool genuinely still works, not merely arithmetically tidy.
    let recheck = tokio::time::timeout(SETTLE_DEADLINE, client.acquire_txn_for(&key_x))
        .await
        .expect("a pool whose permits are all back must hand its slot out immediately");
    drop(recheck);
}
