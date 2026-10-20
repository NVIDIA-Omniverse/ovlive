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

// Service-owned checkout for fred's `ExclusivePool` (the `txn_pool`, b.f4f).
//
// fred 10.1.0's `ExclusivePool::acquire()` scans its slots once for a free
// mutex and, finding none, awaits **one selected mutex**
// (`fred-10.1.0/src/clients/pool.rs:633-646`). A caller parked on that one
// mutex stays parked while every other slot in the pool goes idle, skewing
// checkout latency under saturation for no reason.
//
// This module replaces that call entirely. `ExclusivePool::clients()` is public
// and hands back `&[RefCount<AsyncMutex<Client>>]` — the mutexes themselves
// (`fred-10.1.0/src/clients/pool.rs:510-512`) — so the whole checkout runs
// in-process:
//
//   1. take a permit from a tokio `Semaphore` sized to the slot count
//      (`Semaphore` is FIFO-fair, which is where fairness comes from), then
//   2. scan every slot with `try_lock_owned()`, which never blocks.
//
// fred's `lock_owned().await` fallback is not merely unreached — it is **absent
// from the call path**.
//
// ── Why the scan cannot come up empty ──────────────────────────────────────
//
// Let `h` be the number of permit holders that already hold a slot, `s` the
// number currently scanning, and `w` the number holding a permit while they
// await one specific slot (the targeted checkouts:
// [`ExclusiveCheckout::checkout_slot_for_router_restart`], the router-restart
// repair, and [`ExclusiveCheckout::checkout_slot_for_replacement`], a `txn`
// slot's client replacement). Permits == slots == N and every permit holder
// holds or awaits at most one slot, so `h + s + w <= N`.
//
// A targeted waiter can make at most one slot unavailable to scanners beyond
// the `h` held ones: the slot it awaits, which tokio's FIFO mutex hands to it
// directly on release instead of freeing it for a `try_lock`. So the slots a
// scanner can take number at least `N - h - w >= s >= 1`. With one targeted
// waiter that is `h + s + 1 <= N`, so at least `s + 1` slots are free and at
// most one of them is reserved; each further targeted waiter adds one to both
// sides. Each scanner or targeted waiter that takes its
// slot becomes a holder, which keeps the inequality. Apart from those, the
// locked set during a scan only shrinks — a holder does not release and retake
// mid-operation. A scanner's full pass therefore always finds a free slot.
//
// Two properties carry that argument, and both are structural:
//
// - **Every acquisition of these mutexes goes through this module, because no
//   other module can name one.** The `ExclusivePool` is a private field and
//   nothing hands out a `&[Arc<Mutex<Client>>]`. The mutex slice
//   (`ExclusivePool::clients()`) has these uses, all here: the checkout scan
//   above (`try_take_slot`); [`ExclusiveCheckout::new`], which `try_lock`s
//   every slot once, non-blocking, to read its client id while it wraps a pool
//   nothing else can reach yet; and the named methods the operations outside
//   this module that need pooled clients get, which expose no mutex at all —
//   [`ExclusiveCheckout::clients_for_startup_hooks`], which returns `Client`
//   *clones* for `ValkeyClient::install_observability_hooks` to attach metric
//   hooks to, and the targeted checkouts
//   ([`ExclusiveCheckout::checkout_slot_for_router_restart`],
//   [`ExclusiveCheckout::checkout_slot_for_replacement`]), which take a permit
//   like any checkout. [`ExclusiveCheckout::quit`], for
//   shutdown, delegates to fred's `quit` instead (below).
//   (`ValkeyClient::connect` starts each slot's router before it wraps the
//   pool here, while nothing else can reach a slot.) Everything
//   fred itself offers that locks slots (`init`, `connect_pool`,
//   `wait_for_connect`, `force_reconnection`, `update_perf_config`, `quit` —
//   `fred-10.1.0/src/clients/pool.rs:516-620`, `:673`) is likewise unreachable
//   from outside. That, not a source-text fence, is the proof; the widened
//   `no_unpinned_acquire_txn_call_site_survives_under_src` fence in
//   `tests/integration/storage/txn_checkout_test.rs` is a regression guard on
//   top of it, catching a future edit that re-opens the encapsulation.
// - **The permit is released strictly after the slot.** [`CheckedOutClient`]
//   bundles the guard and the checkout's permit in one value whose field
//   declaration order makes the mutex release first and the permit last (Rust
//   drops struct fields in declaration order). A permit that freed while its
//   slot was still locked would be over-admission: the next waiter would wake
//   into a scan with nothing free. The same property disposes of task *abort*:
//   an aborted task releases both together, slot first, on its next poll.
//
// ── Replacing a slot's client ─────────────────────────────────────────────
//
// A slot's `Client` can be swapped for a fresh one (`storage::member_replacement`,
// a `txn` slot whose client keeps getting fast unresponsive verdicts). The swap
// happens only under that slot's guard, taken by
// [`ExclusiveCheckout::checkout_slot_for_replacement`] with a permit like any
// checkout — so never while a caller has the slot mid-transaction: the
// replacement waits until the slot is released — and is
// [`ExclusiveCheckout::swap_slot_client`], which writes the fresh client into
// the mutex in place (fred's slot holds an `AsyncMutex<Client>`, so no fred
// change is needed) and re-keys the slot's id. The pool, its mutexes, the
// permit count and the scan are untouched, so nothing above changes; the next
// checkout of that slot simply gets the fresh client.
//
// Shutdown is the one interval in which the first property genuinely does not
// hold, and it is handled explicitly rather than argued away: `ExclusivePool::
// quit` locks every slot concurrently while holding none of our permits, so
// [`ExclusiveCheckout::quit`] closes the permit semaphore *before* delegating to
// it. Past that point no new permit is granted, and a checkout reads
// `is_quitting()` as "shutdown took the slots" and waits permit-less instead of
// escalating to `report_escaped_slot`. The bounded re-scan below is defence in
// depth against a future edit that breaks one of the two properties, not a
// mechanism any path relies on.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use fred::clients::{Client, ExclusivePool};
use fred::interfaces::ClientLike;
use tokio::sync::{OwnedMutexGuard, OwnedSemaphorePermit, Semaphore};

use crate::observability::metrics::SharedMetrics;
use crate::storage::client::PoolUseToken;

/// Budget for the bounded re-scan that follows a full pass finding no free slot.
///
/// **Defence in depth, not a mechanism any path relies on.** No path
/// legitimately scans empty while holding a permit (outside
/// [`ExclusiveCheckout::quit`]): an aborted task's checkout releases its slot
/// before its permit. The re-scan exists so that a future edit breaking that
/// property degrades to one re-parked caller plus a loud report, instead of an
/// immediate failure. Exhausting it means a slot escaped its permit, which is
/// why it is loud.
pub const CHECKOUT_RESCAN_DEADLINE: Duration = Duration::from_secs(3);

/// Gap between re-scans after the first (which follows a bare
/// [`tokio::task::yield_now`], the cheapest retry for a defence-in-depth path).
/// A 1 ms sleep rather than a tighter yield loop so an impossible state cannot
/// burn a core for [`CHECKOUT_RESCAN_DEADLINE`].
const CHECKOUT_RESCAN_INTERVAL: Duration = Duration::from_millis(1);

/// A checked-out exclusive Valkey connection: the slot, its gauge token, and
/// the checkout's semaphore permit.
///
/// **The field order is the contract.** Rust drops struct fields in declaration
/// order, so `guard` releases the pool mutex first, `token` then ticks
/// `valkey_pool_in_use{pool}` down, and `entitlement` drops last — after the
/// slot is provably free. That holds on every way the value can drop: a normal
/// return, an early `?`, a cancelled or aborted task, a panic, runtime teardown.
///
/// `Deref<Target = Client>` is preserved so no command site changes.
pub struct CheckedOutClient {
    guard: OwnedMutexGuard<Client>,
    token: PoolUseToken,
    /// What entitled this checkout to its slot, dropped last: the permit this
    /// checkout acquired together with the slot, or `None` for a checkout taken
    /// after [`ExclusiveCheckout::quit`] closed the semaphore, when no permit
    /// can be had (see [`ExclusiveCheckout::acquire`]).
    entitlement: Option<SlotEntitlement>,
}

/// The entitlement a [`CheckedOutClient`] carries in its last field.
///
/// Nothing reads it; it exists only to be dropped after the slot guard. Outside
/// this module it is only carried, through [`CheckedOutClient::into_parts`] and
/// back.
pub(crate) enum SlotEntitlement {
    /// The checkout's own permit.
    Permit { _permit: OwnedSemaphorePermit },
}

impl std::ops::Deref for CheckedOutClient {
    type Target = Client;

    fn deref(&self) -> &Client {
        &self.guard
    }
}

impl CheckedOutClient {
    /// Split the checkout into its slot guard, its gauge token and its
    /// entitlement.
    ///
    /// Only for the one caller that must hand all three to a spawned task and
    /// get them back again ([`ValkeyClient::recycle_txn_slot`], whose repair
    /// body needs a bare `&Client`). Reassembling is
    /// [`Self::reassemble`]; between the two the drop-order contract above is
    /// the caller's to preserve, which is why this is `pub(crate)` and has
    /// exactly one user.
    ///
    /// [`ValkeyClient::recycle_txn_slot`]: crate::storage::client::ValkeyClient
    pub(crate) fn into_parts(
        self,
    ) -> (
        OwnedMutexGuard<Client>,
        PoolUseToken,
        Option<SlotEntitlement>,
    ) {
        (self.guard, self.token, self.entitlement)
    }

    /// Inverse of [`Self::into_parts`].
    pub(crate) fn reassemble(
        guard: OwnedMutexGuard<Client>,
        token: PoolUseToken,
        entitlement: Option<SlotEntitlement>,
    ) -> Self {
        Self {
            guard,
            token,
            entitlement,
        }
    }
}

impl std::fmt::Debug for ExclusiveCheckout {
    /// Hand-written because neither `ExclusivePool` nor `SharedMetrics` is
    /// `Debug`; reports the numbers an operator or a failing assertion actually
    /// wants (`ValkeyClient`'s own `Debug` renders this in place of the raw
    /// pool it used to print).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExclusiveCheckout")
            .field("pool", &self.label)
            .field("slots", &self.slots())
            .field("available_permits", &self.permits.available_permits())
            .field("waiters", &self.waiters.load(Ordering::Acquire))
            .finish()
    }
}

/// The service's own next-free checkout over one fred `ExclusivePool`.
///
/// Not `Clone`: every clone of a `ValkeyClient` must share ONE permit semaphore,
/// or the process would hold `clones × slots` permits against one set of slots —
/// exactly the over-admission the permit exists to prevent. `ValkeyClient` holds
/// it behind an `Arc` for that reason.
pub struct ExclusiveCheckout {
    pool: ExclusivePool,
    /// The fred client id of each slot, in slot order, read at construction
    /// and updated by [`Self::swap_slot_client`] under the slot's guard. A
    /// `Client`'s id is minted in `Client::new` and survives every reconnect,
    /// so only a swap changes it. Resolves a targeted checkout's id to a slot
    /// without locking any other slot.
    slot_ids: std::sync::Mutex<Vec<String>>,
    /// Exactly `pool.clients().len()` permits. FIFO-fair, which is the whole
    /// fairness story; the scan that follows is non-blocking.
    permits: Arc<Semaphore>,
    /// The pool label (`POOL_LABEL_TXN`) for every instrument below.
    label: &'static str,
    /// Callers currently parked waiting for a permit, for
    /// `live_edit_valkey_pool_waiters{pool}`.
    waiters: AtomicUsize,
    /// Shared with `ValkeyClient`; `None` until `install_observability_hooks`
    /// runs, which is the state startup PINGs and metric-free tests operate in.
    metrics: Arc<std::sync::Mutex<Option<SharedMetrics>>>,
    /// Test-only: suppress the post-yield re-scan, making an empty **first**
    /// scan terminal (b.f4f).
    ///
    /// Exists because the re-scan is exactly what hides the bug the permit
    /// ordering prevents. [`CheckedOutClient`] releases its guard before its
    /// permit; if that order were ever inverted, a woken waiter's first scan
    /// would find nothing — and the `yield_now()` re-scan would then silently
    /// rescue it, so the defect would be invisible to any test that only checks
    /// "the waiter eventually got a slot". With this set, the first scan is the
    /// whole test: a waiter woken by a `CheckedOutClient` drop must find the
    /// freed slot immediately, or [`ExclusiveCheckout::acquire`] panics and the
    /// assertion fails.
    ///
    /// **What it does and does not buy** — see
    /// [`ExclusiveCheckout::disable_rescan_for_test`], which states the
    /// multi-thread-runtime requirement in full. In short: this removes the
    /// *rescue*, not the raciness. An inverted drop order is only ever
    /// observable on a multi-thread runtime, so a test built on this seam must
    /// be `#[tokio::test(flavor = "multi_thread")]`.
    ///
    /// Default `false`, and when it is `false` every checkout path runs exactly
    /// as it does in a release build — the flag is read once per empty scan,
    /// which is a path production does not take.
    #[cfg(any(test, feature = "test-fault-injection"))]
    rescan_disabled: std::sync::atomic::AtomicBool,
}

impl ExclusiveCheckout {
    /// Wrap a freshly built `ExclusivePool`, sizing the permit count to its slot
    /// count. `metrics` is the same slot `ValkeyClient` holds, so a checkout
    /// taken before `install_observability_hooks` simply records nothing.
    ///
    /// Reads each slot's client id with a non-blocking `try_lock`: nothing can
    /// hold a slot yet, because the pool has not been handed to anyone. A slot
    /// that is somehow held records an empty id, and the paths that find a
    /// slot by id lose it: [`Self::checkout_slot_for_router_restart`] never
    /// restarts its router, [`Self::checkout_slot_for_replacement`] and
    /// [`Self::swap_slot_client`] never replace its client, and
    /// [`Self::slot_of`] names no slot for it, so its hooks are armed with no
    /// verdict sink (`MemberReplacer::arm_txn`). Checkouts by callers, which
    /// scan, are unaffected.
    pub fn new(
        pool: ExclusivePool,
        label: &'static str,
        metrics: Arc<std::sync::Mutex<Option<SharedMetrics>>>,
    ) -> Self {
        let slots = pool.size();
        let slot_ids = pool
            .clients()
            .iter()
            .map(|slot| {
                slot.try_lock()
                    .map(|client| client.id().to_string())
                    .unwrap_or_default()
            })
            .collect();
        Self {
            pool,
            slot_ids: std::sync::Mutex::new(slot_ids),
            permits: Arc::new(Semaphore::new(slots)),
            label,
            waiters: AtomicUsize::new(0),
            metrics,
            #[cfg(any(test, feature = "test-fault-injection"))]
            rescan_disabled: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Make an empty first scan terminal instead of yielding and re-scanning —
    /// **test-only** (b.f4f).
    ///
    /// Turns [`Self::acquire`] into a single-pass probe so a test can
    /// assert the [`CheckedOutClient`] drop order directly: hold every slot but
    /// one, park a waiter, drop a checkout, and require the waiter's *first*
    /// scan to find the freed slot. Under the production re-scan that assertion
    /// cannot fail, because a permit released an instant before its guard would
    /// be rescued by the yield.
    ///
    /// [`Self::acquire`] — which is infallible — `panic!`s on the empty first
    /// scan, failing the waiter task. (A panic rather than the ordinary
    /// escalation because `report_escaped_slot`'s only hard verdict is a
    /// `debug_assert!`, which `cargo test --release` compiles away; see the call
    /// site.) The *first* scan is the verdict — **unless [`Self::quit`] has
    /// closed the permit semaphore**, in which case `acquire` falls through to
    /// its `is_quitting()` arm without escalating.
    /// Shutdown legitimately holds every slot, so an empty scan there proves
    /// nothing about the drop order, and a seam that accused it would be
    /// reporting the one thing `quit`'s contract promises it will not. A fixture
    /// that wants the seam's verdict must therefore finish before it quits the
    /// client.
    ///
    /// # A test built on this MUST be `#[tokio::test(flavor = "multi_thread")]`
    ///
    /// This seam removes the **rescue**, not the raciness, and the window it
    /// exposes does not exist at all on a single-threaded runtime.
    /// `CheckedOutClient`'s `Drop` releases guard → token → entitlement
    /// synchronously, within one statement and with no await between them. On a
    /// current-thread runtime no other task can be polled inside that statement,
    /// so even a hypothetically inverted order would be repaired before the
    /// waiter ever ran, and the assertion would pass whatever the field order
    /// said — a green test that proves nothing.
    ///
    /// On a multi-thread runtime the waiter can be running on another worker at
    /// the instant the permit is released, which is what makes an inverted order
    /// observable: it would wake, scan, find nothing, and — with the re-scan
    /// suppressed — fail. That is the whole mechanism, so the flavour is part of
    /// the seam's contract rather than an incidental detail of how the test is
    /// written.
    ///
    /// # …and the test MUST join the waiter's `JoinHandle`
    ///
    /// The two requirements are the same requirement seen twice: the section
    /// above puts the waiter on its own task, and `tokio` catches a panic in a
    /// spawned task and surfaces it **only** through that task's `JoinHandle`. A
    /// detached waiter therefore panics, prints to stderr, and leaves the
    /// `#[tokio::test]` green — the failure is visible only to whoever happens
    /// to read the captured output. Await the handle and propagate the outcome
    /// (`handle.await.expect(...)`, or unwrap the `JoinError`) so the panic
    /// actually fails the test.
    ///
    /// Not reversible and not meant to be: a fixture installs it once, before
    /// any checkout, and throws the client away afterwards.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn disable_rescan_for_test(&self) {
        self.rescan_disabled
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Whether the re-scan has been suppressed. Test builds read the flag;
    /// release builds get the constant-`false` twin below, so the branch that
    /// consults it is dead code the optimiser removes and the field does not
    /// exist at all.
    #[cfg(any(test, feature = "test-fault-injection"))]
    fn rescan_suppressed(&self) -> bool {
        self.rescan_disabled
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Release-build twin of [`Self::rescan_suppressed`]. Always `false`.
    #[cfg(not(any(test, feature = "test-fault-injection")))]
    fn rescan_suppressed(&self) -> bool {
        false
    }

    /// Number of exclusive slots — i.e. the permit count.
    pub fn slots(&self) -> usize {
        self.pool.size()
    }

    /// Snapshot one `Client` handle per slot, for
    /// `ValkeyClient::install_observability_hooks` to attach its reconnect /
    /// error hooks to.
    ///
    /// **This is the only way any code outside this module reaches a pooled
    /// client, and it hands back `Client` clones, never the mutexes.** fred
    /// `Client`s are `Clone` and a clone shares the inner state (and therefore
    /// the broadcast channels the hooks subscribe to), so the caller gets
    /// everything it needs without being able to take a slot. That is what makes
    /// "every acquisition of a pool mutex goes through this module" a property of
    /// the type system rather than of a source-text fence — see the module docs.
    ///
    /// **Startup only.** It locks every slot in turn, so calling it while the
    /// server is serving would hold slots that no permit accounts for and a
    /// concurrent scanner could find nothing free. `install_observability_hooks`
    /// runs once, before the gRPC server accepts a connection; keep it that way.
    pub(crate) async fn clients_for_startup_hooks(&self) -> Vec<Client> {
        let mut clients = Vec::with_capacity(self.pool.size());
        for locked in self.pool.clients() {
            clients.push(locked.lock().await.clone());
        }
        clients
    }

    /// Close this pool's connections at shutdown.
    ///
    /// **Closes the permit semaphore first, and that ordering is load-bearing**
    /// (b.f4f). `ExclusivePool::quit` locks every slot concurrently
    /// (`fred-10.1.0/src/clients/pool.rs:673-683`) while holding no permits of
    /// ours, so for the duration of the `QUIT` the pool looks fully occupied to a
    /// scanner. A checkout inside that window would otherwise hold a valid
    /// permit, find nothing free for the whole [`CHECKOUT_RESCAN_DEADLINE`], and
    /// trip
    /// `report_escaped_slot`'s `debug_assert!` — a false accusation that the
    /// soundness argument is broken, when in fact shutdown simply took the slots.
    /// Closing the semaphore makes that window observable instead: no further
    /// entitlement is granted, and every checkout path below reads
    /// `permits.is_closed()` as "the pool is being torn down" and degrades
    /// accordingly rather than escalating.
    ///
    /// Irreversible, which is correct: `ValkeyClient::quit` is terminal.
    pub(crate) async fn quit(&self) -> Result<(), fred::error::Error> {
        self.permits.close();
        self.pool.quit().await
    }

    /// Whether [`Self::quit`] has closed the permit semaphore — i.e. this pool is
    /// being torn down and an empty scan is expected rather than anomalous.
    pub(crate) fn is_quitting(&self) -> bool {
        self.permits.is_closed()
    }

    /// Check out a slot **and its own permit**, parking on the FIFO permit queue
    /// for as long as it takes.
    ///
    /// Infallible, matching the contract every `txn_pool` call site is written
    /// against: a caller that cannot get a connection has no recourse but to
    /// wait, and the caller's own retry / RPC-deadline machinery already bounds
    /// how long it is willing to.
    ///
    /// **After [`Self::quit`] it proceeds without a permit.** A closed semaphore
    /// cannot hand one out, and this signature has no way to refuse; taking the
    /// slot permit-less is safe precisely because shutdown is terminal — the pool
    /// is closing, permit accounting has no future to protect, and the caller's
    /// commands will fail on the quit connection and route through its own error
    /// mapping. It is also exactly the pre-b.f4f behaviour. What it must not do
    /// is escalate: an empty scan during `ExclusivePool::quit` means shutdown
    /// took the slots, not that a slot escaped its permit.
    pub(crate) async fn acquire(&self) -> CheckedOutClient {
        let started = Instant::now();
        loop {
            let permit = self.acquire_permit().await;
            let deadline = Instant::now() + CHECKOUT_RESCAN_DEADLINE;
            let mut first_rescan = true;
            loop {
                if let Some(guard) = self.try_take_slot() {
                    self.observe_checkout(started.elapsed());
                    return CheckedOutClient {
                        guard,
                        token: self.make_pool_use_token(),
                        entitlement: permit
                            .map(|permit| SlotEntitlement::Permit { _permit: permit }),
                    };
                }
                if self.rescan_suppressed() && !self.is_quitting() {
                    // Test-only single-pass mode (`disable_rescan_for_test`).
                    // `acquire` is infallible, so the verdict cannot be a return
                    // value — an empty scan while holding a permit means a slot
                    // escaped its permit, which is exactly the defect an inverted
                    // `CheckedOutClient` drop order would produce, and the test
                    // needs to see it fail.
                    //
                    // `!is_quitting()` is load-bearing, not belt-and-braces. Once
                    // `quit` has closed the semaphore, `acquire_permit` returns
                    // `None` and `ExclusivePool::quit` holds every slot, so an
                    // empty scan here is shutdown doing exactly what it is
                    // supposed to — and the panic below would accuse a caller of
                    // "holding a permit" it demonstrably does not hold. Falling
                    // through leaves the existing `is_quitting()` arm to degrade
                    // gracefully, which is what the pre-seam `break` did and what
                    // `quit`'s own contract promises: a closed semaphore never
                    // escalates.
                    //
                    // `panic!`, not a `break` into `report_escaped_slot`. That
                    // reads as the gentler choice, but its only hard verdict is a
                    // `debug_assert!`, which compiles away under
                    // `cargo test --release` — and with the assertion gone the
                    // outer loop would spin without ever reaching either
                    // `CHECKOUT_RESCAN_INTERVAL` sleep (permit re-acquired,
                    // scan empty, break, `error!`, drop, yield, repeat), turning
                    // a failing assertion into a hot loop. A panic is
                    // deterministic in every build profile, and the task it
                    // fails is the waiter the test is asserting on.
                    //
                    // Unreachable in a release build: `rescan_suppressed()` is
                    // the constant-`false` twin there and this branch is dead.
                    panic!(
                        "ExclusiveCheckout({}) found no free slot on its first scan while holding \
                         a permit, with the re-scan suppressed by \
                         `disable_rescan_for_test`: permits == slots and a permit is released only \
                         after its guard, so this means a slot was locked outside the checkout or \
                         a permit was released before its guard",
                        self.label
                    );
                }
                if first_rescan {
                    // Defence in depth: one yield and a re-scan before the
                    // bounded retry below. No path legitimately reaches an
                    // empty scan here while holding a permit. An aborted task
                    // holding a checkout releases its slot before its
                    // entitlement, so its permit cannot wake us early — and on
                    // the `txn_pool` nothing aborts such a task anyway
                    // (`WatchGuard`'s and `recycle_txn_slot`'s spawned bodies
                    // are drained, never aborted). The yield exists so a future edit that
                    // breaks the drop order is absorbed before it escalates.
                    // See the module docs.
                    first_rescan = false;
                    self.log_rescan();
                    tokio::task::yield_now().await;
                    continue;
                }
                if Instant::now() >= deadline {
                    break;
                }
                tokio::time::sleep(CHECKOUT_RESCAN_INTERVAL).await;
            }
            if self.is_quitting() {
                // `ExclusivePool::quit` holds every slot for the duration of its
                // `QUIT`, which `ValkeyClient::quit` caps at 2 s. Expected, not
                // anomalous: keep waiting rather than accusing the permit
                // arithmetic, and take whichever slot `quit` releases first.
                tracing::debug!(
                    pool = self.label,
                    "exclusive-pool checkout found no free slot while the pool is quitting; \
                     waiting for shutdown to release one"
                );
                tokio::time::sleep(CHECKOUT_RESCAN_INTERVAL).await;
                continue;
            }
            // Past the re-scan deadline, and not shutdown: a slot escaped its
            // permit. Infallible by contract — every `txn_pool` call site takes
            // its connection unconditionally — so the
            // recovery is to release the permit (so the state we are re-testing
            // is not one we are ourselves distorting) and re-park on the FIFO
            // queue, which is exactly what a caller got before b.f4f. Loud,
            // because it means the scan's soundness argument has been broken
            // somewhere.
            self.report_escaped_slot();
            drop(permit);
            tokio::task::yield_now().await;
        }
    }

    /// Check out **one specific slot** — the one whose fred client id is
    /// `client_id` — with its own permit, for restarting that slot's ended
    /// router (`ValkeyClient`'s router-restart repair, driven by
    /// `storage::router_supervisor`). The only other caller that takes a slot
    /// by identity is [`Self::checkout_slot_for_replacement`]; both are the
    /// one targeted checkout, `checkout_slot`.
    ///
    /// Takes a permit first, exactly like [`Self::acquire`], then awaits that
    /// one slot's mutex instead of scanning. The await is what a scan never
    /// does, and the module docs show it keeps the scan sound: this caller
    /// holds a permit and no slot while it waits, so it counts against the
    /// scanners' free slots exactly as one more holder would.
    ///
    /// `None` when no slot has `client_id`, or when [`Self::quit`] has closed
    /// the semaphore: at shutdown there is nothing worth restarting. Also
    /// `None` when the slot's client was replaced while this waited for it
    /// ([`Self::swap_slot_client`]): the router to restart belonged to the
    /// client that left the slot.
    pub(crate) async fn checkout_slot_for_router_restart(
        &self,
        client_id: &str,
    ) -> Option<CheckedOutClient> {
        self.checkout_slot(client_id).await
    }

    /// Check out the slot whose fred client id is `client_id`, with its own
    /// permit, so `storage::member_replacement` can swap a fresh client into it
    /// under its guard ([`Self::swap_slot_client`]). Waits for as long as a
    /// caller holds the slot — a replacement never interrupts a transaction.
    /// `None` as for [`Self::checkout_slot_for_router_restart`].
    pub(crate) async fn checkout_slot_for_replacement(
        &self,
        client_id: &str,
    ) -> Option<CheckedOutClient> {
        self.checkout_slot(client_id).await
    }

    /// The targeted checkout both of the above are: a permit first, exactly
    /// like [`Self::acquire`], then that one slot's mutex, then a check that
    /// the slot still holds `client_id` (a swap may have happened while this
    /// waited; the slot is then released and `None` returned).
    async fn checkout_slot(&self, client_id: &str) -> Option<CheckedOutClient> {
        let index = self
            .slot_ids
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .position(|id| !id.is_empty() && id == client_id)?;
        let permit = self.acquire_permit().await?;
        let guard = self.pool.clients()[index].clone().lock_owned().await;
        if guard.id() != client_id {
            return None;
        }
        Some(CheckedOutClient {
            guard,
            token: self.make_pool_use_token(),
            entitlement: Some(SlotEntitlement::Permit { _permit: permit }),
        })
    }

    /// Swap `fresh` into the slot `checkout` holds, under that slot's guard,
    /// and return the client it held — which no caller can reach any more:
    /// every caller reaches a slot's client only through a checkout, and this
    /// one holds the slot. The slot's id becomes `fresh`'s, so a later targeted
    /// checkout finds it under the new id and none under the old. The pool,
    /// the permits and the scan are unchanged (see the module docs). `None`
    /// when `checkout`'s client is not one of this pool's slots, in which case
    /// nothing changes.
    pub(crate) fn swap_slot_client(
        &self,
        checkout: &mut CheckedOutClient,
        fresh: Client,
    ) -> Option<Client> {
        let mut slot_ids = self
            .slot_ids
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let index = slot_ids
            .iter()
            .position(|id| !id.is_empty() && id.as_str() == checkout.guard.id())?;
        slot_ids[index] = fresh.id().to_string();
        Some(std::mem::replace(&mut *checkout.guard, fresh))
    }

    /// The slot index whose client id is `client_id`, if any.
    pub(crate) fn slot_of(&self, client_id: &str) -> Option<usize> {
        self.slot_ids
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .position(|id| !id.is_empty() && id == client_id)
    }

    /// The fred client id of each slot, in slot order: what the slots hold now,
    /// replacements included. **Test-only.**
    #[cfg(any(test, feature = "test-support"))]
    pub fn slot_client_ids_for_test(&self) -> Vec<String> {
        self.slot_ids
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Park on the FIFO permit queue, counting the wait into
    /// `live_edit_valkey_pool_waiters{pool}`.
    ///
    /// `None` means [`Self::quit`] has closed the semaphore — shutdown. It is a
    /// return value and never a panic: a checkout racing `ValkeyClient::quit` is
    /// an ordinary shutdown interleaving (a GC sweep tick, a checkpoint write, a
    /// `WatchGuard` teardown re-entering the pool), not a bug, and taking the
    /// process down over it would turn an orderly shutdown into a crash.
    async fn acquire_permit(&self) -> Option<OwnedSemaphorePermit> {
        match self.permits.clone().try_acquire_owned() {
            Ok(permit) => return Some(permit),
            Err(tokio::sync::TryAcquireError::Closed) => return None,
            Err(tokio::sync::TryAcquireError::NoPermits) => {}
        }
        let _waiter = WaiterSlot::join_unbounded(self);
        self.permits.clone().acquire_owned().await.ok()
    }

    /// One non-blocking pass over every slot. Never awaits, so the locked set it
    /// observes cannot grow underneath it.
    fn try_take_slot(&self) -> Option<OwnedMutexGuard<Client>> {
        self.pool
            .clients()
            .iter()
            .find_map(|slot| slot.clone().try_lock_owned().ok())
    }

    /// A full pass found nothing free. Not expected outside [`Self::quit`] (see
    /// the module docs); the defence-in-depth re-scan follows, and only
    /// exhausting [`CHECKOUT_RESCAN_DEADLINE`] is a fault, so a single miss is
    /// `debug!`.
    fn log_rescan(&self) {
        tracing::debug!(
            pool = self.label,
            slots = self.slots(),
            "exclusive-pool checkout found no free slot on its first pass; re-scanning after a \
             yield"
        );
    }

    /// A full pass found nothing free while this caller held an entitlement,
    /// **and it kept finding nothing for [`CHECKOUT_RESCAN_DEADLINE`]**. That
    /// means a slot escaped its permit, the one condition the soundness
    /// argument in this module's docs rules out, so it is an `error!` and a
    /// `debug_assert!`, not a retry log.
    fn report_escaped_slot(&self) {
        tracing::error!(
            pool = self.label,
            slots = self.slots(),
            available_permits = self.permits.available_permits(),
            rescan_deadline_ms = CHECKOUT_RESCAN_DEADLINE.as_millis() as u64,
            consequence = "some slot's mutex is being locked outside `ExclusiveCheckout` (only \
                           `install_observability_hooks` and `quit` may do that), or a permit was \
                           released before its guard. The caller re-parks on the permit \
                           queue",
            "exclusive-pool checkout held an entitlement but found no free slot for its whole \
             re-scan budget"
        );
        debug_assert!(
            false,
            "ExclusiveCheckout({}) found no free slot while holding an entitlement: permits == \
             slots and a permit is released only after its guard, so this is unreachable unless a \
             slot was locked outside the checkout",
            self.label
        );
    }

    fn make_pool_use_token(&self) -> PoolUseToken {
        PoolUseToken::new(self.metrics_snapshot(), self.label)
    }

    /// Callers currently parked waiting for an entitlement on this pool.
    pub fn waiters(&self) -> usize {
        self.waiters.load(Ordering::Acquire)
    }

    /// Permits not currently handed out.
    pub fn available_permits(&self) -> usize {
        self.permits.available_permits()
    }

    fn metrics_snapshot(&self) -> Option<SharedMetrics> {
        match self.metrics.lock() {
            Ok(guard) => guard.clone(),
            Err(_) => None,
        }
    }

    fn observe_checkout(&self, elapsed: Duration) {
        if let Some(metrics) = self.metrics_snapshot() {
            metrics.observe_valkey_pool_checkout(self.label, elapsed);
        }
    }
}

/// RAII counter for `live_edit_valkey_pool_waiters{pool}`.
///
/// A guard rather than a bare pair of `fetch_add` / `fetch_sub` calls because
/// the wait it covers can be dropped by a cancelled caller: the decrement has to
/// happen on the dropped-future path too, or the gauge ratchets.
struct WaiterSlot<'a> {
    checkout: &'a ExclusiveCheckout,
}

impl<'a> WaiterSlot<'a> {
    /// Join the permit queue. The semaphore *is* the queue, so there is no cap.
    fn join_unbounded(checkout: &'a ExclusiveCheckout) -> Self {
        checkout.waiters.fetch_add(1, Ordering::AcqRel);
        checkout.inc_waiters_gauge();
        Self { checkout }
    }
}

impl Drop for WaiterSlot<'_> {
    fn drop(&mut self) {
        self.checkout.waiters.fetch_sub(1, Ordering::AcqRel);
        self.checkout.dec_waiters_gauge();
    }
}

impl ExclusiveCheckout {
    fn inc_waiters_gauge(&self) {
        if let Some(metrics) = self.metrics_snapshot() {
            metrics.inc_valkey_pool_waiters(self.label);
        }
    }

    fn dec_waiters_gauge(&self) {
        if let Some(metrics) = self.metrics_snapshot() {
            metrics.dec_valkey_pool_waiters(self.label);
        }
    }
}

#[cfg(test)]
mod tests {
    //! Valkey-free unit tests for the checkout itself (b.f4f).
    //!
    //! **Nothing here connects.** `Builder::build_exclusive_pool` constructs the
    //! slots and their mutexes without opening a socket, and no test below
    //! issues a command — `acquire` is pure
    //! in-process scheduling over a semaphore and a set of mutexes. That is what
    //! makes the ticket's `ExclusivePool::acquire` probe reproducible here as an
    //! ordinary unit test, against this repository's own `Cargo.lock`, rather
    //! than as the hand-linked scratch binary the ticket describes.
    //!
    //! The waits below are **deadlines, not measurements**: the healthy path
    //! resolves in microseconds and pays none of them, and each exists so a
    //! regression fails rather than hangs. None asserts that anything was fast.

    use super::*;
    use fred::interfaces::ClientLike;
    use fred::types::Builder;
    use std::future::Future;
    use std::task::Poll;

    use crate::storage::client::POOL_LABEL_TXN;

    /// Upper bound on any wait in this module. Unreachable on the healthy path;
    /// well inside nextest's 120 s hang-killer so a regression produces a named
    /// panic rather than a SIGKILL with no output.
    const SETTLE_DEADLINE: Duration = Duration::from_secs(5);

    /// Long enough that a "completed only because the wait elapsed" pass is not
    /// possible, short enough that a test asserting *non*-completion stays cheap.
    const STAYS_PENDING_WINDOW: Duration = Duration::from_millis(200);

    /// Iterations of the drop-order probe
    /// ([`a_waiter_woken_by_a_drop_finds_the_freed_slot_on_its_first_scan`]).
    ///
    /// Sized from a measured detection threshold rather than picked: against a
    /// deliberately inverted `CheckedOutClient` field order, 1 and 64 iterations
    /// both passed (i.e. detected nothing) while 1024 failed on 6 of 6 runs.
    /// Four times that leaves margin for a slower or busier machine and still
    /// costs well under a second. See that test's doc comment for why a single
    /// iteration cannot work.
    const DROP_ORDER_PROBES: usize = 4096;

    /// An unconnected `ExclusiveCheckout` over `slots` slots, with **no** metrics
    /// handle installed — the state startup PINGs and most of this module's tests
    /// operate in, where every `observe_*` call is a no-op.
    ///
    /// Tests that assert on a counter must use [`checkout_with_metrics`] instead:
    /// with a `None` handle, deleting an `observe_*` call outright leaves
    /// the suite green, so a test built on this helper can never pin one.
    fn checkout(slots: usize, label: &'static str) -> ExclusiveCheckout {
        build_checkout(slots, label, Arc::new(std::sync::Mutex::new(None)))
    }

    /// An unconnected `ExclusiveCheckout` wired to a real `Metrics`, so the
    /// counters it emits are observable.
    ///
    /// The handle goes into the same `Arc<Mutex<Option<SharedMetrics>>>` slot
    /// `ValkeyClient::install_observability_hooks` fills in production, which is
    /// what makes `observe_checkout` / the waiter gauge
    /// non-no-op. Each call builds its OWN `Metrics`, so counters are per-test and
    /// cannot be contaminated by a sibling running concurrently.
    fn checkout_with_metrics(
        slots: usize,
        label: &'static str,
    ) -> (ExclusiveCheckout, SharedMetrics) {
        let metrics: SharedMetrics =
            Arc::new(crate::observability::metrics::metrics_scrape_only("test"));
        let slot = Arc::new(std::sync::Mutex::new(Some(Arc::clone(&metrics))));
        (build_checkout(slots, label, slot), metrics)
    }

    fn build_checkout(
        slots: usize,
        label: &'static str,
        metrics: Arc<std::sync::Mutex<Option<SharedMetrics>>>,
    ) -> ExclusiveCheckout {
        let pool = Builder::from_config(Default::default())
            .build_exclusive_pool(slots)
            .expect("building an exclusive pool must succeed without connecting");
        ExclusiveCheckout::new(pool, label, metrics)
    }

    /// `live_edit_valkey_pool_waiters{pool}` off the Prometheus scrape — the
    /// EXPORTED gauge, as distinct from the `waiters()` atomic the cap reads.
    ///
    /// The two are separate observables with separate failure modes: the atomic
    /// is what `WaiterSlot::join` enforces the cap against, the gauge is what an
    /// operator sees. `inc_valkey_pool_waiters` / `dec_valkey_pool_waiters` could
    /// both be emptied to no-ops and every `waiters()` assertion in this module
    /// would still pass, so the gauge needs its own read.
    fn read_pool_waiters(metrics: &SharedMetrics, pool: &str) -> i64 {
        use crate::observability::metrics::test_util::{render, sample_value};
        sample_value(
            &render(metrics),
            "live_edit_valkey_pool_waiters",
            &[("pool", pool)],
        )
        .expect("the waiter gauge's pool labels are pre-initialised in `Metrics::new`")
            as i64
    }

    /// The `_count` of a histogram family off the scrape — how many observations
    /// it has recorded.
    ///
    /// `None` when the family is absent entirely, which for an OTel histogram
    /// means no measurement has ever been recorded. That is a real and
    /// distinguishable state here (it is what "this code path never ran" looks
    /// like), so it is returned rather than defaulted to zero.
    fn read_histogram_count(
        metrics: &SharedMetrics,
        name: &str,
        labels: &[(&str, &str)],
    ) -> Option<u64> {
        use crate::observability::metrics::test_util::{render, sample_value};
        sample_value(&render(metrics), &format!("{name}_count"), labels).map(|v| v as u64)
    }

    /// fred assigns each `Client` a distinct id at construction, so this names
    /// the underlying slot a checkout is holding.
    fn slot_id(checked_out: &CheckedOutClient) -> String {
        ClientLike::id(&**checked_out).to_string()
    }

    /// Poll `fut` once and report whether it was ready.
    async fn poll_once<F: Future>(fut: std::pin::Pin<&mut F>) -> Poll<F::Output> {
        let mut fut = Some(fut);
        std::future::poll_fn(move |cx| {
            let f = fut.take().expect("polled once");
            Poll::Ready(f.poll(cx))
        })
        .await
    }

    // ── Next-free scheduling: the ticket's embedded probe, driven through ours ──

    /// The b.f4f defect, reproduced against **our** checkout and required to
    /// come out the other way.
    ///
    /// This is the ticket's embedded two-slot probe, verbatim in shape: hold
    /// both slots, park a third acquire, release the slot fred's `acquire`
    /// would NOT have selected, and require the waiter to complete.
    ///
    /// Against fred 10.1.0's `ExclusivePool::acquire` the waiter stays parked
    /// here — it scanned once, found nothing, and is awaiting one specific
    /// mutex (`fred-10.1.0/src/clients/pool.rs:633-646`). `ExclusiveCheckout`
    /// parks on a FIFO semaphore instead and only ever scans non-blockingly, so
    /// ANY release wakes it.
    ///
    /// The "which slot gets released" detail is what makes the test
    /// discriminating: releasing the *first* checkout would complete the waiter
    /// under both implementations, which is exactly the vacuous version of this
    /// test.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn acquire_completes_a_waiter_when_any_occupied_slot_frees() {
        let checkout = checkout(2, POOL_LABEL_TXN);

        let first = checkout.acquire().await;
        let second = checkout.acquire().await;
        assert_ne!(
            slot_id(&first),
            slot_id(&second),
            "two concurrent holders must hold two DIFFERENT slots — a checkout that handed the \
             same client to both would be exclusive in name only"
        );

        let mut waiter = std::pin::pin!(checkout.acquire());
        assert!(
            poll_once(waiter.as_mut()).await.is_pending(),
            "a third checkout on a two-slot pool must park: both slots are held"
        );
        assert_eq!(
            checkout.available_permits(),
            0,
            "both permits are out, which is what makes the park a permit wait rather than a \
             mutex wait"
        );

        // Release the slot a `lock_owned()`-on-one-selected-mutex waiter would
        // NOT be waiting on. Which of the two that is depends on fred's scan
        // order, so release the SECOND: the ticket's probe shows the waiter
        // selecting the first, and staying blocked when the second frees.
        drop(second);

        let granted = tokio::time::timeout(SETTLE_DEADLINE, &mut waiter)
            .await
            .expect(
                "the parked checkout did not complete after a slot was freed. That is the fred \
                 10.1.0 behaviour b.f4f exists to replace: `ExclusivePool::acquire` scans once \
                 and then awaits ONE selected mutex, so a waiter stays blocked while other slots \
                 go idle. `ExclusiveCheckout` must take a FIFO semaphore permit and re-scan every \
                 slot non-blockingly.",
            );

        assert_ne!(
            slot_id(&granted),
            slot_id(&first),
            "the waiter was handed the slot the FIRST holder is still holding — two holders on \
             one client at the same instant"
        );
    }

    /// The upstream behaviour the test above is the counterpart to, reproduced
    /// against this repository's own `Cargo.lock`.
    ///
    /// This is the ticket's embedded probe run **unchanged** — on fred's
    /// `ExclusivePool::acquire` rather than on ours — and it is what makes the
    /// sibling test above a discriminating regression rather than a description
    /// of current behaviour: without this, "the waiter completed" could be true
    /// of fred's scheduling too and nobody would know. The ticket ran it as a
    /// hand-linked scratch binary against `target/debug/deps`; here it is an
    /// ordinary unit test, so it cannot drift from the version actually built.
    ///
    /// **When this test goes red, it is a finding, not a bug.** The only way it
    /// can fail is that fred fixed its scheduling — at which point
    /// `ExclusiveCheckout`'s next-free rationale (and this pair of tests) should
    /// be revisited deliberately, in the same commit as the dependency bump,
    /// rather than left as a comment describing a version nobody runs.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fred_exclusive_pool_acquire_still_parks_on_one_selected_slot() {
        let pool = Builder::from_config(Default::default())
            .build_exclusive_pool(2)
            .expect("building an exclusive pool must succeed without connecting");

        let first = pool.acquire().await;
        let second = pool.acquire().await;

        let mut waiter = std::pin::pin!(pool.acquire());
        assert!(
            poll_once(waiter.as_mut()).await.is_pending(),
            "fixture self-check: with both slots held the third acquire must park"
        );

        // Free the slot fred's waiter did NOT select.
        drop(second);
        assert!(
            tokio::time::timeout(STAYS_PENDING_WINDOW, &mut waiter)
                .await
                .is_err(),
            "fred's `ExclusivePool::acquire` completed after an arbitrary slot was freed. That \
             is a FIX upstream, not a failure here: `ExclusiveCheckout` exists because fred \
             10.1.0 scans once and then awaits ONE selected mutex \
             (`fred-10.1.0/src/clients/pool.rs:633-646`). Re-read that call site against the \
             version now in Cargo.lock and decide, deliberately, whether the service still needs \
             its own checkout — then update both this test and the module docs."
        );

        // …and completes only when its own selected slot frees.
        drop(first);
        assert!(
            tokio::time::timeout(SETTLE_DEADLINE, &mut waiter)
                .await
                .is_ok(),
            "fixture self-check: releasing the originally selected slot must complete the waiter"
        );
    }

    /// Every slot is handed out before anyone parks, and the (N+1)th parks.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn acquire_hands_out_every_slot_before_any_caller_parks() {
        const SLOTS: usize = 4;
        let checkout = checkout(SLOTS, POOL_LABEL_TXN);

        let mut held = Vec::with_capacity(SLOTS);
        for _ in 0..SLOTS {
            held.push(
                tokio::time::timeout(SETTLE_DEADLINE, checkout.acquire())
                    .await
                    .expect("a free slot must be handed out without waiting"),
            );
        }

        let mut ids: Vec<String> = held.iter().map(slot_id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(
            ids.len(),
            SLOTS,
            "{SLOTS} concurrent holders must hold {SLOTS} distinct slots; got {ids:?}"
        );

        let mut overflow = std::pin::pin!(checkout.acquire());
        assert!(
            poll_once(overflow.as_mut()).await.is_pending(),
            "the ({}th) checkout must park once every slot is held",
            SLOTS + 1
        );
    }

    /// N waiters, M releases ⇒ exactly M completions, never M + 1.
    ///
    /// The failure this rules out is over-admission: a permit released ahead of
    /// its slot, or a waiter woken by a release another waiter already claimed,
    /// would hand out more checkouts than slots were freed — and the surplus
    /// holder would be sharing a client with someone.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn multiple_waiters_complete_exactly_once_per_release() {
        const SLOTS: usize = 2;
        const WAITERS: usize = 4;

        let checkout = Arc::new(checkout(SLOTS, POOL_LABEL_TXN));
        let mut held = Vec::with_capacity(SLOTS);
        for _ in 0..SLOTS {
            held.push(checkout.acquire().await);
        }

        let granted = Arc::new(tokio::sync::Mutex::new(Vec::<String>::new()));
        let mut tasks = Vec::with_capacity(WAITERS);
        for _ in 0..WAITERS {
            let checkout = Arc::clone(&checkout);
            let granted = Arc::clone(&granted);
            tasks.push(tokio::spawn(async move {
                let slot = checkout.acquire().await;
                granted.lock().await.push(slot_id(&slot));
                // Hold it: a completed waiter must not recycle its slot and let
                // a second waiter through on one release.
                std::future::pending::<()>().await;
                drop(slot);
            }));
        }

        // Barrier on observable state rather than a sleep: every waiter is
        // parked once the waiter gauge says so.
        let deadline = Instant::now() + SETTLE_DEADLINE;
        while checkout.waiters() < WAITERS {
            assert!(
                Instant::now() < deadline,
                "only {} of {WAITERS} waiters registered on the permit queue",
                checkout.waiters()
            );
            tokio::task::yield_now().await;
        }

        // One release ⇒ exactly one completion.
        held.pop();
        let deadline = Instant::now() + SETTLE_DEADLINE;
        while granted.lock().await.is_empty() {
            assert!(
                Instant::now() < deadline,
                "no waiter completed within {SETTLE_DEADLINE:?} after one slot was released"
            );
            tokio::task::yield_now().await;
        }
        tokio::time::sleep(STAYS_PENDING_WINDOW).await;
        assert_eq!(
            granted.lock().await.len(),
            1,
            "one release admitted more than one waiter — permits and slots have diverged, so \
             two holders are on one client"
        );

        // A second release ⇒ exactly one more.
        held.pop();
        let deadline = Instant::now() + SETTLE_DEADLINE;
        while granted.lock().await.len() < 2 {
            assert!(
                Instant::now() < deadline,
                "the second waiter did not complete after the second release"
            );
            tokio::task::yield_now().await;
        }
        tokio::time::sleep(STAYS_PENDING_WINDOW).await;
        let ids = granted.lock().await.clone();
        assert_eq!(
            ids.len(),
            2,
            "two releases admitted {} waiters; permits == slots must cap it at two",
            ids.len()
        );
        assert_ne!(
            ids[0], ids[1],
            "the two admitted waiters were handed the SAME slot while both still hold it: {ids:?}"
        );

        for task in tasks {
            task.abort();
        }
    }

    /// The permit queue is **FIFO**: waiters are served in arrival order.
    ///
    /// Fairness is the entire reason the checkout parks on a `Semaphore` at all
    /// — the scan that follows is non-blocking and has no ordering of its own, so
    /// if the queue were LIFO or arbitrary, a steady arrival rate would starve
    /// the earliest waiter indefinitely. That is the same starvation shape as the
    /// fred defect b.f4f replaced, reached by a different route, and the sibling
    /// `multiple_waiters_complete_exactly_once_per_release` cannot see it: it
    /// asserts how MANY waiters completed and that they got distinct slots, both
    /// of which hold under any ordering.
    ///
    /// **Arrival order is program order.** Nothing is spawned: the test task
    /// first-polls three pinned `acquire()` futures itself, in index order. That
    /// fixes the queue order only because a `Pending` first poll has already
    /// joined the permit queue: `acquire` → `acquire_permit` has no await before
    /// the permit semaphore's first poll, and that poll enqueues the waiter before
    /// it returns `Pending` (`tokio-1.53.1/src/sync/batch_semaphore.rs:516`,
    /// `push_front`). An await added before that poll in `acquire_permit` breaks
    /// the premise, and this test then fails rather than passing: the release
    /// passes below poll later waiters first, so the first of them to reach the
    /// semaphore takes the released permit.
    ///
    /// `waiters()` cannot fix the order. `WaiterSlot::join_unbounded` bumps it
    /// before the waiter joins the queue, so a grown counter only means "about to
    /// enqueue"; it is asserted after each first poll as a self-check that the
    /// waiter parked, not as the barrier.
    ///
    /// First polls run under `tokio::task::coop::unconstrained`, so tokio's
    /// cooperative budget cannot return `Pending` before the waiter joins the
    /// queue. Today the budget is never short there — the setup acquisitions take
    /// the uncontended `try_acquire_owned` path, and a `Pending` poll refunds its
    /// unit (`tokio-1.53.1/src/task/coop/mod.rs:278-288`) — so this makes the
    /// property structural rather than fixing anything.
    ///
    /// **Service order is fixed inside the release.** The semaphore hands a
    /// released permit to the oldest waiter within the drop that releases it
    /// (`batch_semaphore.rs:314`, `queue.last()`; `:327`, `pop_back`), before
    /// anything is polled. Each release is followed by passes over the
    /// still-pending waiters newest first, so polling order can only work against
    /// the expected answer. Each served checkout is held to the end of the test,
    /// so one release serves exactly one waiter.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_permit_queue_is_fifo_so_waiters_are_served_in_arrival_order() {
        const SLOTS: usize = 3;

        let checkout = checkout(SLOTS, POOL_LABEL_TXN);
        let mut held = Vec::with_capacity(SLOTS);
        for _ in 0..SLOTS {
            held.push(checkout.acquire().await);
        }

        // `None` once served: a completed future is never polled again.
        let mut waiting: Vec<Option<_>> = (0..SLOTS)
            .map(|_| Some(Box::pin(checkout.acquire())))
            .collect();
        for (index, waiter) in waiting.iter_mut().enumerate() {
            let waiter = waiter
                .as_mut()
                .expect("no waiter is served before a release");
            assert!(
                tokio::task::coop::unconstrained(poll_once(waiter.as_mut()))
                    .await
                    .is_pending(),
                "fixture self-check: with all {SLOTS} slots held, waiter {index} must park"
            );
            assert_eq!(
                checkout.waiters(),
                index + 1,
                "fixture self-check: waiter {index} parked without being counted as waiting"
            );
        }

        let mut order = Vec::with_capacity(SLOTS);
        // Held to the end: a served waiter that released its slot would let one
        // release serve two, and the order below would stop meaning anything.
        let mut served = Vec::with_capacity(SLOTS);
        for release in 0..SLOTS {
            drop(held.pop().expect("one held checkout per release"));
            let deadline = Instant::now() + SETTLE_DEADLINE;
            let index = loop {
                let mut ready = None;
                for index in (0..SLOTS).rev() {
                    let Some(waiter) = waiting[index].as_mut() else {
                        continue;
                    };
                    let Poll::Ready(granted) =
                        tokio::task::coop::unconstrained(poll_once(waiter.as_mut())).await
                    else {
                        continue;
                    };
                    if let Some(earlier) = ready {
                        panic!(
                            "release {release} served waiters {earlier} and {index} in one pass — \
                             permits and slots have diverged, so one release admitted two"
                        );
                    }
                    waiting[index] = None;
                    served.push(granted);
                    ready = Some(index);
                }
                if let Some(index) = ready {
                    break index;
                }
                assert!(
                    Instant::now() < deadline,
                    "release {release} did not serve a waiter within {SETTLE_DEADLINE:?}"
                );
                tokio::task::yield_now().await;
            };
            order.push(index);
        }

        assert_eq!(
            order,
            vec![0, 1, 2],
            "waiters were served out of arrival order. tokio's `Semaphore` is FIFO-fair and that \
             fairness IS the checkout's queueing policy — the `try_lock_owned` scan that follows \
             imposes no order of its own. Serving out of order lets a steady arrival rate starve \
             the earliest waiter, which is the fred 10.1.0 starvation shape reached by a \
             different route. A failure can also mean `acquire` gained an await before its first \
             semaphore poll, so a parked first poll no longer means a queued waiter (see this \
             test's doc)"
        );
        assert_eq!(
            checkout.waiters(),
            0,
            "every waiter was served, so none may still be counted as waiting"
        );
    }

    /// A waiter that gives up returns its permit and its queue position.
    ///
    /// A leak here is invisible and permanent: the pool silently shrinks by one
    /// for the life of the pod. The wait can be dropped by a cancelled caller,
    /// so `WaiterSlot`'s `Drop` is the only thing that can uncount it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cancelled_waiter_returns_its_permit_and_its_queue_position() {
        let checkout = checkout(1, POOL_LABEL_TXN);
        let held = checkout.acquire().await;

        {
            let mut waiter = std::pin::pin!(checkout.acquire());
            assert!(poll_once(waiter.as_mut()).await.is_pending());
            let deadline = Instant::now() + SETTLE_DEADLINE;
            while checkout.waiters() < 1 {
                assert!(
                    Instant::now() < deadline,
                    "the waiter never joined the queue"
                );
                tokio::task::yield_now().await;
            }
        } // <- the acquire future is dropped here, exactly as hyper drops a
          //    cancelled handler's future.

        assert_eq!(
            checkout.waiters(),
            0,
            "a dropped acquire future left its waiter counted; the waiter gauge ratchets"
        );

        drop(held);
        assert_eq!(
            checkout.available_permits(),
            1,
            "the permit count did not return to the slot count after a cancelled waiter and a \
             released holder — a leaked permit shrinks the pool for the life of the process"
        );
        // …and the pool is genuinely usable again, not merely arithmetically tidy.
        let reacquired = tokio::time::timeout(SETTLE_DEADLINE, checkout.acquire())
            .await
            .expect("the pool must still hand out its slot after a waiter was cancelled");
        drop(reacquired);
    }

    /// An **aborted** holder has given its slot and its permit back by the time
    /// its `JoinHandle` resolves.
    ///
    /// [`ValkeyClient::recycle_txn_slot`](crate::storage::client::ValkeyClient)'s
    /// `JoinError` arm takes a **fresh** checkout after the repair task died.
    /// That is only sound if the dead task already returned its permit —
    /// otherwise one caller would consume two permits against one slot and the
    /// pool would shrink on every panicked repair.
    ///
    /// The property is tokio's — an aborted task's `JoinHandle` resolves only
    /// after its future has been dropped — but it is load-bearing enough here
    /// that it is worth pinning against the actual checkout rather than trusting
    /// it from memory.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_aborted_holder_has_returned_its_slot_and_permit_once_its_handle_resolves() {
        let checkout = Arc::new(checkout(1, POOL_LABEL_TXN));

        let held = Arc::clone(&checkout);
        let task = tokio::spawn(async move {
            let _slot = held.acquire().await;
            std::future::pending::<()>().await;
        });

        let deadline = Instant::now() + SETTLE_DEADLINE;
        while checkout.available_permits() != 0 {
            assert!(
                Instant::now() < deadline,
                "the spawned holder never took the pool's only slot"
            );
            tokio::task::yield_now().await;
        }

        task.abort();
        let _ = task.await;

        assert_eq!(
            checkout.available_permits(),
            1,
            "an aborted holder's permit was not back by the time its JoinHandle resolved. \
             `recycle_txn_slot`'s JoinError arm takes a fresh checkout at exactly this point, so \
             a late release means one subscription holding two permits against one slot"
        );
        // …and the slot, not just the arithmetic: a checkout taken right here is
        // what the JoinError arm actually does.
        let fresh = tokio::time::timeout(SETTLE_DEADLINE, checkout.acquire())
            .await
            .expect(
                "the fresh checkout an aborted repair falls back to must not park: the dead \
                 task's guard drops with its future, before its handle resolves",
            );
        drop(fresh);
    }

    // ── Drop order, made observable by suppressing the re-scan ────────────────
    //
    // Round 1 of b.f4f reported this as a gap: `CheckedOutClient` releases its
    // guard before its entitlement, but the `yield_now()` re-scan rescues a woken
    // waiter whose first scan came up empty, so an INVERTED order produced no
    // observable difference and `multiple_waiters_complete_exactly_once_per_release`
    // — the closest indirect test — would still have passed. It catches a gross
    // inversion (a permit freed while its slot stays held for the rest of the
    // test) and nothing finer. `disable_rescan_for_test` removes the rescue, so
    // the first scan becomes the verdict.

    /// A waiter woken by a `CheckedOutClient` drop finds the freed slot on its
    /// **first** scan.
    ///
    /// This is the direct test of the drop-order contract. With the re-scan
    /// suppressed, a permit released even an instant before its guard leaves the
    /// woken waiter scanning a still-locked pool, and `acquire` — being
    /// infallible — `panic!`s. `multi_thread` and the awaited `JoinHandle` are
    /// both part of the seam's contract, not stylistic choices: on a
    /// current-thread runtime the guard→token→entitlement drop cannot be interleaved
    /// at all, so an inverted order would be repaired before the waiter ever ran
    /// and this test would pass whatever the field order said; and a detached
    /// task's panic is swallowed by tokio and surfaced only through its handle.
    ///
    /// # Why it repeats, and why [`DROP_ORDER_PROBES`] is the number it is
    ///
    /// The window an inverted order opens is a few instructions wide — between
    /// the semaphore handing the permit to the woken waiter and the guard's
    /// atomic unlock — while waking that waiter onto another worker costs
    /// microseconds. A single iteration therefore loses the race essentially
    /// always: **measured**, one iteration against a deliberately inverted field
    /// order passed, as did 64 iterations (5/5 runs each). 1024 iterations caught
    /// it on 6 of 6 runs; [`DROP_ORDER_PROBES`] is four times that, and the
    /// correct order passed 6 of 6 runs at that count in ~0.7 s.
    ///
    /// So this is a repeat-until-observed probe, not a timing assertion: it
    /// asserts nothing about how fast anything is, every iteration is an
    /// independent all-or-nothing check, and the count is sized from a measured
    /// detection threshold rather than guessed. A *failing* run is deterministic
    /// enough to act on; a passing one is what the contract predicts.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_waiter_woken_by_a_drop_finds_the_freed_slot_on_its_first_scan() {
        for probe in 0..DROP_ORDER_PROBES {
            let checkout = Arc::new(checkout(1, POOL_LABEL_TXN));
            checkout.disable_rescan_for_test();

            // The pool's only slot and only permit.
            let held = checkout.acquire().await;

            let waiting = Arc::clone(&checkout);
            let waiter = tokio::spawn(async move {
                let slot = waiting.acquire().await;
                slot_id(&slot)
            });

            // Barrier on observable state: the waiter is parked on the permit
            // queue, so the drop below is what wakes it rather than it simply not
            // having started yet.
            let deadline = Instant::now() + SETTLE_DEADLINE;
            while checkout.waiters() == 0 {
                assert!(
                    Instant::now() < deadline,
                    "probe {probe}: the waiter never joined the permit queue, so the drop below \
                     would not be what wakes it and the first-scan assertion would prove nothing"
                );
                tokio::task::yield_now().await;
            }

            let held_id = slot_id(&held);
            drop(held);

            let granted = tokio::time::timeout(SETTLE_DEADLINE, waiter)
                .await
                .unwrap_or_else(|_| panic!("probe {probe}: the woken waiter must complete"))
                .unwrap_or_else(|join_err| {
                    panic!(
                        "probe {probe}: the woken waiter PANICKED. With the re-scan suppressed \
                         its first scan found no free slot while it held a permit; permits == \
                         slots, so the permit it was handed was released BEFORE the guard whose \
                         slot it accounts for. `CheckedOutClient`'s field order (guard, token, \
                         entitlement) is what makes the opposite true, because Rust drops struct \
                         fields in declaration order — check that order first. {join_err}"
                    )
                });
            assert_eq!(
                granted, held_id,
                "probe {probe}: on a one-slot pool the waiter can only have been handed the slot \
                 just released"
            );
        }
    }

    /// The seam is armed, and an empty first scan while holding a permit really
    /// is its verdict.
    ///
    /// The non-vacuity guard for the drop-order test above: were
    /// `disable_rescan_for_test` a no-op, or the branch that reads it gone, the
    /// production re-scan would rescue every empty scan and that test could not
    /// fail. Here the scan is empty by construction — a checkout is split with
    /// `into_parts` and its permit dropped while its slot stays locked, which
    /// is exactly the state an inverted `CheckedOutClient` drop order leaves —
    /// so the next `acquire` is handed that permit, finds nothing, and must
    /// fail with the seam's own panic.
    ///
    /// The panic message is the discriminator, not timing: without the seam the
    /// same waiter re-scans for `CHECKOUT_RESCAN_DEADLINE` and then trips
    /// `report_escaped_slot`'s `debug_assert!`, whose message does not name the
    /// seam.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_suppressed_rescan_makes_an_empty_first_scan_terminal() {
        let checkout = Arc::new(checkout(1, POOL_LABEL_TXN));
        checkout.disable_rescan_for_test();

        let (guard, _token, entitlement) = checkout.acquire().await.into_parts();
        drop(entitlement);
        assert_eq!(
            checkout.available_permits(),
            1,
            "fixture self-check: the permit is free while its slot is still locked"
        );

        let waiting = Arc::clone(&checkout);
        let waiter = tokio::spawn(async move {
            let _slot = waiting.acquire().await;
        });
        let join_err = tokio::time::timeout(SETTLE_DEADLINE, waiter)
            .await
            .expect("an empty first scan under the seam must end the waiter")
            .expect_err("the waiter must not be handed the slot that is still locked");
        let panic = join_err.into_panic();
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or_default();
        assert!(
            message.contains("disable_rescan_for_test"),
            "the waiter must fail with the seam's single-pass verdict; it failed with: {message:?}"
        );
        drop(guard);
    }

    /// `acquire`'s `is_quitting()` arm under the seam: once `quit` has closed
    /// the semaphore, an empty scan is shutdown holding the slots, not an
    /// escaped slot, and the seam must not accuse it.
    ///
    /// The probe's first poll runs its whole first scan — permit-less, since the
    /// semaphore is closed, over a pool whose only slot is held — so a seam
    /// that ignored `is_quitting()` would panic right there. Instead it must
    /// stay pending and take the slot once it is released.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_quitting_pool_is_never_accused_of_an_escaped_slot_under_the_seam() {
        let checkout = checkout(1, POOL_LABEL_TXN);
        checkout.disable_rescan_for_test();
        let held = checkout.acquire().await;

        {
            let mut quitting = std::pin::pin!(checkout.quit());
            assert!(
                poll_once(quitting.as_mut()).await.is_pending(),
                "fixture self-check: fred's pool QUIT waits for the held slot"
            );
        }
        assert!(checkout.is_quitting());

        let mut probe = std::pin::pin!(checkout.acquire());
        assert!(
            poll_once(probe.as_mut()).await.is_pending(),
            "a checkout during shutdown waits for a slot to free rather than being handed one"
        );
        drop(held);
        let granted = tokio::time::timeout(SETTLE_DEADLINE, probe)
            .await
            .expect("the released slot must reach the permit-less checkout");
        drop(granted);
    }

    /// `quit` closes the permit semaphore before its first await, so the window
    /// in which fred's pool `QUIT` holds every slot is already marked as
    /// shutdown when a checkout can observe it.
    ///
    /// A held slot keeps `ExclusivePool::quit` pending, so the first poll stops
    /// at that await; `is_quitting()` must already be true there. With the
    /// close moved after the QUIT, a checkout landing in that window would hold
    /// a permit, scan a full pool and be reported as an escaped slot.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn quit_closes_the_permit_semaphore_before_it_takes_the_slots() {
        let checkout = checkout(1, POOL_LABEL_TXN);
        assert!(!checkout.is_quitting(), "a fresh pool is not quitting");
        let held = checkout.acquire().await;

        let mut quitting = std::pin::pin!(checkout.quit());
        assert!(
            poll_once(quitting.as_mut()).await.is_pending(),
            "fixture self-check: fred's pool QUIT waits for the held slot, so the poll stopped \
             at `quit`'s await"
        );
        assert!(
            checkout.is_quitting(),
            "`quit` reached its first await without closing the permit semaphore"
        );
        drop(held);
    }

    // ── The exported instruments ──────────────────────────────────────────────
    //
    // Everything above asserts on return values and on the `waiters()` atomic.
    // These three assert on what actually leaves the process, because the
    // emitters are separately deletable: `inc_valkey_pool_waiters` and both
    // `record` calls could be emptied without moving a single assertion in the
    // rest of the module.

    /// `live_edit_valkey_pool_checkout_seconds` records one observation per
    /// successful checkout.
    ///
    /// The histogram is the latency signal for the whole b.f4f fix — it is how an
    /// operator sees whether the next-free scheduling is actually keeping
    /// checkout fast under saturation — and nothing else in the suite reads it,
    /// so `observe_checkout` could be deleted silently.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_checkout_latency_histogram_records_one_observation_per_checkout() {
        const NAME: &str = "live_edit_valkey_pool_checkout_seconds";
        let (checkout, metrics) = checkout_with_metrics(2, POOL_LABEL_TXN);
        assert_eq!(
            read_histogram_count(&metrics, NAME, &[("pool", POOL_LABEL_TXN)]),
            None,
            "fixture self-check: an OTel histogram with no recorded measurement emits no \
             family, so this must be absent before the first checkout"
        );

        let first = checkout.acquire().await;
        assert_eq!(
            read_histogram_count(&metrics, NAME, &[("pool", POOL_LABEL_TXN)]),
            Some(1),
            "a successful checkout must record its duration. Without it the pool's latency \
             under contention is invisible, which is the one thing this pool was rebuilt to fix"
        );

        let second = checkout.acquire().await;
        assert_eq!(
            read_histogram_count(&metrics, NAME, &[("pool", POOL_LABEL_TXN)]),
            Some(2),
            "one observation per checkout, not per pool or per process"
        );

        drop(first);
        drop(second);
        assert_eq!(
            read_histogram_count(&metrics, NAME, &[("pool", POOL_LABEL_TXN)]),
            Some(2),
            "releasing a checkout is not itself a checkout and must not record"
        );
    }

    /// The exported waiter gauge rises while a caller is parked on the permit
    /// queue and falls when it leaves, including when its future is dropped.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_exported_waiter_gauge_tracks_parked_acquires() {
        let (checkout, metrics) = checkout_with_metrics(1, POOL_LABEL_TXN);
        assert_eq!(read_pool_waiters(&metrics, POOL_LABEL_TXN), 0);
        let held = checkout.acquire().await;
        assert_eq!(
            read_pool_waiters(&metrics, POOL_LABEL_TXN),
            0,
            "an uncontended acquire must not touch the waiter gauge"
        );
        {
            let mut parked = std::pin::pin!(checkout.acquire());
            assert!(poll_once(parked.as_mut()).await.is_pending());
            assert_eq!(read_pool_waiters(&metrics, POOL_LABEL_TXN), 1);
        }
        assert_eq!(
            read_pool_waiters(&metrics, POOL_LABEL_TXN),
            0,
            "a dropped waiter must leave the gauge"
        );
        drop(held);
    }

    // ── Constant and policy pins ──────────────────────────────────────────────

    /// The re-scan budget is **defence in depth**, not a mechanism any path
    /// relies on: an aborted holder releases its slot before its permit. Pinned
    /// so a future edit that leans on the budget as a correctness mechanism has
    /// to change this line and say why.
    #[test]
    fn checkout_rescan_deadline_is_pinned_to_the_probe_timeout_scale() {
        assert_eq!(
            CHECKOUT_RESCAN_DEADLINE,
            Duration::from_secs(3),
            "Changing CHECKOUT_RESCAN_DEADLINE changes how long a caller hangs before an \
             impossible state is reported, so change it deliberately."
        );
    }
}
