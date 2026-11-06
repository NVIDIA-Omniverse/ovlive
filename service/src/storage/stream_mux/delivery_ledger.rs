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

//! The live-delivery budget `D_live` (SDD §12.6): one [`DeliveryLedger`] per
//! replica, over the relay bytes the replica holds for its subscriptions'
//! undelivered live updates.
//!
//! **What is charged.** Charges are forced: nothing waits for them, and the
//! stream readers never pause.
//!
//! - Each relay payload a stream reader copies out of its reply is a
//!   [`DeliveryLedger::relay_buf`], charged its length plus
//!   [`RELAY_BUF_OVERHEAD_BYTES`] from the copy until the last subscription
//!   holding it drops it. A copy is **deduplicated**: a reader that copies an
//!   entry another live copy already holds — the same stream key and entry id
//!   with the same bytes, as a late joiner's re-read does — shares that copy
//!   instead. An entry of a recreated stream with the same id and other bytes
//!   is never shared.
//! - Each entry queued for a subscription is charged
//!   [`RELAY_ENTRY_OVERHEAD_BYTES`] for that subscription, from the offer
//!   until the subscription receives it from its stream backlog.
//!
//! So `charged` is exact at every moment: every live copy once, plus every
//! queued entry's overhead.
//!
//! **What is excused.** For each stream key, the largest live copy (ties: the
//! lowest sequence) is `E_k`; `exempt` is their sum. A caught-up subscriber
//! receiving one large update, while small updates continue on its layer, is
//! never what puts the replica over its budget.
//!
//! **Holds.** Each `SubscribeReadLayer` RPC registers one [`SubscriptionHold`]
//! for its whole life, cloned into each of its stream registrations. A hold
//! tracks its **pipeline**: every entry offered to it and not yet released —
//! in its stream backlog, in the subscription task's hand, or in its response
//! queue until the stream takes the entry's last message. Each entry rides in
//! a [`PipelineEntry`] guard from offer to release, so a dropped consumer, a
//! cleared response queue, a cancelled task or a panic still settles it.
//! Entries leave a pipeline out of order: one dropped unsent when the
//! subscription receives it (a self-echo, a duplicate) leaves while older
//! ones still wait in the response queue. So a hold counts, per relay copy,
//! the entries in its pipeline carrying it, and whether its pipeline holds
//! `E_k` is that count, exact in any order. A copy is named by its serial,
//! drawn from one ledger-wide counter when the copy is made and never reused;
//! the reader hands it to every offer of the entry with the copy itself
//! ([`RelayCopy`]), so an offer looks nothing up, and the entry keeps only
//! the serial. The count has at most one slot per entry in the pipeline, so
//! it is bounded by the pipeline's own bounds: the backlog's hard cap, the
//! response queue's data slots and the entry in the task's hand. Constant
//! state per hold cannot be exact: when a key's largest copy leaves, the new
//! `E_k` is an older copy that some holds may hold, and nothing of constant
//! size can tell which.
//!
//! **Enforcement** ([`DeliveryLedger::enforce`]) runs after each reply's
//! fan-out, and again when a hold unregisters after a pass that ended over
//! the budget or carrying a pending credit (a credit can exceed what its
//! going frees, since a shared relay copy stays charged). The replica is
//! over when `charged − exempt − pending_release > D_live`. While it is
//! over, the **victim** is the hold with the most
//! chargeable bytes — its pipeline's relay charges and queued overhead, less
//! `E_k` when its pipeline holds `E_k` — ties to the larger whole hold, then
//! the lower id. Its ending token fires with [`STREAM_EXIT_DELIVERY_BUDGET`],
//! and its chargeable bytes count as `pending_release` until it unregisters,
//! so no second subscription is ended for bytes already going away. A hold
//! whose pipeline holds at most two entries, at most one of them in its
//! stream backlog, is **lone** and never a victim: a subscriber sending one
//! update with the next queued is left alone, as its own backlog caps would
//! leave it. When no hold that is not lone has chargeable bytes — every holder
//! over the budget is lone, or only the excused `E_k` is left — the pass
//! counts an **overshoot** (`live_edit_subscriber_delivery_budget_overshoots_total`,
//! one WARN per minute at most).
//!
//! **The residual.** The bytes this budget cannot reclaim are each layer's
//! `E_k` and what lone pipelines hold, at most two entries each. Entries are
//! offered to every subscription of a key in order but not released in order,
//! so a lone pipeline's two are the layer's newest two only when no entry
//! between or after them was dropped unsent (the subscriber's own writes,
//! duplicates); otherwise the older is one it is still sending. Dedupe makes
//! each entry one copy per replica, however many hold it.
//!
//! **Stalled catch-up holders** ([`DeliveryLedger::end_stalled_catchup_holders`]):
//! while a catch-up waits for the catch-up payload budget, a hold whose
//! response queue holds catch-up pages and has had nothing taken for the
//! outer send deadline is ended `slow_reader_deadline`.
//!
//! **Locking.** One `std::sync::Mutex` guards the dedupe map, the per-key
//! index, the holds and `pending_release`; `charged` and a hold's counters are
//! atomics moved without it, and an offer never takes it. A hold's per-copy
//! count has a mutex of its own, taken under the ledger's lock by enforcement
//! and alone by an offer or a release (an offer takes only the hold's map
//! lock), never the other way round. No copy is ever dropped under either
//! lock, because a copy's drop takes the ledger's.

use std::cmp::Reverse;
use std::collections::{BTreeSet, HashMap};
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio_util::sync::CancellationToken;

use crate::observability::metrics::{
    SharedMetrics, STREAM_EXIT_DELIVERY_BUDGET, STREAM_EXIT_SLOW_READER_DEADLINE,
};

use super::core::StreamId;

/// The charge a relay copy carries beyond its bytes: its owner, its dedupe
/// and index nodes, and its allocation's rounding (256 B [assumed]).
pub const RELAY_BUF_OVERHEAD_BYTES: u64 = 256;

/// The charge one entry queued for one subscription carries, from the offer
/// until the subscription receives it: 320 B, at least `e`, the measured
/// per-subscription cost of one queued entry
/// (`config::memory_plan::QUEUED_ENTRY_BYTES_DECI`, 267.9 B, measured with
/// this ledger in place), so `D_live` never under-counts what its queued
/// entries hold (the assertion below).
pub const RELAY_ENTRY_OVERHEAD_BYTES: u64 = 320;

const _: () = assert!(
    RELAY_ENTRY_OVERHEAD_BYTES * 10 >= crate::config::memory_plan::QUEUED_ENTRY_BYTES_DECI,
    "an entry's charge must cover the measured per-subscription cost of a queued entry"
);

/// The shortest gap between two overshoot WARNs.
const OVERSHOOT_WARN_INTERVAL: Duration = Duration::from_secs(60);

/// A pipeline holding at most this many entries is lone, and never a victim.
const LONE_PIPELINE_ENTRIES: u64 = 2;

/// A lone pipeline holds at most this many of its entries in the stream
/// backlog.
const LONE_BACKLOG_ENTRIES: u64 = 1;

/// The `UNAVAILABLE` message a subscription ended by the live-delivery budget
/// ends with.
pub const DELIVERY_BUDGET_ENDING_MSG: &str = "this replica's live-delivery memory is full and \
     this subscription held the most undelivered updates; reconnect to catch up";

/// The replica's live-delivery ledger (module docs).
pub struct DeliveryLedger {
    /// `D_live`, in bytes.
    budget_bytes: AtomicU64,
    /// Every live relay copy's charge plus every queued entry's overhead.
    charged: AtomicU64,
    /// A mirror of [`LedgerState::exempt`], for the gauge.
    exempt: AtomicU64,
    state: StdMutex<LedgerState>,
    metrics: Arc<StdMutex<Option<SharedMetrics>>>,
    /// The origin of the holds' take clocks.
    epoch: Instant,
    /// The next relay copy's serial: from 1, ledger-wide, never reused.
    next_copy: AtomicU64,
}

impl std::fmt::Debug for DeliveryLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeliveryLedger")
            .field("budget_bytes", &self.budget_bytes())
            .field("charged", &self.charged_bytes())
            .field("exempt", &self.exempt_bytes())
            .finish()
    }
}

#[derive(Default)]
struct LedgerState {
    /// Σ over keys of `E_k`'s charge.
    exempt: u64,
    /// The chargeable bytes of ended holds not yet unregistered.
    pending_release: u64,
    /// Whether the last enforcement pass ended over the budget.
    last_pass_over: bool,
    keys: HashMap<Arc<str>, KeyState>,
    holds: HashMap<u64, Arc<HoldShared>>,
    next_hold: u64,
    last_overshoot_warn: Option<Instant>,
}

/// One stream key's live copies.
#[derive(Default)]
struct KeyState {
    /// The newest copy of each entry id, for dedupe.
    copies: HashMap<StreamId, Weak<RelayOwner>>,
    /// Every live copy, ordered so the last is `E_k`.
    live: BTreeSet<LiveCopy>,
}

impl KeyState {
    /// `E_k`, when the key has a live copy.
    fn largest(&self) -> Option<LiveCopy> {
        self.live.last().copied()
    }

    /// `E_k`'s charge, or 0 when the key has no live copy.
    fn largest_charge(&self) -> u64 {
        self.largest().map_or(0, |c| c.charge)
    }
}

/// One live copy in its key's index: ordered by charge, then by lower
/// sequence, then by serial, so the largest is the last and equal charges
/// prefer the lowest sequence. The serial names the copy: drawn from one
/// ledger-wide counter, it is never reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct LiveCopy {
    charge: u64,
    seq: Reverse<u64>,
    serial: NonZeroU64,
}

/// A relay copy's owner: its bytes and its place in the ledger, released when
/// the last [`Bytes`] of it drops.
struct RelayOwner {
    data: Box<[u8]>,
    key: Arc<str>,
    id: StreamId,
    serial: NonZeroU64,
    charge: u64,
    ledger: Arc<DeliveryLedger>,
}

impl Drop for RelayOwner {
    fn drop(&mut self) {
        let copy = LiveCopy {
            charge: self.charge,
            seq: Reverse(self.id.ms),
            serial: self.serial,
        };
        self.ledger.release_copy(&self.key, self.id, copy);
    }
}

/// One more [`Bytes`] view of a shared [`RelayOwner`].
struct SharedCopy(Arc<RelayOwner>);

impl AsRef<[u8]> for SharedCopy {
    fn as_ref(&self) -> &[u8] {
        &self.0.data
    }
}

/// A relay copy ([`DeliveryLedger::relay_buf`]): a whole view of its bytes,
/// and its serial, which names the copy on the ledger and is never reused.
/// The stream reader hands the serial to every offer of the entry beside the
/// notification that carries the bytes, so an offer counts the copy on its
/// hold without the ledger's lock (module docs, *Holds*).
#[derive(Debug, Clone)]
pub struct RelayCopy {
    bytes: Bytes,
    serial: NonZeroU64,
}

impl RelayCopy {
    /// The copy's bytes.
    pub fn bytes(&self) -> &Bytes {
        &self.bytes
    }

    /// The copy's serial.
    pub fn serial(&self) -> NonZeroU64 {
        self.serial
    }

    /// The copy's bytes and its serial.
    pub fn into_parts(self) -> (Bytes, NonZeroU64) {
        (self.bytes, self.serial)
    }
}

impl std::ops::Deref for RelayCopy {
    type Target = Bytes;

    fn deref(&self) -> &Bytes {
        &self.bytes
    }
}

impl DeliveryLedger {
    /// A ledger of `budget_bytes` (`D_live`, from the pod memory plan).
    pub fn new(budget_bytes: u64, metrics: Arc<StdMutex<Option<SharedMetrics>>>) -> Arc<Self> {
        Arc::new(Self {
            budget_bytes: AtomicU64::new(budget_bytes),
            charged: AtomicU64::new(0),
            exempt: AtomicU64::new(0),
            state: StdMutex::new(LedgerState::default()),
            metrics,
            epoch: Instant::now(),
            next_copy: AtomicU64::new(1),
        })
    }

    /// `D_live`, in bytes.
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes.load(Ordering::Acquire)
    }

    /// Every live relay copy's charge plus every queued entry's overhead.
    pub fn charged_bytes(&self) -> u64 {
        self.charged.load(Ordering::Acquire)
    }

    /// The sum of each key's largest live copy's charge.
    pub fn exempt_bytes(&self) -> u64 {
        self.exempt.load(Ordering::Acquire)
    }

    /// The chargeable bytes of ended subscriptions that have not unregistered
    /// yet.
    pub fn pending_release_bytes(&self) -> u64 {
        self.lock().pending_release
    }

    /// Holds registered now.
    pub fn holds(&self) -> usize {
        self.lock().holds.len()
    }

    /// Replaces `D_live` (test seam): a test sizes the budget to its traffic.
    #[cfg(any(test, feature = "test-support"))]
    pub fn set_budget_bytes_for_test(&self, budget_bytes: u64) {
        self.budget_bytes.store(budget_bytes, Ordering::Release);
    }

    fn lock(&self) -> MutexGuard<'_, LedgerState> {
        // Nothing panics while holding the lock and every update leaves the
        // state consistent, so a poisoned lock's state is still sound.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn metrics(&self) -> Option<SharedMetrics> {
        self.metrics.lock().ok().and_then(|g| g.clone())
    }

    /// The milliseconds since the ledger's epoch.
    fn now_ms(&self) -> u64 {
        u64::try_from(self.epoch.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    fn add_charged(&self, bytes: u64) {
        self.charged.fetch_add(bytes, Ordering::AcqRel);
    }

    fn sub_charged(&self, bytes: u64) {
        let _ = self
            .charged
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |c| {
                Some(c.saturating_sub(bytes))
            });
    }

    /// The relay copy of entry `id` on stream `key`, whose bytes are `data`:
    /// a live copy of the same key, id and bytes shared, or a new one, charged
    /// `data.len()` plus [`RELAY_BUF_OVERHEAD_BYTES`] until its last view
    /// drops, with the copy's serial. The bytes are compared and copied
    /// outside the ledger's lock.
    pub fn relay_buf(self: &Arc<Self>, key: &str, id: StreamId, data: &[u8]) -> RelayCopy {
        let existing = {
            let st = self.lock();
            st.keys
                .get(key)
                .and_then(|ks| ks.copies.get(&id))
                .and_then(Weak::upgrade)
        };
        // Dropped, if not shared, after the lock is released: it may be the
        // copy's last strong reference, and its drop takes the lock.
        if let Some(owner) = existing {
            if *owner.data == *data {
                let serial = owner.serial;
                return RelayCopy {
                    bytes: Bytes::from_owner(SharedCopy(owner)),
                    serial,
                };
            }
        }
        let copied: Box<[u8]> = data.into();
        let charge = (copied.len() as u64).saturating_add(RELAY_BUF_OVERHEAD_BYTES);
        // From 1, and a u64 a replica never exhausts: never 0, never reused.
        let serial = NonZeroU64::new(self.next_copy.fetch_add(1, Ordering::Relaxed))
            .expect("relay copy serials start at 1 and never wrap");
        let owner = {
            let mut st = self.lock();
            let key: Arc<str> = match st.keys.get_key_value(key) {
                Some((interned, _)) => Arc::clone(interned),
                None => Arc::from(key),
            };
            let owner = Arc::new(RelayOwner {
                data: copied,
                key: Arc::clone(&key),
                id,
                serial,
                charge,
                ledger: Arc::clone(self),
            });
            let ks = st.keys.entry(key).or_default();
            let before = ks.largest_charge();
            let copy = LiveCopy {
                charge,
                seq: Reverse(id.ms),
                serial,
            };
            ks.copies.insert(id, Arc::downgrade(&owner));
            ks.live.insert(copy);
            let after = ks.largest_charge();
            st.exempt = st.exempt.saturating_add(after).saturating_sub(before);
            self.exempt.store(st.exempt, Ordering::Release);
            self.add_charged(charge);
            owner
        };
        RelayCopy {
            bytes: Bytes::from_owner(SharedCopy(owner)),
            serial,
        }
    }

    /// A copy's last view dropped: out of its key's index and the dedupe map,
    /// `exempt` re-taken from the key's new largest copy, its charge repaid.
    fn release_copy(&self, key: &str, id: StreamId, copy: LiveCopy) {
        let mut st = self.lock();
        let mut exempt = st.exempt;
        let mut empty = false;
        if let Some(ks) = st.keys.get_mut(key) {
            let before = ks.largest_charge();
            ks.live.remove(&copy);
            if ks
                .copies
                .get(&id)
                .is_some_and(|weak| weak.strong_count() == 0)
            {
                ks.copies.remove(&id);
            }
            let after = ks.largest_charge();
            exempt = exempt.saturating_add(after).saturating_sub(before);
            empty = ks.live.is_empty() && ks.copies.is_empty();
        }
        if empty {
            st.keys.remove(key);
        }
        st.exempt = exempt;
        self.exempt.store(exempt, Ordering::Release);
        drop(st);
        self.sub_charged(copy.charge);
    }

    /// Registers a hold for one `SubscribeReadLayer` RPC, named by
    /// `identity` for the log lines that end it.
    pub fn register_hold(self: &Arc<Self>, identity: HoldIdentity) -> SubscriptionHold {
        let now = self.now_ms();
        let mut st = self.lock();
        let id = st.next_hold;
        st.next_hold = st.next_hold.wrapping_add(1);
        let shared = Arc::new(HoldShared {
            id,
            identity,
            ledger: Arc::downgrade(self),
            key: StdMutex::new(None),
            generation: AtomicU64::new(0),
            pipeline_bytes: AtomicU64::new(0),
            pipeline_entries: AtomicU64::new(0),
            backlog_entries: AtomicU64::new(0),
            copies: StdMutex::new(HashMap::new()),
            queued_catchup_bytes: AtomicU64::new(0),
            last_take_ms: AtomicU64::new(now),
            ending: CancellationToken::new(),
            reason: OnceLock::new(),
            stream_ended: AtomicBool::new(false),
            pending_credit: AtomicU64::new(0),
        });
        st.holds.insert(id, Arc::clone(&shared));
        SubscriptionHold {
            shared,
            ledger: Arc::clone(self),
        }
    }

    /// Unregisters `hold`: its pending credit leaves `pending_release`, and
    /// enforcement runs again when the last pass ended over the budget or the
    /// hold carried a pending credit. A credit can exceed what the hold's
    /// going frees — a relay copy it shared with another subscription stays
    /// charged — so the replica may still be over `D_live` once the credit is
    /// gone, and nothing else would enforce until the layer's next delivering
    /// read.
    fn unregister(&self, hold: &HoldShared) {
        let rerun = {
            let mut st = self.lock();
            st.holds.remove(&hold.id);
            let credit = hold.pending_credit.swap(0, Ordering::AcqRel);
            st.pending_release = st.pending_release.saturating_sub(credit);
            st.last_pass_over || credit > 0
        };
        if rerun {
            self.enforce();
        }
    }

    /// `hold`'s stream registration `generation` ended (its stream reader
    /// failed it): its chargeable bytes are going away with it, so they are
    /// credited to `pending_release` and it is no longer a candidate. Once per
    /// registration, and only while it is the hold's current one: the failure
    /// of a registration the hold has already replaced — its reader failing
    /// it before it saw the removal — credits nothing and leaves the new one
    /// a candidate. A hold already ended as a victim keeps its victim credit.
    pub(crate) fn credit_registration_ended(&self, hold: &HoldShared, generation: u64) {
        let mut st = self.lock();
        if hold.generation() != generation || hold.stream_ended.swap(true, Ordering::AcqRel) {
            return;
        }
        let chargeable = hold.chargeable(&st);
        if hold.pending_credit.load(Ordering::Acquire) == 0 {
            hold.pending_credit.store(chargeable, Ordering::Release);
            st.pending_release = st.pending_release.saturating_add(chargeable);
        }
    }

    /// [`Self::credit_registration_ended`] for `hold`'s current registration.
    #[cfg(test)]
    pub(crate) fn credit_stream_ended(&self, hold: &HoldShared) {
        self.credit_registration_ended(hold, hold.generation());
    }

    /// One enforcement pass (module docs): while the replica is over `D_live`,
    /// end the largest chargeable non-lone hold `delivery_budget`; with no
    /// such hold, count an overshoot. Never waits and never drops a copy.
    pub fn enforce(&self) {
        let budget = self.budget_bytes();
        let charged = self.charged_bytes();
        // `Some` when the pass ended over with no hold to end: the exempt
        // bytes and lone holders it saw, and whether to WARN (at most once
        // per `OVERSHOOT_WARN_INTERVAL`).
        let mut overshoot: Option<(u64, usize, bool)> = None;
        {
            let mut st = self.lock();
            loop {
                let held = charged
                    .saturating_sub(st.exempt)
                    .saturating_sub(st.pending_release);
                if held <= budget {
                    st.last_pass_over = false;
                    break;
                }
                let mut victim: Option<(u64, u64, Reverse<u64>, Arc<HoldShared>)> = None;
                let mut lone = 0usize;
                for hold in st.holds.values() {
                    if hold.is_ending() || hold.stream_ended.load(Ordering::Acquire) {
                        continue;
                    }
                    if hold.is_lone() {
                        if hold.pipeline_entries.load(Ordering::Acquire) > 0 {
                            lone += 1;
                        }
                        continue;
                    }
                    let chargeable = hold.chargeable(&st);
                    if chargeable == 0 {
                        continue;
                    }
                    let rank = (chargeable, hold.held_bytes(), Reverse(hold.id));
                    if victim
                        .as_ref()
                        .is_none_or(|(c, t, i, _)| rank > (*c, *t, *i))
                    {
                        victim = Some((rank.0, rank.1, rank.2, Arc::clone(hold)));
                    }
                }
                let Some((chargeable, _, _, hold)) = victim else {
                    st.last_pass_over = true;
                    let now = Instant::now();
                    let warn = st
                        .last_overshoot_warn
                        .is_none_or(|at| now.duration_since(at) >= OVERSHOOT_WARN_INTERVAL);
                    if warn {
                        st.last_overshoot_warn = Some(now);
                    }
                    overshoot = Some((st.exempt, lone, warn));
                    break;
                };
                // Added to any credit it still carries, so unregistering
                // repays every credit `pending_release` took for it.
                hold.pending_credit.fetch_add(chargeable, Ordering::AcqRel);
                st.pending_release = st.pending_release.saturating_add(chargeable);
                hold.end(STREAM_EXIT_DELIVERY_BUDGET);
            }
        }
        if let Some((exempt, lone, warn)) = overshoot {
            if let Some(m) = self.metrics() {
                m.inc_subscriber_delivery_budget_overshoot();
            }
            if warn {
                tracing::warn!(
                    charged_bytes = charged,
                    exempt_bytes = exempt,
                    budget_bytes = budget,
                    lone_holders = lone,
                    "live-delivery memory is over its budget and no subscription can be ended for \
                     it: every holder of undelivered updates holds at most two, or only its \
                     layer's largest pending update"
                );
            }
        }
    }

    /// Ends every hold but `waiter`'s whose response queue holds catch-up
    /// pages and has had nothing taken from it for `stalled_for`,
    /// `slow_reader_deadline`, with one WARN each naming the waiting catch-up
    /// `waiter`. Run while a catch-up waits for the catch-up payload budget
    /// (SDD §12.3); a hold whose queue is being taken from, and every hold
    /// when nothing waits, is never ended this way. Returns how many it ended.
    pub fn end_stalled_catchup_holders(
        &self,
        waiter: &SubscriptionHold,
        stalled_for: Duration,
    ) -> usize {
        let now = self.now_ms();
        let stalled_ms = u64::try_from(stalled_for.as_millis()).unwrap_or(u64::MAX);
        let stalled: Vec<Arc<HoldShared>> = {
            let st = self.lock();
            st.holds
                .values()
                .filter(|hold| {
                    hold.id != waiter.shared.id
                        && !hold.is_ending()
                        && hold.queued_catchup_bytes.load(Ordering::Acquire) > 0
                        && now.saturating_sub(hold.last_take_ms.load(Ordering::Acquire))
                            >= stalled_ms
                })
                .cloned()
                .collect()
        };
        let mut ended = 0;
        for hold in stalled {
            if hold.end(STREAM_EXIT_SLOW_READER_DEADLINE) {
                ended += 1;
                tracing::warn!(
                    session_id = %hold.identity.session_id,
                    layer_name = %hold.identity.layer_name,
                    client_id = %hold.identity.client_id,
                    queued_catchup_bytes = hold.queued_catchup_bytes.load(Ordering::Acquire),
                    stalled_ms = now.saturating_sub(hold.last_take_ms.load(Ordering::Acquire)),
                    waiting_session_id = %waiter.shared.identity.session_id,
                    waiting_layer_name = %waiter.shared.identity.layer_name,
                    waiting_client_id = %waiter.shared.identity.client_id,
                    "subscriber took nothing from its stream for longer than the outer send \
                     deadline while it held catch-up content and another catch-up waited for \
                     catch-up payload memory; ending it so its content is released"
                );
            }
        }
        ended
    }
}

/// Who a hold belongs to, for the log lines that end it.
#[derive(Debug, Clone, Default)]
pub struct HoldIdentity {
    pub session_id: String,
    pub layer_name: String,
    pub client_id: String,
}

/// One subscription's place in the ledger (module docs, *Holds*).
pub struct HoldShared {
    id: u64,
    identity: HoldIdentity,
    /// The ledger it is registered on, for a re-registration's credit
    /// repayment. Weak: the ledger holds the hold.
    ledger: Weak<DeliveryLedger>,
    /// The stream key of its current registration.
    key: StdMutex<Option<Arc<str>>>,
    /// Bumped by each registration, so the failure of a registration it has
    /// replaced credits nothing ([`DeliveryLedger::credit_registration_ended`]).
    generation: AtomicU64,
    /// Σ relay charges of the entries in its pipeline.
    pipeline_bytes: AtomicU64,
    pipeline_entries: AtomicU64,
    backlog_entries: AtomicU64,
    /// The relay copies its pipeline's entries carry, each by its serial,
    /// with how many entries carry it — of any registration, as
    /// `pipeline_bytes` counts them. At most one slot per entry in the
    /// pipeline (module docs, *Holds*).
    copies: StdMutex<HashMap<NonZeroU64, u32>>,
    /// Catch-up page bytes in its response queue.
    queued_catchup_bytes: AtomicU64,
    /// When its response queue was last taken from (or, while it held no
    /// catch-up page, when one was queued), in the ledger's milliseconds.
    last_take_ms: AtomicU64,
    ending: CancellationToken,
    reason: OnceLock<&'static str>,
    stream_ended: AtomicBool,
    /// Its chargeable bytes credited to `pending_release` when it was ended
    /// or its stream registration was, repaid at unregister or at its next
    /// registration.
    pending_credit: AtomicU64,
}

/// Locks a hold's per-copy count. Nothing panics while holding it and every
/// update leaves it consistent, so a poisoned lock's map is still sound.
fn lock_copies(
    copies: &StdMutex<HashMap<NonZeroU64, u32>>,
) -> MutexGuard<'_, HashMap<NonZeroU64, u32>> {
    copies.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A per-copy count's allocation is given back once it is this large and
/// four times what it holds, so a pipeline that drained after a burst does
/// not keep the burst's map for the subscription's life.
const COPIES_SHRINK_FROM: usize = 64;

impl std::fmt::Debug for HoldShared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HoldShared")
            .field("id", &self.id)
            .field(
                "pipeline_entries",
                &self.pipeline_entries.load(Ordering::Acquire),
            )
            .field(
                "backlog_entries",
                &self.backlog_entries.load(Ordering::Acquire),
            )
            .field(
                "pipeline_bytes",
                &self.pipeline_bytes.load(Ordering::Acquire),
            )
            .field("reason", &self.reason.get())
            .finish()
    }
}

impl HoldShared {
    fn is_ending(&self) -> bool {
        self.reason.get().is_some()
    }

    /// At most two entries in the pipeline, at most one of them in the
    /// stream backlog.
    fn is_lone(&self) -> bool {
        self.pipeline_entries.load(Ordering::Acquire) <= LONE_PIPELINE_ENTRIES
            && self.backlog_entries.load(Ordering::Acquire) <= LONE_BACKLOG_ENTRIES
    }

    /// Its pipeline's relay charges and queued overhead.
    fn held_bytes(&self) -> u64 {
        self.pipeline_bytes.load(Ordering::Acquire).saturating_add(
            self.backlog_entries
                .load(Ordering::Acquire)
                .saturating_mul(RELAY_ENTRY_OVERHEAD_BYTES),
        )
    }

    /// [`Self::held_bytes`] less its current registration's key's `E_k` when
    /// an entry in its pipeline carries that copy, by its serial. Called
    /// under the ledger's lock.
    fn chargeable(&self, st: &LedgerState) -> u64 {
        let held = self.held_bytes();
        let key = self.current_key();
        let Some(largest) = key
            .as_deref()
            .and_then(|k| st.keys.get(k))
            .and_then(KeyState::largest)
        else {
            return held;
        };
        if lock_copies(&self.copies).contains_key(&largest.serial) {
            held.saturating_sub(largest.charge)
        } else {
            held
        }
    }

    /// The stream key of its current registration.
    fn current_key(&self) -> Option<Arc<str>> {
        self.key
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// One more entry in its pipeline carries the copy `serial`.
    fn count_copy(&self, serial: NonZeroU64) {
        *lock_copies(&self.copies).entry(serial).or_insert(0) += 1;
    }

    /// One entry carrying the copy `serial` left its pipeline.
    fn uncount_copy(&self, serial: NonZeroU64) {
        let mut copies = lock_copies(&self.copies);
        if let Some(count) = copies.get_mut(&serial) {
            *count -= 1;
            if *count == 0 {
                copies.remove(&serial);
            }
        }
        let len = copies.len();
        if copies.capacity() >= COPIES_SHRINK_FROM && len * 4 <= copies.capacity() {
            copies.shrink_to(len * 2);
        }
    }

    /// Ends the subscription `reason`, once: fires its token. Returns whether
    /// this call ended it.
    fn end(&self, reason: &'static str) -> bool {
        if self.reason.set(reason).is_err() {
            return false;
        }
        self.ending.cancel();
        true
    }

    /// A new stream registration on `key`; returns its generation. Under the
    /// ledger's lock, it is a candidate again and, unless it is ending, the
    /// credit its last registration's stream ending took is repaid: that
    /// registration's entries are gone or going on their own, and the new
    /// registration's bytes are chargeable. An ending hold keeps its credit
    /// until it unregisters, so no other subscription is ended for the bytes
    /// it is about to free.
    pub(crate) fn begin_registration(&self, key: &str) -> u64 {
        let ledger = self.ledger.upgrade();
        let mut st = ledger.as_ref().map(|ledger| ledger.lock());
        *self.key.lock().unwrap_or_else(PoisonError::into_inner) = Some(Arc::from(key));
        self.stream_ended.store(false, Ordering::Release);
        if !self.is_ending() {
            let credit = self.pending_credit.swap(0, Ordering::AcqRel);
            if let Some(st) = st.as_mut() {
                st.pending_release = st.pending_release.saturating_sub(credit);
            }
        }
        let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        drop(st);
        generation
    }

    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }
}

/// A subscription's hold, registered for the RPC's whole life; dropping it
/// unregisters it (module docs, *Holds*).
pub struct SubscriptionHold {
    shared: Arc<HoldShared>,
    ledger: Arc<DeliveryLedger>,
}

impl std::fmt::Debug for SubscriptionHold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.shared.fmt(f)
    }
}

impl SubscriptionHold {
    /// Resolves when the ledger has ended the subscription: the live-delivery
    /// budget picked it ([`STREAM_EXIT_DELIVERY_BUDGET`]), or a stalled
    /// catch-up holder's ending did ([`STREAM_EXIT_SLOW_READER_DEADLINE`]).
    pub async fn ended(&self) {
        self.shared.ending.cancelled().await;
    }

    /// Why the ledger ended the subscription; `None` while it has not.
    pub fn ending_reason(&self) -> Option<&'static str> {
        self.shared.reason.get().copied()
    }

    /// The ledger.
    pub fn ledger(&self) -> &Arc<DeliveryLedger> {
        &self.ledger
    }

    /// The shared state a stream registration and the response stream carry.
    pub(crate) fn shared(&self) -> &Arc<HoldShared> {
        &self.shared
    }

    /// Entries in the pipeline now (tests and diagnostics).
    pub fn pipeline_entries(&self) -> u64 {
        self.shared.pipeline_entries.load(Ordering::Acquire)
    }

    /// Entries in the stream backlog now (tests and diagnostics).
    pub fn backlog_entries(&self) -> u64 {
        self.shared.backlog_entries.load(Ordering::Acquire)
    }

    /// Relay charges in the pipeline now (tests and diagnostics).
    pub fn pipeline_bytes(&self) -> u64 {
        self.shared.pipeline_bytes.load(Ordering::Acquire)
    }

    /// Catch-up page bytes in the response queue now (tests and diagnostics).
    pub fn queued_catchup_bytes(&self) -> u64 {
        self.shared.queued_catchup_bytes.load(Ordering::Acquire)
    }

    /// A guard counting `bytes` of catch-up page in the response queue until
    /// it drops — when the stream takes the page's message, or the message is
    /// dropped unsent. While the queue held no catch-up page, the take clock
    /// restarts here, so a page is never judged stalled for time before it
    /// was queued.
    pub fn queue_catchup_page(&self, bytes: u64) -> QueuedPage {
        let before = self
            .shared
            .queued_catchup_bytes
            .fetch_add(bytes, Ordering::AcqRel);
        if before == 0 {
            let now = self.ledger.now_ms();
            self.shared.last_take_ms.fetch_max(now, Ordering::AcqRel);
        }
        QueuedPage {
            hold: Arc::clone(&self.shared),
            bytes,
        }
    }

    /// A [`TakeClock`] the response stream stamps each take on.
    pub fn take_clock(&self) -> TakeClock {
        TakeClock {
            hold: Arc::clone(&self.shared),
            ledger: Arc::clone(&self.ledger),
        }
    }
}

impl Drop for SubscriptionHold {
    fn drop(&mut self) {
        self.ledger.unregister(&self.shared);
    }
}

/// Stamps the time of each take from a subscription's response queue on its
/// hold.
pub struct TakeClock {
    hold: Arc<HoldShared>,
    ledger: Arc<DeliveryLedger>,
}

impl std::fmt::Debug for TakeClock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TakeClock")
            .field("hold", &self.hold.id)
            .finish()
    }
}

impl TakeClock {
    /// The response stream took a message now.
    pub fn record_take(&self) {
        let now = self.ledger.now_ms();
        self.hold.last_take_ms.fetch_max(now, Ordering::AcqRel);
    }
}

/// Catch-up page bytes counted in a response queue until dropped
/// ([`SubscriptionHold::queue_catchup_page`]).
pub struct QueuedPage {
    hold: Arc<HoldShared>,
    bytes: u64,
}

impl std::fmt::Debug for QueuedPage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueuedPage")
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl Drop for QueuedPage {
    fn drop(&mut self) {
        let _ =
            self.hold
                .queued_catchup_bytes
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |b| {
                    Some(b.saturating_sub(self.bytes))
                });
    }
}

/// One entry in a subscription's pipeline, from the offer until it is
/// released: received from the stream backlog and then dropped — sent, its
/// last message taken by the response stream, or dropped unsent — or dropped
/// in the backlog. Charges [`RELAY_ENTRY_OVERHEAD_BYTES`] while in the
/// backlog, and counts on its hold, if it has one, while in the pipeline —
/// with its relay copy's serial on the hold's per-copy count (module docs,
/// *Holds*).
pub struct PipelineEntry {
    ledger: Arc<DeliveryLedger>,
    hold: Option<Arc<HoldShared>>,
    seq: u64,
    relay_charge: u64,
    /// The serial of the relay copy the entry carries, counted on its hold's
    /// per-copy count; `None` without a hold or a copy.
    copy: Option<NonZeroU64>,
    in_backlog: bool,
}

impl std::fmt::Debug for PipelineEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineEntry")
            .field("seq", &self.seq)
            .field("relay_charge", &self.relay_charge)
            .field("in_backlog", &self.in_backlog)
            .finish()
    }
}

impl PipelineEntry {
    /// An entry of sequence `seq` offered to a subscription (in its stream
    /// backlog), carrying a relay copy of `relay_len` bytes whose serial is
    /// `copy` ([`RelayCopy`]), or none. Takes only the hold's map lock.
    pub(crate) fn offer(
        ledger: &Arc<DeliveryLedger>,
        hold: Option<&Arc<HoldShared>>,
        seq: u64,
        relay_len: Option<usize>,
        copy: Option<NonZeroU64>,
    ) -> Self {
        let relay_charge = relay_len.map_or(0, |len| {
            (len as u64).saturating_add(RELAY_BUF_OVERHEAD_BYTES)
        });
        ledger.add_charged(RELAY_ENTRY_OVERHEAD_BYTES);
        let copy = hold.and(copy);
        if let Some(hold) = hold {
            // Counted on the copy before its bytes are, so a pass between the
            // two can only excuse too much, never charge `E_k` to a hold
            // holding it.
            if let Some(serial) = copy {
                hold.count_copy(serial);
            }
            hold.pipeline_entries.fetch_add(1, Ordering::AcqRel);
            hold.backlog_entries.fetch_add(1, Ordering::AcqRel);
            hold.pipeline_bytes
                .fetch_add(relay_charge, Ordering::AcqRel);
        }
        Self {
            ledger: Arc::clone(ledger),
            hold: hold.cloned(),
            seq,
            relay_charge,
            copy,
            in_backlog: true,
        }
    }

    /// The subscription received the entry from its stream backlog: its queued
    /// overhead is repaid; it stays in the pipeline until dropped.
    pub fn received(&mut self) {
        if !self.in_backlog {
            return;
        }
        self.in_backlog = false;
        self.ledger.sub_charged(RELAY_ENTRY_OVERHEAD_BYTES);
        if let Some(hold) = &self.hold {
            hold.backlog_entries.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

impl Drop for PipelineEntry {
    fn drop(&mut self) {
        self.received();
        if let Some(hold) = &self.hold {
            hold.pipeline_entries.fetch_sub(1, Ordering::AcqRel);
            let _ = hold
                .pipeline_bytes
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |b| {
                    Some(b.saturating_sub(self.relay_charge))
                });
            // Uncounted after its bytes, for the same reason as at the offer.
            if let Some(serial) = self.copy {
                hold.uncount_copy(serial);
            }
        }
    }
}

/// The ledger's rules on its own, with no stream reader, consumer or response
/// queue: each test drives copies ([`DeliveryLedger::relay_buf`]), offers
/// ([`PipelineEntry::offer`]), receives and releases in the order a stream
/// reader and a subscription task make them — entries offered to a
/// registration in sequence order, a reader's copy shared by every holder —
/// and reads `charged`, `exempt`, `pending_release`, the holds' counters and
/// their endings. The end-to-end paths (the reader's fan-out and enforcement,
/// the subscription's ending, the discard) are in
/// `tests/integration/grpc/delivery_budget_test.rs`.
#[cfg(test)]
mod tests {
    use super::*;

    use crate::observability::metrics::test_util::{render, sample_value};

    const KEY: &str = "le:obj:1:stream";

    /// A ledger of `budget` bytes with a metrics handle, so the overshoot
    /// counter can be read.
    fn ledger(budget: u64) -> (Arc<DeliveryLedger>, SharedMetrics) {
        let metrics: SharedMetrics = Arc::new(crate::observability::metrics::metrics_scrape_only(
            "delivery-ledger",
        ));
        let slot = Arc::new(StdMutex::new(Some(Arc::clone(&metrics))));
        (DeliveryLedger::new(budget, slot), metrics)
    }

    fn overshoots(metrics: &SharedMetrics) -> u64 {
        sample_value(
            &render(metrics),
            "live_edit_subscriber_delivery_budget_overshoots_total",
            &[],
        )
        .expect("the overshoot counter is seeded at zero") as u64
    }

    /// The stream id of an update at `seq` (`{seq}-0`).
    fn id(seq: u64) -> StreamId {
        StreamId { ms: seq, seq: 0 }
    }

    /// `len` bytes, all `fill`.
    fn bytes(len: usize, fill: u8) -> Vec<u8> {
        vec![fill; len]
    }

    /// A relay copy's charge: its length and the copy overhead.
    fn copy_charge(len: usize) -> u64 {
        len as u64 + RELAY_BUF_OVERHEAD_BYTES
    }

    /// One subscription: its hold, registered on [`KEY`], and the entries in
    /// its pipeline, each with the view of the relay copy its notification
    /// carries. Dropped whole, its hold goes before its pipeline; a test that
    /// reads what a hold's going does destructures it and drops the pipeline
    /// first, the order a subscription's ending makes (its response queue
    /// cleared, then its hold dropped: `grpc::subscribe`).
    struct Sub {
        hold: SubscriptionHold,
        pipeline: std::collections::VecDeque<(PipelineEntry, Option<Bytes>)>,
    }

    impl Sub {
        fn new(ledger: &Arc<DeliveryLedger>) -> Self {
            Self::on(ledger, KEY)
        }

        /// A subscription registered on stream `key`.
        fn on(ledger: &Arc<DeliveryLedger>, key: &str) -> Self {
            let hold = ledger.register_hold(HoldIdentity::default());
            hold.shared().begin_registration(key);
            Self {
                hold,
                pipeline: std::collections::VecDeque::new(),
            }
        }

        /// The reader offers entry `seq`, carrying `relay`, to this
        /// subscription's backlog.
        fn offer(&mut self, ledger: &Arc<DeliveryLedger>, seq: u64, relay: Option<&RelayCopy>) {
            let entry = PipelineEntry::offer(
                ledger,
                Some(self.hold.shared()),
                seq,
                relay.map(|r| r.len()),
                relay.map(RelayCopy::serial),
            );
            self.pipeline
                .push_back((entry, relay.map(|r| r.bytes().clone())));
        }

        /// The subscription task receives its oldest backlog entry.
        fn receive(&mut self) {
            let (entry, _) = self
                .pipeline
                .iter_mut()
                .find(|(e, _)| e.in_backlog)
                .expect("an entry in the backlog");
            entry.received();
        }

        /// The response stream took the oldest entry's last message: it
        /// leaves the pipeline.
        fn release(&mut self) {
            drop(self.pipeline.pop_front().expect("an entry in the pipeline"));
        }

        /// The subscription task receives entry `seq` and drops it unsent (its
        /// own write echoed back, or a duplicate), out of order: older entries
        /// stay in the pipeline.
        fn drop_unsent(&mut self, seq: u64) {
            let at = self
                .pipeline
                .iter()
                .position(|(e, _)| e.seq == seq)
                .expect("the entry is in the pipeline");
            let (mut entry, view) = self.pipeline.remove(at).expect("present");
            entry.received();
            drop((entry, view));
        }

        fn ended(&self) -> Option<&'static str> {
            self.hold.ending_reason()
        }

        /// Its chargeable bytes now, as an enforcement pass reads them.
        fn chargeable(&self, ledger: &DeliveryLedger) -> u64 {
            let st = ledger.lock();
            self.hold.shared().chargeable(&st)
        }

        /// Its pipeline's relay charges and queued overhead.
        fn held(&self) -> u64 {
            self.hold.shared().held_bytes()
        }
    }

    /// T7 (d): `charged` is every live relay copy once — however many
    /// subscriptions hold it — plus each entry's overhead while it is queued
    /// for a subscription, at every step: two subscriptions offered the same
    /// two entries, received and released one by one, the reader's own view
    /// dropped in between. It returns to zero, as does `exempt`, with the last
    /// view.
    #[test]
    fn charged_is_each_live_copy_once_plus_each_queued_entrys_overhead_at_every_step() {
        let (ledger, _) = ledger(u64::MAX);
        let (a_len, b_len) = (10_000, 3_000);
        let mut subs = [Sub::new(&ledger), Sub::new(&ledger)];
        let a = ledger.relay_buf(KEY, id(1), &bytes(a_len, 1));
        let b = ledger.relay_buf(KEY, id(2), &bytes(b_len, 2));
        let copies = copy_charge(a_len) + copy_charge(b_len);
        assert_eq!(ledger.charged_bytes(), copies, "two copies, nothing queued");
        assert_eq!(
            ledger.exempt_bytes(),
            copy_charge(a_len),
            "E_k is the larger"
        );
        for sub in &mut subs {
            sub.offer(&ledger, 1, Some(&a));
            sub.offer(&ledger, 2, Some(&b));
        }
        let overhead = RELAY_ENTRY_OVERHEAD_BYTES;
        assert_eq!(ledger.charged_bytes(), copies + 4 * overhead);
        // The reader's reply is gone: the copies live on in the notifications.
        drop((a, b));
        assert_eq!(ledger.charged_bytes(), copies + 4 * overhead);

        subs[0].receive();
        assert_eq!(ledger.charged_bytes(), copies + 3 * overhead);
        subs[0].release();
        subs[0].receive();
        subs[0].release();
        assert_eq!(
            ledger.charged_bytes(),
            copies + 2 * overhead,
            "the other subscription still holds both copies"
        );
        subs[1].receive();
        subs[1].release();
        assert_eq!(
            (ledger.charged_bytes(), ledger.exempt_bytes()),
            (copy_charge(b_len) + overhead, copy_charge(b_len)),
            "the first copy's last view dropped: its charge is repaid and E_k moves to the other"
        );
        subs[1].receive();
        subs[1].release();
        assert_eq!((ledger.charged_bytes(), ledger.exempt_bytes()), (0, 0));
        for sub in &subs {
            assert_eq!(
                (
                    sub.hold.pipeline_entries(),
                    sub.hold.backlog_entries(),
                    sub.hold.pipeline_bytes()
                ),
                (0, 0, 0)
            );
        }
    }

    /// T7 (e): a copy of an entry another live copy already holds — the same
    /// stream key, the same id and the same bytes, as a late joiner's re-read
    /// makes — is that copy: one allocation, charged once. The same id with
    /// other bytes (a recreated stream) is a copy of its own, as is the same id
    /// and bytes on another key, and a re-read after the last view dropped.
    #[test]
    fn a_re_read_of_a_live_entry_shares_its_copy_and_a_same_id_entry_with_other_bytes_does_not() {
        let (ledger, _) = ledger(u64::MAX);
        let data = bytes(4_096, 7);
        let first = ledger.relay_buf(KEY, id(5), &data);
        let reread = ledger.relay_buf(KEY, id(5), &data);
        assert_eq!(
            first.as_ptr(),
            reread.as_ptr(),
            "the re-read shares the copy"
        );
        assert_eq!(
            ledger.charged_bytes(),
            copy_charge(data.len()),
            "charged once"
        );

        let recreated = ledger.relay_buf(KEY, id(5), &bytes(4_096, 8));
        assert_ne!(first.as_ptr(), recreated.as_ptr());
        let other_key = ledger.relay_buf("le:obj:2:stream", id(5), &data);
        assert_ne!(first.as_ptr(), other_key.as_ptr());
        assert_eq!(ledger.charged_bytes(), 3 * copy_charge(data.len()));
        assert_eq!(
            &recreated[..],
            &bytes(4_096, 8)[..],
            "each copy holds its own bytes"
        );

        drop((first, reread, recreated, other_key));
        assert_eq!(ledger.charged_bytes(), 0);
        let again = ledger.relay_buf(KEY, id(5), &data);
        assert_eq!(ledger.charged_bytes(), copy_charge(data.len()));
        drop(again);
    }

    /// `exempt` is the sum over keys of each key's largest live copy, ties to
    /// the lowest sequence, re-taken as copies drop.
    #[test]
    fn exempt_is_each_keys_largest_live_copy() {
        let (ledger, _) = ledger(u64::MAX);
        let k1_small = ledger.relay_buf(KEY, id(1), &bytes(100, 1));
        let k1_large = ledger.relay_buf(KEY, id(2), &bytes(900, 2));
        let k2 = ledger.relay_buf("le:obj:2:stream", id(1), &bytes(400, 3));
        assert_eq!(
            ledger.exempt_bytes(),
            copy_charge(900) + copy_charge(400),
            "one exemption per key"
        );
        drop(k1_large);
        assert_eq!(ledger.exempt_bytes(), copy_charge(100) + copy_charge(400));
        drop((k1_small, k2));
        assert_eq!(ledger.exempt_bytes(), 0);
    }

    /// T6, the choice: over the budget, the victim is the hold with the most
    /// chargeable bytes — its pipeline less `E_k` when it holds it — ties to
    /// the larger whole hold. Here two holds have equal chargeable bytes — the
    /// one that sent and released the layer's largest entry is no longer
    /// excused it — and the one still holding it, registered second, is
    /// ended `delivery_budget`; its chargeable bytes count as pending release,
    /// which brings the replica within the budget, so the other is not ended,
    /// nor by a second pass while the victim has not unregistered. When it
    /// does — its pipeline released first, as a subscription's ending clears
    /// its queue before its hold goes — its bytes and its pending credit go
    /// together, and the pass its going runs finds the other still over the
    /// budget on its own (three entries, not lone) and ends it.
    #[test]
    fn the_victim_is_the_largest_chargeable_holder_ties_to_the_larger_hold_and_no_second_is_ended()
    {
        let (ledger, metrics) = ledger(1_000);
        let mut y = Sub::new(&ledger);
        let mut x = Sub::new(&ledger);
        let large = ledger.relay_buf(KEY, id(1), &bytes(50_000, 1));
        let two = ledger.relay_buf(KEY, id(2), &bytes(20_000, 2));
        let three = ledger.relay_buf(KEY, id(3), &bytes(5_000, 3));
        // `y` has sent the large entry already; it holds 2 and 3, and an
        // entry with no relay bytes (4), all in its backlog. Each of `x` and
        // `y` holds three entries, so neither is lone.
        y.offer(&ledger, 1, Some(&large));
        y.receive();
        y.release();
        y.offer(&ledger, 2, Some(&two));
        y.offer(&ledger, 3, Some(&three));
        y.offer(&ledger, 4, None);
        // `x` holds the large entry, and 2 and 3, all in its backlog.
        x.offer(&ledger, 1, Some(&large));
        x.offer(&ledger, 2, Some(&two));
        x.offer(&ledger, 3, Some(&three));
        drop((large, two, three));
        let x_held = x.hold.pipeline_bytes() + 3 * RELAY_ENTRY_OVERHEAD_BYTES;
        let x_chargeable = x_held - copy_charge(50_000);
        // The per-copy count's release leg: `y` sent and released the large
        // entry, `E_k`, while `x` still queues it, so `y` no longer carries
        // that copy and is not excused it.
        assert_eq!(
            y.chargeable(&ledger),
            y.held(),
            "a hold that sent and released `E_k` is charged all it holds"
        );
        assert_eq!(
            x_chargeable,
            y.held(),
            "fixture self-check: equal chargeable bytes, so the tie decides"
        );

        ledger.enforce();
        assert_eq!(
            (x.ended(), y.ended()),
            (Some(STREAM_EXIT_DELIVERY_BUDGET), None),
            "the larger whole hold is the victim, though registered later"
        );
        assert_eq!(ledger.pending_release_bytes(), x_chargeable);
        ledger.enforce();
        assert_eq!(
            y.ended(),
            None,
            "no second victim for bytes already pending release"
        );
        assert_eq!(
            overshoots(&metrics),
            0,
            "a pass that ended a victim is no overshoot"
        );

        let Sub { hold, pipeline } = x;
        drop(pipeline);
        // What `y` alone holds, `E_k` (entry 2, now the layer's largest live
        // copy, in its pipeline) excused: over the 1,000 B budget.
        let y_alone =
            y.hold.pipeline_bytes() + 3 * RELAY_ENTRY_OVERHEAD_BYTES - copy_charge(20_000);
        assert!(y_alone > ledger.budget_bytes(), "fixture self-check");
        drop(hold);
        assert_eq!(
            (y.ended(), ledger.pending_release_bytes()),
            (Some(STREAM_EXIT_DELIVERY_BUDGET), y_alone),
            "the victim's credit leaves with it, and the pass its going runs ends `y`, over the \
             budget on its own, crediting what it holds"
        );
        assert_eq!(overshoots(&metrics), 0);
        let Sub { hold, pipeline } = y;
        drop(pipeline);
        drop(hold);
        assert_eq!(
            (
                ledger.charged_bytes(),
                ledger.pending_release_bytes(),
                ledger.holds()
            ),
            (0, 0, 0)
        );
    }

    /// CR-4: a victim's pending credit can be more than its going frees — a
    /// relay copy it shared with another subscription stays charged — so when
    /// it unregisters the ledger runs a pass at once, rather than leaving the
    /// replica over `D_live` until the layer's next delivering read. Two holds
    /// share every copy: the first victim's going leaves the copies charged,
    /// and the pass it runs ends the other, so the replica is back within the
    /// budget with no further enforcement call. A victim whose going frees
    /// enough runs a pass that ends nothing and counts no overshoot.
    ///
    /// Fails on old code (in-place perturbation): with no pass at unregister,
    /// the other hold is left running and the replica over the budget.
    #[test]
    fn a_victim_leaving_runs_a_pass_that_ends_a_sharer_left_over_the_budget() {
        let (ledger, metrics) = ledger(1_000);
        let mut first = Sub::new(&ledger);
        let mut second = Sub::new(&ledger);
        for seq in 1..=3 {
            let relay = ledger.relay_buf(KEY, id(seq), &bytes(10_000, seq as u8));
            first.offer(&ledger, seq, Some(&relay));
            second.offer(&ledger, seq, Some(&relay));
        }
        ledger.enforce();
        assert_eq!(
            (first.ended(), second.ended()),
            (Some(STREAM_EXIT_DELIVERY_BUDGET), None),
            "one victim; its credit covers the rest"
        );
        let Sub { hold, pipeline } = first;
        drop(pipeline);
        let over =
            |l: &DeliveryLedger| l.charged_bytes() - l.exempt_bytes() - l.pending_release_bytes();
        assert!(
            ledger.charged_bytes() - ledger.exempt_bytes() > ledger.budget_bytes(),
            "fixture self-check: the shared copies stay charged once the victim's pipeline goes"
        );
        drop(hold);
        assert_eq!(
            second.ended(),
            Some(STREAM_EXIT_DELIVERY_BUDGET),
            "the pass the victim's going ran ended the sharer"
        );
        assert!(
            over(&ledger) <= ledger.budget_bytes(),
            "the replica is within D_live, with no further enforcement call"
        );
        assert_eq!(overshoots(&metrics), 0);
        let Sub { hold, pipeline } = second;
        drop(pipeline);
        drop(hold);

        // A victim whose going frees enough: the other subscription holds one
        // small entry of its own, lone, and is within the budget once the
        // victim's copies are gone.
        let mut victim = Sub::new(&ledger);
        let mut small = Sub::new(&ledger);
        for seq in 10..=12 {
            let relay = ledger.relay_buf(KEY, id(seq), &bytes(10_000, seq as u8));
            victim.offer(&ledger, seq, Some(&relay));
        }
        let edit = ledger.relay_buf(KEY, id(13), &bytes(100, 13));
        small.offer(&ledger, 13, Some(&edit));
        drop(edit);
        ledger.enforce();
        assert_eq!(
            (victim.ended(), small.ended()),
            (Some(STREAM_EXIT_DELIVERY_BUDGET), None)
        );
        let Sub { hold, pipeline } = victim;
        drop(pipeline);
        drop(hold);
        assert_eq!(
            (
                small.ended(),
                ledger.pending_release_bytes(),
                overshoots(&metrics)
            ),
            (None, 0, 0),
            "nothing left over the budget: the pass ends nothing and counts no overshoot"
        );
        drop(small);
        assert_eq!((ledger.charged_bytes(), ledger.holds()), (0, 0));
    }

    /// T6, the last tie: equal chargeable bytes and equal whole holds end the
    /// lower id, the hold registered first.
    #[test]
    fn equal_holders_end_the_one_registered_first() {
        let (ledger, _) = ledger(1_000);
        let mut first = Sub::new(&ledger);
        let mut second = Sub::new(&ledger);
        for seq in 1..=3 {
            let relay = ledger.relay_buf(KEY, id(seq), &bytes(10_000 * seq as usize, 1));
            first.offer(&ledger, seq, Some(&relay));
            second.offer(&ledger, seq, Some(&relay));
        }
        ledger.enforce();
        assert_eq!(
            (first.ended(), second.ended()),
            (Some(STREAM_EXIT_DELIVERY_BUDGET), None)
        );
    }

    /// T7 (a): a caught-up subscriber sending one entry larger than the whole
    /// budget, with small edits continuing behind it on the layer, is not
    /// ended: the layer's largest live entry is excused (`E_k`), and what is
    /// left is within the budget. Its pipeline holds three entries, so the
    /// two-entry lone rule does not protect it: only the excusal does.
    #[test]
    fn one_entry_larger_than_the_budget_with_small_edits_behind_it_ends_no_subscriber() {
        let (ledger, metrics) = ledger(64 * 1024);
        let mut sub = Sub::new(&ledger);
        let large = ledger.relay_buf(KEY, id(1), &bytes(1 << 20, 1));
        sub.offer(&ledger, 1, Some(&large));
        sub.receive();
        for seq in 2..=3 {
            let small = ledger.relay_buf(KEY, id(seq), &bytes(1_000, 2));
            sub.offer(&ledger, seq, Some(&small));
        }
        drop(large);
        assert!(
            ledger.charged_bytes() > ledger.budget_bytes(),
            "fixture self-check: the large entry alone is over the budget"
        );
        assert!(
            sub.hold.pipeline_entries() > LONE_PIPELINE_ENTRIES,
            "fixture self-check: not lone, so only the excusal can keep it"
        );
        ledger.enforce();
        assert_eq!(sub.ended(), None, "the layer's largest entry is excused");
        assert_eq!(overshoots(&metrics), 0, "and the replica is not over");
    }

    /// T7 (b) and (c): a subscriber sending one large entry with the next
    /// large entry queued — two in its pipeline, one in its backlog — is lone,
    /// and is never ended by the live-delivery budget, though the second
    /// entry alone is over it; with every holder over the budget lone, no
    /// subscription is ended and the pass counts an overshoot. (A third entry
    /// overflows the subscription's own backlog cap, which ends it
    /// `inner_watchdog`: `stream_mux_fault_test`.)
    #[test]
    fn every_holder_over_the_budget_lone_ends_none_and_counts_an_overshoot() {
        let (ledger, metrics) = ledger(1_000);
        let mut subs: Vec<Sub> = (0..3).map(|_| Sub::new(&ledger)).collect();
        let first = ledger.relay_buf(KEY, id(1), &bytes(100_000, 1));
        let second = ledger.relay_buf(KEY, id(2), &bytes(90_000, 2));
        for sub in &mut subs {
            sub.offer(&ledger, 1, Some(&first));
            sub.receive();
            sub.offer(&ledger, 2, Some(&second));
        }
        drop((first, second));
        assert!(
            ledger.charged_bytes() - ledger.exempt_bytes() > ledger.budget_bytes(),
            "fixture self-check: the second entry is over the budget"
        );
        assert_eq!(overshoots(&metrics), 0);
        ledger.enforce();
        for sub in &subs {
            assert_eq!(sub.ended(), None, "a lone holder is never a victim");
        }
        assert_eq!(overshoots(&metrics), 1, "the pass found no hold to end");

        // One more entry queued makes a holder not lone: it is the victim.
        let third = ledger.relay_buf(KEY, id(3), &bytes(1_000, 3));
        subs[2].offer(&ledger, 3, Some(&third));
        ledger.enforce();
        assert_eq!(
            subs.iter().map(Sub::ended).collect::<Vec<_>>(),
            vec![None, None, Some(STREAM_EXIT_DELIVERY_BUDGET)]
        );
    }

    /// T8, the ledger's side: a hold's counters and `charged` return to zero
    /// on every way an entry leaves a pipeline — dropped in the backlog
    /// unreceived (a consumer dropped with its backlog: a re-bootstrap, a
    /// disconnect), received and sent, or dropped unsent after it was
    /// received (a cleared response queue); after a victim ending, with the
    /// victim's pending credit; after its stream registration failed (an
    /// `inner_watchdog` or `stream_gap` ending), whose credit is taken once;
    /// and after a `slow_reader_deadline` ending whose entries linger past it
    /// before they drop.
    #[test]
    fn hold_counters_and_charged_return_to_zero_on_every_release_path() {
        let (ledger, _) = ledger(1_000);
        let assert_zero = |what: &str| {
            assert_eq!(
                (
                    ledger.charged_bytes(),
                    ledger.exempt_bytes(),
                    ledger.pending_release_bytes(),
                    ledger.holds()
                ),
                (0, 0, 0, 0),
                "{what}: the ledger is back to zero"
            );
        };
        let offer_three = |sub: &mut Sub| {
            for seq in 1..=3 {
                let relay = ledger.relay_buf(KEY, id(seq), &bytes(5_000, seq as u8));
                sub.offer(&ledger, seq, Some(&relay));
            }
        };

        // Dropped in the backlog, and received then dropped unsent.
        let mut sub = Sub::new(&ledger);
        offer_three(&mut sub);
        sub.receive();
        sub.pipeline.clear();
        assert_eq!(
            (
                sub.hold.pipeline_entries(),
                sub.hold.backlog_entries(),
                sub.hold.pipeline_bytes()
            ),
            (0, 0, 0)
        );
        drop(sub);
        assert_zero("dropped in the backlog and dropped unsent");

        // A victim ending, its entries dropped with its queue, then its hold.
        let mut victim = Sub::new(&ledger);
        offer_three(&mut victim);
        ledger.enforce();
        assert_eq!(victim.ended(), Some(STREAM_EXIT_DELIVERY_BUDGET));
        assert!(ledger.pending_release_bytes() > 0);
        victim.pipeline.clear();
        drop(victim);
        assert_zero("a victim ending");

        // A failed stream registration: credited once, released with it.
        let mut failed = Sub::new(&ledger);
        offer_three(&mut failed);
        ledger.credit_stream_ended(failed.hold.shared());
        let credited = ledger.pending_release_bytes();
        assert!(credited > 0);
        ledger.credit_stream_ended(failed.hold.shared());
        assert_eq!(ledger.pending_release_bytes(), credited, "credited once");
        ledger.enforce();
        assert_eq!(
            failed.ended(),
            None,
            "its bytes count as pending release, so no subscription is ended for them"
        );
        failed.pipeline.clear();
        drop(failed);
        assert_zero("a failed stream registration");

        // A `slow_reader_deadline` ending whose entries and catch-up page
        // linger (an HTTP/2 stream not yet reset) and drop later.
        let mut slow = Sub::new(&ledger);
        let waiter = ledger.register_hold(HoldIdentity::default());
        offer_three(&mut slow);
        slow.receive();
        let page = slow.hold.queue_catchup_page(4_096);
        assert_eq!(
            ledger.end_stalled_catchup_holders(&waiter, Duration::ZERO),
            1
        );
        assert_eq!(slow.ended(), Some(STREAM_EXIT_SLOW_READER_DEADLINE));
        assert!(
            ledger.charged_bytes() > 0,
            "the lingering entries still hold their bytes"
        );
        drop(page);
        assert_eq!(slow.hold.queued_catchup_bytes(), 0);
        slow.pipeline.clear();
        drop((slow, waiter));
        assert_zero("a lingering slow-reader ending");
    }

    /// An entry of an earlier registration released after the hold
    /// re-registered (a re-bootstrap onto the same stream) uncounts only
    /// itself. A hold counts its pipeline's entries per relay copy, of any
    /// registration, so when the old registration's lingering entry and the
    /// new one's first entry carry the same copy — the layer's largest, which
    /// the re-read of the live entry shares — the late release leaves the new
    /// entry counted: `a` still holds the layer's largest entry and is
    /// excused it. Were it not, `a`, whose chargeable bytes are small once
    /// that entry is excused, would be the victim instead of `b`, which holds
    /// the most chargeable bytes.
    #[test]
    fn a_late_release_from_an_earlier_registration_leaves_the_new_ones_largest_entry_excused() {
        let (ledger, _) = ledger(2_000);
        let mut a = Sub::new(&ledger);
        let large = bytes(100_000, 1);
        let first = ledger.relay_buf(KEY, id(1), &large);
        a.offer(&ledger, 1, Some(&first));
        let lingering = a.pipeline.pop_front().expect("offered");

        a.hold.shared().begin_registration(KEY);
        let reread = ledger.relay_buf(KEY, id(1), &large);
        assert_eq!(
            reread.serial(),
            first.serial(),
            "fixture self-check: the re-read shares the live copy"
        );
        drop(first);
        a.offer(&ledger, 1, Some(&reread));
        drop(reread);
        for seq in 2..=3 {
            let relay = ledger.relay_buf(KEY, id(seq), &bytes(10, seq as u8));
            a.offer(&ledger, seq, Some(&relay));
        }
        drop(lingering);
        assert_eq!(
            a.chargeable(&ledger),
            a.held() - copy_charge(100_000),
            "the old registration's late release leaves the new entry's count: still excused"
        );

        let other = "le:obj:2:stream";
        let mut b = Sub::on(&ledger, other);
        for seq in 1..=3 {
            let len = if seq == 1 { 6_000 } else { 5_000 };
            let relay = ledger.relay_buf(other, id(seq), &bytes(len, seq as u8));
            b.offer(&ledger, seq, Some(&relay));
        }
        ledger.enforce();
        assert_eq!(
            (a.ended(), b.ended()),
            (None, Some(STREAM_EXIT_DELIVERY_BUDGET)),
            "a still holds its layer's largest entry on its new registration, so it is excused \
             and b holds the most"
        );
    }

    /// #5, a late joiner: a hold registered after its layer's largest entry
    /// `E_k` was committed, holding only later entries, never held `E_k` and
    /// is not excused it — whatever it has or has not released — so it is
    /// chargeable for everything it holds and is chosen over a hold that does
    /// hold `E_k`, whose other bytes are fewer.
    ///
    /// Fails on old code (in-place perturbation: `chargeable`'s membership
    /// test reverted to the offered-through / released-through window): the
    /// late joiner, which has released nothing, reads as holding `E_k`, is
    /// excused it down to nothing, and the hold that does hold it is ended
    /// instead.
    #[test]
    fn a_late_joiner_holding_only_later_entries_is_not_excused_the_layers_largest_entry() {
        let (ledger, _) = ledger(2_000);
        let mut early = Sub::new(&ledger);
        let large = ledger.relay_buf(KEY, id(1), &bytes(100_000, 1));
        early.offer(&ledger, 1, Some(&large));
        // Registered after `E_k` (entry 1) was committed and offered.
        let mut late = Sub::new(&ledger);
        for (seq, len) in [(2, 5_000), (3, 5_000), (4, 30_000)] {
            let relay = ledger.relay_buf(KEY, id(seq), &bytes(len, seq as u8));
            early.offer(&ledger, seq, Some(&relay));
            late.offer(&ledger, seq, Some(&relay));
        }
        drop(large);
        // `early` has received all four (none in its backlog); `late` none.
        for _ in 0..4 {
            early.receive();
        }
        assert_eq!(
            ledger.exempt_bytes(),
            copy_charge(100_000),
            "E_k is entry 1"
        );
        assert_eq!(
            (early.chargeable(&ledger), late.chargeable(&ledger)),
            (early.held() - copy_charge(100_000), late.held()),
            "the hold holding E_k is excused it; the late joiner is not"
        );
        assert!(
            late.chargeable(&ledger) > early.chargeable(&ledger),
            "fixture self-check: the late joiner holds the most chargeable bytes"
        );

        ledger.enforce();
        assert_eq!(
            (late.ended(), early.ended()),
            (Some(STREAM_EXIT_DELIVERY_BUDGET), None),
            "the late joiner is the victim; the hold of E_k is left"
        );
    }

    /// #5, entries released out of order: a subscriber that receives a later
    /// entry and drops it unsent (its own write echoed back) while the
    /// layer's largest entry `E_k` still waits in its response queue still
    /// holds `E_k` and is excused it, so it is not ended ahead of the real
    /// laggard, which holds more of the layer's other entries.
    ///
    /// Fails on old code (in-place perturbation: `chargeable`'s membership
    /// test reverted to the offered-through / released-through window): the
    /// drop moves the writer's released-through past `E_k`, which is then
    /// charged to it, and the writer is ended instead of the laggard.
    #[test]
    fn an_unsent_later_entry_dropped_while_the_largest_is_queued_leaves_the_largest_excused() {
        let (ledger, _) = ledger(2_000);
        let mut writer = Sub::new(&ledger);
        let mut laggard = Sub::new(&ledger);
        for (seq, len) in [(1, 100_000), (2, 1_000), (3, 5_000), (4, 5_000)] {
            let relay = ledger.relay_buf(KEY, id(seq), &bytes(len, seq as u8));
            writer.offer(&ledger, seq, Some(&relay));
            laggard.offer(&ledger, seq, Some(&relay));
        }
        // The writer takes `E_k` (entry 1) into its response queue, then
        // receives its own write (entry 2) and drops it unsent.
        writer.receive();
        writer.drop_unsent(2);
        assert_eq!(
            writer.hold.pipeline_entries(),
            3,
            "fixture self-check: E_k, 3 and 4 are in the writer's pipeline, so it is not lone"
        );
        assert_eq!(
            (writer.chargeable(&ledger), laggard.chargeable(&ledger)),
            (
                writer.held() - copy_charge(100_000),
                laggard.held() - copy_charge(100_000)
            ),
            "both still hold E_k, and both are excused it"
        );
        assert!(
            laggard.chargeable(&ledger) > writer.chargeable(&ledger),
            "fixture self-check: the laggard holds the most chargeable bytes"
        );

        ledger.enforce();
        assert_eq!(
            (laggard.ended(), writer.ended()),
            (Some(STREAM_EXIT_DELIVERY_BUDGET), None),
            "the laggard is the victim; the writer, still sending E_k, is left"
        );
    }

    /// #5, `E_k` falling back: when a layer's largest copy leaves, `E_k` is
    /// the largest older copy still live, and only a hold whose pipeline
    /// holds that copy is excused it. `a` holds entry 1 in its response
    /// queue, having dropped entry 2 (its own write) unsent; `b` joined after
    /// entry 1 and sent entry 2; `c` joined after entry 2. Once entry 2's last
    /// view drops, entry 1 is `E_k`: `a` is excused it, `b` and `c` are not,
    /// and `b`, holding the most chargeable bytes, is the victim.
    ///
    /// Fails on old code (in-place perturbation: `chargeable`'s membership
    /// test reverted to the offered-through / released-through window): `a`,
    /// released through entry 2, reads as not holding entry 1 and is charged
    /// it, and is ended instead of `b`.
    #[test]
    fn when_the_largest_copy_leaves_only_holds_of_the_older_copy_that_becomes_e_k_are_excused() {
        let (ledger, _) = ledger(2_000);
        let mut a = Sub::new(&ledger);
        let one = ledger.relay_buf(KEY, id(1), &bytes(50_000, 1));
        a.offer(&ledger, 1, Some(&one));
        let mut b = Sub::new(&ledger);
        let two = ledger.relay_buf(KEY, id(2), &bytes(100_000, 2));
        a.offer(&ledger, 2, Some(&two));
        b.offer(&ledger, 2, Some(&two));
        let mut c = Sub::new(&ledger);
        assert_eq!(
            ledger.exempt_bytes(),
            copy_charge(100_000),
            "E_k is entry 2"
        );
        assert_eq!(
            (a.chargeable(&ledger), b.chargeable(&ledger)),
            (
                a.held() - copy_charge(100_000),
                b.held() - copy_charge(100_000)
            ),
            "while entry 2 is E_k, both its holders are excused it"
        );

        // `a` takes entry 1 into its response queue and drops entry 2, its
        // own write, unsent; `b` sends entry 2. Its last view drops.
        a.receive();
        a.drop_unsent(2);
        b.receive();
        b.release();
        drop((one, two));
        assert_eq!(
            ledger.exempt_bytes(),
            copy_charge(50_000),
            "E_k falls back to entry 1"
        );

        for (seq, len) in [(3, 1_000), (4, 1_000), (5, 20_000), (6, 10_000)] {
            let relay = ledger.relay_buf(KEY, id(seq), &bytes(len, seq as u8));
            for sub in [&mut a, &mut b, &mut c] {
                sub.offer(&ledger, seq, Some(&relay));
            }
        }
        // `a` has received 3 to 6 behind entry 1; `b` and `c` none.
        for _ in 0..4 {
            a.receive();
        }
        assert_eq!(
            (
                a.chargeable(&ledger),
                b.chargeable(&ledger),
                c.chargeable(&ledger)
            ),
            (a.held() - copy_charge(50_000), b.held(), c.held()),
            "only a holds entry 1, the new E_k, and only a is excused it"
        );
        assert!(
            b.chargeable(&ledger) > a.chargeable(&ledger),
            "fixture self-check: b holds more chargeable bytes than a"
        );

        ledger.enforce();
        assert_eq!(
            (a.ended(), b.ended(), c.ended()),
            (None, Some(STREAM_EXIT_DELIVERY_BUDGET), None),
            "b, holding the most chargeable bytes and registered before c, is the victim; a, \
             still sending E_k, is left"
        );
    }

    /// #6: a hold whose stream registration ended carries that credit in
    /// `pending_release` until its next registration, which repays it; a
    /// victim ending after that adds its own credit, and unregistering repays
    /// it, so nothing is left pending: credit, re-registration, victim and
    /// unregister leave `pending_release` at 0. A hold that re-registers while
    /// ending (a victim) keeps its credit until it unregisters.
    ///
    /// Fails on old code (in-place perturbation): with no repayment at
    /// re-registration, the old credit is still pending once the hold is a
    /// candidate again, and with the victim's credit stored over it, the old
    /// credit is never repaid.
    #[test]
    fn a_registration_credit_is_repaid_at_re_registration_and_a_victim_credit_at_unregister() {
        let (ledger, _) = ledger(1_000);
        let offer_three = |sub: &mut Sub, from: u64| {
            for seq in from..from + 3 {
                let relay = ledger.relay_buf(KEY, id(seq), &bytes(5_000, seq as u8));
                sub.offer(&ledger, seq, Some(&relay));
            }
        };

        let mut sub = Sub::new(&ledger);
        offer_three(&mut sub, 1);
        ledger.credit_stream_ended(sub.hold.shared());
        assert!(
            ledger.pending_release_bytes() > 0,
            "the registration's credit"
        );
        // The re-registration: the old consumer's entries go with it.
        sub.pipeline.clear();
        sub.hold.shared().begin_registration(KEY);
        assert_eq!(
            ledger.pending_release_bytes(),
            0,
            "the old registration's credit is repaid at the next registration"
        );
        offer_three(&mut sub, 4);
        let chargeable = sub.chargeable(&ledger);
        ledger.enforce();
        assert_eq!(
            (sub.ended(), ledger.pending_release_bytes()),
            (Some(STREAM_EXIT_DELIVERY_BUDGET), chargeable),
            "a candidate again, it is the victim, credited only what it holds now"
        );
        let Sub { hold, pipeline } = sub;
        drop(pipeline);
        drop(hold);
        assert_eq!(
            (
                ledger.pending_release_bytes(),
                ledger.charged_bytes(),
                ledger.holds()
            ),
            (0, 0, 0),
            "credit, re-registration, victim and unregister leave nothing pending"
        );

        // A victim re-registering keeps its credit until it unregisters.
        let mut victim = Sub::new(&ledger);
        offer_three(&mut victim, 7);
        ledger.enforce();
        assert_eq!(victim.ended(), Some(STREAM_EXIT_DELIVERY_BUDGET));
        let credit = ledger.pending_release_bytes();
        assert!(credit > 0);
        victim.hold.shared().begin_registration(KEY);
        assert_eq!(
            ledger.pending_release_bytes(),
            credit,
            "an ending hold keeps its credit at re-registration"
        );
        let Sub { hold, pipeline } = victim;
        drop(pipeline);
        drop(hold);
        assert_eq!((ledger.pending_release_bytes(), ledger.holds()), (0, 0));
    }

    /// #6: the failure of a registration the hold has already replaced — its
    /// stream reader failing it before it saw the removal — credits nothing,
    /// and leaves the hold's current registration a candidate; a failure of
    /// the current registration after a victim ending keeps the victim's
    /// credit, not adding a second.
    ///
    /// Fails on old code (in-place perturbation: no generation check): the
    /// stale failure credits the new registration's bytes and takes the hold
    /// out of the candidates, so the pass ends nothing for them.
    #[test]
    fn a_replaced_registrations_late_failure_credits_nothing() {
        let (ledger, metrics) = ledger(1_000);
        let mut sub = Sub::new(&ledger);
        let replaced = sub.hold.shared().generation();
        let current = sub.hold.shared().begin_registration(KEY);
        assert_ne!(replaced, current);
        for seq in 1..=3 {
            let relay = ledger.relay_buf(KEY, id(seq), &bytes(5_000, seq as u8));
            sub.offer(&ledger, seq, Some(&relay));
        }
        ledger.credit_registration_ended(sub.hold.shared(), replaced);
        assert_eq!(
            ledger.pending_release_bytes(),
            0,
            "the replaced registration's failure credits nothing"
        );
        let chargeable = sub.chargeable(&ledger);
        ledger.enforce();
        assert_eq!(
            (
                sub.ended(),
                ledger.pending_release_bytes(),
                overshoots(&metrics)
            ),
            (Some(STREAM_EXIT_DELIVERY_BUDGET), chargeable, 0),
            "still a candidate, it is the victim"
        );
        ledger.credit_registration_ended(sub.hold.shared(), current);
        assert_eq!(
            ledger.pending_release_bytes(),
            chargeable,
            "the current registration's failure after a victim ending adds no second credit"
        );
        let Sub { hold, pipeline } = sub;
        drop(pipeline);
        drop(hold);
        assert_eq!((ledger.pending_release_bytes(), ledger.holds()), (0, 0));
    }

    /// T10, the ledger's side: while a catch-up waits for catch-up payload
    /// memory, a hold whose response queue holds catch-up pages and has had
    /// nothing taken for the deadline is ended `slow_reader_deadline`; one
    /// that took a message within it, one holding no page, the waiting
    /// catch-up's own hold and a hold already ended are not.
    #[test]
    fn stalled_catchup_holders_are_ended_only_when_stalled_holding_pages() {
        let (ledger, _) = ledger(u64::MAX);
        let waiter = ledger.register_hold(HoldIdentity::default());
        let stalled = ledger.register_hold(HoldIdentity::default());
        let reading = ledger.register_hold(HoldIdentity::default());
        let no_pages = ledger.register_hold(HoldIdentity::default());
        let _stalled_page = stalled.queue_catchup_page(1 << 20);
        let _reading_page = reading.queue_catchup_page(1 << 20);
        let _waiter_page = waiter.queue_catchup_page(1 << 20);
        let reading_clock = reading.take_clock();
        let deadline = Duration::from_secs(3_600);

        // Within the deadline nothing is stalled, the one that just took a
        // message least of all.
        reading_clock.record_take();
        assert_eq!(ledger.end_stalled_catchup_holders(&waiter, deadline), 0);
        // Past it — a zero deadline here, so no time need pass — every hold
        // holding pages is, but for the waiter.
        assert_eq!(
            ledger.end_stalled_catchup_holders(&waiter, Duration::ZERO),
            2,
            "with a zero deadline every page holder but the waiter is stalled"
        );
        assert_eq!(
            (
                stalled.ending_reason(),
                reading.ending_reason(),
                no_pages.ending_reason(),
                waiter.ending_reason()
            ),
            (
                Some(STREAM_EXIT_SLOW_READER_DEADLINE),
                Some(STREAM_EXIT_SLOW_READER_DEADLINE),
                None,
                None
            ),
            "a hold with no page, and the waiter itself, are never ended"
        );
        assert_eq!(
            ledger.end_stalled_catchup_holders(&waiter, Duration::ZERO),
            0,
            "a hold already ended is not ended again"
        );
    }

    /// T10, the take clock: a hold that took a message within the deadline is
    /// not stalled, and a page queued into an empty queue restarts the clock,
    /// so a page is never judged stalled for time before it was queued.
    #[test]
    fn a_recent_take_or_a_freshly_queued_page_is_not_stalled() {
        let (ledger, _) = ledger(u64::MAX);
        let waiter = ledger.register_hold(HoldIdentity::default());
        let hold = ledger.register_hold(HoldIdentity::default());
        let deadline = Duration::from_secs(3_600);
        let page = hold.queue_catchup_page(1);
        hold.take_clock().record_take();
        assert_eq!(ledger.end_stalled_catchup_holders(&waiter, deadline), 0);
        drop(page);
        let _page = hold.queue_catchup_page(1);
        assert_eq!(ledger.end_stalled_catchup_holders(&waiter, deadline), 0);
        assert_eq!(hold.ending_reason(), None);
    }

    /// T11, at scale: 6,553 holds (an 8 GiB pod's subscription ceiling) over
    /// 256 layers, each offered a pipeline of three entries on its layer.
    /// `charged` is exact, `exempt` is one largest copy per layer, a pass
    /// ends only non-lone holds until the replica is within the budget — the
    /// largest first, so the fewest — and everything returns to zero.
    #[test]
    fn a_ledger_of_6553_holds_over_256_layers_stays_exact_and_ends_the_fewest_largest_holders() {
        const HOLDS: usize = 6_553;
        const KEYS: usize = 256;
        let (ledger, _) = ledger(u64::MAX);
        let keys: Vec<String> = (0..KEYS).map(|k| format!("le:obj:{k}:stream")).collect();
        // Per layer, entries 1..=3; entry sizes vary by layer so the holds'
        // chargeable bytes differ.
        let copies: Vec<[RelayCopy; 3]> = keys
            .iter()
            .enumerate()
            .map(|(k, key)| {
                [1u64, 2, 3].map(|seq| {
                    ledger.relay_buf(key, id(seq), &bytes(100 + k * 10 + seq as usize, seq as u8))
                })
            })
            .collect();
        let mut subs: Vec<(usize, SubscriptionHold, Vec<PipelineEntry>)> = (0..HOLDS)
            .map(|h| {
                let k = h % KEYS;
                let hold = ledger.register_hold(HoldIdentity::default());
                hold.shared().begin_registration(&keys[k]);
                let entries = (0..3)
                    .map(|i| {
                        PipelineEntry::offer(
                            &ledger,
                            Some(hold.shared()),
                            i as u64 + 1,
                            Some(copies[k][i].len()),
                            Some(copies[k][i].serial()),
                        )
                    })
                    .collect();
                (k, hold, entries)
            })
            .collect();
        let copy_total: u64 = copies.iter().flatten().map(|c| copy_charge(c.len())).sum();
        assert_eq!(
            ledger.charged_bytes(),
            copy_total + (HOLDS * 3) as u64 * RELAY_ENTRY_OVERHEAD_BYTES,
            "every copy once, every queued entry's overhead"
        );
        let exempt: u64 = copies.iter().map(|c| copy_charge(c[2].len())).sum();
        assert_eq!(ledger.exempt_bytes(), exempt, "one exemption per layer");

        // A budget a few holds over: the pass ends the largest chargeable
        // holders, on the layer with the largest entries.
        let held = ledger.charged_bytes() - ledger.exempt_bytes();
        let largest_chargeable = {
            let c = &copies[KEYS - 1];
            copy_charge(c[0].len()) + copy_charge(c[1].len()) + 3 * RELAY_ENTRY_OVERHEAD_BYTES
        };
        ledger.set_budget_bytes_for_test(held - 2 * largest_chargeable);
        ledger.enforce();
        let ended: Vec<usize> = subs
            .iter()
            .filter(|(_, hold, _)| hold.ending_reason().is_some())
            .map(|(k, _, _)| *k)
            .collect();
        assert_eq!(
            ended,
            vec![KEYS - 1, KEYS - 1],
            "two victims, both on the layer with the largest entries"
        );

        // Every subscription receives and releases; every view drops.
        for (_, _, entries) in &mut subs {
            for entry in entries.iter_mut() {
                entry.received();
            }
        }
        drop(subs);
        drop(copies);
        assert_eq!(
            (
                ledger.charged_bytes(),
                ledger.exempt_bytes(),
                ledger.pending_release_bytes(),
                ledger.holds()
            ),
            (0, 0, 0, 0)
        );
    }
}
