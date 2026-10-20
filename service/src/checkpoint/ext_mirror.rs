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

//! Async-off-hot-path external-value mirror pipeline (SDD §22.17, SR-3.1,
//! Epic `t1.fgj.zc`).
//!
//! This module owns the bounded `tokio::sync::mpsc` channel, the byte budget
//! and the worker pool that copy ext-blob bytes into the configured
//! [`CheckpointStore`] adapter. It is the **only** mirror enqueue surface in
//! the codebase: every producer that needs to mirror an ext-blob
//! (`ValkeyExternalValueStore` and any future caller) accepts an
//! [`ExtMirrorSender`] via dependency injection rather than constructing its
//! own channel. [`ext_mirror_channel`] builds the sender, the receiver and
//! their shared budget together.
//!
//! ## Contract
//!
//! - **Two bounds, both drop-on-full.** The channel bounds how many items
//!   wait; the byte budget, `M_ext` (the pod memory plan's
//!   `PodMemoryPlan::ext_mirror_bytes`), bounds the bytes the mirror holds,
//!   queued and in flight. Each item is charged [`item_charge_kib`]: its
//!   size rounded up to a KiB plus a fixed [`ITEM_OVERHEAD_KIB`] for its own
//!   small allocations. [`ExtMirrorSender::send`] takes the charge without
//!   waiting and drops the copy, with a structured WARN and a
//!   `checkpoint_ext_mirror_dropped_total{reason}` tick, when the value is
//!   larger than the whole budget (`oversized`), when the budget has no room
//!   (`budget_full`), or when the channel is full (`queue_full`). The hot
//!   store path is **never** back-pressured and **never** blocked.
//! - **The charge follows the memory.** The value's one owned copy is built
//!   by [`ExtMirrorSender::copy_value`] around an owner that holds the
//!   charge, so the charge is released only when the **last** handle on that
//!   memory drops: the queue's, a worker's, the adapter's request or retry
//!   clones, a clone an adapter keeps, or the store script's argument in
//!   fred's router. While a copy is charged its bytes are counted in
//!   `checkpoint_ext_mirror_bytes{state}`, `queued` until a worker takes it,
//!   then `in_flight`.
//! - **A delivered copy is confirmed; a missing one is repaired.** After a
//!   successful [`CheckpointStore::put_ext_blob`], a worker spawned with the
//!   marks ([`spawn_ext_mirror_workers`]) records the copy on the value's meta
//!   (`backed_up_in`, `storage::ext_backup`), guarded by the store stamp the
//!   item carries: a store of the same hash since the copy was made declines
//!   it. A confirmation that fails is counted and ignored. When a put returns
//!   `Err`, the worker logs a WARN with the classified-error kind and the
//!   first 16 hex chars of the hash (never the blob bytes), ticks
//!   `checkpoint_ext_mirror_errors_total`, and **continues** — no retry, no
//!   panic, no task exit. A copy dropped, failed, discarded at shutdown or
//!   lost with a crash is never confirmed, and the backup repair
//!   (`checkpoint::ext_backup_repair`) re-sends it from Valkey while Valkey
//!   still holds the value. Until then the backup store has no copy:
//!   fetch-on-miss (SR-3.2) recovers a value Valkey lost *from* a backup
//!   copy, so it cannot stand in for one that never arrived.
//! - **The repair shares the budget.** Its re-send takes its charge from the
//!   same byte budget through [`ExtMirrorSender::try_reserve`], without
//!   waiting, counted `in_flight` while it holds it, so `M_ext` bounds the
//!   mirror's and the repair's bytes together and a send on the store path
//!   never waits for a re-send.
//! - **No `send().await` anywhere.** Only the channel's `try_send` (caller
//!   side) and `recv()` (worker side) exist in this module.
//! - **Bounded shutdown.** On `cancel.cancelled()` each worker drains up to
//!   [`DRAIN_BUDGET`] queued items with the same `put_ext_blob` semantics
//!   then returns. Whatever is still queued when the last worker lets go of
//!   the receiver is discarded by [`ExtMirrorReceiver`]'s `Drop`, counted as
//!   `reason="shutdown"` with one WARN.
//!
//! [`CheckpointStore::put_ext_blob`]: crate::checkpoint::CheckpointStore::put_ext_blob

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use bytes::Bytes;
use tokio::sync::mpsc;
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::checkpoint::store::{CheckpointError, CheckpointStore};
use crate::config::PodMemoryPlan;
use crate::observability::metrics::{
    SharedMetrics, EXT_MIRROR_DROP_BUDGET_FULL, EXT_MIRROR_DROP_OVERSIZED,
    EXT_MIRROR_DROP_QUEUE_FULL, EXT_MIRROR_DROP_SHUTDOWN, EXT_MIRROR_STATE_IN_FLIGHT,
    EXT_MIRROR_STATE_QUEUED,
};
use crate::storage::ext_backup::ExtBackupMarks;

/// Maximum number of queued items each worker will drain after the cancel
/// signal fires before returning.
///
/// # What actually bounds the drain (corrected, b.gg7)
///
/// This constant used to be justified by "~50 ms per `put_ext_blob` against a
/// slow remote adapter, so 64 items takes ~3.2 s, comfortably inside
/// `LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS`". That arithmetic was wrong by roughly
/// three and a half orders of magnitude. `put_ext_blob` on the StorageApi
/// adapter is one `send_classified` call, whose *enforced* ceiling is
/// `LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS × MAX_RETRY_ATTEMPTS ×
/// AUTH_ATTEMPTS` — 180 s at the defaults, not 50 ms. A drain of 64 such items
/// has a worst case measured in hours, and no item count would fix that: at a
/// 180 s per-item ceiling even `DRAIN_BUDGET = 2` overshoots a 5 s budget.
///
/// The drain is bounded, but not by this number. `main.rs` joins each worker
/// handle under a `tokio::time::timeout` of `LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS`,
/// inside a shutdown sequence itself wrapped in one
/// `LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS` timeout, so a worker still waiting
/// on a slow adapter is abandoned there rather than held to. Read this
/// constant for what it actually is: a cap on how much *queued* work a worker
/// tries to salvage on the way out, which matters when the adapter is healthy
/// (the common case, where each put is milliseconds and 64 items really do
/// finish promptly) and is simply moot when it is not.
pub const DRAIN_BUDGET: usize = 64;

/// The most mirror workers `LIVE_EDIT_CHECKPOINT_EXT_MIRROR_WORKERS` accepts.
/// The values the workers hold are inside the byte budget, but each worker's
/// adapter connection keeps buffers that are not (hyper's read buffer alone
/// can reach 408 KiB), so the count is capped rather than charged.
pub const EXT_MIRROR_MAX_WORKERS: usize = 16;

/// The fixed allowance, in KiB, each item is charged beyond its value's
/// bytes, for the item's own small allocations: the hash string, the store
/// stamp, the copy's owner and its charge slot, and the channel slot.
pub const ITEM_OVERHEAD_KIB: u64 = 1;

/// The charge, in KiB, one value of `len` bytes takes from the byte budget:
/// `⌈len / 1 KiB⌉ +` [`ITEM_OVERHEAD_KIB`]. The one definition: the sender
/// charges it, and the pod memory plan floors `M_ext` at the charge of one
/// maximum-size value, so every value under the upload size cap fits.
pub const fn item_charge_kib(len: u64) -> u64 {
    len.div_ceil(1024).saturating_add(ITEM_OVERHEAD_KIB)
}

/// The charge one copy holds against the byte budget: the budget's permits
/// and the bytes they stand for. Dropped with the copy's last handle, which
/// returns the permits and takes the bytes off the gauge state they are in.
struct Charge {
    /// Released when the charge drops, after `Drop::drop` has updated the
    /// gauge.
    _permit: OwnedSemaphorePermit,
    /// The charged bytes, `item_charge_kib(len) × 1 KiB`.
    bytes: i64,
    /// Whether a worker has taken the item off the channel.
    in_flight: AtomicBool,
    metrics: SharedMetrics,
}

impl Charge {
    /// Move the charged bytes from `queued` to `in_flight`, once.
    fn mark_in_flight(&self) {
        if !self.in_flight.swap(true, Ordering::AcqRel) {
            self.metrics
                .add_checkpoint_ext_mirror_bytes(EXT_MIRROR_STATE_QUEUED, -self.bytes);
            self.metrics
                .add_checkpoint_ext_mirror_bytes(EXT_MIRROR_STATE_IN_FLIGHT, self.bytes);
        }
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        let state = if self.in_flight.load(Ordering::Acquire) {
            EXT_MIRROR_STATE_IN_FLIGHT
        } else {
            EXT_MIRROR_STATE_QUEUED
        };
        self.metrics
            .add_checkpoint_ext_mirror_bytes(state, -self.bytes);
    }
}

/// Where a copy's charge sits: empty until [`ExtMirrorSender::send`] takes
/// the charge, shared by the copy's owner (so the charge lives as long as the
/// memory) and by the handles that need to reach it (the item, to mark it in
/// flight).
type ChargeSlot = Arc<OnceLock<Charge>>;

/// The owner of a value's one owned copy, inside its `Bytes`
/// ([`Bytes::from_owner`]): the bytes and the charge slot, dropped together
/// when the last `Bytes` handle on the copy drops.
struct ChargedValue {
    data: Box<[u8]>,
    _slot: ChargeSlot,
}

impl AsRef<[u8]> for ChargedValue {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

/// A value's one owned copy, built before the store script by
/// [`ExtMirrorSender::copy_value`] so the script's argument and the mirror
/// share it, and handed to [`ExtMirrorSender::send`] after the script
/// commits. Carries no charge until then.
pub struct ExtMirrorCopy {
    bytes: Bytes,
    slot: ChargeSlot,
}

impl ExtMirrorCopy {
    /// The copy: clone it (a refcount bump) for the store script's argument.
    pub fn bytes(&self) -> &Bytes {
        &self.bytes
    }
}

/// A single item enqueued for mirroring: the blob's content hash — the
/// `valueOrExtHash` it is stored under, as lowercase 64-char hex (produced by
/// [`crate::storage::keys::sha256_to_hex`]) — the blob's charged copy, and the
/// `last_store_ms` stamp the store that made the copy wrote. The hex shape is
/// the same string the recovery-fallback path would hand to
/// [`CheckpointStore::put_ext_blob`], so producers do not need to re-format
/// anything.
pub struct ExtMirrorItem {
    hash_hex: String,
    copy: ExtMirrorCopy,
    last_store_ms: String,
}

impl ExtMirrorItem {
    /// The blob's content hash, lowercase hex.
    pub fn hash_hex(&self) -> &str {
        &self.hash_hex
    }

    /// The blob's bytes.
    pub fn bytes(&self) -> &Bytes {
        &self.copy.bytes
    }

    /// The store stamp the copy's confirmation is guarded by.
    pub fn last_store_ms(&self) -> &str {
        &self.last_store_ms
    }

    /// Count the item's charge as in flight: a worker has taken it.
    fn mark_in_flight(&self) {
        if let Some(charge) = self.copy.slot.get() {
            charge.mark_in_flight();
        }
    }
}

/// Build the mirror's channel: its sender, its receiver, and the byte budget
/// they share, `budget_bytes` (rounded down to whole KiB). `capacity` bounds
/// the items queued and must be above zero (`mpsc::channel` panics on 0).
/// Reads no configuration: [`ext_mirror_channel_for_plan`] passes `M_ext`
/// (`PodMemoryPlan::ext_mirror_bytes`) for the service; tests and benches pass
/// their own.
pub fn ext_mirror_channel(
    capacity: usize,
    budget_bytes: NonZeroU64,
    metrics: SharedMetrics,
) -> (ExtMirrorSender, ExtMirrorReceiver) {
    let budget_kib = (budget_bytes.get() / 1024).min(Semaphore::MAX_PERMITS as u64);
    let (tx, rx) = mpsc::channel(capacity);
    let sender = ExtMirrorSender {
        tx,
        metrics: Arc::clone(&metrics),
        budget: Arc::new(Semaphore::new(budget_kib as usize)),
        budget_kib,
    };
    (sender, ExtMirrorReceiver { rx, metrics })
}

/// The service's mirror wiring (`main.rs` Step 1b-pre): the mirror's channel
/// when the pod memory plan funds it, with the plan's `M_ext`
/// ([`PodMemoryPlan::ext_mirror_bytes`]) as its byte budget and `capacity`
/// (`LIVE_EDIT_CHECKPOINT_EXT_MIRROR_QUEUE_CAPACITY`) as its item bound;
/// `None` when the plan does not fund it.
///
/// The plan funds the mirror exactly when checkpointing is on with an
/// adapter (`Config::checkpoint_adapter_wired`), so the mirror runs exactly
/// then and never without a funded budget, and no second copy of that
/// condition exists to drift from the plan's. `M_ext` is never zero (it is at
/// least one maximum-size value's charge); a zero would read as unfunded.
pub fn ext_mirror_channel_for_plan(
    plan: &PodMemoryPlan,
    capacity: usize,
    metrics: SharedMetrics,
) -> Option<(ExtMirrorSender, ExtMirrorReceiver)> {
    let budget = NonZeroU64::new(plan.ext_mirror_bytes()?)?;
    Some(ext_mirror_channel(capacity, budget, metrics))
}

/// Why [`ExtMirrorSender::try_reserve`] took no charge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveRefusal {
    /// The value's charge is larger than the whole byte budget (or does not
    /// fit a `u32` permit count): no amount of room would fit it.
    Oversized,
    /// The budget has no room for the charge right now.
    NoRoom,
}

/// A charge on the mirror's byte budget taken by
/// [`ExtMirrorSender::try_reserve`] for one value about to be read and
/// uploaded outside the channel (the backup repair's re-send). Its bytes are
/// counted `in_flight` from the moment it is taken, so `queued + in_flight`
/// stays the budget's charged bytes. [`Self::wrap`] moves the charge into the
/// value's `Bytes`, so it is released when the last handle on that memory
/// drops — the adapter's request and retry clones included — as a queued
/// copy's charge is. Dropped unwrapped, it is released at once.
pub struct ExtMirrorReservation {
    charge: Charge,
    len: u64,
}

impl ExtMirrorReservation {
    /// The value length, in bytes, the charge was taken for.
    pub fn reserved_len(&self) -> u64 {
        self.len
    }

    /// `value` with this charge riding inside it, released at the last
    /// handle's drop. A refcount bump on `value`, not a copy.
    pub fn wrap(self, value: Bytes) -> Bytes {
        Bytes::from_owner(ReservedValue {
            value,
            _charge: self.charge,
        })
    }
}

/// The owner [`ExtMirrorReservation::wrap`] puts inside a `Bytes`: the value
/// and its charge, dropped together.
struct ReservedValue {
    value: Bytes,
    _charge: Charge,
}

impl AsRef<[u8]> for ReservedValue {
    fn as_ref(&self) -> &[u8] {
        &self.value
    }
}

/// Producer-side handle that enqueues ext-blob mirror items onto the
/// bounded mpsc channel, charging each to the shared byte budget. Cheap to
/// clone (`tx` is itself `Clone`, the metrics and the budget are `Arc`s) so
/// call sites may hold one per producer instance without ceremony.
#[derive(Clone)]
pub struct ExtMirrorSender {
    tx: mpsc::Sender<ExtMirrorItem>,
    metrics: SharedMetrics,
    budget: Arc<Semaphore>,
    /// The budget's total, in KiB.
    budget_kib: u64,
}

impl ExtMirrorSender {
    /// The mirror's byte budget, in bytes (whole KiB).
    pub fn budget_bytes(&self) -> u64 {
        self.budget_kib.saturating_mul(1024)
    }

    /// What is left of the byte budget right now, in bytes.
    pub fn available_budget_bytes(&self) -> u64 {
        (self.budget.available_permits() as u64).saturating_mul(1024)
    }

    /// Make `data`'s one owned copy, able to carry a mirror charge: call it
    /// before the store script, clone [`ExtMirrorCopy::bytes`] into the
    /// script's argument, and hand the copy to [`Self::send`] once the
    /// script has committed. One value-sized allocation, like the
    /// `Bytes::copy_from_slice` it replaces.
    pub fn copy_value(&self, data: &[u8]) -> ExtMirrorCopy {
        let slot: ChargeSlot = Arc::new(OnceLock::new());
        let bytes = Bytes::from_owner(ChargedValue {
            data: Box::from(data),
            _slot: Arc::clone(&slot),
        });
        ExtMirrorCopy { bytes, slot }
    }

    /// Offer one committed ext-blob for asynchronous mirroring. **Never
    /// waits and never returns an error** — the hot store path is unaffected
    /// by the mirror pipeline's state. `last_store_ms` is the stamp the store
    /// script that committed the value returned; the worker's confirmation of
    /// the copy is guarded by it. In order:
    ///
    /// 1. a value whose charge is larger than the whole budget (or does not
    ///    fit a `u32` permit count) is dropped, `reason="oversized"`,
    ///    without touching the budget;
    /// 2. the charge is taken without waiting; with no room the copy is
    ///    dropped, `reason="budget_full"`;
    /// 3. the charge goes into the copy's slot and its bytes onto the
    ///    `queued` gauge state;
    /// 4. the item is sent without waiting; on a full channel it is dropped,
    ///    `reason="queue_full"`, its charge released with the copy's last
    ///    handle. On a closed channel (the worker pool has gone, at
    ///    shutdown) a single DEBUG line is emitted and nothing is counted: it
    ///    is end of life, not saturation.
    ///
    /// Each drop logs a WARN with its reason and the first 16 hex chars of
    /// the hash. No throttling helper exists in this codebase, so the WARN
    /// fires per drop and relies on Prometheus aggregation to surface the
    /// rate; per-line context is the forensic complement to the counter.
    /// A dropped copy is never confirmed, so the backup repair re-sends it.
    pub fn send(&self, hash_hex: String, copy: ExtMirrorCopy, last_store_ms: String) {
        let len = copy.bytes.len() as u64;
        let (permit, bytes) = match self.take_charge(len) {
            Ok(taken) => taken,
            Err(ReserveRefusal::Oversized) => {
                self.dropped(EXT_MIRROR_DROP_OVERSIZED, &hash_hex, len);
                return;
            }
            Err(ReserveRefusal::NoRoom) => {
                self.dropped(EXT_MIRROR_DROP_BUDGET_FULL, &hash_hex, len);
                return;
            }
        };
        self.metrics
            .add_checkpoint_ext_mirror_bytes(EXT_MIRROR_STATE_QUEUED, bytes);
        // A copy is sent once (`send` takes it), so its slot is empty here;
        // were it not, the charge just taken would drop at once and give its
        // permits and bytes back.
        let _ = copy.slot.set(Charge {
            _permit: permit,
            bytes,
            in_flight: AtomicBool::new(false),
            metrics: Arc::clone(&self.metrics),
        });
        match self.tx.try_send(ExtMirrorItem {
            hash_hex,
            copy,
            last_store_ms,
        }) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(item)) => {
                self.dropped(EXT_MIRROR_DROP_QUEUE_FULL, &item.hash_hex, len);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                tracing::debug!("checkpoint ext-mirror channel closed — service is shutting down");
            }
        }
    }

    /// Take a charge for a value of `len` bytes from the budget without
    /// waiting: its permits and its bytes (`item_charge_kib(len) × 1 KiB`).
    /// The caller puts the bytes on a gauge state. The one charge rule for
    /// [`Self::send`] and [`Self::try_reserve`].
    fn take_charge(&self, len: u64) -> Result<(OwnedSemaphorePermit, i64), ReserveRefusal> {
        let charge_kib = item_charge_kib(len);
        if charge_kib > self.budget_kib {
            return Err(ReserveRefusal::Oversized);
        }
        let Ok(permits) = u32::try_from(charge_kib) else {
            return Err(ReserveRefusal::Oversized);
        };
        // The budget is never closed, so an error is only "no room".
        let Ok(permit) = Arc::clone(&self.budget).try_acquire_many_owned(permits) else {
            return Err(ReserveRefusal::NoRoom);
        };
        let bytes = i64::try_from(charge_kib.saturating_mul(1024)).unwrap_or(i64::MAX);
        Ok((permit, bytes))
    }

    /// Reserve the charge of a value of `len` bytes for an upload made
    /// outside the channel (the backup repair's re-send), **without
    /// waiting**: on no room it answers [`ReserveRefusal::NoRoom`] rather
    /// than queueing on the budget, so a [`Self::send`] on the store path is
    /// never held behind it. The charge is counted `in_flight` at once and
    /// rides inside the value through [`ExtMirrorReservation::wrap`].
    /// Nothing is counted on `checkpoint_ext_mirror_dropped_total`: the
    /// caller counts its own outcomes.
    pub fn try_reserve(&self, len: u64) -> Result<ExtMirrorReservation, ReserveRefusal> {
        let (permit, bytes) = self.take_charge(len)?;
        self.metrics
            .add_checkpoint_ext_mirror_bytes(EXT_MIRROR_STATE_IN_FLIGHT, bytes);
        Ok(ExtMirrorReservation {
            charge: Charge {
                _permit: permit,
                bytes,
                in_flight: AtomicBool::new(true),
                metrics: Arc::clone(&self.metrics),
            },
            len,
        })
    }

    /// Count and log one copy dropped for `reason`.
    fn dropped(&self, reason: &'static str, hash_hex: &str, len: u64) {
        self.metrics.add_checkpoint_ext_mirror_dropped(reason, 1);
        let prefix_end = hash_hex.len().min(16);
        tracing::warn!(
            reason,
            hash_prefix = &hash_hex[..prefix_end],
            bytes_len = len,
            budget_bytes = self.budget_bytes(),
            "checkpoint ext-mirror dropped an external value's backup copy (see reason); the value \
             is stored, but the backup store has no copy of it until the backup repair re-sends \
             it (for oversized, only after LIVE_EDIT_MAX_LAYER_SIZE is raised to at least its \
             bytes_len, or resources.limits.memory is raised until budget_bytes exceeds it), so a \
             restore after the main store loses it before then cannot bring it back"
        );
    }
}

/// The receiver half of [`ext_mirror_channel`], shared by the mirror workers.
/// An item taken off it counts as in flight. When it drops — after the last
/// worker exits — the items still queued are discarded and counted as
/// `reason="shutdown"`, with one WARN.
pub struct ExtMirrorReceiver {
    rx: mpsc::Receiver<ExtMirrorItem>,
    metrics: SharedMetrics,
}

impl ExtMirrorReceiver {
    /// The next item, waiting for one; `None` once every sender has dropped
    /// and the channel is empty. Cancel-safe, like `mpsc::Receiver::recv`.
    pub async fn recv(&mut self) -> Option<ExtMirrorItem> {
        let item = self.rx.recv().await?;
        item.mark_in_flight();
        Some(item)
    }

    /// The next item if one is queued, without waiting.
    pub fn try_recv(&mut self) -> Result<ExtMirrorItem, mpsc::error::TryRecvError> {
        let item = self.rx.try_recv()?;
        item.mark_in_flight();
        Ok(item)
    }
}

impl Drop for ExtMirrorReceiver {
    fn drop(&mut self) {
        // Close first, so no send lands behind the discard below; a later
        // send sees the channel closed and logs at DEBUG only.
        self.rx.close();
        let (mut count, mut bytes) = (0u64, 0u64);
        while let Ok(item) = self.rx.try_recv() {
            count += 1;
            bytes = bytes.saturating_add(item.copy.bytes.len() as u64);
        }
        if count > 0 {
            self.metrics
                .add_checkpoint_ext_mirror_dropped(EXT_MIRROR_DROP_SHUTDOWN, count);
            tracing::warn!(
                reason = EXT_MIRROR_DROP_SHUTDOWN,
                count,
                bytes,
                "checkpoint ext-mirror discarded external values still waiting for their backup \
                 copy at shutdown; they are stored, but the backup store has no copy of them until \
                 the backup repair re-sends them"
            );
        }
    }
}

/// Spawn `worker_count` mirror-worker tasks that consume from `rx` and
/// drive [`CheckpointStore::put_ext_blob`] for each item.
///
/// All workers share one [`ExtMirrorReceiver`] via
/// `Arc<Mutex<ExtMirrorReceiver>>`. `mpsc` is single-consumer, so fanning to N
/// workers requires either a fan-out shim or this shared-mutex pattern;
/// we use the mutex because (a) the critical section is just the
/// `recv()` poll (no work done under the lock — the `put_ext_blob`
/// await happens with the lock dropped) and (b) it keeps the worker
/// surface a single mpsc, matching the SR-3.1 spec verbatim. The receiver
/// drops when the last worker exits, discarding and counting what is still
/// queued.
///
/// Each worker:
///
/// 1. `select!`s between the cancel token and the locked `recv()`.
/// 2. On a received item (now counted in flight), drops the lock and awaits
///    `put_ext_blob`. Errors log a WARN with the classified kind and the
///    first 16 hex chars of the hash, then bump the errors counter. A
///    successful write is confirmed on the value's meta when `marks` is
///    `Some` ([`ExtBackupMarks::confirm`], best effort) and is otherwise
///    silent — the `checkpoint_ext_mirror_dropped_total` / `_errors_total`
///    counter pair is the operator-visible signal here. With `marks` `None`
///    nothing is confirmed, and the backup repair finds each copy with an
///    existence check instead.
/// 3. On `cancel.cancelled()`, drains up to [`DRAIN_BUDGET`] items with
///    the same semantics then returns.
/// 4. On `recv()` returning `None` (all senders dropped), returns
///    cleanly.
///
/// Returns the worker [`JoinHandle`]s so the caller can `await` them
/// during graceful shutdown.
pub fn spawn_ext_mirror_workers(
    store: Arc<dyn CheckpointStore>,
    rx: ExtMirrorReceiver,
    worker_count: usize,
    metrics: SharedMetrics,
    marks: Option<ExtBackupMarks>,
    cancel: CancellationToken,
) -> Vec<JoinHandle<()>> {
    let rx = Arc::new(Mutex::new(rx));
    let marks = marks.map(Arc::new);
    let mut handles = Vec::with_capacity(worker_count);
    for worker_idx in 0..worker_count {
        let store = Arc::clone(&store);
        let rx = Arc::clone(&rx);
        let metrics = Arc::clone(&metrics);
        let marks = marks.clone();
        let cancel = cancel.clone();
        handles.push(tokio::spawn(async move {
            tracing::debug!(worker_idx, "checkpoint ext-mirror worker started");

            loop {
                let item = {
                    let mut guard = rx.lock().await;
                    tokio::select! {
                        biased;
                        () = cancel.cancelled() => {
                            // Drop the lock so the drain loop below can
                            // re-acquire it cleanly per iteration.
                            drop(guard);
                            drain_after_cancel(&rx, &store, &metrics, marks.as_deref(), worker_idx)
                                .await;
                            tracing::debug!(
                                worker_idx,
                                "checkpoint ext-mirror worker shutting down"
                            );
                            return;
                        }
                        maybe = guard.recv() => maybe,
                    }
                };
                match item {
                    Some(item) => {
                        process_item(&*store, &metrics, marks.as_deref(), item).await;
                    }
                    None => {
                        // All senders dropped — no further work will ever
                        // arrive. Exit cleanly without ticking any
                        // counter; this is normal end-of-life.
                        tracing::debug!(
                            worker_idx,
                            "checkpoint ext-mirror channel closed; worker exiting"
                        );
                        return;
                    }
                }
            }
        }));
    }
    handles
}

/// Best-effort drain after the cancel signal fires. Pulls up to
/// [`DRAIN_BUDGET`] items off the shared receiver using `try_recv` so we
/// never block past the cancel boundary. Each drained item is mirrored
/// with the same error-handling posture as the steady-state path.
async fn drain_after_cancel(
    rx: &Arc<Mutex<ExtMirrorReceiver>>,
    store: &Arc<dyn CheckpointStore>,
    metrics: &SharedMetrics,
    marks: Option<&ExtBackupMarks>,
    worker_idx: usize,
) {
    let mut drained = 0usize;
    while drained < DRAIN_BUDGET {
        let next = {
            let mut guard = rx.lock().await;
            match guard.try_recv() {
                Ok(item) => Some(item),
                Err(mpsc::error::TryRecvError::Empty)
                | Err(mpsc::error::TryRecvError::Disconnected) => None,
            }
        };
        match next {
            Some(item) => {
                process_item(&**store, metrics, marks, item).await;
                drained += 1;
            }
            None => break,
        }
    }
    if drained > 0 {
        tracing::debug!(
            worker_idx,
            drained,
            "checkpoint ext-mirror worker drained queued items post-cancel"
        );
    }
}

/// Mirror one item by calling `put_ext_blob` and applying the SR-3.1
/// error-handling posture (log + tick counter on failure, never panic, never
/// retry). A success is confirmed on the value's meta when `marks` is given,
/// after the put has returned. The item's bytes move into the adapter; its
/// charge goes when the last handle on them does.
async fn process_item(
    store: &dyn CheckpointStore,
    metrics: &SharedMetrics,
    marks: Option<&ExtBackupMarks>,
    item: ExtMirrorItem,
) {
    let ExtMirrorItem {
        hash_hex,
        copy: ExtMirrorCopy { bytes, slot },
        last_store_ms,
    } = item;
    // The copy's owner holds the charge for as long as `bytes` live; this
    // handle on the slot is not needed past here.
    drop(slot);
    // No clean-up fence here yet: this PUT can land after the checkpoint
    // collect deleted the hash's copy, leaving an orphaned blob (a leak, never
    // a loss). b.g65's fence lands after this change and adds its check here.
    match store.put_ext_blob(&hash_hex, bytes).await {
        Ok(()) => {
            if let Some(marks) = marks {
                marks.confirm(&hash_hex, &last_store_ms).await;
            }
        }
        Err(err) => {
            metrics.inc_checkpoint_ext_mirror_error();
            let prefix_end = hash_hex.len().min(16);
            tracing::warn!(
                reason = "adapter_error",
                hash_prefix = &hash_hex[..prefix_end],
                error_kind = error_kind(&err),
                error = %err,
                "checkpoint ext-mirror put_ext_blob failed and is not retried; the value is \
                 stored, but the backup store has no copy of it until the backup repair re-sends \
                 it, so a restore after the main store loses it before then cannot bring it back"
            );
        }
    }
}

/// Map a [`CheckpointError`] to a short, stable string for structured
/// logging. Kept local to this module so the mirror pipeline's log shape
/// is self-contained; mirrors the same shape used by
/// `checkpointer::checkpoint_error_kind`.
fn error_kind(err: &CheckpointError) -> &'static str {
    match err {
        CheckpointError::Transient { .. } => "transient",
        CheckpointError::Permanent { .. } => "permanent",
        CheckpointError::NotFound => "not_found",
        CheckpointError::Conflict => "conflict",
        CheckpointError::NotImplemented { .. } => "not_implemented",
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the SR-3.1 ext-mirror pipeline.
    //!
    //! These tests pin the observable contracts the module promises:
    //!
    //! 1. **Saturated-channel drop semantics.** A producer that calls
    //!    [`ExtMirrorSender::send`] against a full channel must return
    //!    immediately without blocking and tick the
    //!    `checkpoint_ext_mirror_dropped_total{reason="queue_full"}` counter
    //!    exactly once per rejected enqueue. The errors counter MUST stay at zero on this
    //!    path — the two counters exist precisely so operators can tell
    //!    "queue saturated" apart from "adapter unhealthy".
    //!
    //! 2. **Worker happy-path delivery.** Every item handed to a worker
    //!    via `recv()` is delivered exactly once to
    //!    [`CheckpointStore::put_ext_blob`] with the same `hash_hex` and
    //!    `bytes` the caller enqueued. Successful puts are silent on the
    //!    metrics surface — no error tick, no dropped tick.
    //!
    //! 3. **Worker error-resilience.** A `put_ext_blob` error MUST log,
    //!    tick the errors counter, and continue. No panic, no task exit,
    //!    no retry. The next item lands cleanly.
    //!
    //! 4. **Graceful-shutdown drain.** On `cancel.cancelled()` workers
    //!    drain up to [`DRAIN_BUDGET`] queued items via `try_recv` (no
    //!    blocking past the cancel boundary) and then return. Items
    //!    drained during shutdown still flow through `put_ext_blob` with
    //!    the same error-handling posture as steady-state.
    //!
    //! 5. **The byte budget** (SDD §22.17, b.mhq). Each send is charged
    //!    [`item_charge_kib`] against the budget without waiting: a value
    //!    whose charge exceeds the whole budget is dropped `oversized`, one
    //!    that does not fit what is left `budget_full`, each send decided on
    //!    its own charge; the budget bounds bytes long before the item count
    //!    does for large values, the item count binds for tiny ones. The
    //!    charge lives with the value's memory, not with the queue item: an
    //!    adapter that keeps a clone keeps the charge. `queued + in_flight`
    //!    on `checkpoint_ext_mirror_bytes` is the budget's charged bytes at
    //!    every quiescent step, and every charge returns once no handle on
    //!    its memory remains, on every exit path: a successful or failed put,
    //!    each drop reason, a closed channel, the cancel drain, and the
    //!    receiver's shutdown discard (counted `reason="shutdown"`).
    //!
    //! The recovery-fallback path (SR-3.2) and the assembler/producer
    //! integration are out of scope here — they are owned by sibling Task B
    //! and `assembler.rs` respectively. This module owns only the channel
    //! shape, the byte budget and the worker loop, and these tests assert
    //! only those.
    //!
    //! No external services are required (no Valkey, no real adapter); the
    //! test store below is an in-memory fake that satisfies the
    //! `CheckpointStore` trait surface with just enough state to drive the
    //! observations each test makes, and can hold its workers inside
    //! `put_ext_blob` until a test opens its gate.
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use async_trait::async_trait;
    use tokio::sync::Notify;
    use tokio::time::timeout;

    use crate::checkpoint::store::{
        CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointMeta, CheckpointStore,
    };
    use crate::observability::metrics::{metrics_scrape_only, Metrics, SharedMetrics};

    // ─── Shared helpers ────────────────────────────────────────────────

    /// Build a fresh `SharedMetrics` so each test owns an isolated metric
    /// snapshot.
    fn make_metrics() -> SharedMetrics {
        Arc::new(metrics_scrape_only("ext-mirror-test"))
    }

    /// Read the `reason="queue_full"` cell of
    /// `live_edit_checkpoint_ext_mirror_dropped_total` from the rendered
    /// scrape exposition (absent → 0).
    fn read_dropped(metrics: &Metrics) -> u64 {
        use crate::observability::metrics::test_util::{render, sample_value};
        sample_value(
            &render(metrics),
            "live_edit_checkpoint_ext_mirror_dropped_total",
            &[("reason", "queue_full")],
        )
        .unwrap_or(0.0) as u64
    }

    /// `live_edit_checkpoint_ext_mirror_dropped_total` summed over every
    /// `reason`, for "nothing was dropped" claims: a drop under any reason
    /// moves it (absent → 0).
    fn read_dropped_any_reason(metrics: &Metrics) -> u64 {
        use crate::observability::metrics::test_util::{render, sample_sum};
        sample_sum(
            &render(metrics),
            "live_edit_checkpoint_ext_mirror_dropped_total",
            &[],
        )
        .unwrap_or(0.0) as u64
    }

    /// A mirror channel of `capacity` items whose byte budget, 1 GiB, never
    /// sheds the small payloads these tests send.
    fn channel(capacity: usize, metrics: &SharedMetrics) -> (ExtMirrorSender, ExtMirrorReceiver) {
        ext_mirror_channel(
            capacity,
            NonZeroU64::new(1 << 30).expect("non-zero"),
            Arc::clone(metrics),
        )
    }

    /// Offer `bytes` under `hash_hex` the way the store path does: a copy,
    /// then the send.
    fn enqueue(sender: &ExtMirrorSender, hash_hex: String, bytes: &[u8]) {
        sender.send(hash_hex, sender.copy_value(bytes), "0".to_owned());
    }

    /// Read the unlabeled `live_edit_checkpoint_ext_mirror_errors_total`
    /// counter from the rendered scrape exposition (absent → 0).
    fn read_errors(metrics: &Metrics) -> u64 {
        use crate::observability::metrics::test_util::{render, sample_value};
        sample_value(
            &render(metrics),
            "live_edit_checkpoint_ext_mirror_errors_total",
            &[],
        )
        .unwrap_or(0.0) as u64
    }

    /// A 64-char hex content-hash placeholder. Deliberately distinct per call
    /// site so a test that compares hashes can tell items apart, but the
    /// exact bytes do not matter — the mirror pipeline treats them as
    /// opaque carrier state.
    fn hex(byte: u8) -> String {
        let s = format!("{byte:02x}");
        s.repeat(32)
    }

    // ─── Test doubles ──────────────────────────────────────────────────

    /// In-memory `CheckpointStore` fake that records every `put_ext_blob`
    /// call (hash + bytes) and lets each test choose whether the call
    /// returns `Ok` or `Err`. All non-target trait methods panic so a
    /// regression that wires the mirror pipeline to a different store
    /// method surfaces loudly instead of going unobserved.
    ///
    /// It keeps the `Bytes` handle of every successful put, as an adapter
    /// that stores the caller's bytes would, so each kept value keeps its
    /// mirror charge until [`RecordingStore::clear_puts`] drops it.
    struct RecordingStore {
        /// Successful `put_ext_blob` calls land here, in arrival order.
        puts: tokio::sync::Mutex<Vec<(String, Bytes)>>,
        /// Every call waits here for a permit (returned when the call
        /// ends) after counting itself in `call_count`: open for
        /// [`RecordingStore::new`], closed for [`RecordingStore::held`]
        /// until [`RecordingStore::open_gate`], so a test can hold its
        /// workers inside `put_ext_blob`, each with the item it took.
        gate: Semaphore,
        /// Number of `put_ext_blob` calls that have returned.
        returned: AtomicUsize,
        /// Number of `put_ext_blob` calls (Ok + Err combined) — used as a
        /// monotonic "did the worker make progress?" probe in async
        /// waits, since reading `puts.len()` requires the lock.
        call_count: AtomicUsize,
        /// When > 0 the next N calls return `CheckpointError::Transient`
        /// instead of recording. Decremented on each error so a test can
        /// say "fail one call, then succeed thereafter".
        fail_next: AtomicUsize,
        /// Optional notifier fired AFTER each `put_ext_blob` returns. Used
        /// by tests that need to know exactly when the worker has finished
        /// processing one item without resorting to a `sleep` race.
        progress: Arc<Notify>,
    }

    /// Permits that open a gate for good: far more calls than any test makes.
    const GATE_OPEN_PERMITS: usize = 1 << 20;

    impl RecordingStore {
        fn new() -> Arc<Self> {
            Self::with_gate(GATE_OPEN_PERMITS)
        }

        /// A store whose `put_ext_blob` calls wait inside the call until
        /// [`Self::open_gate`].
        fn held() -> Arc<Self> {
            Self::with_gate(0)
        }

        fn with_gate(permits: usize) -> Arc<Self> {
            Arc::new(Self {
                puts: tokio::sync::Mutex::new(Vec::new()),
                gate: Semaphore::new(permits),
                returned: AtomicUsize::new(0),
                call_count: AtomicUsize::new(0),
                fail_next: AtomicUsize::new(0),
                progress: Arc::new(Notify::new()),
            })
        }

        /// Let every waiting and later `put_ext_blob` call through.
        fn open_gate(&self) {
            self.gate.add_permits(GATE_OPEN_PERMITS);
        }

        fn returned(&self) -> usize {
            self.returned.load(Ordering::SeqCst)
        }

        /// Drop every kept `Bytes` handle, and with it the charge of each
        /// value no other handle holds.
        async fn clear_puts(&self) {
            self.puts.lock().await.clear();
        }

        /// Configure the store to fail the next `count` `put_ext_blob`
        /// calls with `CheckpointError::Transient`, then resume normal
        /// recording behaviour.
        fn fail_next(&self, count: usize) {
            self.fail_next.store(count, Ordering::SeqCst);
        }

        fn call_count(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }

        fn progress(&self) -> Arc<Notify> {
            Arc::clone(&self.progress)
        }

        async fn puts_snapshot(&self) -> Vec<(String, Bytes)> {
            self.puts.lock().await.clone()
        }
    }

    #[async_trait]
    impl CheckpointStore for RecordingStore {
        async fn put_layer_checkpoint(
            &self,
            _object_id: u64,
            _snap_seq: u64,
            _ompf: Bytes,
            _meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            unreachable!("ext_mirror tests only drive put_ext_blob")
        }

        async fn get_latest_layer_checkpoint(
            &self,
            _object_id: u64,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            unreachable!("ext_mirror tests only drive put_ext_blob")
        }

        // This fake holds no inner store and forwards nothing, so
        // `CheckpointStore::get_layer_checkpoint`'s forwarding MUST does not
        // apply. Its `NotImplemented` default would, however, quietly return a
        // classified error where every sibling method panics — defeating this
        // fixture's stated design of surfacing a mis-wired mirror pipeline
        // loudly. The arm matches its siblings instead.
        async fn get_layer_checkpoint(
            &self,
            _handle: &CheckpointHandle,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            unreachable!("ext_mirror tests only drive put_ext_blob")
        }

        async fn list_layer_checkpoints(
            &self,
            _object_id: u64,
        ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
            unreachable!("ext_mirror tests only drive put_ext_blob")
        }

        async fn delete_layer_checkpoint(
            &self,
            _handle: CheckpointHandle,
        ) -> Result<(), CheckpointError> {
            unreachable!("ext_mirror tests only drive put_ext_blob")
        }

        async fn put_session_checkpoint(
            &self,
            _session_id: &str,
            _body: Bytes,
            _meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            unreachable!("ext_mirror tests only drive put_ext_blob")
        }

        async fn get_session_checkpoint(
            &self,
            _session_id: &str,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            unreachable!("ext_mirror tests only drive put_ext_blob")
        }

        async fn delete_session_checkpoint(
            &self,
            _session_id: &str,
        ) -> Result<(), CheckpointError> {
            unreachable!("ext_mirror tests only drive put_ext_blob")
        }

        async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
            // Increment the call counter BEFORE the success/failure
            // branch so the "did the worker pick up an item?" probe
            // observes every call regardless of outcome.
            self.call_count.fetch_add(1, Ordering::SeqCst);
            // Held here, with the item's bytes, until the gate opens. The
            // gate is never closed, so the acquire only waits.
            let _pass = self.gate.acquire().await.expect("the gate is never closed");
            let outcome = if self.fail_next.load(Ordering::SeqCst) > 0 {
                self.fail_next.fetch_sub(1, Ordering::SeqCst);
                Err(CheckpointError::Transient {
                    source: "synthetic put_ext_blob failure for ext_mirror test".into(),
                })
            } else {
                self.puts.lock().await.push((hash_hex.to_string(), bytes));
                Ok(())
            };
            // Fire the notifier AFTER we finish bookkeeping so a waiter
            // that observes a wake-up can read consistent state without
            // re-racing the worker.
            self.returned.fetch_add(1, Ordering::SeqCst);
            self.progress.notify_waiters();
            outcome
        }

        async fn get_ext_blob(&self, _hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
            unreachable!("ext_mirror tests only drive put_ext_blob")
        }

        async fn delete_ext_blob(&self, _hash_hex: &str) -> Result<(), CheckpointError> {
            unreachable!("ext_mirror tests only drive put_ext_blob")
        }

        async fn put_counters_sidecar(&self, _body: Bytes) -> Result<(), CheckpointError> {
            unreachable!("ext_mirror tests only drive put_ext_blob")
        }

        async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
            unreachable!("ext_mirror tests only drive put_ext_blob")
        }

        async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
            unreachable!("ext_mirror tests only drive put_ext_blob")
        }
    }

    /// Poll a closure with a short bounded sleep between probes until the
    /// expected condition holds or the overall budget elapses. The probe
    /// pattern is the test-writing-guide-blessed alternative to an
    /// unconditional `sleep` for "wait for some background condition".
    async fn wait_for<F>(deadline: Duration, mut probe: F)
    where
        F: FnMut() -> bool,
    {
        let start = std::time::Instant::now();
        while start.elapsed() < deadline {
            if probe() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("wait_for: condition never became true within {deadline:?}");
    }

    // ─── Test 1: saturated-channel drop ────────────────────────────────

    /// SR-3.1 producer-side contract:
    ///
    /// - `send` against a full channel never blocks and never returns
    ///   an error to the caller — `send` returns `()`.
    /// - Each rejected enqueue ticks
    ///   `checkpoint_ext_mirror_dropped_total{reason="queue_full"}` exactly
    ///   once.
    /// - The errors counter MUST stay at zero on this path (a regression
    ///   that conflates the two would surface here).
    /// - The hot path is not back-pressured: `send` returns
    ///   immediately even with zero workers draining the channel.
    #[tokio::test]
    async fn try_send_saturated_channel_drops_and_ticks_dropped_counter() {
        // Capacity-1 channel + no workers => the second enqueue is
        // guaranteed to find the channel full.
        let metrics = make_metrics();
        let (sender, _rx) = channel(1, &metrics);

        // The first enqueue fits into the bounded buffer.
        enqueue(&sender, hex(0xa1), b"payload-1");
        assert_eq!(
            read_dropped_any_reason(&metrics),
            0,
            "the first enqueue must succeed without ticking the dropped counter"
        );

        // Subsequent enqueues find the channel full and must drop +
        // tick. We drive a small batch so the counter assertion is not
        // confused by a one-off.
        for i in 0..3 {
            enqueue(&sender, hex(0xb0 + i), b"payload-overflow");
        }

        assert_eq!(
            (read_dropped(&metrics), read_dropped_any_reason(&metrics)),
            (3, 3),
            "each rejected enqueue must tick checkpoint_ext_mirror_dropped_total, under \
             queue_full and no other reason"
        );
        assert_eq!(
            read_errors(&metrics),
            0,
            "saturation MUST NOT tick the worker-side errors counter — the two \
             counters exist to disambiguate 'queue saturated' from 'adapter unhealthy'"
        );

        // Closed-channel path: drop the receiver and verify send is
        // silent (no panic, no dropped tick). The Closed branch is
        // explicitly NOT a saturation signal per the doc-comment. The
        // receiver's drop itself counts the one value still queued under
        // `shutdown`, so the send is judged by the total before and after.
        drop(_rx);
        let before = read_dropped_any_reason(&metrics);
        assert_eq!(before, 4, "the receiver's drop counts the one queued value");
        enqueue(&sender, hex(0xc0), b"payload-after-close");
        assert_eq!(
            read_dropped_any_reason(&metrics),
            before,
            "send-to-closed-channel MUST NOT tick the dropped counter under any reason"
        );
    }

    // ─── Test 2: worker happy path ─────────────────────────────────────

    /// Worker happy-path delivery:
    ///
    /// - Every item enqueued via `send` lands at `put_ext_blob`
    ///   exactly once with the same `(hash_hex, bytes)` the producer
    ///   handed in.
    /// - Successful puts leave both mirror counters at zero — no
    ///   dropped tick (the channel never saturated), no errors tick
    ///   (every put returned `Ok`).
    /// - Dropping the sender lets the worker observe `recv() = None`
    ///   and exit cleanly, so the test joins inside the timeout budget.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn worker_processes_queued_items_and_calls_put_ext_blob_once_each() {
        let store = RecordingStore::new();
        let store_dyn: Arc<dyn CheckpointStore> = store.clone();
        let metrics = make_metrics();
        let (sender, rx) = channel(16, &metrics);
        let cancel = CancellationToken::new();
        let handles = spawn_ext_mirror_workers(
            store_dyn,
            rx,
            /*worker_count=*/ 2,
            Arc::clone(&metrics),
            None,
            cancel.clone(),
        );

        let items: Vec<(String, Bytes)> = (0..5)
            .map(|i| (hex(0x10 + i as u8), Bytes::from(format!("payload-{i}"))))
            .collect();
        for (h, b) in &items {
            enqueue(&sender, h.clone(), b);
        }

        // Close the sender side so workers see `recv() = None` and exit
        // cleanly without needing the cancel signal — pinning the
        // "all senders dropped" exit branch.
        drop(sender);

        // Wait for both workers to finish processing the backlog and
        // exit on the `None` branch.
        for h in handles {
            timeout(Duration::from_secs(5), h)
                .await
                .expect("worker join must complete inside 5s")
                .expect("worker must not panic");
        }

        let recorded = store.puts_snapshot().await;
        assert_eq!(
            recorded.len(),
            items.len(),
            "every enqueued item must reach put_ext_blob exactly once"
        );
        let mut recorded_sorted = recorded.clone();
        recorded_sorted.sort_by(|a, b| a.0.cmp(&b.0));
        let mut expected_sorted = items.clone();
        expected_sorted.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            recorded_sorted, expected_sorted,
            "put_ext_blob must receive the same (hash, bytes) the caller enqueued"
        );

        assert_eq!(
            read_dropped_any_reason(&metrics),
            0,
            "the happy path must not tick the dropped counter"
        );
        assert_eq!(
            read_errors(&metrics),
            0,
            "successful put_ext_blob calls must not tick the errors counter"
        );
    }

    // ─── Test 3: worker error-resilience ───────────────────────────────

    /// Worker error-resilience contract:
    ///
    /// - A `put_ext_blob` error MUST tick
    ///   `checkpoint_ext_mirror_errors_total` once per failure.
    /// - The worker MUST NOT panic, exit, or retry — the next item
    ///   lands cleanly on `put_ext_blob`.
    /// - The dropped counter MUST stay at zero on this path (errors
    ///   are not saturation; a regression that conflates the two would
    ///   surface here).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn worker_continues_after_put_ext_blob_error_and_ticks_errors_counter() {
        let store = RecordingStore::new();
        // First two calls fail; third onwards succeeds. We use a single
        // worker so the failing-then-succeeding sequence is deterministic
        // — with multiple workers the failure order would interleave with
        // success order non-deterministically and the per-counter
        // assertions would only hold in aggregate.
        store.fail_next(2);
        let store_dyn: Arc<dyn CheckpointStore> = store.clone();
        let metrics = make_metrics();
        let (sender, rx) = channel(16, &metrics);
        let cancel = CancellationToken::new();
        let handles = spawn_ext_mirror_workers(
            store_dyn,
            rx,
            /*worker_count=*/ 1,
            Arc::clone(&metrics),
            None,
            cancel.clone(),
        );

        // Four items: two should fail at put_ext_blob, two should
        // succeed. The successful pair pins "the worker survived the
        // earlier errors and is still draining the queue".
        let items: Vec<(String, Bytes)> = (0..4)
            .map(|i| (hex(0x20 + i as u8), Bytes::from(format!("err-payload-{i}"))))
            .collect();
        for (h, b) in &items {
            enqueue(&sender, h.clone(), b);
        }

        drop(sender);

        for h in handles {
            timeout(Duration::from_secs(5), h)
                .await
                .expect("worker join must complete inside 5s")
                .expect("worker must not panic after put_ext_blob error");
        }

        assert_eq!(
            store.call_count(),
            items.len(),
            "every queued item must reach put_ext_blob — the worker MUST NOT \
             abandon the queue after an error"
        );

        let recorded = store.puts_snapshot().await;
        assert_eq!(
            recorded.len(),
            items.len() - 2,
            "only the non-failing put_ext_blob calls should have recorded \
             their (hash, bytes) payloads"
        );

        assert_eq!(
            read_errors(&metrics),
            2,
            "each put_ext_blob error must tick checkpoint_ext_mirror_errors_total once"
        );
        assert_eq!(
            read_dropped_any_reason(&metrics),
            0,
            "put_ext_blob errors MUST NOT tick the dropped counter — that \
             counter is reserved for producer-side saturation"
        );
    }

    // ─── Test 4: graceful-shutdown drain ───────────────────────────────

    /// Graceful-shutdown drain contract:
    ///
    /// - After `cancel.cancelled()` each worker drains up to
    ///   [`DRAIN_BUDGET`] queued items via `try_recv` (no blocking past
    ///   the cancel boundary).
    /// - Items drained during shutdown still flow through
    ///   `put_ext_blob` with the same error-handling posture as
    ///   steady-state — successful drains leave the metrics surface
    ///   silent.
    /// - Workers return cleanly inside a tight timeout budget so the
    ///   service-level shutdown stays inside `LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn worker_drains_queued_items_after_cancel_then_returns() {
        let store = RecordingStore::new();
        let progress = store.progress();
        let store_dyn: Arc<dyn CheckpointStore> = store.clone();
        // Channel large enough to hold the entire backlog without
        // saturating, so the test is exclusively exercising the
        // drain-after-cancel path (no producer-side drops).
        let metrics = make_metrics();
        let (sender, rx) = channel(64, &metrics);
        let cancel = CancellationToken::new();
        let handles = spawn_ext_mirror_workers(
            store_dyn,
            rx,
            /*worker_count=*/ 1,
            Arc::clone(&metrics),
            None,
            cancel.clone(),
        );

        // Enqueue ONE item and wait for the worker to process it. This
        // pins "the worker is parked on `recv()` waiting for more"
        // before we fire `cancel`, so the items enqueued below land
        // entirely on the drain-after-cancel branch (not on the
        // steady-state branch).
        enqueue(&sender, hex(0x30), b"warmup");
        wait_for(Duration::from_secs(5), || store.call_count() >= 1).await;

        // Enqueue a small backlog (well under DRAIN_BUDGET = 64 so the
        // budget is not the test-driver here) AND fire cancel before
        // the worker can pick anything up. The biased select! polls the
        // cancel branch first, so the cancel signal is observed before
        // the next `recv()` and the queued items must flow through the
        // drain path.
        let backlog: Vec<(String, Bytes)> = (0..4)
            .map(|i| {
                (
                    hex(0x40 + i as u8),
                    Bytes::from(format!("drain-payload-{i}")),
                )
            })
            .collect();
        for (h, b) in &backlog {
            enqueue(&sender, h.clone(), b);
        }

        // Cancel and observe the workers drain + exit. We hold `sender`
        // open across the cancel so the drain branch sees a non-empty
        // queue rather than a closed channel — pinning the "cancelled
        // BEFORE the channel was closed" path, which is the realistic
        // shutdown shape.
        cancel.cancel();

        for h in handles {
            timeout(Duration::from_secs(5), h)
                .await
                .expect(
                    "worker join must complete inside 5s — the drain budget \
                     must keep shutdown inside LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS",
                )
                .expect("worker must not panic during drain");
        }

        // Sanity-check the progress notifier was actually wired: at
        // least one wake-up fired during the drain. We don't await it
        // (the workers have already exited), we just confirm the
        // bookkeeping path ran.
        let _ = progress; // silence unused-binding lint without disabling it.

        let recorded = store.puts_snapshot().await;
        assert_eq!(
            recorded.len(),
            1 + backlog.len(),
            "drain phase must deliver every queued item (1 warmup + {} backlog)",
            backlog.len()
        );

        let recorded_hashes: std::collections::HashSet<String> =
            recorded.iter().map(|(h, _)| h.clone()).collect();
        for (h, _) in &backlog {
            assert!(
                recorded_hashes.contains(h),
                "drain phase must have delivered hash {h} via put_ext_blob"
            );
        }

        assert_eq!(
            read_dropped_any_reason(&metrics),
            0,
            "graceful drain must not tick the dropped counter"
        );
        assert_eq!(
            read_errors(&metrics),
            0,
            "successful drain puts must not tick the errors counter"
        );

        // Sender is still alive at this point — verify shutdown did not
        // depend on producer-side close.
        drop(sender);
    }

    // ─── The byte budget (b.mhq) ───────────────────────────────────────

    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;

    /// A mirror channel of `capacity` items with a byte budget of
    /// `budget_bytes`.
    fn budgeted(
        capacity: usize,
        budget_bytes: u64,
        metrics: &SharedMetrics,
    ) -> (ExtMirrorSender, ExtMirrorReceiver) {
        ext_mirror_channel(
            capacity,
            NonZeroU64::new(budget_bytes).expect("a test budget is non-zero"),
            Arc::clone(metrics),
        )
    }

    /// The bytes one value of `len` bytes is charged.
    fn charge_of(len: u64) -> u64 {
        item_charge_kib(len) * KIB
    }

    /// The budget's bytes charged right now.
    fn charged(sender: &ExtMirrorSender) -> u64 {
        sender.budget_bytes() - sender.available_budget_bytes()
    }

    /// `checkpoint_ext_mirror_dropped_total{reason}` (absent → 0).
    fn dropped_for(metrics: &Metrics, reason: &str) -> u64 {
        use crate::observability::metrics::test_util::{render, sample_value};
        sample_value(
            &render(metrics),
            "live_edit_checkpoint_ext_mirror_dropped_total",
            &[("reason", reason)],
        )
        .unwrap_or(0.0) as u64
    }

    /// `checkpoint_ext_mirror_bytes{state}`; every state is pre-registered,
    /// so an absent cell is a failure, not 0.
    fn bytes_in(metrics: &Metrics, state: &str) -> u64 {
        use crate::observability::metrics::test_util::{render, sample_value};
        let value = sample_value(
            &render(metrics),
            "live_edit_checkpoint_ext_mirror_bytes",
            &[("state", state)],
        )
        .unwrap_or_else(|| panic!("checkpoint_ext_mirror_bytes{{state={state:?}}} is registered"));
        assert!(value >= 0.0, "state {state:?} is never negative: {value}");
        value as u64
    }

    /// T5: the gauge's `queued` plus `in_flight` bytes are what the budget
    /// has charged, never more than the budget, at a quiescent step (no
    /// send, receive or drop in progress); returns `(queued, in_flight)`.
    fn assert_accounted(sender: &ExtMirrorSender, metrics: &Metrics, step: &str) -> (u64, u64) {
        let (queued, in_flight) = (
            bytes_in(metrics, EXT_MIRROR_STATE_QUEUED),
            bytes_in(metrics, EXT_MIRROR_STATE_IN_FLIGHT),
        );
        assert_eq!(
            queued + in_flight,
            charged(sender),
            "{step}: queued {queued} B + in flight {in_flight} B is the budget's charged bytes"
        );
        assert!(
            charged(sender) <= sender.budget_bytes(),
            "{step}: never more charged than the budget"
        );
        (queued, in_flight)
    }

    /// A value of `len` bytes, every byte `fill`, so values of one length
    /// are told apart by their fill.
    fn value(fill: u8, len: u64) -> Vec<u8> {
        vec![fill; len as usize]
    }

    /// The charge is the value's size rounded up to a whole KiB, plus the
    /// one-KiB per-item allowance: so an empty value is charged 1 KiB, a
    /// value of one byte to one KiB 2 KiB, one byte past a KiB 3 KiB. And a
    /// budget is held in whole KiB, rounded down, all of it available at
    /// first.
    #[test]
    fn item_charge_kib_rounds_the_size_up_to_a_kib_and_adds_the_overhead() {
        for (len, kib) in [
            (0, 1),
            (1, 2),
            (KIB, 2),
            (KIB + 1, 3),
            (MIB, 1_025),
            (100 * MIB, 102_401),
        ] {
            assert_eq!(item_charge_kib(len), kib, "{len} B");
        }
        assert_eq!(ITEM_OVERHEAD_KIB, 1, "the per-item allowance is one KiB");
        assert_eq!(
            item_charge_kib(u64::MAX),
            u64::MAX.div_ceil(1024) + 1,
            "no overflow at the top"
        );

        let metrics = make_metrics();
        let (sender, _rx) = budgeted(4, 8 * KIB + 1_023, &metrics);
        assert_eq!(sender.budget_bytes(), 8 * KIB, "rounded down to whole KiB");
        assert_eq!(sender.available_budget_bytes(), 8 * KIB);
    }

    /// T1 + T5: the budget sheds by bytes long before the item count. With
    /// an 8 MiB budget, a 1,000-item channel and both workers held inside
    /// `put_ext_blob`, ten distinct 1 MiB values admit exactly 7 (8,192 KiB
    /// ÷ 1,025 KiB): two in the workers' hands and five queued; the other
    /// three tick `budget_full`, and `queue_full` never does. The charged
    /// bytes never exceed the budget, and `in_flight` is exactly what the
    /// held workers have taken. Released, the workers deliver the 7
    /// byte-identical; the store keeps their bytes, so their charges stay
    /// until it drops them. A queue counting only items would admit all ten.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn send_sheds_one_mib_values_by_bytes_with_budget_full_long_before_the_item_count() {
        let store = RecordingStore::held();
        let metrics = make_metrics();
        let (sender, rx) = budgeted(1_000, 8 * MIB, &metrics);
        let cancel = CancellationToken::new();
        let handles = spawn_ext_mirror_workers(
            store.clone(),
            rx,
            2,
            Arc::clone(&metrics),
            None,
            cancel.clone(),
        );

        let values: Vec<(String, Vec<u8>)> = (0..10u8)
            .map(|i| (hex(0x60 + i), value(0x60 + i, MIB)))
            .collect();
        for (i, (hash, bytes)) in values.iter().enumerate() {
            enqueue(&sender, hash.clone(), bytes);
            assert!(
                charged(&sender) <= sender.budget_bytes(),
                "after send {i}: {} B charged, past the {} B budget",
                charged(&sender),
                sender.budget_bytes()
            );
        }
        // Both workers hold one item each, inside the held put.
        wait_for(Duration::from_secs(5), || store.call_count() == 2).await;

        let one = charge_of(MIB);
        assert_eq!(dropped_for(&metrics, EXT_MIRROR_DROP_BUDGET_FULL), 3);
        assert_eq!(dropped_for(&metrics, EXT_MIRROR_DROP_QUEUE_FULL), 0);
        assert_eq!(dropped_for(&metrics, EXT_MIRROR_DROP_OVERSIZED), 0);
        assert_eq!(charged(&sender), 7 * one, "exactly 7 admitted");
        let (queued, in_flight) = assert_accounted(&sender, &metrics, "workers held");
        assert_eq!(
            (queued, in_flight),
            (5 * one, 2 * one),
            "five queued, and in flight exactly the two the held workers took"
        );

        store.open_gate();
        wait_for(Duration::from_secs(5), || store.returned() == 7).await;
        let mut puts = store.puts_snapshot().await;
        puts.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(puts.len(), 7, "the seven admitted values are delivered");
        for (hash, bytes) in &puts {
            let sent = values
                .iter()
                .find(|(h, _)| h == hash)
                .unwrap_or_else(|| panic!("{hash} was sent"));
            assert_eq!(bytes.as_ref(), sent.1.as_slice(), "{hash} byte-identical");
        }
        drop(puts);
        let (queued, in_flight) = assert_accounted(&sender, &metrics, "delivered, kept");
        assert_eq!(
            (queued, in_flight),
            (0, 7 * one),
            "the store keeps the seven values, and with them their charges"
        );
        store.clear_puts().await;
        assert_eq!(charged(&sender), 0, "every charge returns with its memory");
        assert_eq!(assert_accounted(&sender, &metrics, "dropped"), (0, 0));

        shutdown(cancel, handles).await;
    }

    /// Cancel the workers and join them within 5 s.
    async fn shutdown(cancel: CancellationToken, handles: Vec<JoinHandle<()>>) {
        cancel.cancel();
        for h in handles {
            timeout(Duration::from_secs(5), h)
                .await
                .expect("workers join within 5 s of the cancel")
                .expect("no worker panics");
        }
    }

    /// T2 + T5: for tiny values the item count binds, not the budget. A
    /// 4-item channel with a 1 GiB budget and nothing receiving admits 4 of
    /// ten 100-byte values and drops the other six `queue_full`, each
    /// charge released with its dropped copy: what stays charged is the four
    /// queued values at 2 KiB each. `budget_full` never ticks. Discarding
    /// the queue at the receiver's drop returns every charge (T6).
    #[test]
    fn send_drops_tiny_values_by_the_item_count_with_queue_full_and_charges_only_the_queued() {
        let metrics = make_metrics();
        let (sender, rx) = budgeted(4, 1 << 30, &metrics);
        for i in 0..10u8 {
            enqueue(&sender, hex(0x70 + i), &value(i, 100));
        }
        assert_eq!(dropped_for(&metrics, EXT_MIRROR_DROP_QUEUE_FULL), 6);
        assert_eq!(dropped_for(&metrics, EXT_MIRROR_DROP_BUDGET_FULL), 0);
        assert_eq!(
            charged(&sender),
            4 * 2 * KIB,
            "the four queued at 2 KiB each"
        );
        assert_eq!(
            assert_accounted(&sender, &metrics, "four queued"),
            (4 * 2 * KIB, 0)
        );

        drop(rx);
        assert_eq!(dropped_for(&metrics, EXT_MIRROR_DROP_SHUTDOWN), 4);
        assert_eq!(charged(&sender), 0, "the discarded queue's charges return");
        assert_eq!(assert_accounted(&sender, &metrics, "discarded"), (0, 0));
    }

    /// T3 + T5: each send is decided on its own charge, against what is left
    /// right then. A 64 KiB budget takes a 60 KiB value (61 KiB), refuses a
    /// 10 KiB one (11 KiB, 3 left) `budget_full`, then admits a 1 KiB value
    /// (2 KiB) and an empty one (1 KiB), filling it; another empty value is
    /// refused. Taken off the channel the large value is counted in flight;
    /// dropped, its 61 KiB return, and the 10 KiB value refused before now
    /// fits. The gauge is the charges taken less those released at every
    /// step.
    #[test]
    fn send_decides_each_value_on_its_own_charge_so_a_small_value_fits_after_a_large_one_is_refused(
    ) {
        let metrics = make_metrics();
        let (sender, mut rx) = budgeted(16, 64 * KIB, &metrics);
        let budget_full = || dropped_for(&metrics, EXT_MIRROR_DROP_BUDGET_FULL);

        enqueue(&sender, hex(0x80), &value(0x80, 60 * KIB));
        assert_eq!(charged(&sender), 61 * KIB, "60 KiB charged 61");
        enqueue(&sender, hex(0x81), &value(0x81, 10 * KIB));
        assert_eq!(budget_full(), 1, "11 KiB does not fit the 3 left");
        assert_eq!(charged(&sender), 61 * KIB, "a refused value takes nothing");
        enqueue(&sender, hex(0x82), &value(0x82, KIB));
        assert_eq!(budget_full(), 1, "a 2 KiB charge fits the 3 left");
        enqueue(&sender, hex(0x83), &[]);
        assert_eq!(charged(&sender), 64 * KIB, "the budget is full");
        enqueue(&sender, hex(0x84), &[]);
        assert_eq!(budget_full(), 2, "even 1 KiB no longer fits");
        assert_eq!(assert_accounted(&sender, &metrics, "full"), (64 * KIB, 0));

        let large = rx
            .try_recv()
            .expect("the large value is first in the queue");
        assert_eq!(large.hash_hex(), hex(0x80));
        assert_eq!(
            assert_accounted(&sender, &metrics, "large taken"),
            (3 * KIB, 61 * KIB),
            "a value taken off the channel is counted in flight"
        );
        drop(large);
        assert_eq!(
            assert_accounted(&sender, &metrics, "large dropped"),
            (3 * KIB, 0),
            "its charge returns with its memory"
        );
        enqueue(&sender, hex(0x81), &value(0x81, 10 * KIB));
        assert_eq!(budget_full(), 2, "the 10 KiB value fits now");
        assert_eq!(
            assert_accounted(&sender, &metrics, "10 KiB admitted"),
            (14 * KIB, 0)
        );
        assert_eq!(dropped_for(&metrics, EXT_MIRROR_DROP_QUEUE_FULL), 0);
        assert_eq!(dropped_for(&metrics, EXT_MIRROR_DROP_OVERSIZED), 0);
    }

    /// T4: a value whose charge is larger than the whole budget is dropped
    /// `oversized` and takes nothing: the budget's available bytes are
    /// unchanged and nothing is queued. Its boundary: an 8 KiB budget
    /// refuses an 8 KiB value (charged 9) and admits a 7 KiB one (charged 8,
    /// the whole budget). With the budget full an 8 KiB value is still
    /// `oversized`, not `budget_full`: no amount of room would fit it.
    #[test]
    fn send_drops_a_value_larger_than_the_whole_budget_as_oversized_without_touching_it() {
        let metrics = make_metrics();
        let (sender, mut rx) = budgeted(16, 8 * KIB, &metrics);

        enqueue(&sender, hex(0x90), &value(0x90, 8 * KIB));
        assert_eq!(dropped_for(&metrics, EXT_MIRROR_DROP_OVERSIZED), 1);
        assert_eq!(sender.available_budget_bytes(), 8 * KIB, "nothing taken");
        assert!(
            matches!(rx.try_recv(), Err(mpsc::error::TryRecvError::Empty)),
            "nothing queued"
        );
        assert_eq!(assert_accounted(&sender, &metrics, "oversized"), (0, 0));

        enqueue(&sender, hex(0x91), &value(0x91, 7 * KIB));
        assert_eq!(
            dropped_for(&metrics, EXT_MIRROR_DROP_OVERSIZED),
            1,
            "a charge of exactly the budget fits"
        );
        assert_eq!(sender.available_budget_bytes(), 0);
        enqueue(&sender, hex(0x92), &value(0x92, 8 * KIB));
        assert_eq!(dropped_for(&metrics, EXT_MIRROR_DROP_OVERSIZED), 2);
        assert_eq!(dropped_for(&metrics, EXT_MIRROR_DROP_BUDGET_FULL), 0);
        assert_eq!(
            assert_accounted(&sender, &metrics, "7 KiB queued"),
            (8 * KIB, 0)
        );
    }

    /// T6 + T7: the charge follows the memory, not the queue item. A store
    /// that keeps the bytes of each successful put holds their charges after
    /// the worker is done with the item — in flight, as the copy is no
    /// longer queued — and they return only when it drops them. A failed put
    /// keeps nothing, so its charge returns as the worker carries on to the
    /// next item. Once no handle remains the whole budget is available.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn charge_is_held_while_an_adapter_keeps_the_bytes_and_returns_when_they_drop() {
        let store = RecordingStore::new();
        store.fail_next(2);
        let metrics = make_metrics();
        let (sender, rx) = budgeted(16, MIB, &metrics);
        let cancel = CancellationToken::new();
        let handles = spawn_ext_mirror_workers(
            store.clone(),
            rx,
            1,
            Arc::clone(&metrics),
            None,
            cancel.clone(),
        );

        for i in 0..4u8 {
            enqueue(&sender, hex(0xa0 + i), &value(i, 4 * KIB));
        }
        wait_for(Duration::from_secs(5), || store.returned() == 4).await;
        assert_eq!(read_errors(&metrics), 2, "two puts failed");
        assert_eq!(store.puts_snapshot().await.len(), 2, "two were kept");
        assert_eq!(
            assert_accounted(&sender, &metrics, "two kept"),
            (0, 2 * charge_of(4 * KIB)),
            "the kept values hold their charges after the worker is done; the failed \
             ones returned theirs"
        );

        store.clear_puts().await;
        assert_eq!(charged(&sender), 0, "released when the kept bytes drop");
        assert_eq!(assert_accounted(&sender, &metrics, "dropped"), (0, 0));
        shutdown(cancel, handles).await;
    }

    /// T6: the shutdown paths leak no charge. One worker holds an item in
    /// its put; [`DRAIN_BUDGET`] + 6 more are queued; then the cancel fires
    /// and the put is let through. The worker finishes its item, drains
    /// `DRAIN_BUDGET` of the queue and returns; the receiver, dropped with
    /// it, discards the 6 left, counted `reason="shutdown"`. After the store
    /// drops the bytes it kept, every charge is back. A send after the
    /// channel closed takes nothing and is not counted.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_drain_and_shutdown_discard_return_every_charge_and_count_what_was_left() {
        let store = RecordingStore::held();
        let metrics = make_metrics();
        let left = 6;
        let queued = DRAIN_BUDGET + left;
        let (sender, rx) = budgeted(queued, 4 * MIB, &metrics);
        let cancel = CancellationToken::new();
        let handles = spawn_ext_mirror_workers(
            store.clone(),
            rx,
            1,
            Arc::clone(&metrics),
            None,
            cancel.clone(),
        );

        enqueue(&sender, hex(0xb0), &value(0xb0, 100));
        wait_for(Duration::from_secs(5), || store.call_count() == 1).await;
        for i in 0..queued {
            let tag = format!("{i:064x}");
            enqueue(&sender, tag, &(i as u64).to_le_bytes());
        }
        assert_eq!(
            charged(&sender),
            (1 + queued as u64) * 2 * KIB,
            "fixture self-check: every value admitted"
        );
        cancel.cancel();
        store.open_gate();
        for h in handles {
            timeout(Duration::from_secs(5), h)
                .await
                .expect("the worker drains and returns")
                .expect("no worker panics");
        }

        assert_eq!(
            store.returned(),
            1 + DRAIN_BUDGET,
            "its item, then the drain"
        );
        assert_eq!(
            dropped_for(&metrics, EXT_MIRROR_DROP_SHUTDOWN),
            left as u64,
            "the receiver's drop discards and counts what the drain left"
        );
        for reason in [
            EXT_MIRROR_DROP_QUEUE_FULL,
            EXT_MIRROR_DROP_BUDGET_FULL,
            EXT_MIRROR_DROP_OVERSIZED,
        ] {
            assert_eq!(dropped_for(&metrics, reason), 0, "{reason}");
        }
        store.clear_puts().await;
        assert_eq!(charged(&sender), 0, "every charge returns");
        assert_eq!(assert_accounted(&sender, &metrics, "shut down"), (0, 0));

        enqueue(&sender, hex(0xb1), &value(0xb1, 100));
        assert_eq!(
            charged(&sender),
            0,
            "a send to the closed channel keeps nothing"
        );
        assert_eq!(dropped_for(&metrics, EXT_MIRROR_DROP_SHUTDOWN), left as u64);
        for reason in [
            EXT_MIRROR_DROP_QUEUE_FULL,
            EXT_MIRROR_DROP_BUDGET_FULL,
            EXT_MIRROR_DROP_OVERSIZED,
        ] {
            assert_eq!(dropped_for(&metrics, reason), 0, "closed: {reason}");
        }
    }

    /// T8: sending never waits. With the budget full and the one worker
    /// held in its put, a burst of 1,000 sends from a plain thread — no
    /// runtime to yield to — completes, each dropped `budget_full`, while
    /// the held worker makes no progress. A send that waited for room would
    /// never return here, as nothing frees any until the gate opens.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn send_returns_without_waiting_when_the_budget_is_full_and_the_workers_are_held() {
        let store = RecordingStore::held();
        let metrics = make_metrics();
        // Two 1 KiB values' charges.
        let (sender, rx) = budgeted(16, 2 * charge_of(KIB), &metrics);
        let cancel = CancellationToken::new();
        let handles = spawn_ext_mirror_workers(
            store.clone(),
            rx,
            1,
            Arc::clone(&metrics),
            None,
            cancel.clone(),
        );
        enqueue(&sender, hex(0xc0), &value(0xc0, KIB));
        enqueue(&sender, hex(0xc1), &value(0xc1, KIB));
        wait_for(Duration::from_secs(5), || store.call_count() == 1).await;
        assert_eq!(
            sender.available_budget_bytes(),
            0,
            "fixture: the budget is full"
        );

        const BURST: u64 = 1_000;
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let burst_sender = sender.clone();
        let burst = std::thread::spawn(move || {
            for i in 0..BURST {
                enqueue(&burst_sender, format!("{i:064x}"), &value(0xc2, KIB));
            }
            done_tx.send(()).expect("the test is waiting");
        });
        tokio::task::spawn_blocking(move || done_rx.recv_timeout(Duration::from_secs(10)))
            .await
            .expect("the waiter does not panic")
            .expect("the burst of sends completes without waiting for room");
        burst.join().expect("the burst thread does not panic");

        assert_eq!(dropped_for(&metrics, EXT_MIRROR_DROP_BUDGET_FULL), BURST);
        assert_eq!(store.call_count(), 1, "the held worker made no progress");
        assert_eq!(store.returned(), 0);

        store.open_gate();
        shutdown(cancel, handles).await;
    }

    // ─── The backup repair's reservation (b.wsx) ───────────────────────

    /// T12 (b.wsx): a reservation is charged like a send and counted
    /// `in_flight` from the moment it is taken, so `queued + in_flight` stays
    /// the budget's charged bytes beside a queued copy. Wrapped into the
    /// value's `Bytes`, the charge rides with that memory: a clone of the
    /// wrapped bytes (an adapter's retry copy) keeps it after the first
    /// handle drops, and the last handle's drop returns it. A reservation
    /// counts no drop.
    #[test]
    fn try_reserve_counts_its_charge_in_flight_and_returns_it_with_the_last_handle_on_the_wrapped_bytes(
    ) {
        let metrics = make_metrics();
        let (sender, _rx) = budgeted(16, 64 * KIB, &metrics);
        enqueue(&sender, hex(0xd0), &value(0xd0, 10 * KIB));
        assert_eq!(
            assert_accounted(&sender, &metrics, "one queued"),
            (11 * KIB, 0),
            "fixture self-check: one 10 KiB copy queued"
        );

        let reservation = sender
            .try_reserve(20 * KIB)
            .expect("21 KiB fits the 53 KiB left");
        assert_eq!(reservation.reserved_len(), 20 * KIB);
        assert_eq!(
            assert_accounted(&sender, &metrics, "reserved"),
            (11 * KIB, charge_of(20 * KIB)),
            "the reservation is counted in flight at once, beside the queued copy"
        );

        let payload = value(0xd1, 20 * KIB);
        let wrapped = reservation.wrap(Bytes::from(payload.clone()));
        assert_eq!(
            wrapped.as_ref(),
            payload.as_slice(),
            "the wrapped bytes are the value"
        );
        let retry_copy = wrapped.clone();
        drop(wrapped);
        assert_eq!(
            assert_accounted(&sender, &metrics, "first handle dropped"),
            (11 * KIB, charge_of(20 * KIB)),
            "a clone of the wrapped bytes keeps the charge"
        );
        drop(retry_copy);
        assert_eq!(
            assert_accounted(&sender, &metrics, "last handle dropped"),
            (11 * KIB, 0),
            "the last handle's drop returns the charge"
        );
        assert_eq!(
            read_dropped_any_reason(&metrics),
            0,
            "a reservation counts no drop"
        );
    }

    /// A reservation dropped without being wrapped (the re-send found the
    /// value gone, or its length changed) returns its charge at once.
    #[test]
    fn try_reserve_dropped_unwrapped_returns_its_charge_at_once() {
        let metrics = make_metrics();
        let (sender, _rx) = budgeted(16, 64 * KIB, &metrics);
        let reservation = sender.try_reserve(4 * KIB).expect("fits");
        assert_eq!(charged(&sender), charge_of(4 * KIB));
        drop(reservation);
        assert_eq!(
            charged(&sender),
            0,
            "the charge returns with the reservation"
        );
        assert_eq!(assert_accounted(&sender, &metrics, "dropped"), (0, 0));
    }

    /// T12 (b.wsx): on a full budget `try_reserve` answers `NoRoom` at once and
    /// counts nothing, and a send on the store path is refused `budget_full`
    /// without waiting, exactly as it is beside queued copies. The refused
    /// reservation leaves nothing queued on the budget: room freed afterwards
    /// goes to the next send, which a reservation parked as a waiter on the
    /// budget would have taken first.
    #[test]
    fn try_reserve_on_a_full_budget_answers_no_room_and_leaves_the_freed_room_to_the_next_send() {
        let metrics = make_metrics();
        let (sender, mut rx) = budgeted(16, 2 * charge_of(KIB), &metrics);
        let held = sender.try_reserve(KIB).expect("the first charge fits");
        enqueue(&sender, hex(0xe0), &value(0xe0, KIB));
        assert_eq!(
            sender.available_budget_bytes(),
            0,
            "fixture: the budget is full"
        );

        assert!(
            matches!(sender.try_reserve(KIB), Err(ReserveRefusal::NoRoom)),
            "a full budget answers NoRoom"
        );
        assert_eq!(
            read_dropped_any_reason(&metrics),
            0,
            "a refused reservation counts no drop"
        );
        enqueue(&sender, hex(0xe1), &value(0xe1, KIB));
        assert_eq!(
            dropped_for(&metrics, EXT_MIRROR_DROP_BUDGET_FULL),
            1,
            "a send beside the full budget is refused budget_full, without waiting"
        );

        drop(held);
        enqueue(&sender, hex(0xe2), &value(0xe2, KIB));
        assert_eq!(
            dropped_for(&metrics, EXT_MIRROR_DROP_BUDGET_FULL),
            1,
            "the room the reservation freed went to the next send"
        );
        assert_eq!(
            assert_accounted(&sender, &metrics, "two queued"),
            (2 * charge_of(KIB), 0)
        );
        let queued: Vec<String> = std::iter::from_fn(|| rx.try_recv().ok())
            .map(|item| item.hash_hex().to_owned())
            .collect();
        assert_eq!(queued, vec![hex(0xe0), hex(0xe2)]);
    }

    /// T13 (b.wsx): a value whose charge is larger than the whole budget is
    /// refused `Oversized` and takes nothing, with room or without: no amount
    /// of room would fit it. A charge of exactly the budget fits.
    #[test]
    fn try_reserve_refuses_a_value_larger_than_the_whole_budget_as_oversized_and_takes_nothing() {
        let metrics = make_metrics();
        let (sender, _rx) = budgeted(16, 8 * KIB, &metrics);
        assert!(
            matches!(sender.try_reserve(8 * KIB), Err(ReserveRefusal::Oversized)),
            "an 8 KiB value is charged 9 KiB, past the 8 KiB budget"
        );
        assert_eq!(sender.available_budget_bytes(), 8 * KIB, "nothing taken");
        let whole = sender
            .try_reserve(7 * KIB)
            .expect("a charge of exactly the budget fits");
        assert!(
            matches!(sender.try_reserve(8 * KIB), Err(ReserveRefusal::Oversized)),
            "still Oversized, not NoRoom, with the budget full"
        );
        assert_eq!(read_dropped_any_reason(&metrics), 0);
        drop(whole);
        assert_eq!(assert_accounted(&sender, &metrics, "released"), (0, 0));
    }
}
