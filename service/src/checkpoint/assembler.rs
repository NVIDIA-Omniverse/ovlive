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

//! Layer-checkpoint assembler (SR-2.3).
//!
//! Builds a version-filtered snapshot of a layer's state at a proven cut,
//! suitable for serializing into an OODF reset-delta. The module spans all of
//! SR-2.3: the consistent-snapshot primitive (steps 1-3) in
//! [`load_consistent_snapshot`], the OODF reset-delta + OMPF wrap (steps 4-5)
//! in [`build_reset_ompf`], and the public entrypoint that chains those with
//! the ext-refs sidecar and the `CheckpointStore` PUT (steps 6-8) in
//! [`assemble_layer_checkpoint`]. The periodic leader-elected loop in
//! [`crate::checkpoint::checkpointer`] is the sole caller.
//!
//! # Why the snapshot is bounded by a proven cut, and the two reads that prove it
//!
//! A blob's `snap_seq` claims coverage of every commit at or below it, so the
//! snapshot behind it must be a proven-complete consistent cut. A read whose
//! end-of-read `obj:hdr seq` merely bounds the versions it saw proves nothing of
//! the kind: a commit landing mid-read creates entities the read never saw,
//! whose versions still pass a `version <= snap_seq` filter, and the blob then
//! silently omits them while claiming coverage — a DR content-loss hole
//! (b.pw6). So every snapshot the assembler persists comes from one of two
//! proven reads, chosen by the layer's size (Spec SDD SR-4.1, SR-11.5):
//!
//! - **At or below the atomic-read limits** —
//!   [`crate::storage::diff_read::read_object_snapshot`]'s atomic read, consumer
//!   `checkpoint`: one `EVALSHA` that reads the whole object, header included,
//!   at one cut and returns that cut. The proof is the read's atomicity, so a
//!   commit landing during the read is either wholly in the state or wholly
//!   above the cut.
//! - **Above them** — the fenced read
//!   ([`crate::storage::snapshot_fence::read::read_object_snapshot_fenced`],
//!   Spec SDD SR-5.3), which the atomic read's `TOO_LARGE` answer routes to: a
//!   per-object fence that no commit lands through from its grant to its
//!   release, a bounded fan-out under it, and a release that proves the lease
//!   was held and the header never moved off the cut. Its wait for the fence is
//!   capped at `reader_wait_max(L)`
//!   ([`crate::storage::snapshot_fence::read::CappedFenceWait`]); a broken read
//!   is retried up to `FENCE_READ_ATTEMPTS` times; it takes no catch-up slot,
//!   read permit or memory budget (Spec SDD SR-4.5). It never returns an
//!   unproven state: exhaustion and the wait cap are transient errors.
//!
//! Either way there is no re-read, no raced section to reconcile and no
//! tombstone to re-read: an indexed section without a hash is a corrupt index,
//! never a race. The header `seq` pinned before the read serves only as a
//! regression check: a cut below it skips the tick, and that regression (a DR
//! restore with an undersized band) is the only way a layer of any size is
//! skipped other than an error, a lost leader lease or a shutdown.
//!
//! The version filter (`version > snap_seq` → drop) survives as a subordinate
//! guard rather than the primary mechanism: under a proven cut it should have
//! nothing to do, and it earns its keep only against entries a post-cut read
//! could still surface.
//!
//! # The leader lease during a fenced read
//!
//! The periodic loop holds its leader lease (`CheckpointLease`, TTL 30 s) for
//! the whole tick and re-proves it between layers. A fenced read of a large
//! layer — its wait for the fence plus its hold — can outlast that TTL, so the
//! caller hands [`assemble_layer_checkpoint`] a [`LeaderLeaseKeeper`] and the
//! fenced read runs beside it (Spec SDD SR-11.5): the keeper refreshes the
//! lease on its own period for as long as the read runs, and the moment it
//! cannot re-prove the lease the read is dropped — which releases its fence
//! (the fence lease guard's drop) — and the assembly ends
//! [`AssemblyDecision::LeaseLost`] without persisting anything. So no layer
//! size is skipped for want of time: the per-size checkpoint time lengthens the
//! tick's walk instead. The loop's keeper also stops the read when the loop is
//! cancelled for graceful shutdown, and the assembly then ends
//! [`AssemblyDecision::ShuttingDown`], so a fenced read never outlives the
//! shutdown's join of the loop and its fence is released before the pools
//! close.
//!
//! # `SnapshotLoader` injection seam
//!
//! The assembler calls four primitive operations: pin `obj:hdr seq` before the
//! state read, run the atomic state read, run the fenced state read, and read
//! the session-level header identity. To keep the assembler unit-testable
//! without spinning up `TestValkey`, those operations are gathered into a
//! [`SnapshotLoader`] trait. Sibling Subtask `.7r` (the unit tests) injects a
//! hand-built fake loader against the trait; production code uses
//! [`ValkeySnapshotLoader`], which delegates to `read_object_snapshot`,
//! `read_object_snapshot_fenced` and the fred pool. Note the assembler
//! deliberately does NOT call the storage functions directly — routing them
//! through the trait is what keeps the Valkey-free unit seam intact. The
//! integration test in Subtask `.kn` exercises the real loader against Valkey
//! end-to-end.
//!
//! # Test-only seam: an unprovable cut
//!
//! Since every read proves its cut, nothing but a header-sequence regression
//! reaches [`AssemblyDecision::SkippedNotQuiesced`], and with it the
//! checkpoint loop's tail-capture fallback. Tests that need that fallback arm
//! `ValkeyClient::arm_unprovable_checkpoint_cut_for_test` for an object
//! (compiled only under `test-checkpoint-fault-injection`, and in this crate's
//! unit tests): [`ValkeySnapshotLoader`]'s pin read then answers
//! `u64::MAX`, above every cut, so the proven read that follows is judged
//! regressed by the production arm, WARN included, and the tick persists
//! nothing for that layer.

use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::HashesInterface;
use serde::Serialize;

use crate::checkpoint::chunked_base::{encode_reset_base, ResetBaseParts};
use crate::domain::diff::{collect_external_value_hashes, DiffEngine, DiffPlan};
use crate::domain::limits::MAX_DELTA_ENTRIES;
use crate::domain::types::ObjectState;
use crate::storage::client::{ValkeyClient, POOL_LABEL_DATA};
use crate::storage::diff_read::{
    classify_snapshot_read_err, read_object_snapshot, snapshot_to_object_state, AtomicSnapshot,
    ObjectSnapshot, SnapshotKeyFamily, SnapshotRead, SnapshotReadConsumer, SnapshotTooLargeAt,
};
use crate::storage::keys::{sha256_to_hex, KeySchema};
use crate::storage::snapshot_fence::read::{read_object_snapshot_fenced, CappedFenceWait};
use crate::storage::snapshot_fence::LayerSize;

use super::store::{CheckpointError, CheckpointHandle, CheckpointMeta, CheckpointStore};

/// Keeps the checkpoint loop's leader lease held while a fenced read runs
/// (Spec SDD SR-11.5).
///
/// The assembler polls [`Self::hold_until_lost`] beside the fenced read — its
/// wait for the fence and its hold alike — and drops it when the read ends.
/// The periodic loop's implementation refreshes its `CheckpointLease` every
/// `LEADER_LOCK_TTL_MS / 3` (the same refresh, and the same fail-closed
/// verdicts, as its between-layers lease gate), and also returns
/// [`KeeperStop::ShuttingDown`] when the loop is cancelled for graceful
/// shutdown, so the read is dropped — and its fence released — while the
/// pools are still open. A test that drives the assembler outside the loop
/// passes one that never stops.
///
/// `pub` because [`assemble_layer_checkpoint`] is.
#[async_trait]
pub trait LeaderLeaseKeeper: Send {
    /// Keep the lease held, refreshing it whenever a refresh is due, and return
    /// only once the read must stop, saying why: [`KeeperStop::LeaseLost`]
    /// once the lease can no longer be re-proven (lapsed, held by a peer, or a
    /// refresh that failed), [`KeeperStop::ShuttingDown`] once the
    /// implementation's owner is shutting down. Never returns otherwise.
    /// Dropped, possibly mid-refresh, when the read it runs beside ends.
    async fn hold_until_lost(&mut self) -> KeeperStop;
}

/// Why a [`LeaderLeaseKeeper`] stopped the fenced read it ran beside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeeperStop {
    /// The leader lease could not be re-proven
    /// ([`AssemblyDecision::LeaseLost`]).
    LeaseLost,
    /// The keeper's owner is shutting down — the periodic loop's
    /// cancellation ([`AssemblyDecision::ShuttingDown`]). The lease may still
    /// be held; nothing more is written under it.
    ShuttingDown,
}

/// A [`LeaderLeaseKeeper`] for an assembly run outside the periodic loop,
/// which holds no leader lease: it never reports a loss, so the fenced read
/// is never cut short for one. Test and fixture use only.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct NoLeaseToKeep;

#[cfg(any(test, feature = "test-support"))]
#[async_trait]
impl LeaderLeaseKeeper for NoLeaseToKeep {
    async fn hold_until_lost(&mut self) -> KeeperStop {
        std::future::pending::<KeeperStop>().await
    }
}

/// Minimal storage-layer surface the assembler depends on.
///
/// Threaded as a trait object so unit tests in sibling Subtask `.7r` can
/// inject fakes without spinning up `TestValkey`. The production
/// implementation is [`ValkeySnapshotLoader`]; tests construct an inline fake.
///
/// All methods MUST be classified per the load-bearing-error contract —
/// backend failures bubble up as [`CheckpointError::Transient`] so the caller
/// can decide whether to retry or surface, and a missing object header (i.e.
/// `NotFound` on the underlying snapshot) is preserved as
/// [`CheckpointError::NotFound`] so the leader-elected loop can drop the
/// object from its working set without alarm.
#[async_trait]
pub(crate) trait SnapshotLoader: Send + Sync {
    /// Read the object's whole state in one atomic read (Spec SDD SR-4.1).
    /// Concretely a delegate to `storage::diff_read::read_object_snapshot` with
    /// [`SnapshotReadConsumer::Checkpoint`] for production, or a hand-built
    /// [`SnapshotRead`] for tests.
    ///
    /// Answers [`SnapshotRead::Snapshot`] with the state at its cut, or
    /// [`SnapshotRead::TooLarge`] when the layer is above the atomic-read
    /// limits, with no state — the caller then reads it through
    /// [`Self::read_fenced_snapshot`]. A missing header, or one without a
    /// numeric `seq`, is [`CheckpointError::NotFound`]; any other failure is
    /// classified as on every other method here.
    async fn read_atomic_snapshot(&self, object_id: &str) -> Result<SnapshotRead, CheckpointError>;

    /// Read the object's whole state under its snapshot fence, proven at the
    /// fence's cut (Spec SDD SR-5.3, SR-11.5): the read for a layer above the
    /// atomic-read limits. Concretely a delegate to
    /// `storage::snapshot_fence::read::read_object_snapshot_fenced` with
    /// [`SnapshotReadConsumer::Checkpoint`] and a
    /// [`CappedFenceWait`] hook for production — no capacity taken, the fence
    /// wait capped at `reader_wait_max(L)` — or a hand-built [`AtomicSnapshot`]
    /// for tests.
    ///
    /// `estimate` is the layer's size from the atomic read's `TOO_LARGE`
    /// answer ([`LayerSize::from_too_large`]); the fence scripts use the fence
    /// key's stored size whenever it has one.
    ///
    /// Returns only a verified state. A missing header (at acquire, or
    /// confirmed by the release) is [`CheckpointError::NotFound`]; a read
    /// broken `FENCE_READ_ATTEMPTS` times in a row, a fence wait that would
    /// pass `reader_wait_max(L)`, and every other failure are
    /// [`CheckpointError::Transient`] — the tick skips the layer and the next
    /// one reads it afresh.
    ///
    /// The caller may drop the returned future at any await (a lost leader
    /// lease, [`LeaderLeaseKeeper`]); the read's fence lease is then released
    /// by its guard's drop, with the lease deadline as the backstop.
    async fn read_fenced_snapshot(
        &self,
        object_id: &str,
        estimate: LayerSize,
    ) -> Result<AtomicSnapshot, CheckpointError>;

    /// Read `obj:hdr seq`. The assembler calls this once, BEFORE the state
    /// read, to obtain the regression check's pin. Returns `None` when the
    /// header is absent, which the assembler treats as
    /// [`CheckpointError::NotFound`] — a vanished header does not come back,
    /// so there is nothing to retry.
    async fn read_header_seq(&self, object_id: &str) -> Result<Option<u64>, CheckpointError>;

    /// Read the header columns the assembler stamps into `CheckpointMeta` in
    /// one `HGETALL`: the four session-level identity fields the apply /
    /// recovery path needs to reconstruct `obj:hdr` on
    /// `LayerStore::apply_checkpoint` (`session_id`, `layer_name`,
    /// `created_by`, `last_server_ts`), plus the two cursor columns the
    /// observed-commit-span measurement is derived from (`last_checkpoint_seq`,
    /// `restore_seq_bump` — b.vnw). None of them appear in the OODF reset-delta
    /// payload: the identity fields are session-level metadata rather than
    /// document state, and the cursor columns are checkpoint bookkeeping.
    ///
    /// Returns `None` in **two** cases, which every caller must treat
    /// identically (route onto its absent-header arm):
    ///
    /// 1. `obj:hdr` is absent — the concurrent DeleteLayer /
    ///    CreateLayer-overwrite race between the snapshot and identity reads.
    /// 2. `obj:hdr` exists but its `session_id` or `layer_name` is missing or
    ///    empty. A generation stamped from such a header is refused by
    ///    `validate_generation_identity` at the adapter and by the apply path's
    ///    identity guard, so it is permanently unrestorable — for DR purposes
    ///    the object does not exist, and saying so here is what keeps every
    ///    producer from having to re-check it (b.hka review round 2). The
    ///    implementation WARNs once at that site.
    ///
    /// The *descriptive* columns (`created_by`, `server_ts`) still fold to
    /// defaults rather than failing: no guard consults them, so downgraded
    /// description beats a refused generation. The same fold applies to the
    /// cursor columns, but there it is recorded in `cursor_columns_trusted` so
    /// the caller can tell a fabricated 0 from a read one and decline to measure
    /// a span against it.
    async fn read_header_identity(
        &self,
        object_id: &str,
    ) -> Result<Option<HeaderIdentity>, CheckpointError>;
}

/// Header columns read from `obj:hdr` for stamping into `CheckpointMeta` —
/// the four session-level identity fields plus the two cursor columns the
/// observed-commit-span measurement needs. See
/// [`SnapshotLoader::read_header_identity`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct HeaderIdentity {
    pub session_id: String,
    pub layer_name: String,
    pub created_by: String,
    pub server_ts: u64,
    /// `obj:hdr last_checkpoint_seq` — the snapshot generation that most
    /// recently landed for this object (0 = never checkpointed). Subtracted
    /// from the cut's `snap_seq` to measure how many commits this generation
    /// covers that the previous one did not.
    pub last_checkpoint_seq: u64,
    /// `obj:hdr restore_seq_bump` — the guard band a DR restore added on top
    /// of its restore floor when it seeded `seq`, or 0 when the current header
    /// was not written by a restore (or a checkpoint has landed since, which
    /// clears the column). Subtracted from the raw span so the artificial
    /// restore bump is never measured as if it were real commit traffic.
    pub restore_seq_bump: u64,
    /// `true` when both cursor columns above were read rather than fabricated:
    /// each was either parsed cleanly or genuinely absent from the hash (and
    /// absence *is* a truthful 0 — no checkpoint has landed, no restore bump is
    /// outstanding). `false` when a column was present but unparseable and the
    /// fold to 0 invented a baseline. Only [`measure_observed_commit_span`]
    /// consults it; the identity fields are unaffected either way.
    ///
    /// Defaults to `false` so every synthesized / fabricated `HeaderIdentity`
    /// declines the measurement rather than asserting one — the conservative
    /// direction, since a span measured against an invented baseline is not the
    /// per-interval quantity the DR guard band consumes.
    pub cursor_columns_trusted: bool,
}

/// Value of [`CheckpointMeta::observed_commit_span`] meaning "this generation
/// carries no usable per-interval measurement". `apply_checkpoint` folds it
/// back to the configured `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN`, which is
/// exactly the pre-b.vnw behaviour, so declining to measure is always safe.
/// `pub(crate)` so the SR-7.1 creation-time bootstrap
/// ([`crate::checkpoint::base_bootstrap`]) can name the sentinel it stamps rather
/// than writing a bare `0` whose meaning would have to be re-derived at that
/// write site.
pub(crate) const NO_COMMIT_SPAN_MEASUREMENT: u64 = 0;

/// Measure how many commits the generation being assembled covers that the
/// *previous* generation did not (b.vnw), or return
/// [`NO_COMMIT_SPAN_MEASUREMENT`] when no honest measurement is available.
///
/// The quantity `apply_checkpoint` consumes is a **per-interval** commit count:
/// how far `seq` can be expected to run past a checkpoint before the next one
/// lands. Two inputs cannot produce it, and each returns the sentinel rather
/// than a number that would be read as one:
///
/// - `cursor_columns_trusted == false` — a cursor column was present but
///   unparseable and folded to 0. The blob still assembles (see
///   [`SnapshotLoader::read_header_identity`]); only the measurement is
///   declined.
/// - `last_checkpoint_seq == 0` — unambiguously "no generation has ever landed
///   for this object". The periodic loop's skip-if-no-commits gate (`seq ==
///   last_checkpoint_seq`) makes a *written* generation at `snap_seq == 0`
///   unreachable, so there is no competing reading. Measuring here would report
///   commits-**since-creation**, which for a layer that predates the checkpoint
///   loop being enabled can exceed the per-interval rate by orders of
///   magnitude — a wrong number in the same units as a right one, which is
///   worse for the operator sizing the knob off these values than no number.
///
/// Otherwise the measurement is `snap_seq − last_checkpoint_seq −
/// restore_seq_bump`. The restore-bump subtraction is what keeps it honest
/// across a restore: `apply_checkpoint` stamps `last_checkpoint_seq = snap_seq`
/// (the recovery audit-trail invariant, SDD §22.6 step 6) while seeding `seq =
/// restore_floor + band`, so the raw difference on the first post-restore
/// assembly is `band + (restore_floor − snap_seq) + real commits`. Subtracting
/// the band leaves the two terms that are genuine commit traffic: the
/// stream-probe gap the restore proved had happened past the checkpoint, and
/// everything committed since the restore. The periodic loop clears
/// `restore_seq_bump` in the same HSET that advances `last_checkpoint_seq`, so
/// exactly one assembly per restore pays the correction.
///
/// `saturating_sub` twice rather than a checked arithmetic chain: every way
/// this can go negative is a header that moved under us (a restore landing
/// between the cut and the identity read is impossible — a restore requires an
/// absent header — but a hand-repaired column is not), and the sentinel is the
/// correct fallback for all of them.
///
/// Takes `&HeaderIdentity`, not `Option<&HeaderIdentity>`: the sole caller
/// (`assemble_and_write`) now refuses outright when `obj:hdr` vanished between
/// the cut and the identity read, so "no header" never reaches a measurement
/// decision. A fabricated identity would have declined here anyway
/// (`cursor_columns_trusted` defaults to `false`), so nothing is lost.
fn measure_observed_commit_span(snap_seq: u64, identity: &HeaderIdentity) -> u64 {
    if !identity.cursor_columns_trusted || identity.last_checkpoint_seq == 0 {
        return NO_COMMIT_SPAN_MEASUREMENT;
    }
    snap_seq
        .saturating_sub(identity.last_checkpoint_seq)
        .saturating_sub(identity.restore_seq_bump)
}

/// Production [`SnapshotLoader`] that delegates to `read_object_snapshot` (on
/// the snapshot lane), `read_object_snapshot_fenced` (its fan-out on the
/// snapshot lane, its fence scripts on the data pool) and the fred data pool.
pub(crate) struct ValkeySnapshotLoader<'a> {
    pub client: &'a ValkeyClient,
    pub keys: &'a KeySchema,
}

#[async_trait]
impl<'a> SnapshotLoader for ValkeySnapshotLoader<'a> {
    async fn read_atomic_snapshot(&self, object_id: &str) -> Result<SnapshotRead, CheckpointError> {
        // SR-4.1 direction: checkpoint → storage::diff_read, never the reverse.
        // The storage entry point owns the script, its classification and the
        // `live_edit_snapshot_read_*{consumer="checkpoint"}` ticks. No catch-up
        // read permit or build budget is taken (SR-4.5): the assembler is
        // leader-only and reads one layer at a time, and waiting behind
        // catch-ups would only add a cross-consumer wait.
        read_object_snapshot(
            self.client,
            self.keys,
            object_id,
            SnapshotReadConsumer::Checkpoint,
        )
        .await
        .map_err(service_err_to_checkpoint_err)
    }

    async fn read_fenced_snapshot(
        &self,
        object_id: &str,
        estimate: LayerSize,
    ) -> Result<AtomicSnapshot, CheckpointError> {
        // Every id the assembler reads is the decimal rendering of a `u64`
        // (`assemble_inner`), so this refuses only a caller bug.
        let numeric_id =
            object_id
                .parse::<u64>()
                .map_err(|parse_err| CheckpointError::Permanent {
                    source: format!(
                        "object id '{object_id}' is not a valid u64 for a fenced snapshot read: \
                     {parse_err}"
                    )
                    .into(),
                })?;
        // The storage read owns the fence scripts, the renewal timer, the
        // release's proof, the `BROKEN` retries and the
        // `live_edit_snapshot_read_*` / `live_edit_snapshot_fence_*` ticks
        // under `consumer="checkpoint"`. The hook takes no capacity — no
        // catch-up slot, fenced read permit or memory budget (Spec SDD SR-4.5)
        // — ignores the release events, and refuses a fence wait past
        // `reader_wait_max(L)` with a WARN (Spec SDD SR-11.5). The leader lease
        // is kept by the caller, beside this future ([`LeaderLeaseKeeper`]).
        let mut wait =
            CappedFenceWait::new(self.client, numeric_id, SnapshotReadConsumer::Checkpoint);
        // Boxed: the fenced read's future is large, and the checkpoint task
        // already carries the tick's own.
        Box::pin(read_object_snapshot_fenced(
            self.client,
            self.keys,
            numeric_id,
            SnapshotReadConsumer::Checkpoint,
            estimate,
            &mut wait,
        ))
        .await
        .map_err(service_err_to_checkpoint_err)
    }

    async fn read_header_seq(&self, object_id: &str) -> Result<Option<u64>, CheckpointError> {
        let key = self.keys.obj_header(object_id);
        // b.muu: `classify_snapshot_read_err`, not the bare
        // `map_snapshot_load_err` this used to call — the returned
        // `ServiceError` is identical by construction (that helper ends in the
        // same mapper with the same `op`), so this conversion's verdict is
        // unchanged and only the counting is added. Before b.muu this read was
        // the checkpoint consumer's FIRST touch of `obj:hdr` and it moved no
        // counter, which is what made `key_family="header"` unreachable here.
        let raw: Option<String> = self.client.pool.hget(&key, "seq").await.map_err(|e| {
            service_err_to_checkpoint_err(classify_snapshot_read_err(
                self.client,
                POOL_LABEL_DATA,
                object_id,
                SnapshotKeyFamily::Header,
                &key,
                "HGET obj:hdr seq",
                e,
            ))
        })?;
        match raw {
            None => Ok(None),
            Some(s) => match s.parse::<u64>() {
                // Test-only seam (module docs, "an unprovable cut"): an armed
                // object's pin reads above every cut, so the proven read that
                // follows is judged regressed by the production arm. A header
                // that is absent stays absent — the seam forces a regression,
                // never a deletion.
                #[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
                Ok(_)
                    if object_id
                        .parse::<u64>()
                        .is_ok_and(|id| self.client.unprovable_cut_faults.is_armed(id)) =>
                {
                    tracing::debug!(
                        object_id,
                        "checkpoint assembler: test seam armed for this object; pinning its \
                         header sequence at u64::MAX so its proven read is judged regressed"
                    );
                    Ok(Some(u64::MAX))
                }
                Ok(v) => Ok(Some(v)),
                Err(parse_err) => {
                    // Corrupt-but-present `seq` is a permanent backend
                    // failure, not transient — retrying the read can't fix
                    // a malformed value. Surface to the checkpoint loop.
                    Err(CheckpointError::Permanent {
                        source: format!(
                            "obj:hdr seq for object {object_id} is not a valid u64 ('{s}'): {parse_err}"
                        )
                        .into(),
                    })
                }
            },
        }
    }

    async fn read_header_identity(
        &self,
        object_id: &str,
    ) -> Result<Option<HeaderIdentity>, CheckpointError> {
        let key = self.keys.obj_header(object_id);
        // Single `HGETALL` over the header hash. The header is small (a
        // dozen fields) so reading them all is cheaper than one `HGET` per
        // column — six of them, four identity plus two cursor — and matches
        // the read shape the snapshot fan-out already uses on the same key.
        let data: std::collections::HashMap<String, String> =
            self.client.pool.hgetall(&key).await.map_err(|e| {
                service_err_to_checkpoint_err(classify_snapshot_read_err(
                    self.client,
                    POOL_LABEL_DATA,
                    object_id,
                    SnapshotKeyFamily::Header,
                    &key,
                    "HGETALL obj:hdr",
                    e,
                ))
            })?;
        if data.is_empty() {
            // `obj:hdr` was deleted between the snapshot fan-out and this
            // read — treat as the same transient race as `read_header_seq`'s
            // `None` return so the orchestrating loop can decide whether to
            // retry the assembly or surface.
            return Ok(None);
        }
        // A header that EXISTS but carries no usable `session_id` /
        // `layer_name` is reported as absent, and that is the one place this
        // decision is taken (b.hka review round 2, path (b)).
        //
        // These two columns used to fold to `String::new()` alongside every
        // other column, and the fold was a foot-gun rather than a
        // downgrade-in-place: `validate_generation_identity` refuses such a
        // generation at the adapter with a `Permanent`, so every producer that
        // did not separately re-check emptiness turned a degenerate header into
        // an adapter-health error tick on every tick, forever, from a perfectly
        // healthy adapter. Exactly one producer (the periodic assembler) carried
        // that second check; the tail-capture and bootstrap producers refused
        // only an ABSENT header. Refusing here makes all three route onto the
        // arm they already have for "the object is gone" — which is also the
        // right operator reading: a layer with no identity cannot be restored,
        // so for DR purposes it does not exist.
        //
        // `created_by` and `server_ts` keep folding: they are descriptive, and
        // no guard on the restore path consults them.
        let session_id = data.get("session_id").cloned().unwrap_or_default();
        let layer_name = data.get("layer_name").cloned().unwrap_or_default();
        if session_id.is_empty() || layer_name.is_empty() {
            // One WARN, at the shared site rather than per producer. Not
            // reachable from the shipped write paths — `create_layer` and the
            // two materialization entry points (`write_object` for CreateLayer,
            // `write_restored_object` for the DR restore) write both columns
            // inside the MULTI/EXEC that creates the header — so it means a
            // partially written or
            // hand-repaired header, and it is worth an operator's attention
            // precisely because it is not routine.
            tracing::warn!(
                object_id,
                session_id_empty = session_id.is_empty(),
                layer_name_empty = layer_name.is_empty(),
                "obj:hdr exists but carries empty session_id/layer_name; treating the header as \
                 absent, so no permanently unrestorable generation is written for this object"
            );
            return Ok(None);
        }
        // Every u64 column folds a missing / unparseable value to 0 rather
        // than failing the assembly — dropping a layer out of DR over a
        // bookkeeping column would be strictly worse than a downgraded blob.
        let header_u64 = |field: &str| -> u64 {
            data.get(field)
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0)
        };
        // The two cursor columns (b.vnw) fold the same way but report whether
        // the 0 they may carry was *read* or *invented*. The distinction is not
        // cosmetic: an absent column is a truthful 0 (no checkpoint has landed;
        // no restore bump is outstanding), while a present-but-unparseable one
        // is a fabricated baseline, and measuring `snap_seq − 0` against it
        // would report commits-since-creation as if it were this layer's
        // per-interval commit rate. `measure_observed_commit_span` declines to
        // measure in that case; the blob still assembles and its restore falls
        // back to the configured margin.
        let cursor_u64 = |field: &str| -> (u64, bool) {
            match data.get(field) {
                None => (0, true),
                Some(raw) => match raw.parse::<u64>() {
                    Ok(v) => (v, true),
                    Err(_) => (0, false),
                },
            }
        };
        let (last_checkpoint_seq, seq_read) = cursor_u64("last_checkpoint_seq");
        let (restore_seq_bump, bump_read) = cursor_u64("restore_seq_bump");
        let cursor_columns_trusted = seq_read && bump_read;
        if !cursor_columns_trusted {
            // Only reachable via header corruption or a hand-repair — the
            // checkpoint loop and the apply path both write these columns as
            // decimal u64. Worth a WARN precisely because it is not routine.
            tracing::warn!(
                object_id,
                last_checkpoint_seq_parsed = seq_read,
                restore_seq_bump_parsed = bump_read,
                "obj:hdr checkpoint cursor column present but unparseable; \
                 assembling without an observed-commit-span measurement"
            );
        }
        Ok(Some(HeaderIdentity {
            session_id,
            layer_name,
            created_by: data.get("created_by").cloned().unwrap_or_default(),
            // `last_server_ts` is the most recently-written server timestamp;
            // we use it (rather than `created_at` / `modified_at`) because
            // `write_restored_object` writes the same value into all three on apply
            // and `last_server_ts` is the one that survives subsequent
            // updates. Missing / unparseable values fold to 0 — preferable
            // to permanent failure during recovery.
            server_ts: header_u64("last_server_ts"),
            last_checkpoint_seq,
            restore_seq_bump,
            cursor_columns_trusted,
        }))
    }
}

/// Map a `ServiceError` from the storage layer into a `CheckpointError`.
///
/// `ServiceError::NotFound` (no object header) is preserved as `NotFound` so
/// the leader-elected loop can drop the object from its working set; every
/// other variant is treated as `Transient` — a checkpoint write that fails
/// for any retryable reason will be retried on the next tick by the loop.
///
/// `pub(crate)` so sibling checkpoint-side seams reuse this one mapping rather
/// than growing another copy (`crate::checkpoint::tail_classify`'s
/// `ValkeyTailSliceLoader`).
pub(crate) fn service_err_to_checkpoint_err(err: crate::error::ServiceError) -> CheckpointError {
    match err {
        crate::error::ServiceError::NotFound(_) => CheckpointError::NotFound,
        other => CheckpointError::Transient {
            source: format!("{other}").into(),
        },
    }
}

/// Outcome of [`load_consistent_snapshot`].
///
/// Deliberately NOT a `CheckpointError` variant for the unverified arm: the
/// four-variant `CheckpointError` vocabulary (`Transient` / `Permanent` /
/// `NotFound` / `Conflict`) is the adapter contract the SR-1.3 conformance
/// harness asserts (§22.2, tests #19-#22), and "no cut could be proven for this
/// layer this tick" is a *scheduling* decision the assembler makes, not an
/// adapter failure. Keeping it in the success channel also keeps it out of
/// `result="error"` on `live_edit_checkpoint_writes_total`, whose adapter-health
/// alert would otherwise fire on a healthy adapter.
// The size gap between the variants is real but irrelevant here: exactly one
// of these is constructed per layer per checkpoint tick, on a background loop,
// and it is destructured immediately by the sole caller. Boxing the snapshot
// would add a heap allocation to the hot (verified) path to shrink a move that
// happens once every `LIVE_EDIT_CHECKPOINT_TICK_SECS`.
#[allow(clippy::large_enum_variant)]
pub(crate) enum ConsistentSnapshot {
    /// The cut is proven — by the atomic read, or by the fenced read's
    /// release — and is at or above the pin. `snapshot` is a complete,
    /// version-filtered consistent cut at `snap_seq`: every commit at or below
    /// `snap_seq` is folded in, and none above it is.
    Verified {
        snapshot: ObjectSnapshot,
        snap_seq: u64,
    },
    /// The read proved a cut, but below the pin: the header sequence regressed
    /// between the two reads (a DR restore with an undersized band; Spec SDD
    /// SR-4.1), and the object header was still present — not a deletion race
    /// (which surfaces as [`CheckpointError::NotFound`] instead). The only way
    /// a layer of any size is skipped other than an error, a lost leader
    /// lease or a shutdown.
    ///
    /// There is no snapshot the assembler is willing to persist; the caller
    /// MUST skip the object this tick and write nothing (neither blob nor
    /// cursor), leaving the previous generation as the newest durable copy.
    NotQuiesced,
    /// The leader lease could not be re-proven while the fenced read ran
    /// ([`LeaderLeaseKeeper::hold_until_lost`] returned
    /// [`KeeperStop::LeaseLost`]), so the read was
    /// dropped — releasing its fence — and nothing was read to persist. The
    /// caller MUST stop mutating shared durable state.
    LeaseLost,
    /// The keeper's owner is shutting down
    /// ([`LeaderLeaseKeeper::hold_until_lost`] returned
    /// [`KeeperStop::ShuttingDown`]), so the read was dropped — releasing any
    /// fence lease it held — and nothing was read to persist.
    ShuttingDown,
}

/// How many corrupt-index members a [`CheckpointError::Permanent`] names before
/// it summarises the rest as a count, so one badly corrupted index cannot turn
/// the assembly-failure WARN into an unbounded line.
const CORRUPT_INDEX_MEMBERS_NAMED: usize = 8;

/// Load a provably-complete consistent snapshot of an object's state (SR-2.3
/// steps 1-3, corrected by b.pw6; one proven read per Spec SDD SR-4.1 and
/// SR-11.5).
///
/// # Algorithm
///
/// 1. **Pin `obj:hdr seq` BEFORE the state read.** `loader.read_header_seq` is
///    the pin the read's cut is judged against (step 4), so this read must
///    precede the state read. An absent header is `CheckpointError::NotFound`
///    immediately — a vanished header does not come back, so retrying it is
///    pointless.
/// 2. **One atomic state read.** `loader.read_atomic_snapshot` reads the whole
///    object at one cut, or answers `TOO_LARGE` for a layer above the
///    atomic-read limits. A missing header is `NotFound`; any other failure
///    propagates as classified.
/// 3. **`TOO_LARGE` ⇒ one fenced state read** (`loader.read_fenced_snapshot`,
///    sized from the answer), beside `lease`
///    ([`LeaderLeaseKeeper::hold_until_lost`]) for as long as it runs — its
///    fence wait and its hold. The read returns only a verified state; it ends
///    `NotFound` when the release confirms the header gone, and `Transient`
///    when it was broken `FENCE_READ_ATTEMPTS` times or its fence wait would
///    pass `reader_wait_max(L)` — either way the tick skips the layer, never
///    persisting an unproven cut. If the keeper reports the lease lost first,
///    the read is dropped (its fence released by the lease guard's drop) and
///    the answer is [`ConsistentSnapshot::LeaseLost`]; if it reports its owner
///    shutting down, the read is dropped the same way and the answer is
///    [`ConsistentSnapshot::ShuttingDown`].
/// 4. **Judge the proven read against the pin.** Nothing is retried, re-read
///    or reconciled ([`consistent_from_proven`]):
///    - `cut >= pin` ⇒ [`ConsistentSnapshot::Verified`] at `snap_seq = cut`,
///      whatever commits landed during the read — each is wholly in the state
///      or wholly above the cut. Then the version filter runs as the
///      subordinate guard.
///    - An `obj:secs` member naming no readable section ⇒
///      `CheckpointError::Permanent`: under a proven cut that is a corrupt
///      index, not a race, and persisting the state would lose the section
///      without a trace. Retrying reads the same corruption.
///    - `cut < pin` ⇒ [`ConsistentSnapshot::NotQuiesced`] with a WARN naming
///      both: the header sequence regressed between the pin and the read, which
///      only a DR restore with an undersized band can cause. The next tick
///      re-pins from the current header.
///
/// # Errors
///
/// - `CheckpointError::NotFound` — the object header is absent: at the pin read
///   (step 1), at the atomic read, or as the fenced read's acquire or release
///   found it.
/// - `CheckpointError::Transient` — a backend failure from the loader (which
///   classifies every non-`NotFound` storage error this way), a fenced read
///   broken `FENCE_READ_ATTEMPTS` times in a row, or a fence wait that would
///   pass `reader_wait_max(L)`.
/// - `CheckpointError::Permanent` — the loader observed a malformed
///   on-the-wire value (e.g. a non-numeric `seq`), or the section index is
///   corrupt (a member without a hash or a decimal id).
pub(crate) async fn load_consistent_snapshot(
    loader: &dyn SnapshotLoader,
    object_id: &str,
    lease: &mut dyn LeaderLeaseKeeper,
) -> Result<ConsistentSnapshot, CheckpointError> {
    // Step 1: pin the header sequence BEFORE the state read.
    let Some(pin) = loader.read_header_seq(object_id).await? else {
        return Err(CheckpointError::NotFound);
    };

    // Step 2: exactly one atomic state read.
    let proven = match loader.read_atomic_snapshot(object_id).await? {
        SnapshotRead::Snapshot(atomic) => *atomic,
        // Step 3: above the atomic-read limits. `read_object_snapshot` has
        // already counted the answer (`outcome="too_large"`) and WARNed if it
        // was the item-tally anomaly, so this line only records the route.
        SnapshotRead::TooLarge(too_large) => {
            tracing::debug!(
                object_id = object_id,
                pinned_seq = pin,
                item_bound = too_large.item_bound,
                early_out = matches!(too_large.at, SnapshotTooLargeAt::EarlyOut),
                "checkpoint assembler: layer is above the atomic-read limits; reading it under \
                 its snapshot fence"
            );
            // No fence probe for a recorded bytes-per-item: the estimate only
            // seeds a fresh fence key (a stored size always wins), and the
            // checkpoint reserves no memory from it.
            let estimate = LayerSize::from_too_large(&too_large, None);
            let read = loader.read_fenced_snapshot(object_id, estimate);
            // Biased toward the keeper: a read that completes in the same poll
            // the lease is found lost is still not persisted, since nothing
            // may be written without the lease, and one that completes as the
            // loop shuts down is not persisted either, since shutdown is
            // stopping the loop's writes.
            tokio::select! {
                biased;
                stop = lease.hold_until_lost() => match stop {
                    KeeperStop::LeaseLost => {
                        tracing::debug!(
                            object_id = object_id,
                            pinned_seq = pin,
                            "checkpoint assembler: leader lease lost during the fenced read; \
                             abandoning the read (its fence is released) and persisting nothing"
                        );
                        return Ok(ConsistentSnapshot::LeaseLost);
                    }
                    KeeperStop::ShuttingDown => {
                        tracing::debug!(
                            object_id = object_id,
                            pinned_seq = pin,
                            "checkpoint assembler: shutting down during the fenced read; \
                             abandoning the read (any fence lease it held is released) and \
                             persisting nothing"
                        );
                        return Ok(ConsistentSnapshot::ShuttingDown);
                    }
                },
                read = read => read?,
            }
        }
    };

    // Step 4: proven by construction, judged only against the pin.
    consistent_from_proven(object_id, pin, proven)
}

/// Judge one proven read — atomic, or fenced and verified by its release —
/// against the pin (step 4 of [`load_consistent_snapshot`]). No I/O: the read
/// already proved its own cut, so the only questions left are whether the cut
/// can serve the pin and whether the index it read is sound.
fn consistent_from_proven(
    object_id: &str,
    pin: u64,
    proven: AtomicSnapshot,
) -> Result<ConsistentSnapshot, CheckpointError> {
    let cut = proven.cut;
    let path = proven.path.label();
    if cut < pin {
        // The header sequence went backwards between the pin and the read. The
        // state is a proven cut, but at a sequence below one this tick already
        // observed, so it cannot stand for the layer the pin saw; persist
        // nothing and let the next tick re-pin (Spec SDD SR-4.1, the
        // checkpoint's twin of catch-up's SR-3.2 regression check).
        tracing::warn!(
            object_id = object_id,
            pinned_seq = pin,
            cut,
            path,
            "checkpoint assembler: the layer's header sequence regressed between the pin and \
             the proven read (cut below the pin, which only a DR restore with an undersized \
             band causes); skipping this layer this tick (nothing persisted)"
        );
        return Ok(ConsistentSnapshot::NotQuiesced);
    }
    let corrupt = &proven.corrupt_section_index_members;
    if !corrupt.is_empty() {
        let named: Vec<&str> = corrupt
            .iter()
            .take(CORRUPT_INDEX_MEMBERS_NAMED)
            .map(String::as_str)
            .collect();
        return Err(CheckpointError::Permanent {
            source: format!(
                "obj:secs of object {object_id} is corrupt at proven ({path}) cut {cut}: {} \
                 member(s) name no readable section (no section hash, or not a decimal section \
                 id), first {}: {named:?}; refusing to persist a checkpoint that would drop them",
                corrupt.len(),
                named.len(),
            )
            .into(),
        });
    }
    let mut snapshot = proven.state;
    // Subordinate guard: under a proven cut it has nothing to drop.
    filter_snapshot_by_snap_seq(&mut snapshot, cut);
    Ok(ConsistentSnapshot::Verified {
        snapshot,
        snap_seq: cut,
    })
}

/// Drop sections / fields / time samples / tombstones whose per-entry version
/// is strictly greater than `snap_seq`.
///
/// Subordinate to the proven read in [`load_consistent_snapshot`], not a
/// substitute for it: under a verified cut this should find nothing to drop,
/// because the read observed a single commit boundary (atomically, or under the
/// snapshot fence). It survives as defence-in-depth against an entry a
/// post-cut read could still surface.
///
/// Section retention uses `last_version` (the latest write that touched the
/// section); a section first created before `snap_seq` but later modified at
/// `version > snap_seq` is dropped, because including it would commit the
/// future-write to the durable checkpoint. Field / time-sample / tombstone
/// retention uses each entry's own `version` / `last_version`.
///
/// `children` and `next_set_order` are not filtered: they're header-level
/// metadata that the writer ratchets atomically with `seq`, so `snap_seq`
/// bounds them by construction.
pub(crate) fn filter_snapshot_by_snap_seq(snapshot: &mut ObjectSnapshot, snap_seq: u64) {
    snapshot
        .sections
        .retain(|_, section| section.last_version <= snap_seq);
    snapshot.fields.retain(|f| f.version <= snap_seq);
    snapshot.time_samples.retain(|ts| ts.version <= snap_seq);
    snapshot
        .tombstones
        .retain(|&(_, _, last_ver)| last_ver <= snap_seq);
}

/// Emit an OODF reset-delta from a filtered [`ObjectSnapshot`] (SR-2.3 step 4-5).
///
/// Thin wrapper over [`build_reset_ompf_from_state`] — it converts the
/// storage-layer snapshot to the domain-layer [`ObjectState`] the builder
/// actually consumes and delegates. The whole contract (return shape, the
/// `DiffEngine::plan` routing argument, which `set_order` rank the wire slot
/// carries, and the error taxonomy) is documented on that function.
pub(crate) fn build_reset_ompf(
    snapshot: &ObjectSnapshot,
) -> Result<(Vec<u8>, Vec<[u8; 32]>), CheckpointError> {
    // Convert storage-layer `ObjectSnapshot` to domain-layer `ObjectState` so
    // we can reuse `DiffEngine::plan`'s DFS ordering / orphan handling.
    build_reset_ompf_from_state(&snapshot_to_object_state(snapshot))
}

/// Emit an OODF reset-delta (`isDiff=true, baseVersion=0`) wrapped in an
/// `OmpfContainer`, sourced from a domain-layer [`ObjectState`] (SR-2.3 step 4-5).
///
/// Returns `(ompf_bytes, ordered_ext_hashes)`:
///
/// - `ompf_bytes` — the on-the-wire OMPF container. Carries the delta as Part 0;
///   the external-value parts list is intentionally empty for Epic 1. A state
///   over `MAX_DELTA_ENTRIES` entries is written as a chunked base instead
///   (`checkpoint::chunked_base`), which `decode_checkpoint_ompf` reads back
///   into the same state. Field /
///   time-sample entries that reference external values still carry the
///   `ext_value_index` assigned by [`collect_external_value_hashes`] so the
///   apply-side Task (`t2.fgj.ya.n9`) can resolve them against the side-channel
///   ext blobs once Epic 3 mirrors them.
/// - `ordered_ext_hashes` — the deduplicated ext-value hashes in
///   first-appearance order, ready for the ext-refs sidecar that pairs with the
///   checkpoint blob (sibling Subtask `.wi`'s entrypoint emits this alongside
///   the OMPF bytes).
///
/// # Why route through `DiffEngine::plan`
///
/// The in-RPC diff path's `DiffPlan::FullContent` arm emits sections in DFS
/// parent-first order with orphans appended (SDD §10.3). The reset-delta
/// produced by this helper is the durable cross-Task round-trip contract
/// (`t2.fgj.ya.n9` re-applies it on recovery), so the section ordering MUST
/// match exactly — re-routing through `DiffEngine::plan(state, &[], 0, 0, None)`
/// borrows the same DFS walk rather than duplicating it. The arguments
/// `have_object_id = 0` and `have_version = 0` force the Case-1 (FullContent)
/// branch by construction (any non-zero `state.object_id` triggers the
/// "object_id mismatch" path; even on the degenerate `state.object_id == 0`
/// branch, `have_version == 0` still routes to FullContent). That second half
/// of the argument is what makes this safe for the SR-7.1 creation-time
/// bootstrap, whose `ObjectState` comes straight out of the CreateLayer handler
/// and therefore still carries `object_id == 0` (the id is allocated inside
/// `LayerStore::create_layer`, never written back onto the in-memory state):
/// `have_version == 0` forces FullContent on that input too.
///
/// # Which rank the field `set_order` slot carries
///
/// Each field carries two persisted ranks: the cumulative `set_order` write
/// cursor and the writer's true per-delta `authoring_set_order`. OODF
/// `SetField` has exactly ONE `set_order` wire slot, so this reset-delta can
/// faithfully carry only one of the two across a DR serialize → restore
/// round-trip. This helper serializes the field's `authoring_set_order` into
/// that slot (not the cumulative cursor), because BOTH faithful catch-up
/// builders (`plan_faithful_snapshot` and `plan_faithful_coalesced`)
/// reconstruct authoring order from `authoring_set_order` — losing it would
/// reorder any catch-up served after a DR restore. The cumulative cursor is
/// intentionally NOT serialized: a reset delta always re-derives it from
/// scratch on apply (`apply_reset` zeroes `next_set_order`; `apply_set_field`
/// re-accumulates as it applies the reset's fields), so dropping it loses
/// nothing. Field emission order does not affect
/// this — each field carries its own authoring rank verbatim, so WHICH rank
/// lands in the wire slot is order-independent.
///
/// Emission order is nonetheless ALSO total: `plan_full_content` sorts fields on
/// `(version, authoring_set_order, section_id, key_name)` and time samples on
/// `(version, section_id, time)` (b.gcb / b.vz7), which is what makes the blob
/// this helper emits byte-reproducible for a given `ObjectState` and gives
/// SR-9.2's re-base idempotency argument its structural ground. The two facts
/// are independent and both load-bearing: the first is why emission order was
/// never a *correctness* problem, the second is why it is now a *byte-identity*
/// guarantee.
///
/// This remap lives in the checkpoint assembler only; `DiffEngine::plan` /
/// `plan_full_content` stay **lossy** — no net-live pruning, no dense re-index,
/// cumulative `set_order` emitted verbatim — and **stable** on the ground just
/// stated, and the in-RPC diff path is unaffected (SR-4.1). Only the first half
/// is a prohibition: the lossy semantics MUST NOT change, because this helper
/// depends on them. The stable half is not a fixed wire order to preserve
/// blindly but a deliberate total order that replaced `HashMap` iteration —
/// the same lossy-vs-stable split `plan_faithful_snapshot`'s doc-comment draws.
///
/// # Errors
///
/// - [`CheckpointError::Permanent`] if `DiffEngine::plan` rejects the snapshot
///   (e.g. `SectionTreeTooDeep`), if `build_delta` fails (OODF FFI allocator
///   failure), or if `OmpfContainer::serialize` or the chunked-base framing
///   refuses its input.
///   None of these are retryable — the snapshot bytes we hold are malformed or
///   too large, and retrying the same input produces the same failure.
pub(crate) fn build_reset_ompf_from_state(
    state: &ObjectState,
) -> Result<(Vec<u8>, Vec<[u8; 32]>), CheckpointError> {
    // Force the FullContent branch — see the doc-comment above for why this
    // pair of zeros is the right key. Empty tombstones because a reset-delta
    // describes the *current* state; the apply side rebuilds from scratch and
    // does not need a tombstone list.
    let plan =
        DiffEngine::plan(state, &[], 0, 0, None).map_err(|e| CheckpointError::Permanent {
            source: format!("diff plan rejected snapshot during checkpoint assembly: {e}").into(),
        })?;

    let (sections, mut fields, time_samples) = match plan {
        DiffPlan::FullContent {
            sections,
            fields,
            time_samples,
        } => (sections, fields, time_samples),
        // Unreachable by construction: `have_object_id=0, have_version=0` always
        // produces FullContent. Surface as `Permanent` rather than panic so a
        // future refactor of `DiffEngine::plan` that broke this invariant would
        // be reported cleanly to the checkpoint loop instead of aborting the
        // tokio worker.
        other => {
            return Err(CheckpointError::Permanent {
                source: format!(
                    "DiffEngine::plan produced unexpected variant for reset-delta build: {other:?}"
                )
                .into(),
            });
        }
    };

    // Remap each field so the value serialized into the single OODF
    // `SetField.set_order` wire slot is the writer's true authoring rank, not
    // the cumulative write cursor. `build_delta` serializes `f.set_order`
    // verbatim into that slot, and OODF `SetField` has exactly ONE such slot —
    // so the DR round-trip can faithfully carry only one of the two persisted
    // ranks. The load-bearing one is `authoring_set_order` (the faithful
    // catch-up snapshot reconstructs authoring order from it); the cumulative
    // cursor is intentionally dropped because the reset-apply path re-derives it
    // from scratch (`apply_reset` zeroes `next_set_order`; `apply_set_field`
    // re-accumulates). Perform this before `collect_external_value_hashes` /
    // `build_delta` so the serialized bytes carry the authoring rank.
    for f in &mut fields {
        f.set_order = f.authoring_set_order;
    }

    // Scan for ext-value hash references in the field / time-sample payloads.
    // The hash → 1-based index map produced here drives the `ext_value_index`
    // stamped onto each `SetField` / `TimeSample` inside `build_delta`. Even
    // though we pass `&[]` for the OMPF parts list below (Epic 1 ships hash
    // refs only — actual blob bytes travel via the ext-refs sidecar that
    // Epic 3 mirrors), the per-field `ext_value_index` still encodes the
    // logical ordering so apply-side reassembly can pair it with the sidecar.
    let scan = collect_external_value_hashes(&fields, &time_samples);

    // Encode the base. `is_diff=true, base_version=0` is exactly the shape
    // `DiffPlan::FullContent` emits in the in-RPC path, built by the same
    // `build_delta`. At or below `MAX_DELTA_ENTRIES` entries this is one OMPF
    // (no external-value parts — the ext-refs sidecar carries the blobs),
    // byte-identical to every earlier base; above it, a chunked base that
    // decodes into the same state (`checkpoint::chunked_base`).
    let ompf_bytes = encode_reset_base(
        &ResetBaseParts {
            source_format: &state.source_format,
            sections: &sections,
            fields: &fields,
            time_samples: &time_samples,
            hash_to_index: &scan.hash_to_index,
        },
        MAX_DELTA_ENTRIES,
    )?;

    Ok((ompf_bytes, scan.ordered_hashes))
}

/// Outcome of a successful [`assemble_layer_checkpoint`] call.
///
/// The orchestrating leader-elected loop (Subtask `t3.fgj.ya.78.kn` and beyond)
/// reads `snap_seq` to advance its per-object cursor and `ext_hash_count` to
/// drive the per-tick metric the SDD asks for (SR-2.3 step 8). `handle` is
/// adapter-opaque; callers thread it forward unchanged.
///
/// The shape carries exactly what the subtask body promises (`handle`,
/// `snap_seq`, `ext_hash_count`); byte-length figures used by the
/// success-tracing line are computed inside [`assemble_layer_checkpoint`] and
/// not surfaced in the outcome — operators get them from the log line, the
/// loop doesn't need them for its own bookkeeping.
#[derive(Debug, Clone)]
pub struct AssemblerOutcome {
    pub handle: CheckpointHandle,
    pub snap_seq: u64,
    pub ext_hash_count: usize,
}

/// What one `assemble_layer_checkpoint` call decided to do (b.pw6).
///
/// The assembler has five outcomes: it persisted a generation, it
/// deliberately persisted nothing, its leader lease was lost under it, its
/// owner shut down under it, or it failed. Only the last is a
/// `CheckpointError` — see [`ConsistentSnapshot`] for why the deliberate-skip
/// case is kept out of the four-variant error vocabulary; a lost lease and a
/// shutdown are not adapter failures either.
#[derive(Debug, Clone)]
pub enum AssemblyDecision {
    /// A generation was assembled and persisted; the caller advances the
    /// per-object cursor and runs the retention prune.
    Written(AssemblerOutcome),
    /// The read proved a cut below the header sequence pinned before it: the
    /// sequence regressed between the two reads (a DR restore with an
    /// undersized band; see [`ConsistentSnapshot::NotQuiesced`]). The only way
    /// a layer of any size is skipped other than an error, a lost leader
    /// lease ([`Self::LeaseLost`]) or a shutdown ([`Self::ShuttingDown`]),
    /// since every read proves its cut (the atomic read at or below the
    /// atomic-read limits, the fenced read above them). So nothing was persisted — no blob, no
    /// sidecar, and (caller's responsibility) no `last_checkpoint_seq` cursor
    /// write. The previous generation stays the newest durable copy and the
    /// next tick re-attempts.
    ///
    /// Persisting anyway would be the b.pw6 hole: the blob's `snap_seq` would
    /// claim coverage of commits the snapshot may never have observed, the
    /// cursor write would then silence the next tick's skip-if-no-commits gate,
    /// and retention would eventually prune the good older generations behind
    /// it.
    ///
    /// The cost of skipping is one further tick per regression: the next tick
    /// re-pins from the current header and its read proves a cut however hot
    /// the layer is. The exposure is visible as `outcome="unverified"` on
    /// `live_edit_checkpoint_stable_cut_total`.
    SkippedNotQuiesced,
    /// The leader lease could not be re-proven while the layer's fenced read
    /// ran ([`LeaderLeaseKeeper`]), so the read was abandoned — its fence
    /// released — and nothing was persisted. The caller must stop the tick's
    /// walk here, as at its between-layers lease gate: no further layer, and
    /// no destructive step, without the lease. The layer stays due.
    LeaseLost,
    /// The keeper stopped the layer's fenced read because its owner is
    /// shutting down ([`KeeperStop::ShuttingDown`]: the periodic loop's
    /// cancellation), so the read was abandoned — any fence lease it held
    /// released — and nothing was persisted. Neither a lease loss nor a
    /// failure: the caller stops as at any other cancellation, and the layer
    /// stays due.
    ShuttingDown,
}

/// Ext-refs sidecar JSON shape — `{"ext_hashes": ["<lowercase-64-hex>", ...]}`.
///
/// The shape is intentionally minimal: a single named field with a flat list.
/// Future evolutions (e.g. carrying per-hash byte-length hints, or a schema
/// version) would extend the struct with new optional fields; the InMemory and
/// LocalDisk adapters' `Deserialize` impls tolerate unknown fields per serde's
/// default. We do NOT bump a schema version eagerly — the orphan sweep deserializes
/// every sidecar in the store on every tick, so the on-the-wire shape should stay
/// stable across Epic-1 ticks.
///
/// Deliberately private: every producer goes through
/// [`ext_refs_sidecar_json`], which is what keeps the byte-level determinism
/// argument documented in one place.
#[derive(Debug, Serialize)]
struct Sidecar {
    ext_hashes: Vec<String>,
}

/// Serialize the ext-refs sidecar JSON that pairs with a **base** checkpoint
/// blob, from the ordered ext-value hashes
/// [`build_reset_ompf_from_state`] returned.
///
/// Serialized via `serde_json::to_vec`, in
/// [`ExternalValueScan::ordered_hashes`](crate::domain::diff::ExternalValueScan)
/// order (first-appearance, no resort here). The JSON byte content is
/// byte-for-byte deterministic across replicas seeing the same snapshot, and the
/// ground for that claim is a three-link chain that does NOT start in this file:
///
/// 1. [`DiffEngine::plan`](crate::domain::diff::DiffEngine::plan)'s
///    `plan_full_content` arm — the arm `build_reset_ompf_from_state` forces —
///    emits `fields` and `time_samples` in a **total order over `ObjectState`'s
///    content**: fields on `(version, authoring_set_order, section_id,
///    key_name)`, time samples on `(version, section_id, time)`. Each tuple ends
///    in the respective map's key, so neither can tie, and no `HashMap` table
///    layout survives into the plan (b.gcb / b.vz7).
/// 2. `collect_external_value_hashes` walks those two slices in order into ONE
///    shared first-appearance pool and does not resort, so `ordered_hashes` is a
///    pure function of the plan's emission order — which is why the time-sample
///    half of the sort is load-bearing here, not merely cosmetic.
/// 3. First-appearance order over a deterministic sequence is itself
///    deterministic, and this function maps it 1:1 through `sha256_to_hex`.
///
/// That determinism is load-bearing for cross-replica idempotency of the
/// retention-cascade and orphan-sweep paths (SR-3.4) which read the sidecar
/// independently of the OMPF. `sha256_to_hex` produces the lowercase 64-char
/// hex of the 32-byte content hash that the trait contract requires.
///
/// **Maintenance obligation.** Link 1 lives in another module
/// ([`crate::domain::diff`]), so a reorder of `plan_full_content`'s emission
/// would falsify this paragraph from outside this file — but it does not do so
/// silently. Two unit tests in this file's `mod tests`,
/// `build_reset_ompf_from_state_is_byte_identical_for_equal_state` and
/// `ext_refs_sidecar_json_is_byte_identical_for_equal_state`, fail on exactly
/// that mutation (their doc-comments record the evidence); the same order is
/// fenced at the integration level by
/// `assembler_produces_byte_identical_blobs_across_independent_assemblies` and
/// by `first_base_bootstrap_test`'s Case 2
/// (`bootstrap_base_is_semantically_identical_to_the_assembler_base_for_the_same_state`),
/// and at the source by `plan_full_content_is_deterministic_for_equal_state`.
/// The obligation is therefore to keep those tests green and NOT to delete them
/// as redundant ceremony — they are what makes this chain enforced rather than
/// merely asserted.
///
/// An empty `ordered_hashes` still yields a sidecar (`{"ext_hashes":[]}`), not
/// `None` — every base generation carries one, which is what makes the
/// adapter's publish-order obligation (b.s5y) apply uniformly.
///
/// # Errors
///
/// [`CheckpointError::Permanent`] if `serde_json` fails to serialize — the
/// input is a `Vec<String>` so this is unreachable in practice, and retrying
/// the same input could not fix it.
pub(crate) fn ext_refs_sidecar_json(
    ordered_hashes: &[[u8; 32]],
) -> Result<Vec<u8>, CheckpointError> {
    let sidecar = Sidecar {
        ext_hashes: ordered_hashes.iter().map(sha256_to_hex).collect(),
    };
    serde_json::to_vec(&sidecar).map_err(|e| CheckpointError::Permanent {
        source: format!("ext-refs sidecar JSON serialization failed: {e}").into(),
    })
}

/// Public assembler entrypoint — SR-2.3 steps 6-8.
///
/// Chains:
///
/// 1. [`load_consistent_snapshot`] over a [`ValkeySnapshotLoader`] built from
///    `client` + `keys` — the atomic read on the snapshot lane, and for a layer
///    above the atomic-read limits the fenced read (its fan-out on the snapshot
///    lane, its fence scripts on the data pool), with `lease` kept beside it;
///    never `txn_pool` or `maintenance_pool`.
///    A [`ConsistentSnapshot::NotQuiesced`] outcome short-circuits every step
///    below and returns [`AssemblyDecision::SkippedNotQuiesced`], and a
///    [`ConsistentSnapshot::LeaseLost`] one returns
///    [`AssemblyDecision::LeaseLost`], and a
///    [`ConsistentSnapshot::ShuttingDown`] one
///    [`AssemblyDecision::ShuttingDown`]: nothing is built and nothing is
///    persisted.
/// 2. [`build_reset_ompf`] over the filtered snapshot.
/// 3. `serde_json::to_vec` of the ext-refs [`Sidecar`] in first-appearance
///    order (matching `ExternalValueScan::ordered_hashes`).
/// 4. [`CheckpointStore::put_layer_checkpoint`] with the sidecar JSON threaded
///    through `CheckpointMeta::sidecar_bytes` so the adapter writes OMPF +
///    sidecar atomically (removes the partial-write failure mode that a
///    separate post-put `set_layer_sidecar` call would carry).
///
/// On a write emits a single `tracing::info!` carrying `object_id`, `snap_seq`,
/// the OMPF byte length, the ext-hash count, and the sidecar JSON byte length
/// (NOT the sidecar bytes themselves — operator logs MUST NOT carry the
/// hash payload). The deliberate-skip decision is WARNed once inside
/// [`load_consistent_snapshot`] and again by the periodic loop, which owns the
/// metric ticks.
///
/// On failure emits a single `tracing::warn!` with the same identifying fields
/// plus the classified error. Each inner step already emits its own
/// fine-grained tracing on the snapshot-read / OMPF-build paths; the warn here
/// is the call-site-level summary the leader-elected loop reads.
///
/// Every attempt, whatever its outcome, also emits one `tracing::debug!`
/// (`layer checkpoint timing`) with the elapsed time of the read, the
/// assembly and the write, for the checkpoint's validation measurements.
///
/// # `lease`
///
/// The leader lease the caller holds, kept beside the fenced read for as long
/// as it runs ([`LeaderLeaseKeeper`], Spec SDD SR-11.5): the periodic loop's
/// keeper refreshes its `CheckpointLease`; a caller outside the loop holds none
/// and passes a keeper that never loses. Not consulted on the atomic path,
/// whose one read is bounded by the atomic-read limits.
///
/// # Errors
///
/// - [`CheckpointError::NotFound`] — the object header is absent (caller drops
///   the object from its working set).
/// - [`CheckpointError::Transient`] — a backend failure on the reads or a
///   transient one on the put, a fenced read broken `FENCE_READ_ATTEMPTS`
///   times in a row, or a fence wait that would pass `reader_wait_max(L)`
///   (the layer is skipped this tick; nothing unproven is ever persisted).
/// - [`CheckpointError::Permanent`] — malformed snapshot (a corrupt section
///   index included; see [`load_consistent_snapshot`]), OMPF build failure,
///   sidecar JSON serialization failure (vanishingly unlikely in practice but
///   classified Permanent because retrying produces the same Err), or a
///   permanent adapter put failure.
/// - [`CheckpointError::Conflict`] — adapter surfaced an optimistic-concurrency
///   race on the put (LocalDisk's rename-into-place can produce this; InMemory
///   serializes via its mutex and does not).
pub async fn assemble_layer_checkpoint(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
    store: &dyn CheckpointStore,
    service_label: &str,
    lease: &mut dyn LeaderLeaseKeeper,
) -> Result<AssemblyDecision, CheckpointError> {
    // Per the subtask contract: one `tracing::info!` on success, one
    // `tracing::warn!` on failure. The inner async block computes the
    // assembly and surfaces every intermediate byte-length via the
    // `AssemblyProgress` running counter so the warn arm names them even
    // when an inner step failed before the value was final. Sidecar JSON
    // bytes themselves are NEVER logged — only their byte length — per the
    // operator-log discipline noted in the subtask body.
    let mut progress = AssemblyProgress::default();
    let started = Instant::now();
    let result = assemble_inner(
        client,
        keys,
        object_id,
        store,
        service_label,
        lease,
        &mut progress,
    )
    .await;
    // One event per layer checkpoint attempt, for the checkpoint's validation
    // measurements (per-size checkpoint time, split by stage). A stage the
    // attempt did not finish reads zero. Debug level: the stage clocks are
    // read on every attempt, but the event's fields are evaluated only when
    // the log filter enables debug for this module.
    tracing::debug!(
        object_id = object_id,
        outcome = match &result {
            Ok(AssemblyDecision::Written(_)) => "written",
            Ok(AssemblyDecision::SkippedNotQuiesced) => "skipped_not_quiesced",
            Ok(AssemblyDecision::LeaseLost) => "lease_lost",
            Ok(AssemblyDecision::ShuttingDown) => "shutdown",
            Err(CheckpointError::NotFound) => "not_found",
            Err(_) => "error",
        },
        snap_seq = progress.snap_seq,
        ompf_byte_len = progress.ompf_byte_len,
        read_us = micros(progress.read),
        assembly_us = micros(progress.assembly),
        write_us = micros(progress.write),
        total_us = micros(Some(started.elapsed())),
        "layer checkpoint timing"
    );
    match &result {
        Ok(AssemblyDecision::Written(outcome)) => {
            tracing::info!(
                object_id = object_id,
                snap_seq = outcome.snap_seq,
                // The band a DR restore of this blob would use when it exceeds
                // the configured margin (b.vnw); an operator sizing
                // `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN` reads it straight
                // off this line — together with the two inputs it was derived
                // from, so a `0` sentinel or an outsized reading can be
                // attributed (first generation / post-outage catch-up /
                // post-restore correction) without a second query.
                observed_commit_span = progress.observed_commit_span,
                last_checkpoint_seq = progress.last_checkpoint_seq,
                restore_seq_bump = progress.restore_seq_bump,
                ext_hash_count = outcome.ext_hash_count,
                ompf_byte_len = progress.ompf_byte_len,
                sidecar_byte_len = progress.sidecar_byte_len,
                adapter_key = outcome.handle.adapter_key.as_str(),
                "layer checkpoint assembled and persisted"
            );
        }
        // `load_consistent_snapshot` already WARNed with the pin and the cut;
        // a second line here would just duplicate it. The periodic loop emits
        // the operator-facing per-layer WARN plus the metric ticks.
        Ok(AssemblyDecision::SkippedNotQuiesced) => {}
        // The keeper that lost the lease already WARNed (the loop's
        // `CheckpointLease`, which also counts it), and the loop WARNs once
        // more for the layers it then leaves undone.
        Ok(AssemblyDecision::LeaseLost) => {
            tracing::debug!(
                object_id = object_id,
                "layer checkpoint assembly abandoned: the leader lease was lost during the \
                 fenced read; nothing persisted"
            );
        }
        // Shutdown is not a fault: DEBUG only, like the loop's own line.
        Ok(AssemblyDecision::ShuttingDown) => {
            tracing::debug!(
                object_id = object_id,
                "layer checkpoint assembly abandoned: shutting down during the fenced read; \
                 nothing persisted"
            );
        }
        // A vanished header is a routine DeleteLayer / cleanup / orphan-sweep
        // race, not an assembly failure — `debug!`, so the WARN arm below stays
        // a signal operators can act on. Matches the silent-skip posture the
        // periodic loop takes on the same race.
        Err(CheckpointError::NotFound) => {
            tracing::debug!(
                object_id = object_id,
                "layer checkpoint assembly skipped: object header vanished mid-assembly"
            );
        }
        Err(err) => {
            tracing::warn!(
                object_id = object_id,
                snap_seq = progress.snap_seq,
                ext_hash_count = progress.ext_hash_count,
                ompf_byte_len = progress.ompf_byte_len,
                sidecar_byte_len = progress.sidecar_byte_len,
                error = %err,
                "layer checkpoint assembly failed"
            );
        }
    }
    result
}

/// Running counter mutated by [`assemble_inner`] so the outer success / failure
/// tracing arms can name every intermediate byte length the assembler computed
/// before the call returned. Each field is `0` until the corresponding step
/// lands a value; on failure paths the unset fields stay `0`, which is the
/// natural "no value computed" sentinel.
#[derive(Debug, Default)]
struct AssemblyProgress {
    snap_seq: u64,
    /// Commits this generation covers that the previous one did not, after the
    /// restore-bump correction (b.vnw). `0` until the header read lands, and
    /// legitimately `0` afterwards whenever
    /// [`measure_observed_commit_span`] declined to measure or the correction
    /// consumed the whole raw difference.
    observed_commit_span: u64,
    /// `obj:hdr last_checkpoint_seq` as read at assembly time — the first input
    /// to the span above, surfaced on the success log so an operator can tell a
    /// genuinely hot layer from a first-generation (`0`) or post-outage (far
    /// behind `snap_seq`) artifact without re-deriving it.
    last_checkpoint_seq: u64,
    /// `obj:hdr restore_seq_bump` as read at assembly time — the second input,
    /// surfaced for the same reason: a non-zero value means this line's span is
    /// the one assembly per restore that pays the correction.
    restore_seq_bump: u64,
    ext_hash_count: usize,
    ompf_byte_len: usize,
    sidecar_byte_len: usize,
    /// Elapsed time of each stage the attempt finished, for the timing event
    /// in [`assemble_layer_checkpoint`]; `None` for a stage it did not finish.
    /// `read`: the consistent snapshot ([`load_consistent_snapshot`], every
    /// fence wait and the fenced read included). `assembly`: the reset OMPF,
    /// the ext-refs sidecar and the header-identity read. `write`: the
    /// pre-PUT liveness probe and the adapter's put.
    read: Option<Duration>,
    assembly: Option<Duration>,
    write: Option<Duration>,
}

/// Whole microseconds of `elapsed`, zero for `None` (a stage not finished).
fn micros(elapsed: Option<Duration>) -> u64 {
    elapsed.map_or(0, |d| u64::try_from(d.as_micros()).unwrap_or(u64::MAX))
}

/// Inner implementation that does the actual work; split out so the outer
/// function's tracing wrap is a single match without nesting per inner step.
async fn assemble_inner(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
    store: &dyn CheckpointStore,
    service_label: &str,
    lease: &mut dyn LeaderLeaseKeeper,
    progress: &mut AssemblyProgress,
) -> Result<AssemblyDecision, CheckpointError> {
    let loader = ValkeySnapshotLoader { client, keys };
    let object_id_str = object_id.to_string();

    // Steps 1-5 of SR-2.3: a proven consistent snapshot (atomic, or fenced
    // above the atomic-read limits). A regressed cut or a lost lease aborts
    // here — no build, no put, no cursor.
    let stage = Instant::now();
    let consistent = load_consistent_snapshot(&loader, &object_id_str, lease).await;
    progress.read = Some(stage.elapsed());
    let (snapshot, snap_seq) = match consistent? {
        ConsistentSnapshot::Verified { snapshot, snap_seq } => (snapshot, snap_seq),
        ConsistentSnapshot::NotQuiesced => return Ok(AssemblyDecision::SkippedNotQuiesced),
        ConsistentSnapshot::LeaseLost => return Ok(AssemblyDecision::LeaseLost),
        ConsistentSnapshot::ShuttingDown => return Ok(AssemblyDecision::ShuttingDown),
    };
    progress.snap_seq = snap_seq;
    let stage = Instant::now();

    // Step 5: reset-delta + OMPF wrap + ext-hash ordering.
    let (ompf_bytes, ordered_hashes) = build_reset_ompf(&snapshot)?;
    progress.ompf_byte_len = ompf_bytes.len();
    progress.ext_hash_count = ordered_hashes.len();

    // Step 6: ext-refs sidecar JSON — first-appearance hash order, lowercase
    // hex, byte-for-byte deterministic across replicas (SR-3.4) because
    // `plan_full_content` emits fields and time samples in a total order and the
    // hash scan walks that order without resorting. Chain spelled out on
    // `ext_refs_sidecar_json`.
    let sidecar_bytes = ext_refs_sidecar_json(&ordered_hashes)?;
    progress.sidecar_byte_len = sidecar_bytes.len();

    // Step 6.5: read the session-level identity fields off `obj:hdr` so
    // `LayerStore::apply_checkpoint` can rebuild the header without inventing
    // session metadata. The fields are NOT in the OODF reset-delta payload —
    // they're session-level, not document state. Note this read is
    // deliberately NOT covered by the stable cut: it carries session-level
    // identity, not document state, so a value from a later commit boundary
    // cannot tear the snapshot.
    //
    // A missing `obj:hdr` here is the same transient race `read_header_seq`
    // already guards against (DeleteLayer, `cleanup_old_object_retrying` after
    // a CreateLayer overwrite, orphan sweep). It used to fold to identity-empty
    // defaults and write the blob anyway; that is now refused. Since the
    // identity guard on the apply path (`apply.rs`) rejects a generation with
    // an empty `session_id`/`layer_name`, such a blob is *permanently
    // unrestorable* — writing it spends a PUT, consumes a retention slot, and
    // leaves a generation on the stem that can only ever fail a restore.
    // Declining is strictly better, and `NotFound` is the right shape: it is
    // the arm the checkpoint loop already treats as benign and terminal
    // (silent `debug!`, no cursor stamp, no metric tick — `checkpointer.rs`,
    // the `Err(CheckpointError::NotFound)` arm), which is exactly the truth
    // here — the object is gone and will never be enumerated again.
    // `tail_capture`'s post-restore fossil guard routes the identical
    // condition to the identical arm, and names this fold as the thing it
    // avoids.
    //
    // ONE check, and it covers both refusals: `read_header_identity` reports
    // `None` for an absent header AND for a header whose `session_id` /
    // `layer_name` is empty (a partially written or hand-repaired one). The
    // empty case used to be a second `if` here, and being local to this producer
    // is precisely what made it a defect — the tail-capture and bootstrap
    // producers had no such check, so a degenerate header reached
    // `validate_generation_identity` and ticked an adapter-health error every
    // tick. Refusing at the shared read is what makes "no generation of any kind
    // ever carries empty identity" structural rather than merely usual (b.hka
    // review round 2).
    let Some(identity) = loader.read_header_identity(&object_id_str).await? else {
        return Err(CheckpointError::NotFound);
    };
    let observed_commit_span = measure_observed_commit_span(snap_seq, &identity);
    progress.assembly = Some(stage.elapsed());

    progress.observed_commit_span = observed_commit_span;
    progress.last_checkpoint_seq = identity.last_checkpoint_seq;
    progress.restore_seq_bump = identity.restore_seq_bump;

    // Step 7: put_layer_checkpoint with the sidecar threaded through meta so
    // the adapter writes OMPF + sidecar atomically. The wall-clock stamp
    // uses `SystemTime::now()`; a clock that is pre-epoch (impossible in
    // practice for any sane container) folds down to 0 rather than panicking.
    let written_at_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let meta = CheckpointMeta {
        creator_replica_id: service_label.to_string(),
        written_at_unix_ms,
        snap_seq: Some(snap_seq),
        session_id: identity.session_id,
        layer_name: identity.layer_name,
        created_by: identity.created_by,
        server_ts: identity.server_ts,
        observed_commit_span,
        sidecar_bytes: Some(Bytes::from(sidecar_bytes)),
        ..CheckpointMeta::default()
    };

    // Pre-PUT liveness (b.hka): the layer can be deleted while the snapshot
    // read, the OMPF build and the sidecar serialization above are in flight,
    // and the resulting artifact would be durable garbage nobody ever reads.
    // Shared with the other three durable-generation producers; the residual
    // (it cannot catch a delete that lands after the probe) and why the b.vn6
    // conditional cursor stamp remains the real guard are on
    // `crate::checkpoint::put_guard`.
    //
    // `Ok(None)` folds onto `NotFound`, which is the same benign-and-terminal
    // arm this function's identity read above uses and the checkpoint loop
    // already handles: skipped at `debug!`, no metric tick, no cursor stamp.
    // The condition IS "the object header vanished", so it is the honest
    // spelling rather than a new outcome.
    let stage = Instant::now();
    let put = crate::checkpoint::put_guard::put_layer_checkpoint_if_live(
        Some(crate::checkpoint::put_guard::ObjectLivenessProbe::new(
            client, keys,
        )),
        store,
        object_id,
        snap_seq,
        Bytes::from(ompf_bytes),
        meta,
        "assembler",
    )
    .await;
    progress.write = Some(stage.elapsed());
    let Some(handle) = put? else {
        return Err(CheckpointError::NotFound);
    };

    Ok(AssemblyDecision::Written(AssemblerOutcome {
        handle,
        snap_seq,
        ext_hash_count: progress.ext_hash_count,
    }))
}

#[cfg(test)]
mod tests {
    //! Unit tests for the assembler's two correctness primitives that do not
    //! require Valkey: the per-entry version filter, and the
    //! consistent-snapshot decision flow — pin → one atomic read → verified /
    //! regressed / corrupt index (Spec SDD SR-4.1), or, when the atomic read
    //! answers `TOO_LARGE`, one fenced read beside the leader-lease keeper →
    //! verified / regressed / corrupt index / failed / lease lost / shutting
    //! down (Spec SDD SR-11.5; the keeper's two stops, [`KeeperStop`]).
    //!
    //! All decision-flow tests inject a hand-built `FakeLoader` against the
    //! [`SnapshotLoader`] trait (no `TestValkey`, no multi-thread runtime —
    //! tests use the default single-threaded `#[tokio::test]` per the
    //! Proptest Isolation rule in CLAUDE.md). The fake answers the atomic read
    //! `TOO_LARGE` unless a test queues an atomic answer, and the fenced read
    //! only from its own queue, so each test says which path it is on.
    //!
    //! The production loader's own behaviour — the header-identity contract and
    //! the test-only unprovable-cut seam on its pin read — is driven against
    //! Valkey at the bottom of the module.
    //!
    //! The b.xje block at the end pins sibling order through the base's
    //! encode → decode round trip, also without Valkey.
    //!
    //! Naming follows `<unit>_<scenario>_<expected>`.
    use super::*;
    use crate::domain::ompf::OmpfContainer;
    use crate::domain::types::{Field, FieldValue, Section, TimeSample};
    use ordered_float::OrderedFloat;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    // ── Helpers ─────────────────────────────────────────────────────────────

    /// Build an empty `ObjectSnapshot` with sane defaults that tests can mutate.
    fn empty_snapshot() -> ObjectSnapshot {
        ObjectSnapshot {
            object_id: 1,
            sequence_num: 0,
            first_version: 0,
            source_format: String::new(),
            next_set_order: 0,
            sections: HashMap::new(),
            children: HashMap::new(),
            fields: Vec::new(),
            time_samples: Vec::new(),
            tombstones: Vec::new(),
        }
    }

    fn make_section(id: u64, last_version: u64) -> Section {
        Section {
            section_id: id,
            parent_id: 0,
            name: format!("s{id}"),
            section_type: 0,
            section_order: 0,
            first_version: 0,
            last_version,
        }
    }

    fn make_field(section_id: u64, key: &str, version: u64) -> Field {
        Field {
            section_id,
            key_name: key.to_string(),
            value: FieldValue::Inline(Vec::new()),
            set_order: 0,
            authoring_set_order: 0,
            version,
        }
    }

    fn make_time_sample(section_id: u64, time: f64, version: u64) -> TimeSample {
        TimeSample {
            section_id,
            time: OrderedFloat(time),
            value: FieldValue::Inline(Vec::new()),
            version,
        }
    }

    /// The atomic read's `TOO_LARGE` answer: the layer is above the
    /// atomic-read limits. The fake's default; its sizing fields are zero,
    /// since only the test that pins the fenced read's estimate reads them, and
    /// it builds its own answers with [`too_large_answer`].
    fn too_large() -> SnapshotRead {
        SnapshotRead::TooLarge(crate::storage::diff_read::SnapshotTooLarge {
            cardinalities: crate::storage::diff_read::SnapshotCardinalities {
                sections: 0,
                tombstones: 0,
                fields: 0,
                time_samples: 0,
                children_index: None,
            },
            item_bound: 0,
            at: SnapshotTooLargeAt::EarlyOut,
        })
    }

    /// The atomic read's answer for a layer at or below the limits: `state` at
    /// `cut`, with `corrupt` as the `obj:secs` members that named no readable
    /// section. `state.sequence_num` is set to `cut`, as the decoder does.
    fn atomic_read(mut state: ObjectSnapshot, cut: u64, corrupt: &[&str]) -> SnapshotRead {
        state.sequence_num = cut;
        SnapshotRead::Snapshot(Box::new(AtomicSnapshot {
            state,
            cut,
            last_server_ts: 0,
            last_client_ts: 0,
            reply_bytes: 0,
            item_count: 0,
            entry_count: 0,
            path: crate::storage::diff_read::SnapshotReadPath::Atomic,
            corrupt_section_index_members: corrupt.iter().map(|m| m.to_string()).collect(),
            structural: None,
            memory: Default::default(),
            value_row_bytes_received: 0,
        }))
    }

    /// A `TOO_LARGE` answer with `item_bound` and `at`: the layer is above the
    /// atomic-read limits. The assembler sizes its fenced read from it
    /// (`LayerSize::from_too_large`).
    fn too_large_answer(
        item_bound: u64,
        at: SnapshotTooLargeAt,
    ) -> crate::storage::diff_read::SnapshotTooLarge {
        crate::storage::diff_read::SnapshotTooLarge {
            cardinalities: crate::storage::diff_read::SnapshotCardinalities {
                sections: 0,
                tombstones: 0,
                fields: 0,
                time_samples: 0,
                children_index: None,
            },
            item_bound,
            at,
        }
    }

    /// The fenced read's verified state: `state` at `cut`, on
    /// `SnapshotReadPath::Fenced`, with `corrupt` as the `obj:secs` members
    /// that named no readable section. `state.sequence_num` is set to `cut`.
    fn fenced_read(mut state: ObjectSnapshot, cut: u64, corrupt: &[&str]) -> AtomicSnapshot {
        state.sequence_num = cut;
        AtomicSnapshot {
            state,
            cut,
            last_server_ts: 0,
            last_client_ts: 0,
            reply_bytes: 0,
            item_count: 0,
            entry_count: 0,
            path: crate::storage::diff_read::SnapshotReadPath::Fenced,
            corrupt_section_index_members: corrupt.iter().map(|m| m.to_string()).collect(),
            structural: None,
            memory: Default::default(),
            value_row_bytes_received: 0,
        }
    }

    /// A snapshot whose live sections are exactly `ids`, each stamped at
    /// `version`.
    fn snapshot_with_sections(ids: &[u64], version: u64) -> ObjectSnapshot {
        let mut snap = empty_snapshot();
        for id in ids {
            snap.sections.insert(*id, make_section(*id, version));
        }
        snap
    }

    /// A snapshot read at cut 100 that carries one entry of each kind at 50
    /// and one at 200 — above the cut — for the subordinate version filter to
    /// drop.
    fn snapshot_straddling_the_cut() -> ObjectSnapshot {
        let mut snap = empty_snapshot();
        snap.sections.insert(1, make_section(1, 50));
        snap.sections.insert(2, make_section(2, 200));
        snap.fields.push(make_field(1, "a", 50));
        snap.fields.push(make_field(1, "b", 200));
        snap.time_samples.push(make_time_sample(1, 0.0, 50));
        snap.time_samples.push(make_time_sample(1, 1.0, 200));
        snap.tombstones.push((5, 1, 50));
        snap.tombstones.push((6, 1, 200));
        snap
    }

    /// The straddling snapshot after the version filter at cut 100: only the
    /// entries at 50 survive.
    fn assert_only_entries_at_or_below_the_cut_survive(snapshot: &ObjectSnapshot) {
        assert_eq!(
            snapshot.sections.keys().copied().collect::<Vec<_>>(),
            vec![1]
        );
        assert_eq!(snapshot.fields.len(), 1);
        assert_eq!(snapshot.fields[0].version, 50);
        assert_eq!(snapshot.time_samples.len(), 1);
        assert_eq!(snapshot.time_samples[0].version, 50);
        assert_eq!(snapshot.tombstones, vec![(5, 1, 50)]);
    }

    /// What the fake's fenced read does when the assembler calls it.
    enum FencedAnswer {
        /// Answer at once.
        Ready(Result<AtomicSnapshot, CheckpointError>),
        /// Signal `fenced_started`, then wait for the `Notify` and answer —
        /// a read that becomes ready on an outside event.
        ReadyWhen(
            Arc<tokio::sync::Notify>,
            Result<AtomicSnapshot, CheckpointError>,
        ),
        /// Signal `fenced_started` and never answer — a read still under its
        /// fence — recording in `fenced_dropped` when the assembler drops it.
        HangUnderTheFence,
    }

    /// Sets its flag when dropped: the fake's stand-in for the fence lease
    /// guard whose drop releases the fence.
    struct DropFlag(Arc<std::sync::atomic::AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Hand-rolled `SnapshotLoader` fake — canned answers for the pin read, the
    /// atomic read and the fenced read, and a count of each call. The intent
    /// is to let each test state the reads it wants to simulate without
    /// spinning up Valkey.
    ///
    /// # Every read is canned, not modelled
    ///
    /// Both state reads prove their own cut — the atomic read by atomicity,
    /// the fenced read by its fence and its release — so there is no timing
    /// relationship left for a fake to model: a test states the cut a read
    /// returned and the pin it is judged against, and asserts what the
    /// assembler decides. The call counts are what make "one read, no retry,
    /// no re-pin" assertions rather than readings of the code.
    ///
    /// An exhausted atomic queue answers `TOO_LARGE` (a layer above the
    /// atomic-read limits); an exhausted fenced queue answers a `Permanent`
    /// "test bug" error, so an under-stubbed test fails loudly rather than
    /// passing on a default.
    ///
    /// # The fenced read can be held under its fence
    ///
    /// [`FencedAnswer::HangUnderTheFence`] and [`FencedAnswer::ReadyWhen`] let a
    /// lease test lose the lease while the read is still running and observe
    /// whether the assembler dropped it — the drop being what releases the
    /// fence in production.
    struct FakeLoader {
        /// Canned `read_atomic_snapshot` answers, one per call, in order. An
        /// exhausted queue answers `TOO_LARGE` (see the struct doc-comment).
        atomic_answers: Mutex<Vec<Result<SnapshotRead, CheckpointError>>>,
        atomic_calls: Mutex<u32>,
        /// Canned `read_fenced_snapshot` answers, one per call, in order.
        fenced_answers: Mutex<Vec<FencedAnswer>>,
        /// The `estimate` each fenced read was handed, in call order.
        estimates_seen: Mutex<Vec<LayerSize>>,
        /// Notified (one permit) when a held or deferred fenced read starts.
        fenced_started: Arc<tokio::sync::Notify>,
        /// Set when a [`FencedAnswer::HangUnderTheFence`] read is dropped.
        fenced_dropped: Arc<std::sync::atomic::AtomicBool>,
        /// `None` models an absent `obj:hdr` — the pin read returns `None` and
        /// the assembler must classify `NotFound` without retrying.
        header_seq: Option<u64>,
        /// Canned `read_header_seq` answers, consumed in call order. An
        /// exhausted queue falls back to [`Self::header_seq`], so a test that
        /// only cares about a steady header says nothing about this. A test
        /// queues a second answer to show it is never asked for.
        header_seq_answers: Mutex<Vec<Option<u64>>>,
        header_calls: Mutex<u32>,
    }

    impl FakeLoader {
        fn new() -> Self {
            Self {
                atomic_answers: Mutex::new(Vec::new()),
                atomic_calls: Mutex::new(0),
                fenced_answers: Mutex::new(Vec::new()),
                estimates_seen: Mutex::new(Vec::new()),
                fenced_started: Arc::new(tokio::sync::Notify::new()),
                fenced_dropped: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                header_seq: Some(100),
                header_seq_answers: Mutex::new(Vec::new()),
                header_calls: Mutex::new(0),
            }
        }

        fn with_snap_seq(mut self, snap_seq: u64) -> Self {
            self.header_seq = Some(snap_seq);
            self
        }

        /// Model an absent `obj:hdr` at pin time.
        fn without_header(mut self) -> Self {
            self.header_seq = None;
            self
        }

        /// Canned answers for successive `read_header_seq` calls. See the
        /// field doc-comment.
        fn with_header_seq_answers(self, answers: Vec<Option<u64>>) -> Self {
            let mut rev = answers;
            rev.reverse();
            *self.header_seq_answers.lock().unwrap() = rev;
            self
        }

        /// Canned answers for successive `read_atomic_snapshot` calls. See the
        /// field doc-comment; an exhausted queue answers `TOO_LARGE`.
        fn with_atomic_answers(self, answers: Vec<Result<SnapshotRead, CheckpointError>>) -> Self {
            let mut rev = answers;
            rev.reverse();
            *self.atomic_answers.lock().unwrap() = rev;
            self
        }

        /// Canned answers for successive `read_fenced_snapshot` calls.
        fn with_fenced_answers(self, answers: Vec<FencedAnswer>) -> Self {
            let mut rev = answers;
            rev.reverse();
            *self.fenced_answers.lock().unwrap() = rev;
            self
        }

        /// How many atomic reads the assembler issued.
        fn atomic_call_count(&self) -> u32 {
            *self.atomic_calls.lock().unwrap()
        }

        /// How many fenced reads the assembler started.
        fn fenced_call_count(&self) -> usize {
            self.estimates_seen.lock().unwrap().len()
        }

        /// The size each fenced read was handed, in call order.
        fn estimates_seen(&self) -> Vec<LayerSize> {
            self.estimates_seen.lock().unwrap().clone()
        }

        /// Whether a held fenced read was dropped.
        fn fenced_dropped(&self) -> bool {
            self.fenced_dropped
                .load(std::sync::atomic::Ordering::SeqCst)
        }

        /// How many times the assembler read `obj:hdr seq`: `1` is the pin and
        /// nothing else, on either path.
        fn header_call_count(&self) -> u32 {
            *self.header_calls.lock().unwrap()
        }
    }

    #[async_trait]
    impl SnapshotLoader for FakeLoader {
        async fn read_atomic_snapshot(
            &self,
            _object_id: &str,
        ) -> Result<SnapshotRead, CheckpointError> {
            *self.atomic_calls.lock().unwrap() += 1;
            // Unless a test queued an answer, the layer is above the
            // atomic-read limits, so the assembler routes to the fenced read.
            self.atomic_answers
                .lock()
                .unwrap()
                .pop()
                .unwrap_or_else(|| Ok(too_large()))
        }

        async fn read_fenced_snapshot(
            &self,
            _object_id: &str,
            estimate: LayerSize,
        ) -> Result<AtomicSnapshot, CheckpointError> {
            self.estimates_seen.lock().unwrap().push(estimate);
            let answer = self.fenced_answers.lock().unwrap().pop();
            match answer {
                Some(FencedAnswer::Ready(result)) => result,
                Some(FencedAnswer::ReadyWhen(ready, result)) => {
                    self.fenced_started.notify_one();
                    ready.notified().await;
                    result
                }
                Some(FencedAnswer::HangUnderTheFence) => {
                    let _fence = DropFlag(Arc::clone(&self.fenced_dropped));
                    self.fenced_started.notify_one();
                    std::future::pending::<Result<AtomicSnapshot, CheckpointError>>().await
                }
                None => Err(CheckpointError::Permanent {
                    source: "test bug: read_fenced_snapshot called more times than canned answers"
                        .into(),
                }),
            }
        }

        async fn read_header_seq(&self, _object_id: &str) -> Result<Option<u64>, CheckpointError> {
            *self.header_calls.lock().unwrap() += 1;
            let queued = self.header_seq_answers.lock().unwrap().pop();
            Ok(queued.unwrap_or(self.header_seq))
        }

        async fn read_header_identity(
            &self,
            _object_id: &str,
        ) -> Result<Option<HeaderIdentity>, CheckpointError> {
            // Default to an empty identity — the unit tests in this module
            // exercise the read / filter / version paths, not the meta
            // population. The end-to-end identity round-trip is the
            // integration test's job (Subtask `.xw`).
            Ok(Some(HeaderIdentity::default()))
        }
    }

    // ── Version-filter correctness tests ─────────────────────────────────────

    #[test]
    fn filter_snapshot_by_snap_seq_sections_drops_only_future_writes() {
        let mut snap = empty_snapshot();
        snap.sections.insert(1, make_section(1, 1));
        snap.sections.insert(2, make_section(2, 5));
        snap.sections.insert(3, make_section(3, 10));
        snap.sections.insert(4, make_section(4, 15));

        filter_snapshot_by_snap_seq(&mut snap, 10);

        assert_eq!(
            snap.sections.len(),
            3,
            "expected sections at 1, 5, 10 retained"
        );
        assert!(snap.sections.contains_key(&1));
        assert!(snap.sections.contains_key(&2));
        assert!(snap.sections.contains_key(&3));
        assert!(
            !snap.sections.contains_key(&4),
            "section with last_version=15 must be dropped at snap_seq=10"
        );
    }

    #[test]
    fn filter_snapshot_by_snap_seq_sections_snap_seq_zero_drops_all_nonzero() {
        let mut snap = empty_snapshot();
        snap.sections.insert(1, make_section(1, 1));
        snap.sections.insert(2, make_section(2, 5));
        // A section at last_version=0 is retained (boundary: 0 <= 0).
        snap.sections.insert(3, make_section(3, 0));

        filter_snapshot_by_snap_seq(&mut snap, 0);

        assert_eq!(snap.sections.len(), 1);
        assert!(snap.sections.contains_key(&3));
    }

    #[test]
    fn filter_snapshot_by_snap_seq_sections_snap_seq_u64_max_retains_all() {
        let mut snap = empty_snapshot();
        snap.sections.insert(1, make_section(1, 1));
        snap.sections.insert(2, make_section(2, u64::MAX));

        filter_snapshot_by_snap_seq(&mut snap, u64::MAX);

        assert_eq!(snap.sections.len(), 2);
    }

    #[test]
    fn filter_snapshot_by_snap_seq_empty_snapshot_is_noop() {
        let mut snap = empty_snapshot();
        filter_snapshot_by_snap_seq(&mut snap, 10);
        assert!(snap.sections.is_empty());
        assert!(snap.fields.is_empty());
        assert!(snap.time_samples.is_empty());
        assert!(snap.tombstones.is_empty());
    }

    #[test]
    fn filter_snapshot_by_snap_seq_fields_drops_only_future_writes() {
        let mut snap = empty_snapshot();
        snap.fields.push(make_field(1, "a", 1));
        snap.fields.push(make_field(1, "b", 5));
        snap.fields.push(make_field(1, "c", 10));
        snap.fields.push(make_field(1, "d", 15));

        filter_snapshot_by_snap_seq(&mut snap, 10);

        assert_eq!(snap.fields.len(), 3);
        let versions: Vec<u64> = snap.fields.iter().map(|f| f.version).collect();
        assert!(versions.contains(&1));
        assert!(versions.contains(&5));
        assert!(versions.contains(&10));
        assert!(!versions.contains(&15));
    }

    #[test]
    fn filter_snapshot_by_snap_seq_fields_boundary_cases_match_spec() {
        // snap_seq = 0 — only version 0 entries (none here) retained.
        let mut snap = empty_snapshot();
        snap.fields.push(make_field(1, "a", 1));
        snap.fields.push(make_field(1, "b", 5));
        filter_snapshot_by_snap_seq(&mut snap, 0);
        assert!(snap.fields.is_empty());

        // snap_seq = u64::MAX — every entry retained.
        let mut snap = empty_snapshot();
        snap.fields.push(make_field(1, "a", 1));
        snap.fields.push(make_field(1, "b", u64::MAX));
        filter_snapshot_by_snap_seq(&mut snap, u64::MAX);
        assert_eq!(snap.fields.len(), 2);
    }

    #[test]
    fn filter_snapshot_by_snap_seq_time_samples_drops_only_future_writes() {
        let mut snap = empty_snapshot();
        snap.time_samples.push(make_time_sample(1, 0.0, 1));
        snap.time_samples.push(make_time_sample(1, 1.0, 5));
        snap.time_samples.push(make_time_sample(1, 2.0, 10));
        snap.time_samples.push(make_time_sample(1, 3.0, 15));

        filter_snapshot_by_snap_seq(&mut snap, 10);

        assert_eq!(snap.time_samples.len(), 3);
        let versions: Vec<u64> = snap.time_samples.iter().map(|ts| ts.version).collect();
        assert!(versions.contains(&1));
        assert!(versions.contains(&5));
        assert!(versions.contains(&10));
        assert!(!versions.contains(&15));
    }

    #[test]
    fn filter_snapshot_by_snap_seq_time_samples_boundary_cases_match_spec() {
        let mut snap = empty_snapshot();
        snap.time_samples.push(make_time_sample(1, 0.0, 1));
        snap.time_samples.push(make_time_sample(1, 1.0, 5));
        filter_snapshot_by_snap_seq(&mut snap, 0);
        assert!(snap.time_samples.is_empty());

        let mut snap = empty_snapshot();
        snap.time_samples.push(make_time_sample(1, 0.0, 1));
        snap.time_samples.push(make_time_sample(1, 1.0, u64::MAX));
        filter_snapshot_by_snap_seq(&mut snap, u64::MAX);
        assert_eq!(snap.time_samples.len(), 2);
    }

    #[test]
    fn filter_snapshot_by_snap_seq_tombstones_drops_only_future_writes() {
        let mut snap = empty_snapshot();
        // Tombstone triples are `(section_id, first_version, last_version)`.
        snap.tombstones.push((1, 1, 2));
        snap.tombstones.push((2, 5, 8));
        snap.tombstones.push((3, 10, 12));

        filter_snapshot_by_snap_seq(&mut snap, 10);

        assert_eq!(snap.tombstones.len(), 2);
        let ids: Vec<u64> = snap.tombstones.iter().map(|&(s, _, _)| s).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(
            !ids.contains(&3),
            "tombstone with last_ver=12 must be dropped at snap_seq=10"
        );
    }

    #[test]
    fn filter_snapshot_by_snap_seq_tombstones_boundary_cases_match_spec() {
        // snap_seq = 0 — only tombstones with last_ver = 0 retained.
        let mut snap = empty_snapshot();
        snap.tombstones.push((1, 0, 0));
        snap.tombstones.push((2, 0, 5));
        filter_snapshot_by_snap_seq(&mut snap, 0);
        assert_eq!(snap.tombstones.len(), 1);
        assert_eq!(snap.tombstones[0].0, 1);

        // snap_seq = u64::MAX — none dropped.
        let mut snap = empty_snapshot();
        snap.tombstones.push((1, 0, 1));
        snap.tombstones.push((2, 0, u64::MAX));
        filter_snapshot_by_snap_seq(&mut snap, u64::MAX);
        assert_eq!(snap.tombstones.len(), 2);
    }

    // ── Consistent-snapshot flow tests ───────────────────────────────────────
    //
    // Every state the assembler persists comes from ONE proven read — the
    // atomic read at or below the atomic-read limits, the fenced read above
    // them (Spec SDD SR-4.1, SR-11.5) — judged against the header sequence
    // pinned before it. The fake answers the atomic read `TOO_LARGE` unless a
    // test queues an atomic answer, and answers the fenced read only from its
    // own queue, so each test states which path it is on.

    /// Unwrap the `Verified` arm. `ObjectSnapshot` does not impl `Debug`, so
    /// the match shape stands in for `expect`.
    fn expect_verified(
        result: Result<ConsistentSnapshot, CheckpointError>,
    ) -> (ObjectSnapshot, u64) {
        match result {
            Ok(ConsistentSnapshot::Verified {
                snapshot, snap_seq, ..
            }) => (snapshot, snap_seq),
            Ok(ConsistentSnapshot::NotQuiesced) => {
                panic!("expected a verified cut; got NotQuiesced")
            }
            Ok(ConsistentSnapshot::LeaseLost) => {
                panic!("expected a verified cut; got LeaseLost")
            }
            Ok(ConsistentSnapshot::ShuttingDown) => {
                panic!("expected a verified cut; got ShuttingDown")
            }
            Err(e) => panic!("expected a verified cut; got {e:?}"),
        }
    }

    /// A keeper whose lease is already gone: `hold_until_lost` returns at once.
    struct LeaseAlreadyLost;

    #[async_trait]
    impl LeaderLeaseKeeper for LeaseAlreadyLost {
        async fn hold_until_lost(&mut self) -> KeeperStop {
            KeeperStop::LeaseLost
        }
    }

    /// A keeper that loses its lease once `lost` is notified.
    struct LoseLeaseWhen(Arc<tokio::sync::Notify>);

    #[async_trait]
    impl LeaderLeaseKeeper for LoseLeaseWhen {
        async fn hold_until_lost(&mut self) -> KeeperStop {
            self.0.notified().await;
            KeeperStop::LeaseLost
        }
    }

    /// A keeper whose owner shuts down once `stop` is notified.
    struct ShutDownWhen(Arc<tokio::sync::Notify>);

    #[async_trait]
    impl LeaderLeaseKeeper for ShutDownWhen {
        async fn hold_until_lost(&mut self) -> KeeperStop {
            self.0.notified().await;
            KeeperStop::ShuttingDown
        }
    }

    /// An absent `obj:hdr` at pin time is `NotFound` immediately — no state read
    /// is attempted, atomic or fenced, and no retry (a vanished header does not
    /// come back).
    #[tokio::test]
    async fn load_consistent_snapshot_absent_header_is_not_found_without_state_read() {
        let loader =
            FakeLoader::new()
                .without_header()
                .with_fenced_answers(vec![FencedAnswer::Ready(Ok(fenced_read(
                    empty_snapshot(),
                    10,
                    &[],
                )))]);

        match load_consistent_snapshot(&loader, "obj-1", &mut NoLeaseToKeep).await {
            Err(CheckpointError::NotFound) => {}
            Err(other) => panic!("expected NotFound, got {other:?}"),
            Ok(_) => panic!("expected NotFound for an absent obj:hdr"),
        }
        assert_eq!(
            loader.atomic_call_count(),
            0,
            "an absent header must short-circuit before the atomic read"
        );
        assert_eq!(
            loader.fenced_call_count(),
            0,
            "an absent header must short-circuit before the fenced read"
        );
    }

    // ── Spec SDD SR-4.1: the atomic path — pin, read once, judge the cut ─────
    //
    // A layer at or below the atomic-read limits is read in ONE atomic read
    // that proves its own cut. The pin taken before it serves only the
    // regression check. These tests pin the four answers that path can give —
    // verified at the cut, regressed, corrupt index, failed — and that none of
    // them reaches the fenced read or re-reads the header.

    /// Assert the atomic path ran alone: exactly one atomic read, one header
    /// read (the pin), and no fenced read.
    fn assert_atomic_path_only(loader: &FakeLoader, context: &str) {
        assert_eq!(
            loader.atomic_call_count(),
            1,
            "{context}: exactly one atomic read — there is no retry on this path"
        );
        assert_eq!(
            loader.header_call_count(),
            1,
            "{context}: the pin is the only header read — no re-pin and no confirmation probe"
        );
        assert_eq!(
            loader.fenced_call_count(),
            0,
            "{context}: only a TOO_LARGE answer routes to the fenced read"
        );
    }

    /// **The single-read contract.** Pin 10; commits 11-13 land between the
    /// pin and the read, and the atomic read returns its cut at 13. That cut is
    /// proven by the read's atomicity, so it is `Verified` at `snap_seq = 13`
    /// in one read, which is why a hot layer at or below the limits is written
    /// every eligible tick.
    #[tokio::test]
    async fn load_consistent_snapshot_atomic_cut_above_the_pin_verifies_at_the_cut_in_one_read() {
        let loader = FakeLoader::new()
            .with_snap_seq(10)
            .with_atomic_answers(vec![Ok(atomic_read(
                snapshot_with_sections(&[7, 42], 12),
                13,
                &[],
            ))]);

        match load_consistent_snapshot(&loader, "obj-1", &mut NoLeaseToKeep).await {
            Ok(ConsistentSnapshot::Verified { snapshot, snap_seq }) => {
                assert_eq!(
                    snap_seq, 13,
                    "the proven cut is the atomic read's cut, not the pin"
                );
                assert_eq!(
                    snapshot.sections.len(),
                    2,
                    "the state read at the cut is returned"
                );
            }
            Ok(ConsistentSnapshot::NotQuiesced) => panic!(
                "a cut at or above the pin is proven by atomicity however hot the layer is; got \
                 NotQuiesced"
            ),
            Ok(ConsistentSnapshot::LeaseLost) => {
                panic!("the keeper never loses its lease here; got LeaseLost")
            }
            Ok(ConsistentSnapshot::ShuttingDown) => {
                panic!("the keeper never loses its lease here; got ShuttingDown")
            }
            Err(e) => panic!("expected a verified cut; got {e:?}"),
        }
        assert_atomic_path_only(&loader, "cut above the pin");
    }

    /// The boundary of `cut >= pin`: a cut EQUAL to the pin (nothing committed
    /// between the two reads) is verified too.
    #[tokio::test]
    async fn load_consistent_snapshot_atomic_cut_at_the_pin_verifies() {
        let loader = FakeLoader::new()
            .with_snap_seq(10)
            .with_atomic_answers(vec![Ok(atomic_read(empty_snapshot(), 10, &[]))]);

        let (_, snap_seq) =
            expect_verified(load_consistent_snapshot(&loader, "obj-1", &mut NoLeaseToKeep).await);
        assert_eq!(snap_seq, 10);
        assert_atomic_path_only(&loader, "cut at the pin");
    }

    /// **A header-sequence regression.** The header sequence went backwards
    /// between the pin (10) and the read (cut 9) — a DR restore with an
    /// undersized band. The state is a proven cut but at a sequence below one
    /// this tick already saw, so nothing may be persisted: `NotQuiesced`, after
    /// the one read and with no fenced-read fallback.
    #[tokio::test]
    async fn load_consistent_snapshot_atomic_cut_below_the_pin_is_not_quiesced_in_one_read() {
        let loader = FakeLoader::new()
            .with_snap_seq(10)
            .with_atomic_answers(vec![Ok(atomic_read(
                snapshot_with_sections(&[7], 9),
                9,
                &[],
            ))]);

        match load_consistent_snapshot(&loader, "obj-1", &mut NoLeaseToKeep).await {
            Ok(ConsistentSnapshot::NotQuiesced) => {}
            Ok(ConsistentSnapshot::Verified { snap_seq, .. }) => panic!(
                "a cut below the pin cannot stand for the layer the pin saw; persisting it at \
                 snap_seq={snap_seq} would claim a state older than this tick observed"
            ),
            Ok(ConsistentSnapshot::LeaseLost) => {
                panic!("the keeper never loses its lease here; got LeaseLost")
            }
            Ok(ConsistentSnapshot::ShuttingDown) => {
                panic!("the keeper never loses its lease here; got ShuttingDown")
            }
            Err(e) => panic!("a regression is a skip decision, not an error; got {e:?}"),
        }
        assert_atomic_path_only(&loader, "cut below the pin");
    }

    /// Judge the one proven read `loader` is canned with — atomic, or fenced
    /// behind a `TOO_LARGE` answer — and return the assembler's error,
    /// asserting it is `Permanent`.
    async fn corrupt_index_error(loader: &FakeLoader) -> String {
        match load_consistent_snapshot(loader, "obj-1", &mut NoLeaseToKeep).await {
            Err(CheckpointError::Permanent { source }) => source.to_string(),
            Err(other) => panic!(
                "a corrupt section index under a proven cut must be Permanent — retrying reads \
                 the same corruption; got {other:?}"
            ),
            Ok(ConsistentSnapshot::Verified { snap_seq, .. }) => panic!(
                "a snapshot whose index names a section it could not read must never be \
                 persisted: the blob would drop the section without a trace (snap_seq={snap_seq})"
            ),
            Ok(ConsistentSnapshot::NotQuiesced) => panic!(
                "a corrupt index is not a skip-and-retry: the next tick reads the same corruption"
            ),
            Ok(ConsistentSnapshot::LeaseLost) => {
                panic!("the keeper never loses its lease here; got LeaseLost")
            }
            Ok(ConsistentSnapshot::ShuttingDown) => {
                panic!("the keeper never loses its lease here; got ShuttingDown")
            }
        }
    }

    /// One atomic read whose index named `corrupt` members.
    async fn atomic_corrupt_index_error(corrupt: &[&str]) -> String {
        let loader = FakeLoader::new()
            .with_snap_seq(10)
            .with_atomic_answers(vec![Ok(atomic_read(
                snapshot_with_sections(&[7], 9),
                10,
                corrupt,
            ))]);
        let err = corrupt_index_error(&loader).await;
        assert_atomic_path_only(&loader, "corrupt index");
        err
    }

    /// An indexed section whose hash the atomic read found empty. Under one
    /// atomic cut that cannot be a race — a delete that removed the hash removed
    /// the member too — so it is a corrupt index (Spec SDD SR-4.1).
    #[tokio::test]
    async fn load_consistent_snapshot_atomic_index_member_without_a_hash_is_permanent() {
        let err = atomic_corrupt_index_error(&["42"]).await;
        assert!(
            err.contains("42"),
            "the refusal must name the member an operator has to repair; got {err:?}"
        );
    }

    /// An `obj:secs` member that is not a decimal section id names no section at
    /// all (the decoder reads it as section 0), so it is the same corrupt index.
    #[tokio::test]
    async fn load_consistent_snapshot_atomic_non_decimal_index_member_is_permanent() {
        let err = atomic_corrupt_index_error(&["not-a-section-id"]).await;
        assert!(
            err.contains("not-a-section-id"),
            "the refusal must name the member an operator has to repair; got {err:?}"
        );
    }

    /// A badly corrupted index cannot turn the refusal into an unbounded log
    /// line: it names at most eight members and reports the total, so the count
    /// still tells the operator how much is broken.
    #[tokio::test]
    async fn load_consistent_snapshot_atomic_corrupt_index_names_eight_members_and_the_total() {
        let members: Vec<String> = (0..20).map(|i| format!("ghost-{i:02}")).collect();
        let refs: Vec<&str> = members.iter().map(String::as_str).collect();
        let err = atomic_corrupt_index_error(&refs).await;

        for named in &members[..8] {
            assert!(
                err.contains(named.as_str()),
                "the first eight are named; {named} missing from {err:?}"
            );
        }
        for unnamed in &members[8..] {
            assert!(
                !err.contains(unnamed.as_str()),
                "members past the eighth are summarised, not listed; {unnamed} appears in {err:?}"
            );
        }
        assert!(
            err.contains("20"),
            "the total count must be reported so the summary is not mistaken for the whole; got \
             {err:?}"
        );
    }

    /// The version filter stays as the subordinate guard on the atomic path:
    /// entries stamped above the read's cut are dropped before the snapshot is
    /// returned (SR-4.1 keeps `filter_snapshot_by_snap_seq`).
    #[tokio::test]
    async fn load_consistent_snapshot_atomic_snapshot_is_filtered_by_its_cut() {
        let loader = FakeLoader::new()
            .with_snap_seq(100)
            .with_atomic_answers(vec![Ok(atomic_read(
                snapshot_straddling_the_cut(),
                100,
                &[],
            ))]);

        let (snapshot, snap_seq) =
            expect_verified(load_consistent_snapshot(&loader, "obj-1", &mut NoLeaseToKeep).await);
        assert_eq!(snap_seq, 100);
        assert_only_entries_at_or_below_the_cut_survive(&snapshot);
    }

    /// A failed atomic read surfaces as classified and never falls back to the
    /// fenced read: only `TOO_LARGE` routes there. Falling back on an error
    /// would hide a store fault behind a second, fenced read of a layer the
    /// atomic read could have served.
    #[tokio::test]
    async fn load_consistent_snapshot_failed_atomic_read_propagates_without_the_fenced_read() {
        for (answer, expected) in [
            (CheckpointError::NotFound, "NotFound"),
            (
                CheckpointError::Transient {
                    source: "injected transient".into(),
                },
                "Transient",
            ),
            (
                CheckpointError::Permanent {
                    source: "injected permanent".into(),
                },
                "Permanent",
            ),
        ] {
            let loader = FakeLoader::new()
                .with_snap_seq(10)
                .with_atomic_answers(vec![Err(answer)])
                .with_fenced_answers(vec![FencedAnswer::Ready(Ok(fenced_read(
                    empty_snapshot(),
                    10,
                    &[],
                )))]);

            let got = match load_consistent_snapshot(&loader, "obj-1", &mut NoLeaseToKeep).await {
                Err(CheckpointError::NotFound) => "NotFound",
                Err(CheckpointError::Transient { .. }) => "Transient",
                Err(CheckpointError::Permanent { .. }) => "Permanent",
                Err(other) => panic!("unexpected error variant {other:?}"),
                Ok(_) => panic!("a failed atomic read must fail the assembly ({expected})"),
            };
            assert_eq!(
                got, expected,
                "the atomic read's classification must pass through"
            );
            assert_atomic_path_only(&loader, expected);
        }
    }

    /// The atomic path never consults the leader-lease keeper: its one read is
    /// bounded by the atomic-read limits, so a keeper that has already lost its
    /// lease changes nothing here — the periodic loop's between-layers gate is
    /// what catches that loss.
    #[tokio::test]
    async fn load_consistent_snapshot_atomic_path_does_not_consult_the_lease() {
        let loader = FakeLoader::new()
            .with_snap_seq(10)
            .with_atomic_answers(vec![Ok(atomic_read(empty_snapshot(), 11, &[]))]);

        let (_, snap_seq) = expect_verified(
            load_consistent_snapshot(&loader, "obj-1", &mut LeaseAlreadyLost).await,
        );
        assert_eq!(snap_seq, 11);
        assert_atomic_path_only(&loader, "lease already lost");
    }

    // ── Spec SDD SR-11.5: above the limits, ONE fenced read ─────────────────
    //
    // The atomic read's `TOO_LARGE` answer routes to one fenced read, sized
    // from that answer, judged against the SAME pin the atomic path uses —
    // no re-pin, no quiescence loop, no confirmation probe, no reconciliation.
    // The fenced read returns only a proven state; its exhaustion and its wait
    // cap reach the assembler as errors, never as an unproven state.

    /// Assert the fenced path ran as one read: one atomic attempt (which said
    /// `TOO_LARGE`), one fenced read, and the pin as the only header read.
    fn assert_one_fenced_read(loader: &FakeLoader, context: &str) {
        assert_eq!(
            loader.atomic_call_count(),
            1,
            "{context}: one atomic attempt, which answered TOO_LARGE"
        );
        assert_eq!(
            loader.fenced_call_count(),
            1,
            "{context}: exactly one fenced read — its BROKEN retries live inside it, and the \
             assembler wraps no loop around it"
        );
        assert_eq!(
            loader.header_call_count(),
            1,
            "{context}: the pin is the only header read — no re-pin before the fenced read and \
             no probe after it"
        );
    }

    /// `TOO_LARGE` routes to ONE fenced read, sized from that answer
    /// (`LayerSize::from_too_large`, no recorded bytes-per-item), and the
    /// verified state is judged against the pin read before the atomic
    /// attempt. Two answer shapes, so the estimate is seen to follow the
    /// answer rather than a constant: an early-out sizes items from the item
    /// bound, a mid-read abort from the larger of the bound and the tally.
    #[tokio::test]
    async fn load_consistent_snapshot_too_large_reads_once_under_the_fence_sized_from_the_answer() {
        for answer in [
            too_large_answer(50_000, SnapshotTooLargeAt::EarlyOut),
            too_large_answer(
                40_000,
                SnapshotTooLargeAt::MidRead {
                    limit: crate::storage::diff_read::SnapshotLimit::Bytes,
                    item_tally: 45_000,
                    byte_tally: 900_000,
                },
            ),
        ] {
            let expected = LayerSize::from_too_large(&answer, None);
            let loader = FakeLoader::new()
                .with_snap_seq(10)
                .with_atomic_answers(vec![Ok(SnapshotRead::TooLarge(answer))])
                .with_fenced_answers(vec![FencedAnswer::Ready(Ok(fenced_read(
                    snapshot_with_sections(&[7, 42], 12),
                    12,
                    &[],
                )))]);

            match load_consistent_snapshot(&loader, "obj-1", &mut NoLeaseToKeep).await {
                Ok(ConsistentSnapshot::Verified { snapshot, snap_seq }) => {
                    assert_eq!(snap_seq, 12, "the proven cut is the fenced read's cut");
                    assert_eq!(snapshot.sections.len(), 2);
                }
                Ok(ConsistentSnapshot::NotQuiesced) => panic!(
                    "a fenced cut at or above the pin is proven however hot the layer is; got \
                     NotQuiesced"
                ),
                Ok(ConsistentSnapshot::LeaseLost) => {
                    panic!("the keeper never loses its lease here; got LeaseLost")
                }
                Ok(ConsistentSnapshot::ShuttingDown) => {
                    panic!("the keeper never loses its lease here; got ShuttingDown")
                }
                Err(e) => panic!("expected a verified cut; got {e:?}"),
            }
            assert_eq!(
                loader.estimates_seen(),
                vec![expected],
                "the fenced read is sized from the atomic read's TOO_LARGE answer"
            );
            assert_one_fenced_read(&loader, "too large");
        }
    }

    /// The boundary of `cut >= pin` on the fenced path: a fenced cut equal to
    /// the pin is verified.
    #[tokio::test]
    async fn load_consistent_snapshot_fenced_cut_at_the_pin_verifies() {
        let loader = FakeLoader::new()
            .with_snap_seq(10)
            .with_fenced_answers(vec![FencedAnswer::Ready(Ok(fenced_read(
                empty_snapshot(),
                10,
                &[],
            )))]);

        let (_, snap_seq) =
            expect_verified(load_consistent_snapshot(&loader, "obj-1", &mut NoLeaseToKeep).await);
        assert_eq!(snap_seq, 10);
        assert_one_fenced_read(&loader, "fenced cut at the pin");
    }

    /// **The fenced path's only skip: a header-sequence regression.** A fenced
    /// cut below the pin is judged exactly as an atomic one: `NotQuiesced`,
    /// after the one read, and nothing persisted. This is the arm the test-only
    /// unprovable-cut seam drives (its pin answers `u64::MAX`), and with it
    /// the checkpoint loop's tail-capture fallback.
    #[tokio::test]
    async fn load_consistent_snapshot_fenced_cut_below_the_pin_is_not_quiesced() {
        let loader = FakeLoader::new()
            .with_snap_seq(10)
            .with_fenced_answers(vec![FencedAnswer::Ready(Ok(fenced_read(
                snapshot_with_sections(&[7], 9),
                9,
                &[],
            )))]);

        match load_consistent_snapshot(&loader, "obj-1", &mut NoLeaseToKeep).await {
            Ok(ConsistentSnapshot::NotQuiesced) => {}
            Ok(ConsistentSnapshot::Verified { snap_seq, .. }) => panic!(
                "a fenced cut below the pin cannot stand for the layer the pin saw (snap_seq \
                 {snap_seq})"
            ),
            Ok(ConsistentSnapshot::LeaseLost) => {
                panic!("the keeper never loses its lease here; got LeaseLost")
            }
            Ok(ConsistentSnapshot::ShuttingDown) => {
                panic!("the keeper never loses its lease here; got ShuttingDown")
            }
            Err(e) => panic!("a regression is a skip decision, not an error; got {e:?}"),
        }
        assert_one_fenced_read(&loader, "fenced cut below the pin");
    }

    /// The corrupt-index refusal runs on the fenced path too, and says which
    /// read proved the cut, so an operator can tell the two paths apart.
    #[tokio::test]
    async fn load_consistent_snapshot_fenced_index_member_without_a_hash_is_permanent() {
        let loader = FakeLoader::new()
            .with_snap_seq(10)
            .with_fenced_answers(vec![FencedAnswer::Ready(Ok(fenced_read(
                snapshot_with_sections(&[7], 9),
                10,
                &["42"],
            )))]);

        let err = corrupt_index_error(&loader).await;

        assert!(
            err.contains("42"),
            "the refusal must name the member an operator has to repair; got {err:?}"
        );
        assert!(
            err.contains("fenced"),
            "the refusal names the read that proved the cut; got {err:?}"
        );
        assert_one_fenced_read(&loader, "fenced corrupt index");
    }

    /// The version filter runs on the fenced state too: entries stamped above
    /// the fenced cut are dropped before the snapshot is returned.
    #[tokio::test]
    async fn load_consistent_snapshot_fenced_snapshot_is_filtered_by_its_cut() {
        let loader = FakeLoader::new()
            .with_snap_seq(100)
            .with_fenced_answers(vec![FencedAnswer::Ready(Ok(fenced_read(
                snapshot_straddling_the_cut(),
                100,
                &[],
            )))]);

        let (snapshot, snap_seq) =
            expect_verified(load_consistent_snapshot(&loader, "obj-1", &mut NoLeaseToKeep).await);
        assert_eq!(snap_seq, 100);
        assert_only_entries_at_or_below_the_cut_survive(&snapshot);
    }

    /// A fenced read that ends without a proof surfaces its classified error
    /// after ONE call, and is never turned into a skip or a persisted state:
    /// `Transient` is the shape of a read broken `FENCE_READ_ATTEMPTS` times
    /// and of a fence wait past `reader_wait_max(L)` (the storage read's
    /// `Unavailable`, which the loader classifies `Transient`) — the tick
    /// skips the layer and the next one reads it afresh; `NotFound` is a
    /// header the acquire or the release found gone — the layer is dropped.
    #[tokio::test]
    async fn load_consistent_snapshot_failed_fenced_read_propagates_after_one_read() {
        for (answer, expected) in [
            (CheckpointError::NotFound, "NotFound"),
            (
                CheckpointError::Transient {
                    source: "fenced read broken FENCE_READ_ATTEMPTS times".into(),
                },
                "Transient",
            ),
            (
                CheckpointError::Permanent {
                    source: "injected permanent".into(),
                },
                "Permanent",
            ),
        ] {
            let loader = FakeLoader::new()
                .with_snap_seq(10)
                .with_fenced_answers(vec![FencedAnswer::Ready(Err(answer))]);

            let got = match load_consistent_snapshot(&loader, "obj-1", &mut NoLeaseToKeep).await {
                Err(CheckpointError::NotFound) => "NotFound",
                Err(CheckpointError::Transient { .. }) => "Transient",
                Err(CheckpointError::Permanent { .. }) => "Permanent",
                Err(other) => panic!("unexpected error variant {other:?}"),
                Ok(ConsistentSnapshot::NotQuiesced) => panic!(
                    "a fenced read that ended without a proof is an error, not a regression skip \
                     ({expected})"
                ),
                Ok(_) => panic!("a failed fenced read must fail the assembly ({expected})"),
            };
            assert_eq!(
                got, expected,
                "the fenced read's classification must pass through"
            );
            assert_one_fenced_read(&loader, expected);
        }
    }

    /// **The b.pw6 hole stays shut on the fenced path.** The claimed cut is
    /// the proven read's own cut, never a header sequence read after it. Pin
    /// 10; the fenced read is granted at 11, so its state carries section 42,
    /// which commit 11 created; by the time the read returns the header has
    /// moved on to 12. Stamping 12 would claim commit 12, which the read never
    /// saw; stamping the pin 10 would claim section 42 belongs to a sequence
    /// that predates it. The answer is 11, and the header is not read again.
    #[tokio::test]
    async fn load_consistent_snapshot_claims_the_fenced_reads_own_cut_never_a_later_header() {
        let loader = FakeLoader::new()
            .with_header_seq_answers(vec![Some(10), Some(12)])
            .with_fenced_answers(vec![FencedAnswer::Ready(Ok(fenced_read(
                snapshot_with_sections(&[7, 42], 11),
                11,
                &[],
            )))]);

        let (snapshot, snap_seq) =
            expect_verified(load_consistent_snapshot(&loader, "obj-1", &mut NoLeaseToKeep).await);

        assert_eq!(
            snap_seq, 11,
            "the blob claims exactly the cut the read proved"
        );
        assert!(
            snapshot.sections.contains_key(&42),
            "a checkpoint stamped snap_seq=11 carries the section commit 11 created"
        );
        assert_one_fenced_read(&loader, "claimed cut");
    }

    // ── The keeper's two stops during a fenced read (Spec SDD SR-11.5;
    // post-completion review F12) ─────────────────────────────────────────

    /// The assembly's answer for each stop, by name, for the panics.
    fn stop_name(result: &Result<ConsistentSnapshot, CheckpointError>) -> String {
        match result {
            Ok(ConsistentSnapshot::Verified { snap_seq, .. }) => {
                format!("Verified (snap_seq {snap_seq})")
            }
            Ok(ConsistentSnapshot::NotQuiesced) => "NotQuiesced".to_owned(),
            Ok(ConsistentSnapshot::LeaseLost) => "LeaseLost".to_owned(),
            Ok(ConsistentSnapshot::ShuttingDown) => "ShuttingDown".to_owned(),
            Err(e) => format!("Err({e:?})"),
        }
    }

    /// Whether `result` is the assembly's answer for `stop`.
    fn is_stopped_by(
        result: &Result<ConsistentSnapshot, CheckpointError>,
        stop: KeeperStop,
    ) -> bool {
        matches!(
            (result, stop),
            (Ok(ConsistentSnapshot::LeaseLost), KeeperStop::LeaseLost)
                | (
                    Ok(ConsistentSnapshot::ShuttingDown),
                    KeeperStop::ShuttingDown
                )
        )
    }

    /// **A stop mid-read.** The keeper `keeper` builds (stopping with `stop`
    /// once its `Notify` fires) stops while the fenced read is still under
    /// its fence: the assembly answers `stop`'s own outcome — a lost lease
    /// `LeaseLost`, a shutdown `ShuttingDown`, never the other, a regression
    /// skip or an error — the read is DROPPED, which is what releases its
    /// fence in production (the fence lease guard's drop), and it never
    /// completed, so there is no state to persist.
    async fn stop_mid_fenced_read_drops_the_read<K: LeaderLeaseKeeper>(
        keeper: impl FnOnce(Arc<tokio::sync::Notify>) -> K,
        stop: KeeperStop,
    ) {
        let loader = FakeLoader::new()
            .with_snap_seq(10)
            .with_fenced_answers(vec![FencedAnswer::HangUnderTheFence]);
        let fire = Arc::new(tokio::sync::Notify::new());
        let mut keeper = keeper(Arc::clone(&fire));

        let assembly = load_consistent_snapshot(&loader, "obj-1", &mut keeper);
        let stop_once_the_read_is_under_way = async {
            loader.fenced_started.notified().await;
            assert!(
                !loader.fenced_dropped(),
                "fixture: the read is still running when the keeper stops ({stop:?})"
            );
            fire.notify_one();
        };
        let (result, ()) = tokio::join!(assembly, stop_once_the_read_is_under_way);

        assert!(
            is_stopped_by(&result, stop),
            "a keeper stop ({stop:?}) mid-read is that stop's own decision — nothing persisted, not a \
             regression skip, not an error; got {}",
            stop_name(&result)
        );
        assert!(
            loader.fenced_dropped(),
            "the fenced read must be dropped when the keeper stops ({stop:?}) — its drop releases the \
             fence"
        );
        assert_one_fenced_read(&loader, &format!("{stop:?}"));
    }

    /// **Biased toward the keeper.** When the keeper `keeper` builds stops
    /// (with `stop`) in the same poll the fenced read becomes ready, the
    /// assembly still answers `stop`'s outcome: nothing may be written without
    /// the lease, nor as the loop shuts down, however complete the read. Both
    /// futures are woken by one `notify_waiters`, so on the next poll both are
    /// ready; an unbiased `select!` would pick the read about half the time,
    /// which is why the case runs many rounds.
    async fn stop_as_the_fenced_read_completes_still_wins<K: LeaderLeaseKeeper>(
        keeper: impl Fn(Arc<tokio::sync::Notify>) -> K,
        stop: KeeperStop,
    ) {
        const ROUNDS: usize = 64;
        for round in 0..ROUNDS {
            let both = Arc::new(tokio::sync::Notify::new());
            let loader = FakeLoader::new()
                .with_snap_seq(10)
                .with_fenced_answers(vec![FencedAnswer::ReadyWhen(
                    Arc::clone(&both),
                    Ok(fenced_read(empty_snapshot(), 10, &[])),
                )]);
            let mut keeper = keeper(Arc::clone(&both));

            let assembly = load_consistent_snapshot(&loader, "obj-1", &mut keeper);
            let wake_both = async {
                loader.fenced_started.notified().await;
                both.notify_waiters();
            };
            let (result, ()) = tokio::join!(assembly, wake_both);

            assert!(
                is_stopped_by(&result, stop),
                "round {round}: a read that completes in the poll the keeper stops ({stop:?}) must not \
                 be persisted; got {}",
                stop_name(&result)
            );
        }
    }

    #[tokio::test]
    async fn load_consistent_snapshot_lease_lost_mid_fenced_read_drops_the_read_and_is_lease_lost()
    {
        stop_mid_fenced_read_drops_the_read(LoseLeaseWhen, KeeperStop::LeaseLost).await;
    }

    #[tokio::test]
    async fn load_consistent_snapshot_lease_lost_as_the_fenced_read_completes_is_still_lease_lost()
    {
        stop_as_the_fenced_read_completes_still_wins(LoseLeaseWhen, KeeperStop::LeaseLost).await;
    }

    #[tokio::test]
    async fn load_consistent_snapshot_shutdown_mid_fenced_read_drops_the_read_and_is_shutting_down()
    {
        stop_mid_fenced_read_drops_the_read(ShutDownWhen, KeeperStop::ShuttingDown).await;
    }

    #[tokio::test]
    async fn load_consistent_snapshot_shutdown_as_the_fenced_read_completes_is_still_shutting_down()
    {
        stop_as_the_fenced_read_completes_still_wins(ShutDownWhen, KeeperStop::ShuttingDown).await;
    }

    // ── b.gcb / b.vz7: byte-deterministic base emission ──────────────────────
    //
    // The Task's success criterion (a) in its purest form: two assemblies of the
    // same state produce byte-identical base OMPF and byte-identical sidecar
    // JSON.
    //
    // These live HERE and nowhere else because `build_reset_ompf_from_state` and
    // `ext_refs_sidecar_json` are `pub(crate)` —
    // `tests/integration/checkpoint/e2e_composed_restore_test.rs` records that
    // constraint explicitly ("`build_reset_ompf_from_state` is `pub(crate)`, so
    // the base cannot be [built directly]"). No integration-crate test and no
    // proptest binary can reach them. Both are plain `#[test]`: no Valkey, no
    // tokio runtime (see the module header).
    //
    // The sibling DOMAIN tests — `plan_full_content_is_deterministic_for_equal_state`
    // and `plan_full_content_ext_hash_order_is_deterministic_for_equal_state` in
    // `src/domain/diff.rs` — pin the PLAN order. These pin that the plan order
    // propagates all the way to BYTES, which is a separate claim: FlatBuffers
    // vtable sharing and offset widths follow emission order, which is why
    // `tests/fixtures/tail_lineage/README.md` § "Nothing else needs normalising"
    // records base.ompf moving in *length*, not merely in content, across
    // pre-sort regenerations. The length assertion below is therefore
    // load-bearing evidence, not decoration — and the mutation table on the
    // first test confirms it fires in practice, on 5/10 of the field- and
    // time-sample-sort neutralisation runs.
    //
    // Construction discipline matches the domain siblings: two INDEPENDENTLY
    // built `ObjectState`s, with every unordered container filled in the reverse
    // insertion order for the second build. `RandomState::new()` draws a fresh
    // key per `HashMap` INSTANCE, so an in-process pair iterates differently in
    // exactly the way two replicas' maps do — a genuine witness, not a weaker
    // stand-in (Task amendment 2026-09-01). A `clone()`-based fixture would be
    // vacuous and is prohibited. `children` insertion order is deliberately held
    // FIXED: it is real `LRANGE`-ordered persisted data, so permuting it would
    // change the state's MEANING rather than its map layout.

    /// Deterministic 32-byte test hash — local to this group so the fixture's
    /// hash bytes read as `0xA1`/`0xB2` at the call site.
    fn det_hash(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    /// `ObjectState`-oriented section builder, deliberately separate from this
    /// module's `make_section` (which is `ObjectSnapshot`-shaped, always
    /// `parent_id = 0`, and depended on by the tests above).
    fn det_section(id: u64, parent: u64) -> Section {
        Section {
            section_id: id,
            parent_id: parent,
            name: format!("sec_{id}"),
            section_type: 0,
            section_order: id as u32,
            first_version: 1,
            last_version: 9,
        }
    }

    /// `cumulative_set_order` and `authoring_set_order` are SEPARATE arguments on
    /// purpose — see NON-VACUITY (d) in [`det_state`].
    fn det_field(
        section_id: u64,
        key: &str,
        version: u64,
        cumulative_set_order: u64,
        authoring_set_order: u64,
    ) -> Field {
        Field {
            section_id,
            key_name: key.to_string(),
            value: FieldValue::Inline(vec![version as u8, section_id as u8]),
            set_order: cumulative_set_order,
            authoring_set_order,
            version,
        }
    }

    fn det_ext_field(
        section_id: u64,
        key: &str,
        version: u64,
        cumulative_set_order: u64,
        authoring_set_order: u64,
        hash_byte: u8,
    ) -> Field {
        Field {
            section_id,
            key_name: key.to_string(),
            value: FieldValue::External {
                hash: det_hash(hash_byte),
                size: 1024,
                ext_value_index: 0,
            },
            set_order: cumulative_set_order,
            authoring_set_order,
            version,
        }
    }

    fn det_time_sample(section_id: u64, time: f64, version: u64) -> TimeSample {
        TimeSample {
            section_id,
            time: OrderedFloat(time),
            value: FieldValue::Inline(vec![version as u8]),
            version,
        }
    }

    fn det_ext_time_sample(section_id: u64, time: f64, version: u64, hash_byte: u8) -> TimeSample {
        TimeSample {
            section_id,
            time: OrderedFloat(time),
            value: FieldValue::External {
                hash: det_hash(hash_byte),
                size: 2048,
                ext_value_index: 0,
            },
            version,
        }
    }

    /// The shared fixture for both byte-determinism tests.
    ///
    /// `reverse` flips the insertion order of every unordered container
    /// (`sections`' orphan tail, `fields`, `time_samples`) so the two builds
    /// differ in `HashMap` layout while being logically identical. `children` is
    /// held fixed — see the group comment above.
    ///
    /// Every NON-VACUITY condition below is annotated with what its removal
    /// would let through. Each one guards a distinct half of the landed sort;
    /// dropping any of them makes at least one of the three sorts untested here.
    fn det_state(reverse: bool) -> ObjectState {
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 9,
            first_version: 1,
            source_format: "usda".to_string(),
            next_set_order: 908,
            ..Default::default()
        };

        // Reachable tree, root(0) └─1 ├─2 └─3 ; 2 └─4. `children`'s Vec order is
        // identical in both builds by construction.
        for s in [
            det_section(1, 0),
            det_section(2, 1),
            det_section(3, 1),
            det_section(4, 2),
        ] {
            state
                .children
                .entry(s.parent_id)
                .or_default()
                .push(s.section_id);
            state.sections.insert(s.section_id, s);
        }

        // NON-VACUITY (a) — FIVE ORPHAN sections (dangling parent 900, never
        // registered in `children`, so the DFS cannot reach them and they land
        // in the orphan tail). Their insertion order (83, 80, 84, 81, 82) is
        // deliberately not ascending, so an ascending emitted tail cannot happen
        // by accident.
        //
        // Removing them would let a FIELDS-ONLY sort pass: the emitted
        // `struct_commands` order — not just the field order — is what the OMPF
        // byte comparison covers here, and with a purely reachable tree that
        // order is already deterministic from `children` alone. This is the
        // condition most likely to be omitted and the one that catches the
        // fields-only regression (seam E-1(ii)).
        //
        // FIVE rather than the minimum two so a red is not a coin flip: an
        // unordered tail has to hit one of 120 permutations in BOTH builds to
        // slip past, instead of one of two.
        let mut orphans = vec![
            det_section(83, 900),
            det_section(80, 900),
            det_section(84, 900),
            det_section(81, 900),
            det_section(82, 900),
        ];

        // NON-VACUITY (b) — fields TIE on the leading `(version,
        // authoring_set_order)` pair `(6, 1)` ACROSS four different sections AND
        // WITHIN one section (`e_c` / `e_d`, both section 1). Without the
        // cross-section tie the `section_id` component of the key is never
        // exercised; without the within-section tie the `key_name` tiebreak is
        // never exercised, and a 3-tuple sort (or a *stable* sort riding on
        // `HashMap` order for the ties) would pass (seam E-2).
        //
        // FIVE of the seven carry distinct ext hashes, for the same
        // non-flakiness reason as the orphan tail: the sidecar's hash order then
        // has 120 ways to be wrong per build rather than two. `early` and
        // `later` stay INLINE at both ends of the sorted sequence, so the hash
        // scan's "skip non-external" branch is exercised inside the sequence
        // rather than around it.
        //
        // NON-VACUITY (d) — every field's cumulative `set_order` (the 900s) is
        // deliberately out of line with its `authoring_set_order` (0..2), and
        // the cumulative cursors are not even ascending in emission order.
        // Removing that divergence would let a broken
        // `set_order = authoring_set_order` remap (assembler.rs, before
        // `build_delta`) through: the wire slot would carry the same number
        // either way and the decode assertion below would be vacuous.
        let mut fields = vec![
            det_ext_field(2, "e_a", 6, 900, 1, 0xA3),
            det_ext_field(3, "e_b", 6, 901, 1, 0xA4),
            det_ext_field(1, "e_c", 6, 902, 1, 0xA1),
            det_ext_field(1, "e_d", 6, 903, 1, 0xA2),
            det_ext_field(4, "s4_k", 6, 905, 1, 0xA5),
            det_field(1, "early", 5, 907, 2),
            det_field(3, "later", 7, 904, 0),
        ];

        // NON-VACUITY (c) — `0xB1`..`0xB4` appear FIRST on a TIME SAMPLE and on
        // NO field, and the four samples carrying them all share `(version = 7,
        // section_id = 2)`, differing only in `time`.
        //
        // `collect_external_value_hashes` scans fields THEN time samples into
        // ONE shared first-appearance pool (seam E-1(i)), so removing them would
        // leave the sidecar assertion a fields-only claim and the time-sample
        // sort untested here — a state whose ext refs sit on time samples would
        // still produce a non-deterministic sidecar. Sharing `(version,
        // section_id)` is what exercises the `time` component; without it a
        // `(version, section_id)`-only sort would pass. Four rather than two,
        // again so an unsorted red is not a coin flip.
        //
        // `0xA2` reappears on a later sample already seen on a field, so the
        // first-appearance dedup path is covered too (no second entry).
        let mut samples = vec![
            det_ext_time_sample(2, 3.0, 7, 0xB3),
            det_ext_time_sample(2, 1.0, 7, 0xB1),
            det_ext_time_sample(2, 4.0, 7, 0xB4),
            det_ext_time_sample(2, 2.0, 7, 0xB2),
            det_time_sample(3, 2.0, 7),
            det_ext_time_sample(3, 2.5, 8, 0xA2),
            det_time_sample(4, 0.5, 9),
        ];

        if reverse {
            orphans.reverse();
            fields.reverse();
            samples.reverse();
        }
        for o in orphans {
            state.sections.insert(o.section_id, o);
        }
        for f in fields {
            state.fields.insert((f.section_id, f.key_name.clone()), f);
        }
        for ts in samples {
            state.time_samples.insert((ts.section_id, ts.time), ts);
        }
        state
    }

    /// The ext-value hashes [`det_state`] pins, in the order the landed total
    /// order forces:
    ///
    /// - fields sorted `(version, authoring_set_order, section_id, key_name)`:
    ///   `(5,2,1,early)` inline, `(6,1,1,e_c)`→`A1`, `(6,1,1,e_d)`→`A2`,
    ///   `(6,1,2,e_a)`→`A3`, `(6,1,3,e_b)`→`A4`, `(6,1,4,s4_k)`→`A5`,
    ///   `(7,0,3,later)` inline
    /// - then time samples sorted `(version, section_id, time)`:
    ///   `(7,2,1.0)`→`B1`, `(7,2,2.0)`→`B2`, `(7,2,3.0)`→`B3`, `(7,2,4.0)`→`B4`,
    ///   `(7,3,2.0)` inline, `(8,3,2.5)`→`A2` (dup, no new entry), `(9,4,0.5)`
    ///   inline
    fn det_expected_hashes() -> Vec<[u8; 32]> {
        vec![
            det_hash(0xA1),
            det_hash(0xA2),
            det_hash(0xA3),
            det_hash(0xA4),
            det_hash(0xA5),
            det_hash(0xB1),
            det_hash(0xB2),
            det_hash(0xB3),
            det_hash(0xB4),
        ]
    }

    /// b.gcb / b.vz7 — two independently-built, logically identical
    /// `ObjectState`s assemble to BYTE-IDENTICAL base OMPF.
    ///
    /// This is the structural ground SR-9.2's duplicate-leader re-base
    /// idempotency argument stands on: identical inputs ⇒ identical bytes ⇒ a
    /// re-base PUT at the same key is byte-idempotent, rather than "the two
    /// blobs mean the same thing" which every future reader would have to
    /// re-derive.
    ///
    /// **Mutation evidence.** Each of the three landed sorts in
    /// `src/domain/diff.rs` was neutralised in turn (scratch edit, restored
    /// byte-identically; never committed), five `--retries 0` repeats each. This
    /// test observed red 5/5 on all three:
    ///
    /// | neutralised | first assertion red | content-only or length-too |
    /// |---|---|---|
    /// | `Self::sort_fields_by_authoring(&mut fields)` in `plan_full_content` | MUTATION-EVIDENCE-LENGTH (2/5) or MUTATION-EVIDENCE-CONTENT (3/5) | **length-too** on the runs that reddened on length (1672 vs 1680 / 1672 vs 1688 bytes); content-only at equal length (first divergence at byte 164) on the rest |
    /// | `time_samples.sort_by_key(…)` in `plan_full_content` | MUTATION-EVIDENCE-LENGTH (3/5) or MUTATION-EVIDENCE-CONTENT (2/5) | **length-too** (1664 vs 1680 bytes); content-only at equal length (byte 92 / 96) on the rest |
    /// | `orphans.sort_unstable_by_key(…)` in `sections_in_dfs_order` | MUTATION-EVIDENCE-CONTENT (5/5) | **content-only**, never length — the five orphans differ only in `section_id`, and their names (`sec_80`..`sec_84`) are equal-length, so a permuted tail changes no vtable count and no offset width |
    ///
    /// Which of LENGTH / CONTENT fires for the first two is a property of the
    /// particular `HashMap` permutation that run drew, not of the defect — both
    /// are the same ordering regression. That is exactly why the length
    /// assertion is separate and first: a run that diverges at equal length must
    /// not read as "a value differs".
    ///
    /// Each neutralisation was then re-run with the byte and ordered-hash
    /// assertions temporarily inert, to confirm the decoded spot checks NAME the
    /// regression rather than leaving it an opaque byte inequality (3/3 each):
    ///
    /// | neutralised | spot check red |
    /// |---|---|
    /// | field sort | MUTATION-EVIDENCE-FIELDS — `the wire field order must follow (version, authoring_set_order, section_id, key_name)` |
    /// | time-sample sort | MUTATION-EVIDENCE-SAMPLES — `the wire time-sample order must follow (version, section_id, time)` |
    /// | orphan sort | MUTATION-EVIDENCE-ORPHANS — `the emitted orphan tail must be section_id-ascending, got [1, 2, 4, 3, 83, 80, …]` |
    #[test]
    fn build_reset_ompf_from_state_is_byte_identical_for_equal_state() {
        let (ompf_a, hashes_a) = build_reset_ompf_from_state(&det_state(false))
            .expect("the fixture state must assemble");
        let (ompf_b, hashes_b) =
            build_reset_ompf_from_state(&det_state(true)).expect("the fixture state must assemble");

        // LENGTH FIRST, and as its own assertion. A length divergence is a
        // different diagnosis from a content divergence: the FlatBuffers builder
        // shares one vtable between tables emitted with identical field-presence
        // layouts and widens offsets as the buffer grows, so a reordered
        // field / time-sample / struct-command sequence changes how many vtables
        // are shared and how wide the offsets are. `tail_lineage/README.md`
        // records exactly that for the pre-sort `base.ompf`. Read a red here as
        // "emission order moved", never as "a value differs".
        assert_eq!(
            ompf_a.len(),
            ompf_b.len(),
            "MUTATION-EVIDENCE-LENGTH: base OMPF LENGTH diverged ({} vs {} bytes) for two \
             independently-built logically identical states. Length moves with emission order via \
             FlatBuffers vtable sharing and offset widths — this is an ordering defect in \
             `plan_full_content` / `sections_in_dfs_order`, not a differing value",
            ompf_a.len(),
            ompf_b.len()
        );

        // The headline. Deliberately NOT weakened to semantic isomorphism: Task
        // 3's SR-9.2 duplicate-leader tests and Task 7's cross-replica prose are
        // designed on this byte basis, and downgrading here would make both
        // unsound. If this reddens, decode BOTH (`OmpfContainer::parse` +
        // `oodf_ffi::parse_delta`) and report which vector, which entries and
        // which order diverged.
        let first_divergence = ompf_a
            .iter()
            .zip(ompf_b.iter())
            .position(|(a, b)| a != b)
            .map_or("none".to_string(), |i| i.to_string());
        assert_eq!(
            ompf_a, ompf_b,
            "MUTATION-EVIDENCE-CONTENT: base OMPF bytes diverged at EQUAL length (first differing \
             byte index: {first_divergence}) for two independently-built logically identical \
             states — one of `plan_full_content`'s three emitted vectors is still HashMap-derived, \
             or a fourth un-ordered vector reached `build_delta`. Decode both with \
             `OmpfContainer::parse` + `oodf_ffi::parse_delta` and report the exact divergence; do \
             NOT downgrade this to an isomorphism assertion"
        );

        // Repeating on the SAME state is also stable — weaker than the pair
        // above, but it separates "emission order varies" from "some non-order
        // nondeterminism (clock, address, interior mutability) leaks into the
        // blob" when diagnosing a red.
        let state = det_state(false);
        let (again, _) = build_reset_ompf_from_state(&state).expect("re-assembly must succeed");
        let (and_again, _) = build_reset_ompf_from_state(&state).expect("re-assembly must succeed");
        assert_eq!(
            again, and_again,
            "two assemblies of ONE state must agree byte-for-byte"
        );

        // Ordered ext hashes compared as VECTORS, never as sets — a set
        // comparison is precisely the weak observable b.vz7 records as unable to
        // witness order, and would be satisfied by every permutation the
        // pre-sort emitter produced.
        assert_eq!(
            hashes_a, hashes_b,
            "MUTATION-EVIDENCE-HASHES: the ordered ext-value hashes returned alongside the base \
             must be equal as VECTORS across two independently-built equal states"
        );
        assert_eq!(
            hashes_a,
            det_expected_hashes(),
            "the first-appearance hash order must follow the emitted field order, then the \
             emitted time-sample order"
        );

        // ── Non-vacuity: the bytes really carry the fixture ──────────────────
        //
        // Without this block a future refactor that emptied `det_state` would
        // leave both equalities trivially true. It also witnesses the claim the
        // Task actually makes — that the PLAN order propagates to the BYTES —
        // rather than inferring it.
        let container = OmpfContainer::parse(&ompf_a).expect("the assembled base must parse");
        let delta =
            oodf_ffi::parse_delta(container.delta_bytes).expect("the base delta must parse");

        let emitted_sections: Vec<u64> = delta
            .struct_commands
            .iter()
            .map(|c| match c {
                oodf_ffi::StructCommand::DiffChangeSection(s) => s.section_id,
                other => panic!("expected DiffChangeSection, got {other:?}"),
            })
            .collect();
        assert_eq!(
            emitted_sections.len(),
            9,
            "4 reachable + 5 orphan sections must reach the wire"
        );
        // E-1(ii): the ORPHAN TAIL is the half that gained an order. E-5: the
        // reachable DFS prefix MUST NOT move, so nothing new is asserted about
        // it beyond the parent-before-child shape existing coverage carries.
        let (prefix, tail) = emitted_sections.split_at(4);
        assert_eq!(
            tail,
            [80, 81, 82, 83, 84],
            "MUTATION-EVIDENCE-ORPHANS: the emitted orphan tail must be section_id-ascending, got \
             {emitted_sections:?}"
        );
        let mut prefix_sorted = prefix.to_vec();
        prefix_sorted.sort_unstable();
        assert_eq!(
            prefix_sorted,
            vec![1, 2, 3, 4],
            "every reachable section must precede every orphan, got {emitted_sections:?}"
        );
        let pos = |id: u64| prefix.iter().position(|&x| x == id).unwrap();
        assert!(
            pos(1) < pos(2) && pos(1) < pos(3) && pos(2) < pos(4),
            "the reachable prefix must stay parent-before-child, got {prefix:?}"
        );

        let emitted_fields: Vec<(u64, &str)> = delta
            .set_fields
            .iter()
            .map(|f| (f.section_id, f.key_name.as_str()))
            .collect();
        assert_eq!(
            emitted_fields,
            vec![
                (1, "early"),
                (1, "e_c"),
                (1, "e_d"),
                (2, "e_a"),
                (3, "e_b"),
                (4, "s4_k"),
                (3, "later"),
            ],
            "MUTATION-EVIDENCE-FIELDS: the wire field order must follow (version, \
             authoring_set_order, section_id, key_name)"
        );
        // NON-VACUITY (d) — the wire `set_order` slot carries the writer's
        // AUTHORING rank, not the cumulative cursor (the 900s). The fixture's
        // two ranks differ on every field, so this cannot pass by coincidence.
        assert_eq!(
            delta
                .set_fields
                .iter()
                .map(|f| f.set_order)
                .collect::<Vec<u64>>(),
            vec![2, 1, 1, 1, 1, 1, 0],
            "the assembler's set_order = authoring_set_order remap must survive the sort — the \
             cumulative cursors (900..907) must not reach the wire"
        );

        let emitted_samples: Vec<(u64, f64)> = delta
            .time_samples
            .iter()
            .map(|ts| (ts.section_id, ts.time))
            .collect();
        assert_eq!(
            emitted_samples,
            vec![
                (2, 1.0),
                (2, 2.0),
                (2, 3.0),
                (2, 4.0),
                (3, 2.0),
                (3, 2.5),
                (4, 0.5),
            ],
            "MUTATION-EVIDENCE-SAMPLES: the wire time-sample order must follow (version, \
             section_id, time)"
        );
    }

    /// b.gcb / b.vz7 — the ext-refs sidecar JSON is byte-identical for two
    /// independently-built, logically identical states.
    ///
    /// This is what closes b.vz7 directly: `ext_refs_sidecar_json`'s
    /// doc-comment claims byte-for-byte determinism, and until the sort landed
    /// that claim rested on a false ground. The assertion is on BYTES, never on
    /// a `HashSet<String>` round-trip — the set-semantic observable is precisely
    /// what b.vz7 records as unable to witness order.
    ///
    /// **Mutation evidence.** Same three scratch neutralisations as the sibling
    /// above, five `--retries 0` repeats each. The sidecar is a JSON array of
    /// hex strings, so there is no separate length claim to make here — a
    /// reordering of nine equal-length hashes moves content at constant length
    /// by construction, which is precisely the b.vz7 observation that the OMPF
    /// half needs its own length assertion and this half does not:
    ///
    /// | neutralised | first assertion red | note |
    /// |---|---|---|
    /// | `Self::sort_fields_by_authoring(&mut fields)` | MUTATION-EVIDENCE-SIDECAR (5/5) | content-only, constant length |
    /// | `time_samples.sort_by_key(…)` | MUTATION-EVIDENCE-SIDECAR (5/5) | content-only, constant length. This row is the whole reason the `0xB1`..`0xB4` half of the fixture exists: sorting fields alone leaves the sidecar non-deterministic (seam E-1(i)) |
    /// | `orphans.sort_unstable_by_key(…)` | — **green 5/5** | expected and not a gap: orphan SECTIONS carry no ext-value references, so the orphan tail cannot reach `collect_external_value_hashes`. That sort is fenced by the sibling test, which reddened 5/5 on it |
    ///
    /// An earlier, weaker fixture (two ext fields / two ext time samples)
    /// reddened only ~4/6 under the time-sample neutralisation, because two
    /// hashes have only two orderings and both builds can draw the right one.
    /// The fixture now carries five field-first and four time-sample-first
    /// hashes for that reason — see NON-VACUITY (b)/(c) in [`det_state`].
    #[test]
    fn ext_refs_sidecar_json_is_byte_identical_for_equal_state() {
        let (_, hashes_a) = build_reset_ompf_from_state(&det_state(false))
            .expect("the fixture state must assemble");
        let (_, hashes_b) =
            build_reset_ompf_from_state(&det_state(true)).expect("the fixture state must assemble");

        let sidecar_a =
            ext_refs_sidecar_json(&hashes_a).expect("sidecar serialization must succeed");
        let sidecar_b =
            ext_refs_sidecar_json(&hashes_b).expect("sidecar serialization must succeed");

        // Non-vacuity first: an empty hash list would make the equality below
        // trivially true, and the fixture's whole point is that four of its nine
        // hashes arrive via the TIME-SAMPLE half of the scan.
        assert_eq!(
            hashes_a.len(),
            9,
            "the fixture must carry nine distinct ext hashes (five field-first, four \
             time-sample-first), or the sidecar comparison is vacuous"
        );

        assert_eq!(
            sidecar_a, sidecar_b,
            "MUTATION-EVIDENCE-SIDECAR: the ext-refs sidecar JSON must be byte-identical across \
             two independently-built equal states. This is asserted on BYTES on purpose — a \
             HashSet round-trip cannot witness order, which is the b.vz7 defect itself"
        );

        // The concrete bytes, so a red names the order rather than leaving it an
        // opaque inequality between two blobs.
        let quoted: Vec<String> = det_expected_hashes()
            .iter()
            .map(|h| format!("\"{}\"", sha256_to_hex(h)))
            .collect();
        let expected: Vec<u8> = format!("{{\"ext_hashes\":[{}]}}", quoted.join(",")).into_bytes();
        assert_eq!(
            sidecar_a, expected,
            "MUTATION-EVIDENCE-SIDECAR-ORDER: the sidecar's hash order must follow the emitted \
             field order, then the emitted time-sample order"
        );

        // The empty case still emits a sidecar rather than an error or `None` —
        // the doc-comment makes that a uniformity promise the adapter
        // publish-order obligation (b.s5y) rides on.
        assert_eq!(
            ext_refs_sidecar_json(&[]).expect("an empty hash list must still serialize"),
            br#"{"ext_hashes":[]}"#.to_vec(),
            "every base generation carries a sidecar, including one with no ext refs"
        );
    }

    // ── `ValkeySnapshotLoader::read_header_identity`'s absence contract ──────
    //
    // b.hka review round 2, path (b). `Ok(None)` means "for DR purposes this
    // object does not exist", and since round 2 it covers TWO header states, not
    // one: the key is gone, or the key is there with an empty `session_id` /
    // `layer_name`.
    //
    // The second used to fold to `String::new()` alongside every other column,
    // and the fold was a foot-gun rather than a downgrade-in-place: every
    // durable-generation producer that did not separately re-check emptiness
    // handed the degenerate meta to `validate_generation_identity`, which refuses
    // it with a `Permanent` — an adapter-health error tick every tick, forever,
    // from a perfectly healthy adapter. Exactly one of the four producers carried
    // that second check. Taking the decision here is what makes "no generation of
    // any kind ever carries empty identity" structural rather than merely usual.
    //
    // These drive the PRODUCTION loader against a real header hash, because the
    // translation under test IS the loader's — a `FakeLoader` answering `None`
    // would only restate the caller's `match`. The three producers' downstream
    // arms are pinned at their own sites: `assembler_test.rs` (assembler →
    // `NotFound`), `reconcile_dead_object_test.rs` (tail capture → `NotFound`),
    // `first_base_bootstrap_test.rs` (bootstrap → `SkippedHeaderAbsent`).

    /// `HSET` a whole header hash and read it back through the production loader.
    async fn identity_of(
        tv: &crate::storage::test_support::TestValkey,
        object_id: u64,
        columns: &[(&str, &str)],
    ) -> Option<HeaderIdentity> {
        let key = tv.key_schema().obj_header(&object_id.to_string());
        let pairs: Vec<(String, String)> = columns
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        let _: u64 = tv
            .client()
            .pool
            .hset(&key, pairs)
            .await
            .expect("HSET obj:hdr must succeed");
        ValkeySnapshotLoader {
            client: tv.client(),
            keys: tv.key_schema(),
        }
        .read_header_identity(&object_id.to_string())
        .await
        .expect("a present, well-formed header must not be an Err")
    }

    /// The four columns a healthy `create_layer` header carries that this unit
    /// reads, in the shape it writes them.
    const HEALTHY_HEADER: &[(&str, &str)] = &[
        ("seq", "12"),
        ("session_id", "sess-1"),
        ("layer_name", "layer-1"),
        ("created_by", "user-1"),
        ("last_server_ts", "1700000000000"),
    ];

    /// The positive control, first: a complete header reads back as `Some` with
    /// both identity columns intact.
    ///
    /// Without it every case below is satisfiable by a loader that returns `None`
    /// unconditionally — which would take all four durable-generation producers
    /// permanently offline while every "must refuse" assertion stayed green.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_header_identity_returns_some_for_a_complete_header() {
        let tv = crate::storage::test_support::TestValkey::setup().await;

        let identity = identity_of(&tv, 1, HEALTHY_HEADER)
            .await
            .expect("a complete header is not absent");

        assert_eq!(identity.session_id, "sess-1");
        assert_eq!(identity.layer_name, "layer-1");
        assert_eq!(identity.created_by, "user-1");
        assert!(
            identity.cursor_columns_trusted,
            "both cursor columns are ABSENT here, which is a truthful 0 (nothing has been \
             checkpointed) and must stay distinguishable from a fabricated one"
        );
    }

    /// An ABSENT `session_id` column — the partially-written / hand-repaired
    /// header — reads as `Ok(None)`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_header_identity_reports_a_missing_session_id_as_absent() {
        let tv = crate::storage::test_support::TestValkey::setup().await;
        let columns: Vec<(&str, &str)> = HEALTHY_HEADER
            .iter()
            .copied()
            .filter(|(name, _)| *name != "session_id")
            .collect();

        assert!(
            identity_of(&tv, 2, &columns).await.is_none(),
            "a generation stamped from this header is refused by \
             `validate_generation_identity` at every adapter and by the apply path's identity \
             guard, so it is permanently unrestorable. Reporting it as present is what made a \
             degenerate header tick adapter-health on every tick"
        );
    }

    /// An EMPTY-STRING `session_id` reads the same way as an absent one.
    ///
    /// Separate from the case above because the two arrive by different routes —
    /// `HDEL`/partial write versus a write that stored `""` — and the pre-fix fold
    /// made them indistinguishable in the WRONG direction: `unwrap_or_default()`
    /// turned the absent one INTO the empty one and then admitted both.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_header_identity_reports_an_empty_session_id_as_absent() {
        let tv = crate::storage::test_support::TestValkey::setup().await;
        let mut columns: Vec<(&str, &str)> = HEALTHY_HEADER.to_vec();
        columns.retain(|(name, _)| *name != "session_id");
        columns.push(("session_id", ""));

        assert!(identity_of(&tv, 3, &columns).await.is_none());
    }

    /// And the same for `layer_name`, in both shapes.
    ///
    /// Asserted rather than inferred from the `session_id` pair: the production
    /// check is a two-clause `||`, and an implementation that tested only
    /// `session_id` would pass every case above while writing an unrestorable
    /// generation for any layer whose name column was lost. The apply path needs
    /// BOTH — it cannot rebuild `obj:hdr` knowing the session but not the layer.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_header_identity_reports_a_degenerate_layer_name_as_absent() {
        let tv = crate::storage::test_support::TestValkey::setup().await;

        let missing: Vec<(&str, &str)> = HEALTHY_HEADER
            .iter()
            .copied()
            .filter(|(name, _)| *name != "layer_name")
            .collect();
        assert!(
            identity_of(&tv, 4, &missing).await.is_none(),
            "absent `layer_name`"
        );

        let mut empty: Vec<(&str, &str)> = missing;
        empty.push(("layer_name", ""));
        assert!(
            identity_of(&tv, 5, &empty).await.is_none(),
            "empty-string `layer_name`"
        );
    }

    /// The DESCRIPTIVE columns keep folding — the round-2 change is scoped to the
    /// two the restore path guards on, and widening it would be a second bug.
    ///
    /// `created_by` and `last_server_ts` are consulted by no guard anywhere, so a
    /// header missing them still describes a restorable layer and must still
    /// assemble. A loader that refused on any empty column would drop those
    /// layers out of DR entirely, which is strictly worse than a downgraded
    /// description.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_header_identity_still_folds_the_descriptive_columns() {
        let tv = crate::storage::test_support::TestValkey::setup().await;

        let identity = identity_of(
            &tv,
            6,
            &[("seq", "12"), ("session_id", "s"), ("layer_name", "l")],
        )
        .await
        .expect("missing DESCRIPTIVE columns must not be reported as an absent header");

        assert_eq!(identity.created_by, "");
        assert_eq!(identity.server_ts, 0);
        assert_eq!(
            (identity.session_id, identity.layer_name),
            ("s".to_owned(), "l".to_owned())
        );
    }

    /// Sink for [`capture_warns`].
    #[derive(Clone)]
    struct CapturingWriter(std::sync::Arc<Mutex<Vec<u8>>>);

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturingWriter {
        type Writer = CapturingWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl std::io::Write for CapturingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .expect("capture buffer poisoned")
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Drive `future` under a capturing subscriber and return its value alongside
    /// everything it logged at WARN or above.
    ///
    /// `WithSubscriber` rather than `set_default`: this future awaits Valkey I/O
    /// on a multi-thread runtime and may resume on a different worker thread, and
    /// a thread-local dispatcher would silently stop capturing at the first
    /// `.await` — the failure mode that makes a log assertion read as "the WARN
    /// was never emitted". Same shape as
    /// `adapters::local_disk::tests::capture_warns`.
    async fn capture_warns<F, T>(future: F) -> (T, String)
    where
        F: std::future::Future<Output = T>,
    {
        use tracing::instrument::WithSubscriber;
        let buf = std::sync::Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(CapturingWriter(buf.clone()))
            .with_max_level(tracing::Level::WARN)
            // The assertions match plain substrings and ANSI escapes land inside
            // them.
            .with_ansi(false)
            .finish();
        let value = future
            .with_subscriber(tracing::Dispatch::new(subscriber))
            .await;
        let text =
            String::from_utf8_lossy(&buf.lock().expect("capture buffer poisoned")).into_owned();
        (value, text)
    }

    /// The degenerate header is reported to the operator, and says WHICH column
    /// was empty.
    ///
    /// `Ok(None)` is deliberately indistinguishable — at the type level — from
    /// the ordinary "this layer was deleted" answer, and every one of the four
    /// producers routes it onto an arm that is silent by design (`NotFound`
    /// dropped at `debug!` with no metric tick, or the bootstrap's
    /// `SkippedHeaderAbsent` cell). So for a header that EXISTS but has lost an
    /// identity column, this WARN is the only operator-visible evidence that
    /// anything is wrong — a layer that silently stops being checkpointed
    /// forever, with a `debug!` as its sole trace, is a DR hole nobody would find.
    /// That makes the log an interface here rather than incidental observability,
    /// which is why it is asserted at all (cf. the test-writing guide's default
    /// "do not assert on log strings").
    ///
    /// The complete-header half is the non-vacuity control: an implementation
    /// that WARNed unconditionally — or one that moved the WARN above the
    /// emptiness check — would satisfy the first half alone while burying a real
    /// signal in per-tick noise for every healthy layer in the fleet.
    ///
    /// Both column flags are asserted, not just the empty one: the WARN's value
    /// to an operator is telling them which column to repair, and a message that
    /// reported both as empty would send them looking for a header state that is
    /// not the one on disk.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_header_identity_warns_naming_the_empty_column_for_a_degenerate_header() {
        let tv = crate::storage::test_support::TestValkey::setup().await;

        // A distinctive id, so `contains` cannot match a timestamp or a byte
        // count that happens to share its digits.
        const DEGENERATE_OBJECT_ID: u64 = 918_273;
        const HEALTHY_OBJECT_ID: u64 = 918_274;

        let mut degenerate: Vec<(&str, &str)> = HEALTHY_HEADER.to_vec();
        degenerate.retain(|(name, _)| *name != "session_id");
        degenerate.push(("session_id", ""));

        let (identity, warns) =
            capture_warns(identity_of(&tv, DEGENERATE_OBJECT_ID, &degenerate)).await;

        assert!(
            identity.is_none(),
            "precondition: this is the degenerate-header arm, and the WARN below belongs to it"
        );
        assert!(
            warns.contains(&DEGENERATE_OBJECT_ID.to_string()),
            "the WARN must name the object, or an operator reading it has no layer to go and \
             repair. Captured: {warns}"
        );
        assert!(
            warns.contains("session_id_empty=true"),
            "and it must name the column that was actually empty. Captured: {warns}"
        );
        assert!(
            warns.contains("layer_name_empty=false"),
            "and report the intact one as intact — a message that flagged both would misdirect \
             the repair. Captured: {warns}"
        );

        let ((), healthy_warns) = capture_warns(async {
            assert!(
                identity_of(&tv, HEALTHY_OBJECT_ID, HEALTHY_HEADER)
                    .await
                    .is_some(),
                "precondition: the control header must read back as present"
            );
        })
        .await;

        assert!(
            !healthy_warns.contains("session_id_empty"),
            "the control: a complete header must pass without this WARN. Emitting it on every \
             healthy read would make the signal worthless at fleet scale — the operator learns to \
             ignore it, and the one degenerate layer it exists for is lost in the noise. Matched \
             on the FIELD NAME rather than on total silence so an unrelated WARN from the client \
             library cannot fail the case for the wrong reason. Captured: {healthy_warns}"
        );
    }

    /// An absent header — the original `Ok(None)` — still reads as `None`.
    ///
    /// The fence on the round-2 widening: the new empty-identity arm must be
    /// ADDITIVE, and a refactor that replaced the `data.is_empty()` check with
    /// the identity one would pass every case above while turning a vanished
    /// layer into an `Err` or a fabricated `Some`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_header_identity_reports_an_absent_header_as_absent() {
        let tv = crate::storage::test_support::TestValkey::setup().await;

        let identity = ValkeySnapshotLoader {
            client: tv.client(),
            keys: tv.key_schema(),
        }
        .read_header_identity("7")
        .await
        .expect("an absent header is `Ok(None)`, never an error");

        assert!(identity.is_none());
    }

    // ── b.xje: sibling order through the checkpoint base ────────────────────
    //
    // The base carries each row at its STORED key (`plan_full_content`), and
    // `decode_checkpoint_ompf` rebuilds every children list by placing the rows
    // by those keys. A restored layer therefore has the live list only if the
    // stored keys order it — which the engine's appends, and the keys a
    // reorder rewrites, now guarantee.

    fn xje_change(
        parent_id: u64,
        section_id: u64,
        name: &str,
        order: u32,
    ) -> oodf_ffi::StructCommand {
        xje_typed_change(parent_id, section_id, name, 0, order)
    }

    /// [`xje_change`] with an explicit `section_type` (an `SdfSpecType`).
    fn xje_typed_change(
        parent_id: u64,
        section_id: u64,
        name: &str,
        section_type: u8,
        order: u32,
    ) -> oodf_ffi::StructCommand {
        oodf_ffi::StructCommand::DiffChangeSection(oodf_ffi::CommandDiffChangeSection {
            parent_id,
            section_id,
            section_name: name.to_string(),
            section_type,
            section_order: order,
        })
    }

    fn xje_create(parent_id: u64, section_id: u64, name: &str) -> oodf_ffi::StructCommand {
        oodf_ffi::StructCommand::CreateSection(oodf_ffi::CommandCreateSection {
            parent_id,
            section_id,
            section_name: name.to_string(),
            section_type: 0,
        })
    }

    /// Apply one non-reset delta of `cmds`.
    fn xje_apply(state: &mut ObjectState, cmds: Vec<oodf_ffi::StructCommand>) {
        crate::domain::delta::apply_delta(
            state,
            Some(oodf_ffi::OwnedDelta {
                is_diff: false,
                struct_commands: cmds,
                ..Default::default()
            }),
        )
        .expect("delta must apply");
    }

    /// The state the checkpoint round trip rebuilds: the real base encoder,
    /// then the real DR decoder.
    fn xje_round_trip(state: &ObjectState) -> ObjectState {
        let (ompf, _) =
            build_reset_ompf_from_state(state).expect("the state must assemble into a base");
        crate::checkpoint::apply::decode_checkpoint_ompf(state.object_id, &ompf)
            .expect("the base must decode")
    }

    /// Key saturation: under a parent whose largest key is `u32::MAX`, every
    /// further appended child is stored at `u32::MAX` too, and the ties keep
    /// arrival order — identically in the live list, in a fresh snapshot, in
    /// the checkpoint round trip, against an independent append model, and on
    /// every repeated run.
    ///
    /// The ids descend in arrival order (S 12, C1 11, C2 10), so arrival order
    /// and id order disagree: a reader that broke the key ties by section id
    /// would give `[C2, C1, S]` and fail here.
    ///
    /// Fails on the pre-change engine, which stored each create at key 0 and
    /// put both in front of the seeded child: `[C1, C2, S]`.
    #[test]
    fn appended_children_at_the_key_ceiling_keep_arrival_order_in_every_reader() {
        const P: u64 = 1;
        const S: u64 = 12;
        const C1: u64 = 11;
        const C2: u64 = 10;

        // The independent model: an append puts each child last, in the order
        // the commits arrive, whatever its key.
        let appended = vec![S, C1, C2];

        let mut runs = Vec::new();
        for _ in 0..3 {
            let mut state = ObjectState {
                object_id: 1,
                ..Default::default()
            };
            xje_apply(
                &mut state,
                vec![xje_change(0, P, "P", 0), xje_change(P, S, "S", u32::MAX)],
            );
            xje_apply(&mut state, vec![xje_create(P, C1, "C1")]);
            xje_apply(&mut state, vec![xje_create(P, C2, "C2")]);

            assert_eq!(
                state.children[&P], appended,
                "the live list keeps arrival order"
            );
            assert!(
                [S, C1, C2]
                    .iter()
                    .all(|id| state.sections[id].section_order == u32::MAX),
                "the append key saturates at u32::MAX"
            );

            let DiffPlan::FullContent { sections, .. } =
                DiffEngine::plan_faithful_snapshot(&state).expect("the snapshot must plan")
            else {
                panic!("a snapshot plans FullContent");
            };
            let snapshot: Vec<(u64, u32)> = sections
                .iter()
                .filter(|s| s.parent_id == P)
                .map(|s| (s.section_id, s.section_order))
                .collect();
            assert_eq!(
                snapshot,
                vec![(S, 0), (C1, 1), (C2, 2)],
                "a fresh snapshot emits the ties in arrival order, at their sibling indexes"
            );

            let restored = xje_round_trip(&state);
            assert_eq!(
                restored.children[&P], appended,
                "the checkpoint round trip keeps the ties in arrival order"
            );

            let (base, _) = build_reset_ompf_from_state(&state).expect("must assemble");
            runs.push((
                state.children[&P].clone(),
                restored.children[&P].clone(),
                base,
            ));
        }
        assert!(
            runs.windows(2).all(|w| w[0] == w[1]),
            "every run gives the same live list, restored list and base bytes"
        );
    }

    /// A partial reorder of one child list on a parent whose prims and
    /// attributes interleave survives the checkpoint round trip: P (a prim)
    /// lists `[A, p1, B, p2, C]` keyed 0..4, and reordering its prims by
    /// `[C, A]` gives `[C, p1, A, p2, B]` — the prims ordered among their own
    /// positions, the attributes in place. The reorder gives C, A and B their
    /// positions' keys (0, 2, 4), so the base's stored keys rebuild exactly the
    /// live list.
    ///
    /// Discriminated by: dropping the re-key's `section_order` write in
    /// `apply_reorder_children`; the restore then sorts by the old keys and
    /// rebuilds `[A, p1, B, p2, C]`.
    #[test]
    fn partial_reorder_of_one_child_list_survives_the_checkpoint_round_trip() {
        const PRIM: u8 = 6;
        const ATTRIBUTE: u8 = 1;
        const P: u64 = 1;
        const A: u64 = 10;
        const B: u64 = 11;
        const C: u64 = 12;
        const P1: u64 = 20;
        const P2: u64 = 21;

        let mut state = ObjectState {
            object_id: 1,
            ..Default::default()
        };
        xje_apply(
            &mut state,
            vec![
                xje_typed_change(0, P, "P", PRIM, 0),
                xje_typed_change(P, A, "A", PRIM, 0),
                xje_typed_change(P, P1, "p1", ATTRIBUTE, 1),
                xje_typed_change(P, B, "B", PRIM, 2),
                xje_typed_change(P, P2, "p2", ATTRIBUTE, 3),
                xje_typed_change(P, C, "C", PRIM, 4),
            ],
        );
        xje_apply(
            &mut state,
            vec![oodf_ffi::StructCommand::ReorderChildren(
                oodf_ffi::CommandReorderChildren {
                    section_id: P,
                    children_list_id: 0,
                    children_list: vec![C, A],
                },
            )],
        );
        assert_eq!(
            state.children[&P],
            vec![C, P1, A, P2, B],
            "precondition: the prims are reordered in their own positions"
        );

        let restored = xje_round_trip(&state);
        assert_eq!(
            restored.children[&P], state.children[&P],
            "the checkpoint round trip rebuilds the live list"
        );
    }

    // ── The test-only unprovable-cut seam on the production pin read ────────
    //
    // `ValkeyClient::arm_unprovable_checkpoint_cut_for_test` is the only route
    // by which a test reaches the checkpoint loop's tail-capture fallback, so
    // its contract is pinned on the production loader it bends: an armed
    // object's pin reads `u64::MAX` (above every cut, so the real read that
    // follows is judged regressed by the production arm), an absent header
    // still reads absent, and the seam is per object, shared by every clone
    // of the client, and undone by disarming.

    /// `HSET obj:hdr seq` for `object_id` and read it back through the
    /// production loader on `client`.
    async fn pin_through(
        tv: &crate::storage::test_support::TestValkey,
        client: &ValkeyClient,
        object_id: u64,
    ) -> Option<u64> {
        ValkeySnapshotLoader {
            client,
            keys: tv.key_schema(),
        }
        .read_header_seq(&object_id.to_string())
        .await
        .expect("a present, numeric header seq must not be an Err")
    }

    async fn set_header_seq(
        tv: &crate::storage::test_support::TestValkey,
        object_id: u64,
        seq: u64,
    ) {
        let key = tv.key_schema().obj_header(&object_id.to_string());
        let _: u64 = tv
            .client()
            .pool
            .hset(&key, [("seq", seq.to_string())])
            .await
            .expect("HSET obj:hdr must succeed");
    }

    /// Armed, the pin reads above every cut — on every clone of the client,
    /// since the checkpoint loop is handed its client by value — and only for
    /// the armed object; disarmed, it reads the real header again.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_header_seq_armed_unprovable_cut_seam_pins_above_every_cut_until_disarmed() {
        let tv = crate::storage::test_support::TestValkey::setup().await;
        const ARMED: u64 = 31;
        const UNARMED: u64 = 32;
        set_header_seq(&tv, ARMED, 12).await;
        set_header_seq(&tv, UNARMED, 7).await;
        let loop_client = tv.client().clone();

        assert_eq!(
            pin_through(&tv, &loop_client, ARMED).await,
            Some(12),
            "precondition: an unarmed pin reads the real header sequence"
        );

        tv.client().arm_unprovable_checkpoint_cut_for_test(ARMED);

        assert_eq!(
            pin_through(&tv, &loop_client, ARMED).await,
            Some(u64::MAX),
            "armed through one clone, the pin of the armed object reads above every cut on \
             another"
        );
        assert_eq!(
            pin_through(&tv, &loop_client, UNARMED).await,
            Some(7),
            "the seam is per object: another object's pin is untouched"
        );

        tv.client().disarm_unprovable_checkpoint_cut_for_test(ARMED);

        assert_eq!(
            pin_through(&tv, &loop_client, ARMED).await,
            Some(12),
            "disarmed, the pin reads the real header sequence again"
        );
    }

    /// The seam forces a regression, never a deletion: an armed object whose
    /// header is absent still pins `None`, so the assembler drops it as
    /// `NotFound` exactly as it would unarmed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_header_seq_armed_unprovable_cut_seam_leaves_an_absent_header_absent() {
        let tv = crate::storage::test_support::TestValkey::setup().await;
        const ABSENT: u64 = 33;
        tv.client().arm_unprovable_checkpoint_cut_for_test(ABSENT);

        assert_eq!(pin_through(&tv, tv.client(), ABSENT).await, None);
    }
}
