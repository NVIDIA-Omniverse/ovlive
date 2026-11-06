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

//! The restore and accept-loss Jobs' object writes (SDD §22.22, *The restore
//! writer*; SR-9.6, SR-9.10, SR-10): the batched writer, its header and
//! clear-flags transactions, accept-loss's in-place value removal, the
//! retirement-carrier removal and the Job-owned wipe.
//!
//! The Job runs these only after its quiet period, under its lease, with its
//! write permit open (`dr::job::execute`). Each function here is ONE attempt:
//! the Job retries a step in place and decides what an outcome means; nothing
//! here retries, logs a decision, or makes a compensating write.
//!
//! # The batches ([`RestoreObjectWrite`])
//!
//! A restored object's rows are the units `queue_object_state` queues
//! (`StateUnit`), sliced greedily, in order, into batches of at most
//! [`RESTORE_BATCH_COMMANDS`] commands and at most
//! [`RESTORE_BATCH_ARG_BYTES`] of the arguments whose length follows the
//! state (`StateUnit::arg_bytes`). A unit is whole in one batch, except a
//! children list over the command limit, which is cut (below). A field or
//! time sample whose value alone is over the byte limit is a batch of its
//! own; no other unit can pass a limit. When every unit plus the stream
//! `UNLINK` and the header `HSET` fit both limits, the object is ONE
//! transaction: the units, then `UNLINK obj:stream`, then the header.
//! Otherwise the row batches go first and the header transaction — `UNLINK
//! obj:stream` and the header `HSET` — last.
//!
//! **A children list over the command limit** (more than
//! `RESTORE_BATCH_COMMANDS − 2` children) is cut at the limits like the rows
//! around it, its children in order: the first slice carries the list's
//! `DEL`, its first children's `RPUSH`es and its `obj:chs` `SADD`, filling
//! what the current batch has left; each later slice carries `LTRIM 0
//! <children already pushed − 1>` and the next children, a whole batch or
//! the rest, and the batch holding the last slice goes on with the units
//! after it. So no batch passes either limit but one holding a single value
//! over the byte limit. Cutting a list is safe because nothing reads it
//! before the header lands: the fleet is held, so every replica refuses
//! every `LiveService` RPC, and the Job writes the batches back to back, the
//! header last. A Job killed mid-list leaves no header, and the re-run wipes
//! the object and writes it again. The first slice indexes the list in
//! `obj:chs` with its `DEL`, so the wipe finds every list key a batch wrote;
//! a later slice's `LTRIM` makes it idempotent under the Job's in-place
//! retry of a torn or in-doubt batch, whose `EXEC` may have run: a bare
//! `RPUSH` would push its children twice. The commit path's materializer
//! (`queue_state_unit`) is unchanged: only this slicer cuts a list.
//!
//! **Why both limits.** Each command a batch queues holds client memory
//! until the batch's reply is read: 2.6–2.9 KB per command at small
//! arguments, plus about one byte per argument byte above them (record
//! `restore-memory-envelope-tv3-20261022`). The command limit bounds the
//! first; the byte limit the second, which a batch of large inline values
//! would otherwise grow without bound. Together they bound what one batch
//! holds by a constant, which the writer's memory charge carries
//! (`dr::job::execute::memory::DR_WRITE_CHARGE_BATCH_BYTES`). A unit over
//! the byte limit holds more, alone: the writer's peak heap grew 3.667 bytes
//! per byte of one such inline value, at 16, 64 and 96 MiB [measured; same
//! record, `raw/e-recal2-measure-inline16.jsonl`, `-inline64`, `-inline96`],
//! which the writer's charge prices apart, from the largest inline value the
//! planner recorded (`dr::job::execute::memory::inline_value_charge`).
//!
//! **Why the header goes last.** "Complete" (SR-9.10) is "the header is
//! structurally complete and not retiring", so a header that lands last
//! proves every row batch, the tail backfill and the stream band landed
//! before it. A re-run skips a complete object and rewrites a partial one.
//!
//! **Why the stream `UNLINK` rides the header transaction.** The stream and
//! the header share the `{oid}` slot. Kept until the header lands, a Job
//! killed before it still finds the stream and its high-water, so a re-run
//! bands at the same floor (PM r2 N4).
//!
//! Each batch's client timeout is `txn_timeout(command timeout, its
//! commands)` (`storage::client::txn_timeout`): the command families M1′
//! measured, every batch at most 10,000 commands, a thirtieth of its
//! smallest measured shape (300,003).
//!
//! # The fence on every restore transaction
//!
//! (The clear-flags write and the value removal are fenced on the header's
//! `seq` instead: [`seq_fenced_txn`].)
//!
//! `acquire_txn_for(obj:hdr)`, a [`WatchGuard`], `watch_with_baseline` on
//! `obj:hdr`, then `EXISTS obj:hdr` on the watched connection: a header
//! present answers [`FencedTxn::HeaderPresent`] and writes nothing. Then a
//! [`WatchedTxn`] opened with the batch's timeout, the batch queued, the L2
//! pre-dispatch check, the `EXEC`. So no batch lands over a header that
//! appeared after the wipe, and no header lands twice. A void `WATCH` ticks
//! `storage_watch_voided_total{path="dr_job"}`.
//!
//! The header transaction carries a `commit_chain` the writer generated once
//! for the object ([`RestoreObjectWrite::commit_chain`]): a present header
//! carrying it is the Job's own header from an in-doubt attempt, which the
//! Job counts as landed ([`read_header_commit_chain`]).
//!
//! # The clear-flags write ([`write_clear_flags_header`])
//!
//! One transaction on `{oid}`, fenced on `obj:hdr` reading `seq` at the
//! expected value with `superseded` or `ext_gc_scheduled` present: `UNLINK
//! obj:stream`, `HSET seq restore_seq_bump commit_chain`, `HDEL superseded
//! superseded_by ext_gc_scheduled`. Never the wipe: the survivor keeps its
//! content. The cursor columns are left as they are. It is the banded-header
//! write ([`write_banded_header`]) under [`BandedFence::Flagged`].
//!
//! # The value removal ([`ValueRemovalBatch`])
//!
//! Accept-loss removes the rows of the external values that are in no store
//! from a live object in place, never by a rewrite: the header stays
//! complete throughout, so a Job killed part-way finds the object complete,
//! its removed hashes still in `obj:ext`, and removes the rest. The Job walks
//! the value indices by descending rank pages (`dr::job::execute::discard`,
//! D5) and fills
//! batches as it reads: each row naming a removed value is deleted by its
//! member as read ([`RemovedRow`]: the key's `DEL` and the member's `ZREM`,
//! through the commit path's own delete definition,
//! `super::queue_value_row_delete`), deduped within the batch. Once every
//! row is gone, batches of `HDEL obj:ext <hash>` remove every removed hash
//! but the last, in hash order; the last hash's rides the header
//! transaction. The batches share the restore batches' two limits,
//! [`RESTORE_BATCH_COMMANDS`] and [`RESTORE_BATCH_ARG_BYTES`], a removed row
//! counting its member twice ([`RemovedRow::arg_bytes`]) and an `HDEL` its
//! hash's hex digits. Each batch is a transaction fenced on `obj:hdr`
//! reading `seq` at the value the Job read with no retirement flag
//! ([`remove_value_rows`]). Then the header, last: the banded-header write
//! under [`BandedFence::NotRetiring`] ([`write_value_removal_header`]),
//! which removes the stream, removes the last hash from `obj:ext`, stamps the
//! banded `seq` and adds the band to `restore_seq_bump`, so the next backup
//! of the object is a fresh base without the removed values. The batches and
//! both header writes share one fenced transaction ([`seq_fenced_txn`]).
//!
//! **Why the hashes leave `obj:ext` after the rows, the last with the
//! header.** Until the header lands at least one removed hash is still in
//! `obj:ext`, and until the last row batch lands every one is, so a Job
//! killed at any point before the header finds a removed value referenced
//! again, plans the removal again at the same `seq`, walks the indices from
//! the start and finishes it with the header. Had every `HDEL` landed before
//! the header, a kill just before it would leave nothing for a re-run to
//! find, the header unbanded, and the old backup still naming the removed
//! values. The header transaction stays three commands (`UNLINK`, `HSET`,
//! one `HDEL`).
//!
//! # The Job-owned wipe ([`wipe_object_for_restore`])
//!
//! Unfenced, and safe only because the Job is the store's one writer after
//! its quiet period: the paged teardown behind the server's retirement
//! (`storage::retirement::Teardown`: the header and every fixed key,
//! `obj:ext` included, first; then the rows through the five indices; then
//! the indices), then `obj:extret` and its attempts, and the stream only for
//! an abandoned delete ([`WipeStream`]). `checkpoint::residue`'s wipe cannot
//! serve here: it refuses a complete header, keeps `obj:ext` and the
//! retirement carrier, and reads whole indices.
//!
//! # No releases
//!
//! The external-value index plan is built over an empty pre-image, so it
//! owes no release; nothing here enqueues one or schedules an object's
//! external-value collection. The Job's authoritative recount, after its
//! last write (`storage::ext_reconcile::recount_authoritative`), owns every
//! count.

use std::collections::HashMap;
use std::time::Duration;

use fred::interfaces::{
    ClientLike, HashesInterface, KeysInterface, ListInterface, SetsInterface, SortedSetsInterface,
};
use fred::types::ClusterHash;

use super::{
    object_header_pairs, plan_materialization_ext_index, queue_state_unit, queue_value_row_delete,
    state_units, CheckpointCursorStamp, CommitBytes, ExtIndexPlan, QueuedTally, StateUnit,
    SurplusPolicy, ValueRow, HEADER_FIELD_COMMIT_CHAIN, HEADER_FIELD_SEQ, HEADER_FIELD_SUPERSEDED,
    HEADER_FIELD_SUPERSEDED_BY,
};
use crate::domain::types::ObjectState;
use crate::error::ServiceError;
use crate::observability::metrics::{
    WATCH_VOIDED_ABORTED_PRE_EXEC, WATCH_VOIDED_COMMITTED_UNWATCHED,
    WATCH_VOIDED_FINGERPRINT_UNAVAILABLE, WATCH_VOIDED_PATH_DR_JOB,
    WATCH_VOIDED_WITNESS_UNAVAILABLE,
};
use crate::storage::client::{
    map_snapshot_load_err, txn_timeout, FredErrOrigin, ValkeyClient, POOL_LABEL_DATA,
    POOL_LABEL_TXN,
};
use crate::storage::commit_chain::fresh_commit_chain;
use crate::storage::keys::{sha256_to_hex, KeySchema};
use crate::storage::retirement::{RetirementPaging, Teardown};
use crate::storage::watch_guard::WatchGuard;
use crate::storage::watched_txn::{
    dispatch_verdict, warn_fingerprint_unavailable, warn_witness_unavailable, watch_with_baseline,
    DispatchVerdict, WatchFingerprint, WatchedExec, WatchedTxn,
};

/// The most commands one restore batch queues: 10,000. Each holds 2.6–2.9 KB
/// of client memory until the batch's reply is read [measured: the writer's
/// peak heap stood 260.9 MB above its state with 100,000-command batches and
/// 67.9 MB with 25,000-command ones at 27,231 specs, and 29.1 MB with one
/// 9,999-command batch at 947 specs (the record's `batch` shape); record
/// `restore-memory-envelope-tv3-20261022`], so a full batch holds about
/// 29 MB, where 100,000 held about 260 MB. A thirtieth of the smallest shape
/// M1′ measured (300,003), so `txn_timeout`'s 120 µs per command is
/// justified for every batch without a new measurement (C3): no batch passes
/// it, a children list over it being cut across batches (module docs).
/// Accept-loss's value removal batches share it and
/// [`RESTORE_BATCH_ARG_BYTES`] ([`ValueRemovalBatch::fits`]).
pub(crate) const RESTORE_BATCH_COMMANDS: u64 = 10_000;

/// The most argument bytes one restore batch queues, by
/// `StateUnit::arg_bytes`: 4 MiB. Each such byte holds about one byte of
/// client memory while its batch is queued [measured: 1.00 per base64 byte
/// of the inline values, between 8-byte and 1,024-byte values at
/// 100,000-command batches; same record], so the byte limit bounds that term
/// at about 4 MiB. Without it a batch of 10,000 commands carrying large
/// inline values would hold their encoded bytes, up to the command limit.
/// Public for the restore memory record's tool, which prints it.
pub const RESTORE_BATCH_ARG_BYTES: u64 = 4 * 1024 * 1024;

/// A restore batch's two limits ([`RESTORE_BATCH_COMMANDS`] and
/// [`RESTORE_BATCH_ARG_BYTES`] in production).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BatchLimit {
    /// The most commands.
    pub commands: u64,
    /// The most argument bytes, by `StateUnit::arg_bytes`.
    pub arg_bytes: u64,
}

/// One unit's cost against a [`BatchLimit`]: its commands and its argument
/// bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct UnitCost {
    commands: u64,
    arg_bytes: u64,
}

impl UnitCost {
    fn of(unit: StateUnit<'_>) -> Self {
        Self {
            commands: unit.commands(),
            arg_bytes: unit.arg_bytes(),
        }
    }
}

/// One unit as the slicer sees it: its cost, and its children when it is a
/// children list, the one kind of unit the slicer may cut (module docs, *A
/// children list over the command limit*).
#[derive(Clone, Copy, Debug)]
struct SlicedUnit {
    cost: UnitCost,
    /// The children of a children-list unit; 0 for every other kind.
    children: u64,
}

impl SlicedUnit {
    fn of(unit: StateUnit<'_>) -> Self {
        let children = match unit {
            StateUnit::Children { ids, .. } => u64::try_from(ids.len()).unwrap_or(u64::MAX),
            _ => 0,
        };
        Self {
            cost: UnitCost::of(unit),
            children,
        }
    }
}

/// The commands a children list's first slice queues beside its `RPUSH`es:
/// the list's `DEL` and its `obj:chs` `SADD`, the unit's own head and tail.
const CHILDREN_FIRST_SLICE_COMMANDS: u64 = 2;

/// The commands a later slice of a children list queues beside its
/// `RPUSH`es: one `LTRIM` back to the children the earlier slices pushed,
/// which makes the slice idempotent under the Job's in-place retry.
const CHILDREN_LATER_SLICE_COMMANDS: u64 = 1;

/// The header `HSET` field of the guard band a restore applied.
const HEADER_FIELD_RESTORE_SEQ_BUMP: &str = "restore_seq_bump";

/// The header field the server sets when it schedules a retiring object's
/// external-value collection.
const HEADER_FIELD_EXT_GC_SCHEDULED: &str = "ext_gc_scheduled";

/// The header fields a restored object's header carries beside its state's.
#[derive(Clone, Debug)]
pub(crate) struct RestoreHeader {
    /// The session the restored layer belongs to.
    pub session_id: String,
    /// The layer.
    pub layer_name: String,
    /// The backup's `created_by`.
    pub created_by: String,
    /// The backup's `server_ts`: `created_at`, `modified_at` and
    /// `last_server_ts`.
    pub server_ts: u64,
    /// The cursor columns: `last_checkpoint_seq` = the restored head,
    /// `last_checkpoint_ms`, and `restore_seq_bump` = the band.
    pub cursor: CheckpointCursorStamp,
}

/// One row batch: units `[start, start + len)` of the object's units,
/// queueing `commands` commands and `arg_bytes` argument bytes
/// (`StateUnit::arg_bytes`). Every unit is whole but where a children list
/// over the command limit is cut: then the batch's first unit may start at
/// child `first_child` (a later slice of a list an earlier batch began), and
/// its last unit may stop before child `last_child_end` (a list a later batch
/// continues). A batch that is one middle slice of a list has both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchSpec {
    start: usize,
    len: usize,
    commands: u64,
    arg_bytes: u64,
    /// The child the batch's first unit starts at: 0 unless that unit is a
    /// children list an earlier batch began.
    first_child: u64,
    /// `Some(end)` when the batch's last unit is a children list a later
    /// batch continues: the batch pushes its children up to `end`.
    last_child_end: Option<u64>,
}

impl BatchSpec {
    /// An empty batch whose first unit is unit `start`, from child
    /// `first_child`.
    fn starting(start: usize, first_child: u64) -> Self {
        Self {
            start,
            len: 0,
            commands: 0,
            arg_bytes: 0,
            first_child,
            last_child_end: None,
        }
    }

    /// The children of the batch's `offset`-th unit it pushes, when that
    /// unit is a list of `children` children the batch carries only part
    /// of; `None` when it carries the whole unit.
    fn children_slice(&self, offset: usize, children: u64) -> Option<std::ops::Range<u64>> {
        let from = if offset == 0 { self.first_child } else { 0 };
        let to = if offset + 1 == self.len {
            self.last_child_end.unwrap_or(children)
        } else {
            children
        };
        (from != 0 || to != children).then_some(from..to)
    }
}

/// One restored object's write, planned: its rows sliced into batches, and
/// its header. Built once per object attempt over the decoded, banded and
/// flattened `state`; the Job writes [`Self::row_batches`] batches, then the
/// header transaction.
pub(crate) struct RestoreObjectWrite<'a> {
    object_id: u64,
    oid_str: String,
    state: &'a ObjectState,
    ext_plan: ExtIndexPlan,
    header: RestoreHeader,
    commit_chain: String,
    /// The row batches; empty when the object is one transaction.
    batches: Vec<BatchSpec>,
    /// Every unit's commands, summed.
    unit_commands: u64,
    /// Every unit's argument bytes, summed.
    unit_arg_bytes: u64,
    /// Whether the header transaction carries every unit (the object is one
    /// transaction).
    single: bool,
}

impl<'a> RestoreObjectWrite<'a> {
    /// Plan object `object_id`'s write of `state` with `header`, in batches
    /// of at most `batch_commands` commands ([`RESTORE_BATCH_COMMANDS`] in
    /// production; the Job's test seam varies it) and at most
    /// [`RESTORE_BATCH_ARG_BYTES`] argument bytes ([`Self::plan_within`]).
    pub(crate) fn plan(
        object_id: u64,
        state: &'a ObjectState,
        header: RestoreHeader,
        batch_commands: u64,
    ) -> Self {
        Self::plan_within(
            object_id,
            state,
            header,
            BatchLimit {
                commands: batch_commands,
                arg_bytes: RESTORE_BATCH_ARG_BYTES,
            },
        )
    }

    /// Plan object `object_id`'s write of `state` with `header`, in batches
    /// within `limit`. The external-value index is planned over an empty
    /// pre-image, so it owes no release.
    pub(crate) fn plan_within(
        object_id: u64,
        state: &'a ObjectState,
        header: RestoreHeader,
        limit: BatchLimit,
    ) -> Self {
        let ext_plan = plan_materialization_ext_index(
            object_id,
            state,
            &HashMap::new(),
            SurplusPolicy::Suppress,
        );
        debug_assert!(
            ext_plan.releases.is_empty(),
            "an index planned over an empty pre-image owes no release"
        );
        let units = state_units(state, &ext_plan).fold(UnitCost::default(), |sum, unit| {
            let unit = UnitCost::of(unit);
            UnitCost {
                commands: sum.commands.saturating_add(unit.commands),
                arg_bytes: sum.arg_bytes.saturating_add(unit.arg_bytes),
            }
        });
        // The header transaction's own: the stream `UNLINK` and the header
        // `HSET`, whose content-length arguments are its names.
        let single = units.commands.saturating_add(2) <= limit.commands
            && units
                .arg_bytes
                .saturating_add(header_arg_bytes(state, &header))
                <= limit.arg_bytes;
        let unit_commands = units.commands;
        let unit_arg_bytes = units.arg_bytes;
        let batches = if single {
            Vec::new()
        } else {
            slice_state_units(state_units(state, &ext_plan).map(SlicedUnit::of), limit)
        };
        Self {
            object_id,
            oid_str: object_id.to_string(),
            state,
            ext_plan,
            header,
            commit_chain: fresh_commit_chain(),
            batches,
            unit_commands,
            unit_arg_bytes,
            single,
        }
    }

    /// How many row batches precede the header transaction: 0 when the
    /// object is one transaction.
    pub(crate) fn row_batches(&self) -> usize {
        self.batches.len()
    }

    /// The commands row batch `batch` queues.
    pub(crate) fn row_batch_commands(&self, batch: usize) -> u64 {
        self.batches[batch].commands
    }

    /// The commands the header transaction queues: the stream `UNLINK` and
    /// the header `HSET`, plus every unit when the object is one
    /// transaction.
    pub(crate) fn header_commands(&self) -> u64 {
        let own = 2;
        if self.single {
            self.unit_commands.saturating_add(own)
        } else {
            own
        }
    }

    /// The argument bytes row batch `batch` queues (`StateUnit::arg_bytes`).
    pub(crate) fn row_batch_arg_bytes(&self, batch: usize) -> u64 {
        self.batches[batch].arg_bytes
    }

    /// The argument bytes the header transaction queues: the header `HSET`'s
    /// content-length arguments, plus every unit's when the object is one
    /// transaction.
    pub(crate) fn header_arg_bytes(&self) -> u64 {
        let own = header_arg_bytes(self.state, &self.header);
        if self.single {
            self.unit_arg_bytes.saturating_add(own)
        } else {
            own
        }
    }

    /// The `commit_chain` the header carries: generated once per planned
    /// write, so a present header carrying it is this write's.
    pub(crate) fn commit_chain(&self) -> &str {
        &self.commit_chain
    }

    /// The units of row batch `batch`, each with the children it pushes when
    /// the batch carries only a slice of a children list (`None`: the whole
    /// unit).
    fn batch_units(
        &self,
        batch: usize,
    ) -> impl Iterator<Item = (StateUnit<'_>, Option<std::ops::Range<u64>>)> {
        let spec = self.batches[batch];
        state_units(self.state, &self.ext_plan)
            .skip(spec.start)
            .take(spec.len)
            .enumerate()
            .map(move |(offset, unit)| {
                let slice = match unit {
                    StateUnit::Children { ids, .. } => {
                        spec.children_slice(offset, u64::try_from(ids.len()).unwrap_or(u64::MAX))
                    }
                    _ => None,
                };
                (unit, slice)
            })
    }
}

/// Test support: the transactions [`RestoreObjectWrite::plan`] plans for
/// object `object_id`'s `state` under a header naming `session_id`,
/// `layer_name` and `created_by`, in batches of at most `batch_commands`
/// commands and [`RESTORE_BATCH_ARG_BYTES`] argument bytes: each row batch's
/// `(commands, argument bytes)`, then the header transaction's. For the
/// restore memory record's tool, which reads the batches' argument bytes the
/// Job's report does not carry. Compiled out of release builds.
#[cfg(any(test, feature = "test-support"))]
pub fn planned_restore_transactions_for_test(
    object_id: u64,
    state: &ObjectState,
    session_id: &str,
    layer_name: &str,
    created_by: &str,
    batch_commands: u64,
) -> Vec<(u64, u64)> {
    let header = RestoreHeader {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
        created_by: created_by.to_owned(),
        server_ts: 0,
        cursor: CheckpointCursorStamp {
            snap_seq: 0,
            written_at_ms: 0,
            restore_seq_bump: 0,
        },
    };
    let write = RestoreObjectWrite::plan(object_id, state, header, batch_commands);
    (0..write.row_batches())
        .map(|batch| {
            (
                write.row_batch_commands(batch),
                write.row_batch_arg_bytes(batch),
            )
        })
        .chain(std::iter::once((
            write.header_commands(),
            write.header_arg_bytes(),
        )))
        .collect()
}

/// The header `HSET`'s arguments whose length follows the write's content:
/// the session id, the layer name, `created_by` and the source format.
fn header_arg_bytes(state: &ObjectState, header: &RestoreHeader) -> u64 {
    [
        header.session_id.len(),
        header.layer_name.len(),
        header.created_by.len(),
        state.source_format.len(),
    ]
    .into_iter()
    .fold(0u64, |sum, len| sum.saturating_add(len as u64))
}

/// Greedy slicing of units whose command counts are `commands`, in order,
/// into `(start, len, commands)` batches of at most `limit` commands, with no
/// byte limit: [`slice_units_within`] for units that carry no argument bytes.
#[cfg(test)]
fn slice_units(commands: impl Iterator<Item = u64>, limit: u64) -> Vec<BatchSpec> {
    slice_units_within(
        commands.map(|commands| UnitCost {
            commands,
            arg_bytes: 0,
        }),
        BatchLimit {
            commands: limit,
            arg_bytes: u64::MAX,
        },
    )
}

/// [`slice_state_units`] over units of the given costs, none of them a
/// children list: a batch ends where the next unit would take it over either
/// limit, and a unit over either limit is a batch of its own, whole.
#[cfg(test)]
fn slice_units_within(units: impl Iterator<Item = UnitCost>, limit: BatchLimit) -> Vec<BatchSpec> {
    slice_state_units(units.map(|cost| SlicedUnit { cost, children: 0 }), limit)
}

/// Greedy slicing of an object's units, in order, into batches within
/// `limit`: a batch ends where the next unit would take it over either
/// limit. Every unit is whole in one batch, except a children list over the
/// command limit, which is cut at the limits (module docs, *A children list
/// over the command limit*): its first slice carries the list's `DEL` and
/// `obj:chs` `SADD` and its first children, filling what the current batch
/// has left; each later slice carries an `LTRIM` and the next children, a
/// whole batch's worth, or the rest; the batch holding its last slice goes on
/// with the units after it. Any other unit over either limit is a batch of
/// its own: a field or time sample whose value alone is over the byte limit
/// (no other kind can pass the command limit). Under a command limit too
/// small for a list's own commands and one child, each slice still carries
/// one child, over the limit.
fn slice_state_units(units: impl Iterator<Item = SlicedUnit>, limit: BatchLimit) -> Vec<BatchSpec> {
    let mut batches = Vec::new();
    let mut current = BatchSpec::starting(0, 0);
    for (index, unit) in units.enumerate() {
        let SlicedUnit { cost, children } = unit;
        if children > 0 && cost.commands > limit.commands {
            // A children list over the command limit: cut it. Its slices
            // carry no content-length argument (a child is a decimal id), so
            // only the command limit cuts them.
            let mut from = 0u64;
            while from < children {
                let own = if from == 0 {
                    CHILDREN_FIRST_SLICE_COMMANDS
                } else {
                    CHILDREN_LATER_SLICE_COMMANDS
                };
                // A slice never joins a batch with no room for its own
                // commands and one child, nor one a value over the byte
                // limit already holds by itself.
                if current.len > 0
                    && (current.commands.saturating_add(own + 1) > limit.commands
                        || current.arg_bytes > limit.arg_bytes)
                {
                    batches.push(current);
                    current = BatchSpec::starting(index, from);
                }
                if current.len == 0 {
                    current = BatchSpec::starting(index, from);
                }
                let room = limit
                    .commands
                    .saturating_sub(current.commands.saturating_add(own))
                    .max(1);
                let to = from.saturating_add(room).min(children);
                current.len += 1;
                current.commands = current
                    .commands
                    .saturating_add(own)
                    .saturating_add(to - from);
                if to < children {
                    current.last_child_end = Some(to);
                    batches.push(current);
                    current = BatchSpec::starting(index, to);
                }
                from = to;
            }
            continue;
        }
        if current.len > 0
            && (current.commands.saturating_add(cost.commands) > limit.commands
                || current.arg_bytes.saturating_add(cost.arg_bytes) > limit.arg_bytes)
        {
            batches.push(current);
            current = BatchSpec::starting(index, 0);
        }
        if current.len == 0 {
            current = BatchSpec::starting(index, 0);
        }
        current.len += 1;
        current.commands = current.commands.saturating_add(cost.commands);
        current.arg_bytes = current.arg_bytes.saturating_add(cost.arg_bytes);
    }
    if current.len > 0 {
        batches.push(current);
    }
    batches
}

/// The commands the restore Job's writer queues for object `object_id`'s
/// `state` over an empty pre-image at the production limits, header and
/// stream `UNLINK` included (`MULTI` and `EXEC` not): what the planner
/// records in a lineage's footprint. Every unit's commands, plus one `LTRIM`
/// per later slice of a children list over [`RESTORE_BATCH_COMMANDS`].
pub(crate) fn restore_object_command_count(object_id: u64, state: &ObjectState) -> u64 {
    let ext_plan =
        plan_materialization_ext_index(object_id, state, &HashMap::new(), SurplusPolicy::Suppress);
    let units = state_units(state, &ext_plan).fold(0u64, |sum, u| sum.saturating_add(u.commands()));
    // Only a list over the command limit is cut, and only then are there
    // slices (and their `LTRIM`s) to count; an object that is one
    // transaction has none.
    let cut = state_units(state, &ext_plan).any(|unit| {
        matches!(unit, StateUnit::Children { .. }) && unit.commands() > RESTORE_BATCH_COMMANDS
    });
    let rows = if cut {
        slice_state_units(
            state_units(state, &ext_plan).map(SlicedUnit::of),
            BatchLimit {
                commands: RESTORE_BATCH_COMMANDS,
                arg_bytes: RESTORE_BATCH_ARG_BYTES,
            },
        )
        .iter()
        .fold(0u64, |sum, batch| sum.saturating_add(batch.commands))
    } else {
        units
    };
    rows.saturating_add(2)
}

/// What one fenced restore transaction did.
#[derive(Debug)]
pub(crate) enum FencedTxn {
    /// The `EXEC` committed.
    Committed,
    /// `obj:hdr` was present under the `WATCH`, before the `MULTI`: nothing
    /// was written.
    HeaderPresent,
    /// The `EXEC` tripped on the `WATCH`: something wrote `obj:hdr` between
    /// the `EXISTS` and the `EXEC`, and nothing of this transaction ran.
    Conflict,
    /// The `EXEC` ran and reported an error frame, or the server discarded
    /// the transaction: which frames ran is not knowable. Each unit is
    /// idempotent, so a retry converges.
    Torn(String),
    /// The reply never came back: whether the `EXEC` ran is unknown.
    InDoubt(String),
}

/// What a fenced restore transaction queues.
enum TxnBody<'w, 'a> {
    /// Row batch `batch`.
    Rows(&'w RestoreObjectWrite<'a>, usize),
    /// The header transaction: the units first when the object is one
    /// transaction, then `UNLINK obj:stream`, then the header `HSET`.
    Header(&'w RestoreObjectWrite<'a>),
}

/// Write row batch `batch` of `write`: one attempt of one fenced
/// transaction ([`FencedTxn`]). `Err` for a store error before the `EXEC`
/// (transient ones are `Aborted` or `Unavailable`).
pub(crate) async fn write_restore_batch(
    client: &ValkeyClient,
    keys: &KeySchema,
    write: &RestoreObjectWrite<'_>,
    batch: usize,
) -> Result<FencedTxn, ServiceError> {
    fenced_restore_txn(client, keys, TxnBody::Rows(write, batch)).await
}

/// Write `write`'s header transaction (with every unit, when the object is
/// one transaction): one attempt, as [`write_restore_batch`].
pub(crate) async fn write_restore_header(
    client: &ValkeyClient,
    keys: &KeySchema,
    write: &RestoreObjectWrite<'_>,
) -> Result<FencedTxn, ServiceError> {
    fenced_restore_txn(client, keys, TxnBody::Header(write)).await
}

/// The `commit_chain` object `object_id`'s header carries, `None` when the
/// header or the field is absent. One `HGET` on the data pool.
pub(crate) async fn read_header_commit_chain(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
) -> Result<Option<String>, ServiceError> {
    client
        .pool
        .next()
        .hget(
            keys.obj_header(&object_id.to_string()),
            HEADER_FIELD_COMMIT_CHAIN,
        )
        .await
        .map_err(|e| {
            map_snapshot_load_err(
                FredErrOrigin::client(client, POOL_LABEL_DATA),
                "HGET obj header commit_chain (restore Job)",
                e,
            )
        })
}

/// Count and log a `WATCH` this attempt could not fully arm or keep.
fn voided(client: &ValkeyClient, disposition: &'static str) {
    if let Some(metrics) = client.metrics_snapshot() {
        metrics.inc_storage_watch_voided(WATCH_VOIDED_PATH_DR_JOB, disposition);
    }
}

/// One fenced restore transaction (module docs, *The fence on every restore
/// transaction*).
async fn fenced_restore_txn(
    client: &ValkeyClient,
    keys: &KeySchema,
    body: TxnBody<'_, '_>,
) -> Result<FencedTxn, ServiceError> {
    let write = match &body {
        TxnBody::Rows(write, _) | TxnBody::Header(write) => *write,
    };
    let oid_str = write.oid_str.as_str();
    let header_key = keys.obj_header(oid_str);
    let txn_origin = || FredErrOrigin::client(client, POOL_LABEL_TXN);

    let checkout = client.acquire_txn_for(&header_key).await;
    let command_timeout = checkout.perf_config().default_command_timeout;
    let mut conn = WatchGuard::new(checkout, client.watch_unwatch_tracker())
        .with_metrics(client.metrics_snapshot());
    let watch_hash = ClusterHash::from(header_key.as_str());
    let baseline = watch_with_baseline(&conn, &watch_hash, &[&header_key])
        .await
        .map_err(|e| map_snapshot_load_err(txn_origin(), "WATCH obj header (restore Job)", e))?;
    conn.mark_watched(&header_key, &[]);
    if let Some(reason) = baseline.unavailable_reason() {
        voided(client, WATCH_VOIDED_WITNESS_UNAVAILABLE);
        warn_witness_unavailable(WATCH_VOIDED_PATH_DR_JOB, reason);
    }
    let fingerprint = WatchFingerprint::take(&conn, &watch_hash);

    // The fence: the header must be absent, read on the watched connection.
    let present: u64 = match conn.exists(&header_key).await {
        Ok(present) => present,
        Err(e) => {
            let _ = conn.unwatch_pinned().await;
            return Err(map_snapshot_load_err(
                txn_origin(),
                "EXISTS obj header (restore Job fence)",
                e,
            ));
        }
    };
    if present > 0 {
        let _ = conn.unwatch_pinned().await;
        return Ok(FencedTxn::HeaderPresent);
    }

    let commands = match &body {
        TxnBody::Rows(write, batch) => write.row_batch_commands(*batch),
        TxnBody::Header(write) => write.header_commands(),
    };
    let timeout: Duration = txn_timeout(command_timeout, commands);
    let txn = match WatchedTxn::open_with_timeout(&conn, &watch_hash, &baseline, timeout).await {
        Ok(txn) => txn,
        Err(e) => {
            let _ = conn.unwatch_pinned().await;
            return Err(map_snapshot_load_err(
                txn_origin(),
                "MULTI queue (restore Job)",
                e,
            ));
        }
    };
    if let Err(e) = queue_restore_txn(txn_origin(), txn.commands(), keys, &body).await {
        let _ = conn.unwatch_pinned().await;
        return Err(e);
    }

    match dispatch_verdict(&conn, &fingerprint, &watch_hash) {
        DispatchVerdict::Dispatch => {}
        DispatchVerdict::DispatchUncomparable => {
            voided(client, WATCH_VOIDED_FINGERPRINT_UNAVAILABLE);
            warn_fingerprint_unavailable(WATCH_VOIDED_PATH_DR_JOB);
        }
        DispatchVerdict::Refuse => {
            let _ = conn.unwatch_pinned().await;
            voided(client, WATCH_VOIDED_ABORTED_PRE_EXEC);
            return Err(ServiceError::Unavailable(format!(
                "the restore Job's write of object {} was aborted before EXEC: the WATCH \
                 connection was replaced, so the obj:hdr fence would not have held; nothing was \
                 written",
                write.object_id
            )));
        }
    }

    match txn.exec(txn_origin()).await {
        WatchedExec::Committed { witness } => {
            conn.mark_unwatched();
            if baseline.contradicted_by(witness) {
                voided(client, WATCH_VOIDED_COMMITTED_UNWATCHED);
                tracing::warn!(
                    object_id = write.object_id,
                    "the restore Job's write of object {} ran on a connection that never held its \
                     WATCH on the header; the header fence did not hold for this write",
                    write.object_id
                );
            }
            Ok(FencedTxn::Committed)
        }
        WatchedExec::Conflict => {
            conn.mark_unwatched();
            Ok(FencedTxn::Conflict)
        }
        // The guard stays armed: an errored or discarded `EXEC` does not
        // prove the watch set was cleared, and `Drop` unwatches.
        WatchedExec::ExecError(e) | WatchedExec::Discarded { exec: e, .. } => {
            Ok(FencedTxn::Torn(e.to_string()))
        }
        WatchedExec::InDoubt(e) => Ok(FencedTxn::InDoubt(e.to_string())),
    }
}

/// Queue the commands of one restore transaction into `trx`: a row batch's
/// units, a children list's slice through [`queue_children_slice`] and every
/// whole unit through `queue_state_unit`; or, for the header transaction,
/// every unit when the object is one
/// transaction, then the stream `UNLINK`, then the header `HSET`, last.
/// [`fenced_restore_txn`] queues through it into its open transaction, and
/// the unit tests into a never-dialled one, so both count the same
/// commands.
async fn queue_restore_txn<C>(
    origin: FredErrOrigin<'_>,
    trx: &C,
    keys: &KeySchema,
    body: &TxnBody<'_, '_>,
) -> Result<(), ServiceError>
where
    C: HashesInterface + KeysInterface + SetsInterface + SortedSetsInterface + ListInterface,
{
    let write = match body {
        TxnBody::Rows(write, _) | TxnBody::Header(write) => *write,
    };
    let oid_str = write.oid_str.as_str();
    let ext_index_key = keys.obj_ext_index(oid_str);
    match body {
        TxnBody::Rows(write, batch) => {
            for (unit, slice) in write.batch_units(*batch) {
                match (unit, slice) {
                    (StateUnit::Children { parent, ids }, Some(slice)) => {
                        queue_children_slice(origin, trx, keys, oid_str, parent, ids, slice)
                            .await?;
                    }
                    (unit, _) => {
                        queue_state_unit(origin, trx, keys, oid_str, &ext_index_key, unit).await?;
                    }
                }
            }
        }
        TxnBody::Header(write) => {
            if write.single {
                for unit in state_units(write.state, &write.ext_plan) {
                    queue_state_unit(origin, trx, keys, oid_str, &ext_index_key, unit).await?;
                }
            }
            let _: () = trx
                .unlink(keys.layer_stream(write.object_id))
                .await
                .map_err(|e| map_snapshot_load_err(origin, "UNLINK obj stream queue", e))?;
            let header = &write.header;
            let pairs = object_header_pairs(
                write.state,
                &header.session_id,
                &header.layer_name,
                &header.created_by,
                header.server_ts,
                Some(header.cursor),
                write.commit_chain.clone(),
            );
            let _: () = trx
                .hset(keys.obj_header(oid_str), pairs)
                .await
                .map_err(|e| map_snapshot_load_err(origin, "HSET obj header queue", e))?;
        }
    }
    Ok(())
}

/// Queue one slice of a children list cut across row batches (module docs,
/// *A children list over the command limit*): children `slice` of `ids`,
/// the list of section `parent`. The first slice (from child 0) queues what
/// the whole unit's head and tail are, the list's `DEL` and its `obj:chs`
/// `SADD`, around its `RPUSH`es, so every list key a batch writes is indexed
/// for the wipe; a later slice queues `LTRIM 0 <start − 1>` before its
/// `RPUSH`es, so the list holds exactly the children the earlier slices
/// pushed when they land, even after an attempt of this slice whose `EXEC`
/// ran but whose reply was lost: the Job retries a batch in place, and a
/// bare `RPUSH` would append its children twice.
async fn queue_children_slice<C>(
    origin: FredErrOrigin<'_>,
    trx: &C,
    keys: &KeySchema,
    oid_str: &str,
    parent: u64,
    ids: &[u64],
    slice: std::ops::Range<u64>,
) -> Result<(), ServiceError>
where
    C: KeysInterface + SetsInterface + ListInterface,
{
    let parent_str = parent.to_string();
    let children_key = keys.obj_children(oid_str, &parent_str);
    let start = usize::try_from(slice.start).unwrap_or(usize::MAX);
    let end = usize::try_from(slice.end).unwrap_or(usize::MAX);
    if start == 0 {
        let _: () = trx
            .del(&children_key)
            .await
            .map_err(|e| map_snapshot_load_err(origin, "DEL children list queue", e))?;
    } else {
        let keep = i64::try_from(slice.start - 1).unwrap_or(i64::MAX);
        let _: () = trx
            .ltrim(&children_key, 0, keep)
            .await
            .map_err(|e| map_snapshot_load_err(origin, "LTRIM children list queue", e))?;
    }
    for child_id in &ids[start..end] {
        let _: () = trx
            .rpush(&children_key, child_id.to_string())
            .await
            .map_err(|e| map_snapshot_load_err(origin, "RPUSH children queue", e))?;
    }
    if start == 0 {
        let _: () = trx
            .sadd(keys.obj_children_index(oid_str), &parent_str)
            .await
            .map_err(|e| map_snapshot_load_err(origin, "SADD children index queue", e))?;
    }
    Ok(())
}

/// The commands the clear-flags transaction queues: the stream `UNLINK`,
/// the header `HSET` and its `HDEL`. Its timeout is `txn_timeout(command
/// timeout, 3)`.
pub(crate) const CLEAR_FLAGS_COMMANDS: u64 = 3;

/// The commands accept-loss's value-removal header transaction queues: the
/// stream `UNLINK`, the header `HSET` and the last removed hash's `HDEL
/// obj:ext` ([`BandedHeaderWrite::removing_ext_hash`]). Its timeout is
/// `txn_timeout(command timeout, 3)`. A constant, whatever the removal's
/// size: the header carries one hash's `HDEL`, never a group's rows.
pub(crate) const VALUE_REMOVAL_HEADER_COMMANDS: u64 = 3;

/// What the banded-header write's fence requires of the header beside `seq`
/// at the expected value ([`write_banded_header`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BandedFence {
    /// A flagged survivor: `superseded` or `ext_gc_scheduled` present. The
    /// write also clears the flags (the restore writer's flag clear).
    Flagged,
    /// A live object: neither flag present. The write leaves the flags
    /// alone (accept-loss's value removal, which must never touch a
    /// retiring object).
    NotRetiring,
}

/// One banded-header write (module docs, *The clear-flags write* and *The
/// value removal*): the header moves from `expected_seq` to `seeded_seq`,
/// adding `band` to `restore_seq_bump`.
#[derive(Clone, Debug)]
pub(crate) struct BandedHeaderWrite {
    /// The object.
    pub object_id: u64,
    /// The `seq` the header must carry.
    pub expected_seq: u64,
    /// The banded `seq` to stamp.
    pub seeded_seq: u64,
    /// The band, added to the header's `restore_seq_bump`.
    pub band: u64,
    /// The `commit_chain` to stamp, generated once per write.
    pub commit_chain: String,
    /// The value removal's last hash, removed from `obj:ext` in the same
    /// transaction (module docs, *The value removal*); `None` for the flag
    /// clear.
    pub ext_hash: Option<[u8; 32]>,
}

impl BandedHeaderWrite {
    /// A banded-header write of `object_id` from `expected_seq` to
    /// `seeded_seq`, adding `band`, with a fresh `commit_chain`.
    pub(crate) fn new(object_id: u64, expected_seq: u64, seeded_seq: u64, band: u64) -> Self {
        Self {
            object_id,
            expected_seq,
            seeded_seq,
            band,
            commit_chain: fresh_commit_chain(),
            ext_hash: None,
        }
    }

    /// The write also removing `hash` from `obj:ext` (`HDEL`), in its own
    /// transaction: the value removal's last hash, in hash order.
    pub(crate) fn removing_ext_hash(mut self, hash: Option<[u8; 32]>) -> Self {
        self.ext_hash = hash;
        self
    }

    /// The commands the write queues under `fence`: the flag clear's
    /// [`CLEAR_FLAGS_COMMANDS`], or the value removal's `UNLINK` and `HSET`;
    /// plus the `obj:ext` `HDEL` when it carries one (the value removal's
    /// [`VALUE_REMOVAL_HEADER_COMMANDS`]).
    pub(crate) fn commands(&self, fence: BandedFence) -> u64 {
        match (fence, self.ext_hash.is_some()) {
            (BandedFence::Flagged, hdel) => CLEAR_FLAGS_COMMANDS + u64::from(hdel),
            (BandedFence::NotRetiring, true) => VALUE_REMOVAL_HEADER_COMMANDS,
            (BandedFence::NotRetiring, false) => VALUE_REMOVAL_HEADER_COMMANDS - 1,
        }
    }
}

/// The restore writer's flag clear is a [`BandedHeaderWrite`] under
/// [`BandedFence::Flagged`].
pub(crate) type ClearFlagsWrite = BandedHeaderWrite;

/// What one `seq`-fenced attempt did: a banded-header write's, or a value
/// removal batch's.
#[derive(Debug)]
pub(crate) enum SeqFencedOutcome {
    /// The `EXEC` committed.
    Committed,
    /// A banded-header write only: the header already carries this write —
    /// `seq` is the banded value, no flag is set and `commit_chain` is this
    /// write's (an in-doubt attempt landed).
    AlreadyLanded,
    /// The header is not the object the plan saw, nor this write's result:
    /// nothing was written.
    FenceMismatch {
        /// The header's `seq`, as read.
        seq: Option<String>,
        /// Whether `superseded` or `ext_gc_scheduled` is present.
        flagged: bool,
    },
    /// The `EXEC` tripped on the `WATCH`.
    Conflict,
    /// The `EXEC` ran with an error frame, or was discarded.
    Torn(String),
    /// The reply never came back.
    InDoubt(String),
}

/// The flag clear's outcome is a [`SeqFencedOutcome`].
pub(crate) type ClearFlagsOutcome = SeqFencedOutcome;

/// Clear a kept survivor's retirement flags with its banded `seq`: one
/// attempt of one fenced transaction on `{oid}` (module docs).
/// [`write_banded_header`] under [`BandedFence::Flagged`].
pub(crate) async fn write_clear_flags_header(
    client: &ValkeyClient,
    keys: &KeySchema,
    write: &ClearFlagsWrite,
) -> Result<ClearFlagsOutcome, ServiceError> {
    write_banded_header(client, keys, write, BandedFence::Flagged).await
}

/// Stamp a value-removed object's banded `seq` and remove its last hash from
/// `obj:ext` (accept-loss's D5, after its row and `obj:ext` batches): one attempt of one
/// fenced transaction on `{oid}`. [`write_banded_header`] under
/// [`BandedFence::NotRetiring`].
pub(crate) async fn write_value_removal_header(
    client: &ValkeyClient,
    keys: &KeySchema,
    write: &BandedHeaderWrite,
) -> Result<SeqFencedOutcome, ServiceError> {
    write_banded_header(client, keys, write, BandedFence::NotRetiring).await
}

/// One banded-header write: one fenced transaction on `{oid}`, fenced on
/// `obj:hdr` reading `seq` at `write.expected_seq` and on `fence`: `UNLINK
/// obj:stream`, `HSET seq restore_seq_bump commit_chain`, under
/// [`BandedFence::Flagged`] `HDEL superseded superseded_by
/// ext_gc_scheduled`, and with [`BandedHeaderWrite::ext_hash`] `HDEL obj:ext
/// <hash>`. The cursor columns are left as they are. A header
/// already carrying this write answers [`SeqFencedOutcome::AlreadyLanded`].
/// `Err` for a store error before the `EXEC`.
pub(crate) async fn write_banded_header(
    client: &ValkeyClient,
    keys: &KeySchema,
    write: &BandedHeaderWrite,
    fence: BandedFence,
) -> Result<SeqFencedOutcome, ServiceError> {
    seq_fenced_txn(
        client,
        keys,
        write.object_id,
        write.expected_seq,
        fence,
        SeqFencedBody::Header(write),
    )
    .await
}

/// What a `seq`-fenced transaction queues.
enum SeqFencedBody<'w> {
    /// A banded-header write.
    Header(&'w BandedHeaderWrite),
    /// One batch of a value removal: rows, `obj:ext` `HDEL`s, or both.
    Rows(&'w ValueRemovalBatch),
}

/// One `seq`-fenced transaction on `{oid}`: `acquire_txn_for(obj:hdr)`, a
/// [`WatchGuard`], `watch_with_baseline` on `obj:hdr`, an `HMGET` of `seq`,
/// the flags, `restore_seq_bump` and `commit_chain` on the watched
/// connection, the fence (`seq` at `expected_seq`, the flags as `fence`
/// requires), then a [`WatchedTxn`] with `body` queued, the L2 pre-dispatch
/// check and the `EXEC`.
async fn seq_fenced_txn(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
    expected_seq: u64,
    fence: BandedFence,
    body: SeqFencedBody<'_>,
) -> Result<SeqFencedOutcome, ServiceError> {
    let oid_str = object_id.to_string();
    let header_key = keys.obj_header(&oid_str);
    let txn_origin = || FredErrOrigin::client(client, POOL_LABEL_TXN);
    // What the error labels and lines name: the clear-flags write's, as the
    // restore writer has always named it, or the value removal's.
    let (watch_label, read_label, multi_label, what) = match &body {
        SeqFencedBody::Header(_) if fence == BandedFence::Flagged => (
            "WATCH obj header (restore Job clear-flags)",
            "HMGET obj header (restore Job clear-flags fence)",
            "MULTI queue (restore Job clear-flags)",
            "the restore Job's clear-flags write",
        ),
        SeqFencedBody::Header(_) => (
            "WATCH obj header (accept-loss value removal)",
            "HMGET obj header (accept-loss value-removal fence)",
            "MULTI queue (accept-loss value removal)",
            "the accept-loss Job's value-removal header write",
        ),
        SeqFencedBody::Rows(..) => (
            "WATCH obj header (accept-loss value removal)",
            "HMGET obj header (accept-loss value-removal fence)",
            "MULTI queue (accept-loss value removal)",
            "the accept-loss Job's value-removal batch",
        ),
    };

    let checkout = client.acquire_txn_for(&header_key).await;
    let commands = match &body {
        SeqFencedBody::Header(write) => write.commands(fence),
        SeqFencedBody::Rows(batch) => batch.commands(),
    };
    let timeout = txn_timeout(checkout.perf_config().default_command_timeout, commands);
    let mut conn = WatchGuard::new(checkout, client.watch_unwatch_tracker())
        .with_metrics(client.metrics_snapshot());
    let watch_hash = ClusterHash::from(header_key.as_str());
    let baseline = watch_with_baseline(&conn, &watch_hash, &[&header_key])
        .await
        .map_err(|e| map_snapshot_load_err(txn_origin(), watch_label, e))?;
    conn.mark_watched(&header_key, &[]);
    if let Some(reason) = baseline.unavailable_reason() {
        voided(client, WATCH_VOIDED_WITNESS_UNAVAILABLE);
        warn_witness_unavailable(WATCH_VOIDED_PATH_DR_JOB, reason);
    }
    let fingerprint = WatchFingerprint::take(&conn, &watch_hash);

    let read: Vec<Option<String>> = match conn
        .hmget(
            &header_key,
            vec![
                HEADER_FIELD_SEQ,
                HEADER_FIELD_SUPERSEDED,
                HEADER_FIELD_EXT_GC_SCHEDULED,
                HEADER_FIELD_RESTORE_SEQ_BUMP,
                HEADER_FIELD_COMMIT_CHAIN,
            ],
        )
        .await
    {
        Ok(read) => read,
        Err(e) => {
            let _ = conn.unwatch_pinned().await;
            return Err(map_snapshot_load_err(txn_origin(), read_label, e));
        }
    };
    let field = |i: usize| read.get(i).cloned().flatten();
    let seq = field(0);
    let flagged = field(1).is_some() || field(2).is_some();
    let seq_number = seq.as_deref().and_then(|s| s.trim().parse::<u64>().ok());
    let wants_flagged = fence == BandedFence::Flagged;
    if seq_number != Some(expected_seq) || flagged != wants_flagged {
        let _ = conn.unwatch_pinned().await;
        if let SeqFencedBody::Header(write) = &body {
            if seq_number == Some(write.seeded_seq)
                && !flagged
                && field(4).as_deref() == Some(write.commit_chain.as_str())
            {
                return Ok(SeqFencedOutcome::AlreadyLanded);
            }
        }
        return Ok(SeqFencedOutcome::FenceMismatch { seq, flagged });
    }
    let prior_bump = field(3)
        .and_then(|b| b.trim().parse::<u64>().ok())
        .unwrap_or(0);

    let txn = match WatchedTxn::open_with_timeout(&conn, &watch_hash, &baseline, timeout).await {
        Ok(txn) => txn,
        Err(e) => {
            let _ = conn.unwatch_pinned().await;
            return Err(map_snapshot_load_err(txn_origin(), multi_label, e));
        }
    };
    let queued: Result<(), ServiceError> = async {
        match &body {
            SeqFencedBody::Header(write) => {
                let _: () = txn
                    .commands()
                    .unlink(keys.layer_stream(write.object_id))
                    .await
                    .map_err(|e| {
                        map_snapshot_load_err(txn_origin(), "UNLINK obj stream queue", e)
                    })?;
                let _: () = txn
                    .commands()
                    .hset(
                        &header_key,
                        [
                            (HEADER_FIELD_SEQ, write.seeded_seq.to_string()),
                            (
                                HEADER_FIELD_RESTORE_SEQ_BUMP,
                                prior_bump.saturating_add(write.band).to_string(),
                            ),
                            (HEADER_FIELD_COMMIT_CHAIN, write.commit_chain.clone()),
                        ],
                    )
                    .await
                    .map_err(|e| map_snapshot_load_err(txn_origin(), "HSET obj header queue", e))?;
                if fence == BandedFence::Flagged {
                    let _: () = txn
                        .commands()
                        .hdel(
                            &header_key,
                            vec![
                                HEADER_FIELD_SUPERSEDED,
                                HEADER_FIELD_SUPERSEDED_BY,
                                HEADER_FIELD_EXT_GC_SCHEDULED,
                            ],
                        )
                        .await
                        .map_err(|e| {
                            map_snapshot_load_err(txn_origin(), "HDEL obj header queue", e)
                        })?;
                }
                if let Some(hash) = &write.ext_hash {
                    let _: () = txn
                        .commands()
                        .hdel(keys.obj_ext_index(&oid_str), sha256_to_hex(hash))
                        .await
                        .map_err(|e| {
                            map_snapshot_load_err(txn_origin(), "HDEL obj ext queue", e)
                        })?;
                }
                Ok(())
            }
            SeqFencedBody::Rows(batch) => {
                queue_value_removal_batch(txn_origin(), txn.commands(), keys, batch).await
            }
        }
    }
    .await;
    if let Err(e) = queued {
        let _ = conn.unwatch_pinned().await;
        return Err(e);
    }

    match dispatch_verdict(&conn, &fingerprint, &watch_hash) {
        DispatchVerdict::Dispatch => {}
        DispatchVerdict::DispatchUncomparable => {
            voided(client, WATCH_VOIDED_FINGERPRINT_UNAVAILABLE);
            warn_fingerprint_unavailable(WATCH_VOIDED_PATH_DR_JOB);
        }
        DispatchVerdict::Refuse => {
            let _ = conn.unwatch_pinned().await;
            voided(client, WATCH_VOIDED_ABORTED_PRE_EXEC);
            return Err(ServiceError::Unavailable(format!(
                "{what} of object {object_id} was aborted before EXEC: the WATCH \
                 connection was replaced; nothing was written"
            )));
        }
    }

    match txn.exec(txn_origin()).await {
        WatchedExec::Committed { witness } => {
            conn.mark_unwatched();
            if baseline.contradicted_by(witness) {
                voided(client, WATCH_VOIDED_COMMITTED_UNWATCHED);
                tracing::warn!(
                    object_id,
                    "{what} of object {object_id} ran on a connection that never \
                     held its WATCH on the header"
                );
            }
            Ok(SeqFencedOutcome::Committed)
        }
        WatchedExec::Conflict => {
            conn.mark_unwatched();
            Ok(SeqFencedOutcome::Conflict)
        }
        WatchedExec::ExecError(e) | WatchedExec::Discarded { exec: e, .. } => {
            Ok(SeqFencedOutcome::Torn(e.to_string()))
        }
        WatchedExec::InDoubt(e) => Ok(SeqFencedOutcome::InDoubt(e.to_string())),
    }
}

/// One value row accept-loss's value removal deletes, named by its index
/// member exactly as the removal's `ZRANGE` page read it: a field row by its
/// `obj:flds` member, a time-sample row by its `obj:tss` member. Deleted by
/// that member ([`ValueRow::FieldMember`] / [`ValueRow::TimeSampleMember`]),
/// so a legacy `-0.0` sample's row is its own, apart from `+0.0`'s.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct RemovedRow {
    /// An `obj:tss` member (a time-sample row); else an `obj:flds` member.
    pub time_sample: bool,
    /// The member, as read.
    pub member: String,
}

impl RemovedRow {
    /// The row, borrowed, for its delete.
    fn as_row(&self) -> ValueRow<'_> {
        if self.time_sample {
            ValueRow::TimeSampleMember(&self.member)
        } else {
            ValueRow::FieldMember(&self.member)
        }
    }

    /// The argument bytes its delete queues whose length follows its
    /// content, counted from above: its member twice (the `DEL`'s row key
    /// carries it, and the `ZREM` names it). The key prefix is the
    /// per-command cost's, which the command limit prices.
    pub(crate) fn arg_bytes(&self) -> u64 {
        (self.member.len() as u64).saturating_mul(2)
    }
}

/// The argument bytes one `HDEL obj:ext <hash>` of the value removal queues
/// beside its key: the hash's 64 hex digits.
pub(crate) const VALUE_REMOVAL_HDEL_ARG_BYTES: u64 = 64;

/// One transaction of accept-loss's value removal (D5; module docs, *The
/// value removal*), before its header: the rows it deletes, each by its
/// member as read, deduped (a row pushed twice joins once), and
/// the hashes it removes from `obj:ext`. The removal fills it up to the
/// batch's two limits ([`Self::fits`]) and sends it fenced on `obj:hdr`
/// reading `seq` at `expected_seq` with no retirement flag
/// ([`remove_value_rows`]). Each command is idempotent, so a torn or
/// in-doubt batch is sent again as it is.
#[derive(Clone, Debug)]
pub(crate) struct ValueRemovalBatch {
    object_id: u64,
    oid_str: String,
    expected_seq: u64,
    rows: std::collections::BTreeSet<RemovedRow>,
    hdels: Vec<[u8; 32]>,
    arg_bytes: u64,
}

impl ValueRemovalBatch {
    /// An empty batch of object `object_id`'s removal under a header at
    /// `expected_seq`.
    pub(crate) fn new(object_id: u64, expected_seq: u64) -> Self {
        Self {
            object_id,
            oid_str: object_id.to_string(),
            expected_seq,
            rows: std::collections::BTreeSet::new(),
            hdels: Vec::new(),
            arg_bytes: 0,
        }
    }

    /// Whether the batch already deletes `row`.
    pub(crate) fn contains(&self, row: &RemovedRow) -> bool {
        self.rows.contains(row)
    }

    /// Whether `commands` more commands carrying `arg_bytes` more argument
    /// bytes keep the batch within `limit`. An empty batch takes anything,
    /// so one row over a limit still goes, alone.
    pub(crate) fn fits(&self, commands: u64, arg_bytes: u64, limit: BatchLimit) -> bool {
        self.is_empty()
            || (self.commands().saturating_add(commands) <= limit.commands
                && self.arg_bytes.saturating_add(arg_bytes) <= limit.arg_bytes)
    }

    /// Delete `row` too (two commands); a row it already deletes is not
    /// added again.
    pub(crate) fn push_row(&mut self, row: RemovedRow) {
        let bytes = row.arg_bytes();
        if self.rows.insert(row) {
            self.arg_bytes = self.arg_bytes.saturating_add(bytes);
        }
    }

    /// Remove `hash` from `obj:ext` too (one command).
    pub(crate) fn push_hdel(&mut self, hash: [u8; 32]) {
        self.hdels.push(hash);
        self.arg_bytes = self.arg_bytes.saturating_add(VALUE_REMOVAL_HDEL_ARG_BYTES);
    }

    /// The commands the batch queues: two per row (its key's `DEL` and its
    /// member's `ZREM`) and one per `HDEL`.
    pub(crate) fn commands(&self) -> u64 {
        2 * self.rows.len() as u64 + self.hdels.len() as u64
    }

    /// The rows it deletes.
    pub(crate) fn rows(&self) -> usize {
        self.rows.len()
    }

    /// Whether it queues nothing.
    pub(crate) fn is_empty(&self) -> bool {
        self.rows.is_empty() && self.hdels.is_empty()
    }

    /// Empty it, once it landed.
    pub(crate) fn clear(&mut self) {
        self.rows.clear();
        self.hdels.clear();
        self.arg_bytes = 0;
    }
}

/// Send `batch`: one attempt of one transaction fenced on `obj:hdr` reading
/// `seq` at the batch's expected value with no retirement flag
/// ([`SeqFencedOutcome`]; never `AlreadyLanded`). `Err` for a store error
/// before the `EXEC`.
pub(crate) async fn remove_value_rows(
    client: &ValkeyClient,
    keys: &KeySchema,
    batch: &ValueRemovalBatch,
) -> Result<SeqFencedOutcome, ServiceError> {
    seq_fenced_txn(
        client,
        keys,
        batch.object_id,
        batch.expected_seq,
        BandedFence::NotRetiring,
        SeqFencedBody::Rows(batch),
    )
    .await
}

/// Queue `batch` into `trx`: its rows' deletes through
/// `queue_value_row_delete` (the commit path's own delete definition, in its
/// member-as-read form), then its `HDEL obj:ext`s, so a hash leaves
/// `obj:ext` no earlier than the rows in the same transaction.
/// [`seq_fenced_txn`] queues through it, and the unit tests into a
/// never-dialled transaction.
async fn queue_value_removal_batch<C>(
    origin: FredErrOrigin<'_>,
    trx: &C,
    keys: &KeySchema,
    batch: &ValueRemovalBatch,
) -> Result<(), ServiceError>
where
    C: HashesInterface + KeysInterface + SetsInterface + SortedSetsInterface + ListInterface,
{
    let oid_str = batch.oid_str.as_str();
    let mut bytes = CommitBytes::default();
    let mut queued = QueuedTally::default();
    for row in &batch.rows {
        queue_value_row_delete(
            origin,
            trx,
            keys,
            oid_str,
            row.as_row(),
            &mut bytes,
            &mut queued,
        )
        .await?;
    }
    if !batch.hdels.is_empty() {
        let ext_index_key = keys.obj_ext_index(oid_str);
        for hash in &batch.hdels {
            let _: () = trx
                .hdel(&ext_index_key, sha256_to_hex(hash))
                .await
                .map_err(|e| map_snapshot_load_err(origin, "HDEL obj ext queue", e))?;
        }
    }
    Ok(())
}

/// `UNLINK obj:extret obj:extret:attempts` of `object_id` (one command; both
/// keys share the `{oid}` slot): the retirement carrier the clear-flags
/// write's survivor no longer owes. Idempotent.
pub(crate) async fn remove_retirement_carrier(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
) -> Result<(), ServiceError> {
    let oid_str = object_id.to_string();
    let _: u64 = client
        .pool
        .next()
        .unlink(vec![
            keys.obj_ext_retire_carrier(&oid_str),
            keys.obj_ext_retire_attempts(&oid_str),
        ])
        .await
        .map_err(|e| {
            map_snapshot_load_err(
                FredErrOrigin::client(client, POOL_LABEL_DATA),
                "UNLINK obj extret (restore Job)",
                e,
            )
        })?;
    Ok(())
}

/// Whether the Job-owned wipe removes the object's stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WipeStream {
    /// Keep it: the restore writer removes it with the header, so a Job
    /// killed before the header keeps its high-water (a restored object).
    Keep,
    /// Remove it: the object is not restored (an abandoned delete).
    Unlink,
}

/// The Job-owned, unfenced wipe of `object_id` (module docs): the paged
/// teardown (`storage::retirement::Teardown`, retrying a later page's
/// transient failure in place), then `obj:extret` and its attempts, and the
/// stream with [`WipeStream::Unlink`]. Idempotent: the header goes first and
/// the indices last, so a wipe stopped part-way is finished by the next one.
pub(crate) async fn wipe_object_for_restore(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
    stream: WipeStream,
) -> Result<(), ServiceError> {
    let oid_str = object_id.to_string();
    let origin = FredErrOrigin::client(client, POOL_LABEL_DATA);
    Teardown::new(&client.pool, keys, &oid_str, RetirementPaging::reference())
        .retrying_later_pages()
        .run()
        .await
        .map_err(|failure| map_snapshot_load_err(origin, &failure.label(), failure.error))?;
    let mut rest = vec![
        keys.obj_ext_retire_carrier(&oid_str),
        keys.obj_ext_retire_attempts(&oid_str),
    ];
    if stream == WipeStream::Unlink {
        rest.push(keys.layer_stream(object_id));
    }
    let _: u64 =
        client.pool.next().unlink(rest).await.map_err(|e| {
            map_snapshot_load_err(origin, "UNLINK obj extret (restore Job wipe)", e)
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    //! U5: the restore batches against what they queue. Each batch, and the
    //! header transaction, is queued through `queue_restore_txn`, the code
    //! `fenced_restore_txn` queues through, into a real fred `Transaction`
    //! over `Client::default()`, which never dials (queueing is a local buffer
    //! push; nothing here calls `exec`), so the transaction's `len()` is the
    //! truth the batch's command count is read against. The real
    //! transactions, fence included, are `fence_tests`' (below) and the
    //! integration tests' (`tests/integration/dr/restore_write_test.rs`).
    //! A children list over the command limit is cut across batches: the
    //! slicer's cut unit by unit, the slices a plan carries and what they
    //! queue, and the footprint's count of their `LTRIM`s; the cut list's
    //! content in a store, its retry and its wipe are `fence_tests`'. The
    //! batches' byte limit is pinned here too: the slicer and the
    //! one-transaction decision on both limits, production `plan`'s byte
    //! limit, `StateUnit::arg_bytes` per unit kind and against the arguments
    //! a unit queues, and both limits against their committed derivation
    //! record. A value removal's batch is read against what it queues too.

    use fred::interfaces::TransactionInterface;
    use ordered_float::OrderedFloat;

    use super::super::object_state_command_count;
    use super::*;
    use crate::domain::types::{Field, FieldValue, Section, TimeSample};
    use crate::storage::store_oom::StoreOomSignal;

    const OID: u64 = 42;

    fn schema() -> KeySchema {
        KeySchema::new("le:")
    }

    fn section(id: u64, parent: u64) -> Section {
        Section {
            section_id: id,
            parent_id: parent,
            name: format!("s{id}"),
            section_type: 6,
            section_order: 0,
            first_version: 1,
            last_version: 9,
        }
    }

    fn inline_field(section_id: u64, key: &str) -> Field {
        Field {
            section_id,
            key_name: key.to_owned(),
            value: FieldValue::Inline(vec![7]),
            set_order: 0,
            authoring_set_order: 0,
            version: 9,
        }
    }

    /// `prims` prims under one root, each with two fields and a time sample,
    /// and one field naming an external value.
    fn state_of(prims: u64) -> ObjectState {
        let mut state = ObjectState {
            sequence_num: 9,
            ..ObjectState::default()
        };
        state.sections.insert(1, section(1, 0));
        let children: Vec<u64> = (2..2 + prims).collect();
        for &id in &children {
            state.sections.insert(id, section(id, 1));
            for key in ["a", "b"] {
                state
                    .fields
                    .insert((id, key.to_owned()), inline_field(id, key));
            }
            state.time_samples.insert(
                (id, OrderedFloat(1.0)),
                TimeSample {
                    section_id: id,
                    time: OrderedFloat(1.0),
                    value: FieldValue::Inline(vec![3]),
                    version: 9,
                },
            );
        }
        state.children.insert(1, children);
        state.fields.insert(
            (1, "ext".to_owned()),
            Field {
                value: FieldValue::External {
                    hash: [0xE1; 32],
                    size: 64,
                    ext_value_index: 1,
                },
                ..inline_field(1, "ext")
            },
        );
        state
    }

    fn header() -> RestoreHeader {
        RestoreHeader {
            session_id: "s".to_owned(),
            layer_name: "l.usda".to_owned(),
            created_by: "user".to_owned(),
            server_ts: 1,
            cursor: CheckpointCursorStamp {
                snap_seq: 6,
                written_at_ms: 1_000,
                restore_seq_bump: 1_000,
            },
        }
    }

    fn origin() -> FredErrOrigin<'static> {
        FredErrOrigin::metrics(None, POOL_LABEL_TXN, StoreOomSignal::unwired())
    }

    /// A transaction over a client that never dials. Opened through the
    /// trait path rather than the method call: `b_289_tests` inventories
    /// every method-call `MULTI` under `src/storage/` as a production
    /// transaction site, and a never-sent buffer is not one.
    fn never_dialled_transaction() -> fred::clients::Transaction {
        TransactionInterface::multi(&fred::clients::Client::default())
    }

    /// Queue `body` through `queue_restore_txn`, the queueing
    /// `fenced_restore_txn` runs; the commands it queued.
    async fn queued(body: TxnBody<'_, '_>) -> usize {
        let trx = never_dialled_transaction();
        queue_restore_txn(origin(), &trx, &schema(), &body)
            .await
            .expect("queueing into a never-dialled transaction cannot fail");
        trx.len()
    }

    /// Queue row batch `batch` of `write`; the commands it queued.
    async fn queued_row_batch(write: &RestoreObjectWrite<'_>, batch: usize) -> usize {
        queued(TxnBody::Rows(write, batch)).await
    }

    /// Queue `write`'s header transaction; the commands it queued.
    async fn queued_header(write: &RestoreObjectWrite<'_>) -> usize {
        queued(TxnBody::Header(write)).await
    }

    /// The slicer keeps units that are not children lists whole, fills
    /// greedily and puts an oversized one alone.
    #[test]
    fn slice_units_fills_greedily_and_keeps_an_oversized_unit_alone() {
        let specs = slice_units([3, 3, 3, 12, 2, 2].into_iter(), 8);
        let shape: Vec<(usize, usize, u64)> =
            specs.iter().map(|s| (s.start, s.len, s.commands)).collect();
        assert_eq!(shape, vec![(0, 2, 6), (2, 1, 3), (3, 1, 12), (4, 2, 4)]);
    }

    /// An object over the batch limit: every row batch queues exactly the
    /// commands it counts and at most the limit; the batches, in order, are
    /// every unit once, none split; the header transaction queues the two
    /// commands `header_commands()` counts (its timeout's size), the stream
    /// `UNLINK` and the header `HSET`; and the batches and the header
    /// together are `object_state_command_count` (header `HSET` plus every
    /// unit) plus the `UNLINK`, which is the footprint's count. Every count
    /// is read off `queue_restore_txn`, the queueing the real transaction
    /// runs.
    #[tokio::test]
    async fn restore_batches_queue_what_they_count_and_the_header_goes_alone_last() {
        let state = state_of(6);
        let limit = 10;
        let write = RestoreObjectWrite::plan(OID, &state, header(), limit);
        let unit_count = state_units(&state, &write.ext_plan).count();
        assert!(write.row_batches() >= 2, "fixture: several row batches");

        let mut next_unit = 0;
        let mut row_total = 0u64;
        for batch in 0..write.row_batches() {
            let spec = write.batches[batch];
            assert_eq!(
                spec.start, next_unit,
                "batch {batch} starts where the last ended"
            );
            next_unit += spec.len;
            let counted = write.row_batch_commands(batch);
            assert_eq!(
                queued_row_batch(&write, batch).await as u64,
                counted,
                "batch {batch} queues what it counts"
            );
            assert!(counted <= limit, "batch {batch}: {counted} commands");
            row_total += counted;
        }
        assert_eq!(next_unit, unit_count, "every unit is in exactly one batch");
        assert_eq!(write.header_commands(), 2);
        assert_eq!(queued_header(&write).await, 2, "the header goes alone");

        let ext_plan =
            plan_materialization_ext_index(OID, &state, &HashMap::new(), SurplusPolicy::Suppress);
        assert_eq!(
            row_total + write.header_commands(),
            object_state_command_count(&state, &ext_plan) + 1,
            "the units, the header HSET and the stream UNLINK"
        );
        assert_eq!(
            row_total + write.header_commands(),
            restore_object_command_count(OID, &state)
        );
    }

    /// With no children list over the limit (the fixture's is 8 commands),
    /// no unit is split across batches: each batch's units' commands sum to
    /// its count, and a batch ends only where the next unit would overflow
    /// it.
    #[test]
    fn restore_batches_end_only_where_the_next_unit_would_overflow() {
        let state = state_of(6);
        let limit = 10;
        let write = RestoreObjectWrite::plan(OID, &state, header(), limit);
        let commands: Vec<u64> = state_units(&state, &write.ext_plan)
            .map(|u| u.commands())
            .collect();
        for (i, spec) in write.batches.iter().enumerate() {
            let units = &commands[spec.start..spec.start + spec.len];
            assert_eq!(units.iter().sum::<u64>(), spec.commands, "batch {i}");
            if let Some(next) = commands.get(spec.start + spec.len) {
                assert!(
                    spec.commands + next > limit,
                    "batch {i} closed while unit {} still fitted",
                    spec.start + spec.len
                );
            }
        }
    }

    /// The children slices `write`'s row batches carry, in batch order: each
    /// as `(batch, children pushed)`. A whole list (one batch carries all of
    /// it) is not a slice.
    fn children_slices(write: &RestoreObjectWrite<'_>) -> Vec<(usize, std::ops::Range<u64>)> {
        (0..write.row_batches())
            .flat_map(|batch| {
                write
                    .batch_units(batch)
                    .filter_map(move |(_, slice)| slice.map(|slice| (batch, slice)))
            })
            .collect()
    }

    /// A children list longer than the command limit (here 30 children, 32
    /// commands, at a limit of 10) is cut, not put in a batch of its own:
    /// every row batch is within the limit and queues exactly what it
    /// counts; the list's slices are consecutive, in order, from child 0 to
    /// its last, each in its own batch; the first slice fills what the batch
    /// before it left (that batch carries a section too), and the batch
    /// holding the last slice goes on with the units after it. The batches
    /// together queue every unit's commands plus one `LTRIM` per later slice.
    #[tokio::test]
    async fn restore_batches_cut_a_children_list_over_the_command_limit_into_slices_within_it() {
        let state = state_of(30);
        let limit = 10;
        let write = RestoreObjectWrite::plan(OID, &state, header(), limit);

        let slices = children_slices(&write);
        assert!(slices.len() >= 3, "fixture: the list is cut: {slices:?}");
        let mut next_child = 0;
        for (i, (batch, slice)) in slices.iter().enumerate() {
            assert_eq!(slice.start, next_child, "slice {i} continues the last");
            assert!(slice.end > slice.start, "slice {i} pushes a child");
            if i > 0 {
                assert_eq!(*batch, slices[i - 1].0 + 1, "slice {i} is the next batch's");
            }
            next_child = slice.end;
        }
        assert_eq!(next_child, 30, "the slices push every child, once");
        let (first_batch, _) = slices[0];
        assert!(
            write.batches[first_batch].len > 1,
            "the first slice joins the batch with room left"
        );
        let (last_batch, _) = slices[slices.len() - 1];
        assert!(
            write.batches[last_batch].len > 1,
            "the batch holding the last slice goes on with the units after it"
        );

        let mut row_total = 0u64;
        for batch in 0..write.row_batches() {
            let counted = write.row_batch_commands(batch);
            assert!(counted <= limit, "batch {batch}: {counted} commands");
            assert_eq!(
                queued_row_batch(&write, batch).await as u64,
                counted,
                "batch {batch} queues what it counts"
            );
            row_total += counted;
        }
        let units: u64 = state_units(&state, &write.ext_plan)
            .map(|u| u.commands())
            .sum();
        assert_eq!(
            row_total,
            units + (slices.len() as u64 - 1),
            "every unit's commands and one LTRIM per later slice"
        );
    }

    /// A root section with `children` child sections and nothing else: a
    /// children list as long as the test needs, cheap to build.
    fn wide_state(children: u64) -> ObjectState {
        let mut state = ObjectState {
            sequence_num: 9,
            ..ObjectState::default()
        };
        state.sections.insert(1, section(1, 0));
        let ids: Vec<u64> = (2..2 + children).collect();
        for &id in &ids {
            state.sections.insert(id, section(id, 1));
        }
        state.children.insert(1, ids);
        state
    }

    /// The footprint's command count is what the writer queues at the
    /// production limits, the `LTRIM`s of a cut list included: for a list of
    /// 25,000 children, longer than `RESTORE_BATCH_COMMANDS`,
    /// `restore_object_command_count` is every row batch's queued commands
    /// plus the header transaction's, which is every unit plus one `LTRIM`
    /// per later slice and the two header commands.
    ///
    /// Fails with the count not counting the cut (in-place perturbation of
    /// `restore_object_command_count`, its `cut` forced false): the count is
    /// the units and the header alone, the `LTRIM`s short.
    #[tokio::test]
    async fn restore_object_command_count_counts_the_ltrims_of_a_cut_list() {
        let state = wide_state(25_000);
        let write = RestoreObjectWrite::plan(OID, &state, header(), RESTORE_BATCH_COMMANDS);
        let later_slices = children_slices(&write).len() as u64 - 1;
        assert!(
            later_slices >= 2,
            "fixture: the list is cut into three slices or more"
        );

        let mut queued = 0u64;
        for batch in 0..write.row_batches() {
            let counted = write.row_batch_commands(batch);
            assert!(
                counted <= RESTORE_BATCH_COMMANDS,
                "batch {batch}: {counted}"
            );
            assert_eq!(
                queued_row_batch(&write, batch).await as u64,
                counted,
                "batch {batch}"
            );
            queued += counted;
        }
        queued += queued_header(&write).await as u64;

        let units: u64 = state_units(&state, &write.ext_plan)
            .map(|u| u.commands())
            .sum();
        assert_eq!(restore_object_command_count(OID, &state), queued);
        assert_eq!(
            queued,
            units + later_slices + 2,
            "the units, the LTRIMs and the header"
        );
    }

    /// The slicer's cut of a children list, unit by unit: a list over the
    /// command limit starts in the batch before it when that batch has room
    /// for the list's own two commands (`DEL`, `SADD`) and one child, filling
    /// what is left; else it starts a batch. Each later slice carries one
    /// `LTRIM` and a whole batch's worth of children, or the rest; the batch
    /// holding the last slice goes on with the units after it. A list within
    /// the limit is whole.
    #[test]
    fn slice_state_units_cuts_a_children_list_over_the_command_limit_at_the_limits() {
        let limit = BatchLimit {
            commands: 8,
            arg_bytes: 10,
        };
        let unit = |commands: u64, arg_bytes: u64| SlicedUnit {
            cost: cost(commands, arg_bytes),
            children: 0,
        };
        let list = |children: u64| SlicedUnit {
            cost: cost(children + 2, 0),
            children,
        };
        /// A batch's `(start, len, commands, first_child, last_child_end)`.
        type Shape = (usize, usize, u64, u64, Option<u64>);
        let cases: [(&str, Vec<SlicedUnit>, Vec<Shape>); 3] = [
            (
                "the first slice fills the batch before it; the last goes on",
                vec![unit(3, 0), list(20), unit(3, 0)],
                vec![
                    (0, 2, 8, 0, Some(3)),
                    (1, 1, 8, 3, Some(10)),
                    (1, 1, 8, 10, Some(17)),
                    (1, 2, 7, 17, None),
                ],
            ),
            (
                "a batch with no room for the list's own commands and a child",
                vec![unit(6, 0), list(10)],
                vec![
                    (0, 1, 6, 0, None),
                    (1, 1, 8, 0, Some(6)),
                    (1, 1, 5, 6, None),
                ],
            ),
            (
                "a list at the limit is whole, a unit like any other",
                vec![unit(2, 0), list(6), unit(1, 0)],
                vec![(0, 1, 2, 0, None), (1, 1, 8, 0, None), (2, 1, 1, 0, None)],
            ),
        ];
        for (case, units, expected) in cases {
            let specs = slice_state_units(units.into_iter(), limit);
            let shape: Vec<Shape> = specs
                .iter()
                .map(|s| (s.start, s.len, s.commands, s.first_child, s.last_child_end))
                .collect();
            assert_eq!(shape, expected, "{case}");
            for spec in &specs {
                assert!(spec.commands <= limit.commands, "{case}: {spec:?}");
            }
        }
    }

    /// A slice never joins a batch that a value over the byte limit already
    /// holds by itself, though the batch has commands to spare: the value
    /// stays alone and the list starts the next batch. (`state_units` puts
    /// every value after every children list, so only the slicer's own
    /// contract, for any order, reaches this.)
    ///
    /// Fails without the guard (in-place perturbation of `slice_state_units`,
    /// its byte condition on a slice dropped): the list's first slice joins
    /// the value's batch.
    #[test]
    fn slice_state_units_never_puts_a_slice_beside_a_value_over_the_byte_limit() {
        let limit = BatchLimit {
            commands: 8,
            arg_bytes: 10,
        };
        let units = [
            SlicedUnit {
                cost: cost(1, 11),
                children: 0,
            },
            SlicedUnit {
                cost: cost(22, 0),
                children: 20,
            },
        ];
        let specs = slice_state_units(units.into_iter(), limit);
        let shape: Vec<(usize, usize, u64, u64, Option<u64>)> = specs
            .iter()
            .map(|s| (s.start, s.len, s.commands, s.first_child, s.last_child_end))
            .collect();
        assert_eq!(
            shape,
            vec![
                (0, 1, 1, 0, None),
                (1, 1, 8, 0, Some(6)),
                (1, 1, 8, 6, Some(13)),
                (1, 1, 8, 13, None),
            ]
        );
    }

    /// Production `plan` cuts on the byte limit, `RESTORE_BATCH_ARG_BYTES`:
    /// an object of six 1 MiB inline values, about 8 MiB of arguments in a
    /// few dozen commands, far under `RESTORE_BATCH_COMMANDS`, goes in
    /// several row batches, each within the byte limit (none holds a value
    /// over it alone), the header transaction last.
    ///
    /// Fails with `plan` passing no byte limit (in-place perturbation,
    /// `u64::MAX` for `RESTORE_BATCH_ARG_BYTES`): the object is one
    /// transaction.
    #[test]
    fn restore_plan_cuts_on_the_production_byte_limit_when_the_commands_fit() {
        let mut state = state_of(6);
        for id in 2..8u64 {
            state.fields.insert(
                (id, "big".to_owned()),
                Field {
                    value: FieldValue::Inline(vec![0x5A; 1 << 20]),
                    ..inline_field(id, "big")
                },
            );
        }
        let units = units_arg_bytes(&state);
        assert!(
            units > RESTORE_BATCH_ARG_BYTES,
            "fixture: {units} B of arguments, over the byte limit"
        );
        assert!(
            restore_object_command_count(OID, &state) < RESTORE_BATCH_COMMANDS,
            "fixture: the commands fit one batch"
        );

        let write = RestoreObjectWrite::plan(OID, &state, header(), RESTORE_BATCH_COMMANDS);

        assert!(
            write.row_batches() >= 2,
            "the bytes cut several row batches: {}",
            write.row_batches()
        );
        for batch in 0..write.row_batches() {
            assert!(
                write.row_batch_arg_bytes(batch) <= RESTORE_BATCH_ARG_BYTES,
                "batch {batch}: {} B",
                write.row_batch_arg_bytes(batch)
            );
        }
        assert_eq!(write.header_commands(), 2, "the header goes alone");
    }

    /// An object whose units, the stream `UNLINK` and the header fit the
    /// limit is one transaction: no row batch, and the header transaction,
    /// queued by `queue_restore_txn`, carries every unit, the `UNLINK` and
    /// the header `HSET` — exactly the count its timeout is sized by — at the
    /// limit itself; one command under it, the object splits and the header
    /// transaction is the two commands alone. (A transaction's length is all
    /// a never-dialled one shows, not its order; the header landing with
    /// the rows is `fence_tests`' and the integration tests'.)
    #[tokio::test]
    async fn restore_of_a_small_object_is_one_transaction_with_the_header_last() {
        let state = state_of(2);
        let total = restore_object_command_count(OID, &state);
        let write = RestoreObjectWrite::plan(OID, &state, header(), total);
        assert_eq!(
            write.row_batches(),
            0,
            "one transaction at exactly the limit"
        );
        assert_eq!(write.header_commands(), total);
        assert_eq!(queued_header(&write).await as u64, total);

        let split = RestoreObjectWrite::plan(OID, &state, header(), total - 1);
        assert!(split.row_batches() >= 1, "one command under it splits");
        assert_eq!(split.header_commands(), 2);
    }

    // ── The byte limit: `RESTORE_BATCH_ARG_BYTES`, `StateUnit::arg_bytes` ──

    /// A unit cost of `commands` commands and `arg_bytes` argument bytes.
    fn cost(commands: u64, arg_bytes: u64) -> UnitCost {
        UnitCost {
            commands,
            arg_bytes,
        }
    }

    /// The slicer ends a batch where the next unit would take it over either
    /// limit, and only there: on bytes while the commands still fit, on
    /// commands while the bytes still fit, and not at all when a batch meets
    /// both limits exactly. A unit over either limit is a batch of its own.
    /// In every case each unit is in exactly one batch, in order, none split.
    ///
    /// Fails without the byte limit (in-place perturbation of
    /// `slice_units_within`, its byte condition dropped): the case where
    /// bytes bind comes out as one batch.
    #[test]
    fn slice_units_within_ends_a_batch_where_either_limit_would_be_passed() {
        let limit = BatchLimit {
            commands: 8,
            arg_bytes: 10,
        };
        /// A batch's `(start, len, commands)`.
        type Shape = (usize, usize, u64);
        let cases: [(&str, Vec<UnitCost>, Vec<Shape>); 5] = [
            (
                "bytes bind while the commands fit",
                vec![cost(1, 4), cost(1, 4), cost(1, 4)],
                vec![(0, 2, 2), (2, 1, 1)],
            ),
            (
                "commands bind while the bytes fit",
                vec![cost(3, 0), cost(3, 1), cost(3, 1)],
                vec![(0, 2, 6), (2, 1, 3)],
            ),
            (
                "both limits met exactly",
                vec![cost(4, 5), cost(4, 5), cost(1, 0)],
                vec![(0, 2, 8), (2, 1, 1)],
            ),
            (
                "a unit over the byte limit is alone",
                vec![cost(1, 1), cost(1, 11), cost(1, 1)],
                vec![(0, 1, 1), (1, 1, 1), (2, 1, 1)],
            ),
            (
                "a unit over the command limit is alone",
                vec![cost(1, 1), cost(9, 0), cost(1, 1)],
                vec![(0, 1, 1), (1, 1, 9), (2, 1, 1)],
            ),
        ];
        for (case, units, expected) in cases {
            let specs = slice_units_within(units.iter().copied(), limit);
            let shape: Vec<Shape> = specs.iter().map(|s| (s.start, s.len, s.commands)).collect();
            assert_eq!(shape, expected, "{case}");
            let mut next = 0;
            for spec in &specs {
                assert_eq!(spec.start, next, "{case}: batches are contiguous, in order");
                next += spec.len;
            }
            assert_eq!(next, units.len(), "{case}: every unit in exactly one batch");
        }
    }

    /// `state`'s units' argument bytes, summed, as the planner counts them.
    fn units_arg_bytes(state: &ObjectState) -> u64 {
        let ext_plan =
            plan_materialization_ext_index(OID, state, &HashMap::new(), SurplusPolicy::Suppress);
        state_units(state, &ext_plan).map(|u| u.arg_bytes()).sum()
    }

    /// The object is one transaction only when its units and the header
    /// transaction fit both limits, the header `HSET`'s content-length
    /// arguments (session id, layer name, `created_by`, source format)
    /// counted against the byte limit: at exactly the units' and the
    /// header's bytes it is one transaction, and one byte under, with the
    /// command limit far away, it splits though its units alone would fit.
    ///
    /// Fails without the header's bytes in the decision (in-place
    /// perturbation of `plan_within`): one byte under is still one
    /// transaction.
    #[test]
    fn plan_within_is_one_transaction_only_when_the_units_and_the_header_fit_the_byte_limit() {
        let state = state_of(2);
        let units = units_arg_bytes(&state);
        let header_bytes = header_arg_bytes(&state, &header());
        assert!(
            units > 0 && header_bytes > 0,
            "fixture: the units and the header both carry argument bytes"
        );
        let total = restore_object_command_count(OID, &state);
        let within = |arg_bytes: u64| {
            RestoreObjectWrite::plan_within(
                OID,
                &state,
                header(),
                BatchLimit {
                    commands: 10 * total,
                    arg_bytes,
                },
            )
        };

        let single = within(units + header_bytes);
        assert_eq!(
            single.row_batches(),
            0,
            "one transaction at exactly the bytes"
        );
        assert_eq!(single.header_commands(), total);

        let split = within(units + header_bytes - 1);
        assert!(
            split.row_batches() >= 1,
            "one byte under it splits, though the units alone fit"
        );
        assert_eq!(split.header_commands(), 2);
    }

    /// An object over the byte limit and far under the command limit is cut
    /// on bytes: every row batch queues the commands it counts and holds at
    /// most the byte limit's argument bytes unless it is one unit alone (the
    /// external-value field here, over the limit by itself), each ends only
    /// where its next unit would pass a limit, and the batches are every unit
    /// once, in order; the header transaction is the two commands alone.
    #[tokio::test]
    async fn plan_within_cuts_row_batches_on_bytes_when_the_commands_fit() {
        let state = state_of(6);
        let limit = BatchLimit {
            commands: 1_000,
            arg_bytes: 40,
        };
        let write = RestoreObjectWrite::plan_within(OID, &state, header(), limit);
        let costs: Vec<UnitCost> = state_units(&state, &write.ext_plan)
            .map(UnitCost::of)
            .collect();
        assert!(
            costs.iter().map(|c| c.commands).sum::<u64>() + 2 <= limit.commands,
            "fixture: the commands alone fit one transaction"
        );
        assert!(
            costs.iter().any(|c| c.arg_bytes > limit.arg_bytes),
            "fixture: one unit over the byte limit by itself"
        );
        assert!(
            write.row_batches() >= 3,
            "fixture: the bytes cut several batches"
        );

        let mut next = 0;
        for batch in 0..write.row_batches() {
            let spec = write.batches[batch];
            assert_eq!(
                spec.start, next,
                "batch {batch} starts where the last ended"
            );
            let units = &costs[spec.start..spec.start + spec.len];
            let bytes: u64 = units.iter().map(|c| c.arg_bytes).sum();
            assert!(
                spec.len == 1 || bytes <= limit.arg_bytes,
                "batch {batch}: {bytes} B in {} units",
                spec.len
            );
            if let Some(following) = costs.get(spec.start + spec.len) {
                assert!(
                    bytes + following.arg_bytes > limit.arg_bytes
                        || spec.commands + following.commands > limit.commands,
                    "batch {batch} closed while its next unit still fitted"
                );
            }
            assert_eq!(
                queued_row_batch(&write, batch).await as u64,
                write.row_batch_commands(batch),
                "batch {batch} queues what it counts"
            );
            next += spec.len;
        }
        assert_eq!(next, costs.len(), "every unit is in exactly one batch");
        assert_eq!(write.header_commands(), 2);
        assert_eq!(queued_header(&write).await, 2, "the header goes alone");
    }

    /// `StateUnit::arg_bytes` per unit kind: a section its name; a field six
    /// times its name (hex-encoded in its row key, which its `HDEL` and its
    /// `HSET` both carry, and in its index member) plus its value as queued,
    /// an inline value's base64 length (4 per 3 bytes, rounded up) and an
    /// external one 84 (its hash's 64 hex digits and at most 20 of size); a
    /// time sample its value alone; a children list and an index write
    /// nothing.
    #[test]
    fn state_unit_arg_bytes_counts_names_and_values_by_unit_kind() {
        let named = Section {
            name: "Xform_12".to_owned(),
            ..section(5, 1)
        };
        let field = |value: FieldValue| Field {
            value,
            ..inline_field(5, "points")
        };
        let sample = |value: FieldValue| TimeSample {
            section_id: 5,
            time: OrderedFloat(2.0),
            value,
            version: 9,
        };
        let external = |size: u64| FieldValue::External {
            hash: [0xE2; 32],
            size,
            ext_value_index: 1,
        };
        let (inline_field_7, external_field) = (
            field(FieldValue::Inline(vec![0; 7])),
            field(external(u64::MAX)),
        );
        let (inline_sample_3, external_sample) =
            (sample(FieldValue::Inline(vec![0; 3])), sample(external(1)));
        let children = [2u64, 3, 4];
        let ext_set = ("ab".repeat(32), "3".to_owned());
        let ext_del = "cd".repeat(32);
        let cases = [
            ("section", StateUnit::Section(&named), 8),
            (
                "inline field",
                StateUnit::Field(&inline_field_7),
                6 * 6 + 12,
            ),
            (
                "external field",
                StateUnit::Field(&external_field),
                6 * 6 + 84,
            ),
            (
                "inline time sample",
                StateUnit::TimeSample(&inline_sample_3),
                4,
            ),
            (
                "external time sample",
                StateUnit::TimeSample(&external_sample),
                84,
            ),
            (
                "children list",
                StateUnit::Children {
                    parent: 1,
                    ids: &children,
                },
                0,
            ),
            ("index HSET", StateUnit::ExtSet(&ext_set), 0),
            ("index HDEL", StateUnit::ExtDel(&ext_del), 0),
        ];
        for (kind, unit, bytes) in cases {
            assert_eq!(unit.arg_bytes(), bytes, "{kind}");
        }
    }

    /// The commands `queue_state_unit` queues for `unit` of object [`OID`],
    /// into a never-dialled transaction.
    async fn queued_unit(unit: StateUnit<'_>) -> usize {
        let trx = never_dialled_transaction();
        let oid = OID.to_string();
        queue_state_unit(
            origin(),
            &trx,
            &schema(),
            &oid,
            &schema().obj_ext_index(&oid),
            unit,
        )
        .await
        .expect("queueing into a never-dialled transaction cannot fail");
        trx.len()
    }

    /// The commands and the argument bytes `queue_state_unit` queues for a
    /// section, field or time-sample `unit` of object [`OID`], its arguments
    /// rebuilt from the builders it calls: the key schema's row and index
    /// keys and members, `section_hash_fields`, `field_value_hash_fields`,
    /// `time_sample_hash_fields` and `VALUE_HFIELDS`. The command names and a
    /// sorted-set score's spelling are left out: the same whatever the
    /// unit's content.
    fn rebuilt_args(unit: StateUnit<'_>) -> (usize, u64) {
        use super::super::{
            field_value_hash_fields, section_hash_fields, time_sample_hash_fields, VALUE_HFIELDS,
        };
        let keys = schema();
        let oid = OID.to_string();
        let pairs = |pairs: &[(String, String)]| -> u64 {
            pairs.iter().map(|(k, v)| (k.len() + v.len()) as u64).sum()
        };
        let hdel_fields: u64 = VALUE_HFIELDS.iter().map(|f| f.len() as u64).sum();
        match unit {
            StateUnit::Section(section) => {
                let sid = section.section_id.to_string();
                let hset: u64 = keys.obj_section(&oid, &sid).len() as u64
                    + section_hash_fields(section)
                        .iter()
                        .map(|(k, v)| (k.len() + v.len()) as u64)
                        .sum::<u64>();
                let sadd = (keys.obj_sections_index(&oid).len() + sid.len()) as u64;
                (2, hset + sadd)
            }
            StateUnit::Field(field) => {
                let sid = field.section_id.to_string();
                let row = keys.obj_field(&oid, &sid, &field.key_name).len() as u64;
                let hset = pairs(&field_value_hash_fields(
                    &field.value,
                    field.set_order,
                    field.authoring_set_order,
                    field.version,
                ));
                let zadd = (keys.obj_fields_index(&oid).len()
                    + KeySchema::field_member(&sid, &field.key_name).len())
                    as u64;
                (3, (row + hdel_fields) + (row + hset) + zadd)
            }
            StateUnit::TimeSample(sample) => {
                let sid = sample.section_id.to_string();
                let time = sample.time.into_inner();
                let row = keys.obj_time_sample(&oid, &sid, time).len() as u64;
                let hset = pairs(&time_sample_hash_fields(&sample.value, sample.version));
                let zadd = (keys.obj_time_samples_index(&oid).len()
                    + KeySchema::time_sample_member(&sid, time).len())
                    as u64;
                (3, (row + hdel_fields) + (row + hset) + zadd)
            }
            other => {
                unreachable!("rebuilt only for the kinds whose arguments follow content: {other:?}")
            }
        }
    }

    /// `StateUnit::arg_bytes` counts from above: as a section's name, a
    /// field's name, or a field's or a time sample's inline value grows, the
    /// arguments `queue_state_unit` queues for the unit grow by no more than
    /// its count does, and an external value's hash and size arguments are
    /// at most its 84 at the largest size. What else a unit queues stays the
    /// same as its content grows: the per-command constant the batch's
    /// command limit prices. The rebuilt arguments' command count is checked
    /// against what `queue_state_unit` really queues, so a unit that came to
    /// queue another command fails here before its bytes are compared.
    ///
    /// Fails with a field's name counted four times rather than six
    /// (in-place perturbation of `arg_bytes`): its rebuilt arguments grow 6
    /// bytes per name byte, its count 4.
    #[tokio::test]
    async fn state_unit_arg_bytes_grows_at_least_as_fast_as_the_arguments_it_queues() {
        const GROWTH: usize = 3_000;
        let named = |name: String| Section {
            name,
            ..section(5, 1)
        };
        let keyed = |key: &str, len: usize| Field {
            value: FieldValue::Inline(vec![0x5A; len]),
            ..inline_field(5, key)
        };
        let sampled = |len: usize| TimeSample {
            section_id: 5,
            time: OrderedFloat(2.5),
            value: FieldValue::Inline(vec![0x5A; len]),
            version: 9,
        };
        let (short_section, long_section) = (named("n".to_owned()), named("n".repeat(GROWTH)));
        let long_name = "k".repeat(GROWTH);
        let (short_key, long_key) = (keyed("k", 1), keyed(&long_name, 1));
        let (small_value, large_value) = (keyed("k", 1), keyed("k", 1 + GROWTH));
        let (small_sample, large_sample) = (sampled(1), sampled(1 + GROWTH));
        let pairs = [
            (
                "a section's name",
                StateUnit::Section(&short_section),
                StateUnit::Section(&long_section),
            ),
            (
                "a field's name",
                StateUnit::Field(&short_key),
                StateUnit::Field(&long_key),
            ),
            (
                "a field's inline value",
                StateUnit::Field(&small_value),
                StateUnit::Field(&large_value),
            ),
            (
                "a time sample's inline value",
                StateUnit::TimeSample(&small_sample),
                StateUnit::TimeSample(&large_sample),
            ),
        ];
        for (grown, small, large) in pairs {
            for unit in [small, large] {
                let (commands, _) = rebuilt_args(unit);
                assert_eq!(
                    (commands, queued_unit(unit).await),
                    (unit.commands() as usize, unit.commands() as usize),
                    "{grown}: the rebuilt arguments are the commands the unit queues"
                );
            }
            let real = rebuilt_args(large).1 - rebuilt_args(small).1;
            let counted = large.arg_bytes() - small.arg_bytes();
            assert!(real > 0, "fixture: {grown} grows the arguments queued");
            assert!(
                real <= counted,
                "{grown}: the arguments queued grew {real} B, arg_bytes only {counted} B"
            );
        }

        let mut value_args = Vec::new();
        super::super::encode_value_fields(
            &FieldValue::External {
                hash: [0xFF; 32],
                size: u64::MAX,
                ext_value_index: 1,
            },
            &mut value_args,
        );
        let external: u64 = value_args.iter().map(|(_, v)| v.len() as u64).sum();
        assert!(
            external <= 84,
            "an external value's hash and size arguments are {external} B, above the 84 counted"
        );
    }

    /// A value removal's batch, queued through `queue_value_removal_batch`
    /// (the code the fenced transaction queues through) into a
    /// never-dialled transaction, queues exactly the commands it counts: two
    /// per row (its `DEL` and its member's `ZREM`), a row pushed twice once,
    /// and one per `HDEL`.
    #[tokio::test]
    async fn a_value_removal_batch_queues_what_it_counts() {
        let row = |time_sample: bool, member: String| RemovedRow {
            time_sample,
            member,
        };
        let mut batch = ValueRemovalBatch::new(OID, 9);
        batch.push_row(row(false, KeySchema::field_member("2", "default")));
        batch.push_row(row(true, KeySchema::time_sample_member("2", -0.0)));
        batch.push_row(row(true, KeySchema::time_sample_member("2", 0.0)));
        batch.push_row(row(true, KeySchema::time_sample_member("2", 0.0)));
        batch.push_hdel([0xA1; 32]);
        assert_eq!(batch.rows(), 3, "the repeated row joins once");
        assert_eq!(batch.commands(), 2 * 3 + 1);

        let trx = never_dialled_transaction();
        queue_value_removal_batch(origin(), &trx, &schema(), &batch)
            .await
            .expect("queueing into a never-dialled transaction cannot fail");
        assert_eq!(trx.len() as u64, batch.commands());
    }

    /// A value-removal batch fills as D5 fills it — a row, or an `HDEL`,
    /// joins only while [`ValueRemovalBatch::fits`] — until it is full. The
    /// rows each `name_len` bytes of field name, then `hdels` `HDEL`s.
    fn fill_value_removal_batch(
        name_len: usize,
        rows: usize,
        hdels: usize,
        limit: BatchLimit,
    ) -> ValueRemovalBatch {
        let mut batch = ValueRemovalBatch::new(OID, 9);
        for r in 0..rows {
            let row = RemovedRow {
                time_sample: false,
                member: KeySchema::field_member(
                    "2",
                    &format!("{r:04}{}", "k".repeat(name_len - 4)),
                ),
            };
            if !batch.fits(2, row.arg_bytes(), limit) {
                return batch;
            }
            batch.push_row(row);
        }
        for h in 0..hdels {
            if !batch.fits(1, VALUE_REMOVAL_HDEL_ARG_BYTES, limit) {
                return batch;
            }
            batch.push_hdel([u8::try_from(h % 256).expect("fits"); 32]);
        }
        batch
    }

    /// A value-removal batch cuts on argument bytes while its commands fit
    /// (fail-on-old). At the production limits, rows of 4,096-byte field
    /// names (each counting its hex-encoded member twice, about 16 KiB) fill
    /// a batch past `RESTORE_BATCH_ARG_BYTES` at a few hundred rows, far
    /// under `RESTORE_BATCH_COMMANDS`; and the `HDEL obj:ext` batches that
    /// follow a removal's rows cut on bytes too, under a byte limit that
    /// binds before the command limit (at production the command limit
    /// binds first: 10,000 `HDEL`s are 640,000 B). A batch never passes
    /// either limit, and an empty batch takes a row over a limit, alone.
    ///
    /// Fails without the byte limit (in-place perturbation of `fits`, its
    /// byte condition dropped): the batches fill to the command limit.
    #[test]
    fn a_value_removal_batch_cuts_on_argument_bytes_while_its_commands_fit() {
        let production = BatchLimit {
            commands: RESTORE_BATCH_COMMANDS,
            arg_bytes: RESTORE_BATCH_ARG_BYTES,
        };
        let rows = fill_value_removal_batch(4_096, 1_000, 0, production);
        assert!(
            rows.rows() > 1 && rows.commands() < RESTORE_BATCH_COMMANDS,
            "the bytes cut first: {} rows, {} commands",
            rows.rows(),
            rows.commands()
        );
        assert!(
            rows.arg_bytes <= RESTORE_BATCH_ARG_BYTES,
            "{} B",
            rows.arg_bytes
        );
        assert!(
            rows.arg_bytes + 2 * (4_096 * 2 + 2) as u64 > RESTORE_BATCH_ARG_BYTES,
            "full: one more row would pass the byte limit ({} B)",
            rows.arg_bytes
        );

        let bytes_bind = BatchLimit {
            commands: 1_000,
            arg_bytes: 10 * VALUE_REMOVAL_HDEL_ARG_BYTES,
        };
        let hdels = fill_value_removal_batch(8, 0, 100, bytes_bind);
        assert_eq!(
            (hdels.commands(), hdels.arg_bytes),
            (10, 10 * VALUE_REMOVAL_HDEL_ARG_BYTES),
            "ten HDELs fill the byte limit"
        );

        let one_row_over = BatchLimit {
            commands: 1_000,
            arg_bytes: 16,
        };
        let alone = fill_value_removal_batch(64, 3, 0, one_row_over);
        assert_eq!(
            alone.rows(),
            1,
            "an empty batch takes a row over the limit, alone"
        );
    }

    /// A value-removal batch cuts on commands: two per row and one per
    /// `HDEL`, never past the limit, a row needing room for both of its.
    #[test]
    fn a_value_removal_batch_cuts_on_commands() {
        let limit = BatchLimit {
            commands: 5,
            arg_bytes: u64::MAX,
        };
        assert_eq!(
            fill_value_removal_batch(8, 10, 0, limit).commands(),
            4,
            "two rows: a third needs 2 more"
        );
        let mixed = {
            let mut batch = fill_value_removal_batch(8, 2, 0, limit);
            assert!(
                batch.fits(1, VALUE_REMOVAL_HDEL_ARG_BYTES, limit),
                "one HDEL fits beside two rows"
            );
            batch.push_hdel([0xA1; 32]);
            batch
        };
        assert_eq!(mixed.commands(), 5);
        assert!(
            !mixed.fits(1, VALUE_REMOVAL_HDEL_ARG_BYTES, limit),
            "full at 5"
        );
        assert_eq!(
            fill_value_removal_batch(8, 0, 10, limit).commands(),
            5,
            "five HDELs"
        );
    }

    /// The header transaction of a value removal stays three commands — the
    /// stream `UNLINK`, the header `HSET` and the last hash's `HDEL obj:ext`
    /// ([`VALUE_REMOVAL_HEADER_COMMANDS`]) — however many hashes and rows
    /// the removal had: every other `HDEL` and every row goes in the
    /// batches before it. A removal of no hash carries none.
    #[test]
    fn a_value_removal_header_carries_one_hdel_and_is_three_commands() {
        let header = BandedHeaderWrite::new(OID, 9, 1_009, 1_000);
        assert_eq!(
            header
                .clone()
                .removing_ext_hash(Some([0xB2; 32]))
                .commands(BandedFence::NotRetiring),
            VALUE_REMOVAL_HEADER_COMMANDS
        );
        assert_eq!(VALUE_REMOVAL_HEADER_COMMANDS, 3);
        assert_eq!(
            header
                .removing_ext_hash(None)
                .commands(BandedFence::NotRetiring),
            VALUE_REMOVAL_HEADER_COMMANDS - 1
        );
    }

    /// Each planned write carries its own `commit_chain`, the value its header
    /// `HSET` stamps.
    #[test]
    fn restore_writes_carry_a_commit_chain_of_their_own() {
        let state = state_of(1);
        let a = RestoreObjectWrite::plan(OID, &state, header(), 1_000);
        let b = RestoreObjectWrite::plan(OID, &state, header(), 1_000);
        assert_ne!(a.commit_chain(), b.commit_chain());
        let pairs = object_header_pairs(
            &state,
            "s",
            "l.usda",
            "user",
            1,
            None,
            a.commit_chain().to_owned(),
        );
        assert!(pairs
            .iter()
            .any(|(k, v)| *k == HEADER_FIELD_COMMIT_CHAIN && v == a.commit_chain()));
    }

    /// The restore memory envelope's record (plan b.tv3, ruling w6m2), which
    /// derives the restore batch's two limits.
    const RESTORE_ENVELOPE_DERIVATION_JSON: &str =
        "docs-internal/validation/restore-memory-envelope-tv3-20261022/derivation.json";

    /// The restore batch's limits are the record's: [`RESTORE_BATCH_COMMANDS`]
    /// its `writer_batch_limits.restore_batch_commands` and
    /// [`RESTORE_BATCH_ARG_BYTES`] its `restore_batch_arg_bytes`, the limits
    /// the writer's batch term is measured and priced at. Either edited
    /// without the record, or the reverse, fails here; the published source
    /// tree, which has no `docs-internal/`, skips.
    #[test]
    fn restore_batch_limits_are_the_recorded_envelope_derivation() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        if !root.join("docs-internal").is_dir() {
            eprintln!(
                "skipped: no docs-internal/ (published source tree), so no \
                 {RESTORE_ENVELOPE_DERIVATION_JSON}"
            );
            return;
        }
        let raw = std::fs::read_to_string(root.join(RESTORE_ENVELOPE_DERIVATION_JSON))
            .unwrap_or_else(|e| {
                panic!("the committed record {RESTORE_ENVELOPE_DERIVATION_JSON}: {e}")
            });
        let record: serde_json::Value =
            serde_json::from_str(&raw).expect("the record is valid JSON");
        let recorded = |pointer: &str| {
            record
                .pointer(pointer)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_else(|| panic!("the record carries {pointer} as a non-negative integer"))
        };
        assert_eq!(
            RESTORE_BATCH_COMMANDS,
            recorded("/writer_batch_limits/restore_batch_commands"),
            "the command limit is the record's"
        );
        assert_eq!(
            RESTORE_BATCH_ARG_BYTES,
            recorded("/writer_batch_limits/restore_batch_arg_bytes"),
            "the byte limit is the record's"
        );
    }
}

#[cfg(test)]
mod fence_tests {
    //! The fences of the restore and accept-loss Jobs' transactions against
    //! a real store: the header-absent fence of every batch and of the
    //! header transaction, the `commit_chain` read that settles an in-doubt
    //! header, the clear-flags fence with each of its answers, and the value
    //! removal's fence (its row batch and its header) with each of its
    //! answers. Which outcome the Job makes of each answer is
    //! `dr::job::execute::object`'s tests' and, for the value removal,
    //! `dr::job::execute::discard`'s.

    use std::collections::HashMap;

    use fred::interfaces::StreamsInterface;

    use super::*;
    use crate::domain::types::Section;
    use crate::storage::test_support::TestValkey;

    const OID: u64 = 4_242;

    fn oid() -> String {
        OID.to_string()
    }

    /// A root prim with `prims` children.
    fn state_of(prims: u64) -> ObjectState {
        let mut state = ObjectState {
            sequence_num: 9,
            ..ObjectState::default()
        };
        let section = |section_id: u64, parent_id: u64| Section {
            section_id,
            parent_id,
            name: format!("s{section_id}"),
            section_type: 6,
            section_order: 0,
            first_version: 1,
            last_version: 9,
        };
        state.sections.insert(1, section(1, 0));
        let children: Vec<u64> = (2..2 + prims).collect();
        for &id in &children {
            state.sections.insert(id, section(id, 1));
        }
        state.children.insert(1, children);
        state
    }

    fn header() -> RestoreHeader {
        RestoreHeader {
            session_id: "s".to_owned(),
            layer_name: "l.usda".to_owned(),
            created_by: "user".to_owned(),
            server_ts: 1,
            cursor: CheckpointCursorStamp {
                snap_seq: 6,
                written_at_ms: 1_000,
                restore_seq_bump: 1_000,
            },
        }
    }

    async fn hgetall(tv: &TestValkey) -> HashMap<String, String> {
        tv.client()
            .pool
            .hgetall(tv.key_schema().obj_header(&oid()))
            .await
            .expect("HGETALL")
    }

    async fn hset(tv: &TestValkey, fields: &[(&str, &str)]) {
        let fields: Vec<(String, String)> = fields
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        let _: i64 = tv
            .client()
            .pool
            .hset(tv.key_schema().obj_header(&oid()), fields)
            .await
            .expect("HSET");
    }

    async fn exists(tv: &TestValkey, key: String) -> bool {
        let n: i64 = tv.client().pool.exists(key).await.expect("EXISTS");
        n > 0
    }

    /// A surviving stream on the object, so a test can see whether a
    /// transaction removed it.
    async fn seed_stream(tv: &TestValkey) {
        let _: String = tv
            .client()
            .pool
            .xadd(
                tv.key_schema().layer_stream(OID),
                false,
                None,
                "5-0",
                vec![("payload", "x")],
            )
            .await
            .expect("XADD");
    }

    /// A header present on the object (another writer's): a row batch and
    /// the header transaction each answer `HeaderPresent` and write nothing:
    /// no row, no index, the header as it was, the stream still there.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_present_header_fences_every_restore_transaction() {
        let tv = TestValkey::setup().await;
        hset(&tv, &[("seq", "3"), ("session_id", "other")]).await;
        seed_stream(&tv).await;
        let before = hgetall(&tv).await;
        let state = state_of(6);
        let batched = RestoreObjectWrite::plan(OID, &state, header(), 4);
        assert!(batched.row_batches() >= 2, "fixture: several row batches");
        let single = RestoreObjectWrite::plan(OID, &state, header(), 1_000);
        assert_eq!(single.row_batches(), 0, "fixture: one transaction");

        let answers = [
            write_restore_batch(tv.client(), tv.key_schema(), &batched, 0).await,
            write_restore_header(tv.client(), tv.key_schema(), &batched).await,
            write_restore_header(tv.client(), tv.key_schema(), &single).await,
        ];

        for (i, answer) in answers.iter().enumerate() {
            assert!(
                matches!(answer, Ok(FencedTxn::HeaderPresent)),
                "transaction {i}: {answer:?}"
            );
        }
        assert_eq!(hgetall(&tv).await, before, "the header is as it was");
        assert!(
            !exists(&tv, tv.key_schema().obj_sections_index(&oid())).await,
            "no row batch landed"
        );
        assert!(
            !exists(&tv, tv.key_schema().obj_section(&oid(), "1")).await,
            "no row landed"
        );
        assert!(
            exists(&tv, tv.key_schema().layer_stream(OID)).await,
            "the stream is not removed"
        );
    }

    /// With no header the transactions commit, the header last carrying the
    /// write's own `commit_chain`; the chain read answers `None` before and
    /// that chain after, so an in-doubt header attempt is recognised as the
    /// write's own. The same write's header transaction sent again, and
    /// another write's, find the header present.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_header_lands_last_with_the_writes_commit_chain() {
        let tv = TestValkey::setup().await;
        seed_stream(&tv).await;
        let state = state_of(6);
        let write = RestoreObjectWrite::plan(OID, &state, header(), 4);
        assert_eq!(
            read_header_commit_chain(tv.client(), tv.key_schema(), OID)
                .await
                .expect("HGET"),
            None,
            "no header, no chain"
        );

        for batch in 0..write.row_batches() {
            let answer = write_restore_batch(tv.client(), tv.key_schema(), &write, batch).await;
            assert!(
                matches!(answer, Ok(FencedTxn::Committed)),
                "batch {batch}: {answer:?}"
            );
        }
        assert!(
            hgetall(&tv).await.is_empty(),
            "no row batch writes the header"
        );
        assert!(exists(&tv, tv.key_schema().obj_sections_index(&oid())).await);
        assert!(exists(&tv, tv.key_schema().layer_stream(OID)).await);

        let answer = write_restore_header(tv.client(), tv.key_schema(), &write).await;
        assert!(matches!(answer, Ok(FencedTxn::Committed)), "{answer:?}");
        assert_eq!(
            read_header_commit_chain(tv.client(), tv.key_schema(), OID)
                .await
                .expect("HGET")
                .as_deref(),
            Some(write.commit_chain())
        );
        assert!(
            !exists(&tv, tv.key_schema().layer_stream(OID)).await,
            "the header transaction removes the stream"
        );

        let again = write_restore_header(tv.client(), tv.key_schema(), &write).await;
        assert!(matches!(again, Ok(FencedTxn::HeaderPresent)), "{again:?}");
        let other = RestoreObjectWrite::plan(OID, &state, header(), 4);
        assert_ne!(other.commit_chain(), write.commit_chain());
        let answer = write_restore_header(tv.client(), tv.key_schema(), &other).await;
        assert!(matches!(answer, Ok(FencedTxn::HeaderPresent)), "{answer:?}");
    }

    /// The root's children list as stored, in order.
    async fn root_children(tv: &TestValkey) -> Vec<String> {
        tv.client()
            .pool
            .lrange(tv.key_schema().obj_children(&oid(), "1"), 0, -1)
            .await
            .expect("LRANGE")
    }

    /// Whether `obj:chs` indexes the root's children list.
    async fn root_list_indexed(tv: &TestValkey) -> bool {
        tv.client()
            .pool
            .sismember(tv.key_schema().obj_children_index(&oid()), "1")
            .await
            .expect("SISMEMBER")
    }

    /// `state`'s write at a command limit of 10, its one children list (25
    /// children, 27 commands) cut into slices after the sections' batches.
    /// Returns the write and the batches carrying a slice, in order.
    fn cut_list_write(state: &ObjectState) -> (RestoreObjectWrite<'_>, Vec<usize>) {
        let write = RestoreObjectWrite::plan(OID, state, header(), 10);
        let sliced: Vec<usize> = (0..write.row_batches())
            .filter(|&batch| write.batch_units(batch).any(|(_, slice)| slice.is_some()))
            .collect();
        assert!(sliced.len() >= 3, "fixture: the list is cut: {sliced:?}");
        (write, sliced)
    }

    /// A children list cut across row batches lands whole and in order, and
    /// sending any batch that carries a slice twice — the Job's in-place
    /// retry of a batch whose `EXEC` ran but whose reply was lost — pushes
    /// nothing twice: a later slice's `LTRIM` takes the list back to the
    /// children before it first. The first slice's batch indexes the list in
    /// `obj:chs` with it. The header lands after the batches, the list as it
    /// was.
    ///
    /// Fails without the `LTRIM` (in-place perturbation of
    /// `queue_children_slice`, its later-slice `LTRIM` not queued): each
    /// retried later slice's children are in the list twice.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cut_children_list_lands_whole_in_order_and_a_retried_slice_adds_nothing() {
        let tv = TestValkey::setup().await;
        let state = state_of(25);
        let (write, sliced) = cut_list_write(&state);
        let mut indexed_with_the_first_slice = None;

        for batch in 0..write.row_batches() {
            let sends = if sliced.contains(&batch) { 2 } else { 1 };
            for send in 0..sends {
                let answer = write_restore_batch(tv.client(), tv.key_schema(), &write, batch).await;
                assert!(
                    matches!(answer, Ok(FencedTxn::Committed)),
                    "batch {batch}, send {send}: {answer:?}"
                );
            }
            if batch == sliced[0] {
                indexed_with_the_first_slice = Some(root_list_indexed(&tv).await);
            }
        }

        let expected: Vec<String> = (2..27u64).map(|id| id.to_string()).collect();
        assert_eq!(
            root_children(&tv).await,
            expected,
            "every child once, in order, every sliced batch sent twice"
        );
        assert_eq!(
            indexed_with_the_first_slice,
            Some(true),
            "the first slice's batch indexes the list"
        );
        let answer = write_restore_header(tv.client(), tv.key_schema(), &write).await;
        assert!(matches!(answer, Ok(FencedTxn::Committed)), "{answer:?}");
        assert_eq!(
            root_children(&tv).await,
            expected,
            "the header leaves the list"
        );
    }

    /// The parent of [`unparented_list_state`]'s children list: no section
    /// of the state, and not the pseudo-root 0, whose list key the wipe
    /// removes as a fixed key.
    const UNPARENTED: u64 = 999;

    /// `prims` sections listed as the children of [`UNPARENTED`], which has
    /// no section, so no section's keys name their list: `obj:chs` is the
    /// only index that does.
    fn unparented_list_state(prims: u64) -> ObjectState {
        let mut state = state_of(0);
        state.sections.clear();
        state.children.clear();
        let ids: Vec<u64> = (2..2 + prims).collect();
        for &id in &ids {
            state.sections.insert(
                id,
                Section {
                    section_id: id,
                    parent_id: UNPARENTED,
                    name: format!("s{id}"),
                    section_type: 6,
                    section_order: 0,
                    first_version: 1,
                    last_version: 9,
                },
            );
        }
        state.children.insert(UNPARENTED, ids);
        state
    }

    /// A cut list written only in part — its first slice and the next, no
    /// header, as a Job killed between batches leaves it — is indexed in
    /// `obj:chs` by its first slice, so the Job-owned wipe the re-run starts
    /// with finds it and removes it with the rest of the object. The list's
    /// parent here has no section, so the index is the only way the wipe can
    /// find it (a section parent's list the wipe also finds through
    /// `obj:secs`, and the pseudo-root's as a fixed key).
    ///
    /// Fails without the first slice's `SADD` (in-place perturbation of
    /// `queue_children_slice`): the partial list survives the wipe.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_partly_written_cut_list_is_indexed_so_the_wipe_removes_it() {
        let tv = TestValkey::setup().await;
        let state = unparented_list_state(25);
        let (write, sliced) = cut_list_write(&state);
        for batch in 0..=sliced[1] {
            let answer = write_restore_batch(tv.client(), tv.key_schema(), &write, batch).await;
            assert!(
                matches!(answer, Ok(FencedTxn::Committed)),
                "batch {batch}: {answer:?}"
            );
        }
        let list = tv
            .key_schema()
            .obj_children(&oid(), &UNPARENTED.to_string());
        let partial: Vec<String> = tv.client().pool.lrange(&list, 0, -1).await.expect("LRANGE");
        assert!(
            !partial.is_empty() && partial.len() < 25,
            "fixture: part of the list: {partial:?}"
        );

        wipe_object_for_restore(tv.client(), tv.key_schema(), OID, WipeStream::Keep)
            .await
            .expect("the wipe");

        assert!(!exists(&tv, list).await, "the partial list is wiped");
        assert!(!exists(&tv, tv.key_schema().obj_children_index(&oid())).await);
        assert!(!exists(&tv, tv.key_schema().obj_sections_index(&oid())).await);
    }

    /// The clear-flags fence: a flagged survivor at the expected `seq` is
    /// cleared (`Committed`: the banded `seq`, the band added to the prior
    /// `restore_seq_bump`, the write's `commit_chain`, no flag, no stream);
    /// the same write again answers `AlreadyLanded`; another write over the
    /// cleared header, a flagged header whose `seq` moved, and an unflagged
    /// one at the expected `seq` each answer `FenceMismatch` and write
    /// nothing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_clear_flags_fence_answers_committed_already_landed_or_mismatch() {
        let tv = TestValkey::setup().await;
        hset(
            &tv,
            &[
                ("seq", "10"),
                ("superseded", "1"),
                ("superseded_by", "99"),
                ("ext_gc_scheduled", "1"),
                ("restore_seq_bump", "7"),
                ("commit_chain", "old"),
            ],
        )
        .await;
        seed_stream(&tv).await;
        let write = ClearFlagsWrite::new(OID, 10, 1_010, 1_000);

        let answer = write_clear_flags_header(tv.client(), tv.key_schema(), &write).await;
        assert!(
            matches!(answer, Ok(ClearFlagsOutcome::Committed)),
            "{answer:?}"
        );
        let cleared = hgetall(&tv).await;
        assert_eq!(cleared["seq"], "1010");
        assert_eq!(cleared["restore_seq_bump"], "1007");
        assert_eq!(cleared["commit_chain"], write.commit_chain);
        for flag in ["superseded", "superseded_by", "ext_gc_scheduled"] {
            assert!(!cleared.contains_key(flag), "{flag}: {cleared:?}");
        }
        assert!(!exists(&tv, tv.key_schema().layer_stream(OID)).await);

        let again = write_clear_flags_header(tv.client(), tv.key_schema(), &write).await;
        assert!(
            matches!(again, Ok(ClearFlagsOutcome::AlreadyLanded)),
            "the same write landed: {again:?}"
        );

        let other = ClearFlagsWrite::new(OID, 10, 1_010, 1_000);
        let answer = write_clear_flags_header(tv.client(), tv.key_schema(), &other).await;
        assert!(
            matches!(
                &answer,
                Ok(ClearFlagsOutcome::FenceMismatch { seq: Some(seq), flagged: false })
                    if seq == "1010"
            ),
            "another write's chain is not this header's: {answer:?}"
        );
        assert_eq!(hgetall(&tv).await, cleared, "nothing written");

        hset(&tv, &[("seq", "11"), ("superseded", "1")]).await;
        seed_stream(&tv).await;
        let moved = hgetall(&tv).await;
        let answer = write_clear_flags_header(tv.client(), tv.key_schema(), &other).await;
        assert!(
            matches!(
                &answer,
                Ok(ClearFlagsOutcome::FenceMismatch { seq: Some(seq), flagged: true })
                    if seq == "11"
            ),
            "a moved seq: {answer:?}"
        );
        assert_eq!(hgetall(&tv).await, moved, "nothing written");
        assert!(exists(&tv, tv.key_schema().layer_stream(OID)).await);

        let unflagged = ClearFlagsWrite::new(OID, 11, 1_011, 1_000);
        let _: i64 = tv
            .client()
            .pool
            .hdel(tv.key_schema().obj_header(&oid()), "superseded")
            .await
            .expect("HDEL");
        let answer = write_clear_flags_header(tv.client(), tv.key_schema(), &unflagged).await;
        assert!(
            matches!(
                &answer,
                Ok(ClearFlagsOutcome::FenceMismatch { seq: Some(seq), flagged: false })
                    if seq == "11"
            ),
            "no flag: {answer:?}"
        );
        assert!(exists(&tv, tv.key_schema().layer_stream(OID)).await);
    }

    /// The value removal's two hashes: `H1`'s group has one field row,
    /// `H2`'s (the last, whose `HDEL` the header carries) none.
    const H1: [u8; 32] = [0xA1; 32];
    const H2: [u8; 32] = [0xB2; 32];

    /// Whether `obj:ext` names `hash`.
    async fn ext_named(tv: &TestValkey, hash: &[u8; 32]) -> bool {
        tv.client()
            .pool
            .hexists(tv.key_schema().obj_ext_index(&oid()), sha256_to_hex(hash))
            .await
            .expect("HEXISTS")
    }

    /// The field row `(1, "a")`'s key and its `obj:flds` member.
    fn field_row(tv: &TestValkey) -> (String, String) {
        let member = KeySchema::field_member("1", "a");
        (
            tv.key_schema().obj_field_from_member(&oid(), &member),
            member,
        )
    }

    /// Whether the field row `(1, "a")` or its `obj:flds` member stands.
    async fn field_row_stands(tv: &TestValkey) -> (bool, bool) {
        let (key, member) = field_row(tv);
        let score: Option<f64> = tv
            .client()
            .pool
            .zscore(tv.key_schema().obj_fields_index(&oid()), member)
            .await
            .expect("ZSCORE");
        (exists(tv, key).await, score.is_some())
    }

    /// A live header at `seq` 10 (bump 7), a stream, both hashes in
    /// `obj:ext` and `H1`'s field row with its index member.
    async fn seed_value_removal(tv: &TestValkey) {
        hset(
            tv,
            &[
                ("seq", "10"),
                ("restore_seq_bump", "7"),
                ("commit_chain", "old"),
            ],
        )
        .await;
        seed_stream(tv).await;
        let ext: Vec<(String, String)> = [H1, H2]
            .iter()
            .map(|h| (sha256_to_hex(h), "1".to_owned()))
            .collect();
        let _: i64 = tv
            .client()
            .pool
            .hset(tv.key_schema().obj_ext_index(&oid()), ext)
            .await
            .expect("HSET obj:ext");
        let (key, member) = field_row(tv);
        let _: i64 = tv
            .client()
            .pool
            .hset(key, [("value", "x")])
            .await
            .expect("HSET field row");
        let _: i64 = tv
            .client()
            .pool
            .zadd(
                tv.key_schema().obj_fields_index(&oid()),
                None,
                None,
                false,
                false,
                (1.0, member),
            )
            .await
            .expect("ZADD obj:flds");
    }

    /// A value-removal batch under a header at `expected_seq`: `H1`'s field
    /// row, by its member, and `H1`'s `HDEL` (`H2`'s is the header's).
    fn value_removal(expected_seq: u64) -> ValueRemovalBatch {
        let mut batch = ValueRemovalBatch::new(OID, expected_seq);
        batch.push_row(RemovedRow {
            time_sample: false,
            member: KeySchema::field_member("1", "a"),
        });
        batch.push_hdel(H1);
        batch
    }

    /// The value removal's fence (`BandedFence::NotRetiring`; test review 4;
    /// fail-on-old), on the post-K1 shape: a live header at the expected
    /// `seq` lets the row batch commit (the row, its member and `H1` gone,
    /// `H2` kept) and then the header (`Committed`: the banded `seq`, the
    /// band added to the prior bump, the write's `commit_chain`, no stream,
    /// and `H2` gone with it); the same header write again answers
    /// `AlreadyLanded`. A header whose `seq` moved, and a flagged (retiring)
    /// header at the expected `seq`, make both the row batch and the header
    /// write answer `FenceMismatch` and write nothing. With the fence's flag
    /// check dropped, the flagged header's row batch commits.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_value_removal_fence_answers_committed_already_landed_or_mismatch() {
        let tv = TestValkey::setup().await;
        seed_value_removal(&tv).await;
        let write = value_removal(10);
        let header = BandedHeaderWrite::new(OID, 10, 1_010, 1_000).removing_ext_hash(Some(H2));

        let rows = remove_value_rows(tv.client(), tv.key_schema(), &write).await;
        assert!(matches!(rows, Ok(SeqFencedOutcome::Committed)), "{rows:?}");
        assert_eq!(
            field_row_stands(&tv).await,
            (false, false),
            "the row is gone"
        );
        assert!(!ext_named(&tv, &H1).await, "H1's HDEL rode its batch");
        assert!(ext_named(&tv, &H2).await, "H2's waits for the header");
        let answer = write_value_removal_header(tv.client(), tv.key_schema(), &header).await;
        assert!(
            matches!(answer, Ok(SeqFencedOutcome::Committed)),
            "{answer:?}"
        );
        let banded = hgetall(&tv).await;
        assert_eq!(banded["seq"], "1010");
        assert_eq!(banded["restore_seq_bump"], "1007");
        assert_eq!(banded["commit_chain"], header.commit_chain);
        assert!(!ext_named(&tv, &H2).await, "H2's HDEL rode the header");
        assert!(!exists(&tv, tv.key_schema().layer_stream(OID)).await);
        let again = write_value_removal_header(tv.client(), tv.key_schema(), &header).await;
        assert!(
            matches!(again, Ok(SeqFencedOutcome::AlreadyLanded)),
            "the same write landed: {again:?}"
        );

        // A moved seq: the plan saw 10, the header reads 1010.
        seed_value_removal(&tv).await;
        hset(&tv, &[("seq", "1010")]).await;
        let moved = hgetall(&tv).await;
        let stale = BandedHeaderWrite::new(OID, 10, 1_010, 1_000).removing_ext_hash(Some(H2));
        let rows = remove_value_rows(tv.client(), tv.key_schema(), &write).await;
        let answer = write_value_removal_header(tv.client(), tv.key_schema(), &stale).await;
        for (what, answer) in [("the row batch", rows), ("the header", answer)] {
            assert!(
                matches!(
                    &answer,
                    Ok(SeqFencedOutcome::FenceMismatch { seq: Some(seq), flagged: false })
                        if seq == "1010"
                ),
                "{what} on a moved seq: {answer:?}"
            );
        }
        assert_eq!(hgetall(&tv).await, moved, "nothing written");
        assert_eq!(field_row_stands(&tv).await, (true, true));
        assert!(ext_named(&tv, &H1).await && ext_named(&tv, &H2).await);
        assert!(exists(&tv, tv.key_schema().layer_stream(OID)).await);

        // A retiring header at the expected seq.
        hset(&tv, &[("seq", "10"), ("superseded", "1")]).await;
        let flagged = hgetall(&tv).await;
        let rows = remove_value_rows(tv.client(), tv.key_schema(), &write).await;
        let answer = write_value_removal_header(tv.client(), tv.key_schema(), &stale).await;
        for (what, answer) in [("the row batch", rows), ("the header", answer)] {
            assert!(
                matches!(
                    &answer,
                    Ok(SeqFencedOutcome::FenceMismatch { seq: Some(seq), flagged: true })
                        if seq == "10"
                ),
                "{what} on a retiring header: {answer:?}"
            );
        }
        assert_eq!(hgetall(&tv).await, flagged, "nothing written");
        assert_eq!(field_row_stands(&tv).await, (true, true));
        assert!(ext_named(&tv, &H1).await && ext_named(&tv, &H2).await);
        assert!(exists(&tv, tv.key_schema().layer_stream(OID)).await);
    }
}
