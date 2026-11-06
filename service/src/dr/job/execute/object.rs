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

//! The objects stage (SDD SR-9.5, SR-9.6): every restore and flag clear of a
//! plan, admitted by the Job's restore memory, largest charge first.
//!
//! **A restore** (SR-9.6, in order):
//!
//! 1. re-fetch the planned lineage by its exact handles, with no request
//!    deadline (each call bounded by the adapter's own ceiling). A backup
//!    the plan selected that now reads absent contradicts the plan: exit 1,
//!    never a step-down to another backup;
//! 2. decode it as the planner did (`plan::lineage::decode_planned`); a
//!    failure, a composed head that is not the planned one, or a backup whose
//!    identity is not the mapped layer's (PD-4) also exits 1;
//! 3. read the stream's high-water (`storage::stream_high_water`);
//! 4. band and flatten (`checkpoint::seq_band`), before any write;
//! 5. wipe the object, keeping its stream;
//! 6. backfill the tail-carried values: to Valkey (`SET NX` and the meta
//!    increment the recount overwrites), then to the backup store, each PUT
//!    recorded in the run's own PUTs, before any header write;
//! 7. write the batches, the header last, with the stream `UNLINK`
//!    (`storage::layer::dr_restore`);
//! 8. emit no external-value release.
//!
//! **A flag clear** (PM B2's order; never the wipe): read the high-water,
//! band, remove the retirement carrier and its attempts, then one
//! transaction that removes the stream, stamps the banded `seq` and clears
//! the flags.

use std::sync::Arc;

use bytes::Bytes;
use futures::StreamExt;

use super::memory::{inline_value_charge, write_charge, RestoreGate, RestoreMemory};
use super::{
    step_failed, write_step, At, BackfillReport, BatchRecord, ExecSeams, ExecutionReport,
    MemoryReport, ObjectWrite, StepError, WriteKind,
};
use crate::checkpoint::apply::OUTSIZED_BAND_MARGIN_MULTIPLE;
use crate::checkpoint::compose::ExtValuePart;
use crate::checkpoint::seq_band::{
    band_is_outsized, flatten_to_seeded_seq, seq_band, SeqBand, SeqCeiling,
};
use crate::checkpoint::store::{CheckpointHandle, CheckpointStore};
use crate::dr::job::plan::lineage::{decode_planned, DecodedLineage};
use crate::dr::job::plan::model::{
    BandInputs, ObjectAction, ObjectPlan, PlannedLineage, RestorePlan,
};
use crate::dr::job::{BackupAccess, JobExit, JobMode, JobRun};
use crate::storage::client::txn_timeout;
use crate::storage::external_values::{
    ExternalValueStore, ValkeyExternalValueStore, STORE_VALUES_MAX_INFLIGHT,
};
use crate::storage::keys::sha256_to_hex;
use crate::storage::layer::dr_restore::{
    read_header_commit_chain, remove_retirement_carrier, wipe_object_for_restore,
    write_clear_flags_header, write_restore_batch, write_restore_header, ClearFlagsOutcome,
    ClearFlagsWrite, FencedTxn, RestoreHeader, RestoreObjectWrite, WipeStream,
    CLEAR_FLAGS_COMMANDS,
};
use crate::storage::layer::CheckpointCursorStamp;
use crate::storage::stream_high_water::{read_stream_floor, StreamFloor};

/// One object of the stage, with its charge.
struct Item<'p> {
    object: &'p ObjectPlan,
    charge: u64,
}

/// What one object's pipeline produced.
struct Written {
    object_id: u64,
    write: ObjectWrite,
    backfill: BackfillReport,
}

/// Write every restore and flag clear of `plan` (module docs), filling
/// `report`'s objects, backfills and memory.
pub(super) async fn write_objects(
    run: &JobRun,
    plan: &RestorePlan,
    memory: &RestoreMemory,
    seams: &ExecSeams,
    report: &mut ExecutionReport,
) -> Result<(), JobExit> {
    let mut items: Vec<Item<'_>> = plan
        .objects
        .iter()
        .filter_map(|object| match &object.action {
            ObjectAction::Restore { lineage, .. } => Some(Item {
                object,
                charge: write_charge(
                    lineage.footprint.fetched_bytes,
                    lineage.footprint.state_bytes,
                )
                .saturating_add(inline_value_charge(lineage.footprint.largest_inline_bytes)),
            }),
            ObjectAction::ClearFlags { .. } => Some(Item { object, charge: 0 }),
            ObjectAction::Keep | ObjectAction::Remap { .. } | ObjectAction::Nothing => None,
        })
        .collect();
    // Largest charge first, then by object: a FIFO budget admits in this
    // order, which shortens the tail.
    items.sort_by(|a, b| {
        b.charge
            .cmp(&a.charge)
            .then(a.object.object_id.cmp(&b.object.object_id))
    });
    let total = items.len();
    let gate = RestoreGate::new(memory, seams.budget_override());
    report.memory = MemoryReport::idle(memory);
    if total == 0 {
        return Ok(());
    }

    // By index, not by item: a stream closure whose argument borrows trips
    // the compiler's higher-ranked closure inference under the body's `Send`
    // bound.
    let items = &items;
    let mut writes = futures::stream::iter(0..total)
        .map(|index| {
            let gate = &gate;
            let item = &items[index];
            async move {
                let Some(_admission) = gate.admit(item.charge).await else {
                    tracing::error!(
                        object_id = item.object.object_id,
                        "the {mode} Job's memory budget refused an object; this is an internal \
                         defect; it exits 1",
                        mode = run.mode
                    );
                    return Err(JobExit::Failed);
                };
                match &item.object.action {
                    ObjectAction::Restore { lineage, band, .. } => {
                        restore_object(run, item.object, lineage, band, seams).await
                    }
                    ObjectAction::ClearFlags { band } => {
                        clear_flags(run, item.object, band, seams).await
                    }
                    _ => unreachable!("only restores and flag clears are items"),
                }
            }
        })
        .buffer_unordered(gate.parallelism());

    let mut done = 0usize;
    // `interval` panics on a zero period, and `JobTimings::wait_log_interval`
    // is a public field the frame's own waits accept at zero.
    let mut ticker = tokio::time::interval(
        run.clock
            .log_interval()
            .max(std::time::Duration::from_millis(1)),
    );
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    ticker.tick().await;
    loop {
        tokio::select! {
            next = writes.next() => match next {
                None => break,
                Some(Ok(written)) => {
                    done += 1;
                    report.backfill.valkey_values += written.backfill.valkey_values;
                    report.backfill.backup_puts += written.backfill.backup_puts;
                    report.objects.insert(written.object_id, written.write);
                }
                Some(Err(exit)) => return Err(exit),
            },
            _ = ticker.tick() => log_progress(run.mode, &gate, done, total),
        }
    }
    report.memory.record(&gate);
    Ok(())
}

/// The progress line, every wait-log interval.
fn log_progress(mode: JobMode, gate: &RestoreGate, done: usize, total: usize) {
    const MIB: u64 = 1024 * 1024;
    let in_flight = gate.in_flight();
    match gate.budget_bytes() {
        Some(budget) => {
            let reserved = gate.reserved_bytes() / MIB;
            let budget = budget / MIB;
            tracing::info!(
                done,
                total,
                in_flight,
                reserved_mib = reserved,
                budget_mib = budget,
                "the {mode} Job has written {done} of {total} layers; {in_flight} in flight \
                 holding {reserved} MiB of its {budget} MiB budget"
            );
        }
        None => tracing::info!(
            done,
            total,
            in_flight,
            "the {mode} Job has written {done} of {total} layers; {in_flight} in flight"
        ),
    }
}

/// The plan-contradiction ERROR (defer-45): the plan selected a generation
/// the Job cannot restore now.
fn contradiction(mode: JobMode, object: &ObjectPlan, head_seq: u64, why: &str) -> JobExit {
    let (layer, session, oid) = (&object.layer_name, &object.session_id, object.object_id);
    tracing::error!(
        object_id = oid,
        session_id = %session,
        layer_name = %layer,
        generation = head_seq,
        why = %why,
        "the {mode} Job could not restore layer {layer} of session {session} (object {oid}): \
         {why}; its plan selected generation {head_seq}, so the Job does not fall back to another \
         backup; it stops writing and exits 1 so Kubernetes retries it, and the service stays \
         held"
    );
    JobExit::Failed
}

/// The header-appeared ERROR: another writer is active.
fn header_appeared(mode: JobMode, object: &ObjectPlan) -> StepError {
    let (layer, session, oid) = (&object.layer_name, &object.session_id, object.object_id);
    tracing::error!(
        object_id = oid,
        session_id = %session,
        layer_name = %layer,
        "the {mode} Job found a header on object {oid} (layer {layer} of session {session}) \
         while it was restoring it; another writer is active; it stops writing and exits 1"
    );
    StepError::Exit(JobExit::Failed)
}

/// Fetch one planned generation by its exact handle: the bytes, or the
/// plan-contradiction exit.
async fn fetch_generation(
    mode: JobMode,
    store: &dyn CheckpointStore,
    object: &ObjectPlan,
    head_seq: u64,
    handle: &CheckpointHandle,
) -> Result<Bytes, JobExit> {
    match store.get_layer_checkpoint(handle).await {
        Ok(Some(blob)) => Ok(blob.bytes),
        Ok(None) => Err(contradiction(
            mode,
            object,
            head_seq,
            &format!("backup generation {} reads as absent", handle.seq()),
        )),
        Err(e) => Err(contradiction(
            mode,
            object,
            head_seq,
            &format!("backup generation {} could not be read: {e}", handle.seq()),
        )),
    }
}

/// Read the stream key's floor, retried in place.
pub(super) async fn stream_floor(
    run: &JobRun,
    object_id: u64,
    snap_seq: u64,
) -> Result<StreamFloor, JobExit> {
    let stream_key = run.keys.layer_stream(object_id);
    write_step(
        run,
        &|| format!("read the stream high-water of object {object_id}"),
        || async {
            read_stream_floor(&run.valkey, &stream_key, snap_seq)
                .await
                .map_err(StepError::from)
        },
    )
    .await
}

/// The band at `snap_seq` over `floor`, or the exit for a band at the
/// sequence ceiling; the outsized-band WARN as the on-access restore logs it.
/// `layer` names the layer the ceiling's ERROR names, and `verb` what the Job
/// could not do to it ("restore", or accept-loss's value removal's).
pub(super) fn band_for(
    run: &JobRun,
    layer: LayerRef<'_>,
    verb: &str,
    snap_seq: u64,
    floor: u64,
    observed_commit_span: u64,
) -> Result<SeqBand, JobExit> {
    let margin = run.config.checkpoint_seq_safety_margin;
    let mode = run.mode;
    let object_id = layer.object_id;
    let banded = seq_band(snap_seq, floor, observed_commit_span, margin);
    let band = match &banded {
        Ok(b) | Err(SeqCeiling(b)) => b.band,
    };
    if band_is_outsized(observed_commit_span, margin) {
        tracing::warn!(
            object_id,
            snap_seq,
            observed_commit_span,
            safety_margin = margin,
            margin_multiple = OUTSIZED_BAND_MARGIN_MULTIPLE,
            band,
            "apply_checkpoint: measured commit span exceeds \
             LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN by more than the expected \
             multiple; applying the measured band unclamped (clamping would \
             re-open the sequence-reuse window). Either the margin is sized \
             below this layer's real per-interval commit rate, or this \
             generation spans a checkpoint outage. Note every sequence in the \
             band is skipped and gap-filled client-side by each reconnecting \
             subscriber"
        );
    }
    banded.map_err(|SeqCeiling(b)| {
        let (layer, session) = (layer.layer_name, layer.session_id);
        tracing::error!(
            object_id,
            session_id = %session,
            layer_name = %layer,
            snap_seq,
            restore_floor = b.restore_floor,
            band = b.band,
            seeded_seq = b.seeded_seq,
            "the {mode} Job could not {verb} layer {layer} of session {session} (object \
             {object_id}): its banded seq {} reaches the sequence ceiling (2^53 - 1), at which \
             no write to it could commit; it stops writing and exits 1, and the service stays \
             held",
            b.seeded_seq
        );
        JobExit::Failed
    })
}

/// A layer an ERROR names: its object, session and layer name.
#[derive(Clone, Copy, Debug)]
pub(super) struct LayerRef<'a> {
    /// The object.
    pub object_id: u64,
    /// The session.
    pub session_id: &'a str,
    /// The layer.
    pub layer_name: &'a str,
}

impl<'a> LayerRef<'a> {
    /// The layer `object` plans.
    pub(super) fn of(object: &'a ObjectPlan) -> Self {
        Self {
            object_id: object.object_id,
            session_id: &object.session_id,
            layer_name: &object.layer_name,
        }
    }
}

/// The Valkey command timeout every pool is built with
/// (`LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`): the `T` of each transaction's
/// `txn_timeout(T, commands)`.
pub(super) fn command_timeout(run: &JobRun) -> std::time::Duration {
    std::time::Duration::from_millis(run.config.valkey_command_timeout_ms)
}

/// The foreign-stream WARN (PD-3).
pub(super) fn warn_foreign(object_id: u64, floor: &StreamFloor) {
    if let StreamFloor::Foreign(kind) = floor {
        tracing::warn!(
            object_id,
            key_type = %kind,
            "object {object_id}'s stream key held a {kind}, not a stream; the restore removes it \
             with the header write so the restored layer can be written to"
        );
    }
}

/// Restore one object from its planned lineage (module docs).
async fn restore_object(
    run: &JobRun,
    object: &ObjectPlan,
    lineage: &PlannedLineage,
    band_inputs: &BandInputs,
    seams: &ExecSeams,
) -> Result<Written, JobExit> {
    let object_id = object.object_id;
    let head_seq = lineage.head_seq;
    let BackupAccess::Store(store) = &run.backup else {
        return Err(contradiction(
            run.mode,
            object,
            head_seq,
            "there is no backup store to read the planned backup from",
        ));
    };

    // 1. The re-fetch, by the plan's exact handles.
    let base = fetch_generation(run.mode, store.as_ref(), object, head_seq, &lineage.base).await?;
    let mut tails: Vec<Bytes> = Vec::with_capacity(lineage.tails.len());
    for tail in &lineage.tails {
        tails.push(fetch_generation(run.mode, store.as_ref(), object, head_seq, tail).await?);
    }

    // 2. The decode, as the planner decoded it.
    let base_seq = lineage.base.seq();
    let decoded =
        tokio::task::spawn_blocking(move || decode_planned(object_id, base_seq, &base, &tails))
            .await
            .map_err(|join| {
                tracing::error!(
                    object_id,
                    error = %join,
                    "the {mode} Job's decode of a backup failed to join; this is an internal \
                     defect, not a broken backup; it exits 1",
                    mode = run.mode
                );
                JobExit::Failed
            })?;
    let (mut state, ext_values) = match decoded {
        DecodedLineage::Ok {
            state,
            ext_values,
            composed_seq,
        } => {
            if let Some(composed) = composed_seq.filter(|c| *c != head_seq) {
                return Err(contradiction(
                    run.mode,
                    object,
                    head_seq,
                    &format!("its backup composes to generation {composed}"),
                ));
            }
            (state, ext_values)
        }
        DecodedLineage::OverLimit => {
            return Err(contradiction(
                run.mode,
                object,
                head_seq,
                "its newest full backup is over the per-delta entry limit",
            ))
        }
        DecodedLineage::Failed(error) => {
            return Err(contradiction(
                run.mode,
                object,
                head_seq,
                &format!("its backup could not be decoded: {error}"),
            ))
        }
    };
    // A replayed reset leaves `first_version` from the pre-crash timeline;
    // a restored object starts at 0, as the on-access restore does.
    state.first_version = 0;
    // PD-4: never write one layer's backup under another layer's identity.
    if lineage.meta.session_id != object.session_id || lineage.meta.layer_name != object.layer_name
    {
        return Err(contradiction(
            run.mode,
            object,
            head_seq,
            &format!(
                "its backup names layer {} of session {}",
                lineage.meta.layer_name, lineage.meta.session_id
            ),
        ));
    }

    // 3. The stream's high-water.
    let floor_reading = stream_floor(run, object_id, head_seq).await?;
    seams.stop_at(run.mode, At::ObjectAfterHighWater(object_id))?;
    let floor = floor_reading.floor(head_seq);

    // 4. The band and the flattening: pure, before any write.
    let banded = band_for(
        run,
        LayerRef::of(object),
        "restore",
        head_seq,
        floor,
        lineage.meta.observed_commit_span,
    )?;
    flatten_to_seeded_seq(&mut state, banded.seeded_seq);

    // 5. The wipe, keeping the stream.
    write_step(run, &|| format!("wipe object {object_id}"), || async {
        wipe_object_for_restore(&run.valkey, &run.keys, object_id, WipeStream::Keep)
            .await
            .map_err(StepError::from)
    })
    .await?;
    seams.stop_at(run.mode, At::ObjectAfterWipe(object_id))?;

    // 6. The tail backfill, before any header write.
    let backfill = backfill(run, store.as_ref(), object_id, ext_values, seams).await?;

    // 7. The batches, the header last.
    let written_at_ms = if band_inputs.written_at_ms == 0 {
        crate::checkpoint::now_unix_ms()
    } else {
        band_inputs.written_at_ms
    };
    let header = RestoreHeader {
        session_id: object.session_id.clone(),
        layer_name: object.layer_name.clone(),
        created_by: lineage.meta.created_by.clone(),
        server_ts: lineage.meta.server_ts,
        cursor: CheckpointCursorStamp {
            snap_seq: head_seq,
            written_at_ms,
            restore_seq_bump: banded.band,
        },
    };
    let write = RestoreObjectWrite::plan(object_id, &state, header, seams.batch_commands);
    warn_foreign(object_id, &floor_reading);
    let batches = write_batches(run, object, &write, seams).await?;

    Ok(Written {
        object_id,
        write: ObjectWrite {
            kind: WriteKind::Restored {
                head_seq,
                base_seq,
                tails: lineage.tails.len(),
            },
            floor,
            high_water: floor_reading.high_water(),
            band: banded.band,
            seeded_seq: banded.seeded_seq,
            batches,
        },
        backfill,
    })
}

/// SR-9.6 step 6: store the tail-carried values in Valkey (no mirror), then
/// PUT each not already in this run's own PUTs to the backup store, once,
/// recording each. The values' bytes drop when it returns.
async fn backfill(
    run: &JobRun,
    store: &dyn CheckpointStore,
    object_id: u64,
    ext_values: Vec<ExtValuePart>,
    seams: &ExecSeams,
) -> Result<BackfillReport, JobExit> {
    let mut report = BackfillReport::default();
    if !ext_values.is_empty() {
        let values_store = Arc::new(ValkeyExternalValueStore::new(
            run.valkey.clone(),
            run.keys.clone(),
        ));
        for chunk in ext_values.chunks(STORE_VALUES_MAX_INFLIGHT) {
            let values: Vec<(&[u8; 32], &[u8])> = chunk
                .iter()
                .map(|(hash, bytes)| (hash, bytes.as_ref()))
                .collect();
            write_step(
                run,
                &|| format!("store the backup-carried values of object {object_id} in Valkey"),
                || async {
                    values_store
                        .store_values(&values)
                        .await
                        .map_err(StepError::from)
                },
            )
            .await?;
            report.valkey_values += chunk.len() as u64;
        }
    }
    seams.stop_at(run.mode, At::ObjectAfterBackfillValkey(object_id))?;
    for (hash, bytes) in &ext_values {
        let hash_hex = sha256_to_hex(hash);
        if run.own_puts.contains(&hash_hex) {
            continue;
        }
        if run.permit.begin_unit().await.is_err() {
            return Err(JobExit::Failed);
        }
        // Never retried by the Job: the adapter retries inside the call.
        if let Err(e) = store.put_ext_blob(&hash_hex, bytes.clone()).await {
            step_failed(
                run.mode,
                &|| {
                    format!("store backup-carried value {hash_hex} of object {object_id} in the backup store")
                },
                &e.to_string(),
                1,
            );
            return Err(JobExit::Failed);
        }
        run.own_puts.record(&hash_hex);
        report.backup_puts += 1;
    }
    seams.stop_at(run.mode, At::ObjectAfterBackfillBackup(object_id))?;
    Ok(report)
}

/// Write `write`'s row batches, then its header transaction; each batch
/// retried in place.
async fn write_batches(
    run: &JobRun,
    object: &ObjectPlan,
    write: &RestoreObjectWrite<'_>,
    seams: &ExecSeams,
) -> Result<Vec<BatchRecord>, JobExit> {
    let object_id = object.object_id;
    let command_timeout = command_timeout(run);
    let mut batches = Vec::with_capacity(write.row_batches() + 1);
    for batch in 0..write.row_batches() {
        write_step(
            run,
            &|| format!("write batch {} of object {object_id}", batch + 1),
            || async {
                match write_restore_batch(&run.valkey, &run.keys, write, batch).await {
                    Ok(FencedTxn::Committed) => Ok(()),
                    Ok(FencedTxn::HeaderPresent) => Err(header_appeared(run.mode, object)),
                    Ok(FencedTxn::Conflict) => Err(StepError::Retry(
                        "the batch lost its WATCH on the header".to_owned(),
                    )),
                    Ok(FencedTxn::Torn(e)) | Ok(FencedTxn::InDoubt(e)) => Err(StepError::Retry(e)),
                    Err(e) => Err(StepError::from(e)),
                }
            },
        )
        .await?;
        let commands = write.row_batch_commands(batch);
        batches.push(BatchRecord {
            commands,
            timeout: txn_timeout(command_timeout, commands),
        });
        seams.stop_at(run.mode, At::ObjectAfterBatch(object_id, batch))?;
    }
    seams.stop_at(run.mode, At::ObjectBeforeHeader(object_id))?;
    write_step(
        run,
        &|| format!("write the header of object {object_id}"),
        || async {
            match write_restore_header(&run.valkey, &run.keys, write).await {
                Ok(FencedTxn::Committed) => Ok(()),
                // A header carrying this write's `commit_chain` is this
                // write's, landed by an attempt whose reply was lost.
                Ok(FencedTxn::HeaderPresent) => {
                    match read_header_commit_chain(&run.valkey, &run.keys, object_id).await {
                        Ok(Some(chain)) if chain == write.commit_chain() => Ok(()),
                        Ok(_) => Err(header_appeared(run.mode, object)),
                        Err(e) => Err(StepError::from(e)),
                    }
                }
                Ok(FencedTxn::Conflict) => Err(StepError::Retry(
                    "the header transaction lost its WATCH on the header".to_owned(),
                )),
                Ok(FencedTxn::Torn(e)) | Ok(FencedTxn::InDoubt(e)) => Err(StepError::Retry(e)),
                Err(e) => Err(StepError::from(e)),
            }
        },
    )
    .await?;
    let commands = write.header_commands();
    batches.push(BatchRecord {
        commands,
        timeout: txn_timeout(command_timeout, commands),
    });
    Ok(batches)
}

/// Clear a kept survivor's retirement flags (module docs).
async fn clear_flags(
    run: &JobRun,
    object: &ObjectPlan,
    band_inputs: &BandInputs,
    seams: &ExecSeams,
) -> Result<Written, JobExit> {
    let object_id = object.object_id;
    let snap_seq = band_inputs.floor_seq;
    let floor_reading = stream_floor(run, object_id, snap_seq).await?;
    seams.stop_at(run.mode, At::ClearFlagsAfterHighWater(object_id))?;
    let floor = floor_reading.floor(snap_seq);
    let banded = band_for(run, LayerRef::of(object), "restore", snap_seq, floor, 0)?;
    warn_foreign(object_id, &floor_reading);

    write_step(
        run,
        &|| format!("remove the retirement carrier of object {object_id}"),
        || async {
            remove_retirement_carrier(&run.valkey, &run.keys, object_id)
                .await
                .map_err(StepError::from)
        },
    )
    .await?;
    seams.stop_at(run.mode, At::ClearFlagsAfterCarrier(object_id))?;

    let write = ClearFlagsWrite::new(object_id, snap_seq, banded.seeded_seq, banded.band);
    write_step(
        run,
        &|| format!("clear the retirement flags of object {object_id}"),
        || async {
            match write_clear_flags_header(&run.valkey, &run.keys, &write).await {
                Ok(ClearFlagsOutcome::Committed | ClearFlagsOutcome::AlreadyLanded) => Ok(()),
                Ok(ClearFlagsOutcome::FenceMismatch { seq, flagged }) => {
                    Err(StepError::Fatal(format!(
                        "its header reads seq {} with the retirement flags {}, not the survivor \
                         the plan saw at seq {snap_seq}",
                        seq.as_deref().unwrap_or("(absent)"),
                        if flagged { "set" } else { "clear" }
                    )))
                }
                Ok(ClearFlagsOutcome::Conflict) => Err(StepError::Retry(
                    "the flag clear lost its WATCH on the header".to_owned(),
                )),
                Ok(ClearFlagsOutcome::Torn(e)) | Ok(ClearFlagsOutcome::InDoubt(e)) => {
                    Err(StepError::Retry(e))
                }
                Err(e) => Err(StepError::from(e)),
            }
        },
    )
    .await?;
    Ok(Written {
        object_id,
        write: ObjectWrite {
            kind: WriteKind::FlagsCleared,
            floor,
            high_water: floor_reading.high_water(),
            band: banded.band,
            seeded_seq: banded.seeded_seq,
            batches: vec![BatchRecord {
                commands: CLEAR_FLAGS_COMMANDS,
                timeout: txn_timeout(command_timeout(run), CLEAR_FLAGS_COMMANDS),
            }],
        },
        backfill: BackfillReport::default(),
    })
}

#[cfg(test)]
mod tests {
    //! What the pipelines make of a fence's answer, against a real store
    //! with the write permit open: a header present at the header
    //! transaction is the write's own when it carries the write's
    //! `commit_chain` (an attempt whose reply was lost) and another writer's
    //! otherwise; a header present at a row batch is always another
    //! writer's; a flag clear whose survivor moved exits 1. The fences
    //! themselves are `storage::layer::dr_restore`'s tests'.

    use std::collections::HashMap;

    use fred::interfaces::HashesInterface;

    use super::super::test_harness::{open_run, LogCapture};
    use super::*;
    use crate::checkpoint::adapters::in_memory::InMemoryCheckpointStore;
    use crate::domain::types::{ObjectState, Section};
    use crate::storage::test_support::TestValkey;

    const OID: u64 = 7_401;

    fn object() -> ObjectPlan {
        ObjectPlan {
            object_id: OID,
            session_id: "s".to_owned(),
            layer_name: "l.usda".to_owned(),
            action: ObjectAction::Keep,
        }
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

    async fn run_on(tv: &TestValkey) -> JobRun {
        open_run(tv, Arc::new(InMemoryCheckpointStore::new())).await
    }

    async fn header_fields(tv: &TestValkey) -> HashMap<String, String> {
        tv.client()
            .pool
            .hgetall(tv.key_schema().obj_header(&OID.to_string()))
            .await
            .expect("HGETALL")
    }

    /// The header-appeared ERROR for [`object`].
    fn header_appeared_error() -> String {
        format!(
            "the restore Job found a header on object {OID} (layer l.usda of session s) while it \
             was restoring it; another writer is active; it stops writing and exits 1"
        )
    }

    /// A header transaction whose `EXEC` reply was lost but which landed: the
    /// retry finds the header present carrying this write's `commit_chain`,
    /// and counts it as written, with no ERROR.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_header_carrying_the_writes_own_chain_is_the_write_landed() {
        let tv = TestValkey::setup().await;
        let run = run_on(&tv).await;
        let state = state_of(2);
        let write =
            RestoreObjectWrite::plan(OID, &state, header(), RESTORE_BATCH_COMMANDS_FOR_TEST);
        assert_eq!(write.row_batches(), 0, "fixture: one transaction");
        let landed = write_restore_header(&run.valkey, &run.keys, &write).await;
        assert!(
            matches!(landed, Ok(FencedTxn::Committed)),
            "fixture: the lost attempt landed: {landed:?}"
        );
        let capture = LogCapture::default();
        let guard = capture.install();

        let batches = write_batches(&run, &object(), &write, &ExecSeams::production()).await;
        drop(guard);

        let batches = batches.expect("the landed header is the write's own");
        assert_eq!(
            batches,
            vec![BatchRecord {
                commands: write.header_commands(),
                timeout: txn_timeout(command_timeout(&run), write.header_commands()),
            }]
        );
        assert!(
            capture.messages("ERROR").is_empty(),
            "{:?}",
            capture.messages("ERROR")
        );
    }

    /// A header present that another write left — at the header transaction
    /// (another `commit_chain`) or at a row batch — is another writer's: one
    /// header-appeared ERROR, exit 1, no retry, and the header as it was.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_header_another_write_left_stops_the_writer_with_one_error() {
        let tv = TestValkey::setup().await;
        let run = run_on(&tv).await;
        let state = state_of(6);
        let theirs =
            RestoreObjectWrite::plan(OID, &state, header(), RESTORE_BATCH_COMMANDS_FOR_TEST);
        let landed = write_restore_header(&run.valkey, &run.keys, &theirs).await;
        assert!(matches!(landed, Ok(FencedTxn::Committed)), "{landed:?}");
        let before = header_fields(&tv).await;

        let single =
            RestoreObjectWrite::plan(OID, &state, header(), RESTORE_BATCH_COMMANDS_FOR_TEST);
        let batched = RestoreObjectWrite::plan(OID, &state, header(), 4);
        assert!(batched.row_batches() >= 2, "fixture: row batches");
        for (case, write) in [("header transaction", &single), ("row batch", &batched)] {
            let capture = LogCapture::default();
            let guard = capture.install();
            let result = write_batches(&run, &object(), write, &ExecSeams::production()).await;
            drop(guard);
            assert_eq!(result, Err(JobExit::Failed), "{case}");
            assert_eq!(
                capture.messages("ERROR"),
                vec![header_appeared_error()],
                "{case}: one ERROR, no retry"
            );
            assert_eq!(header_fields(&tv).await, before, "{case}: nothing written");
        }
    }

    /// A flag clear whose survivor no longer reads as the plan saw it (its
    /// `seq` moved) exits 1 with one ERROR saying what it read, and writes no
    /// header field.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_flag_clear_over_a_moved_survivor_exits_1_and_writes_no_header_field() {
        let tv = TestValkey::setup().await;
        let run = run_on(&tv).await;
        let _: i64 = tv
            .client()
            .pool
            .hset(
                tv.key_schema().obj_header(&OID.to_string()),
                vec![("seq", "11"), ("superseded", "1")],
            )
            .await
            .expect("HSET");
        let before = header_fields(&tv).await;
        let band = BandInputs {
            floor_seq: 10,
            observed_commit_span: 0,
            written_at_ms: 0,
            cursor: None,
        };
        let capture = LogCapture::default();
        let guard = capture.install();

        let result = clear_flags(&run, &object(), &band, &ExecSeams::production()).await;
        drop(guard);

        assert!(matches!(result, Err(JobExit::Failed)));
        assert_eq!(
            capture.messages("ERROR"),
            vec![format!(
                "the restore Job could not clear the retirement flags of object {OID} after 1 \
                 attempts: its header reads seq 11 with the retirement flags set, not the \
                 survivor the plan saw at seq 10; it stops writing and exits 1 so Kubernetes \
                 retries it, and the service stays held"
            )]
        );
        assert_eq!(header_fields(&tv).await, before, "no header field written");
    }

    /// One transaction for every fixture object here.
    const RESTORE_BATCH_COMMANDS_FOR_TEST: u64 = 1_000;
}
