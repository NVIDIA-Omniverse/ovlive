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

//! The restore coverage check (SDD SR-9.3; PD-3, PD-5): every external value
//! a restored or surviving layer references must be somewhere — Valkey, a
//! restored lineage's tails, this run's own backup-store PUT, or the backup
//! store — checked in that order, so the backup store is probed only for what
//! the first three miss.
//!
//! - **Probes** run at most [`DR_COVERAGE_PROBE_CONCURRENCY`] (32) at once,
//!   each one `ext_blob_exists` call under the adapter's call ceiling, which
//!   retries inside the call; the Job never retries a probe (defer-7 F2).
//!   Each call ticks `live_edit_dr_restore_coverage_probes_total{result}`.
//! - **Answers.** Present is covered. An error or a call past the ceiling is
//!   unknown. Absent is uncovered only when observed at or after `T_safe`;
//!   one observed before it is re-probed once, at or after `T_safe`, when the
//!   check is finalised ([`CoveragePlan::finalize`]), and that second answer
//!   decides (PD-3). A value this run stored itself is covered by that PUT
//!   and never probed again.
//! - **No backup store:** a value missing from Valkey and the tails is
//!   uncovered, with no probe and no wait.
//! - **The breaker** (PD-5): [`DR_COVERAGE_PROBE_BREAKER`] (32) consecutive
//!   probe errors, counted in completion order, end the stage with exit 1, so
//!   a backup-store brownout costs one ceiling, not one per value. Its ERROR
//!   says whether the run had written anything ([`ProbePhase`]).
//!
//! `finalize` then names each uncovered or unknown value's rows — session,
//! layer, prim path, attribute, count, hash prefix — re-fetching a restored
//! layer's planned lineage, or reading a survivor's state through the
//! `dr_restore` snapshot-read consumer, sets
//! `live_edit_dr_restore_values_uncovered`, and logs the report.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use bytes::Bytes;
use futures::{StreamExt, TryStreamExt};

use super::decide::{probe_verdict, settle_without_probe};
use super::model::{
    CoverageObject, CoveragePlan, CoverageReport, CoverageSource, ExtHash, HashVerdict,
    LayerCoverageReport, PlannedLineage, ValueSite,
};
use super::report::{log_coverage_report, value_sites};
use super::{
    DR_COVERAGE_PROBE_BREAKER, DR_COVERAGE_PROBE_CONCURRENCY, DR_PLAN_VALKEY_READ_CONCURRENCY,
};
use crate::checkpoint::apply::decode_checkpoint_ompf;
use crate::checkpoint::compose::compose_from_artifacts;
use crate::checkpoint::store::CheckpointStore;
use crate::dr::job::{BackupAccess, JobExit, JobRun};
use crate::observability::metrics::{
    DR_COVERAGE_PROBE_RESULT_ABSENT, DR_COVERAGE_PROBE_RESULT_ERROR,
    DR_COVERAGE_PROBE_RESULT_PRESENT,
};
use crate::storage::diff_read::{
    read_object_snapshot_slim, SlimSnapshotReadConsumer, SnapshotReadOf,
};
use crate::storage::dr_job::ext_value_stored;
use crate::storage::keys::sha256_to_hex;

/// Check the coverage of `universe` (each referenced value and the objects
/// referencing it): Valkey (`EXISTS`, at most
/// [`DR_PLAN_VALKEY_READ_CONCURRENCY`] at once), then `tails`, then this
/// run's own PUTs, then one probe each against the backup store. A Valkey
/// error, or the probe breaker, ends the plan with exit 1.
pub async fn check_coverage(
    run: &JobRun,
    universe: BTreeMap<ExtHash, BTreeSet<u64>>,
    tails: &HashSet<ExtHash>,
    objects: BTreeMap<u64, CoverageObject>,
) -> Result<CoveragePlan, JobExit> {
    let mode = run.mode;
    let in_valkey: Vec<(ExtHash, bool)> = futures::stream::iter(universe.keys().copied())
        .map(|hash| async move {
            ext_value_stored(&run.valkey, &run.keys, &sha256_to_hex(&hash))
                .await
                .map(|stored| (hash, stored))
        })
        .buffer_unordered(DR_PLAN_VALKEY_READ_CONCURRENCY)
        .try_collect()
        .await
        .map_err(|e| {
            tracing::error!(
                error = %e,
                "the {mode} Job could not read the store to plan the restore (the referenced \
                 values' blobs), wrote nothing and exits 1"
            );
            JobExit::Failed
        })?;
    let mut verdicts: BTreeMap<ExtHash, HashVerdict> = BTreeMap::new();
    let mut to_probe: Vec<ExtHash> = Vec::new();
    for (hash, stored) in in_valkey {
        let own_put = run.own_puts.contains(&sha256_to_hex(&hash));
        match settle_without_probe(stored, tails.contains(&hash), own_put) {
            Some(verdict) => {
                verdicts.insert(hash, verdict);
            }
            None => to_probe.push(hash),
        }
    }
    match &run.backup {
        BackupAccess::None(_) => {
            for hash in to_probe {
                verdicts.insert(hash, HashVerdict::Uncovered);
            }
        }
        BackupAccess::Store(store) => {
            if !to_probe.is_empty() {
                let at_or_after_t_safe = run.clock.at_or_after_t_safe().await.map_err(|e| {
                    tracing::error!(
                        error = %e,
                        "the {mode} Job could not read the store's clock to plan the restore, \
                         wrote nothing and exits 1"
                    );
                    JobExit::Failed
                })?;
                let answers = probe_all(run, store.as_ref(), to_probe, ProbePhase::Plan).await?;
                for (hash, answer) in answers {
                    verdicts.insert(hash, probe_verdict(&answer, at_or_after_t_safe));
                }
            }
        }
    }
    Ok(CoveragePlan {
        verdicts,
        references: universe,
        objects,
    })
}

/// When a probe stage runs: its breaker's ERROR says whether the run had
/// written anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbePhase {
    /// While the plan is made: nothing is written yet.
    Plan,
    /// When the body finalises the check, after the writes.
    Finalize,
}

/// One probe per value in `hashes`, at most [`DR_COVERAGE_PROBE_CONCURRENCY`]
/// at once, each under the adapter's call ceiling, with no retry. Each call
/// ticks the probe counter. [`DR_COVERAGE_PROBE_BREAKER`] consecutive errors,
/// in completion order, stop the stage (the calls in flight are dropped) and
/// end the run with exit 1; the ERROR's text follows `phase`.
pub async fn probe_all(
    run: &JobRun,
    store: &dyn CheckpointStore,
    hashes: Vec<ExtHash>,
    phase: ProbePhase,
) -> Result<Vec<(ExtHash, Result<bool, String>)>, JobExit> {
    let mode = run.mode;
    let ceiling = run.config.checkpoint_adapter_call_ceiling();
    let mut probes = futures::stream::iter(hashes)
        .map(|hash| async move {
            let answer =
                tokio::time::timeout(ceiling, store.ext_blob_exists(&sha256_to_hex(&hash))).await;
            let answer = match answer {
                Ok(Ok(present)) => Ok(present),
                Ok(Err(e)) => Err(e.to_string()),
                Err(_elapsed) => Err(format!(
                    "no answer within the backup store's call ceiling of {}s",
                    ceiling.as_secs()
                )),
            };
            (hash, answer)
        })
        .buffer_unordered(DR_COVERAGE_PROBE_CONCURRENCY);
    let mut answers = Vec::new();
    let mut consecutive_errors = 0usize;
    while let Some((hash, answer)) = probes.next().await {
        let result = match &answer {
            Ok(true) => DR_COVERAGE_PROBE_RESULT_PRESENT,
            Ok(false) => DR_COVERAGE_PROBE_RESULT_ABSENT,
            Err(_) => DR_COVERAGE_PROBE_RESULT_ERROR,
        };
        run.metrics.inc_dr_restore_coverage_probe(result);
        if answer.is_err() {
            consecutive_errors += 1;
            if consecutive_errors >= DR_COVERAGE_PROBE_BREAKER {
                let last_error = answer.as_ref().err().map_or("", String::as_str);
                match phase {
                    ProbePhase::Plan => tracing::error!(
                        consecutive_errors,
                        last_error = %last_error,
                        "the backup store failed {DR_COVERAGE_PROBE_BREAKER} existence checks in \
                         a row; the {mode} Job stops checking, writes nothing, and exits 1 so \
                         Kubernetes retries it"
                    ),
                    ProbePhase::Finalize => tracing::error!(
                        consecutive_errors,
                        last_error = %last_error,
                        "the backup store failed {DR_COVERAGE_PROBE_BREAKER} existence checks in \
                         a row; the {mode} Job stops checking and exits 1 so Kubernetes retries \
                         it; what it has written stays, and a re-run continues from it"
                    ),
                }
                return Err(JobExit::Failed);
            }
        } else {
            consecutive_errors = 0;
        }
        answers.push((hash, answer));
    }
    Ok(answers)
}

impl CoveragePlan {
    /// Finalise the coverage check: re-probe once, at or after `T_safe`, each
    /// value whose backup-store answer was "absent" before it (waiting for
    /// `T_safe` first, logging the wait), unless this run has since stored it
    /// itself; then name the rows of every uncovered and unknown value, set
    /// `live_edit_dr_restore_values_uncovered`, and log the report.
    ///
    /// Called by the restore body after the writer's writes
    /// (`dr::job::execute`), so a value the writer stored in the backup store
    /// itself (`JobRun::own_puts`) is covered by that PUT and not probed;
    /// accept-loss calls it before its discards. A failed read of the
    /// store's clock, or the probe breaker, ends the run with exit 1; a layer
    /// whose rows cannot be read is still reported, with why.
    pub async fn finalize(&self, run: &JobRun) -> Result<CoverageReport, JobExit> {
        let mode = run.mode;
        let mut verdicts = self.verdicts.clone();
        let pending: Vec<ExtHash> = verdicts
            .iter()
            .filter(|(_, v)| matches!(v, HashVerdict::PendingReprobe))
            .map(|(h, _)| *h)
            .collect();
        if !pending.is_empty() {
            match &run.backup {
                BackupAccess::Store(store) => {
                    run.clock
                        .wait_until_t_safe(&format!(
                            "{} referenced values read as absent from the backup store before \
                             T_safe; checking them again then",
                            pending.len()
                        ))
                        .await
                        .map_err(|e| {
                            tracing::error!(
                                error = %e,
                                "the {mode} Job could not read the store's clock to finish its \
                                 coverage check and exits 1"
                            );
                            JobExit::Failed
                        })?;
                    let mut to_probe = Vec::with_capacity(pending.len());
                    for hash in pending {
                        if run.own_puts.contains(&sha256_to_hex(&hash)) {
                            verdicts.insert(hash, HashVerdict::OwnPut);
                        } else {
                            to_probe.push(hash);
                        }
                    }
                    for (hash, answer) in
                        probe_all(run, store.as_ref(), to_probe, ProbePhase::Finalize).await?
                    {
                        verdicts.insert(hash, probe_verdict(&answer, true));
                    }
                }
                BackupAccess::None(_) => {
                    for hash in pending {
                        verdicts.insert(hash, HashVerdict::Uncovered);
                    }
                }
            }
        }

        // Name the rows of every uncovered and unknown value, per layer.
        let mut affected: BTreeMap<u64, (BTreeSet<ExtHash>, BTreeSet<ExtHash>)> = BTreeMap::new();
        for (hash, verdict) in &verdicts {
            let uncovered = matches!(verdict, HashVerdict::Uncovered);
            let unknown = matches!(verdict, HashVerdict::Unknown(_));
            if !uncovered && !unknown {
                continue;
            }
            for object_id in self.references.get(hash).into_iter().flatten() {
                let entry = affected.entry(*object_id).or_default();
                if uncovered {
                    entry.0.insert(*hash);
                } else {
                    entry.1.insert(*hash);
                }
            }
        }
        let mut layers = Vec::with_capacity(affected.len());
        for (object_id, (uncovered, unknown)) in affected {
            let Some(object) = self.objects.get(&object_id) else {
                continue;
            };
            let mut all: BTreeSet<ExtHash> = uncovered.clone();
            all.extend(unknown.iter().copied());
            let (sites, naming_error) = match name_sites(run, object_id, object, all).await {
                Ok(sites) => (sites, None),
                Err(error) => (Vec::new(), Some(error)),
            };
            let (uncovered_sites, unknown_sites): (Vec<ValueSite>, Vec<ValueSite>) =
                sites.into_iter().partition(|s| uncovered.contains(&s.hash));
            layers.push(LayerCoverageReport {
                object_id,
                session_id: object.session_id.clone(),
                layer_name: object.layer_name.clone(),
                uncovered,
                unknown,
                uncovered_sites,
                unknown_sites,
                naming_error,
            });
        }
        layers.sort_by(|a, b| {
            (a.session_id.as_str(), a.layer_name.as_str(), a.object_id).cmp(&(
                b.session_id.as_str(),
                b.layer_name.as_str(),
                b.object_id,
            ))
        });
        let report = CoverageReport { verdicts, layers };
        run.metrics
            .set_dr_restore_values_uncovered(report.uncovered_count() as u64);
        log_coverage_report(&report);
        Ok(report)
    }
}

/// Where `hashes` sit in one covered object: a restored one's planned lineage
/// re-fetched by its exact handles and decoded; a survivor's state read from
/// the store (`dr_restore` consumer). `Err` names why they could not be read.
async fn name_sites(
    run: &JobRun,
    object_id: u64,
    object: &CoverageObject,
    hashes: BTreeSet<ExtHash>,
) -> Result<Vec<ValueSite>, String> {
    match &object.source {
        CoverageSource::Survivor => {
            match read_object_snapshot_slim(
                &run.valkey,
                &run.keys,
                &object_id.to_string(),
                SlimSnapshotReadConsumer::DrRestore,
            )
            .await
            {
                Ok(SnapshotReadOf::Snapshot(snapshot)) => Ok(value_sites(&snapshot.state, &hashes)),
                Ok(SnapshotReadOf::TooLarge(_)) => {
                    Err("the surviving layer's read answered too large".to_owned())
                }
                Err(e) => Err(format!("the surviving layer could not be read: {e}")),
            }
        }
        CoverageSource::Restored(lineage) => {
            let BackupAccess::Store(store) = &run.backup else {
                return Err("there is no backup store to read the restored backup from".to_owned());
            };
            let (base, tails) = fetch_lineage(store.as_ref(), lineage).await?;
            let base_seq = lineage.base.seq();
            tokio::task::spawn_blocking(move || {
                if tails.is_empty() {
                    decode_checkpoint_ompf(object_id, &base)
                        .map(|state| value_sites(&state, &hashes))
                        .map_err(|e| e.to_string())
                } else {
                    compose_from_artifacts(object_id, &base, base_seq, &tails)
                        .map(|composed| value_sites(&composed.state, &hashes))
                        .map_err(|e| e.to_string())
                }
            })
            .await
            .map_err(|e| format!("the decode of the restored backup failed to join: {e}"))?
            .map_err(|e| format!("the restored backup could not be decoded again: {e}"))
        }
    }
}

/// A planned lineage's bytes, by its exact handles.
async fn fetch_lineage(
    store: &dyn CheckpointStore,
    lineage: &PlannedLineage,
) -> Result<(Bytes, Vec<Bytes>), String> {
    let fetch = |handle| async move {
        match store.get_layer_checkpoint(handle).await {
            Ok(Some(blob)) => Ok(blob.bytes),
            Ok(None) => Err(format!(
                "backup generation {} of object {} reads as absent",
                handle.seq(),
                handle.object_id
            )),
            Err(e) => Err(format!(
                "backup generation {} of object {} could not be read: {e}",
                handle.seq(),
                handle.object_id
            )),
        }
    };
    let base = fetch(&lineage.base).await?;
    let mut tails = Vec::with_capacity(lineage.tails.len());
    for tail in &lineage.tails {
        tails.push(fetch(tail).await?);
    }
    Ok((base, tails))
}
