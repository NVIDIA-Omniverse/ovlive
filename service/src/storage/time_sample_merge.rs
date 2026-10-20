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

//! The signed-zero merge: one object's legacy `-0.0` time-sample rows folded
//! into the `+0.0` key, so the store holds one row per section and USD time.
//!
//! `-0.0` and `0.0` are one USD time code, but a time-sample row key and its
//! `obj:tss` member spell the time by its IEEE-754 bits
//! (`keys::hex_encode_f64`), so the two are different stored rows. This build
//! never writes `-0.0`: `oodf_ffi::parse_delta` canonicalizes every time to
//! `+0.0`, and `domain::validation::validate_delta_bounds` refuses one that
//! bypassed the parser. Older builds stored the client's bits, so a `-0.0` row
//! — alone, or beside a `+0.0` row of the same section, a *pair* — can exist
//! from before an upgrade, from an older replica during a rolling upgrade, or
//! after a rollback.
//!
//! Readers already serve one sample per pair: `diff_read::snapshot_to_object_state`
//! resolves it by `domain::types::resolve_signed_zero_pair` and holds the
//! survivor at `+0.0`. This module makes the store agree, by the same rule, so
//! a merge changes nothing a reader serves.
//!
//! **Who runs it, and when.** The reference-count reconciler's object walk
//! (`storage::ext_reconcile`), on every pass, under the shared `le:gc:leader`
//! lease: the walk already lists every live object's `obj:tss` members, so an
//! object needing the merge is found by a member ending `:8000000000000000`
//! at no extra read. The walk merges before it reads the object's references,
//! then re-lists the object, so the same pass's index audit sees the merged
//! rows. No operator action; `LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS=0`, which
//! disables the reconciler, disables it too.
//!
//! **How.** Per call of at most [`MERGE_MAX_ROWS_PER_CALL`] sections:
//!
//! 1. [`plan_signed_zero_merge`] reads, in one pipeline on the maintenance
//!    pool, the header's `seq` and then, per section, both zero-time rows and
//!    whether `obj:tss` lists the `+0.0` member; decodes the rows with the
//!    full readers' decoder (`diff_read::decode_time_sample_row`) and decides
//!    each section by `resolve_signed_zero_pair`: keep the `+0.0` row, or
//!    take the `-0.0` one (which is also how a lone `-0.0` row is re-keyed).
//!    A `+0.0` row whose member is not listed is planned as absent, because
//!    readers find rows through their members and so serve the `-0.0` row:
//!    the plan takes it, and the take's `RENAME` overwrites the unlisted key.
//!    A pair with a row the decoder refuses is left alone and counted
//!    `undecodable`; readers refuse the same row.
//! 2. [`apply_signed_zero_merge`] applies the decisions in one atomic
//!    `scripts/time_sample_signed_zero_merge.lua`, which first checks that the
//!    object is live, that its `seq` is still the one read (the rewrite does
//!    not advance `seq`, so without this a stale take could overwrite a newer
//!    `+0.0` write), and that no snapshot reader holds its fence; its first
//!    write then stamps [`HEADER_FIELD_TIME_SAMPLES_MERGED_MS`] on the header,
//!    which cancels in-flight watched txn commits planned from the old rows.
//!    The script header carries the full contract.
//!
//! **Idempotent and safe under two leaders**: the `seq` guard plus the
//! script's skip of a row whose `-0.0` key or member is already gone. The
//! script also skips a keep whose `+0.0` key or member is gone by the time it
//! runs — the backstop for a race between the read and the script, the plan
//! itself never deciding keep over an unlisted `+0.0` row — and the next pass
//! re-plans that section from a fresh read.
//!
//! **External values.** The merge never writes `obj:ext` and enqueues no
//! release. The row it drops can only leave the object's index counting one
//! row too many — the leak direction — and the same pass's index audit
//! recounts the object, after which the reconciler drains any surplus as
//! usual. A take over an unlisted `+0.0` key drops no row the index audit
//! counts (it counts the rows `obj:tss` lists), so it moves nothing the audit
//! sees; whatever count the unlisted row's hash still holds is a surplus, the
//! same leak direction.
//!
//! **Never needed for correctness**, so it never costs a reconciler pass: a
//! failure counts `failed`, logs a WARN, and the walk goes on with a fresh
//! read of the object. Every outcome is counted per row on
//! `live_edit_time_sample_signed_zero_merge_total{outcome}`.

use std::collections::HashMap;
use std::sync::LazyLock;

use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
use fred::interfaces::{HashesInterface, SortedSetsInterface};
use fred::types::scripts::Script;
use fred::types::Value;

use crate::domain::types::{resolve_signed_zero_pair, SignedZeroSurvivor};
use crate::observability::metrics::{
    Metrics, TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_CHANGED,
    TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_FAILED, TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_FENCED,
    TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_MERGED, TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_NOT_LIVE,
    TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_REKEYED,
    TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_UNDECODABLE,
};
use crate::storage::client::{FredErrOrigin, ValkeyClient, POOL_LABEL_MAINTENANCE};
use crate::storage::diff_read::decode_time_sample_row;
use crate::storage::keys::KeySchema;

/// The compiled merge script. Same lazy-`SHA-1` / `NOSCRIPT`-reload story as
/// `storage::lua::UPDATE_LAYER_FAST_SCRIPT`.
pub(crate) static TIME_SAMPLE_SIGNED_ZERO_MERGE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/time_sample_signed_zero_merge.lua")));

/// The `obj:hdr` column a merge stamps with the Valkey-side millisecond time
/// of the merge.
///
/// Written only by `scripts/time_sample_signed_zero_merge.lua`, as its first
/// write, and only when it merges or re-keys a row. Nothing reads it: its
/// purpose is the write itself, which fires the header `WATCH` of every
/// in-flight **watched** txn commit on the object, so none of them can `EXEC`
/// rows planned from the pre-merge keyspace — the precedent is
/// `ext_index_audit::HEADER_FIELD_EXT_INDEX_REPAIRED_MS`. It is removed with
/// the header (retirement, the restore residue wipe) and ignored by the
/// snapshot decoder, so it never reaches a checkpoint. It does not advance
/// `seq`.
///
/// `pub` so integration tests in the external test crate can read it by name;
/// the script spells it literally and the two must match.
pub const HEADER_FIELD_TIME_SAMPLES_MERGED_MS: &str = "time_samples_merged_ms";

/// The most sections one merge call reads and writes; the script refuses
/// more. Bounds one script's duration on an object with many legacy rows.
pub const MERGE_MAX_ROWS_PER_CALL: usize = 256;

/// The time part of a `-0.0` time-sample member or row key:
/// `keys::hex_encode_f64(-0.0)`.
pub const NEGATIVE_ZERO_TIME_HEX: &str = "8000000000000000";

/// The section id of an `obj:tss` member whose time is `-0.0`
/// (`{section_id}:8000000000000000`, `KeySchema::time_sample_member`), or
/// `None` for any other member.
pub fn negative_zero_member_section(member: &str) -> Option<&str> {
    let (section_id, time_hex) = member.rsplit_once(':')?;
    (time_hex == NEGATIVE_ZERO_TIME_HEX).then_some(section_id)
}

/// The section ids of every `-0.0` member of an object's `obj:tss` listing:
/// the sections [`merge_signed_zero_rows`] works on. Empty for an object with
/// no legacy row, which is every object this build wrote.
pub fn negative_zero_sections(time_sample_members: &[String]) -> Vec<String> {
    time_sample_members
        .iter()
        .filter_map(|member| negative_zero_member_section(member))
        .map(str::to_owned)
        .collect()
}

/// What the merge does with one section's zero-time rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeAction {
    /// The `+0.0` row survives: the `-0.0` row and member go.
    Keep,
    /// The `-0.0` row survives, or is the section's only zero-time row
    /// readers see (the `+0.0` key absent, or present with no `obj:tss`
    /// member): it moves to the `+0.0` key and member.
    Take,
}

/// One section's decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedRow {
    /// The section id, as the `obj:tss` member spells it.
    pub section_id: String,
    pub action: MergeAction,
}

/// What [`plan_signed_zero_merge`] read and decided for one call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergePlan {
    pub object_id: String,
    /// The header `seq`, exactly as stored, read before the rows; the
    /// script writes nothing unless it still holds.
    pub seq: String,
    /// The sections to merge, in the order asked.
    pub rows: Vec<PlannedRow>,
    /// Sections left alone because a row of the pair failed to decode.
    pub undecodable: Vec<String>,
    /// Sections whose `-0.0` row was absent at the read (merged by another
    /// run, or an index member with no row): nothing to do.
    pub absent: u64,
}

/// The outcome of [`plan_signed_zero_merge`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanRead {
    Planned(MergePlan),
    /// The header is gone or has no decimal `seq`. Nothing to merge.
    NotLive,
}

/// The script's answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeReply {
    /// `merged`: rows whose `+0.0` key existed; `rekeyed`: lone `-0.0` rows
    /// moved to `+0.0`; `skipped`: rows left as they were (the `-0.0` key
    /// already gone, its member absent, or a keep whose `+0.0` key or member
    /// is gone).
    Merged {
        merged: u64,
        rekeyed: u64,
        skipped: u64,
    },
    /// The object's `seq` moved since the read. Nothing written.
    Changed,
    /// A snapshot reader holds the object's fence. Nothing written.
    Fenced { retry_after_ms: u64 },
    /// The object is not live (gone, replaced, or retiring). Nothing written.
    NotLive,
}

/// Per-row counts of one object's merge, by outcome — the cells of
/// `live_edit_time_sample_signed_zero_merge_total` this object added to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MergeSummary {
    pub merged: u64,
    pub rekeyed: u64,
    pub changed: u64,
    pub fenced: u64,
    pub not_live: u64,
    pub undecodable: u64,
    pub failed: u64,
}

impl MergeSummary {
    /// Rows the merge left as one row at `+0.0`: `merged + rekeyed`.
    pub fn rows_merged(&self) -> u64 {
        self.merged.saturating_add(self.rekeyed)
    }
}

/// Read one call's worth of `section_ids` (at most
/// [`MERGE_MAX_ROWS_PER_CALL`]) and decide each by
/// `domain::types::resolve_signed_zero_pair`.
///
/// One pipeline on the maintenance pool: `HGET obj:hdr seq` first, then per
/// section `HGETALL` of its `+0.0` and `-0.0` rows and `ZSCORE obj:tss` of
/// its `+0.0` member. The `seq` is read first so that any commit landing
/// after it moves it, and the script's `seq` guard then refuses a plan made
/// from rows that commit changed.
///
/// A `+0.0` row whose member `obj:tss` does not list is planned as absent
/// (readers do not see it, and serve the `-0.0` row), so the section is
/// taken; it is never decided keep. The script's own skip of a keep whose
/// `+0.0` member is gone is the backstop for a member removed between this
/// read and the script.
///
/// # Errors
///
/// A store or decode failure of the pipeline, already observed on
/// `FredErrOrigin` (a command timeout counted, an `-OOM` noted).
///
/// # Panics
///
/// When `section_ids` is longer than [`MERGE_MAX_ROWS_PER_CALL`].
pub async fn plan_signed_zero_merge(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    section_ids: &[String],
) -> Result<PlanRead, FredError> {
    assert!(
        section_ids.len() <= MERGE_MAX_ROWS_PER_CALL,
        "one merge call takes at most {MERGE_MAX_ROWS_PER_CALL} sections"
    );
    let observe = |e: FredError| {
        FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE).observe_error(&e);
        e
    };
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let pipeline = maint_pool.next().pipeline();
    let _: () = pipeline
        .hget(keys.obj_header(object_id), "seq")
        .await
        .map_err(observe)?;
    let tss_key = keys.obj_time_samples_index(object_id);
    for section_id in section_ids {
        let _: () = pipeline
            .hgetall(keys.obj_time_sample(object_id, section_id, 0.0))
            .await
            .map_err(observe)?;
        let _: () = pipeline
            .hgetall(keys.obj_time_sample(object_id, section_id, -0.0))
            .await
            .map_err(observe)?;
        let _: () = pipeline
            .zscore(&tss_key, KeySchema::time_sample_member(section_id, 0.0))
            .await
            .map_err(observe)?;
    }
    let replies: Vec<Value> = pipeline.all().await.map_err(observe)?;
    if replies.len() != 1 + 3 * section_ids.len() {
        return Err(FredError::new(
            FredErrorKind::Parse,
            format!(
                "signed-zero merge read: {} replies for {} sections",
                replies.len(),
                section_ids.len()
            ),
        ));
    }
    let mut replies = replies.into_iter();
    let seq: Option<String> = replies
        .next()
        .unwrap_or(Value::Null)
        .convert()
        .map_err(observe)?;
    let Some(seq) = seq.filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())) else {
        return Ok(PlanRead::NotLive);
    };

    let mut plan = MergePlan {
        object_id: object_id.to_owned(),
        seq,
        rows: Vec::with_capacity(section_ids.len()),
        undecodable: Vec::new(),
        absent: 0,
    };
    for section_id in section_ids {
        let positive: HashMap<String, String> = replies
            .next()
            .unwrap_or(Value::Null)
            .convert()
            .map_err(observe)?;
        let negative: HashMap<String, String> = replies
            .next()
            .unwrap_or(Value::Null)
            .convert()
            .map_err(observe)?;
        // `ZSCORE` answers nil for a member `obj:tss` does not list.
        let positive_listed = !matches!(replies.next(), None | Some(Value::Null));
        // A `+0.0` row whose member is not listed is one no reader finds
        // (readers reach rows through `obj:tss`), so readers serve the `-0.0`
        // row: plan it as absent, so the section is taken, and the take's
        // `RENAME` overwrites the unlisted key. Deciding keep would drop the
        // row readers serve; the script refuses that keep, and the section
        // would never converge.
        let positive = if positive_listed {
            decode_time_sample_row(object_id, section_id, 0.0, &positive)
        } else {
            Ok(None)
        };
        let negative = decode_time_sample_row(object_id, section_id, -0.0, &negative);
        let action = match (positive, negative) {
            // Checked first: with no `-0.0` row there is nothing to merge,
            // whatever the `+0.0` row holds.
            (_, Ok(None)) => {
                plan.absent += 1;
                continue;
            }
            (Err(_), _) | (_, Err(_)) => {
                plan.undecodable.push(section_id.clone());
                continue;
            }
            (Ok(None), Ok(Some(_))) => MergeAction::Take,
            (Ok(Some(positive)), Ok(Some(negative))) => {
                match resolve_signed_zero_pair(&positive, &negative) {
                    SignedZeroSurvivor::PositiveZero => MergeAction::Keep,
                    SignedZeroSurvivor::NegativeZero => MergeAction::Take,
                }
            }
        };
        plan.rows.push(PlannedRow {
            section_id: section_id.clone(),
            action,
        });
    }
    Ok(PlanRead::Planned(plan))
}

/// Apply `plan` in one atomic `time_sample_signed_zero_merge.lua` on the
/// maintenance pool. A plan with no row to act on sends nothing and answers
/// `Merged` with all counts zero.
///
/// # Errors
///
/// A store failure or a script refusal (`MERGE_ARGS`, `WRONGTYPE_KEY`, an
/// out-of-memory or other write-gate refusal of its first write, a slot
/// migration's): nothing was written. Already observed on `FredErrOrigin`.
pub async fn apply_signed_zero_merge(
    client: &ValkeyClient,
    keys: &KeySchema,
    plan: &MergePlan,
) -> Result<MergeReply, FredError> {
    if plan.rows.is_empty() {
        return Ok(MergeReply::Merged {
            merged: 0,
            rekeyed: 0,
            skipped: 0,
        });
    }
    let object_id = plan.object_id.as_str();
    let mut script_keys = vec![
        keys.obj_header(object_id),
        keys.obj_time_samples_index(object_id),
        keys.obj_snapshot_fence(object_id),
        // Never written: makes the store refuse the script while the object's
        // slot migrates, so it cannot stop after its first write (see the
        // script's "Migration sentinel").
        keys.migration_sentinel(object_id),
    ];
    script_keys.reserve(2 * plan.rows.len());
    let mut argv = Vec::with_capacity(3 + 2 * plan.rows.len());
    argv.push(keys.prefix().to_owned());
    argv.push(object_id.to_owned());
    argv.push(plan.seq.clone());
    for row in &plan.rows {
        script_keys.push(keys.obj_time_sample(object_id, &row.section_id, 0.0));
        script_keys.push(keys.obj_time_sample(object_id, &row.section_id, -0.0));
        argv.push(row.section_id.clone());
        argv.push(
            match row.action {
                MergeAction::Keep => "K",
                MergeAction::Take => "T",
            }
            .to_owned(),
        );
    }
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let reply: Vec<String> = TIME_SAMPLE_SIGNED_ZERO_MERGE_SCRIPT
        .evalsha_with_reload::<Vec<String>, _, _>(maint_pool.next(), script_keys, argv)
        .await
        .inspect_err(|e| FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE).observe_error(e))?;
    parse_reply(&reply)
}

/// Decode the script's flat all-string reply. An unrecognised shape is a
/// protocol fault (a stale cached script body, say) and comes back as an
/// error, so the caller treats it exactly like a failed call.
fn parse_reply(reply: &[String]) -> Result<MergeReply, FredError> {
    let bad = || {
        FredError::new(
            FredErrorKind::Parse,
            format!("time_sample_signed_zero_merge returned an unrecognised reply: {reply:?}"),
        )
    };
    let num = |i: usize| -> Result<u64, FredError> {
        reply
            .get(i)
            .and_then(|s| s.parse::<u64>().ok())
            .ok_or_else(bad)
    };
    match reply.first().map(String::as_str) {
        Some("NOT_LIVE") if reply.len() == 1 => Ok(MergeReply::NotLive),
        Some("CHANGED") if reply.len() == 2 => Ok(MergeReply::Changed),
        Some("FENCED") if reply.len() == 2 => Ok(MergeReply::Fenced {
            retry_after_ms: num(1)?,
        }),
        Some("MERGED") if reply.len() == 4 => Ok(MergeReply::Merged {
            merged: num(1)?,
            rekeyed: num(2)?,
            skipped: num(3)?,
        }),
        _ => Err(bad()),
    }
}

/// Merge `object_id`'s legacy `-0.0` time-sample rows in the sections
/// `section_ids` ([`negative_zero_sections`] of its `obj:tss` listing): plan
/// and apply per call of at most [`MERGE_MAX_ROWS_PER_CALL`] sections.
///
/// Never fails: every outcome is counted per row on
/// `live_edit_time_sample_signed_zero_merge_total{outcome}` and returned. A
/// call answered `CHANGED` moves on to the next call, which reads afresh; one
/// answered `FENCED` or `NOT_LIVE`, or that failed, ends this object's merge
/// for this pass, and the sections it did not reach are left for the next
/// pass, uncounted. A failure or an undecodable row logs a WARN naming the
/// layer; nothing else logs above DEBUG here (the reconciler's pass-complete
/// line carries the pass's total).
pub async fn merge_signed_zero_rows(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    section_ids: &[String],
    metrics: Option<&Metrics>,
) -> MergeSummary {
    let mut summary = MergeSummary::default();
    for chunk in section_ids.chunks(MERGE_MAX_ROWS_PER_CALL) {
        let rows = chunk.len() as u64;
        let plan = match plan_signed_zero_merge(client, keys, object_id, chunk).await {
            Ok(PlanRead::Planned(plan)) => plan,
            Ok(PlanRead::NotLive) => {
                summary.not_live += rows;
                break;
            }
            Err(e) => {
                summary.failed += rows;
                warn_failed(object_id, rows, &e);
                break;
            }
        };
        let undecodable = plan.undecodable.len() as u64;
        if undecodable > 0 {
            summary.undecodable += undecodable;
            tracing::warn!(
                object_id,
                sections = ?plan.undecodable,
                "a layer holds a time-sample row at time 0.0 or -0.0 that cannot be decoded, \
                 beside a -0.0 row of the same section; the signed-zero merge left these sections \
                 alone, and readers of the layer refuse the same row. Keyspace corruption: the \
                 row needs an operator"
            );
        }
        let planned = plan.rows.len() as u64;
        match apply_signed_zero_merge(client, keys, &plan).await {
            Ok(MergeReply::Merged {
                merged,
                rekeyed,
                skipped,
            }) => {
                summary.merged += merged;
                summary.rekeyed += rekeyed;
                tracing::debug!(
                    object_id,
                    merged,
                    rekeyed,
                    skipped,
                    absent = plan.absent,
                    "merged a layer's -0.0 time-sample rows into 0.0"
                );
            }
            Ok(MergeReply::Changed) => {
                summary.changed += planned;
                tracing::debug!(
                    object_id,
                    rows = planned,
                    "signed-zero merge skipped rows: a commit landed on the layer since the read"
                );
            }
            Ok(MergeReply::Fenced { retry_after_ms }) => {
                summary.fenced += planned;
                tracing::debug!(
                    object_id,
                    rows = planned,
                    retry_after_ms,
                    "signed-zero merge deferred to the next pass: a snapshot reader holds the \
                     layer's fence"
                );
                break;
            }
            Ok(MergeReply::NotLive) => {
                summary.not_live += planned;
                break;
            }
            Err(e) => {
                summary.failed += planned;
                warn_failed(object_id, planned, &e);
                break;
            }
        }
    }
    if let Some(m) = metrics {
        for (outcome, n) in [
            (TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_MERGED, summary.merged),
            (
                TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_REKEYED,
                summary.rekeyed,
            ),
            (
                TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_CHANGED,
                summary.changed,
            ),
            (TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_FENCED, summary.fenced),
            (
                TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_NOT_LIVE,
                summary.not_live,
            ),
            (
                TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_UNDECODABLE,
                summary.undecodable,
            ),
            (TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_FAILED, summary.failed),
        ] {
            m.inc_time_sample_signed_zero_merge(outcome, n);
        }
    }
    summary
}

fn warn_failed(object_id: &str, rows: u64, error: &FredError) {
    tracing::warn!(
        object_id,
        rows,
        error = %error,
        "the signed-zero merge of a layer's -0.0 time-sample rows failed; nothing was written for \
         these rows, readers still serve one sample per time, and the next reference-count \
         reconciler pass retries"
    );
}
