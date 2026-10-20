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

//! The external-value backup repair (SDD §22.17): re-sends, from Valkey, the
//! backup copies the ext-mirror never confirmed.
//!
//! The ext-mirror (`checkpoint::ext_mirror`) is fire-and-forget: a copy it
//! drops, fails to upload, discards at shutdown or loses with a crash leaves
//! the value in Valkey with no copy in the backup store, and a restore after
//! Valkey loses the value cannot bring it back. What records the difference is
//! the value's own meta: a delivered copy is confirmed there
//! (`backed_up_in = <store id>:<ms>`, `storage::ext_backup`), and every other
//! outcome leaves the field absent. This task finds the values Valkey holds
//! with no confirmation and repairs each one: an existence check on the backup
//! store, then, for one the store lacks, an upload of Valkey's bytes.
//!
//! ## One pass
//!
//! 1. Valkey's clock (`TIME`) sets the cutoffs: the age gate `G`
//!    ([`ExtBackupRepairTiming::age_gate`], `max(adapter call ceiling, 60 s)`)
//!    and the audit age (24 h).
//! 2. A walk of every `le:ext:meta:*` key (`orphan_sweep::scan_keyspace`), one
//!    pipelined `HMGET backed_up_in last_store_ms` per chunk of
//!    [`META_CHUNK`] keys, sorts each value into one of four classes
//!    ([`MetaClass`]): **backed** (confirmed under this store's identity
//!    within the audit age), **young** (stored within `G`: the mirror is
//!    still on it), **audit** (confirmed here, but longer ago than the audit
//!    age) or **owed** (anything else: no field, another store's identity, an
//!    unparseable field). Unparseable reads as owed because acting costs one
//!    existence check or one upload, never a delete.
//! 3. Owed values, chunk by chunk, then up to
//!    [`ExtBackupRepairTiming::audit_per_pass`] audit values after the walk:
//!    `ext_blob_exists` at most [`EXT_BACKUP_PROBE_CONCURRENCY`] at a time.
//!    Present: confirmed (`already_stored`). Absent: re-sent (an audited value
//!    also counts `confirmed_copy_missing`, with a WARN). A failed check
//!    (`probe_failed`) ends checking and re-sending for the pass; the walk
//!    still finishes, for the owed gauge. So does a spent time budget:
//!    checks and re-sends together get [`ExtBackupRepairTiming::work_budget`]
//!    from the pass's start ([`EXT_BACKUP_REPAIR_WORK_BUDGET`] in
//!    production); once it is spent, the check or re-send in flight is dropped
//!    and its value stays owed, and checking and re-sending end for the pass.
//! 4. A re-send, one value at a time: the value's length (`STRLEN`); a
//!    reservation of its charge on the mirror's byte budget
//!    ([`ExtMirrorSender::try_reserve`], never waiting on the budget,
//!    retried once a [`ExtBackupRepairTiming::reserve_retry`] for up to
//!    [`ExtBackupRepairTiming::reserve_wait`]); the bytes (`GETRANGE`s of at
//!    most `storage::ext_backup::EXT_VALUE_READ_CHUNK_BYTES` into one buffer
//!    of the reserved length, through the allowlisted reader
//!    `storage::ext_backup::read_ext_value_bytes`); the
//!    upload, under [`ExtBackupRepairTiming::upload_ceiling`] (`G`: 180 s on
//!    StorageApi at the defaults, 60 s on the local adapters, whose 5 s call
//!    ceiling bounds a rename, not a write of up to the upload size cap), with
//!    the reservation riding in the bytes; then the confirmation. Existence
//!    checks stay under the adapter call ceiling. A value larger than the whole budget
//!    is `oversized` and stays owed; a budget with no room for the whole wait
//!    (`budget_busy`) and a failed or timed-out upload (`resend_failed`) each
//!    end re-sending for the pass; a failed Valkey read, or a value that
//!    changed between the size and the end of the read, is `resend_failed`
//!    for that value alone; a value Valkey no longer holds, before or during
//!    the read, is `value_gone`.
//! 5. A complete walk publishes `checkpoint_ext_backup_owed`: the values older
//!    than `G` still without a copy, every one counted, including those a
//!    spent time budget left unchecked. A pass whose walk failed, or that was
//!    cancelled, publishes nothing, because its count would miss the metas it
//!    never read; the next pass, or the next leader's, publishes. So does a
//!    pass that runs past its deadline, the interval: a backstop that, with
//!    checks and re-sends cut at the time budget, only a walk slower than the
//!    rest of the interval or a stuck Valkey read reaches.
//!
//! ## Why it is safe
//!
//! - **It never deletes and never creates a meta.** It reads Valkey and adds
//!   copies to the backup store; the confirmation script writes only to a meta
//!   that exists and still carries the stamp read with the value
//!   (`ext_backup_confirm.lua`). A value the GC collects mid-pass is
//!   `value_gone` at the read, or declines its confirmation.
//! - **Absent is acted on, never trusted.** An `Ok(false)` from
//!   `ext_blob_exists` may be a cached not-found (`CheckpointStore::
//!   ext_blob_exists`); the worst a wrong one costs is an upload of bytes the
//!   store already holds.
//! - **Memory is constant.** One chunk of keys and its reply, at most
//!   [`EXT_BACKUP_PROBE_CONCURRENCY`] existence checks (no bodies), the audit
//!   list (at most [`ExtBackupRepairTiming::audit_per_pass`] hashes), and one
//!   value in flight, charged to `M_ext` with the mirror's copies: a re-send
//!   never takes room the budget does not have, and never waits for it. The
//!   value is read into one buffer of the length its charge was taken for,
//!   one range at a time, so what a re-send holds is its charge, and its peak
//!   is the charge plus one range, which stays inside the data-pool
//!   connection's counted read buffer.
//! - **One replica sweeps.** It is a consumer of `le:gc:leader`, like the GC,
//!   the orphan sweep and the reference-count reconciler, shaped like the
//!   reconciler (`storage::ext_reconcile::spawn_ext_ref_reconcile`): the lock
//!   is taken before each pass, and a non-leader publishes 0. The pass does
//!   not renew the lock; the GC task's refresh ticker does. Two leaders after
//!   a takeover cost only duplicate checks and uploads, because every step is
//!   idempotent.
//! - **No content re-hash.** Valkey's bytes are uploaded as stored, the same
//!   posture fetch-on-miss and the re-base backfill take toward the bytes they
//!   copy (SDD §11.1).
//!
//! What it does not cover: a value Valkey loses within `G` plus one interval
//! of a skipped copy; a value larger than the mirror's whole byte budget
//! (reachable only for one uploaded under a larger upload size limit);
//! no clean-up fence yet guards its uploads against the checkpoint collect
//! (see the upload's call site); and, until b.g65's fence, a confirmation
//! from an upload or existence check that answered just before the collect
//! deleted the copy, landing after the collect cleared the field: the value
//! reads as backed up, with no copy, until the daily audit re-checks it
//! (about 24 h).

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::checkpoint::ext_mirror::{ExtMirrorSender, ReserveRefusal};
use crate::checkpoint::store::CheckpointStore;
use crate::observability::metrics::{
    SharedMetrics, EXT_BACKUP_REPAIR_ALREADY_STORED, EXT_BACKUP_REPAIR_BUDGET_BUSY,
    EXT_BACKUP_REPAIR_CONFIRMED_COPY_MISSING, EXT_BACKUP_REPAIR_OVERSIZED,
    EXT_BACKUP_REPAIR_PROBE_FAILED, EXT_BACKUP_REPAIR_RESEND_FAILED, EXT_BACKUP_REPAIR_RESENT,
    EXT_BACKUP_REPAIR_VALUE_GONE,
};
use crate::storage::client::ValkeyClient;
use crate::storage::ext_backup::{
    parse_backed_up_in, read_backup_marks, read_ext_value_bytes, stored_ext_value_len,
    valkey_time_ms, BackupMarks, ExtBackupMarks, ExtValueRead,
};
use crate::storage::external_values::{hash_from_hex, try_acquire_gc_leader};
use crate::storage::keys::KeySchema;
use crate::storage::leader_lease::LeaderIdentity;
use crate::storage::orphan_sweep::scan_keyspace;

/// How often the leader runs a pass, and each pass's deadline.
pub const EXT_BACKUP_REPAIR_INTERVAL: Duration = Duration::from_secs(300);

/// How long, from a pass's start, its existence checks and re-sends may run
/// before the pass stops both and finishes its walk for the owed gauge
/// (module docs, step 3). 240 s leaves the walk 60 s of the
/// [`EXT_BACKUP_REPAIR_INTERVAL`] deadline for its remaining chunks, each one
/// pipelined `HMGET` round trip per [`META_CHUNK`] metas, so a slow backup
/// store cuts the pass's work short instead of its owed count. And it is the
/// reserve wait plus the StorageApi upload ceiling at the defaults
/// ([`EXT_BACKUP_RESERVE_WAIT`] + 180 s), so the pass's first re-send can use
/// about both in full: less the time the checks before it and its own Valkey
/// reads took, which the budget also counts.
pub const EXT_BACKUP_REPAIR_WORK_BUDGET: Duration = Duration::from_secs(240);

// The budget must end before the pass deadline, or the deadline cuts the pass
// first and it publishes nothing.
const _: () =
    assert!(EXT_BACKUP_REPAIR_WORK_BUDGET.as_secs() < EXT_BACKUP_REPAIR_INTERVAL.as_secs());

/// The floor of the age gate `G`: a value stored more recently than `G` is
/// left to the mirror. `G` is the adapter call ceiling when that is longer,
/// so an upload the mirror may still have in flight is not repeated.
pub const EXT_BACKUP_MIN_AGE: Duration = Duration::from_secs(60);

/// How old a confirmation may grow before the audit checks it again.
pub const EXT_BACKUP_AUDIT_AGE: Duration = Duration::from_secs(24 * 60 * 60);

/// The most confirmed values one pass audits.
pub const EXT_BACKUP_AUDIT_PER_PASS: usize = 2_000;

/// The most existence checks in flight at once.
pub const EXT_BACKUP_PROBE_CONCURRENCY: usize = 8;

/// How long a re-send waits for room in the mirror's byte budget before the
/// pass stops re-sending (`budget_busy`).
pub const EXT_BACKUP_RESERVE_WAIT: Duration = Duration::from_secs(60);

/// How often a re-send asks the budget for room again while it waits.
pub const EXT_BACKUP_RESERVE_RETRY: Duration = Duration::from_secs(1);

/// How many meta keys share one pipelined `HMGET`.
pub const META_CHUNK: usize = 128;

/// The repair's timing. Production uses [`Self::production`], whose values are
/// fixed constants, not settings; tests build their own to make values old
/// enough to act on without waiting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtBackupRepairTiming {
    /// How often the leader runs a pass, and each pass's deadline.
    pub interval: Duration,
    /// `G`: values whose `last_store_ms` is newer than `now - G` are left to
    /// the mirror.
    pub age_gate: Duration,
    /// Confirmations older than this are audited.
    pub audit_age: Duration,
    /// The most confirmed values one pass audits.
    pub audit_per_pass: usize,
    /// How long a re-send waits for room in the byte budget.
    pub reserve_wait: Duration,
    /// How often it asks again while it waits.
    pub reserve_retry: Duration,
    /// How long a re-send's upload may take before the pass gives up on it
    /// (`resend_failed`) and stops re-sending. In production it equals `G`,
    /// the time the design allows for one upload, the mirror's or the
    /// repair's: 180 s on StorageApi at the defaults, 60 s on the local
    /// adapters, whose 5 s call ceiling bounds a rename, not a write of up to
    /// the upload size cap. A field of its own rather than a read of
    /// [`Self::age_gate`], so a test can zero the age gate and still bound
    /// uploads. Existence checks keep the adapter call ceiling.
    pub upload_ceiling: Duration,
    /// How long, from the pass's start, checks and re-sends may run before
    /// the pass stops both and finishes its walk
    /// ([`EXT_BACKUP_REPAIR_WORK_BUDGET`] in production). One at or above
    /// [`Self::interval`] never fires: the pass deadline comes first.
    pub work_budget: Duration,
}

impl ExtBackupRepairTiming {
    /// The fixed production timing, with `G = max(call_ceiling, 60 s)` as
    /// both the age gate and the upload ceiling; `call_ceiling` is
    /// `Config::checkpoint_adapter_call_ceiling`.
    pub fn production(call_ceiling: Duration) -> Self {
        Self {
            interval: EXT_BACKUP_REPAIR_INTERVAL,
            age_gate: call_ceiling.max(EXT_BACKUP_MIN_AGE),
            audit_age: EXT_BACKUP_AUDIT_AGE,
            audit_per_pass: EXT_BACKUP_AUDIT_PER_PASS,
            reserve_wait: EXT_BACKUP_RESERVE_WAIT,
            reserve_retry: EXT_BACKUP_RESERVE_RETRY,
            upload_ceiling: call_ceiling.max(EXT_BACKUP_MIN_AGE),
            work_budget: EXT_BACKUP_REPAIR_WORK_BUDGET,
        }
    }
}

/// Which of the four classes a value's meta reads as (module docs, step 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaClass {
    /// Confirmed under this store's identity within the audit age.
    Backed,
    /// Stored within the age gate; the mirror is still on it.
    Young,
    /// Confirmed under this store's identity, longer ago than the audit age.
    Audit,
    /// No usable confirmation: absent, another store's, or unparseable.
    Owed,
}

/// Classify one meta's [`BackupMarks`] against this store's identity and the
/// pass's cutoffs (Valkey milliseconds): `young_cutoff_ms = now - G`,
/// `audit_cutoff_ms = now - audit age`. A `last_store_ms` that does not parse
/// is not young.
pub fn classify(
    marks: &BackupMarks,
    store_id: &str,
    young_cutoff_ms: u64,
    audit_cutoff_ms: u64,
) -> MetaClass {
    let confirmed_at = marks
        .backed_up_in
        .as_deref()
        .and_then(parse_backed_up_in)
        .filter(|(id, _)| *id == store_id)
        .map(|(_, ms)| ms);
    if confirmed_at.is_some_and(|ms| ms >= audit_cutoff_ms) {
        return MetaClass::Backed;
    }
    let young = marks
        .last_store_ms
        .as_deref()
        .and_then(|raw| raw.parse::<u64>().ok())
        .is_some_and(|ms| ms >= young_cutoff_ms);
    if young {
        MetaClass::Young
    } else if confirmed_at.is_some() {
        MetaClass::Audit
    } else {
        MetaClass::Owed
    }
}

/// What one completed pass saw and did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PassSummary {
    /// Meta keys the walk read.
    pub metas: u64,
    /// Values classed backed.
    pub backed: u64,
    /// Values classed young.
    pub young: u64,
    /// Values classed owed.
    pub owed_seen: u64,
    /// Values classed audit within the pass's audit cap, checked after the
    /// walk unless checking stopped (a failed check, or the spent time
    /// budget).
    pub audited: u64,
    /// Values classed audit beyond the pass's audit cap, left for a later
    /// pass.
    pub audit_deferred: u64,
    // The pass's outcomes, one per value, each counted on the
    // `checkpoint_ext_backup_repair_total` cell of the same name as it
    // happens.
    pub already_stored: u64,
    pub resent: u64,
    pub resend_failed: u64,
    pub probe_failed: u64,
    pub value_gone: u64,
    pub budget_busy: u64,
    pub oversized: u64,
    pub confirmed_copy_missing: u64,
    /// Values older than `G` still without a copy at the end of the pass: the
    /// owed gauge's reading.
    pub owed: u64,
    /// Whether the time budget ([`ExtBackupRepairTiming::work_budget`]) ran
    /// out while the pass could still check or re-send, ending both.
    pub work_budget_spent: bool,
}

impl PassSummary {
    /// Whether the pass acted on, or failed to act on, anything.
    fn eventful(&self) -> bool {
        self.work_budget_spent
            || self.already_stored
                + self.resent
                + self.resend_failed
                + self.probe_failed
                + self.value_gone
                + self.budget_busy
                + self.oversized
                + self.confirmed_copy_missing
                + self.owed
                > 0
    }
}

/// Why a pass ended without a complete walk: it publishes nothing.
#[derive(Debug)]
pub enum PassError {
    /// A Valkey read the walk depends on failed: the clock, the scan, or a
    /// chunk's `HMGET`.
    Valkey {
        op: &'static str,
        source: fred::error::Error,
    },
    /// The pass was cancelled (shutdown).
    Cancelled,
    /// The pass ran past its deadline, the interval: a backstop, since checks
    /// and re-sends stop at the time budget before it.
    Deadline,
}

impl std::fmt::Display for PassError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PassError::Valkey { op, source } => write!(f, "{op} failed: {source}"),
            PassError::Cancelled => write!(f, "the pass was cancelled"),
            PassError::Deadline => write!(f, "the pass ran past its deadline"),
        }
    }
}

/// One value to check: its hash and the `last_store_ms` read with it (`""`
/// when the meta carried none), which guards its confirmation.
struct Candidate {
    hash_hex: String,
    last_store_ms: String,
}

/// Whether a candidate was owed, or confirmed and audited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateKind {
    Owed,
    Audit,
}

/// How one re-send ended.
enum ResendOutcome {
    /// Uploaded.
    Resent,
    /// Larger than the whole budget; stays owed.
    Oversized,
    /// Valkey no longer holds it.
    ValueGone,
    /// A per-value failure (a Valkey read, or a value that changed between
    /// the size and the end of the read); the pass carries on.
    Failed,
    /// No room in the budget for the whole wait; the pass stops re-sending.
    BudgetBusy,
    /// The upload failed or timed out; the pass stops re-sending.
    UploadFailed,
}

/// The pass's running state: what it may still do, until when, and its
/// summary.
struct PassState {
    probing_stopped: bool,
    resending_stopped: bool,
    /// When the time budget for checks and re-sends runs out.
    work_deadline: tokio::time::Instant,
    summary: PassSummary,
}

impl PassState {
    fn new(work_deadline: tokio::time::Instant) -> Self {
        Self {
            probing_stopped: false,
            resending_stopped: false,
            work_deadline,
            summary: PassSummary::default(),
        }
    }

    /// Whether the time budget has run out, ending checks and re-sends
    /// ([`Self::spend_budget`]) when it has.
    fn budget_spent(&mut self) -> bool {
        if tokio::time::Instant::now() < self.work_deadline {
            return false;
        }
        self.spend_budget();
        true
    }

    /// End checking and re-sending for the pass because the time budget ran
    /// out, logging it once.
    fn spend_budget(&mut self) {
        if self.summary.work_budget_spent || (self.probing_stopped && self.resending_stopped) {
            return;
        }
        self.probing_stopped = true;
        self.resending_stopped = true;
        self.summary.work_budget_spent = true;
        tracing::info!(
            "external-value backup repair spent its time for checks and re-sends; it stops both \
             for this pass and finishes its walk for the owed count"
        );
    }
}

/// The backup repair: its store, the backup store, the mirror's budget and
/// timing. Build it with [`Self::new`] and run it with
/// [`spawn_ext_backup_repair`]; [`Self::run_pass`] runs one pass without the
/// leader lock, for tests.
pub struct ExtBackupRepair {
    client: ValkeyClient,
    keys: KeySchema,
    store: Arc<dyn CheckpointStore>,
    mirror: ExtMirrorSender,
    marks: ExtBackupMarks,
    call_ceiling: Duration,
    timing: ExtBackupRepairTiming,
    metrics: Option<SharedMetrics>,
}

impl ExtBackupRepair {
    /// A repair over `client`'s store and the backup store `store`, whose
    /// identity is `store_id` (`Config::checkpoint_store_identity`), sharing
    /// `mirror`'s byte budget, with each existence check bounded by
    /// `call_ceiling` (`Config::checkpoint_adapter_call_ceiling`) and the
    /// production timing, whose upload ceiling,
    /// [`ExtBackupRepairTiming::upload_ceiling`], bounds each re-send's upload
    /// (`max(call_ceiling, 60 s)`).
    pub fn new(
        client: ValkeyClient,
        keys: KeySchema,
        store: Arc<dyn CheckpointStore>,
        mirror: ExtMirrorSender,
        store_id: String,
        call_ceiling: Duration,
        metrics: Option<SharedMetrics>,
    ) -> Self {
        let marks = ExtBackupMarks::new(client.clone(), keys.clone(), store_id, metrics.clone());
        Self {
            client,
            keys,
            store,
            mirror,
            marks,
            call_ceiling,
            timing: ExtBackupRepairTiming::production(call_ceiling),
            metrics,
        }
    }

    /// This repair with `timing` in place of the production one.
    pub fn with_timing(mut self, timing: ExtBackupRepairTiming) -> Self {
        self.timing = timing;
        self
    }

    /// The timing in force.
    pub fn timing(&self) -> &ExtBackupRepairTiming {
        &self.timing
    }

    /// The backup store identity the repair confirms under.
    pub fn store_id(&self) -> &str {
        self.marks.store_id()
    }

    fn count(&self, outcome: &'static str) {
        if let Some(m) = self.metrics.as_ref() {
            m.inc_checkpoint_ext_backup_repair(outcome, 1);
        }
    }

    /// Run one pass (module docs): checks and re-sends bounded by the timing's
    /// work budget, the whole pass by its interval. The spawned task runs it
    /// only while this replica holds `le:gc:leader`; called directly, it
    /// takes no lock. Outcomes are counted as they
    /// happen, so a pass that ends early has still counted what it did.
    pub async fn run_pass(&self, cancel: &CancellationToken) -> Result<PassSummary, PassError> {
        match tokio::time::timeout(self.timing.interval, self.walk(cancel)).await {
            Ok(result) => result,
            Err(_elapsed) => Err(PassError::Deadline),
        }
    }

    /// The pass's body: the clock, the walk with each chunk's owed values,
    /// then the audit.
    async fn walk(&self, cancel: &CancellationToken) -> Result<PassSummary, PassError> {
        let started = tokio::time::Instant::now();
        // A budget too large to add never runs out (the pass deadline does).
        let work_deadline = started
            .checked_add(self.timing.work_budget)
            .unwrap_or_else(|| started + Duration::from_secs(u64::from(u32::MAX)));
        let mut state = PassState::new(work_deadline);
        let now_ms = valkey_time_ms(&self.client)
            .await
            .map_err(|source| PassError::Valkey { op: "TIME", source })?;
        let young_cutoff_ms = now_ms.saturating_sub(millis(self.timing.age_gate));
        let audit_cutoff_ms = now_ms.saturating_sub(millis(self.timing.audit_age));

        let pattern = self.keys.ext_meta_scan_pattern();
        let (maint_pool, _maint_token) = self.client.acquire_maintenance();
        let mut stream = scan_keyspace(maint_pool.next(), &pattern);

        let mut chunk: Vec<String> = Vec::with_capacity(META_CHUNK);
        let mut audit: Vec<Candidate> = Vec::new();
        loop {
            let next = tokio::select! {
                biased;
                () = cancel.cancelled() => return Err(PassError::Cancelled),
                item = stream.next() => item,
            };
            let finished = next.is_none();
            if let Some(item) = next {
                let key = item.map_err(|source| PassError::Valkey {
                    op: "SCAN ext:meta",
                    source,
                })?;
                match key.as_str() {
                    Some(key) => chunk.push(key.to_owned()),
                    None => tracing::debug!(
                        key = ?key,
                        "external-value backup repair skipped a meta key that is not UTF-8"
                    ),
                }
            }
            if chunk.len() >= META_CHUNK || (finished && !chunk.is_empty()) {
                let owed = self
                    .classify_chunk(
                        &chunk,
                        young_cutoff_ms,
                        audit_cutoff_ms,
                        &mut audit,
                        &mut state,
                    )
                    .await?;
                chunk.clear();
                self.serve(owed, CandidateKind::Owed, &mut state, cancel)
                    .await?;
            }
            if finished {
                break;
            }
        }
        state.summary.audited = audit.len() as u64;
        self.serve(audit, CandidateKind::Audit, &mut state, cancel)
            .await?;
        Ok(state.summary)
    }

    /// Read one chunk's marks and sort its values: owed ones are returned for
    /// the caller to serve, audit ones go onto `audit` up to the pass's cap.
    async fn classify_chunk(
        &self,
        chunk: &[String],
        young_cutoff_ms: u64,
        audit_cutoff_ms: u64,
        audit: &mut Vec<Candidate>,
        state: &mut PassState,
    ) -> Result<Vec<Candidate>, PassError> {
        let replies = read_backup_marks(&self.client, chunk)
            .await
            .map_err(|source| PassError::Valkey {
                op: "HMGET ext:meta",
                source,
            })?;
        let mut owed = Vec::new();
        for (key, marks) in chunk.iter().zip(replies) {
            state.summary.metas += 1;
            // Only a well-formed content hash is ever uploaded under.
            let Some(hash_hex) = self
                .keys
                .ext_meta_hash_hex(key)
                .filter(|h| hash_from_hex(h).is_some())
            else {
                tracing::debug!(
                    key = %key,
                    "external-value backup repair skipped a meta key with no content hash"
                );
                continue;
            };
            let class = classify(&marks, self.store_id(), young_cutoff_ms, audit_cutoff_ms);
            let candidate = || Candidate {
                hash_hex: hash_hex.to_owned(),
                last_store_ms: marks.last_store_ms.clone().unwrap_or_default(),
            };
            match class {
                MetaClass::Backed => state.summary.backed += 1,
                MetaClass::Young => state.summary.young += 1,
                MetaClass::Owed => {
                    state.summary.owed_seen += 1;
                    owed.push(candidate());
                }
                MetaClass::Audit if audit.len() < self.timing.audit_per_pass => {
                    audit.push(candidate());
                }
                MetaClass::Audit => state.summary.audit_deferred += 1,
            }
        }
        Ok(owed)
    }

    /// Check `candidates` against the backup store, then re-send the ones it
    /// lacks, adding to the owed reading every one still without a copy.
    async fn serve(
        &self,
        candidates: Vec<Candidate>,
        kind: CandidateKind,
        state: &mut PassState,
        cancel: &CancellationToken,
    ) -> Result<(), PassError> {
        if candidates.is_empty() {
            return Ok(());
        }
        if !state.probing_stopped {
            state.budget_spent();
        }
        if state.probing_stopped {
            // Unchecked: an owed value stays owed, a confirmed one keeps its
            // confirmation.
            if kind == CandidateKind::Owed {
                state.summary.owed += candidates.len() as u64;
            }
            return Ok(());
        }

        let absent = self.probe(candidates, kind, state, cancel).await?;
        for candidate in absent {
            if !state.resending_stopped {
                state.budget_spent();
            }
            if kind == CandidateKind::Audit {
                state.summary.confirmed_copy_missing += 1;
                self.count(EXT_BACKUP_REPAIR_CONFIRMED_COPY_MISSING);
                if state.resending_stopped {
                    tracing::warn!(
                        hash_prefix = prefix(&candidate.hash_hex),
                        "external-value backup repair found a copy confirmed in the backup store \
                         missing from it; re-sending has stopped for this pass, so a later pass \
                         re-sends it"
                    );
                } else {
                    tracing::warn!(
                        hash_prefix = prefix(&candidate.hash_hex),
                        "external-value backup repair found a copy confirmed in the backup store \
                         missing from it; re-sending it from the store"
                    );
                }
            }
            if state.resending_stopped {
                state.summary.owed += 1;
                continue;
            }
            let work_deadline = state.work_deadline;
            let outcome = tokio::select! {
                biased;
                outcome = self.resend(&candidate, cancel) => outcome?,
                () = tokio::time::sleep_until(work_deadline) => {
                    // Dropped mid-way, with its upload if one is in flight;
                    // the value stays owed.
                    state.spend_budget();
                    state.summary.owed += 1;
                    continue;
                }
            };
            match outcome {
                ResendOutcome::Resent => {
                    state.summary.resent += 1;
                    self.count(EXT_BACKUP_REPAIR_RESENT);
                }
                ResendOutcome::ValueGone => {
                    state.summary.value_gone += 1;
                    self.count(EXT_BACKUP_REPAIR_VALUE_GONE);
                }
                ResendOutcome::Oversized => {
                    state.summary.oversized += 1;
                    state.summary.owed += 1;
                    self.count(EXT_BACKUP_REPAIR_OVERSIZED);
                }
                ResendOutcome::Failed => {
                    state.summary.resend_failed += 1;
                    state.summary.owed += 1;
                    self.count(EXT_BACKUP_REPAIR_RESEND_FAILED);
                }
                ResendOutcome::UploadFailed => {
                    state.summary.resend_failed += 1;
                    state.summary.owed += 1;
                    state.resending_stopped = true;
                    self.count(EXT_BACKUP_REPAIR_RESEND_FAILED);
                }
                ResendOutcome::BudgetBusy => {
                    state.summary.budget_busy += 1;
                    state.summary.owed += 1;
                    state.resending_stopped = true;
                    self.count(EXT_BACKUP_REPAIR_BUDGET_BUSY);
                }
            }
        }
        Ok(())
    }

    /// `ext_blob_exists` for each candidate, at most
    /// [`EXT_BACKUP_PROBE_CONCURRENCY`] at once, each under the call ceiling.
    /// A present value is confirmed (`already_stored`); the absent ones are
    /// returned. The first failed check, or the time budget running out, stops
    /// the pass's checks and re-sends: the checks still in flight are dropped,
    /// and every unanswered owed value stays owed.
    async fn probe(
        &self,
        candidates: Vec<Candidate>,
        kind: CandidateKind,
        state: &mut PassState,
        cancel: &CancellationToken,
    ) -> Result<Vec<Candidate>, PassError> {
        let total = candidates.len() as u64;
        let mut answered = 0u64;
        let mut absent = Vec::new();
        let ceiling = self.call_ceiling;
        let store = &self.store;
        let mut checks = futures::stream::iter(candidates)
            .map(|candidate| async move {
                let answer =
                    tokio::time::timeout(ceiling, store.ext_blob_exists(&candidate.hash_hex)).await;
                (candidate, answer)
            })
            .buffer_unordered(EXT_BACKUP_PROBE_CONCURRENCY);
        let work_deadline = state.work_deadline;
        loop {
            let next = tokio::select! {
                biased;
                () = cancel.cancelled() => return Err(PassError::Cancelled),
                next = checks.next() => next,
                () = tokio::time::sleep_until(work_deadline) => {
                    // The checks in flight are dropped; every unanswered
                    // owed value stays owed.
                    state.spend_budget();
                    break;
                }
            };
            let Some((candidate, answer)) = next else {
                break;
            };
            match answer {
                Ok(Ok(true)) => {
                    answered += 1;
                    state.summary.already_stored += 1;
                    self.count(EXT_BACKUP_REPAIR_ALREADY_STORED);
                    self.marks
                        .confirm(&candidate.hash_hex, &candidate.last_store_ms)
                        .await;
                }
                Ok(Ok(false)) => {
                    answered += 1;
                    absent.push(candidate);
                }
                Ok(Err(e)) => {
                    self.probe_failed(&candidate, &e.to_string(), state);
                    break;
                }
                Err(_elapsed) => {
                    self.probe_failed(&candidate, "the existence check timed out", state);
                    break;
                }
            }
        }
        if state.probing_stopped && kind == CandidateKind::Owed {
            // The failed check and every one never answered.
            state.summary.owed += total - answered;
        }
        Ok(absent)
    }

    fn probe_failed(&self, candidate: &Candidate, error: &str, state: &mut PassState) {
        state.summary.probe_failed += 1;
        state.probing_stopped = true;
        state.resending_stopped = true;
        self.count(EXT_BACKUP_REPAIR_PROBE_FAILED);
        tracing::warn!(
            hash_prefix = prefix(&candidate.hash_hex),
            error,
            "external-value backup repair could not check the backup store; it stops checking \
             and re-sending for this pass and finishes its walk for the owed count"
        );
    }

    /// Re-send one value from Valkey (module docs, step 4). The value is read
    /// only after its charge is reserved, into one buffer of the reserved
    /// length, so the charge covers all the re-send holds.
    async fn resend(
        &self,
        candidate: &Candidate,
        cancel: &CancellationToken,
    ) -> Result<ResendOutcome, PassError> {
        let hash_hex = candidate.hash_hex.as_str();
        // An absent key reads 0 here, like an empty value; the read below
        // tells them apart, at the cost of the smallest charge.
        let len = match stored_ext_value_len(&self.client, &self.keys, hash_hex).await {
            Ok(len) => len,
            Err(e) => {
                tracing::warn!(
                    hash_prefix = prefix(hash_hex),
                    error = %e,
                    "external-value backup repair could not read a value's size; it stays owed"
                );
                return Ok(ResendOutcome::Failed);
            }
        };

        let reservation = match self.reserve(len, cancel).await? {
            Ok(reservation) => reservation,
            Err(ReserveRefusal::Oversized) => {
                tracing::warn!(
                    hash_prefix = prefix(hash_hex),
                    bytes_len = len,
                    budget_bytes = self.mirror.budget_bytes(),
                    "external-value backup repair cannot re-send a value larger than the mirror's \
                     whole byte budget; it stays without a backup copy until \
                     LIVE_EDIT_MAX_LAYER_SIZE is raised to at least its bytes_len, or \
                     resources.limits.memory is raised until the copy memory \
                     (live_edit_checkpoint_ext_mirror_budget_bytes) exceeds its bytes_len"
                );
                return Ok(ResendOutcome::Oversized);
            }
            Err(ReserveRefusal::NoRoom) => {
                tracing::info!(
                    hash_prefix = prefix(hash_hex),
                    bytes_len = len,
                    waited_secs = self.timing.reserve_wait.as_secs(),
                    "external-value backup repair found no room in the mirror's byte budget; it \
                     stops re-sending for this pass"
                );
                return Ok(ResendOutcome::BudgetBusy);
            }
        };

        // The read fills one buffer of exactly the reserved length, so the
        // charge covers all it holds (module docs, Memory is constant).
        let read = read_ext_value_bytes(
            &self.client,
            &self.keys,
            hash_hex,
            reservation.reserved_len(),
        )
        .await;
        let bytes = match read {
            Ok(ExtValueRead::Bytes(bytes)) => bytes,
            Ok(ExtValueRead::Gone) => return Ok(ResendOutcome::ValueGone),
            Ok(ExtValueRead::Changed { now_len }) => {
                // The key changed between the size and the end of the read (a
                // verified upload's heal replaces a damaged copy). Its bytes
                // are no longer the reserved length; the next pass re-reads.
                tracing::debug!(
                    hash_prefix = prefix(hash_hex),
                    reserved_len = reservation.reserved_len(),
                    now_len,
                    "external-value backup repair read a value that changed since its size was \
                     read; it stays owed"
                );
                return Ok(ResendOutcome::Failed);
            }
            Err(e) => {
                tracing::warn!(
                    hash_prefix = prefix(hash_hex),
                    error = %e,
                    "external-value backup repair could not read a value from the store; it stays \
                     owed"
                );
                return Ok(ResendOutcome::Failed);
            }
        };
        let bytes = reservation.wrap(bytes);

        // No clean-up fence here yet: this upload can land after the
        // checkpoint collect deleted the hash's copy, leaving an orphaned blob
        // (a leak, never a loss). b.g65's fence lands after this change and
        // adds its check here.
        // The upload ceiling, not the call ceiling: on the local adapters the
        // latter is a 5 s rename bound, too short for a large value's write.
        let upload = tokio::time::timeout(
            self.timing.upload_ceiling,
            self.store.put_ext_blob(hash_hex, bytes),
        );
        let answer = tokio::select! {
            biased;
            () = cancel.cancelled() => return Err(PassError::Cancelled),
            answer = upload => answer,
        };
        match answer {
            Ok(Ok(())) => {
                self.marks.confirm(hash_hex, &candidate.last_store_ms).await;
                Ok(ResendOutcome::Resent)
            }
            Ok(Err(e)) => {
                tracing::warn!(
                    hash_prefix = prefix(hash_hex),
                    error = %e,
                    "external-value backup repair could not upload a value's backup copy; it \
                     stops re-sending for this pass"
                );
                Ok(ResendOutcome::UploadFailed)
            }
            Err(_elapsed) => {
                tracing::warn!(
                    hash_prefix = prefix(hash_hex),
                    ceiling_secs = self.timing.upload_ceiling.as_secs(),
                    "external-value backup repair's upload of a value's backup copy timed out; it \
                     stops re-sending for this pass"
                );
                Ok(ResendOutcome::UploadFailed)
            }
        }
    }

    /// A reservation for a value of `len` bytes, asking again every
    /// `reserve_retry` while the budget has no room, for at most
    /// `reserve_wait`. Never queues on the budget, so a send on the store
    /// path is never held behind it.
    async fn reserve(
        &self,
        len: u64,
        cancel: &CancellationToken,
    ) -> Result<Result<crate::checkpoint::ExtMirrorReservation, ReserveRefusal>, PassError> {
        let started = Instant::now();
        loop {
            match self.mirror.try_reserve(len) {
                Err(ReserveRefusal::NoRoom) if started.elapsed() < self.timing.reserve_wait => {
                    tokio::select! {
                        biased;
                        () = cancel.cancelled() => return Err(PassError::Cancelled),
                        () = tokio::time::sleep(self.timing.reserve_retry) => {}
                    }
                }
                answer => return Ok(answer),
            }
        }
    }

    /// Run one pass as the leader, log it, and publish the owed gauge from a
    /// complete one.
    async fn run_one_pass(&self, replica_id: &str, cancel: &CancellationToken) {
        let started = Instant::now();
        match self.run_pass(cancel).await {
            Ok(summary) => {
                if let Some(m) = self.metrics.as_ref() {
                    m.record_checkpoint_ext_backup_owed(summary.owed, replica_id);
                }
                macro_rules! summary_line {
                    ($level:ident) => {
                        tracing::$level!(
                            elapsed_ms = started.elapsed().as_millis() as u64,
                            metas = summary.metas,
                            backed = summary.backed,
                            young = summary.young,
                            owed_seen = summary.owed_seen,
                            audited = summary.audited,
                            audit_deferred = summary.audit_deferred,
                            already_stored = summary.already_stored,
                            resent = summary.resent,
                            resend_failed = summary.resend_failed,
                            probe_failed = summary.probe_failed,
                            value_gone = summary.value_gone,
                            budget_busy = summary.budget_busy,
                            oversized = summary.oversized,
                            confirmed_copy_missing = summary.confirmed_copy_missing,
                            owed = summary.owed,
                            work_budget_spent = summary.work_budget_spent,
                            "external-value backup repair pass complete"
                        )
                    };
                }
                if summary.eventful() {
                    summary_line!(info);
                } else {
                    summary_line!(debug);
                }
            }
            Err(PassError::Cancelled) => tracing::debug!(
                elapsed_ms = started.elapsed().as_millis() as u64,
                "external-value backup repair pass cancelled; the owed count is not published"
            ),
            Err(e) => tracing::warn!(
                elapsed_ms = started.elapsed().as_millis() as u64,
                error = %e,
                "external-value backup repair pass did not complete its walk; the owed count is not \
                 published (what it repaired stays repaired)"
            ),
        }
    }
}

/// Spawn the backup repair as a periodic task: a pass every
/// `repair.timing().interval`, the first at once, each run only while this
/// replica holds `le:gc:leader` (taken with `leader`, the process-wide
/// identity the GC, the orphan sweep and the reconciler share). A non-leader
/// publishes 0 on the owed gauge; a failed lock probe publishes nothing.
/// Exits on `cancel`, abandoning a pass in progress; every step a pass
/// completed stays done, and the next leader's pass resumes the work.
pub fn spawn_ext_backup_repair(
    repair: ExtBackupRepair,
    leader: LeaderIdentity,
    cancel: CancellationToken,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        tracing::info!(
            interval_secs = repair.timing.interval.as_secs(),
            age_gate_secs = repair.timing.age_gate.as_secs(),
            upload_ceiling_secs = repair.timing.upload_ceiling.as_secs(),
            work_budget_secs = repair.timing.work_budget.as_secs(),
            backup_store_id = repair.store_id(),
            replica_id = %leader.service_label,
            lock_value = %leader.gc_lock_value(),
            "external-value backup repair started"
        );
        let replica_id = leader.service_label.as_str();
        loop {
            match try_acquire_gc_leader(
                &repair.client,
                &repair.keys,
                &leader,
                repair.metrics.as_ref(),
            )
            .await
            {
                Ok(true) => repair.run_one_pass(replica_id, &cancel).await,
                Ok(false) => {
                    if let Some(m) = repair.metrics.as_ref() {
                        m.record_checkpoint_ext_backup_owed(0, replica_id);
                    }
                    tracing::debug!(
                        replica_id = %leader.service_label,
                        lock_value = %leader.gc_lock_value(),
                        "external-value backup repair skipped: another replica holds the leader \
                         lock"
                    );
                }
                Err(e) => tracing::warn!(
                    replica_id = %leader.service_label,
                    lock_value = %leader.gc_lock_value(),
                    error = %e,
                    "external-value backup repair leader lock acquisition failed"
                ),
            }
            tokio::select! {
                () = tokio::time::sleep(repair.timing.interval) => {}
                () = cancel.cancelled() => {
                    tracing::info!("external-value backup repair shutting down");
                    return;
                }
            }
        }
    })
}

/// `d` in whole milliseconds, saturating.
fn millis(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

/// The first 16 hex characters of a hash, for log lines.
fn prefix(hash_hex: &str) -> &str {
    &hash_hex[..hash_hex.len().min(16)]
}

#[cfg(test)]
mod tests {
    //! The pure parts of the repair: [`classify`] and the production timing.
    //! The pass itself runs against Valkey and a recording backup store in
    //! `tests/integration/checkpoint/ext_backup_repair_test.rs`.
    use super::*;

    const STORE: &str = "0123456789abcdef";
    const OTHER_STORE: &str = "fedcba9876543210";
    /// The pass's clock, in Valkey milliseconds.
    const NOW_MS: u64 = 10_000_000_000;
    /// `now - G`, with G = 60 s.
    const YOUNG_CUTOFF_MS: u64 = NOW_MS - 60_000;
    /// `now - audit age`, with the audit age = 24 h.
    const AUDIT_CUTOFF_MS: u64 = NOW_MS - 86_400_000;
    /// A store stamp older than the age gate.
    const OLD_STAMP_MS: u64 = YOUNG_CUTOFF_MS - 1;

    fn marks(backed_up_in: Option<&str>, last_store_ms: Option<&str>) -> BackupMarks {
        BackupMarks {
            backed_up_in: backed_up_in.map(str::to_owned),
            last_store_ms: last_store_ms.map(str::to_owned),
        }
    }

    fn class_of(backed_up_in: Option<&str>, last_store_ms: Option<&str>) -> MetaClass {
        classify(
            &marks(backed_up_in, last_store_ms),
            STORE,
            YOUNG_CUTOFF_MS,
            AUDIT_CUTOFF_MS,
        )
    }

    fn stamp(ms: u64) -> String {
        ms.to_string()
    }

    fn confirmed(store_id: &str, ms: u64) -> String {
        format!("{store_id}:{ms}")
    }

    /// T5: a value confirmed under this store's identity within the audit
    /// age is backed, whatever its stamp says: the mirror's own fresh
    /// confirmation of a young value, an old value's, and one confirmed at
    /// exactly the audit cutoff.
    #[test]
    fn classify_a_recent_confirmation_under_this_store_is_backed_whatever_the_stamp() {
        let recent = confirmed(STORE, NOW_MS - 1_000);
        assert_eq!(
            class_of(Some(&recent), Some(&stamp(NOW_MS))),
            MetaClass::Backed
        );
        assert_eq!(
            class_of(Some(&recent), Some(&stamp(OLD_STAMP_MS))),
            MetaClass::Backed
        );
        assert_eq!(class_of(Some(&recent), None), MetaClass::Backed);
        assert_eq!(
            class_of(
                Some(&confirmed(STORE, AUDIT_CUTOFF_MS)),
                Some(&stamp(OLD_STAMP_MS))
            ),
            MetaClass::Backed,
            "a confirmation exactly at the audit cutoff is still recent"
        );
    }

    /// T7: an unconfirmed value stored within the age gate is young (the
    /// mirror is still on it), including one stored exactly at the cutoff;
    /// one stored before it is owed.
    #[test]
    fn classify_the_age_gate_leaves_a_young_value_to_the_mirror_and_owes_an_old_one() {
        assert_eq!(class_of(None, Some(&stamp(NOW_MS))), MetaClass::Young);
        assert_eq!(
            class_of(None, Some(&stamp(YOUNG_CUTOFF_MS))),
            MetaClass::Young,
            "a stamp exactly at the cutoff is young"
        );
        assert_eq!(class_of(None, Some(&stamp(OLD_STAMP_MS))), MetaClass::Owed);
    }

    /// T6: a confirmation under another store's identity is no confirmation:
    /// an old value is owed, a young one young.
    #[test]
    fn classify_a_confirmation_under_another_store_counts_as_none() {
        let foreign = confirmed(OTHER_STORE, NOW_MS - 1_000);
        assert_eq!(
            class_of(Some(&foreign), Some(&stamp(OLD_STAMP_MS))),
            MetaClass::Owed
        );
        assert_eq!(class_of(Some(&foreign), None), MetaClass::Owed);
        assert_eq!(
            class_of(Some(&foreign), Some(&stamp(NOW_MS))),
            MetaClass::Young
        );
    }

    /// T15: a meta with neither field (a recount's, or an older build's), a
    /// stamp that does not parse, and every garbled confirmation are owed:
    /// acting on one costs an existence check or an upload, never a delete.
    #[test]
    fn classify_a_missing_or_garbled_field_is_owed() {
        assert_eq!(class_of(None, None), MetaClass::Owed, "no fields at all");
        assert_eq!(
            class_of(None, Some("not-a-number")),
            MetaClass::Owed,
            "an unparseable stamp is not young"
        );
        for garbled in [
            "",
            "garbage",
            ":123",
            "0123456789abcdef:",
            "0123456789abcdef:12x",
            "0123456789abcdef:-5",
            "0123456789abcdef",
        ] {
            assert_eq!(
                class_of(Some(garbled), Some(&stamp(OLD_STAMP_MS))),
                MetaClass::Owed,
                "{garbled:?} is not a confirmation"
            );
            assert_eq!(
                class_of(Some(garbled), None),
                MetaClass::Owed,
                "{garbled:?} without a stamp"
            );
        }
    }

    /// T16: a confirmation under this store's identity older than the audit
    /// age is audited, unless the value was stored within the age gate, in
    /// which case it is young: the mirror's fresh copy will confirm it again.
    #[test]
    fn classify_an_old_confirmation_under_this_store_is_audited_unless_the_value_is_young() {
        let old = confirmed(STORE, AUDIT_CUTOFF_MS - 1);
        assert_eq!(
            class_of(Some(&old), Some(&stamp(OLD_STAMP_MS))),
            MetaClass::Audit
        );
        assert_eq!(
            class_of(Some(&old), None),
            MetaClass::Audit,
            "a stampless meta confirmed long ago"
        );
        assert_eq!(class_of(Some(&old), Some(&stamp(NOW_MS))), MetaClass::Young);
    }

    /// The age gate is the adapter call ceiling, floored at 60 s, so an
    /// upload the mirror may still have in flight is not repeated. T23: the
    /// upload ceiling is that same `G`, so a re-send's upload gets the time
    /// the design allows one upload, not the local adapters' 5 s call
    /// ceiling, which bounds a rename. The rest of the timing is the fixed
    /// production constants.
    #[test]
    fn production_timing_floors_the_age_gate_at_sixty_seconds_and_uses_the_constants() {
        let local = ExtBackupRepairTiming::production(Duration::from_secs(10));
        assert_eq!(local.age_gate, Duration::from_secs(60));
        let storage_api = ExtBackupRepairTiming::production(Duration::from_secs(180));
        assert_eq!(storage_api.age_gate, Duration::from_secs(180));
        assert_eq!(
            ExtBackupRepairTiming::production(Duration::from_secs(5)).upload_ceiling,
            Duration::from_secs(60),
            "the local adapters' 5 s call ceiling is floored at 60 s"
        );
        assert_eq!(
            storage_api.upload_ceiling,
            Duration::from_secs(180),
            "StorageApi's 180 s call ceiling stands"
        );
        assert_eq!(
            local,
            ExtBackupRepairTiming {
                interval: Duration::from_secs(300),
                age_gate: Duration::from_secs(60),
                audit_age: Duration::from_secs(86_400),
                audit_per_pass: 2_000,
                reserve_wait: Duration::from_secs(60),
                reserve_retry: Duration::from_secs(1),
                upload_ceiling: Duration::from_secs(60),
                work_budget: Duration::from_secs(240),
            }
        );
    }
}
