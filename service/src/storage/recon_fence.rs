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

//! The checkpoint clean-up's delete fence on the `{recon}` slot, and the
//! Valkey reads it makes (b.g65, SDD §22.17.3, §22.17.5).
//!
//! The checkpoint collect deletes the backup-store copy of an external value
//! whose tombstone (`le:ckpt:{recon}:tomb`) has aged and that no retained
//! backup's ext-refs sidecar names. Every such delete is licensed here, in
//! Valkey, rather than by holding a lock that keeps writers out:
//!
//! - **The writer re-arm** ([`recon_ext_rearm`], `scripts/recon_ext_rearm.lua`).
//!   Every backup writer — each layer publish, the backup repair's re-send,
//!   the ext-mirror worker's upload — runs it immediately before its adapter
//!   PUT. It records on `le:ckpt:{recon}:rearm` that the tombstoned hashes the
//!   writer names are referenced again, answers which of them the collect
//!   deleted inside the writer's reliance window (`le:ckpt:{recon}:collected`),
//!   and settles the writer's own stem against the orphan and deletion lists.
//!   With no hash and no stem it is the collect's snapshot clock
//!   ([`recon_clock_ms`]).
//! - **The delete-time fence** ([`recon_ext_commit`],
//!   `scripts/recon_ext_commit.lua`): licenses a delete only when the hash's
//!   tombstone and latest re-arm both predate the collect's snapshot
//!   threshold and the run is at most one grace past the collect's snapshot
//!   clock, and records the license on `:collected` before the DELETE.
//! - **The settle** ([`recon_ext_settle`], `scripts/recon_ext_settle.lua`):
//!   retires and dates one batch's outcome and reports the kept hashes an
//!   earlier delete was never settled for.
//! - **The deleters' record** ([`recon_tomb_record`],
//!   `scripts/recon_tomb_record.lua`): a base generation's hashes tombstoned
//!   and its stem announced on `le:ckpt:{recon}:deleting` in one run, before
//!   any of its artifacts is deleted.
//! - **The orphan commit** ([`recon_orphan_commit`],
//!   `scripts/recon_orphan_commit.lua`): licenses the reap of a claimed orphan
//!   stem only while no publish has healed it, announcing it in the same run.
//! - **The re-date** ([`recon_redate`], `scripts/recon_redate.lua`): moves a
//!   deferred member of a work list later (`ZADD XX GT`), so the oldest-first
//!   claim reaches the members behind it.
//! - **Two reads**: whether a value is live in Valkey
//!   ([`ext_values_live`], `EXISTS le:ext:meta:{hash}`), and which stems are
//!   announced on `:deleting` ([`announced_stems`]).
//! - **The withdrawal** ([`withdraw_object_announcements`]: a `ZSCAN` of
//!   `:deleting` for the object's stems, then a plain `ZREM` of them): the
//!   dead-object reap removes every announcement of an object before it
//!   retires that object as restored, and retires nothing when it fails.
//!
//! **Pools** (SDD §22.8): the writer re-arm runs on the data pool, beside the
//! writer's own data-pool work; everything the clean-up itself sends — the
//! clock read, the fence, the settle, the orphan commit, the re-date, the
//! two reads and the withdrawal — runs on the maintenance pool. The deleters' record runs on the
//! maintenance pool, as the register-only tombstone record it replaces did.
//!
//! Every helper counts a command timeout and notes a store write refused
//! `-OOM` where it receives the error ([`FredErrOrigin::observe_error`]);
//! callers only log the error they get back. Every script answers a refusal
//! during a slot migration as an error, which every caller treats as "do not
//! act" (a writer defers its PUT, the clean-up deletes nothing).

use std::sync::LazyLock;
use std::time::Duration;

use fred::interfaces::{KeysInterface, SortedSetsInterface};
use fred::types::scan::Scanner;
use fred::types::scripts::Script;
use fred::types::Value;
use futures::StreamExt;

use crate::storage::client::{
    FredErrOrigin, ValkeyClient, POOL_LABEL_DATA, POOL_LABEL_MAINTENANCE,
};
use crate::storage::keys::KeySchema;
use crate::storage::orphan_sweep::complete_pages;

/// `scripts/recon_ext_rearm.lua`: a writer's re-arm. The script header
/// carries the contract.
pub static RECON_EXT_REARM_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/recon_ext_rearm.lua")));

/// `scripts/recon_ext_commit.lua`: the collect's delete-time fence.
pub static RECON_EXT_COMMIT_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/recon_ext_commit.lua")));

/// `scripts/recon_ext_settle.lua`: one collect batch's settle.
pub static RECON_EXT_SETTLE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/recon_ext_settle.lua")));

/// `scripts/recon_orphan_commit.lua`: the orphan reap's license.
pub static RECON_ORPHAN_COMMIT_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/recon_orphan_commit.lua")));

/// `scripts/recon_tomb_record.lua`: a deleter's tombstone record and stem
/// announcement.
pub static RECON_TOMB_RECORD_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/recon_tomb_record.lua")));

/// `scripts/recon_redate.lua`: a work list's deferred members moved later.
pub static RECON_REDATE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/recon_redate.lua")));

/// How long `:collected` keeps an entry, and the `PEXPIRE` of `:rearm`,
/// `:collected` and `:deleting`: 24 h. The scripts spell the same number;
/// a writer whose reliance window is longer than this cannot be answered from
/// `:collected` and refuses (SDD §22.17.5, R8).
pub const RECON_FENCE_HORIZON: Duration = Duration::from_secs(24 * 60 * 60);

/// How many members ride on one fence `EVALSHA` or one pipelined read batch:
/// a network-framing bound on one command's argument list, the order of the
/// tombstone record's chunk.
pub const RECON_FENCE_CHUNK: usize = 256;

/// `Duration` → whole milliseconds, saturating.
fn millis(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

/// A reply element the scripts return as an integer.
fn reply_u64(value: &Value, what: &str) -> Result<u64, fred::error::Error> {
    value.as_u64().ok_or_else(|| {
        fred::error::Error::new(
            fred::error::ErrorKind::Parse,
            format!("a {{recon}} fence script answered a non-integer {what}: {value:?}"),
        )
    })
}

/// A reply element the scripts return as a string.
fn reply_string(value: Value, what: &str) -> Result<String, fred::error::Error> {
    value.as_string().ok_or_else(|| {
        fred::error::Error::new(
            fred::error::ErrorKind::Parse,
            format!("a {{recon}} fence script answered a non-string {what}"),
        )
    })
}

/// What one writer's [`recon_ext_rearm`] answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RearmReply {
    /// The `{recon}` clock of the (first) run, in Valkey ms.
    pub now_ms: u64,
    /// The writer's stem is announced on `:deleting` within the grace: a
    /// deleter is removing an earlier copy of it, and nothing was written for
    /// the stem.
    pub stem_refused: bool,
    /// The writer's hashes with a `:collected` entry inside its lookback: the
    /// collect licensed the delete of their backup copy within the writer's
    /// reliance window.
    pub collected: Vec<String>,
}

impl RearmReply {
    /// Whether the writer must not PUT: its stem is being deleted, or a hash
    /// it names was collected within its reliance window.
    pub fn refuses(&self) -> bool {
        self.stem_refused || !self.collected.is_empty()
    }
}

/// One run of `recon_ext_rearm.lua`, on the maintenance pool when
/// `on_maintenance` and on the data pool otherwise.
async fn rearm_once(
    client: &ValkeyClient,
    keys: &KeySchema,
    on_maintenance: bool,
    lookback_ms: u64,
    grace_ms: u64,
    stem: &str,
    hashes: &[String],
) -> Result<(u64, bool, Vec<String>), fred::error::Error> {
    let script_keys = vec![
        keys.checkpoint_reconcile_rearm(),
        keys.checkpoint_reconcile_tombstones(),
        keys.checkpoint_reconcile_collected(),
        keys.checkpoint_reconcile_orphans(),
        keys.checkpoint_reconcile_deleting(),
    ];
    let mut argv: Vec<String> = Vec::with_capacity(hashes.len() + 3);
    argv.push(lookback_ms.to_string());
    argv.push(grace_ms.to_string());
    argv.push(stem.to_owned());
    argv.extend(hashes.iter().cloned());
    let reply: Vec<Value> = if on_maintenance {
        let origin = FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE);
        let (maint_pool, _maint_token) = client.acquire_maintenance();
        RECON_EXT_REARM_SCRIPT
            .evalsha_with_reload::<Vec<Value>, _, _>(maint_pool.next(), script_keys, argv)
            .await
            .inspect_err(|e| origin.observe_error(e))?
    } else {
        let origin = FredErrOrigin::client(client, POOL_LABEL_DATA);
        RECON_EXT_REARM_SCRIPT
            .evalsha_with_reload::<Vec<Value>, _, _>(client.pool.next(), script_keys, argv)
            .await
            .inspect_err(|e| origin.observe_error(e))?
    };
    let mut it = reply.into_iter();
    let (Some(now), Some(refused)) = (it.next(), it.next()) else {
        return Err(fred::error::Error::new(
            fred::error::ErrorKind::Parse,
            "recon_ext_rearm answered fewer than two elements",
        ));
    };
    let now_ms = reply_u64(&now, "clock")?;
    let stem_refused = reply_u64(&refused, "stem verdict")? == 1;
    let collected = it
        .map(|v| reply_string(v, "collected hash"))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((now_ms, stem_refused, collected))
}

/// A backup writer's re-arm, immediately before its adapter PUT, on the data
/// pool: `hashes` are the ext hashes the writer is about to publish a
/// reference to (or upload), `lookback` its reliance window (the time since
/// it started relying on the store's state, plus how late its PUT can land),
/// `grace` the reconciliation grace, and `stem` the writer's stem in the
/// `:orphans` spelling (`None` for the backup repair and the ext-mirror
/// worker, which publish no generation).
///
/// More than [`RECON_FENCE_CHUNK`] hashes go out in several runs; the stem
/// rides on the last, and only when no earlier run already found a collected
/// hash, so a refused publish writes nothing for its stem whichever run
/// refuses it. The writer must not PUT when the reply
/// [`refuses`](RearmReply::refuses), and must treat an error as a refusal
/// (fail closed, SDD §22.17.5 P3) unless its own contract says otherwise.
pub async fn recon_ext_rearm(
    client: &ValkeyClient,
    keys: &KeySchema,
    hashes: &[String],
    lookback: Duration,
    grace: Duration,
    stem: Option<&str>,
) -> Result<RearmReply, fred::error::Error> {
    let lookback_ms = millis(lookback);
    let grace_ms = millis(grace);
    let stem = stem.unwrap_or("");
    let chunks: Vec<&[String]> = if hashes.is_empty() {
        vec![&[] as &[String]]
    } else {
        hashes.chunks(RECON_FENCE_CHUNK).collect()
    };
    let last = chunks.len() - 1;
    let mut reply: Option<RearmReply> = None;
    for (i, chunk) in chunks.into_iter().enumerate() {
        let refused_already = reply.as_ref().is_some_and(RearmReply::refuses);
        let chunk_stem = if i == last && !refused_already {
            stem
        } else {
            ""
        };
        let (now_ms, stem_refused, collected) = rearm_once(
            client,
            keys,
            false,
            lookback_ms,
            grace_ms,
            chunk_stem,
            chunk,
        )
        .await?;
        match reply.as_mut() {
            None => {
                reply = Some(RearmReply {
                    now_ms,
                    stem_refused,
                    collected,
                })
            }
            Some(r) => {
                r.stem_refused |= stem_refused;
                r.collected.extend(collected);
            }
        }
    }
    Ok(reply.expect("at least one run"))
}

/// The `{recon}` clock in Valkey ms: the re-arm script run with no hash and no
/// stem, on the maintenance pool. The collect reads it immediately before its
/// live-set walk; its delete threshold is this minus the grace (SDD §22.17.5,
/// R1). The run writes nothing but the memory-neutral opener.
pub async fn recon_clock_ms(
    client: &ValkeyClient,
    keys: &KeySchema,
) -> Result<u64, fred::error::Error> {
    let (now_ms, _, _) = rearm_once(client, keys, true, 0, 0, "", &[]).await?;
    Ok(now_ms)
}

/// One candidate's verdict from [`recon_ext_commit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitVerdict {
    /// Recorded on `:collected`: the backup copy may be deleted.
    Licensed,
    /// A writer re-armed the hash after the threshold; its tombstone was
    /// re-dated to now. Not deleted.
    Rearmed,
    /// The tombstone is newer than the threshold, or no longer a member, or
    /// the run was more than one grace past the collect's `t0`. Nothing was
    /// written; the claim serves oldest first, so the drain ends.
    TooYoung,
}

/// The collect's delete-time fence over `hashes` (claimed, absent from the
/// live set, not live in Valkey), on the maintenance pool: one verdict per
/// hash, in order. `t0_ms` is the collect's snapshot clock
/// ([`recon_clock_ms`], read just before its live-set walk); the script
/// derives the threshold `t0 − grace` from it, and answers
/// [`CommitVerdict::TooYoung`] for every hash, writing nothing, when its own
/// clock is more than one grace past `t0`.
pub async fn recon_ext_commit(
    client: &ValkeyClient,
    keys: &KeySchema,
    t0_ms: u64,
    grace: Duration,
    hashes: &[String],
) -> Result<Vec<CommitVerdict>, fred::error::Error> {
    let origin = FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE);
    let grace_ms = millis(grace);
    let mut verdicts = Vec::with_capacity(hashes.len());
    for chunk in hashes.chunks(RECON_FENCE_CHUNK) {
        let mut argv: Vec<String> = Vec::with_capacity(chunk.len() + 2);
        argv.push(t0_ms.to_string());
        argv.push(grace_ms.to_string());
        argv.extend(chunk.iter().cloned());
        let reply: Vec<String> = {
            let (maint_pool, _maint_token) = client.acquire_maintenance();
            RECON_EXT_COMMIT_SCRIPT
                .evalsha_with_reload::<Vec<String>, _, _>(
                    maint_pool.next(),
                    vec![
                        keys.checkpoint_reconcile_tombstones(),
                        keys.checkpoint_reconcile_rearm(),
                        keys.checkpoint_reconcile_collected(),
                    ],
                    argv,
                )
                .await
                .inspect_err(|e| origin.observe_error(e))?
        };
        if reply.len() != chunk.len() {
            return Err(fred::error::Error::new(
                fred::error::ErrorKind::Parse,
                format!(
                    "recon_ext_commit answered {} verdicts for {} hashes",
                    reply.len(),
                    chunk.len()
                ),
            ));
        }
        for status in reply {
            verdicts.push(match status.as_str() {
                "licensed" => CommitVerdict::Licensed,
                "rearmed" => CommitVerdict::Rearmed,
                "too_young" => CommitVerdict::TooYoung,
                other => {
                    return Err(fred::error::Error::new(
                        fred::error::ErrorKind::Parse,
                        format!("recon_ext_commit answered an unknown verdict {other:?}"),
                    ))
                }
            });
        }
    }
    Ok(verdicts)
}

/// One collect batch's outcome, by group, for [`recon_ext_settle`]. See
/// `scripts/recon_ext_settle.lua` for what each group's settle writes.
#[derive(Debug, Default, Clone, Copy)]
pub struct SettleGroups<'a> {
    /// In the live set: tombstone and re-arm retired, unless the hash has a
    /// `:collected` entry and is not in [`Self::healed`] — then both stay,
    /// so a later claim serves it again for heal-on-keep.
    pub spared: &'a [String],
    /// Backup copy deleted this batch: tombstone retired unless re-armed
    /// since the threshold.
    pub deleted: &'a [String],
    /// Deleted, and the value was gone from Valkey: `:collected` re-dated to
    /// the settle time.
    pub not_live: &'a [String],
    /// Deleted, and re-put from Valkey: `:collected` entry removed.
    pub healed: &'a [String],
    /// Not deleted: only read, for the heal of an unsettled earlier delete.
    pub kept: &'a [String],
}

/// Settle one collect batch on the maintenance pool. Returns the `kept`
/// hashes that still have a `:collected` entry (an earlier delete of their
/// copy was never settled), which the caller heals from Valkey and settles as
/// `healed` — a spared one as `spared` too, which retires its tombstone. A
/// collect batch is at most one claim (100 members), so one run carries it.
pub async fn recon_ext_settle(
    client: &ValkeyClient,
    keys: &KeySchema,
    threshold_ms: u64,
    groups: SettleGroups<'_>,
) -> Result<Vec<String>, fred::error::Error> {
    let lists = [
        groups.spared,
        groups.deleted,
        groups.not_live,
        groups.healed,
        groups.kept,
    ];
    if lists.iter().all(|l| l.is_empty()) {
        return Ok(Vec::new());
    }
    let mut argv: Vec<String> =
        Vec::with_capacity(6 + lists.iter().map(|l| l.len()).sum::<usize>());
    argv.push(threshold_ms.to_string());
    for list in lists {
        argv.push(list.len().to_string());
    }
    for list in lists {
        argv.extend(list.iter().cloned());
    }
    let origin = FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE);
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    RECON_EXT_SETTLE_SCRIPT
        .evalsha_with_reload::<Vec<String>, _, _>(
            maint_pool.next(),
            vec![
                keys.checkpoint_reconcile_tombstones(),
                keys.checkpoint_reconcile_rearm(),
                keys.checkpoint_reconcile_collected(),
            ],
            argv,
        )
        .await
        .inspect_err(|e| origin.observe_error(e))
}

/// The orphan reap's license for one claimed stem `member`, on the
/// maintenance pool: `Ok(true)` when the member is still on `:orphans` and the
/// stem is now announced on `:deleting`; `Ok(false)` when a publish healed it
/// and nothing may be deleted.
pub async fn recon_orphan_commit(
    client: &ValkeyClient,
    keys: &KeySchema,
    member: &str,
    grace: Duration,
) -> Result<bool, fred::error::Error> {
    let origin = FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE);
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let licensed: i64 = RECON_ORPHAN_COMMIT_SCRIPT
        .evalsha_with_reload::<i64, _, _>(
            maint_pool.next(),
            vec![
                keys.checkpoint_reconcile_orphans(),
                keys.checkpoint_reconcile_deleting(),
            ],
            vec![member.to_owned(), millis(grace).to_string()],
        )
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    Ok(licensed == 1)
}

/// Record `hashes` as tombstoned (`ZADD NX`) and announce `stems` on
/// `:deleting`, on the maintenance pool, before the caller deletes the
/// generations those stems name. More than [`RECON_FENCE_CHUNK`] hashes go
/// out in several runs and the stems ride on the last, so an announcement
/// implies every hash is recorded. A no-op when both are empty.
pub async fn recon_tomb_record(
    client: &ValkeyClient,
    keys: &KeySchema,
    grace: Duration,
    stems: &[String],
    hashes: &[String],
) -> Result<(), fred::error::Error> {
    if stems.is_empty() && hashes.is_empty() {
        return Ok(());
    }
    let origin = FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE);
    let grace_ms = millis(grace).to_string();
    let chunks: Vec<&[String]> = if hashes.is_empty() {
        vec![&[] as &[String]]
    } else {
        hashes.chunks(RECON_FENCE_CHUNK).collect()
    };
    let last = chunks.len() - 1;
    for (i, chunk) in chunks.into_iter().enumerate() {
        let chunk_stems: &[String] = if i == last { stems } else { &[] };
        let mut argv: Vec<String> = Vec::with_capacity(2 + chunk_stems.len() + chunk.len());
        argv.push(grace_ms.clone());
        argv.push(chunk_stems.len().to_string());
        argv.extend(chunk_stems.iter().cloned());
        argv.extend(chunk.iter().cloned());
        let (maint_pool, _maint_token) = client.acquire_maintenance();
        let _: i64 = RECON_TOMB_RECORD_SCRIPT
            .evalsha_with_reload::<i64, _, _>(
                maint_pool.next(),
                vec![
                    keys.checkpoint_reconcile_tombstones(),
                    keys.checkpoint_reconcile_deleting(),
                ],
                argv,
            )
            .await
            .inspect_err(|e| origin.observe_error(e))?;
    }
    Ok(())
}

/// Move `members` of the `{recon}` work list `zset_key` later, to the Valkey
/// clock (`ZADD XX GT`), on the maintenance pool; returns how many moved. A
/// no-op on an empty slice. Callers treat an error as non-fatal: a member left
/// at its old score costs only head-of-line order.
pub async fn recon_redate(
    client: &ValkeyClient,
    zset_key: &str,
    members: &[String],
) -> Result<u64, fred::error::Error> {
    let origin = FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE);
    let mut moved: u64 = 0;
    for chunk in members.chunks(RECON_FENCE_CHUNK) {
        let (maint_pool, _maint_token) = client.acquire_maintenance();
        let n: i64 = RECON_REDATE_SCRIPT
            .evalsha_with_reload::<i64, _, _>(
                maint_pool.next(),
                vec![zset_key.to_owned()],
                chunk.to_vec(),
            )
            .await
            .inspect_err(|e| origin.observe_error(e))?;
        moved += u64::try_from(n).unwrap_or(0);
    }
    Ok(moved)
}

/// Whether Valkey holds each of `hashes` (`EXISTS le:ext:meta:{hash}`), in
/// pipelines of at most [`RECON_FENCE_CHUNK`] on the maintenance pool; one
/// answer per hash, in order. A present meta is a value something still
/// references, which a later backup can name. Outside a DR restore a meta
/// never exists without its data (SDD §11.1); the restore Job's recount
/// creates one with no data for a referenced value whose bytes only the
/// backup store holds (`storage::ext_reconcile`), and this answer then keeps
/// the collect from deleting that last copy.
pub async fn ext_values_live(
    client: &ValkeyClient,
    keys: &KeySchema,
    hashes: &[String],
) -> Result<Vec<bool>, fred::error::Error> {
    let origin = FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE);
    let mut live = Vec::with_capacity(hashes.len());
    for chunk in hashes.chunks(RECON_FENCE_CHUNK) {
        let (maint_pool, _maint_token) = client.acquire_maintenance();
        let pipeline = maint_pool.next().pipeline();
        for hash in chunk {
            let _: () = pipeline
                .exists(keys.ext_meta(hash))
                .await
                .inspect_err(|e| origin.observe_error(e))?;
        }
        let replies: Vec<i64> = pipeline
            .all()
            .await
            .inspect_err(|e| origin.observe_error(e))?;
        if replies.len() != chunk.len() {
            return Err(fred::error::Error::new(
                fred::error::ErrorKind::Parse,
                format!(
                    "an ext:meta EXISTS pipeline answered {} replies for {} keys",
                    replies.len(),
                    chunk.len()
                ),
            ));
        }
        live.extend(replies.into_iter().map(|n| n > 0));
    }
    Ok(live)
}

/// Withdraw every announcement of `object_id` from `le:ckpt:{recon}:deleting`,
/// on the maintenance pool: [`announced_stems_of_object`], then
/// [`withdraw_announced_stems`]; returns how many matching stems it found and
/// sent to `ZREM` (a stem a peer removed between the two is counted). The
/// dead-object reap runs it before it retires a member as spared because its
/// object reads live again: an announcement left on a base the restore keeps
/// would let the collect skip a stale not-found of that base's sidecar (N2).
/// It works from the key, not from a listing of the object's backups, so it
/// withdraws every announced stem of the object whatever a listing says. The
/// caller must not retire the member on an error.
pub async fn withdraw_object_announcements(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
) -> Result<usize, fred::error::Error> {
    let stems = announced_stems_of_object(client, keys, object_id).await?;
    withdraw_announced_stems(client, keys, &stems).await?;
    Ok(stems.len())
}

/// Every member of `le:ckpt:{recon}:deleting` that names `object_id`'s stems:
/// one `ZSCAN … MATCH "<object_id>.*"` on the maintenance pool, walked to its
/// terminal page through [`complete_pages`] (one terminal page: a keyed cursor
/// routes by its key to one node in both modes), then filtered to members
/// whose text before the first `.` is exactly `object_id` in decimal. The glob
/// already keeps `123.4` out of object `12`'s answer because its `.` is
/// literal; the filter keeps it out whatever stem spelling a later build adds.
/// A member that is not UTF-8 names no stem this service writes and is
/// skipped. The key holds announcements younger than two grace windows (its
/// writers trim it), so the walk is a few pages at most.
pub async fn announced_stems_of_object(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
) -> Result<Vec<String>, fred::error::Error> {
    let origin = FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE);
    let key = keys.checkpoint_reconcile_deleting();
    let oid = object_id.to_string();
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let mut pages = Box::pin(complete_pages(
        maint_pool.next().zscan(
            key.as_str(),
            format!("{oid}.*"),
            Some(RECON_FENCE_CHUNK as u32),
        ),
        1,
        &key,
    ));
    let mut stems = Vec::new();
    while let Some(page) = pages.next().await {
        let mut page = page.inspect_err(|e| origin.observe_error(e))?;
        for (member, _score) in page.take_results().unwrap_or_default() {
            let Some(member) = member.as_str() else {
                continue;
            };
            if member.split_once('.').map(|(head, _)| head) == Some(oid.as_str()) {
                stems.push(member.into_owned());
            }
        }
    }
    Ok(stems)
}

/// Withdraw `stems` from `le:ckpt:{recon}:deleting` (`ZREM`, on the
/// maintenance pool, at most [`RECON_FENCE_CHUNK`] per command): a plain
/// `ZREM` of named members on one key, which writes nothing for a member or
/// key that is absent. A no-op on an empty slice.
pub async fn withdraw_announced_stems(
    client: &ValkeyClient,
    keys: &KeySchema,
    stems: &[String],
) -> Result<(), fred::error::Error> {
    let origin = FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE);
    let key = keys.checkpoint_reconcile_deleting();
    for chunk in stems.chunks(RECON_FENCE_CHUNK) {
        let (maint_pool, _maint_token) = client.acquire_maintenance();
        let _: i64 = maint_pool
            .next()
            .zrem(key.as_str(), chunk.to_vec())
            .await
            .inspect_err(|e| origin.observe_error(e))?;
    }
    Ok(())
}

/// Which of `stems` are announced on `le:ckpt:{recon}:deleting` (`ZMSCORE`,
/// on the maintenance pool); one answer per stem, in order. The collect asks
/// this for the sidecars its walk listed and then read absent.
pub async fn announced_stems(
    client: &ValkeyClient,
    keys: &KeySchema,
    stems: &[String],
) -> Result<Vec<bool>, fred::error::Error> {
    let origin = FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE);
    let key = keys.checkpoint_reconcile_deleting();
    let mut announced = Vec::with_capacity(stems.len());
    for chunk in stems.chunks(RECON_FENCE_CHUNK) {
        let (maint_pool, _maint_token) = client.acquire_maintenance();
        let scores: Vec<Option<f64>> = maint_pool
            .next()
            .zmscore(key.as_str(), chunk.to_vec())
            .await
            .inspect_err(|e| origin.observe_error(e))?;
        if scores.len() != chunk.len() {
            return Err(fred::error::Error::new(
                fred::error::ErrorKind::Parse,
                format!(
                    "a ZMSCORE of {} stems answered {} scores",
                    chunk.len(),
                    scores.len()
                ),
            ));
        }
        announced.extend(scores.into_iter().map(|s| s.is_some()));
    }
    Ok(announced)
}
