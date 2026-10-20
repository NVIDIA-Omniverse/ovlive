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

//! Atomic presence store operations against one session's `{sid}` slot.
//!
//! Each multi-key presence write is one `EVALSHA` here, with every key the
//! script touches declared in KEYS and all of them carrying the `{sid}` hash
//! tag:
//!
//! - [`register_connection`] — lease, info hash (with `lease_expires_at_ms`
//!   from the store's clock) and membership, publishing `joined` iff the
//!   membership was added;
//! - [`refresh_connection`] — renews the lease only while the connection is
//!   still a member with an intact info hash, and otherwise says why;
//! - [`unregister_connection`] — arms the teardown marker
//!   ([`KeySchema::presence_gone`]) and removes the connection, publishing
//!   `left` iff the membership was removed;
//! - [`sweep_member_batch`] — for one chunk of at most [`SWEEP_MEMBER_BATCH`]
//!   members, removes each whose lease has expired and which is still a
//!   member, deleting its info hash and publishing its `left`, and reports
//!   each removal's delay past `lease_expires_at_ms`. It only checks leases,
//!   never writes one, and does not touch the teardown marker.
//!
//! A register or refresh that reaches the store after its connection's
//! teardown (a client-side timeout that still ran, or a fred re-send after a
//! reconnect) writes and publishes nothing, however late it arrives. The
//! fence has two parts. For one lease TTL the teardown marker refuses both.
//! After that, a register is refused by its own deadline, `not_after_ms`, a
//! point on the store's clock that the caller derives from [`store_time_ms`]
//! read before registering, plus one lease TTL. The caller must take that read
//! while the connection's teardown has not yet started (see
//! [`register_connection`]); then the read precedes teardown and the deadline
//! falls before the marker expires. With no register able to recreate the
//! member, a late refresh finds it gone and writes nothing.
//!
//! Membership changes and their announcements share one script execution, so
//! a lost reply can never separate them, and every `joined` / `left` of a
//! session is emitted by the node owning `{sid}` in the order the membership
//! changed. The event payload is built by the caller
//! (`crate::presence::PresenceEvent::encode`) and passed through as an opaque
//! string: `storage` has no production dependency on `crate::presence`.
//!
//! The fleet-wide session index (`le:presence:active_sessions`) is untagged and
//! lives in another slot, so no script touches it. A session is added to it by
//! [`add_session_to_index`], and removed through the one shared
//! remove-then-recheck helper, [`remove_session_from_index`]. The orphan
//! sweep's index backstop (`storage::orphan_sweep`) also adds back any session
//! whose member set it finds un-indexed.
//!
//! [`scan_session_index_page`] and [`scan_member_page`] read one `SSCAN` page of
//! the session index and of a member set, for the presence sweeper's walk.
//!
//! [`read_member_infos`] reads one page of connections' info hashes, at most
//! [`REPLAY_PAGE`] of them, in one pipeline of keyed `HGETALL`s on the `{sid}`
//! slot; a caller with a longer member list splits it into pages.
//!
//! # Error counting
//!
//! Every function here takes a [`FredErrOrigin`] and, when a data-pool command
//! times out, counts it on that origin once, where the error is received,
//! before returning the error. A caller that hands the error on to `map_fred_err_for_grpc` —
//! which counts on its own — passes an origin that counts nothing
//! (`FredErrOrigin::metrics(None, POOL_LABEL_DATA, <the client's store
//! out-of-memory signal>)`), so the timeout is still counted once. The same
//! origin notes a store write refused `-OOM` on that signal (b.pa1).

use std::sync::LazyLock;
use std::time::Duration;

use fred::interfaces::{ClientLike, HashesInterface, SetsInterface};
use fred::types::config::Options;
use fred::types::scripts::Script;
use fred::types::{ClusterHash, CustomCommand, Value};

use crate::storage::client::{FredErrOrigin, ValkeyClient};
use crate::storage::keys::KeySchema;

/// `presence_register.lua`; sent by [`register_connection`].
static PRESENCE_REGISTER_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/presence_register.lua")));

/// `presence_refresh.lua`; sent by [`refresh_connection`].
static PRESENCE_REFRESH_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/presence_refresh.lua")));

/// `presence_unregister.lua`; sent by [`unregister_connection`].
static PRESENCE_UNREGISTER_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/presence_unregister.lua")));

/// `presence_sweep_batch.lua`; sent by [`sweep_member_batch`].
static PRESENCE_SWEEP_BATCH_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/presence_sweep_batch.lua")));

/// The most connection ids one [`read_member_infos`] call reads: one page of a
/// membership replay (SDD SR-1.6). A constant, not an operator setting.
pub const REPLAY_PAGE: usize = 100;

/// The `SSCAN COUNT` hint the presence sweeper uses when it pages through the
/// session index (`le:presence:active_sessions`) (SDD SR-1.6). A constant, not
/// an operator setting.
pub const SWEEP_SESSION_PAGE: usize = 100;

/// The `SSCAN COUNT` hint the presence sweeper uses when it pages through one
/// session's member set (SDD SR-1.6). `COUNT` is only a hint, so a page can
/// hold more ids than this; the sweeper splits each page into chunks of at most
/// [`SWEEP_MEMBER_BATCH`]. A constant, not an operator setting.
pub const SWEEP_MEMBER_PAGE: usize = 100;

/// The most connection ids one [`sweep_member_batch`] call takes, so at most
/// `1 + 2 × 100` KEYS per script call (SDD SR-1.6). A constant, not an operator
/// setting.
pub const SWEEP_MEMBER_BATCH: usize = 100;

/// What [`register_connection`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterOutcome {
    /// The lease, info hash and membership were written. `added` is true iff
    /// the membership was newly added, which is exactly when `joined` was
    /// published; false means the connection was already a member (an info
    /// repair or a duplicate register), and nothing was published.
    Registered { added: bool },
    /// The connection's teardown marker exists: nothing was written and
    /// nothing was published. The connection has been torn down and must not
    /// be registered again.
    Gone,
    /// The store's clock (on the `{sid}` node) had passed the caller's
    /// `not_after_ms` when the script ran: nothing was written and nothing was
    /// published. The register reached the store too late to be trusted as
    /// coming from a live connection; a caller that still wants the connection
    /// registered reads a fresh [`store_time_ms`] and tries again.
    Expired,
}

/// What [`refresh_connection`] did. Every outcome but [`Self::Renewed`] wrote
/// nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefreshOutcome {
    /// The lease and `lease_expires_at_ms` were renewed.
    Renewed,
    /// The connection is not in the session's member set (it was evicted).
    Evicted,
    /// The connection is a member but its info hash has no `user_name`. The
    /// refresh never creates the info hash; re-registering repairs it.
    InfoMissing,
    /// The connection's teardown marker exists; it has been torn down.
    Gone,
}

/// What [`unregister_connection`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unregistered {
    /// True iff this call removed the membership, which is exactly when `left`
    /// was published. False on a repeated call, or after a sweep already
    /// removed (and announced) the connection.
    pub removed: bool,
    /// Members left in the session's member set after the removal. At 0 the
    /// caller runs [`remove_session_from_index`].
    pub remaining: u64,
}

/// What one [`sweep_member_batch`] call did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SweepBatch {
    /// The chunk's connection ids this call removed, in chunk order, each with
    /// its removal delay: how long after its info hash's `lease_expires_at_ms`
    /// the store removed it, on the store's clock, in whole milliseconds and
    /// never negative (a lease deleted before its recorded expiry reads as
    /// zero). `None` when the info hash's `lease_expires_at_ms` was absent (one
    /// written before the field existed, or a test seed), or negative or
    /// non-numeric (only corrupted data). Every id here had `left` published by
    /// this call, and no other id did. Removals made by an execution whose reply
    /// was lost and re-sent by fred are not reported; their `left` was already
    /// published.
    pub removed: Vec<(String, Option<Duration>)>,
    /// Members left in the session's member set after the removals. At 0, once
    /// the session's member iteration ends, the caller runs
    /// [`remove_session_from_index`].
    pub remaining: u64,
}

/// Register a connection's presence in one atomic step
/// (`scripts/presence_register.lua`).
///
/// Checked in order, each writing and publishing nothing: the teardown marker
/// exists ([`RegisterOutcome::Gone`]); the store's `TIME` is past
/// `not_after_ms` ([`RegisterOutcome::Expired`]). Otherwise sets the lease to
/// `lease_ttl_secs`, writes the info hash (`user_name`, `user_application`, and
/// `lease_expires_at_ms` from the same `TIME` sample), adds the connection to
/// the member set, and publishes `joined_payload` on
/// [`KeySchema::presence_events_channel`] iff the add was new. Serves the
/// initial registration, a self-rejoin after eviction, and an info-hash repair.
///
/// `not_after_ms` is the register deadline on the store's clock: a
/// [`store_time_ms`] read for this session taken immediately before the call,
/// plus `lease_ttl_secs × 1000`.
///
/// **Caller obligation:** the store-time read must be taken while the
/// connection's teardown has not yet started. Only then does the read execute
/// before teardown's unregister, so the deadline (read + TTL) falls before the
/// teardown marker expires (unregister + TTL). That is what lets the two
/// refusals together fence a register however late it reaches the store. If
/// teardown ran first, the marker expires before the deadline, and a register
/// landing between the two would recreate the member and info hash. A caller
/// that can race its own teardown, such as a lease keeper re-registering while
/// its connection is being torn down, must confirm after the read that teardown
/// had not begun (check its teardown signal, read the time, check the signal
/// again, then register) and must not register if it had.
///
/// `joined_payload` is the caller-encoded `joined` event. `lease_ttl_secs` is
/// the configured lease TTL and must be positive; the script rejects anything
/// else before writing.
///
/// Runs on the data pool. A command timeout is counted once on `origin`.
// Every argument is a distinct input the script needs; grouping the strings
// into a struct would only rename them at the one call site that builds them.
#[allow(clippy::too_many_arguments)]
pub async fn register_connection(
    client: &ValkeyClient,
    origin: FredErrOrigin<'_>,
    keys: &KeySchema,
    session_id: &str,
    connection_id: &str,
    user_name: &str,
    user_application: &str,
    lease_ttl_secs: u64,
    not_after_ms: u64,
    joined_payload: &str,
) -> Result<RegisterOutcome, fred::error::Error> {
    let script_keys = connection_keys(keys, session_id, connection_id);
    let script_argv = vec![
        connection_id.to_owned(),
        user_name.to_owned(),
        user_application.to_owned(),
        lease_ttl_secs.to_string(),
        keys.presence_events_channel(session_id),
        joined_payload.to_owned(),
        not_after_ms.to_string(),
    ];
    // b.zt3 re-send audit: fred keeps its default write attempts here, so a lost
    // connection can run this twice: safe, its late-write fence covers a register
    // whose reply was lost (script header).
    let (status, added): (i64, i64) = PRESENCE_REGISTER_SCRIPT
        .evalsha_with_reload::<(i64, i64), _, _>(client.pool.next(), script_keys, script_argv)
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    match (status, added) {
        (0, _) => Ok(RegisterOutcome::Gone),
        (3, _) => Ok(RegisterOutcome::Expired),
        (1, 0) => Ok(RegisterOutcome::Registered { added: false }),
        (1, 1) => Ok(RegisterOutcome::Registered { added: true }),
        _ => Err(unexpected_reply(
            "presence_register.lua",
            format!("{{{status}, {added}}}"),
        )),
    }
}

/// Read the store's clock, in milliseconds, on the node that owns `session_id`'s
/// `{sid}` slot.
///
/// This is the base of [`register_connection`]'s `not_after_ms` deadline, which
/// the register script compares against its own `TIME` on that same node — so
/// the comparison involves one clock, not this process's and not another
/// node's. `TIME` is keyless; left to fred's default routing it would go to an
/// arbitrary primary, so it is pinned to the slot of
/// [`KeySchema::presence_members`] with a per-command `cluster_hash` (a no-op on
/// a standalone store).
///
/// Runs on the data pool. A command timeout is counted once on `origin`; a
/// reply that is not two integers is a `Protocol` error.
pub async fn store_time_ms(
    client: &ValkeyClient,
    origin: FredErrOrigin<'_>,
    keys: &KeySchema,
    session_id: &str,
) -> Result<u64, fred::error::Error> {
    let pin = Options {
        cluster_hash: Some(ClusterHash::from(
            keys.presence_members(session_id).as_str(),
        )),
        ..Default::default()
    };
    let reply: Vec<String> = client
        .pool
        .next()
        .with_options(&pin)
        .custom(
            CustomCommand::new_static("TIME", ClusterHash::FirstKey, false),
            Vec::<Value>::new(),
        )
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    let secs = reply.first().and_then(|s| s.parse::<u64>().ok());
    let usecs = reply.get(1).and_then(|s| s.parse::<u64>().ok());
    match (secs, usecs, reply.len()) {
        (Some(s), Some(us), 2) => Ok(s.saturating_mul(1000).saturating_add(us / 1000)),
        _ => Err(unexpected_reply("TIME", format!("{reply:?}"))),
    }
}

/// Renew a connection's lease iff it is still fully registered
/// (`scripts/presence_refresh.lua`).
///
/// Checked in order, each writing nothing: the teardown marker exists
/// ([`RefreshOutcome::Gone`]); the connection is not a member
/// ([`RefreshOutcome::Evicted`]); its info hash has no `user_name`
/// ([`RefreshOutcome::InfoMissing`]). Otherwise the lease is set to
/// `lease_ttl_secs` and `lease_expires_at_ms` rewritten from the store's `TIME`
/// ([`RefreshOutcome::Renewed`]). Never creates the info hash, the membership
/// or a lease for a connection that has neither.
///
/// `lease_ttl_secs` must be positive; the script rejects anything else before
/// reading.
///
/// Runs on the data pool. A command timeout is counted once on `origin`.
pub async fn refresh_connection(
    client: &ValkeyClient,
    origin: FredErrOrigin<'_>,
    keys: &KeySchema,
    session_id: &str,
    connection_id: &str,
    lease_ttl_secs: u64,
) -> Result<RefreshOutcome, fred::error::Error> {
    let script_keys = connection_keys(keys, session_id, connection_id);
    let script_argv = vec![connection_id.to_owned(), lease_ttl_secs.to_string()];
    // b.zt3 re-send audit: fred keeps its default write attempts here, so a lost
    // connection can run this twice: a second renewal of the same lease, harmless
    // (script header).
    let code: i64 = PRESENCE_REFRESH_SCRIPT
        .evalsha_with_reload::<i64, _, _>(client.pool.next(), script_keys, script_argv)
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    match code {
        1 => Ok(RefreshOutcome::Renewed),
        0 => Ok(RefreshOutcome::Evicted),
        3 => Ok(RefreshOutcome::InfoMissing),
        2 => Ok(RefreshOutcome::Gone),
        other => Err(unexpected_reply("presence_refresh.lua", other.to_string())),
    }
}

/// Tear a connection out of its session's presence state in one atomic step
/// (`scripts/presence_unregister.lua`).
///
/// Arms the teardown marker for `lease_ttl_secs` (the configured lease TTL),
/// removes the connection from the member set, deletes its info hash and lease,
/// and publishes `left_payload` on [`KeySchema::presence_events_channel`] iff
/// the removal happened here. Idempotent: a repeated call re-arms the marker and
/// returns `removed: false` without publishing.
///
/// `left_payload` is the caller-encoded `left` event. `lease_ttl_secs` must be
/// positive; the script rejects anything else before writing.
///
/// Does not touch the session index; when [`Unregistered::remaining`] is 0 the
/// caller runs [`remove_session_from_index`].
///
/// Runs on the data pool. A command timeout is counted once on `origin`.
pub async fn unregister_connection(
    client: &ValkeyClient,
    origin: FredErrOrigin<'_>,
    keys: &KeySchema,
    session_id: &str,
    connection_id: &str,
    lease_ttl_secs: u64,
    left_payload: &str,
) -> Result<Unregistered, fred::error::Error> {
    let script_keys = connection_keys(keys, session_id, connection_id);
    let script_argv = vec![
        connection_id.to_owned(),
        lease_ttl_secs.to_string(),
        keys.presence_events_channel(session_id),
        left_payload.to_owned(),
    ];
    // b.zt3 re-send audit: fred keeps its default write attempts here, so a lost
    // connection can run this twice: idempotent (script header).
    let (removed, remaining): (i64, i64) = PRESENCE_UNREGISTER_SCRIPT
        .evalsha_with_reload::<(i64, i64), _, _>(client.pool.next(), script_keys, script_argv)
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    removal_reply("presence_unregister.lua", removed, remaining)
        .map(|(removed, remaining)| Unregistered { removed, remaining })
}

/// Remove the stale members of one chunk of a session's member set, and
/// announce each removal, in one atomic step (`scripts/presence_sweep_batch.lua`).
///
/// For each `cids[i]`, the script removes it only if its lease
/// ([`KeySchema::presence_lease`]) is absent **and** it is still in the member
/// set. Each removal deletes the connection's info hash
/// ([`KeySchema::presence_user`]) and publishes `left_payloads[i]` on
/// [`KeySchema::presence_events_channel`] in the same execution. A cid with a
/// live lease, or one that is no longer a member (removed by another
/// replica's sweep, by its teardown, or by an earlier call for a duplicate
/// `SSCAN` entry), is left untouched, unreported and unannounced. So each
/// removal is announced exactly once, by the call that made it.
///
/// Each removal's delay is `now − lease_expires_at_ms`, both on the store's
/// clock of the node owning `{sid}` (see [`SweepBatch::removed`]). Leases are
/// checked, never written. The teardown marker ([`KeySchema::presence_gone`])
/// and the session index are neither read nor written; when
/// [`SweepBatch::remaining`] is 0 the caller runs [`remove_session_from_index`].
///
/// `left_payloads[i]` is the caller-encoded `left` event for `cids[i]`. An
/// empty `cids`, more than [`SWEEP_MEMBER_BATCH`] of them, or a
/// `left_payloads` of a different length is a caller bug: it fails with
/// `InvalidArgument` before any command is sent. A reply outside the script's
/// contract, including a removed id that is not in `cids`, is a `Protocol`
/// error.
///
/// One `EVALSHA` on the data pool. A command timeout is counted once on
/// `origin`.
pub async fn sweep_member_batch(
    client: &ValkeyClient,
    origin: FredErrOrigin<'_>,
    keys: &KeySchema,
    session_id: &str,
    cids: &[String],
    left_payloads: &[String],
) -> Result<SweepBatch, fred::error::Error> {
    if cids.is_empty() || cids.len() > SWEEP_MEMBER_BATCH {
        return Err(fred::error::Error::new(
            fred::error::ErrorKind::InvalidArgument,
            format!(
                "sweep_member_batch takes 1 to {SWEEP_MEMBER_BATCH} connection ids per call, got {}",
                cids.len()
            ),
        ));
    }
    if left_payloads.len() != cids.len() {
        return Err(fred::error::Error::new(
            fred::error::ErrorKind::InvalidArgument,
            format!(
                "sweep_member_batch needs one left payload per connection id, got {} for {} ids",
                left_payloads.len(),
                cids.len()
            ),
        ));
    }

    let mut script_keys = Vec::with_capacity(1 + 2 * cids.len());
    script_keys.push(keys.presence_members(session_id));
    let mut script_argv = Vec::with_capacity(1 + 2 * cids.len());
    script_argv.push(keys.presence_events_channel(session_id));
    for (cid, left_payload) in cids.iter().zip(left_payloads) {
        script_keys.push(keys.presence_lease(session_id, cid));
        script_keys.push(keys.presence_user(session_id, cid));
        script_argv.push(cid.clone());
        script_argv.push(left_payload.clone());
    }

    // b.zt3 re-send audit: fred keeps its default write attempts here, so a lost
    // connection can run this twice: each removal is re-checked, so nothing is
    // removed or announced twice (script header).
    let reply: Value = PRESENCE_SWEEP_BATCH_SCRIPT
        .evalsha_with_reload::<Value, _, _>(client.pool.next(), script_keys, script_argv)
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    decode_sweep_reply(reply, cids)
}

/// Add `session_id` to the fleet-wide session index
/// (`le:presence:active_sessions`) with one idempotent `SADD`.
///
/// A joining connection calls this before it registers and again after its
/// register completes (the re-add half of [`remove_session_from_index`]'s
/// protocol). Lease keepers call it after every rejoin or info-repair
/// register, and after a renewal at most once per session per replica per
/// refresh interval (`LocalPresenceRouter::index_renewal_due`). So an entry
/// lost by any other route comes back within one refresh interval while the
/// connection whose renewal last re-added the session keeps renewing, and
/// otherwise within two.
///
/// Runs on the data pool. A command timeout is counted once on `origin`.
pub async fn add_session_to_index(
    client: &ValkeyClient,
    origin: FredErrOrigin<'_>,
    keys: &KeySchema,
    session_id: &str,
) -> Result<(), fred::error::Error> {
    let _: i64 = client
        .pool
        .next()
        .sadd(keys.presence_active_sessions(), session_id)
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    Ok(())
}

/// Remove `session_id` from the fleet-wide session index
/// (`le:presence:active_sessions`) unless it still has members, and report
/// whether it is indexed when this returns.
///
/// The index is in a different slot from the session's keys, so "no members →
/// remove from the index" cannot be one atomic step against a concurrent join.
/// This is the remove-then-recheck protocol instead, and the only
/// implementation of it — teardown and the sweeper both call this rather than
/// re-implementing it:
///
/// 1. `SREM` the session from the index;
/// 2. `SCARD` its member set;
/// 3. if that is non-zero, `SADD` the session back.
///
/// Each command completes before the next is issued. With a join that
/// registers and only then re-adds its session to the index, that ordering
/// never leaves a registered member un-indexed, provided the joining replica
/// survives from its register to its re-add: if the join's re-add lands
/// before step 1, its register completed before step 2, which therefore sees
/// the member and re-adds the entry; otherwise the join's own re-add restores
/// it. The argument relies only on each command being atomic on its own key
/// and on each side awaiting completion, so it holds across shards. A joining
/// replica that dies between its register and its re-add can leave the member
/// un-indexed; this helper does not cover that, and the orphan sweep's index
/// backstop (`storage::orphan_sweep`) re-adds the session, within the bound
/// stated on its backstop phase.
///
/// Returns `Ok(true)` if the session was re-added (it still has members) and
/// `Ok(false)` if it was left out of the index. On an error the protocol stops
/// at the failed step, so the session may be left un-indexed while it still
/// has members. A live connection's replica re-adds it on a keeper renewal:
/// within one refresh interval while the connection whose renewal last
/// re-added the session keeps renewing, otherwise within two. A member whose
/// replica died before that re-add is left to the same backstop.
///
/// Runs on the data pool. A command timeout is counted once on `origin`.
pub async fn remove_session_from_index(
    client: &ValkeyClient,
    origin: FredErrOrigin<'_>,
    keys: &KeySchema,
    session_id: &str,
) -> Result<bool, fred::error::Error> {
    let index_key = keys.presence_active_sessions();
    let members_key = keys.presence_members(session_id);

    let _: i64 = client
        .pool
        .next()
        .srem(&index_key, session_id)
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    let members: i64 = client
        .pool
        .next()
        .scard(&members_key)
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    if members == 0 {
        return Ok(false);
    }
    let _: i64 = client
        .pool
        .next()
        .sadd(&index_key, session_id)
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    Ok(true)
}

/// One page of a keyed `SSCAN` ([`scan_session_index_page`],
/// [`scan_member_page`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanPage {
    /// The members this page returned. `SSCAN COUNT` is only a hint: a set
    /// stored compactly (a listpack, up to `set-max-listpack-entries`, 128 by
    /// default) comes back whole in one page whatever the count, and a page of
    /// a hashtable set can hold somewhat more than the count, or nothing at
    /// all. A member present for the whole scan is returned at least once, and
    /// can be returned again on a later page.
    pub ids: Vec<String>,
    /// The cursor to pass for the next page, or `None` when the scan is
    /// complete.
    pub next_cursor: Option<String>,
}

/// Read one page of the session index (`le:presence:active_sessions`) with
/// `SSCAN <cursor> COUNT <count>`. Pass `None` to start a scan and each page's
/// [`ScanPage::next_cursor`] to continue it; nothing is kept between calls.
///
/// One request/response command on the data pool, under the pool's command
/// timeout; a timeout is counted once on `origin`. A `count` of 0 (a syntax
/// error to the store) fails with `InvalidArgument` before anything is sent.
pub async fn scan_session_index_page(
    client: &ValkeyClient,
    origin: FredErrOrigin<'_>,
    keys: &KeySchema,
    cursor: Option<&str>,
    count: usize,
) -> Result<ScanPage, fred::error::Error> {
    sscan_page(
        client,
        origin,
        keys.presence_active_sessions(),
        cursor,
        count,
    )
    .await
}

/// Read one page of `session_id`'s member set ([`KeySchema::presence_members`])
/// with `SSCAN <cursor> COUNT <count>`; see [`scan_session_index_page`] for the
/// cursor, the timeout and `count`. A session whose member set does not exist
/// reads as one empty, final page.
pub async fn scan_member_page(
    client: &ValkeyClient,
    origin: FredErrOrigin<'_>,
    keys: &KeySchema,
    session_id: &str,
    cursor: Option<&str>,
    count: usize,
) -> Result<ScanPage, fred::error::Error> {
    sscan_page(
        client,
        origin,
        keys.presence_members(session_id),
        cursor,
        count,
    )
    .await
}

/// `SSCAN key cursor COUNT count`, as one plain command.
///
/// Sent as a custom command rather than through fred's `sscan` stream, which
/// suits a walk that must keep a single command in flight poorly on two
/// counts. fred applies its command timeout only to request/response
/// commands, so a stalled stream page would wait for the connection's
/// unresponsive check rather than time out. And the stream requests each next
/// page from the previous page's `Drop`, so a page abandoned while a reply is
/// due makes fred walk the rest of the set in the background. A plain command
/// gets the pool's timeout, and an abandoned one is just a reply nobody reads.
/// The key is the command's first argument, so it routes to the key's slot.
async fn sscan_page(
    client: &ValkeyClient,
    origin: FredErrOrigin<'_>,
    key: String,
    cursor: Option<&str>,
    count: usize,
) -> Result<ScanPage, fred::error::Error> {
    if count == 0 {
        return Err(fred::error::Error::new(
            fred::error::ErrorKind::InvalidArgument,
            "SSCAN COUNT must be at least 1",
        ));
    }
    let args = vec![
        key,
        cursor.unwrap_or(SCAN_START).to_owned(),
        "COUNT".to_owned(),
        count.to_string(),
    ];
    let (next, ids): (String, Vec<String>) = client
        .pool
        .next()
        .custom(
            CustomCommand::new_static("SSCAN", ClusterHash::Offset(0), false),
            args,
        )
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    Ok(ScanPage {
        ids,
        next_cursor: (next != SCAN_START).then_some(next),
    })
}

/// The cursor that starts a scan, and that the store returns when a scan is
/// complete.
const SCAN_START: &str = "0";

/// Read the info hashes ([`KeySchema::presence_user`]) of up to
/// [`REPLAY_PAGE`] of `session_id`'s connections in one pipeline, one `HGETALL`
/// per connection id.
///
/// Returns one entry per element of `cids`, in `cids` order:
/// `Some((user_name, user_application))`, or `None` when the hash is empty or
/// malformed. Malformed means the hash has no `user_name`, or a field value that
/// is not a UTF-8 string. A missing `user_application` is not malformed and reads
/// as an empty string: register always writes both fields, and
/// `presence_refresh.lua` counts only a missing `user_name` as incomplete. Each
/// hash is decoded on its own, so a malformed one is `None` at its position and
/// the rest of the page still decodes.
///
/// An empty `cids` returns an empty `Vec` without a round trip. A `cids` longer
/// than [`REPLAY_PAGE`] is a caller bug: it fails with `InvalidArgument` before
/// any command is sent, and the caller does the paging.
///
/// The whole call fails, rather than yielding `None`, when the pipeline fails
/// (fred answers a failure of the whole pipeline, such as a timeout, with one
/// error), when any one `HGETALL` is answered with an error (an info key that
/// is not a hash breaks the key schema, so the page is not guessed around), or
/// when the replies do not pair one-for-one with the commands, or one is not a
/// field/value map (`Protocol` errors; never a truncated or padded result).
///
/// Every key carries the `{sid}` hash tag, so the pipeline goes to the one node
/// owning the slot. It holds keyed commands only: no script, `MULTI` or `SCAN`.
///
/// Runs on the data pool. A command timeout is counted once on `origin`.
pub async fn read_member_infos(
    client: &ValkeyClient,
    origin: FredErrOrigin<'_>,
    keys: &KeySchema,
    session_id: &str,
    cids: &[String],
) -> Result<Vec<Option<(String, String)>>, fred::error::Error> {
    if cids.len() > REPLAY_PAGE {
        return Err(fred::error::Error::new(
            fred::error::ErrorKind::InvalidArgument,
            format!(
                "read_member_infos reads at most {REPLAY_PAGE} connection ids per call, got {}",
                cids.len()
            ),
        ));
    }
    if cids.is_empty() {
        return Ok(Vec::new());
    }

    let pipeline = client.pool.next().pipeline();
    for cid in cids {
        // Queuing only buffers the command; nothing is sent before `try_all`.
        let _: () = pipeline
            .hgetall(keys.presence_user(session_id, cid))
            .await
            .inspect_err(|e| origin.observe_error(e))?;
    }
    // `try_all` yields one `Result` per queued command, in queue order. fred
    // collects a pipeline's replies into an array of per-command frames even
    // when one command is queued, so a one-id page still gets exactly one
    // entry rather than the `HGETALL` reply's own fields. The helper returns
    // at most one error, so a timeout is counted once here.
    decode_member_info_replies(pipeline.try_all::<Value>().await, cids.len())
        .inspect_err(|e| origin.observe_error(e))
}

/// Turn [`read_member_infos`]'s per-command pipeline results into its answer,
/// one entry per command in command order, for `expected` commands.
///
/// The first `Err` entry is returned before the reply count is checked: a
/// failure of the pipeline as a whole (a timeout, a send error) arrives as a
/// single `Err` entry, and must surface as itself, not as a short reply. A
/// count that differs from `expected` is a `Protocol` error, never a truncated
/// or padded result. Each reply is then decoded by [`decode_member_info`].
fn decode_member_info_replies(
    replies: Vec<Result<Value, fred::error::Error>>,
    expected: usize,
) -> Result<Vec<Option<(String, String)>>, fred::error::Error> {
    let replies = replies.into_iter().collect::<Result<Vec<Value>, _>>()?;
    if replies.len() != expected {
        return Err(unexpected_reply(
            "HGETALL presence info pipeline",
            format!("{} replies for {expected} commands", replies.len()),
        ));
    }
    replies.into_iter().map(decode_member_info).collect()
}

/// The four `{sid}`-tagged keys every connection script declares, in the
/// KEYS order the scripts' contracts fix: members, info, lease, gone.
fn connection_keys(keys: &KeySchema, session_id: &str, connection_id: &str) -> Vec<String> {
    vec![
        keys.presence_members(session_id),
        keys.presence_user(session_id, connection_id),
        keys.presence_lease(session_id, connection_id),
        keys.presence_gone(session_id, connection_id),
    ]
}

/// Decode a `{removed, remaining}` reply: `removed` must be 0 or 1 and
/// `remaining` non-negative.
fn removal_reply(
    script: &str,
    removed: i64,
    remaining: i64,
) -> Result<(bool, u64), fred::error::Error> {
    let removed_flag = match removed {
        0 => false,
        1 => true,
        _ => {
            return Err(unexpected_reply(
                script,
                format!("{{{removed}, {remaining}}}"),
            ))
        }
    };
    let remaining_count = u64::try_from(remaining)
        .map_err(|_| unexpected_reply(script, format!("{{{removed}, {remaining}}}")))?;
    Ok((removed_flag, remaining_count))
}

/// Decode `presence_sweep_batch.lua`'s `{removed, remaining}` reply for the
/// chunk `cids`.
///
/// `removed` must be a flat array of `(cid, delay_ms)` pairs, each cid one of
/// `cids` and each delay an integer that is -1 (no usable field → `None`) or
/// non-negative; `remaining` must be a non-negative integer. Anything else is a
/// `Protocol` error, never a guessed result.
fn decode_sweep_reply(reply: Value, cids: &[String]) -> Result<SweepBatch, fred::error::Error> {
    let bad = || unexpected_reply("presence_sweep_batch.lua", format!("{reply:?}"));

    let Value::Array(top) = &reply else {
        return Err(bad());
    };
    let [Value::Array(flat), Value::Integer(remaining)] = top.as_slice() else {
        return Err(bad());
    };
    if flat.len() % 2 != 0 || flat.len() / 2 > cids.len() {
        return Err(bad());
    }
    let remaining = u64::try_from(*remaining).map_err(|_| bad())?;

    let mut removed = Vec::with_capacity(flat.len() / 2);
    for pair in flat.chunks_exact(2) {
        let (Value::String(cid), Value::Integer(delay_ms)) = (&pair[0], &pair[1]) else {
            return Err(bad());
        };
        if !cids.iter().any(|c| c.as_str() == &**cid) {
            return Err(bad());
        }
        let delay = match *delay_ms {
            -1 => None,
            ms => Some(Duration::from_millis(u64::try_from(ms).map_err(|_| bad())?)),
        };
        removed.push((cid.to_string(), delay));
    }
    Ok(SweepBatch { removed, remaining })
}

/// Decode one presence info hash's `HGETALL` reply for [`read_member_infos`]:
/// `Ok(None)` for an empty or malformed hash, as that function defines them.
/// `Err` only for a reply that is not a field/value map at all, which
/// `HGETALL` never sends.
fn decode_member_info(reply: Value) -> Result<Option<(String, String)>, fred::error::Error> {
    let kind = reply.kind();
    // A pipelined reply is the raw frame, not `hgetall`'s post-processed map:
    // a map under RESP3, a flat field/value array under RESP2.
    let fields = match reply {
        Value::Map(_) | Value::Array(_) => reply.into_map().map_err(|_| {
            unexpected_reply(
                "HGETALL presence info",
                format!("{kind} that is not a field/value map"),
            )
        })?,
        _ => return Err(unexpected_reply("HGETALL presence info", kind.to_string())),
    };
    let mut user_name = None;
    let mut user_application = None;
    for (field, value) in fields.inner() {
        // fred decodes a value that is not valid UTF-8 as `Bytes`.
        let Value::String(value) = value else {
            return Ok(None);
        };
        match field.as_bytes() {
            b"user_name" => user_name = Some(value.to_string()),
            b"user_application" => user_application = Some(value.to_string()),
            _ => {}
        }
    }
    Ok(user_name.map(|name| (name, user_application.unwrap_or_default())))
}

/// A script or command replied with a value outside its documented contract.
/// `Protocol` because the store answered, but not in the shape agreed with it;
/// it is not a timeout, so no origin counts it.
fn unexpected_reply(script: &str, reply: String) -> fred::error::Error {
    fred::error::Error::new(
        fred::error::ErrorKind::Protocol,
        format!("{script} returned an unexpected reply: {reply}"),
    )
}

/// `presence_test_evict.lua`; sent only by [`evict_connection_for_test`].
#[cfg(any(test, feature = "test-support"))]
static PRESENCE_TEST_EVICT_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/presence_test_evict.lua")));

/// What [`evict_connection_for_test`] did.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TestEviction {
    /// True iff this call removed the membership, which is exactly when the
    /// info hash was deleted and `left` was published.
    pub removed: bool,
    /// Members left in the session's member set after the removal.
    pub remaining: u64,
}

/// **Test-only.** Evict a connection the way a sweep does, in one atomic step
/// (`scripts/presence_test_evict.lua`): if it is a member, remove it, delete
/// its info hash and publish `left_payload` on
/// [`KeySchema::presence_events_channel`]; otherwise change nothing.
///
/// Tests evict through this rather than through the live sweeper, which removes
/// a connection only once its lease has expired and only on its own leader-gated
/// cadence, so a test cannot choose the moment of the eviction. It neither
/// checks nor touches the lease: a test that needs the lease gone (so a
/// keeper's refresh sees the eviction rather than a live lease) deletes it
/// itself. Nor does it touch the session index or the teardown marker.
///
/// Runs on the data pool. A command timeout is counted once on `origin`.
#[cfg(any(test, feature = "test-support"))]
pub async fn evict_connection_for_test(
    client: &ValkeyClient,
    origin: FredErrOrigin<'_>,
    keys: &KeySchema,
    session_id: &str,
    connection_id: &str,
    left_payload: &str,
) -> Result<TestEviction, fred::error::Error> {
    let script_keys = vec![
        keys.presence_members(session_id),
        keys.presence_user(session_id, connection_id),
    ];
    let script_argv = vec![
        connection_id.to_owned(),
        keys.presence_events_channel(session_id),
        left_payload.to_owned(),
    ];
    let (removed, remaining): (i64, i64) = PRESENCE_TEST_EVICT_SCRIPT
        .evalsha_with_reload::<(i64, i64), _, _>(client.pool.next(), script_keys, script_argv)
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    removal_reply("presence_test_evict.lua", removed, remaining)
        .map(|(removed, remaining)| TestEviction { removed, remaining })
}

// The reply decoding behind `read_member_infos` and `sweep_member_batch`, fed
// hand-built replies: the shapes a real Valkey never sends (a short or long
// reply list, a non-map reply, a sweep reply outside the script's contract)
// and both RESP encodings of a hash, which one store connection cannot show
// side by side. The store-backed behaviour is pinned in
// `tests/integration/storage/presence_ops_test.rs`.
#[cfg(test)]
mod tests {
    use std::time::Duration;

    use fred::error::{Error, ErrorKind};
    use fred::types::{Map, Value};

    use super::{decode_member_info_replies, decode_sweep_reply, SweepBatch};

    type MemberInfo = Option<(String, String)>;

    /// The fields register writes for a member.
    fn member_fields(name: &str, app: &str) -> Vec<(&'static str, Value)> {
        vec![
            ("user_name", Value::from(name)),
            ("user_application", Value::from(app)),
            ("lease_expires_at_ms", Value::from("1700000000000")),
        ]
    }

    /// An `HGETALL` reply as a RESP2 connection receives it: one flat
    /// field/value array.
    fn resp2_hash(fields: &[(&str, Value)]) -> Value {
        Value::Array(
            fields
                .iter()
                .flat_map(|(field, value)| [Value::from(*field), value.clone()])
                .collect(),
        )
    }

    /// An `HGETALL` reply as a RESP3 connection receives it: a map.
    fn resp3_hash(fields: &[(&str, Value)]) -> Value {
        Value::Map(
            fields
                .iter()
                .map(|(field, value)| (*field, value.clone()))
                .collect::<Map>(),
        )
    }

    fn member(name: &str, app: &str) -> MemberInfo {
        Some((name.to_owned(), app.to_owned()))
    }

    /// `replies` as a pipeline that answered every command.
    fn answered(replies: Vec<Value>) -> Vec<Result<Value, Error>> {
        replies.into_iter().map(Ok).collect()
    }

    fn members(n: usize) -> Vec<Value> {
        (0..n)
            .map(|i| resp2_hash(&member_fields(&format!("user-{i}"), "app")))
            .collect()
    }

    fn assert_protocol_error(result: Result<Vec<MemberInfo>, Error>, what: &str) {
        match result {
            Ok(infos) => panic!("{what} must be a Protocol error; got Ok({infos:?})"),
            Err(e) => assert_eq!(
                *e.kind(),
                ErrorKind::Protocol,
                "{what} must be a Protocol error; got {e:?}"
            ),
        }
    }

    #[test]
    fn decode_member_info_replies_one_reply_short_is_protocol_error() {
        let result = decode_member_info_replies(answered(members(2)), 3);

        assert_protocol_error(result, "two replies for three commands");
    }

    #[test]
    fn decode_member_info_replies_one_reply_over_is_protocol_error() {
        let result = decode_member_info_replies(answered(members(4)), 3);

        assert_protocol_error(result, "four replies for three commands");
    }

    /// A failed pipeline arrives as one `Err` entry, shorter than the command
    /// count. It must surface as itself, so a timeout stays a timeout (and is
    /// counted as one) rather than becoming an arity `Protocol` error.
    #[test]
    fn decode_member_info_replies_lone_timeout_entry_surfaces_before_arity_check() {
        let replies = vec![Err(Error::new(ErrorKind::Timeout, "pipeline timed out"))];

        let err = decode_member_info_replies(replies, 3)
            .expect_err("a failed pipeline must fail the page");

        assert_eq!(*err.kind(), ErrorKind::Timeout, "got {err:?}");
    }

    #[test]
    fn decode_member_info_replies_integer_reply_is_protocol_error() {
        let result = decode_member_info_replies(answered(vec![Value::Integer(1)]), 1);

        assert_protocol_error(result, "an integer HGETALL reply");
    }

    #[test]
    fn decode_member_info_replies_odd_length_array_reply_is_protocol_error() {
        let odd = Value::Array(vec![
            Value::from("user_name"),
            Value::from("someone"),
            Value::from("user_application"),
        ]);

        let result = decode_member_info_replies(answered(vec![odd]), 1);

        assert_protocol_error(result, "an odd-length HGETALL array");
    }

    /// The same hashes decode identically whichever RESP version the data
    /// pool's connection speaks.
    #[test]
    fn decode_member_info_replies_resp3_map_decodes_like_resp2_array() {
        let hashes = [
            member_fields("someone", "some-app"),
            vec![],
            vec![("user_application", Value::from("nameless-app"))],
        ];

        let resp2 = decode_member_info_replies(
            answered(hashes.iter().map(|h| resp2_hash(h)).collect()),
            hashes.len(),
        )
        .expect("RESP2 replies must decode");
        let resp3 = decode_member_info_replies(
            answered(hashes.iter().map(|h| resp3_hash(h)).collect()),
            hashes.len(),
        )
        .expect("RESP3 replies must decode");

        let expected = vec![member("someone", "some-app"), None, None];
        assert_eq!(resp2, expected, "RESP2 flat arrays");
        assert_eq!(
            resp3, expected,
            "RESP3 maps decode the same as RESP2 arrays"
        );
    }

    /// Every entry is decoded on its own: each malformed hash is `None` at its
    /// position, a missing `user_application` reads as empty, and the rest of
    /// the page decodes.
    #[test]
    fn decode_member_info_replies_mixed_page_decodes_each_entry_in_place() {
        let not_utf8 = Value::Bytes(bytes::Bytes::from_static(&[0xff, 0xfe]));
        let replies = vec![
            resp2_hash(&member_fields("first", "app-1")),
            resp2_hash(&[]),
            resp2_hash(&[("user_application", Value::from("app-no-name"))]),
            resp2_hash(&[
                ("user_name", Value::from("bad-app")),
                ("user_application", not_utf8),
            ]),
            resp2_hash(&[("user_name", Value::from("no-app"))]),
            resp2_hash(&member_fields("last", "app-2")),
        ];

        let infos = decode_member_info_replies(answered(replies), 6)
            .expect("a malformed hash must not fail the page");

        assert_eq!(
            infos,
            vec![
                member("first", "app-1"),
                None,
                None,
                None,
                member("no-app", ""),
                member("last", "app-2"),
            ]
        );
    }

    // ── decode_sweep_reply ──────────────────────────────────────────────────

    fn chunk(cids: &[&str]) -> Vec<String> {
        cids.iter().map(|cid| (*cid).to_owned()).collect()
    }

    /// A `presence_sweep_batch.lua` reply: `{removed, remaining}`.
    fn sweep_reply(removed: Vec<Value>, remaining: Value) -> Value {
        Value::Array(vec![Value::Array(removed), remaining])
    }

    fn assert_sweep_protocol_error(reply: Value, cids: &[String], what: &str) {
        match decode_sweep_reply(reply, cids) {
            Ok(batch) => panic!("{what} must be a Protocol error; got Ok({batch:?})"),
            Err(e) => assert_eq!(
                *e.kind(),
                ErrorKind::Protocol,
                "{what} must be a Protocol error; got {e:?}"
            ),
        }
    }

    /// Every accepted shape: -1 is "no usable field" (`None`), 0 and a positive
    /// delay are milliseconds, removals keep the reply's order, and
    /// `remaining` passes through.
    #[test]
    fn decode_sweep_reply_accepts_absent_zero_and_positive_delays_in_reply_order() {
        let cids = chunk(&["conn-a", "conn-b", "conn-c", "conn-d"]);
        let reply = sweep_reply(
            vec![
                Value::from("conn-c"),
                Value::Integer(-1),
                Value::from("conn-a"),
                Value::Integer(0),
                Value::from("conn-d"),
                Value::Integer(1_234),
            ],
            Value::Integer(7),
        );

        let batch = decode_sweep_reply(reply, &cids).expect("a contract reply must decode");

        assert_eq!(
            batch,
            SweepBatch {
                removed: vec![
                    ("conn-c".to_owned(), None),
                    ("conn-a".to_owned(), Some(Duration::ZERO)),
                    ("conn-d".to_owned(), Some(Duration::from_millis(1_234))),
                ],
                remaining: 7,
            }
        );
    }

    #[test]
    fn decode_sweep_reply_accepts_nothing_removed() {
        let batch = decode_sweep_reply(sweep_reply(vec![], Value::Integer(0)), &chunk(&["c"]))
            .expect("an empty removal list must decode");

        assert_eq!(
            batch,
            SweepBatch {
                removed: vec![],
                remaining: 0,
            }
        );
    }

    #[test]
    fn decode_sweep_reply_odd_length_removed_is_protocol_error() {
        let reply = sweep_reply(
            vec![Value::from("c"), Value::Integer(0), Value::from("d")],
            Value::Integer(0),
        );

        assert_sweep_protocol_error(reply, &chunk(&["c", "d"]), "an odd-length removed array");
    }

    #[test]
    fn decode_sweep_reply_more_pairs_than_cids_is_protocol_error() {
        let reply = sweep_reply(
            vec![
                Value::from("c"),
                Value::Integer(0),
                Value::from("c"),
                Value::Integer(0),
            ],
            Value::Integer(0),
        );

        assert_sweep_protocol_error(reply, &chunk(&["c"]), "two removals from a one-id chunk");
    }

    #[test]
    fn decode_sweep_reply_cid_not_in_chunk_is_protocol_error() {
        let reply = sweep_reply(
            vec![Value::from("someone-else"), Value::Integer(0)],
            Value::Integer(0),
        );

        assert_sweep_protocol_error(reply, &chunk(&["c"]), "a removed id not in the chunk");
    }

    /// Only -1 means "no usable field"; any other negative delay is outside
    /// the contract rather than a guessed `None`.
    #[test]
    fn decode_sweep_reply_negative_delay_other_than_minus_one_is_protocol_error() {
        for delay in [-2, i64::MIN] {
            let reply = sweep_reply(
                vec![Value::from("c"), Value::Integer(delay)],
                Value::Integer(0),
            );

            assert_sweep_protocol_error(reply, &chunk(&["c"]), &format!("a delay of {delay}"));
        }
    }

    #[test]
    fn decode_sweep_reply_non_integer_delay_is_protocol_error() {
        let reply = sweep_reply(vec![Value::from("c"), Value::from("5")], Value::Integer(0));

        assert_sweep_protocol_error(reply, &chunk(&["c"]), "a string delay");
    }

    #[test]
    fn decode_sweep_reply_non_string_cid_is_protocol_error() {
        let reply = sweep_reply(
            vec![Value::Integer(1), Value::Integer(0)],
            Value::Integer(0),
        );

        assert_sweep_protocol_error(reply, &chunk(&["1"]), "an integer cid");
    }

    #[test]
    fn decode_sweep_reply_non_integer_remaining_is_protocol_error() {
        for remaining in [Value::from("3"), Value::Null, Value::Array(vec![])] {
            let what = format!("a remaining of {remaining:?}");
            let reply = sweep_reply(vec![], remaining);

            assert_sweep_protocol_error(reply, &chunk(&["c"]), &what);
        }
    }

    #[test]
    fn decode_sweep_reply_negative_remaining_is_protocol_error() {
        let reply = sweep_reply(vec![], Value::Integer(-1));

        assert_sweep_protocol_error(reply, &chunk(&["c"]), "a negative remaining");
    }

    /// The top level must be exactly `{removed, remaining}` with `removed` an
    /// array.
    #[test]
    fn decode_sweep_reply_wrong_top_level_shape_is_protocol_error() {
        let cids = chunk(&["c"]);
        let shapes = [
            ("an integer reply", Value::Integer(0)),
            (
                "a one-element array",
                Value::Array(vec![Value::Array(vec![])]),
            ),
            (
                "a three-element array",
                Value::Array(vec![
                    Value::Array(vec![]),
                    Value::Integer(0),
                    Value::Integer(0),
                ]),
            ),
            (
                "a non-array removed",
                Value::Array(vec![Value::Integer(0), Value::Integer(0)]),
            ),
        ];

        for (what, reply) in shapes {
            assert_sweep_protocol_error(reply, &cids, what);
        }
    }
}
