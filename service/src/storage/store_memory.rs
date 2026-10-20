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

//! The backing store's memory, read from the store itself (b.yuo, b.qhk),
//! for two consumers: the external-value release budget `H`, and how full
//! the store is (`live_edit_store_memory_used_ratio`).
//!
//! **The reading.** [`read_store_memory`] reads the cluster map's slot ranges
//! when `LIVE_EDIT_VALKEY_MEMORY_BYTES` is set — one per entry of a
//! `CLUSTER SLOTS` it sends on every pass, so a reshard is seen within one
//! pass; one range covering every slot on a store that is not clustered —
//! then sends `INFO memory` to every distinct primary endpoint and records
//! each one's `maxmemory` and `used_memory`. A failed `INFO` keeps the map,
//! and a failed or malformed `CLUSTER SLOTS` leaves the pass with no map
//! while the `INFO`s go on. Only when no primary reports a `maxmemory` field at all and
//! `LIVE_EDIT_VALKEY_MEMORY_BYTES` is set does it also send `INFO
//! commandstats`, for the store-wide count of replica acknowledgements
//! (`cmdstat_replconf`, [`store_copies`]). Startup reads once whatever is
//! configured.
//!
//! **Budget precedence.**
//!
//! 1. `LIVE_EDIT_EXT_RELEASE_MAX_BYTES` when set
//!    ([`ExtReleaseBudgetSource::Explicit`], `Config::ext_release_budget`).
//!    Nothing here ever moves it.
//! 2. Else, when EVERY primary reports a `maxmemory` above zero, the smallest
//!    ÷ 8 ([`ExtReleaseBudgetSource::StoreInfo`], `source="info"`). A store
//!    that reports its own limit wins over `LIVE_EDIT_VALKEY_MEMORY_BYTES`.
//! 3. Else, when `LIVE_EDIT_VALKEY_MEMORY_BYTES` — the store's TOTAL memory
//!    as the provider states it — is set and the map is known, one shard's
//!    share of it ÷ 8 ([`memory_bytes_budget`];
//!    [`ExtReleaseBudgetSource::ValkeyMemory`], `source="override"`). It
//!    needs only the map: a failed or refused `INFO` does not stop it, though
//!    a failed `INFO` keeps a rule-2 budget already in force.
//! 4. Else the fallback, `min(8 × LIVE_EDIT_MAX_LAYER_SIZE, 100 MB)`
//!    (`Config::ext_release_budget`), and startup logs why.
//!
//! A 0 or missing `maxmemory` never reads as "unlimited": 0 is Valkey's "no
//! limit configured", and a missing field is a managed store that does not
//! expose it (Azure Managed Redis's `INFO memory` has no `maxmemory` and
//! reports the whole database, the same figure on every endpoint), so neither
//! says how much one primary can hold.
//!
//! **One shard's memory, never the sum.** On a cluster every released blob
//! can sit on one shard, so `H` derives from one shard: the smallest reported
//! `maxmemory`, or the smallest slot range's share of the setting's total.
//! The shard count is the number of slot-range entries in the cluster map,
//! never the number of distinct endpoints: on Azure Managed Redis one
//! endpoint (a node) fronts several primary shards, each listed as its own
//! `CLUSTER SLOTS` entry (`CLUSTER NODES` merges them per node and is never
//! used). The endpoint count is only logged, beside the shard count, so an
//! operator can compare both with the provider's portal.
//!
//! **How full ([`store_memory_gauge`]).** On a store whose primaries all
//! report a `maxmemory` above zero, the highest `used_memory ÷ maxmemory`
//! across them. On a store none of whose primaries reports a `maxmemory`
//! field, with `LIVE_EDIT_VALKEY_MEMORY_BYTES` set, the highest whole-database
//! `used_memory` ÷ the copies of the data ÷ [`store_usable_bytes`] of the
//! setting. That store's `used_memory` counts replica copies too, and its
//! `INFO replication` is a placeholder, so the copies are seen instead: 2
//! when the replica acknowledgement count rose since the previous read, 1
//! when it did not move ([`store_copies`]) — so the figure is absent at a
//! replica's first read and present from its second (later, when a count
//! could not be read). Otherwise — and
//! whenever the latest read failed or a primary reported no `used_memory` —
//! no figure: the series is absent, never 0. A figure above 1 is kept as it
//! is: the store is past its limit.
//!
//! **Logged by reason (b.1jt).** Why a reading's `maxmemory` was unusable is
//! typed ([`UnusableReason`]), and a failed `INFO memory` (an error, or one an
//! ACL denied) is [`StoreMemoryProblem::ReadFailed`]. A store that has no
//! `maxmemory` field is the expected case on a managed store: silent when
//! `LIVE_EDIT_VALKEY_MEMORY_BYTES` is set (it answers both consumers), INFO
//! otherwise. Every other reason — `maxmemory:0` (self-hosted Valkey with no
//! limit), a malformed value, an unknown routing table, no primary, a failed
//! or denied `INFO` — is a fault, logged at WARN. Two more faults are logged
//! WARN once per kind, apart from those: a cluster map read that failed, a
//! cluster map reply that was malformed, and an unreadable `INFO
//! commandstats`. Startup logs one line at its reason's level, and the
//! refresh task is seeded with what startup logged, so the same kind is never
//! logged twice in a row. A change of the gauge's arm or copies logs one INFO.
//!
//! **Refresh.** A task ([`spawn_store_memory_refresh`]) reads again every
//! [`STORE_MEMORY_REFRESH_INTERVAL`] whenever a consumer can use the read
//! ([`store_memory_refresh_needed`]): the budget, unless it is explicit; the
//! gauge, when `LIVE_EDIT_VALKEY_MEMORY_BYTES` is set or the store may report
//! its own limit. Every pass records the gauge. Unless the budget is explicit,
//! a pass that resolves to rule 2 or 3 with a different budget moves the
//! upload gate's budget — and with it the GC loop's pressure threshold
//! `H / 2`, which the loop reads off the gate every pass — so a store resized
//! in place, or resharded, needs no redeploy. A reading of no map or no
//! primary, or one neither rule answers, keeps the last good value: the
//! budget a good read set stays, and a replica that never had one stays at
//! the fallback. Each budget it sets is recorded on
//! `live_edit_ext_release_budget_bytes{source}`.
//!
//! **Routing.** `INFO` is keyless and node-scoped (SDD §15.3's keyless-command
//! inventory): on a cluster it is sent once per distinct primary endpoint the
//! maintenance client's cached routing table lists, pinned with
//! `with_cluster_node`; on standalone, once. The slot ranges come from a
//! `CLUSTER SLOTS` sent on the same client as an ordinary keyless command,
//! bounded by the command timeout like the `INFO`s — never fred's
//! `sync_cluster`, which retries a failing `CLUSTER SLOTS` inside the shared
//! maintenance client's router, stalling every other command on it. All of it
//! runs on the maintenance pool, off the data pool's request traffic, writes
//! nothing, and costs well under a millisecond per primary. Valkey classes
//! `INFO` `@slow @dangerous`; an ACL user denied it gets an error. Managed
//! stores restrict `CONFIG`, not `INFO`.

use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{ClientLike, ClusterInterface};
use fred::types::config::Server;
use fred::types::InfoKind;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::config::{ExtReleaseBudget, ExtReleaseBudgetSource, EXT_RELEASE_BUDGET_MEMORY_DIVISOR};
use crate::observability::metrics::SharedMetrics;
use crate::storage::client::{
    map_transient_fred_err, FredErrOrigin, ValkeyClient, POOL_LABEL_MAINTENANCE,
};
use crate::storage::ext_release_gate::ExtReleaseGate;

/// How often the store's memory is read again: often enough that a resized
/// store moves the budget, and the fullness gauge follows the store, within a
/// minute; rarely enough that the read is noise — per replica per minute, one
/// `INFO memory` per primary endpoint; with `LIVE_EDIT_VALKEY_MEMORY_BYTES`
/// set, one `CLUSTER SLOTS` per pass on a clustered store, and, on a store
/// with no `maxmemory` field, one `INFO commandstats` per primary endpoint.
pub const STORE_MEMORY_REFRESH_INTERVAL: Duration = Duration::from_secs(60);

/// The percentage of `LIVE_EDIT_VALKEY_MEMORY_BYTES` treated as usable: a
/// fixed allowance for the memory a managed provider reserves (Azure Managed
/// Redis reserves about 20 %). The one place the 0.8 is stated; read through
/// [`store_usable_bytes`].
pub const STORE_TOTAL_USABLE_PERCENT: u64 = 80;

/// The hash slots of a Valkey / Redis cluster, which a shard's share of the
/// store's memory is measured against ([`memory_bytes_budget`]).
pub const CLUSTER_SLOT_COUNT: u64 = 16_384;

/// The one slot range of a store that is not clustered: every slot.
pub const STANDALONE_SLOT_RANGE: (u16, u16) = (0, 16_383);

/// The copies of the data [`store_copies`] answers when the store's replicas
/// are seen working: a primary and one replica. Azure Managed Redis runs at
/// most one replica per primary.
pub const STORE_REPLICATED_COPIES: u64 = 2;

/// The `maxmemory` field of one `INFO memory` reply
/// ([`parse_info_maxmemory`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxmemoryField {
    /// `maxmemory:<bytes>`. Valkey reports 0 when no limit is configured.
    Value(u64),
    /// No `maxmemory:` line: a store that does not expose it.
    Absent,
    /// A `maxmemory:` line whose value is not a non-negative integer.
    Malformed,
}

/// The value of the exact field `name` (`name` ends in `:`) in an `INFO`
/// reply, or `None` when no line starts with it. Lines are `field:value`,
/// `\r\n`-, `\r\r\n`- or `\n`-separated, with `#` section headers; callers
/// trim the value.
fn info_field<'a>(info: &'a str, name: &str) -> Option<&'a str> {
    info.lines().find_map(|line| line.strip_prefix(name))
}

/// Read the `maxmemory:<bytes>` line of an `INFO memory` reply.
/// `maxmemory_human:` and `maxmemory_policy:` are other fields.
pub fn parse_info_maxmemory(info: &str) -> MaxmemoryField {
    info_field(info, "maxmemory:").map_or(MaxmemoryField::Absent, |value| {
        value
            .trim()
            .parse::<u64>()
            .map_or(MaxmemoryField::Malformed, MaxmemoryField::Value)
    })
}

/// Read the `used_memory:<bytes>` line of an `INFO memory` reply: `None`
/// when it is absent or not a non-negative integer. Only the exact field
/// counts; `used_memory_human:`, `used_memory_rss:`, `used_memory_peak:`,
/// `used_memory_lua:`, `used_memory_dataset:` and the rest are other fields.
pub fn parse_info_used_memory(info: &str) -> Option<u64> {
    info_field(info, "used_memory:").and_then(|value| value.trim().parse::<u64>().ok())
}

/// Read the `calls=<n>` of the `cmdstat_replconf:` line of an `INFO
/// commandstats` reply: how many `REPLCONF` commands the store has run, which
/// every replica sends about once a second to acknowledge replication. A
/// reply with no such line reads 0 (the store lists only commands it has
/// run); a line whose `calls` is missing or not a non-negative integer is
/// `None`. The line's other fields (`usec=`, `rejected_calls=`, ...) are
/// ignored.
pub fn parse_info_replconf_calls(info: &str) -> Option<u64> {
    let Some(stats) = info_field(info, "cmdstat_replconf:") else {
        return Some(0);
    };
    stats
        .trim()
        .split(',')
        .find_map(|field| field.trim().strip_prefix("calls="))
        .and_then(|calls| calls.trim().parse::<u64>().ok())
}

/// The copies of the data a store's whole-database `used_memory` counts, from
/// the store-wide replica acknowledgement count of this task's previous good
/// read (`previous`) and of this one (`current`): [`STORE_REPLICATED_COPIES`]
/// when it rose — replicas are acknowledging, so each shard is held twice —
/// and 1 when it did not move. `None` (the gauge absent) with no previous
/// read, or when the count fell (the store's statistics were reset; the
/// caller re-baselines on `current`). The one place the copies rule is
/// stated; a count that cannot be read never reaches it.
pub fn store_copies(previous: Option<u64>, current: u64) -> Option<u64> {
    match previous? {
        prev if current > prev => Some(STORE_REPLICATED_COPIES),
        prev if current == prev => Some(1),
        _ => None,
    }
}

/// What one primary answered `INFO memory` ([`read_store_memory`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimaryMemory {
    /// Who answered: `"primary <host:port>"` on a cluster, `"the store"` on
    /// standalone.
    pub primary: String,
    /// Its `maxmemory` ([`parse_info_maxmemory`]).
    pub maxmemory: MaxmemoryField,
    /// Its `used_memory` ([`parse_info_used_memory`]).
    pub used_memory: Option<u64>,
}

/// The replica acknowledgement count of one reading ([`read_store_memory`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplconfCalls {
    /// `INFO commandstats` was not sent: the reading is not on the arm that
    /// needs it (no primary reports a `maxmemory` field, and
    /// `LIVE_EDIT_VALKEY_MEMORY_BYTES` is set).
    NotRead,
    /// The highest `cmdstat_replconf` call count any endpoint reported
    /// ([`parse_info_replconf_calls`]).
    Calls(u64),
    /// `INFO commandstats` failed on an endpoint, or a reply's
    /// `cmdstat_replconf` line was malformed; why, as text.
    Unreadable(String),
}

/// What one read of the store found ([`read_store_memory`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreMemoryReading {
    /// The routing table was known: the map, and what `INFO` answered.
    Read {
        /// The cluster map's slot ranges, `(start, end)` inclusive, one per
        /// entry of this pass's `CLUSTER SLOTS` reply — one per primary
        /// shard, however many shards share an endpoint;
        /// [`STANDALONE_SLOT_RANGE`] alone on a store that is not clustered.
        /// Empty when there is no map: `LIVE_EDIT_VALKEY_MEMORY_BYTES` unset
        /// (nothing needs it, so `CLUSTER SLOTS` is not sent), or a reply that
        /// failed or was malformed ([`StoreMemoryRead::map_problem`]).
        slot_ranges: Vec<(u16, u16)>,
        /// The distinct primary endpoints: those of the `CLUSTER SLOTS` reply
        /// when there is one, else those the `INFO`s were addressed to. A log
        /// field only: one endpoint can front several shards, so it never
        /// enters any arithmetic.
        endpoints: usize,
        /// Every endpoint's `INFO memory`, or why the read failed.
        info: Result<Vec<PrimaryMemory>, String>,
        /// The replica acknowledgement count.
        replconf_calls: ReplconfCalls,
    },
    /// The cluster's routing table is not known yet: nothing was read.
    RoutingUnknown,
    /// The cluster's routing table lists no primary: nothing was read.
    NoPrimary,
}

impl StoreMemoryReading {
    /// The primaries' `INFO memory` answers, or `None` when none was read or
    /// the read failed.
    pub fn primaries(&self) -> Option<&[PrimaryMemory]> {
        match self {
            StoreMemoryReading::Read { info: Ok(p), .. } => Some(p),
            _ => None,
        }
    }

    /// The smallest `maxmemory`, and how many primaries reported one, when
    /// every primary reports one above zero; otherwise why not — the failed
    /// read, the reading's own reason, or the first primary, in routing-table
    /// order, without a usable `maxmemory`.
    pub fn maxmemory_limit(&self) -> Result<(u64, usize), StoreMemoryProblem> {
        let unusable = StoreMemoryProblem::Unusable;
        let primaries = match self {
            StoreMemoryReading::Read { info: Ok(p), .. } => p,
            StoreMemoryReading::Read { info: Err(e), .. } => {
                return Err(StoreMemoryProblem::ReadFailed(e.clone()))
            }
            StoreMemoryReading::RoutingUnknown => {
                return Err(unusable(UnusableReason::RoutingUnknown))
            }
            StoreMemoryReading::NoPrimary => return Err(unusable(UnusableReason::NoPrimary)),
        };
        if primaries.is_empty() {
            return Err(unusable(UnusableReason::NoPrimary));
        }
        let mut min_bytes = u64::MAX;
        for p in primaries {
            let primary = p.primary.clone();
            let reason = match p.maxmemory {
                MaxmemoryField::Value(0) => UnusableReason::ZeroLimit { primary },
                MaxmemoryField::Value(bytes) => {
                    min_bytes = min_bytes.min(bytes);
                    continue;
                }
                MaxmemoryField::Absent => UnusableReason::FieldAbsent { primary },
                MaxmemoryField::Malformed => UnusableReason::Malformed { primary },
            };
            return Err(unusable(reason));
        }
        Ok((min_bytes, primaries.len()))
    }

    /// Whether primaries answered and none of them reports a `maxmemory`
    /// field at all: the shape of a managed store that does not expose it
    /// (Azure Managed Redis), as opposed to `maxmemory:0`, a malformed value
    /// or a mix.
    pub fn reports_no_maxmemory_field(&self) -> bool {
        self.primaries().is_some_and(|primaries| {
            !primaries.is_empty()
                && primaries
                    .iter()
                    .all(|p| p.maxmemory == MaxmemoryField::Absent)
        })
    }
}

/// Why a store reading gave no `maxmemory` the budget can use (b.1jt). Each
/// `primary` names who answered: `"primary <host:port>"` on a cluster, `"the
/// store"` on standalone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnusableReason {
    /// `INFO memory` has no `maxmemory:` line: a managed store that does not
    /// expose it (Azure Managed Redis). The expected case there: silent when
    /// `LIVE_EDIT_VALKEY_MEMORY_BYTES` is set, logged INFO otherwise.
    FieldAbsent { primary: String },
    /// `maxmemory:0`: Valkey with no memory limit configured. A fault (a
    /// `noeviction` store with no limit grows until the host runs out),
    /// logged WARN.
    ZeroLimit { primary: String },
    /// A `maxmemory:` value that is not a number of bytes. A fault, WARN.
    Malformed { primary: String },
    /// The cluster's routing table is not known yet. A fault, WARN.
    RoutingUnknown,
    /// The cluster's routing table lists no primary. A fault, WARN.
    NoPrimary,
}

impl UnusableReason {
    /// The reason's kind, without the primary it names.
    pub fn kind(&self) -> StoreMemoryProblemKind {
        match self {
            UnusableReason::FieldAbsent { .. } => StoreMemoryProblemKind::FieldAbsent,
            UnusableReason::ZeroLimit { .. } => StoreMemoryProblemKind::ZeroLimit,
            UnusableReason::Malformed { .. } => StoreMemoryProblemKind::Malformed,
            UnusableReason::RoutingUnknown => StoreMemoryProblemKind::RoutingUnknown,
            UnusableReason::NoPrimary => StoreMemoryProblemKind::NoPrimary,
        }
    }
}

impl std::fmt::Display for UnusableReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnusableReason::FieldAbsent { primary } => write!(
                f,
                "{primary} reports no maxmemory field in INFO memory (a managed store that does \
                 not expose it)"
            ),
            UnusableReason::ZeroLimit { primary } => write!(
                f,
                "{primary} reports maxmemory 0 (no memory limit configured)"
            ),
            UnusableReason::Malformed { primary } => write!(
                f,
                "{primary} reports a maxmemory that is not a number of bytes"
            ),
            UnusableReason::RoutingUnknown => {
                f.write_str("the cluster's routing table is not known yet")
            }
            UnusableReason::NoPrimary => {
                f.write_str("the cluster's routing table lists no primary")
            }
        }
    }
}

/// Why the store's own `maxmemory` was not usable (b.1jt): a reading with no
/// usable `maxmemory`, or an `INFO memory` that failed — errored, or an ACL
/// denied it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreMemoryProblem {
    /// The store answered, without a usable `maxmemory`.
    Unusable(UnusableReason),
    /// The `INFO memory` read failed; the error, as text.
    ReadFailed(String),
}

impl StoreMemoryProblem {
    /// The problem's kind.
    pub fn kind(&self) -> StoreMemoryProblemKind {
        match self {
            StoreMemoryProblem::Unusable(reason) => reason.kind(),
            StoreMemoryProblem::ReadFailed(_) => StoreMemoryProblemKind::ReadFailed,
        }
    }

    /// The problem a reading carries ([`StoreMemoryReading::maxmemory_limit`]'s
    /// error), or `None` when every primary reports a `maxmemory` above zero.
    pub fn of_reading(reading: &StoreMemoryReading) -> Option<Self> {
        reading.maxmemory_limit().err()
    }
}

impl std::fmt::Display for StoreMemoryProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreMemoryProblem::Unusable(reason) => reason.fmt(f),
            StoreMemoryProblem::ReadFailed(error) => {
                write!(f, "the INFO memory read failed: {error}")
            }
        }
    }
}

/// The kind of a [`StoreMemoryProblem`], which decides its log level and is
/// what the refresh task compares to log each kind once (b.1jt).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreMemoryProblemKind {
    /// [`UnusableReason::FieldAbsent`].
    FieldAbsent,
    /// [`UnusableReason::ZeroLimit`].
    ZeroLimit,
    /// [`UnusableReason::Malformed`].
    Malformed,
    /// [`UnusableReason::RoutingUnknown`].
    RoutingUnknown,
    /// [`UnusableReason::NoPrimary`].
    NoPrimary,
    /// [`StoreMemoryProblem::ReadFailed`].
    ReadFailed,
}

impl StoreMemoryProblemKind {
    /// Whether this kind is a fault (logged WARN) rather than the expected
    /// case on a managed store. Only a missing `maxmemory` field is expected:
    /// Azure Managed Redis never reports one.
    pub fn is_fault(self) -> bool {
        !matches!(self, StoreMemoryProblemKind::FieldAbsent)
    }
}

/// One store read, with why its `CLUSTER SLOTS` gave no map, if it did not
/// ([`read_store_memory`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreMemoryRead {
    /// What the read found.
    pub reading: StoreMemoryReading,
    /// Why this pass's `CLUSTER SLOTS` gave no map; `None` when it gave one
    /// or was not sent (`LIVE_EDIT_VALKEY_MEMORY_BYTES` unset, a store that is
    /// not clustered, or no routing table to address the read). With a
    /// problem the reading's `slot_ranges` are empty: the pass behaves as one
    /// with no map.
    pub map_problem: Option<ClusterMapProblem>,
}

/// The cluster map one `CLUSTER SLOTS` reply gives
/// ([`parse_cluster_slots_reply`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterSlotsMap {
    /// Each entry's `(start, end)`, inclusive, in reply order: one per
    /// primary shard, however many shards share an endpoint.
    pub slot_ranges: Vec<(u16, u16)>,
    /// The distinct primary `host:port`s the entries name, a NULL host keyed
    /// as the replying node. A log field only.
    pub endpoints: usize,
}

/// Why a pass's `CLUSTER SLOTS` gave no map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusterMapProblem {
    /// The command failed; the error, as text.
    ReadFailed(String),
    /// The reply was empty, or one of its entries was malformed
    /// ([`parse_cluster_slots_reply`]).
    Malformed,
}

impl ClusterMapProblem {
    /// The problem's kind, which the refresh task compares to log each kind
    /// once.
    pub fn kind(&self) -> ClusterMapProblemKind {
        match self {
            ClusterMapProblem::ReadFailed(_) => ClusterMapProblemKind::ReadFailed,
            ClusterMapProblem::Malformed => ClusterMapProblemKind::Malformed,
        }
    }
}

impl std::fmt::Display for ClusterMapProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClusterMapProblem::ReadFailed(error) => {
                write!(f, "the CLUSTER SLOTS read failed: {error}")
            }
            ClusterMapProblem::Malformed => {
                f.write_str("the CLUSTER SLOTS reply was empty or had a malformed entry")
            }
        }
    }
}

/// The kind of a [`ClusterMapProblem`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterMapProblemKind {
    /// [`ClusterMapProblem::ReadFailed`].
    ReadFailed,
    /// [`ClusterMapProblem::Malformed`].
    Malformed,
}

/// One hash slot of a `CLUSTER SLOTS` entry: an integer below
/// [`CLUSTER_SLOT_COUNT`].
fn cluster_slot(value: &fred::types::Value) -> Option<u16> {
    match value {
        fred::types::Value::Integer(n) => u16::try_from(*n)
            .ok()
            .filter(|slot| u64::from(*slot) < CLUSTER_SLOT_COUNT),
        _ => None,
    }
}

/// Read a `CLUSTER SLOTS` reply strictly: every entry must be
/// `[start, end, [primary_host, primary_port, ...], replicas...]` with
/// integer `start <= end <= 16383`, a string or NULL host and an integer
/// port; the node id, metadata and replica elements are not read. A NULL host
/// is Redis 7+ / Valkey's `cluster-preferred-endpoint-type unknown-endpoint`
/// answer and means the replying node, as which [`ClusterSlotsMap::endpoints`]
/// keys it. Each entry gives one slot range, in reply order, never merged
/// with another on the same endpoint. `None` — no map — for an empty reply
/// or any malformed entry: a partial map would undercount the shards.
pub fn parse_cluster_slots_reply(reply: &fred::types::Value) -> Option<ClusterSlotsMap> {
    use fred::types::Value;
    let Value::Array(entries) = reply else {
        return None;
    };
    if entries.is_empty() {
        return None;
    }
    let mut slot_ranges = Vec::with_capacity(entries.len());
    let mut endpoints = std::collections::BTreeSet::new();
    for entry in entries {
        let Value::Array(fields) = entry else {
            return None;
        };
        let start = cluster_slot(fields.first()?)?;
        let end = cluster_slot(fields.get(1)?)?;
        if start > end {
            return None;
        }
        let Value::Array(primary) = fields.get(2)? else {
            return None;
        };
        let host = match primary.first()? {
            Value::String(host) => host.to_string(),
            Value::Bytes(host) => std::str::from_utf8(host).ok()?.to_owned(),
            // "The node you asked": keyed as "" for the log-only endpoint set.
            Value::Null => String::new(),
            _ => return None,
        };
        let Value::Integer(port) = primary.get(1)? else {
            return None;
        };
        let port = u16::try_from(*port).ok()?;
        slot_ranges.push((start, end));
        endpoints.insert((host, port));
    }
    Some(ClusterSlotsMap {
        slot_ranges,
        endpoints: endpoints.len(),
    })
}

/// Send `CLUSTER SLOTS` as an ordinary command on `node_client` — bounded by
/// that client's command timeout like the `INFO`s, its error classified by
/// [`map_transient_fred_err`] — and read the reply
/// ([`parse_cluster_slots_reply`]). Never fred's `sync_cluster`, which
/// retries a failing `CLUSTER SLOTS` inside the shared client's router and
/// stalls every other command on it meanwhile.
async fn read_cluster_map(
    client: &ValkeyClient,
    node_client: &fred::clients::Client,
) -> Result<ClusterSlotsMap, ClusterMapProblem> {
    let reply: fred::types::Value = node_client.cluster_slots().await.map_err(|e| {
        ClusterMapProblem::ReadFailed(
            map_transient_fred_err(
                FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE),
                "CLUSTER SLOTS",
                e,
            )
            .to_string(),
        )
    })?;
    parse_cluster_slots_reply(&reply).ok_or(ClusterMapProblem::Malformed)
}

/// Send `INFO <kind>` to `node` (a cluster primary, pinned) or to the store
/// (standalone, `None`), named `command` in the error mapping.
async fn info_on(
    client: &ValkeyClient,
    node_client: &fred::clients::Client,
    node: Option<&Server>,
    kind: InfoKind,
    command: &'static str,
) -> Result<String, String> {
    match node {
        Some(server) => {
            node_client
                .with_cluster_node(server.clone())
                .info(Some(kind))
                .await
        }
        None => node_client.info(Some(kind)).await,
    }
    .map_err(|e| {
        map_transient_fred_err(
            FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE),
            command,
            e,
        )
        .to_string()
    })
}

/// Read the store's memory on one maintenance-pool client.
///
/// 1. On a cluster, take the `INFO` addresses — the distinct primaries — from
///    that client's cached routing table: fred keys its node connections by
///    the routing table's own `host:port`, which can differ from the raw
///    `CLUSTER SLOTS` host (fred prefers an announced hostname to the IP, and
///    a missing host to the replying node's), and a node-pinned command to a
///    server it holds no connection for makes it reconnect. No routing table
///    yet, or one with no primary, ends the read.
/// 2. When `memory_bytes` (`LIVE_EDIT_VALKEY_MEMORY_BYTES`) is set, the only
///    consumer of the map, send `CLUSTER SLOTS` on that client as an
///    ordinary command and read the map from its reply
///    ([`parse_cluster_slots_reply`]): one slot range per entry, and the
///    distinct primaries it names. A failed or malformed reply leaves the
///    pass with no map ([`StoreMemoryRead::map_problem`]). A store that is
///    not clustered is one range covering every slot, with no command sent.
/// 3. `INFO memory` to every primary — via `with_cluster_node` on a
///    cluster, once on standalone — recording each one's `maxmemory` and
///    `used_memory`; a failed `INFO` ends this step, keeps the map, and is
///    classified by [`map_transient_fred_err`] (its timeout counted).
/// 4. When no primary reports a `maxmemory` field and `memory_bytes` is
///    set, `INFO commandstats` to every primary, for the highest
///    `cmdstat_replconf` call count.
pub async fn read_store_memory(
    client: &ValkeyClient,
    memory_bytes: Option<u64>,
) -> StoreMemoryRead {
    let (pool, _maint_token) = client.acquire_maintenance();
    let node_client = pool.next();
    let (nodes, slot_ranges, endpoints, map_problem) = if node_client.is_clustered() {
        let Some(routing) = node_client.cached_cluster_state() else {
            return StoreMemoryRead {
                reading: StoreMemoryReading::RoutingUnknown,
                map_problem: None,
            };
        };
        // INFO addresses come from fred's routing table, not CLUSTER SLOTS: a pinned
        // send matches its connection by `Server` host:port, and fred's host can
        // differ from the reply's (metadata hostname; replying node for NULL) — a
        // miss fails "Failed to route command" and stalls the shared member in
        // `defer_reconnection`. Ranges drive all arithmetic; endpoints only log.
        let nodes: Vec<Option<Server>> = routing
            .unique_primary_nodes()
            .into_iter()
            .map(Some)
            .collect();
        if nodes.is_empty() {
            return StoreMemoryRead {
                reading: StoreMemoryReading::NoPrimary,
                map_problem: None,
            };
        }
        let map = if memory_bytes.is_some() {
            Some(read_cluster_map(client, node_client).await)
        } else {
            None
        };
        match map {
            Some(Ok(map)) => (nodes, map.slot_ranges, map.endpoints, None),
            Some(Err(problem)) => {
                let endpoints = nodes.len();
                (nodes, Vec::new(), endpoints, Some(problem))
            }
            None => {
                let endpoints = nodes.len();
                (nodes, Vec::new(), endpoints, None)
            }
        }
    } else {
        (vec![None], vec![STANDALONE_SLOT_RANGE], 1, None)
    };
    let info = read_info_memory(client, node_client, &nodes).await;
    let mut reading = StoreMemoryReading::Read {
        slot_ranges,
        endpoints,
        info,
        replconf_calls: ReplconfCalls::NotRead,
    };
    if memory_bytes.is_some() && reading.reports_no_maxmemory_field() {
        let calls = read_replconf_calls(client, node_client, &nodes).await;
        if let StoreMemoryReading::Read { replconf_calls, .. } = &mut reading {
            *replconf_calls = calls;
        }
    }
    StoreMemoryRead {
        reading,
        map_problem,
    }
}

/// `INFO memory` on every one of `nodes`, in order, stopping at the first
/// failure.
async fn read_info_memory(
    client: &ValkeyClient,
    node_client: &fred::clients::Client,
    nodes: &[Option<Server>],
) -> Result<Vec<PrimaryMemory>, String> {
    let mut primaries = Vec::with_capacity(nodes.len());
    for node in nodes {
        let info = info_on(
            client,
            node_client,
            node.as_ref(),
            InfoKind::Memory,
            "INFO memory",
        )
        .await?;
        primaries.push(PrimaryMemory {
            primary: primary_name(node.as_ref()),
            maxmemory: parse_info_maxmemory(&info),
            used_memory: parse_info_used_memory(&info),
        });
    }
    Ok(primaries)
}

/// `INFO commandstats` on every one of `nodes`: the highest `cmdstat_replconf`
/// call count, or why it could not be read.
async fn read_replconf_calls(
    client: &ValkeyClient,
    node_client: &fred::clients::Client,
    nodes: &[Option<Server>],
) -> ReplconfCalls {
    let mut highest = 0;
    for node in nodes {
        let info = match info_on(
            client,
            node_client,
            node.as_ref(),
            InfoKind::CommandStats,
            "INFO commandstats",
        )
        .await
        {
            Ok(info) => info,
            Err(e) => return ReplconfCalls::Unreadable(format!("INFO commandstats failed: {e}")),
        };
        match parse_info_replconf_calls(&info) {
            Some(calls) => highest = highest.max(calls),
            None => {
                return ReplconfCalls::Unreadable(format!(
                    "{} reports a cmdstat_replconf line without a call count",
                    primary_name(node.as_ref())
                ))
            }
        }
    }
    ReplconfCalls::Calls(highest)
}

/// Who answered, as a reason names it: `"primary <host:port>"` on a cluster,
/// `"the store"` on standalone.
fn primary_name(node: Option<&Server>) -> String {
    node.map_or_else(
        || "the store".to_owned(),
        |server| ["primary ", &server.to_string()].concat(),
    )
}

/// The usable part of a provider-stated store total `total_bytes`:
/// [`STORE_TOTAL_USABLE_PERCENT`] of it, rounded down. Never overflows.
pub fn store_usable_bytes(total_bytes: u64) -> u64 {
    let usable = u128::from(total_bytes) * u128::from(STORE_TOTAL_USABLE_PERCENT) / 100;
    u64::try_from(usable).unwrap_or(u64::MAX)
}

/// How [`memory_bytes_budget`] derived a budget from
/// `LIVE_EDIT_VALKEY_MEMORY_BYTES`: carried by
/// [`ExtReleaseBudgetSource::ValkeyMemory`], so the startup and refresh lines
/// name it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBytesDerivation {
    /// `LIVE_EDIT_VALKEY_MEMORY_BYTES`: the store's total memory as the
    /// provider states it.
    pub total_bytes: u64,
    /// The cluster map's slot-range entries: the primary shards.
    pub shards: usize,
    /// The slots of the smallest entry.
    pub smallest_shard_slots: u64,
    /// The smallest shard's share of the usable total, in bytes.
    pub per_shard_bytes: u64,
}

/// Rule 3: the release budget `LIVE_EDIT_VALKEY_MEMORY_BYTES`
/// (`memory_bytes`, the store's total memory) gives over the cluster map's
/// slot ranges (`(start, end)` inclusive, one per `CLUSTER SLOTS` entry), and
/// how it was derived; `None` for an empty map.
///
/// The smallest shard's share of the usable total,
/// `per_shard = memory_bytes × 0.8 × (smallest entry's slots ÷ 16384)`
/// ([`STORE_TOTAL_USABLE_PERCENT`], [`CLUSTER_SLOT_COUNT`]), and
/// `H = max(1, per_shard ÷ 8)` ([`EXT_RELEASE_BUDGET_MEMORY_DIVISOR`]): at
/// least 1, so it never reads as the disabling 0. On equal stripes this is
/// the usable total ÷ the shard count, whatever the number of endpoints.
/// Integer arithmetic in u128, rounded down, without overflow at `u64::MAX`.
/// A range whose end precedes its start counts as one slot (the smaller
/// budget), and no range counts as more than every slot.
pub fn memory_bytes_budget(
    slot_ranges: &[(u16, u16)],
    memory_bytes: u64,
) -> Option<(u64, MemoryBytesDerivation)> {
    let smallest_shard_slots = slot_ranges
        .iter()
        .map(|&(start, end)| (u64::from(end.saturating_sub(start)) + 1).min(CLUSTER_SLOT_COUNT))
        .min()?;
    let per_shard = u128::from(memory_bytes)
        * u128::from(STORE_TOTAL_USABLE_PERCENT)
        * u128::from(smallest_shard_slots)
        / (100 * u128::from(CLUSTER_SLOT_COUNT));
    let per_shard_bytes = u64::try_from(per_shard).unwrap_or(u64::MAX);
    let bytes = (per_shard_bytes / EXT_RELEASE_BUDGET_MEMORY_DIVISOR).max(1);
    Some((
        bytes,
        MemoryBytesDerivation {
            total_bytes: memory_bytes,
            shards: slot_ranges.len(),
            smallest_shard_slots,
            per_shard_bytes,
        },
    ))
}

/// Rules 2 and 3 for one reading: the smallest `maxmemory` ÷ 8 when every
/// primary reports one above zero (at least 1 byte); else, when
/// `memory_bytes` is set and the map is known, [`memory_bytes_budget`] over
/// its slot ranges — whatever `INFO` answered, a failure included; else
/// `None` — no map, an empty map, or neither rule applies.
pub fn budget_from_reading(
    reading: &StoreMemoryReading,
    memory_bytes: Option<u64>,
) -> Option<ExtReleaseBudget> {
    if let Ok((min_bytes, primaries)) = reading.maxmemory_limit() {
        return Some(ExtReleaseBudget {
            bytes: (min_bytes / EXT_RELEASE_BUDGET_MEMORY_DIVISOR).max(1),
            source: ExtReleaseBudgetSource::StoreInfo {
                min_maxmemory_bytes: min_bytes,
                primaries,
            },
        });
    }
    let StoreMemoryReading::Read {
        slot_ranges,
        endpoints,
        ..
    } = reading
    else {
        return None;
    };
    let (bytes, derivation) = memory_bytes_budget(slot_ranges, memory_bytes?)?;
    Some(ExtReleaseBudget {
        bytes,
        source: ExtReleaseBudgetSource::ValkeyMemory {
            derivation,
            endpoints: *endpoints,
        },
    })
}

/// Which arm of `live_edit_store_memory_used_ratio` a reading takes
/// ([`store_memory_gauge`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreMemoryGaugeArm {
    /// Every primary reports a `maxmemory` above zero: the highest
    /// `used_memory ÷ maxmemory`.
    Maxmemory,
    /// No primary reports a `maxmemory` field, `LIVE_EDIT_VALKEY_MEMORY_BYTES`
    /// is set, and the copies are known.
    StoreTotal {
        /// [`store_copies`].
        copies: u64,
    },
    /// As [`Self::StoreTotal`], but the copies are not known yet: a replica's
    /// first read, the read after the store's statistics were reset, or an
    /// unreadable `INFO commandstats`. The series is absent.
    StoreTotalPending,
    /// No arm applies: the series is absent.
    Absent,
}

/// `live_edit_store_memory_used_ratio` for a reading: its arm, and how full
/// the store is, or `None` (the series is absent) when no figure applies.
///
/// - Every primary reports a `maxmemory` above zero: the highest
///   `used_memory ÷ maxmemory` across primaries.
/// - Otherwise, when no primary reports a `maxmemory` field at all and
///   `memory_bytes` (`LIVE_EDIT_VALKEY_MEMORY_BYTES`) is set: the highest
///   whole-database `used_memory` any primary reports ÷ `copies`
///   ([`store_copies`]; `None` leaves the figure absent) ÷
///   [`store_usable_bytes`] of `memory_bytes`.
/// - `None` for anything else: `maxmemory:0`, a malformed value or a mix, a
///   primary that reported no `used_memory`, no primary read, or a failed
///   read.
///
/// Not clamped: above 1 means the store is past its limit.
#[allow(clippy::cast_precision_loss)] // byte counts as a ratio; f64 is ample
pub fn store_memory_gauge(
    reading: &StoreMemoryReading,
    memory_bytes: Option<u64>,
    copies: Option<u64>,
) -> (StoreMemoryGaugeArm, Option<f64>) {
    let Some(primaries) = reading.primaries() else {
        return (StoreMemoryGaugeArm::Absent, None);
    };
    if reading.maxmemory_limit().is_ok() {
        let mut highest: Option<f64> = None;
        for p in primaries {
            let (Some(used), MaxmemoryField::Value(limit)) = (p.used_memory, p.maxmemory) else {
                return (StoreMemoryGaugeArm::Maxmemory, None);
            };
            let ratio = used as f64 / limit as f64;
            highest = Some(highest.map_or(ratio, |h| h.max(ratio)));
        }
        return (StoreMemoryGaugeArm::Maxmemory, highest);
    }
    let Some(total) = memory_bytes.filter(|_| reading.reports_no_maxmemory_field()) else {
        return (StoreMemoryGaugeArm::Absent, None);
    };
    let Some(copies) = copies else {
        return (StoreMemoryGaugeArm::StoreTotalPending, None);
    };
    let arm = StoreMemoryGaugeArm::StoreTotal { copies };
    let usable = store_usable_bytes(total);
    let used = primaries
        .iter()
        .map(|p| p.used_memory)
        .collect::<Option<Vec<u64>>>()
        .and_then(|used| used.into_iter().max());
    match used {
        Some(used) if usable > 0 && copies > 0 => {
            (arm, Some(used as f64 / copies as f64 / usable as f64))
        }
        _ => (arm, None),
    }
}

/// Whether the store's memory may set the release budget: true unless
/// `Config::ext_release_budget` answered an explicit
/// `LIVE_EDIT_EXT_RELEASE_MAX_BYTES` (it answers that or the fallback).
pub fn store_sets_ext_release_budget(configured: &ExtReleaseBudget) -> bool {
    !matches!(configured.source, ExtReleaseBudgetSource::Explicit)
}

/// What startup's store read resolves to ([`resolve_startup`]).
#[derive(Debug, Clone, PartialEq)]
pub struct StoreMemoryResolution {
    /// The release budget to start with.
    pub budget: ExtReleaseBudget,
    /// Why the store's own `maxmemory` was not usable, if it was not: the
    /// fallback line names it, and the refresh task is seeded with its kind.
    pub problem: Option<StoreMemoryProblem>,
    /// The gauge's arm at startup ([`store_memory_gauge`]).
    pub gauge_arm: StoreMemoryGaugeArm,
    /// `live_edit_store_memory_used_ratio` at startup; always `None` on the
    /// store-total arm, whose copies need a second read.
    pub used_ratio: Option<f64>,
    /// The replica acknowledgement count startup read, the refresh task's
    /// first `previous` for [`store_copies`].
    pub replconf_calls: Option<u64>,
    /// Why `INFO commandstats` could not be read at startup, if it could not.
    pub commandstats_unreadable: Option<String>,
    /// Why startup's `CLUSTER SLOTS` gave no map, if it did not.
    pub map_problem: Option<ClusterMapProblem>,
    /// Whether the refresh task should run ([`store_memory_refresh_needed`]).
    pub refresh_needed: bool,
}

impl StoreMemoryResolution {
    /// What the refresh task starts from: what startup logged and read.
    pub fn seed(&self) -> RefreshSeed {
        RefreshSeed {
            problem: self.problem.as_ref().map(StoreMemoryProblem::kind),
            gauge_arm: Some(self.gauge_arm),
            replconf_calls: self.replconf_calls,
            map_problem: self.map_problem.as_ref().map(ClusterMapProblem::kind),
            commandstats_unreadable: self.commandstats_unreadable.is_some(),
        }
    }
}

/// Whether a later read can serve either consumer, decided once from
/// startup's read: the budget, unless it is explicit; the fullness gauge,
/// when `memory_bytes` (`LIVE_EDIT_VALKEY_MEMORY_BYTES`) is set or the store
/// may report its own limit. Only a store that answered with no `maxmemory`
/// field on any primary — a managed store that does not expose it, which does
/// not start exposing it while it runs — rules the gauge out; a failed read,
/// a `maxmemory` of 0 (a limit can be set in place) or a malformed one leaves
/// it open, so the refresh runs.
pub fn store_memory_refresh_needed(
    configured: &ExtReleaseBudget,
    reading: &StoreMemoryReading,
    memory_bytes: Option<u64>,
) -> bool {
    store_sets_ext_release_budget(configured)
        || memory_bytes.is_some()
        || !reading.reports_no_maxmemory_field()
}

/// Startup's resolution of `configured` (`Config::ext_release_budget`: the
/// explicit budget, or the fallback) against one read: an explicit budget
/// stands; otherwise [`budget_from_reading`] (rule 3 from the map alone when
/// `INFO` failed), else the configured fallback (no map — `CLUSTER SLOTS`
/// failed or malformed included — or nothing usable and `memory_bytes`
/// unset). The problem, the gauge and whether the refresh
/// runs are the read's, whatever the budget; the gauge's copies are unknown,
/// so its store-total arm is pending.
pub fn resolve_startup(
    configured: ExtReleaseBudget,
    read: &StoreMemoryRead,
    memory_bytes: Option<u64>,
) -> StoreMemoryResolution {
    let reading = &read.reading;
    let budget = if store_sets_ext_release_budget(&configured) {
        budget_from_reading(reading, memory_bytes).unwrap_or(configured)
    } else {
        configured
    };
    let (gauge_arm, used_ratio) = store_memory_gauge(reading, memory_bytes, None);
    let (replconf_calls, commandstats_unreadable) = match reading {
        StoreMemoryReading::Read {
            replconf_calls: ReplconfCalls::Calls(n),
            ..
        } => (Some(*n), None),
        StoreMemoryReading::Read {
            replconf_calls: ReplconfCalls::Unreadable(why),
            ..
        } => (None, Some(why.clone())),
        _ => (None, None),
    };
    StoreMemoryResolution {
        budget,
        problem: StoreMemoryProblem::of_reading(reading),
        gauge_arm,
        used_ratio,
        replconf_calls,
        commandstats_unreadable,
        map_problem: read.map_problem.clone(),
        refresh_needed: store_memory_refresh_needed(&configured, reading, memory_bytes),
    }
}

/// Startup's store read, for every configuration: one [`read_store_memory`],
/// resolved by [`resolve_startup`].
pub async fn startup_store_memory(
    client: &ValkeyClient,
    configured: ExtReleaseBudget,
    memory_bytes: Option<u64>,
) -> StoreMemoryResolution {
    let read = read_store_memory(client, memory_bytes).await;
    resolve_startup(configured, &read, memory_bytes)
}

/// The budget one refresh moves the gate to, or `None` to keep `current`
/// (b.yuo, b.qhk): nothing for an explicit `current`; nothing when `INFO
/// memory` failed while a rule-2 (`info`) budget is in force, which it keeps;
/// else rule 2 or rule 3 ([`budget_from_reading`]) when it differs from
/// `current` — in bytes or in source, so a switch between `info` and
/// `override`, or a changed shard count, moves it; nothing for a reading of
/// no map, no primary or an empty map, or one neither rule answers, which
/// keep the last good value — never a step down to the fallback, and never
/// an "unlimited".
pub fn refreshed_budget(
    current: &ExtReleaseBudget,
    reading: &StoreMemoryReading,
    memory_bytes: Option<u64>,
) -> Option<ExtReleaseBudget> {
    match (current.source, reading) {
        (ExtReleaseBudgetSource::Explicit, _) => return None,
        (
            ExtReleaseBudgetSource::StoreInfo { .. },
            StoreMemoryReading::Read { info: Err(_), .. },
        ) => return None,
        _ => {}
    }
    let next = budget_from_reading(reading, memory_bytes)?;
    (next != *current).then_some(next)
}

/// What a store-memory problem leaves the release budget as, for its log
/// line, from the budget in force after the read.
fn budget_consequence(budget: &ExtReleaseBudget) -> &'static str {
    match budget.source {
        ExtReleaseBudgetSource::Explicit => {
            "the external-value release budget is LIVE_EDIT_EXT_RELEASE_MAX_BYTES"
        }
        ExtReleaseBudgetSource::ValkeyMemory { .. } => {
            "the external-value release budget derives from LIVE_EDIT_VALKEY_MEMORY_BYTES and \
             the cluster map"
        }
        ExtReleaseBudgetSource::StoreInfo { .. } | ExtReleaseBudgetSource::Fallback => {
            "keeping the last good external-value release budget"
        }
    }
}

/// What a reading leaves `live_edit_store_memory_used_ratio` as, for a log
/// line.
fn gauge_consequence(arm: StoreMemoryGaugeArm, memory_bytes_set: bool) -> &'static str {
    match (arm, memory_bytes_set) {
        (StoreMemoryGaugeArm::Maxmemory, _) => {
            "live_edit_store_memory_used_ratio is the highest used_memory / maxmemory across \
             primaries"
        }
        (StoreMemoryGaugeArm::StoreTotal { .. }, _) => {
            "live_edit_store_memory_used_ratio derives from LIVE_EDIT_VALKEY_MEMORY_BYTES"
        }
        (StoreMemoryGaugeArm::StoreTotalPending, _) => {
            "live_edit_store_memory_used_ratio derives from LIVE_EDIT_VALKEY_MEMORY_BYTES; its \
             value comes once two successive reads of the store's replica acknowledgement \
             count (cmdstat_replconf) can be compared, normally at the next read"
        }
        (StoreMemoryGaugeArm::Absent, true) => {
            "live_edit_store_memory_used_ratio is absent until a read gives one"
        }
        (StoreMemoryGaugeArm::Absent, false) => {
            "live_edit_store_memory_used_ratio is absent: the store reports no usable maxmemory \
             and LIVE_EDIT_VALKEY_MEMORY_BYTES is unset"
        }
    }
}

/// Log the first of a kind of store-memory problem at its level: WARN for a
/// fault; for a store with no `maxmemory` field, INFO when
/// `LIVE_EDIT_VALKEY_MEMORY_BYTES` is unset and nothing when it is set (it
/// answers both consumers). Each line says what the budget in force
/// (`budget`) and the fullness gauge now are.
fn log_store_memory_problem(
    problem: &StoreMemoryProblem,
    budget: &ExtReleaseBudget,
    arm: StoreMemoryGaugeArm,
    memory_bytes_set: bool,
) {
    let budget_effect = budget_consequence(budget);
    let gauge_effect = gauge_consequence(arm, memory_bytes_set);
    if problem.kind().is_fault() {
        tracing::warn!(
            problem = %problem,
            ext_release_max_bytes = budget.bytes,
            source = budget.source.label(),
            "store memory read gave nothing usable: {budget_effect}; {gauge_effect}"
        );
    } else if !memory_bytes_set {
        tracing::info!(
            problem = %problem,
            ext_release_max_bytes = budget.bytes,
            source = budget.source.label(),
            "store memory read: the store does not report its size (expected on a managed \
             store such as Azure Managed Redis): {budget_effect}; {gauge_effect}. Set \
             LIVE_EDIT_VALKEY_MEMORY_BYTES to the store's total memory, as the provider states \
             it, for the fullness gauge and, unless LIVE_EDIT_EXT_RELEASE_MAX_BYTES is set, \
             the release budget"
        );
    }
}

/// Log a pass's `CLUSTER SLOTS` that gave no map (WARN), by kind: the read
/// failed, or the reply was malformed. Either way the pass has no map, so a
/// budget derived from `LIVE_EDIT_VALKEY_MEMORY_BYTES` is not re-derived.
fn log_cluster_map_problem(problem: &ClusterMapProblem) {
    match problem {
        ClusterMapProblem::ReadFailed(_) => tracing::warn!(
            problem = %problem,
            "store memory read: the cluster map read failed; this pass has no map, so the \
             release budget derived from LIVE_EDIT_VALKEY_MEMORY_BYTES is not re-derived (the \
             last good budget stays; at startup, the fallback unless the store reports \
             maxmemory)"
        ),
        ClusterMapProblem::Malformed => tracing::warn!(
            problem = %problem,
            "store memory read: the cluster map was malformed; this pass has no map, so the \
             release budget derived from LIVE_EDIT_VALKEY_MEMORY_BYTES is not re-derived (the \
             last good budget stays; at startup, the fallback unless the store reports \
             maxmemory)"
        ),
    }
}

/// Log an unreadable `INFO commandstats` (WARN): the store-total arm of the
/// fullness gauge is absent until it reads again.
fn log_commandstats_unreadable(why: &str) {
    tracing::warn!(
        problem = %why,
        "store memory read: the store's replica acknowledgement count (cmdstat_replconf in INFO \
         commandstats) could not be read; live_edit_store_memory_used_ratio is absent until it \
         can"
    );
}

/// Log startup's store-memory read, once per process: one INFO line naming
/// the fullness gauge's arm and value, or why it is absent; WARN for a
/// `CLUSTER SLOTS` that gave no map or an unreadable `INFO commandstats`; and,
/// when the
/// budget is not the fallback (whose own startup line,
/// `Config::log_ext_release_budget_startup`, names its reason), the
/// maxmemory problem at the level the refresh task would log it first. The
/// refresh task is seeded with what this logged
/// ([`StoreMemoryResolution::seed`]).
pub fn log_store_memory_startup(resolution: &StoreMemoryResolution, memory_bytes: Option<u64>) {
    let memory_bytes_set = memory_bytes.is_some();
    tracing::info!(
        gauge_arm = ?resolution.gauge_arm,
        store_memory_used_ratio = ?resolution.used_ratio,
        valkey_memory_bytes = memory_bytes,
        refresh = resolution.refresh_needed,
        "store memory read at startup: {}",
        gauge_consequence(resolution.gauge_arm, memory_bytes_set)
    );
    if let Some(problem) = &resolution.map_problem {
        log_cluster_map_problem(problem);
    }
    if let Some(why) = &resolution.commandstats_unreadable {
        log_commandstats_unreadable(why);
    }
    if matches!(resolution.budget.source, ExtReleaseBudgetSource::Fallback) {
        return;
    }
    if let Some(problem) = &resolution.problem {
        log_store_memory_problem(
            problem,
            &resolution.budget,
            resolution.gauge_arm,
            memory_bytes_set,
        );
    }
}

/// What the refresh task remembers from startup's read
/// ([`StoreMemoryResolution::seed`]), so it logs nothing startup already
/// logged and its first copies compare against startup's count. The default
/// remembers nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RefreshSeed {
    /// The kind of the maxmemory problem startup logged, if any (b.1jt).
    pub problem: Option<StoreMemoryProblemKind>,
    /// The gauge's arm at startup.
    pub gauge_arm: Option<StoreMemoryGaugeArm>,
    /// The replica acknowledgement count startup read.
    pub replconf_calls: Option<u64>,
    /// The kind of `CLUSTER SLOTS` problem startup logged, if any.
    pub map_problem: Option<ClusterMapProblemKind>,
    /// Startup's `INFO commandstats` was unreadable (and was logged).
    pub commandstats_unreadable: bool,
}

/// What the refresh task starts from ([`spawn_store_memory_refresh`]).
pub struct StoreMemoryRefresh {
    /// The upload gate whose budget the task moves unless the budget is
    /// explicit; `None` when it is disabled (`LIVE_EDIT_EXT_RELEASE_MAX_BYTES=0`).
    pub gate: Option<Arc<ExtReleaseGate>>,
    /// The budget startup resolved ([`StoreMemoryResolution::budget`]).
    pub initial: ExtReleaseBudget,
    /// `LIVE_EDIT_VALKEY_MEMORY_BYTES`.
    pub memory_bytes: Option<u64>,
    /// What startup logged and read.
    pub seed: RefreshSeed,
    /// Where the budget and the fullness gauge are recorded.
    pub metrics: Option<SharedMetrics>,
}

/// A test-only replacement for the refresh task's store read: each call is
/// one read. See [`spawn_store_memory_refresh_with_reader_for_test`].
#[cfg(any(test, feature = "test-support"))]
pub type StoreMemoryReaderForTest =
    Arc<dyn Fn() -> futures::future::BoxFuture<'static, StoreMemoryRead> + Send + Sync>;

/// Where the refresh task reads the store's memory from.
// One per replica, built once at startup, so the size gap between the store
// variant and the test-only reader costs nothing worth a box on the
// production variant.
#[allow(clippy::large_enum_variant)]
enum ReadSource {
    /// [`read_store_memory`] on `client`: the only source in production.
    Store(ValkeyClient),
    /// A test's reader ([`spawn_store_memory_refresh_with_reader_for_test`]).
    #[cfg(any(test, feature = "test-support"))]
    Injected(StoreMemoryReaderForTest),
}

impl ReadSource {
    async fn read(&self, memory_bytes: Option<u64>) -> StoreMemoryRead {
        match self {
            ReadSource::Store(client) => read_store_memory(client, memory_bytes).await,
            #[cfg(any(test, feature = "test-support"))]
            ReadSource::Injected(reader) => reader().await,
        }
    }
}

/// Re-read the store's memory every [`STORE_MEMORY_REFRESH_INTERVAL`] until
/// `cancel` fires. Spawn it only when startup's read says a consumer can use
/// it ([`StoreMemoryResolution::refresh_needed`]). Every pass records
/// `live_edit_store_memory_used_ratio` ([`store_memory_gauge`]; `None` makes
/// the series absent), its copies from the previous good replica
/// acknowledgement count ([`store_copies`]). Unless the budget is explicit,
/// and when a gate exists, a pass moves the gate's budget by
/// [`refreshed_budget`], starting from `refresh.initial`; it never sets it
/// to 0.
///
/// Logs: INFO on every change of the budget (recorded on
/// `live_edit_ext_release_budget_bytes`), naming for a budget derived from
/// `LIVE_EDIT_VALKEY_MEMORY_BYTES` the shards, the endpoints, the per-shard
/// bytes and the total; INFO on every change of the gauge's arm or copies. A
/// reading whose `maxmemory` is unusable is logged only when its kind
/// ([`StoreMemoryProblemKind`]) differs from the last one logged — so a store
/// that keeps answering the same way is logged once, not every minute — at
/// that kind's level (see the module docs), and at DEBUG on its repeats; a
/// usable reading resets the last kind. A `CLUSTER SLOTS` that failed, one
/// that was malformed, and an unreadable `INFO commandstats` WARN once per
/// kind until they succeed again.
pub fn spawn_store_memory_refresh(
    client: ValkeyClient,
    refresh: StoreMemoryRefresh,
    cancel: CancellationToken,
) -> JoinHandle<()> {
    spawn_refresh_every(
        ReadSource::Store(client),
        refresh,
        cancel,
        STORE_MEMORY_REFRESH_INTERVAL,
    )
}

/// [`spawn_store_memory_refresh`] re-reading every `every` instead of
/// [`STORE_MEMORY_REFRESH_INTERVAL`], so a test can watch several refreshes
/// in milliseconds. **Test-only** (`test-support`).
#[cfg(any(test, feature = "test-support"))]
pub fn spawn_store_memory_refresh_every_for_test(
    client: ValkeyClient,
    refresh: StoreMemoryRefresh,
    cancel: CancellationToken,
    every: Duration,
) -> JoinHandle<()> {
    spawn_refresh_every(ReadSource::Store(client), refresh, cancel, every)
}

/// [`spawn_store_memory_refresh`] re-reading every `every`, with each read a
/// call of `reader` instead of [`read_store_memory`], so a test can script
/// every read — a limit, a store without `maxmemory`, a changed cluster map,
/// a moving acknowledgement count, a cluster map that failed or was
/// malformed, a failure — with no
/// store or proxy. **Test-only** (`test-support`), mirroring
/// `ExtReleaseGate::with_total_reader_for_test`.
#[cfg(any(test, feature = "test-support"))]
pub fn spawn_store_memory_refresh_with_reader_for_test(
    reader: StoreMemoryReaderForTest,
    refresh: StoreMemoryRefresh,
    cancel: CancellationToken,
    every: Duration,
) -> JoinHandle<()> {
    spawn_refresh_every(ReadSource::Injected(reader), refresh, cancel, every)
}

fn spawn_refresh_every(
    source: ReadSource,
    refresh: StoreMemoryRefresh,
    cancel: CancellationToken,
    every: Duration,
) -> JoinHandle<()> {
    let StoreMemoryRefresh {
        gate,
        initial,
        memory_bytes,
        seed,
        metrics,
    } = refresh;
    tokio::spawn(async move {
        let mut current = initial;
        let mut last_problem = seed.problem;
        let mut last_arm = seed.gauge_arm;
        let mut last_replconf = seed.replconf_calls;
        let mut last_map_problem = seed.map_problem;
        let mut commandstats_unreadable = seed.commandstats_unreadable;
        let mut tick = tokio::time::interval_at(tokio::time::Instant::now() + every, every);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                biased;
                () = cancel.cancelled() => return,
                _ = tick.tick() => {}
            }
            let StoreMemoryRead {
                reading,
                map_problem,
            } = source.read(memory_bytes).await;
            // Once per kind of map problem, until a pass gets a map again.
            match &map_problem {
                Some(problem) if last_map_problem != Some(problem.kind()) => {
                    log_cluster_map_problem(problem);
                }
                Some(problem) => tracing::debug!(
                    problem = %problem,
                    "store memory read: the cluster map gave no map again"
                ),
                None => {}
            }
            last_map_problem = map_problem.as_ref().map(ClusterMapProblem::kind);

            // The copies: this pass's acknowledgement count against the last
            // good one, which a fall re-baselines and a failure keeps.
            let copies = match &reading {
                StoreMemoryReading::Read {
                    replconf_calls: ReplconfCalls::Calls(calls),
                    ..
                } => {
                    commandstats_unreadable = false;
                    let copies = store_copies(last_replconf, *calls);
                    last_replconf = Some(*calls);
                    copies
                }
                StoreMemoryReading::Read {
                    replconf_calls: ReplconfCalls::Unreadable(why),
                    ..
                } => {
                    if !commandstats_unreadable {
                        log_commandstats_unreadable(why);
                    }
                    commandstats_unreadable = true;
                    None
                }
                _ => None,
            };
            let (arm, used_ratio) = store_memory_gauge(&reading, memory_bytes, copies);
            if let Some(metrics) = &metrics {
                metrics.record_store_memory_used_ratio(used_ratio);
            }
            if last_arm != Some(arm) {
                tracing::info!(
                    gauge_arm = ?arm,
                    previous_gauge_arm = ?last_arm,
                    store_memory_used_ratio = ?used_ratio,
                    "store memory re-read: {}",
                    gauge_consequence(arm, memory_bytes.is_some())
                );
                last_arm = Some(arm);
            }

            if let Some(gate) = &gate {
                if let Some(next) = refreshed_budget(&current, &reading, memory_bytes) {
                    gate.set_budget_bytes(next.bytes);
                    if let Some(metrics) = &metrics {
                        metrics.record_ext_release_budget(next.source.label(), next.bytes);
                    }
                    let map = next.source.cluster_map();
                    tracing::info!(
                        ext_release_max_bytes = next.bytes,
                        previous_bytes = current.bytes,
                        source = next.source.label(),
                        previous_source = current.source.label(),
                        shards = map.map(|(d, _)| d.shards),
                        endpoints = map.map(|(_, endpoints)| endpoints),
                        per_shard_bytes = map.map(|(d, _)| d.per_shard_bytes),
                        total_bytes = map.map(|(d, _)| d.total_bytes),
                        derived_from = ?next.source,
                        "external-value release budget re-read from the store"
                    );
                    current = next;
                }
            }

            // One kind per reason (and one for every failed read), so neither
            // a primary that changes nor a flapping error text logs again.
            let Some(problem) = StoreMemoryProblem::of_reading(&reading) else {
                last_problem = None;
                continue;
            };
            let kind = problem.kind();
            if last_problem == Some(kind) {
                tracing::debug!(
                    problem = %problem,
                    ext_release_max_bytes = current.bytes,
                    source = current.source.label(),
                    "store memory re-read gave nothing usable again: {}",
                    budget_consequence(&current)
                );
            } else {
                log_store_memory_problem(&problem, &current, arm, memory_bytes.is_some());
                last_problem = Some(kind);
            }
        }
    })
}

#[cfg(test)]
mod tests {
    //! The pure halves of the store reading — the `INFO` parses, the copies
    //! rule, the budget a reading gives at startup and on a refresh, the
    //! fullness gauge's arms, whether the refresh runs, the startup line —
    //! and the refresh task itself, driven through its reader seam
    //! ([`spawn_store_memory_refresh_with_reader_for_test`]) on a paused
    //! clock, so each pass is one scripted reading. The reads against a store
    //! (standalone, staged replies, a staged cluster map, a real cluster) are
    //! in `tests/integration/storage/b_yuo_store_memory_tests.rs`.

    use super::*;

    /// A fallback budget, `8 × 100 MiB` (any value serves: these tests only
    /// check that the fallback is kept or replaced).
    const FALLBACK: u64 = 8 * 104_857_600;

    /// 3 GiB, Valkey's `maxmemory 3072mb`.
    const THREE_GIB: u64 = 3_221_225_472;

    /// One primary, `name`, reporting `maxmemory` and `used_memory:1`.
    fn primary(name: &str, maxmemory: MaxmemoryField) -> PrimaryMemory {
        PrimaryMemory {
            primary: name.to_owned(),
            maxmemory,
            used_memory: Some(1),
        }
    }

    /// A reading of `primaries`, over a standalone map.
    fn read(primaries: Vec<PrimaryMemory>) -> StoreMemoryReading {
        StoreMemoryReading::Read {
            slot_ranges: vec![STANDALONE_SLOT_RANGE],
            endpoints: primaries.len(),
            info: Ok(primaries),
            replconf_calls: ReplconfCalls::NotRead,
        }
    }

    /// `primaries` primaries, each reporting `maxmemory:{min_bytes}`.
    fn limit(min_bytes: u64, primaries: usize) -> StoreMemoryReading {
        read(
            (0..primaries)
                .map(|i| {
                    primary(
                        &format!("primary 10.0.0.{i}:6379"),
                        MaxmemoryField::Value(min_bytes),
                    )
                })
                .collect(),
        )
    }

    /// One standalone store answering `maxmemory`.
    fn one(maxmemory: MaxmemoryField) -> StoreMemoryReading {
        read(vec![primary("the store", maxmemory)])
    }

    /// A read whose `CLUSTER SLOTS` raised no problem.
    fn no_map_problem(reading: StoreMemoryReading) -> StoreMemoryRead {
        StoreMemoryRead {
            reading,
            map_problem: None,
        }
    }

    /// A standalone read whose `INFO memory` failed with `why`.
    fn failed(why: &str) -> StoreMemoryReading {
        StoreMemoryReading::Read {
            slot_ranges: vec![STANDALONE_SLOT_RANGE],
            endpoints: 1,
            info: Err(why.to_owned()),
            replconf_calls: ReplconfCalls::NotRead,
        }
    }

    fn from_store(min_maxmemory_bytes: u64, primaries: usize) -> ExtReleaseBudget {
        ExtReleaseBudget {
            bytes: min_maxmemory_bytes / EXT_RELEASE_BUDGET_MEMORY_DIVISOR,
            source: ExtReleaseBudgetSource::StoreInfo {
                min_maxmemory_bytes,
                primaries,
            },
        }
    }

    fn fallback() -> ExtReleaseBudget {
        ExtReleaseBudget {
            bytes: FALLBACK,
            source: ExtReleaseBudgetSource::Fallback,
        }
    }

    // ── parse_info_maxmemory ────────────────────────────────────────────────

    #[test]
    fn parse_info_maxmemory_reads_the_maxmemory_line_of_a_real_reply() {
        let reply = "# Memory\r\nused_memory:34883416\r\nused_memory_human:33.27M\r\n\
                     maxmemory:3221225472\r\nmaxmemory_human:3.00G\r\n\
                     maxmemory_policy:noeviction\r\n";
        assert_eq!(
            parse_info_maxmemory(reply),
            MaxmemoryField::Value(THREE_GIB)
        );
    }

    #[test]
    fn parse_info_maxmemory_reads_zero_as_zero_not_as_unlimited() {
        assert_eq!(
            parse_info_maxmemory("# Memory\r\nmaxmemory:0\r\nmaxmemory_human:0B\r\n"),
            MaxmemoryField::Value(0)
        );
    }

    /// `maxmemory_human:` and `maxmemory_policy:` share the prefix
    /// `maxmemory`; only the exact `maxmemory:` field counts, wherever it
    /// sits, and a reply with only its neighbours has none.
    #[test]
    fn parse_info_maxmemory_ignores_its_neighbouring_fields() {
        assert_eq!(
            parse_info_maxmemory("maxmemory_human:3.00G\r\nmaxmemory_policy:noeviction\r\n"),
            MaxmemoryField::Absent
        );
        assert_eq!(
            parse_info_maxmemory("maxmemory_policy:noeviction\nmaxmemory:42\n"),
            MaxmemoryField::Value(42),
            "a LF-only reply parses too"
        );
    }

    /// A store that does not expose the field (Azure Managed Redis), or an
    /// empty reply, is `Absent`.
    #[test]
    fn parse_info_maxmemory_without_the_field_is_absent() {
        assert_eq!(
            parse_info_maxmemory("# Memory\r\nused_memory:1\r\n"),
            MaxmemoryField::Absent
        );
        assert_eq!(parse_info_maxmemory(""), MaxmemoryField::Absent);
    }

    /// Anything but a non-negative integer that fits `u64` is `Malformed`;
    /// whitespace around the number is tolerated.
    #[test]
    fn parse_info_maxmemory_refuses_a_value_that_is_not_a_byte_count() {
        for value in ["", "-1", "3gb", "3.5", "18446744073709551616"] {
            assert_eq!(
                parse_info_maxmemory(&format!("maxmemory:{value}\r\n")),
                MaxmemoryField::Malformed,
                "maxmemory:{value:?}"
            );
        }
        assert_eq!(
            parse_info_maxmemory("maxmemory: 1024 \r\n"),
            MaxmemoryField::Value(1024)
        );
        assert_eq!(
            parse_info_maxmemory("maxmemory:18446744073709551615\r\n"),
            MaxmemoryField::Value(u64::MAX)
        );
    }

    // ── budget_from_reading / resolve_startup ───────────────────────────────

    /// Every primary above zero: the smallest `maxmemory` ÷ 8, with the
    /// `info` source and no problem.
    #[test]
    fn a_store_limit_gives_one_eighth_of_the_smallest_primary() {
        let resolution = resolve_startup(fallback(), &no_map_problem(limit(THREE_GIB, 3)), None);
        assert_eq!(resolution.budget, from_store(THREE_GIB, 3));
        assert_eq!(resolution.budget.bytes, 402_653_184);
        assert_eq!(resolution.budget.source.label(), "info");
        assert_eq!(resolution.problem, None);
    }

    /// A `maxmemory` below 8 bytes still gives a budget of 1, never the 0
    /// that would disable the gate.
    #[test]
    fn a_tiny_store_limit_never_reads_as_a_disabled_budget() {
        let resolution = resolve_startup(fallback(), &no_map_problem(limit(7, 1)), None);
        assert_eq!(resolution.budget.bytes, 1);
    }

    /// Only a missing `maxmemory` field is the expected case; every other
    /// reason a reading is unusable, and a failed read, is a fault (logged
    /// WARN), each its own kind (b.1jt). The kind drops the primary, so a
    /// fault that moves between primaries is still one kind and is not
    /// logged again.
    #[test]
    fn only_an_absent_maxmemory_field_is_not_a_fault() {
        let zero = |primary: &str| UnusableReason::ZeroLimit {
            primary: primary.to_owned(),
        };
        let kinds = [
            (
                StoreMemoryProblem::Unusable(UnusableReason::FieldAbsent {
                    primary: "the store".to_owned(),
                }),
                StoreMemoryProblemKind::FieldAbsent,
                false,
            ),
            (
                StoreMemoryProblem::Unusable(zero("primary 10.0.0.2:6379")),
                StoreMemoryProblemKind::ZeroLimit,
                true,
            ),
            (
                StoreMemoryProblem::Unusable(UnusableReason::Malformed {
                    primary: "the store".to_owned(),
                }),
                StoreMemoryProblemKind::Malformed,
                true,
            ),
            (
                StoreMemoryProblem::Unusable(UnusableReason::RoutingUnknown),
                StoreMemoryProblemKind::RoutingUnknown,
                true,
            ),
            (
                StoreMemoryProblem::Unusable(UnusableReason::NoPrimary),
                StoreMemoryProblemKind::NoPrimary,
                true,
            ),
            (
                StoreMemoryProblem::ReadFailed("NOPERM".to_owned()),
                StoreMemoryProblemKind::ReadFailed,
                true,
            ),
        ];
        for (problem, kind, fault) in kinds {
            assert_eq!(problem.kind(), kind, "{problem}");
            assert_eq!(kind.is_fault(), fault, "{problem}");
        }
        assert_eq!(
            StoreMemoryProblem::Unusable(zero("primary 10.0.0.3:6379")).kind(),
            StoreMemoryProblem::Unusable(zero("primary 10.0.0.2:6379")).kind(),
            "the kind does not name the primary"
        );
    }

    /// With `LIVE_EDIT_VALKEY_MEMORY_BYTES` unset, an unusable reading (a
    /// primary at 0) and a failed read both keep the fallback, with the
    /// problem the startup line names.
    #[test]
    fn an_unusable_or_failed_reading_falls_back_with_its_reason() {
        let resolution = resolve_startup(
            fallback(),
            &no_map_problem(one(MaxmemoryField::Value(0))),
            None,
        );
        assert_eq!(resolution.budget, fallback());
        assert_eq!(resolution.budget.source.label(), "fallback");
        assert_eq!(
            resolution.problem.map(|r| r.to_string()).as_deref(),
            Some("the store reports maxmemory 0 (no memory limit configured)")
        );

        let resolution = resolve_startup(
            fallback(),
            &no_map_problem(failed("connection refused")),
            None,
        );
        assert_eq!(resolution.budget, fallback());
        let reason = resolution
            .problem
            .expect("a failed read names why")
            .to_string();
        assert!(
            reason.starts_with("the INFO memory read failed: ")
                && reason.contains("connection refused"),
            "{reason}"
        );
    }

    // ── refreshed_budget ─────────────────────────────────────────────────────

    /// A reading that changes the budget moves it, up or down, and so does
    /// the first good reading after the fallback.
    #[test]
    fn a_refresh_follows_a_changed_store_limit() {
        let current = from_store(THREE_GIB, 3);
        assert_eq!(
            refreshed_budget(&current, &limit(2 * THREE_GIB, 3), None),
            Some(from_store(2 * THREE_GIB, 3)),
            "a grown store raises the budget"
        );
        assert_eq!(
            refreshed_budget(&current, &limit(THREE_GIB / 2, 3), None),
            Some(from_store(THREE_GIB / 2, 3)),
            "a shrunk store lowers it"
        );
        assert_eq!(
            refreshed_budget(&fallback(), &limit(THREE_GIB, 1), None),
            Some(from_store(THREE_GIB, 1)),
            "a replica on the fallback takes the first good reading"
        );
        assert_eq!(
            refreshed_budget(&current, &limit(THREE_GIB, 3), None),
            None,
            "an unchanged store changes nothing"
        );
    }

    /// A reading that is unusable or failed keeps the last good budget — it
    /// never steps down to the fallback and never reads as unlimited — and a
    /// replica on the fallback stays there.
    #[test]
    fn a_refresh_with_nothing_usable_keeps_the_last_good_budget() {
        let current = from_store(THREE_GIB, 3);
        for reading in [
            one(MaxmemoryField::Value(0)),
            StoreMemoryReading::RoutingUnknown,
            failed("timed out"),
        ] {
            assert_eq!(
                refreshed_budget(&current, &reading, None),
                None,
                "{reading:?}"
            );
            assert_eq!(
                refreshed_budget(&fallback(), &reading, None),
                None,
                "{reading:?}"
            );
        }
    }

    // ── b.qhk fixtures ───────────────────────────────────────────────────────

    /// The 24 GB Compute Optimized Azure Managed Redis cache measured on
    /// 2026-10-13: `LIVE_EDIT_VALKEY_MEMORY_BYTES` as the provider states it.
    const X20_TOTAL: u64 = 24_000_000_000;

    /// That cache's distinct endpoints: 2 nodes, each fronting 4 shards.
    const X20_ENDPOINTS: usize = 2;

    /// That cache's `CLUSTER SLOTS`: 8 entries of 2,048 slots (interleaved
    /// across its 2 endpoints, which these ranges do not carry).
    fn x20_map() -> Vec<(u16, u16)> {
        (0..8u16).map(|i| (i * 2048, i * 2048 + 2047)).collect()
    }

    /// Dev's `Balanced_B10`: 12 GB.
    const DEV_TOTAL: u64 = 12_000_000_000;

    /// Dev's `CLUSTER SLOTS`: 2 entries of 8,192 slots, on 2 endpoints.
    fn dev_map() -> Vec<(u16, u16)> {
        vec![(0, 8191), (8192, 16383)]
    }

    /// Dev's whole-database `used_memory`, high availability on, as measured
    /// (the Contract's worked example).
    const DEV_USED: u64 = 470_333_232;

    /// An Azure Managed Redis reading: `endpoints` endpoints over
    /// `slot_ranges`, each reporting no `maxmemory` field and the
    /// whole-database `used`, with `replconf_calls`.
    fn amr(
        slot_ranges: Vec<(u16, u16)>,
        endpoints: usize,
        used: u64,
        replconf_calls: ReplconfCalls,
    ) -> StoreMemoryReading {
        StoreMemoryReading::Read {
            slot_ranges,
            endpoints,
            info: Ok((0..endpoints)
                .map(|i| PrimaryMemory {
                    primary: format!("primary 10.0.0.{i}:8500"),
                    maxmemory: MaxmemoryField::Absent,
                    used_memory: Some(used),
                })
                .collect()),
            replconf_calls,
        }
    }

    /// A reading over the X20 map whose primaries answered `primaries`, each
    /// a `(maxmemory, used_memory)` pair.
    fn over_x20(primaries: &[(MaxmemoryField, Option<u64>)]) -> StoreMemoryReading {
        StoreMemoryReading::Read {
            slot_ranges: x20_map(),
            endpoints: primaries.len(),
            info: Ok(primaries
                .iter()
                .enumerate()
                .map(|(i, &(maxmemory, used_memory))| PrimaryMemory {
                    primary: format!("primary 10.0.0.{i}:6379"),
                    maxmemory,
                    used_memory,
                })
                .collect()),
            replconf_calls: ReplconfCalls::NotRead,
        }
    }

    /// A read whose `INFO memory` failed with `why`, over `slot_ranges`.
    fn failed_over(
        slot_ranges: Vec<(u16, u16)>,
        endpoints: usize,
        why: &str,
    ) -> StoreMemoryReading {
        StoreMemoryReading::Read {
            slot_ranges,
            endpoints,
            info: Err(why.to_owned()),
            replconf_calls: ReplconfCalls::NotRead,
        }
    }

    /// The rule-3 budget `total` gives over `slot_ranges` listed by
    /// `endpoints` endpoints. Its arithmetic is pinned once, in
    /// [`memory_bytes_budget_is_one_eighth_of_the_smallest_entrys_share_of_the_usable_total`].
    fn from_total(slot_ranges: &[(u16, u16)], total: u64, endpoints: usize) -> ExtReleaseBudget {
        let (bytes, derivation) =
            memory_bytes_budget(slot_ranges, total).expect("a non-empty map gives a budget");
        ExtReleaseBudget {
            bytes,
            source: ExtReleaseBudgetSource::ValkeyMemory {
                derivation,
                endpoints,
            },
        }
    }

    fn explicit(bytes: u64) -> ExtReleaseBudget {
        ExtReleaseBudget {
            bytes,
            source: ExtReleaseBudgetSource::Explicit,
        }
    }

    /// `actual` is a value within rounding of `expected`.
    fn assert_ratio(actual: Option<f64>, expected: f64, what: &str) {
        let actual = actual.unwrap_or_else(|| panic!("{what}: expected a figure, got none"));
        assert!(
            (actual - expected).abs() < 1e-12,
            "{what}: {actual} is not {expected}"
        );
    }

    // ── parse_cluster_slots_reply ────────────────────────────────────────────

    /// One `CLUSTER SLOTS` entry: `[start, end, [host, port, id]]`.
    fn slots_entry(start: i64, end: i64, host: &str, port: i64) -> fred::types::Value {
        use fred::types::Value;
        Value::Array(vec![
            Value::Integer(start),
            Value::Integer(end),
            Value::Array(vec![
                Value::String(host.into()),
                Value::Integer(port),
                Value::String("0123456789abcdef".into()),
            ]),
        ])
    }

    /// A `CLUSTER SLOTS` node element: `[host, port, id, extra...]`.
    fn slots_node(
        host: fred::types::Value,
        port: i64,
        extra: Vec<fred::types::Value>,
    ) -> fred::types::Value {
        let mut node = vec![
            host,
            fred::types::Value::Integer(port),
            fred::types::Value::String("0123456789abcdef".into()),
        ];
        node.extend(extra);
        fred::types::Value::Array(node)
    }

    /// A `CLUSTER SLOTS` entry: `[start, end, primary, replicas...]`.
    fn slots_entry_of(start: i64, end: i64, nodes: Vec<fred::types::Value>) -> fred::types::Value {
        let mut entry = vec![
            fred::types::Value::Integer(start),
            fred::types::Value::Integer(end),
        ];
        entry.extend(nodes);
        fred::types::Value::Array(entry)
    }

    /// A string host.
    fn host(name: &str) -> fred::types::Value {
        fred::types::Value::String(name.into())
    }

    /// Every well-formed shape parses to its entries' ranges, one per entry,
    /// in reply order and never merged (the X20 cache: 8 entries on 2
    /// endpoints are 8 ranges and 2 endpoints), the endpoints counted by
    /// primary `host:port`: a NULL host (the replying node) is one
    /// placeholder endpoint, beside a named host another; the metadata a
    /// Redis 7+ node element carries — a RESP3 map, or its RESP2 flat array —
    /// and the replica elements are not read; a host sent as bytes is the
    /// same host as one sent as a string. One collected table, so a run that
    /// breaks several shapes shows each.
    #[test]
    fn parse_cluster_slots_reply_reads_every_well_formed_shape() {
        use fred::types::Value;
        let map = |slot_ranges: Vec<(u16, u16)>, endpoints| {
            Some(ClusterSlotsMap {
                slot_ranges,
                endpoints,
            })
        };
        let mut resp3_metadata = fred::types::Map::new();
        resp3_metadata.insert(
            fred::types::Key::from("hostname"),
            host("cache.example.com"),
        );
        let resp2_metadata = Value::Array(vec![host("hostname"), host("cache.example.com")]);
        let cases: Vec<(&str, Value, Option<ClusterSlotsMap>)> = vec![
            (
                "X20: 8 entries interleaved on 2 endpoints, one range each",
                Value::Array(
                    (0..8i64)
                        .map(|i| {
                            slots_entry(
                                i * 2048,
                                i * 2048 + 2047,
                                "172.160.111.250",
                                8500 + (i + 1) % 2,
                            )
                        })
                        .collect(),
                ),
                map(x20_map(), 2),
            ),
            (
                "a NULL host on every entry: one placeholder endpoint",
                Value::Array(vec![
                    slots_entry_of(0, 8191, vec![slots_node(Value::Null, 6379, vec![])]),
                    slots_entry_of(8192, 16383, vec![slots_node(Value::Null, 6379, vec![])]),
                ]),
                map(dev_map(), 1),
            ),
            (
                "a NULL host beside a named one: two endpoints",
                Value::Array(vec![
                    slots_entry_of(0, 8191, vec![slots_node(Value::Null, 6379, vec![])]),
                    slots_entry_of(
                        8192,
                        16383,
                        vec![slots_node(host("10.0.0.1"), 6379, vec![])],
                    ),
                ]),
                map(dev_map(), 2),
            ),
            (
                "RESP3 metadata map on the primary",
                Value::Array(vec![slots_entry_of(
                    0,
                    16383,
                    vec![slots_node(
                        host("10.0.0.1"),
                        6379,
                        vec![Value::Map(resp3_metadata)],
                    )],
                )]),
                map(vec![STANDALONE_SLOT_RANGE], 1),
            ),
            (
                "RESP2 metadata array on the primary",
                Value::Array(vec![slots_entry_of(
                    0,
                    16383,
                    vec![slots_node(host("10.0.0.1"), 6379, vec![resp2_metadata])],
                )]),
                map(vec![STANDALONE_SLOT_RANGE], 1),
            ),
            (
                "replica elements after the primary are not endpoints",
                Value::Array(vec![slots_entry_of(
                    0,
                    16383,
                    vec![
                        slots_node(host("10.0.0.1"), 6379, vec![]),
                        slots_node(host("10.0.0.9"), 6380, vec![]),
                        slots_node(host("10.0.0.10"), 6381, vec![]),
                    ],
                )]),
                map(vec![STANDALONE_SLOT_RANGE], 1),
            ),
            (
                "a bytes host is the same host as a string one",
                Value::Array(vec![
                    slots_entry_of(
                        0,
                        8191,
                        vec![slots_node(
                            Value::Bytes(fred::bytes::Bytes::from_static(b"10.0.0.1")),
                            6379,
                            vec![],
                        )],
                    ),
                    slots_entry_of(
                        8192,
                        16383,
                        vec![slots_node(host("10.0.0.1"), 6379, vec![])],
                    ),
                ]),
                map(dev_map(), 1),
            ),
        ];
        let got: Vec<_> = cases
            .iter()
            .map(|(what, reply, _)| (*what, parse_cluster_slots_reply(reply)))
            .collect();
        let expected: Vec<_> = cases
            .into_iter()
            .map(|(what, _, expected)| (what, expected))
            .collect();
        assert_eq!(got, expected);
    }

    /// An empty reply, a reply that is not an array, or any one malformed
    /// entry — even beside a good one — is no map at all: a partial map would
    /// undercount the shards. One collected table, so a run that lets several
    /// through shows each.
    #[test]
    fn parse_cluster_slots_reply_refuses_an_empty_reply_and_every_malformed_field() {
        use fred::types::Value;
        let good = || slots_entry(0, 8191, "10.0.0.1", 6379);
        let with_second = |second: Value| Value::Array(vec![good(), second]);
        let cases: Vec<(&str, Value)> = vec![
            ("an empty reply", Value::Array(Vec::new())),
            ("a reply that is not an array", Value::Null),
            (
                "an end slot above 16383",
                with_second(slots_entry(8192, 16384, "10.0.0.2", 6379)),
            ),
            (
                "a start slot after its end",
                with_second(slots_entry(9000, 8192, "10.0.0.2", 6379)),
            ),
            (
                "a negative start slot",
                with_second(slots_entry(-1, 16383, "10.0.0.2", 6379)),
            ),
            (
                "a port above 65535",
                with_second(slots_entry(8192, 16383, "10.0.0.2", 70_000)),
            ),
            (
                "a negative port",
                with_second(slots_entry(8192, 16383, "10.0.0.2", -1)),
            ),
            (
                "a host that is an integer",
                with_second(Value::Array(vec![
                    Value::Integer(8192),
                    Value::Integer(16383),
                    Value::Array(vec![Value::Integer(10), Value::Integer(6379)]),
                ])),
            ),
            (
                "an entry with no primary",
                with_second(Value::Array(vec![
                    Value::Integer(8192),
                    Value::Integer(16383),
                ])),
            ),
            (
                "a primary that is not an array",
                with_second(Value::Array(vec![
                    Value::Integer(8192),
                    Value::Integer(16383),
                    host("10.0.0.2"),
                ])),
            ),
            (
                "a port sent as a string",
                with_second(slots_entry_of(
                    8192,
                    16383,
                    vec![Value::Array(vec![host("10.0.0.2"), host("6379")])],
                )),
            ),
            (
                "a primary with no port",
                with_second(slots_entry_of(
                    8192,
                    16383,
                    vec![Value::Array(vec![host("10.0.0.2")])],
                )),
            ),
            (
                "a host whose bytes are not UTF-8",
                with_second(slots_entry_of(
                    8192,
                    16383,
                    vec![slots_node(
                        Value::Bytes(fred::bytes::Bytes::from_static(&[0xff, 0xfe])),
                        6379,
                        vec![],
                    )],
                )),
            ),
            (
                "an entry that is not an array",
                with_second(Value::Integer(5)),
            ),
            (
                "a start slot sent as a string",
                with_second(Value::Array(vec![
                    host("8192"),
                    Value::Integer(16383),
                    slots_node(host("10.0.0.2"), 6379, vec![]),
                ])),
            ),
            (
                "an end slot sent as a string",
                with_second(Value::Array(vec![
                    Value::Integer(8192),
                    host("16383"),
                    slots_node(host("10.0.0.2"), 6379, vec![]),
                ])),
            ),
        ];
        let got: Vec<_> = cases
            .iter()
            .map(|(what, reply)| (*what, parse_cluster_slots_reply(reply)))
            .collect();
        let expected: Vec<_> = cases.iter().map(|(what, _)| (*what, None)).collect();
        assert_eq!(got, expected);
    }

    // ── T1: parse_info_used_memory / parse_info_replconf_calls ──────────────

    /// T1: only the exact `used_memory:` field counts — not its neighbours
    /// `used_memory_human`, `_rss`, `_peak`, `_lua` or `_dataset`, which sort
    /// before it here — under LF, CRLF and Azure Managed Redis's `\r\r\n` line
    /// ends; a reply holding only the neighbours has none.
    #[test]
    fn parse_info_used_memory_reads_only_the_exact_field() {
        for (end, what) in [("\r\n", "CRLF"), ("\n", "LF"), ("\r\r\n", "CR CR LF")] {
            let neighbours = [
                "used_memory_human:448.54M",
                "used_memory_rss:9999",
                "used_memory_peak:8888",
                "used_memory_lua:7777",
                "used_memory_dataset:6666",
            ]
            .join(end);
            let reply = format!(
                "# Memory{end}{neighbours}{end}used_memory:470333232{end}maxmemory_policy:noeviction{end}"
            );
            assert_eq!(parse_info_used_memory(&reply), Some(DEV_USED), "{what}");
            let only_neighbours = format!("# Memory{end}{neighbours}{end}");
            assert_eq!(
                parse_info_used_memory(&only_neighbours),
                None,
                "{what}: the neighbours alone are not the field"
            );
        }
    }

    /// T1: a reply without the field, or with a value that is not a byte
    /// count, gives no `used_memory`.
    #[test]
    fn parse_info_used_memory_without_a_byte_count_is_none() {
        assert_eq!(parse_info_used_memory(""), None);
        assert_eq!(parse_info_used_memory("# Memory\r\nmaxmemory:0\r\n"), None);
        for value in ["", "-1", "448.54M", "abc", "18446744073709551616"] {
            assert_eq!(
                parse_info_used_memory(&format!("used_memory:{value}\r\n")),
                None,
                "used_memory:{value:?}"
            );
        }
    }

    /// T1: the `calls=` of `cmdstat_replconf:`, in Azure Managed Redis's form
    /// (`\r\r\n` line ends, no `rejected_calls`/`failed_calls`) and Valkey's
    /// (with them), beside other commands' lines — `cmdstat_psync`'s count is
    /// not the replica acknowledgement count — and wherever `calls` sits in
    /// the line.
    #[test]
    fn parse_info_replconf_calls_reads_the_call_count_in_either_stores_form() {
        let amr = "# Commandstats\r\r\n\
                   cmdstat_psync:calls=8,usec=1234,usec_per_call=154.25\r\r\n\
                   cmdstat_replconf:calls=1590380,usec=2449185,usec_per_call=1.54\r\r\n\
                   cmdstat_info:calls=42,usec=100,usec_per_call=2.38\r\r\n";
        assert_eq!(parse_info_replconf_calls(amr), Some(1_590_380), "AMR");
        let valkey = "# Commandstats\r\n\
                      cmdstat_replconf:calls=2131,usec=3000,usec_per_call=1.41,\
                      rejected_calls=0,failed_calls=0\r\n\
                      cmdstat_ping:calls=7,usec=7,usec_per_call=1.00,rejected_calls=0,\
                      failed_calls=0\r\n";
        assert_eq!(parse_info_replconf_calls(valkey), Some(2_131), "Valkey");
        assert_eq!(
            parse_info_replconf_calls("cmdstat_replconf:usec=3,calls=12,usec_per_call=0.25\n"),
            Some(12),
            "calls after usec"
        );
    }

    /// T1: a store lists only the commands it has run, so a reply with no
    /// `cmdstat_replconf:` line — no replica ever acknowledged — reads 0.
    #[test]
    fn parse_info_replconf_calls_without_the_line_reads_zero() {
        assert_eq!(parse_info_replconf_calls(""), Some(0));
        assert_eq!(
            parse_info_replconf_calls(
                "# Commandstats\r\ncmdstat_info:calls=42,usec=100,usec_per_call=2.38\r\n"
            ),
            Some(0)
        );
    }

    /// T1: a `cmdstat_replconf:` line whose `calls` is missing or not a
    /// non-negative integer is unreadable, never 0 (which would read as "no
    /// replica").
    #[test]
    fn parse_info_replconf_calls_with_a_malformed_line_is_none() {
        for line in [
            "cmdstat_replconf:",
            "cmdstat_replconf:usec=3,usec_per_call=0.25",
            "cmdstat_replconf:calls=,usec=3",
            "cmdstat_replconf:calls=abc,usec=3",
            "cmdstat_replconf:calls=-1,usec=3",
            "cmdstat_replconf:calls=1.5,usec=3",
        ] {
            assert_eq!(
                parse_info_replconf_calls(&format!("{line}\r\n")),
                None,
                "{line}"
            );
        }
    }

    // ── T2: store_copies ─────────────────────────────────────────────────────

    /// T2: the copies are 2 only when the acknowledgement count rose since the
    /// previous good read; 1 when it stood still; unknown (the gauge absent)
    /// with no previous read or when it fell (a statistics reset). The rule
    /// sees only the count; that a placeholder `connected_slaves:1` with a
    /// standing count reads copies 1 is the integration test
    /// `a_store_whose_replicas_never_acknowledge_reads_one_copy`.
    #[test]
    fn store_copies_is_two_only_when_the_acknowledgement_count_rose() {
        for (previous, current, expected, what) in [
            (None, 651, None, "no previous read"),
            (
                Some(651),
                2_131,
                Some(STORE_REPLICATED_COPIES),
                "the X20 capture, 22:35:33 to 22:38:38",
            ),
            (Some(651), 652, Some(2), "rose by one"),
            (Some(651), 651, Some(1), "stood still"),
            (Some(0), 0, Some(1), "no replconf line on either read"),
            (
                Some(2_131),
                651,
                None,
                "fell: the store's statistics were reset",
            ),
        ] {
            assert_eq!(store_copies(previous, current), expected, "{what}");
        }
        assert_eq!(STORE_REPLICATED_COPIES, 2);
    }

    // ── T3: memory_bytes_budget ──────────────────────────────────────────────

    /// T3, the D2 arithmetic, every case in this one table:
    /// `per_shard = total × 0.8 × the smallest entry's slots ÷ 16384` and
    /// `H = max(1, per_shard ÷ 8)`, in u128, rounded down. The shard count is
    /// the map's entries, never its endpoints: the X20 map's budget is the
    /// same whether 1, 2 or 8 endpoints list it.
    #[test]
    fn memory_bytes_budget_is_one_eighth_of_the_smallest_entrys_share_of_the_usable_total() {
        struct Case {
            what: &'static str,
            ranges: Vec<(u16, u16)>,
            total: u64,
            smallest: u64,
            per_shard: u64,
            budget: u64,
        }
        let one_slot_first = vec![(0, 0), (1, 16_383)];
        let cases = [
            Case {
                what: "X20: 8 x 2,048 on 2 endpoints, 24 GB (dividing by the 2 endpoints would \
                       give 1,200,000,000)",
                ranges: x20_map(),
                total: X20_TOTAL,
                smallest: 2_048,
                per_shard: 2_400_000_000,
                budget: 300_000_000,
            },
            Case {
                what: "dev: 2 x 8,192, 12 GB",
                ranges: dev_map(),
                total: DEV_TOTAL,
                smallest: 8_192,
                per_shard: 4_800_000_000,
                budget: 600_000_000,
            },
            Case {
                what: "standalone: one range over every slot, 12 GB",
                ranges: vec![STANDALONE_SLOT_RANGE],
                total: DEV_TOTAL,
                smallest: 16_384,
                per_shard: 9_600_000_000,
                budget: 1_200_000_000,
            },
            Case {
                what: "unequal entries 5,461 / 5,462 / 5,461: the smallest",
                ranges: vec![(0, 5_460), (5_461, 10_922), (10_923, 16_383)],
                total: DEV_TOTAL,
                smallest: 5_461,
                per_shard: 3_199_804_687,
                budget: 399_975_585,
            },
            Case {
                what: "fragmented: one node owns 0-99 and 5000-10000; the smallest ENTRY, 100 \
                       slots, understates that node (pinned)",
                ranges: vec![(0, 99), (100, 4_999), (5_000, 10_000), (10_001, 16_383)],
                total: DEV_TOTAL,
                smallest: 100,
                per_shard: 58_593_750,
                budget: 7_324_218,
            },
            Case {
                what: "the 100 MB floor with a 1-slot entry",
                ranges: one_slot_first.clone(),
                total: crate::config::VALKEY_MEMORY_BYTES_MIN,
                smallest: 1,
                per_shard: 4_882,
                budget: 610,
            },
            Case {
                what: "below the floor (reachable only past config): a per-shard figure under 8 \
                       clamps to 1, never the disabling 0",
                ranges: one_slot_first,
                total: 100,
                smallest: 1,
                per_shard: 0,
                budget: 1,
            },
            Case {
                what: "u64::MAX over every slot: no overflow",
                ranges: vec![STANDALONE_SLOT_RANGE],
                total: u64::MAX,
                smallest: 16_384,
                per_shard: 14_757_395_258_967_641_292,
                budget: 1_844_674_407_370_955_161,
            },
            Case {
                what: "a range whose end precedes its start counts as one slot",
                ranges: vec![(5, 4), (6, 16_383)],
                total: DEV_TOTAL,
                smallest: 1,
                per_shard: 585_937,
                budget: 73_242,
            },
        ];
        for case in cases {
            let (bytes, derivation) = memory_bytes_budget(&case.ranges, case.total)
                .unwrap_or_else(|| panic!("{}: a non-empty map gives a budget", case.what));
            assert_eq!(
                (
                    derivation.smallest_shard_slots,
                    derivation.per_shard_bytes,
                    bytes
                ),
                (case.smallest, case.per_shard, case.budget),
                "{}: (smallest entry's slots, per-shard bytes, budget)",
                case.what
            );
            assert_eq!(
                (derivation.shards, derivation.total_bytes),
                (case.ranges.len(), case.total),
                "{}: the shards are the entries",
                case.what
            );
        }

        assert_eq!(
            memory_bytes_budget(&[], X20_TOTAL),
            None,
            "an empty map gives no budget"
        );

        for endpoints in [1, X20_ENDPOINTS, 8] {
            let reading = amr(x20_map(), endpoints, 1, ReplconfCalls::NotRead);
            let budget = budget_from_reading(&reading, Some(X20_TOTAL))
                .unwrap_or_else(|| panic!("{endpoints} endpoints: rule 3 answers"));
            assert_eq!(
                budget.bytes, 300_000_000,
                "{endpoints} endpoints: the endpoint count never divides"
            );
            assert_eq!(
                budget.source.cluster_map().map(|(d, e)| (d.shards, e)),
                Some((8, endpoints)),
                "{endpoints} endpoints: logged beside the shards, not used"
            );
        }
    }

    // ── T4: precedence ───────────────────────────────────────────────────────

    /// T4: an explicit `LIVE_EDIT_EXT_RELEASE_MAX_BYTES` stands over a store
    /// limit and over the store total, at startup and on every refresh.
    #[test]
    fn an_explicit_budget_stands_over_the_store_limit_and_the_store_total() {
        for (what, reading) in [
            ("a store limit", limit(THREE_GIB, 3)),
            (
                "a store without maxmemory",
                amr(x20_map(), X20_ENDPOINTS, 1, ReplconfCalls::Calls(5)),
            ),
        ] {
            let resolution = resolve_startup(
                explicit(123_456),
                &no_map_problem(reading.clone()),
                Some(X20_TOTAL),
            );
            assert_eq!(resolution.budget, explicit(123_456), "{what}");
            assert_eq!(
                refreshed_budget(&explicit(123_456), &reading, Some(X20_TOTAL)),
                None,
                "{what}: a refresh never moves it"
            );
        }
    }

    /// T4 (I2): a store whose every primary reports a `maxmemory` wins over
    /// `LIVE_EDIT_VALKEY_MEMORY_BYTES` — `info`, not `override` — at startup
    /// and on a refresh that finds an `override` budget in force.
    ///
    /// **Fails on the build before b.qhk**, where the configuration answered
    /// `override` (the setting ÷ 8) from the setting alone and the store was
    /// never read.
    #[test]
    fn a_store_reported_maxmemory_wins_over_the_store_total() {
        let resolution = resolve_startup(
            fallback(),
            &no_map_problem(limit(THREE_GIB, 3)),
            Some(X20_TOTAL),
        );
        assert_eq!(resolution.budget, from_store(THREE_GIB, 3));
        assert_eq!(resolution.budget.source.label(), "info");
        assert_eq!(resolution.problem, None);

        assert_eq!(
            refreshed_budget(
                &from_total(&x20_map(), X20_TOTAL, X20_ENDPOINTS),
                &limit(THREE_GIB, 3),
                Some(X20_TOTAL)
            ),
            Some(from_store(THREE_GIB, 3)),
            "a store that starts reporting a limit moves an override budget to info"
        );
    }

    /// T4: with the setting set and the map known, a store with no usable
    /// `maxmemory` — no field at all (Azure Managed Redis), `maxmemory:0`, a
    /// malformed value, or a mix — takes rule 3 under `override`, and the
    /// reading's problem is still named.
    #[test]
    fn an_unusable_maxmemory_with_the_store_total_set_derives_from_the_map() {
        let used = Some(1);
        for (what, reading, kind) in [
            (
                "no maxmemory field",
                amr(x20_map(), X20_ENDPOINTS, 1, ReplconfCalls::Calls(5)),
                StoreMemoryProblemKind::FieldAbsent,
            ),
            (
                "maxmemory:0",
                over_x20(&[
                    (MaxmemoryField::Value(0), used),
                    (MaxmemoryField::Value(0), used),
                ]),
                StoreMemoryProblemKind::ZeroLimit,
            ),
            (
                "a malformed maxmemory",
                over_x20(&[
                    (MaxmemoryField::Malformed, used),
                    (MaxmemoryField::Malformed, used),
                ]),
                StoreMemoryProblemKind::Malformed,
            ),
            (
                "a limit on one primary, no field on the other",
                over_x20(&[
                    (MaxmemoryField::Value(THREE_GIB), used),
                    (MaxmemoryField::Absent, used),
                ]),
                StoreMemoryProblemKind::FieldAbsent,
            ),
        ] {
            let resolution = resolve_startup(fallback(), &no_map_problem(reading), Some(X20_TOTAL));
            assert_eq!(
                resolution.budget,
                from_total(&x20_map(), X20_TOTAL, X20_ENDPOINTS),
                "{what}"
            );
            assert_eq!(resolution.budget.bytes, 300_000_000, "{what}");
            assert_eq!(resolution.budget.source.label(), "override", "{what}");
            assert_eq!(
                resolution.problem.map(|p| p.kind()),
                Some(kind),
                "{what}: the problem is still named"
            );
        }
    }

    /// T4 (PD3): with the setting set and the map known, an `INFO` that failed
    /// or was refused at startup starts on rule 3, not the fallback, and names
    /// the failure; without the setting it is the fallback, as before.
    #[test]
    fn a_failed_info_with_the_store_total_set_starts_on_the_map() {
        let reading = failed_over(
            x20_map(),
            X20_ENDPOINTS,
            "NOPERM this user has no permissions to run the 'info' command",
        );
        let resolution = resolve_startup(
            fallback(),
            &no_map_problem(reading.clone()),
            Some(X20_TOTAL),
        );
        assert_eq!(
            resolution.budget,
            from_total(&x20_map(), X20_TOTAL, X20_ENDPOINTS)
        );
        assert_eq!(
            resolution.problem.map(|p| p.kind()),
            Some(StoreMemoryProblemKind::ReadFailed)
        );
        assert_eq!(
            resolve_startup(fallback(), &no_map_problem(reading), None).budget,
            fallback(),
            "the setting unset: the fallback"
        );
    }

    /// T4: on a refresh a failed `INFO` keeps an `info` budget in force — the
    /// store's own limit was the last thing seen — while a budget on the
    /// fallback or on rule 3 is (re-)derived from the map, which a failed
    /// `INFO` still carries.
    #[test]
    fn a_failed_info_on_a_refresh_keeps_an_info_budget_and_rederives_from_the_map() {
        let failed = failed_over(x20_map(), X20_ENDPOINTS, "timed out");
        assert_eq!(
            refreshed_budget(&from_store(THREE_GIB, 2), &failed, Some(X20_TOTAL)),
            None,
            "an info budget stands"
        );
        let x20 = from_total(&x20_map(), X20_TOTAL, X20_ENDPOINTS);
        for current in [fallback(), from_total(&dev_map(), X20_TOTAL, X20_ENDPOINTS)] {
            assert_eq!(
                refreshed_budget(&current, &failed, Some(X20_TOTAL)),
                Some(x20),
                "{current:?}: re-derived from the map"
            );
        }
        assert_eq!(
            refreshed_budget(&x20, &failed, Some(X20_TOTAL)),
            None,
            "an unchanged map changes nothing"
        );
    }

    /// T4: a refresh moves the budget on a changed shard count, and on a
    /// change of source alone: an `info` budget equal in bytes to the map's
    /// is replaced by the map's when the store stops reporting its limit.
    #[test]
    fn a_refresh_follows_a_changed_map_and_a_changed_source() {
        let x20 = from_total(&x20_map(), X20_TOTAL, X20_ENDPOINTS);
        let reading = amr(x20_map(), X20_ENDPOINTS, 1, ReplconfCalls::Calls(5));
        assert_eq!(
            refreshed_budget(
                &from_total(&dev_map(), X20_TOTAL, X20_ENDPOINTS),
                &reading,
                Some(X20_TOTAL)
            ),
            Some(x20),
            "2 entries to 8"
        );
        let same_bytes_from_info = from_store(2_400_000_000, 2);
        assert_eq!(same_bytes_from_info.bytes, x20.bytes, "precondition");
        assert_eq!(
            refreshed_budget(&same_bytes_from_info, &reading, Some(X20_TOTAL)),
            Some(x20),
            "info to override at equal bytes"
        );
    }

    /// T4: no map — the routing table not known yet, no primary, or a map
    /// with no entry — leaves the setting nothing to divide: startup stays on
    /// the fallback and names why, and a refresh keeps the last good budget,
    /// whatever it is.
    #[test]
    fn no_map_starts_on_the_fallback_and_a_refresh_keeps_the_last_good_budget() {
        let empty_map = StoreMemoryReading::Read {
            slot_ranges: vec![],
            endpoints: 1,
            info: Ok(vec![primary(
                "primary 10.0.0.1:8500",
                MaxmemoryField::Absent,
            )]),
            replconf_calls: ReplconfCalls::NotRead,
        };
        for (what, reading) in [
            ("routing unknown", StoreMemoryReading::RoutingUnknown),
            ("no primary", StoreMemoryReading::NoPrimary),
            ("a map with no entry", empty_map),
        ] {
            let resolution = resolve_startup(
                fallback(),
                &no_map_problem(reading.clone()),
                Some(X20_TOTAL),
            );
            assert_eq!(resolution.budget, fallback(), "{what}");
            assert!(
                resolution.problem.is_some(),
                "{what}: the fallback names why"
            );
            for current in [
                from_total(&x20_map(), X20_TOTAL, X20_ENDPOINTS),
                from_store(THREE_GIB, 3),
                fallback(),
            ] {
                assert_eq!(
                    refreshed_budget(&current, &reading, Some(X20_TOTAL)),
                    None,
                    "{what}: {current:?} stands"
                );
            }
        }
    }

    // ── T6: store_memory_gauge ───────────────────────────────────────────────

    /// T6: on a store whose primaries all report a limit, the figure is the
    /// fullest primary's — not the mean (0.316…) and not the
    /// smallest-`maxmemory` primary's (0.1) — and the setting changes nothing.
    #[test]
    fn the_gauge_on_a_store_limit_is_the_fullest_primary() {
        let reading = with_used(&[
            (MaxmemoryField::Value(100), Some(10)),
            (MaxmemoryField::Value(200), Some(150)),
            (MaxmemoryField::Value(400), Some(40)),
        ]);
        for memory_bytes in [None, Some(DEV_TOTAL)] {
            assert_eq!(
                store_memory_gauge(&reading, memory_bytes, Some(2)),
                (StoreMemoryGaugeArm::Maxmemory, Some(0.75)),
                "LIVE_EDIT_VALKEY_MEMORY_BYTES {memory_bytes:?}"
            );
        }
    }

    /// T6: on a store with no `maxmemory` field and the setting set, the
    /// figure is the whole-database `used_memory` ÷ copies ÷ 80 % of the
    /// setting — the Contract's dev example with copies 2 and 1 — taking the
    /// highest `used_memory` any endpoint reported.
    #[test]
    fn the_gauge_on_a_store_without_maxmemory_divides_by_the_copies_and_the_usable_total() {
        let reading = amr(dev_map(), 2, DEV_USED, ReplconfCalls::Calls(5));
        for (copies, expected) in [(2, 0.024_496_522_5), (1, 0.048_993_045)] {
            let (arm, ratio) = store_memory_gauge(&reading, Some(DEV_TOTAL), Some(copies));
            assert_eq!(arm, StoreMemoryGaugeArm::StoreTotal { copies });
            assert_ratio(ratio, expected, &format!("copies {copies}"));
        }

        let uneven = StoreMemoryReading::Read {
            slot_ranges: dev_map(),
            endpoints: 2,
            info: Ok(vec![
                PrimaryMemory {
                    primary: "primary 10.0.0.1:8500".to_owned(),
                    maxmemory: MaxmemoryField::Absent,
                    used_memory: Some(960_000_000),
                },
                PrimaryMemory {
                    primary: "primary 10.0.0.2:8500".to_owned(),
                    maxmemory: MaxmemoryField::Absent,
                    used_memory: Some(1_920_000_000),
                },
            ]),
            replconf_calls: ReplconfCalls::Calls(5),
        };
        assert_ratio(
            store_memory_gauge(&uneven, Some(DEV_TOTAL), Some(1)).1,
            0.2,
            "the highest used_memory, 1.92 GB of 9.6 GB",
        );
    }

    /// T6: on that store, with the copies not known yet (a replica's first
    /// read, a statistics reset, an unreadable count), the arm is pending and
    /// the series absent.
    #[test]
    fn the_gauge_on_a_store_without_maxmemory_is_absent_until_the_copies_are_known() {
        let reading = amr(dev_map(), 2, DEV_USED, ReplconfCalls::Calls(5));
        assert_eq!(
            store_memory_gauge(&reading, Some(DEV_TOTAL), None),
            (StoreMemoryGaugeArm::StoreTotalPending, None)
        );
    }

    /// T6: no figure — the series absent, never 0 — for `maxmemory:0`, a
    /// malformed or mixed reading, a store with no `maxmemory` field and the
    /// setting unset, a failed read, no routing table or no primary; and for a
    /// primary that reported no `used_memory`, on either arm.
    #[test]
    fn the_gauge_is_absent_when_no_arm_gives_a_figure() {
        let used = Some(1);
        let one_used_missing = StoreMemoryReading::Read {
            slot_ranges: dev_map(),
            endpoints: 2,
            info: Ok(vec![
                PrimaryMemory {
                    primary: "primary 10.0.0.1:8500".to_owned(),
                    maxmemory: MaxmemoryField::Absent,
                    used_memory: Some(DEV_USED),
                },
                PrimaryMemory {
                    primary: "primary 10.0.0.2:8500".to_owned(),
                    maxmemory: MaxmemoryField::Absent,
                    used_memory: None,
                },
            ]),
            replconf_calls: ReplconfCalls::Calls(5),
        };
        for (what, reading, memory_bytes, arm) in [
            (
                "maxmemory:0",
                with_used(&[(MaxmemoryField::Value(0), used)]),
                Some(DEV_TOTAL),
                StoreMemoryGaugeArm::Absent,
            ),
            (
                "a malformed maxmemory",
                with_used(&[(MaxmemoryField::Malformed, used)]),
                Some(DEV_TOTAL),
                StoreMemoryGaugeArm::Absent,
            ),
            (
                "a limit on one primary, no field on the other",
                with_used(&[
                    (MaxmemoryField::Value(100), used),
                    (MaxmemoryField::Absent, used),
                ]),
                Some(DEV_TOTAL),
                StoreMemoryGaugeArm::Absent,
            ),
            (
                "no maxmemory field, the setting unset",
                amr(dev_map(), 2, DEV_USED, ReplconfCalls::NotRead),
                None,
                StoreMemoryGaugeArm::Absent,
            ),
            (
                "a failed read",
                failed_over(dev_map(), 2, "timed out"),
                Some(DEV_TOTAL),
                StoreMemoryGaugeArm::Absent,
            ),
            (
                "routing unknown",
                StoreMemoryReading::RoutingUnknown,
                Some(DEV_TOTAL),
                StoreMemoryGaugeArm::Absent,
            ),
            (
                "no primary",
                StoreMemoryReading::NoPrimary,
                Some(DEV_TOTAL),
                StoreMemoryGaugeArm::Absent,
            ),
            (
                "a limit, one primary without used_memory",
                with_used(&[
                    (MaxmemoryField::Value(100), used),
                    (MaxmemoryField::Value(100), None),
                ]),
                None,
                StoreMemoryGaugeArm::Maxmemory,
            ),
            (
                "no maxmemory field, one endpoint without used_memory",
                one_used_missing,
                Some(DEV_TOTAL),
                StoreMemoryGaugeArm::StoreTotal { copies: 2 },
            ),
        ] {
            assert_eq!(
                store_memory_gauge(&reading, memory_bytes, Some(2)),
                (arm, None),
                "{what}"
            );
        }
    }

    /// T6: a store past its limit reads above 1, on either arm.
    #[test]
    fn the_gauge_is_not_clamped_above_one() {
        assert_eq!(
            store_memory_gauge(
                &with_used(&[(MaxmemoryField::Value(100), Some(150))]),
                None,
                None
            ),
            (StoreMemoryGaugeArm::Maxmemory, Some(1.5))
        );
        // 24 GB whole-database over 2 copies against 9.6 GB usable: 1.25.
        let reading = amr(dev_map(), 2, 24_000_000_000, ReplconfCalls::Calls(5));
        assert_ratio(
            store_memory_gauge(&reading, Some(DEV_TOTAL), Some(2)).1,
            1.25,
            "store total",
        );
    }

    /// A standalone reading of `primaries`, each a `(maxmemory, used_memory)`
    /// pair.
    fn with_used(primaries: &[(MaxmemoryField, Option<u64>)]) -> StoreMemoryReading {
        read(
            primaries
                .iter()
                .enumerate()
                .map(|(i, &(maxmemory, used_memory))| PrimaryMemory {
                    primary: format!("primary 10.0.0.{i}:6379"),
                    maxmemory,
                    used_memory,
                })
                .collect(),
        )
    }

    // ── store_memory_refresh_needed / resolve_startup ────────────────────────

    /// The refresh is skipped only when nothing can use it: an explicit
    /// budget, the setting unset, and a store that answered with no
    /// `maxmemory` field on any primary. Every other startup read — a limit,
    /// a `maxmemory:0` that can be set in place, a malformed value, a failed
    /// read, no routing table — keeps it running.
    #[test]
    fn the_refresh_is_skipped_only_when_neither_consumer_can_use_it() {
        let no_field = amr(dev_map(), 2, 1, ReplconfCalls::NotRead);
        assert!(
            !store_memory_refresh_needed(&explicit(1), &no_field, None),
            "the one skip"
        );
        assert!(
            !resolve_startup(explicit(1), &no_map_problem(no_field.clone()), None).refresh_needed,
            "startup carries it"
        );
        for (what, configured, reading, memory_bytes) in [
            (
                "a budget that is not explicit",
                fallback(),
                no_field.clone(),
                None,
            ),
            ("the setting set", explicit(1), no_field, Some(DEV_TOTAL)),
            ("a store limit", explicit(1), limit(THREE_GIB, 1), None),
            (
                "maxmemory:0",
                explicit(1),
                one(MaxmemoryField::Value(0)),
                None,
            ),
            (
                "a malformed maxmemory",
                explicit(1),
                one(MaxmemoryField::Malformed),
                None,
            ),
            ("a failed read", explicit(1), failed("timed out"), None),
            (
                "routing unknown",
                explicit(1),
                StoreMemoryReading::RoutingUnknown,
                None,
            ),
            ("explicit 0", explicit(0), limit(THREE_GIB, 1), None),
        ] {
            assert!(
                store_memory_refresh_needed(&configured, &reading, memory_bytes),
                "{what}"
            );
        }
    }

    /// Startup on a store with no `maxmemory` field: the gauge's arm is
    /// pending and its value absent (its copies need a second read), and the
    /// refresh is seeded with startup's acknowledgement count, its arm, its
    /// problem and its cluster map problem. A `CLUSTER SLOTS` that gave no map
    /// leaves startup on the fallback, the kind carried to the refresh. An
    /// unreadable count is carried as such, with no count to compare against.
    #[test]
    fn startup_on_a_store_without_maxmemory_seeds_the_refresh_with_what_it_read() {
        let read = no_map_problem(amr(dev_map(), 2, DEV_USED, ReplconfCalls::Calls(651)));
        let resolution = resolve_startup(fallback(), &read, Some(DEV_TOTAL));
        assert_eq!(
            (resolution.gauge_arm, resolution.used_ratio),
            (StoreMemoryGaugeArm::StoreTotalPending, None)
        );
        assert_eq!(resolution.budget.bytes, 600_000_000);
        assert_eq!(
            resolution.seed(),
            RefreshSeed {
                problem: Some(StoreMemoryProblemKind::FieldAbsent),
                gauge_arm: Some(StoreMemoryGaugeArm::StoreTotalPending),
                replconf_calls: Some(651),
                map_problem: None,
                commandstats_unreadable: false,
            }
        );

        let no_map = StoreMemoryRead {
            reading: amr(Vec::new(), 2, DEV_USED, ReplconfCalls::Calls(651)),
            map_problem: Some(ClusterMapProblem::ReadFailed("timed out".to_owned())),
        };
        let resolution = resolve_startup(fallback(), &no_map, Some(DEV_TOTAL));
        assert_eq!(resolution.budget, fallback(), "no map: the fallback");
        assert_eq!(
            resolution.seed().map_problem,
            Some(ClusterMapProblemKind::ReadFailed)
        );

        let unreadable = no_map_problem(amr(
            dev_map(),
            2,
            DEV_USED,
            ReplconfCalls::Unreadable("INFO commandstats failed".to_owned()),
        ));
        let seed = resolve_startup(fallback(), &unreadable, Some(DEV_TOTAL)).seed();
        assert_eq!(
            (
                seed.replconf_calls,
                seed.commandstats_unreadable,
                seed.map_problem
            ),
            (None, true, None)
        );
    }

    /// On a store that reports its limit, startup has the figure at once.
    #[test]
    fn startup_on_a_store_limit_has_the_gauge_at_once() {
        let resolution = resolve_startup(
            fallback(),
            &no_map_problem(with_used(&[(MaxmemoryField::Value(100), Some(25))])),
            None,
        );
        assert_eq!(
            (resolution.gauge_arm, resolution.used_ratio),
            (StoreMemoryGaugeArm::Maxmemory, Some(0.25))
        );
    }

    // ── log capture ──────────────────────────────────────────────────────────

    /// Every line logged at DEBUG and above on this thread while its guard
    /// lives. The refresh tests run on a current-thread runtime, so the
    /// spawned task logs on this thread.
    #[derive(Clone, Default)]
    struct LogCapture(Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for LogCapture {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogCapture {
        type Writer = LogCapture;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl LogCapture {
        fn install(&self) -> tracing::subscriber::DefaultGuard {
            tracing::subscriber::set_default(
                tracing_subscriber::fmt()
                    .with_writer(self.clone())
                    .with_max_level(tracing::Level::DEBUG)
                    .with_ansi(false)
                    .finish(),
            )
        }

        fn text(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).expect("UTF-8 log output")
        }

        /// What `log` logs on this thread.
        fn of(log: impl FnOnce()) -> Self {
            let capture = Self::default();
            let guard = capture.install();
            log();
            drop(guard);
            capture
        }

        /// The lines at `level` (`"INFO"`, `"WARN"`, ...) containing `text`.
        fn lines(&self, level: &str, text: &str) -> Vec<String> {
            self.text()
                .lines()
                .filter(|line| line.contains(&format!(" {level} ")) && line.contains(text))
                .map(str::to_owned)
                .collect()
        }
    }

    /// The startup line's message.
    const STARTUP_LINE: &str = "store memory read at startup";

    /// The startup line names the gauge's arm or why it is absent: on a store
    /// with no `maxmemory` field and the setting set, that the value comes at
    /// the next read, and no "does not report its size" line (the setting
    /// answers both consumers); with the setting unset, that it is absent for
    /// that reason. A failed cluster map read and an unreadable count each
    /// WARN once.
    #[test]
    fn the_startup_line_names_the_gauges_arm_or_why_it_is_absent() {
        let pending = resolve_startup(
            fallback(),
            &no_map_problem(amr(dev_map(), 2, DEV_USED, ReplconfCalls::Calls(651))),
            Some(DEV_TOTAL),
        );
        let capture = LogCapture::of(|| log_store_memory_startup(&pending, Some(DEV_TOTAL)));
        let lines = capture.lines("INFO", STARTUP_LINE);
        assert_eq!(lines.len(), 1, "{}", capture.text());
        assert!(
            lines[0].contains("gauge_arm=StoreTotalPending")
                && lines[0].contains("normally at the next read"),
            "{}",
            lines[0]
        );
        assert!(
            capture.lines("INFO", "does not report its size").is_empty()
                && capture.lines("WARN", "").is_empty(),
            "the setting answers both consumers: nothing else; got {}",
            capture.text()
        );

        let unset = resolve_startup(
            fallback(),
            &no_map_problem(amr(dev_map(), 2, DEV_USED, ReplconfCalls::NotRead)),
            None,
        );
        let capture = LogCapture::of(|| log_store_memory_startup(&unset, None));
        let lines = capture.lines("INFO", STARTUP_LINE);
        assert_eq!(lines.len(), 1, "{}", capture.text());
        assert!(
            lines[0].contains("gauge_arm=Absent")
                && lines[0].contains("LIVE_EDIT_VALKEY_MEMORY_BYTES is unset"),
            "{}",
            lines[0]
        );

        let faults = resolve_startup(
            fallback(),
            &StoreMemoryRead {
                reading: amr(
                    Vec::new(),
                    X20_ENDPOINTS,
                    DEV_USED,
                    ReplconfCalls::Unreadable("INFO commandstats failed: NOPERM".to_owned()),
                ),
                map_problem: Some(ClusterMapProblem::ReadFailed(
                    "Timeout Error: Request timed out.".to_owned(),
                )),
            },
            Some(X20_TOTAL),
        );
        let capture = LogCapture::of(|| log_store_memory_startup(&faults, Some(X20_TOTAL)));
        assert_eq!(
            (
                capture.lines("WARN", "the cluster map read failed").len(),
                capture.lines("WARN", "could not be read").len(),
                capture.lines("WARN", "").len(),
            ),
            (1, 1, 2),
            "one WARN each; got {}",
            capture.text()
        );
    }

    // ── T5, T6: the refresh task through its reader seam ─────────────────────

    /// What one refresh pass left behind: the gate's budget (`None` without a
    /// gate), and `/metrics` as rendered right after the pass.
    #[derive(Debug)]
    struct Pass {
        budget: Option<u64>,
        metrics: String,
    }

    impl Pass {
        fn ratio(&self) -> Option<f64> {
            crate::observability::metrics::test_util::sample_value(
                &self.metrics,
                "live_edit_store_memory_used_ratio",
                &[],
            )
        }

        fn budget_cell(&self, source: &str) -> Option<f64> {
            crate::observability::metrics::test_util::sample_value(
                &self.metrics,
                "live_edit_ext_release_budget_bytes",
                &[("source", source)],
            )
        }
    }

    /// An upload gate at `budget_bytes` whose release total is never read.
    fn test_gate(budget_bytes: u64) -> Arc<ExtReleaseGate> {
        use futures::FutureExt;
        Arc::new(ExtReleaseGate::with_total_reader_for_test(
            Arc::new(|| futures::future::ready(Some(0)).boxed()),
            budget_bytes,
            Duration::from_secs(30),
            None,
        ))
    }

    /// Run the refresh task at its real interval over `reads`, one per pass,
    /// and return what each pass left. Call it from a `start_paused` test:
    /// the paused clock jumps to each tick, so the passes take no wall time.
    ///
    /// The reader records the previous pass when it is called for the next
    /// one — the task finishes a pass before it waits for its next tick — and,
    /// past the last read, parks for good, so the task is aborted there.
    async fn run_refresh(
        reads: Vec<StoreMemoryRead>,
        gate: Option<Arc<ExtReleaseGate>>,
        initial: ExtReleaseBudget,
        memory_bytes: Option<u64>,
        seed: RefreshSeed,
    ) -> Vec<Pass> {
        use futures::FutureExt;
        let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only(
            "store-memory-refresh",
        ));
        let script = Arc::new(std::sync::Mutex::new(std::collections::VecDeque::from(
            reads,
        )));
        let passes = Arc::new(std::sync::Mutex::new(Vec::new()));
        let first = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let done = Arc::new(tokio::sync::Notify::new());
        let reader: StoreMemoryReaderForTest = {
            let (script, passes, first, done, gate, metrics) = (
                Arc::clone(&script),
                Arc::clone(&passes),
                Arc::clone(&first),
                Arc::clone(&done),
                gate.clone(),
                Arc::clone(&metrics),
            );
            Arc::new(move || {
                if !first.swap(false, std::sync::atomic::Ordering::SeqCst) {
                    passes.lock().unwrap().push(Pass {
                        budget: gate.as_ref().map(|g| g.budget_bytes()),
                        metrics: crate::observability::metrics::test_util::render(&metrics),
                    });
                }
                match script.lock().unwrap().pop_front() {
                    Some(read) => futures::future::ready(read).boxed(),
                    None => {
                        done.notify_one();
                        futures::future::pending().boxed()
                    }
                }
            })
        };
        let task = spawn_store_memory_refresh_with_reader_for_test(
            reader,
            StoreMemoryRefresh {
                gate,
                initial,
                memory_bytes,
                seed,
                metrics: Some(metrics),
            },
            CancellationToken::new(),
            STORE_MEMORY_REFRESH_INTERVAL,
        );
        tokio::time::timeout(Duration::from_secs(24 * 3600), done.notified())
            .await
            .expect("the refresh task reads every scripted reading");
        task.abort();
        assert!(
            task.await.expect_err("aborted while parked").is_cancelled(),
            "the task was still running, parked on its next read"
        );
        Arc::try_unwrap(passes)
            .expect("the aborted task dropped its reader, the only other holder")
            .into_inner()
            .unwrap()
    }

    /// The budget-change line's message.
    const CHANGE_LINE: &str = "external-value release budget re-read from the store";

    /// T5: the cluster map growing from 2 entries to 8 at the same total moves
    /// the budget on the next pass, without a restart, and the one change
    /// line names the shards, the endpoints, the per-shard bytes and the
    /// total.
    #[tokio::test(start_paused = true)]
    async fn a_resharded_map_moves_the_budget_on_the_next_pass_and_logs_the_map() {
        let capture = LogCapture::default();
        let _guard = capture.install();
        let initial = from_total(&dev_map(), X20_TOTAL, X20_ENDPOINTS);
        assert_eq!(
            initial.bytes, 1_200_000_000,
            "precondition: 24 GB over 2 shards"
        );
        let reads = [dev_map(), x20_map(), x20_map()]
            .into_iter()
            .map(|map| no_map_problem(amr(map, X20_ENDPOINTS, 1, ReplconfCalls::Calls(5))))
            .collect();

        let passes = run_refresh(
            reads,
            Some(test_gate(initial.bytes)),
            initial,
            Some(X20_TOTAL),
            RefreshSeed::default(),
        )
        .await;

        assert_eq!(
            passes.iter().map(|p| p.budget).collect::<Vec<_>>(),
            [Some(1_200_000_000), Some(300_000_000), Some(300_000_000)]
        );
        assert_eq!(passes[1].budget_cell("override"), Some(300_000_000.0));
        let changes = capture.lines("INFO", CHANGE_LINE);
        assert_eq!(
            changes.len(),
            1,
            "one change, one line; got {}",
            capture.text()
        );
        for field in [
            "ext_release_max_bytes=300000000",
            "previous_bytes=1200000000",
            "source=\"override\"",
            "shards=8",
            "endpoints=2",
            "per_shard_bytes=2400000000",
            "total_bytes=24000000000",
        ] {
            assert!(changes[0].contains(field), "{field}: {}", changes[0]);
        }
    }

    /// T5 (I2): the budget moves from `info` to `override` and back without a
    /// restart, as the store stops and starts reporting its limit, and
    /// `live_edit_ext_release_budget_bytes` follows: the source in use carries
    /// the budget, the other 0.
    #[tokio::test(start_paused = true)]
    async fn the_budget_moves_between_info_and_override_both_ways() {
        let reads = vec![
            no_map_problem(limit(THREE_GIB, 2)),
            no_map_problem(amr(x20_map(), X20_ENDPOINTS, 1, ReplconfCalls::Calls(5))),
            no_map_problem(limit(THREE_GIB, 2)),
        ];

        let passes = run_refresh(
            reads,
            Some(test_gate(fallback().bytes)),
            fallback(),
            Some(X20_TOTAL),
            RefreshSeed::default(),
        )
        .await;

        let info_bytes = THREE_GIB / 8;
        assert_eq!(
            passes.iter().map(|p| p.budget).collect::<Vec<_>>(),
            [Some(info_bytes), Some(300_000_000), Some(info_bytes)]
        );
        #[allow(clippy::cast_precision_loss)]
        let info_cell = info_bytes as f64;
        assert_eq!(
            passes
                .iter()
                .map(|p| (p.budget_cell("info"), p.budget_cell("override")))
                .collect::<Vec<_>>(),
            [
                (Some(info_cell), Some(0.0)),
                (Some(0.0), Some(300_000_000.0)),
                (Some(info_cell), Some(0.0)),
            ]
        );
    }

    /// T5: whatever a pass reads, the gate's budget is never 0 — a `maxmemory`
    /// under 8 gives 1, the smallest setting over a 1-slot entry gives 610 —
    /// and a reading of no map or no primary keeps the last good budget.
    #[tokio::test(start_paused = true)]
    async fn the_refresh_never_sets_the_budget_to_zero() {
        let floor = crate::config::VALKEY_MEMORY_BYTES_MIN;
        let tiny = vec![(0, 0), (1, 16_383)];
        let reads = vec![
            no_map_problem(limit(7, 1)),
            no_map_problem(amr(tiny.clone(), 1, 1, ReplconfCalls::Calls(5))),
            no_map_problem(StoreMemoryReading::RoutingUnknown),
            no_map_problem(failed_over(tiny, 1, "timed out")),
            no_map_problem(StoreMemoryReading::NoPrimary),
            no_map_problem(limit(7, 1)),
        ];

        let passes = run_refresh(
            reads,
            Some(test_gate(fallback().bytes)),
            fallback(),
            Some(floor),
            RefreshSeed::default(),
        )
        .await;

        assert_eq!(
            passes.iter().map(|p| p.budget).collect::<Vec<_>>(),
            [Some(1), Some(610), Some(610), Some(610), Some(610), Some(1)]
        );
    }

    /// T6 through the task, on a store with no `maxmemory` field and the
    /// setting set: absent at startup's read, the figure from the next pass —
    /// copies 2 while the acknowledgement count rises, 1 while it stands —
    /// absent after a fall (a statistics reset, which re-baselines), while the
    /// count is unreadable (which keeps the last good count) and after a failed
    /// read, which keeps it too, so the next good read compares against it.
    /// Each change of arm or copies logs one INFO; an unreadable count WARNs
    /// once.
    #[tokio::test(start_paused = true)]
    async fn on_a_store_without_maxmemory_the_gauge_follows_the_acknowledgement_count() {
        let capture = LogCapture::default();
        let _guard = capture.install();
        let at = |calls| no_map_problem(amr(dev_map(), 2, DEV_USED, ReplconfCalls::Calls(calls)));
        let startup = resolve_startup(fallback(), &at(100), Some(DEV_TOTAL));
        assert_eq!(startup.used_ratio, None, "precondition: absent at startup");
        let unreadable = || {
            no_map_problem(amr(
                dev_map(),
                2,
                DEV_USED,
                ReplconfCalls::Unreadable("NOPERM".to_owned()),
            ))
        };
        let reads = vec![
            at(160),
            at(220),
            at(220),
            at(10),
            at(18),
            unreadable(),
            unreadable(),
            at(26),
            no_map_problem(failed_over(dev_map(), 2, "timed out")),
            at(34),
        ];

        let passes =
            run_refresh(reads, None, startup.budget, Some(DEV_TOTAL), startup.seed()).await;

        let (two, one) = (0.024_496_522_5, 0.048_993_045);
        let expected = [
            Some(two),
            Some(two),
            Some(one),
            None,
            Some(two),
            None,
            None,
            Some(two),
            None,
            // M2: the failed read kept the last good count, 26, so 34 rose.
            Some(two),
        ];
        assert_eq!(passes.len(), expected.len());
        for (i, (pass, expected)) in passes.iter().zip(expected).enumerate() {
            match expected {
                Some(ratio) => assert_ratio(pass.ratio(), ratio, &format!("pass {i}")),
                None => assert_eq!(pass.ratio(), None, "pass {i}: the series is absent"),
            }
        }
        assert_eq!(
            capture.lines("INFO", "store memory re-read: ").len(),
            8,
            "one line per change of arm or copies; got {}",
            capture.text()
        );
        assert_eq!(
            capture
                .lines(
                    "WARN",
                    "(cmdstat_replconf in INFO commandstats) could not be read"
                )
                .len(),
            1,
            "{}",
            capture.text()
        );
    }

    /// T5 (PD4) at the task: a pass whose `CLUSTER SLOTS` failed or was
    /// malformed has no map, so the budget derived from the setting keeps its
    /// last good value; the next pass with a map moves it. Each kind WARNs
    /// once until a pass gets a map, then again at its next occurrence.
    #[tokio::test(start_paused = true)]
    async fn a_pass_without_a_map_keeps_the_budget_and_each_kind_warns_once() {
        let capture = LogCapture::default();
        let _guard = capture.install();
        let pass = |problem: Option<ClusterMapProblem>| StoreMemoryRead {
            reading: amr(
                if problem.is_some() {
                    Vec::new()
                } else {
                    x20_map()
                },
                X20_ENDPOINTS,
                1,
                ReplconfCalls::Calls(5),
            ),
            map_problem: problem,
        };
        let failed = || Some(ClusterMapProblem::ReadFailed("timed out".to_owned()));
        // The second read-failed WARN follows a pass with a map: without the
        // reset, it would be a repeat of the last kind and log nothing.
        let reads = vec![
            pass(failed()),
            pass(failed()),
            pass(None),
            pass(failed()),
            pass(Some(ClusterMapProblem::Malformed)),
            pass(Some(ClusterMapProblem::Malformed)),
        ];
        let initial = from_total(&dev_map(), X20_TOTAL, X20_ENDPOINTS);

        let passes = run_refresh(
            reads,
            Some(test_gate(initial.bytes)),
            initial,
            Some(X20_TOTAL),
            RefreshSeed::default(),
        )
        .await;

        assert_eq!(
            passes[0].budget,
            Some(initial.bytes),
            "no map: the last good budget stands"
        );
        assert_eq!(passes[2].budget, Some(300_000_000), "the map is used");
        assert_eq!(passes[4].budget, Some(300_000_000), "and then kept");
        assert_eq!(
            (
                capture.lines("WARN", "the cluster map read failed").len(),
                capture.lines("WARN", "the cluster map was malformed").len(),
            ),
            (2, 1),
            "{}",
            capture.text()
        );
    }

    /// The refresh task inherits startup's WARN-once state
    /// ([`StoreMemoryResolution::seed`]): when startup's cluster map read
    /// failed and its `INFO commandstats` was unreadable — each logged once by
    /// [`log_store_memory_startup`] — re-reads that fail the same way log no
    /// WARN for either.
    #[tokio::test(start_paused = true)]
    async fn the_refresh_does_not_warn_again_for_failures_startup_logged() {
        let failing = || StoreMemoryRead {
            reading: amr(
                Vec::new(),
                X20_ENDPOINTS,
                1,
                ReplconfCalls::Unreadable("INFO commandstats failed: NOPERM".to_owned()),
            ),
            map_problem: Some(ClusterMapProblem::ReadFailed("timed out".to_owned())),
        };
        let startup = resolve_startup(fallback(), &failing(), Some(X20_TOTAL));
        assert!(
            startup.seed().map_problem == Some(ClusterMapProblemKind::ReadFailed)
                && startup.seed().commandstats_unreadable,
            "precondition: startup saw both failures"
        );
        let capture = LogCapture::default();
        let _guard = capture.install();

        let passes = run_refresh(
            vec![failing(), failing()],
            None,
            startup.budget,
            Some(X20_TOTAL),
            startup.seed(),
        )
        .await;

        assert_eq!(passes.len(), 2, "precondition: both passes ran");
        assert_eq!(
            (
                capture.lines("WARN", "the cluster map read failed").len(),
                capture
                    .lines(
                        "WARN",
                        "(cmdstat_replconf in INFO commandstats) could not be read"
                    )
                    .len(),
            ),
            (0, 0),
            "(cluster map WARNs, commandstats WARNs); got {}",
            capture.text()
        );
        assert!(
            capture.lines("WARN", "").is_empty(),
            "no WARN at all; got {}",
            capture.text()
        );
    }

    /// The refresh task inherits startup's gauge arm
    /// ([`RefreshSeed::gauge_arm`]): startup logged the arm, so a re-read on
    /// the same arm logs no arm-change line.
    #[tokio::test(start_paused = true)]
    async fn the_refresh_does_not_log_the_gauge_arm_startup_logged() {
        let limit_read = || no_map_problem(with_used(&[(MaxmemoryField::Value(100), Some(25))]));
        let startup = resolve_startup(fallback(), &limit_read(), None);
        assert_eq!(
            startup.seed().gauge_arm,
            Some(StoreMemoryGaugeArm::Maxmemory),
            "precondition"
        );
        let capture = LogCapture::default();
        let _guard = capture.install();

        let passes = run_refresh(
            vec![limit_read()],
            None,
            startup.budget,
            None,
            startup.seed(),
        )
        .await;

        assert_eq!(passes[0].ratio(), Some(0.25), "precondition: the pass ran");
        assert_eq!(
            capture.lines("INFO", "store memory re-read: ").len(),
            0,
            "the arm did not change; got {}",
            capture.text()
        );
    }
}
