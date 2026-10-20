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

//! Connection buffer limits: what each Valkey connection may keep once its
//! traffic drains, per connection class, and the memory bound they give.
//!
//! A stock fred connection keeps the largest load it ever carried until the
//! connection is replaced: its in-flight command queue, its write and read
//! buffers, and its client's retry buffer only grow. Retained memory then
//! depends on history, not configuration. The vendored fred's retention limits
//! (`fred::live_edit::retention`; `vendor/fred-10.1.0.SOURCE.md`) cut each
//! holder back once it drains: each client keeps one warm connection at its
//! class limits and cuts every other drained connection to a small idle
//! shape, so what a client keeps is a constant of its class whatever the
//! store's primary count. Growth above a limit while a call is in flight
//! belongs to that call, and the upload memory envelope counts it in the
//! call's processing phase; the connection lets go of it at its next drain.
//!
//! Each class is set on a builder with [`apply_limits`]:
//!
//! - [`GENERAL_LIMITS`], by
//!   [`ValkeyClient::connect`](crate::storage::client::ValkeyClient::connect):
//!   on the data-pool builder before the txn pool's clone and the maintenance
//!   and snapshot builds from it, and on the pub/sub client's builder.
//! - [`READER_LIMITS`], by `ValkeyClient::reader_builder`: the template every
//!   stream reader's client is built from.
//!
//! The limits are constants, not operator settings: the memory bound
//! ([`at_rest_bound_bytes`]) follows the configured pool sizes, which are the
//! operator's lever.
//!
//! # The bound
//!
//! Once traffic drains, the connections of one replica keep at most
//!
//! ```text
//! (G + K) × (general connection + (P − 1) × idle connection + c × backchannel)
//!   + R × (reader connection + backchannel)
//! ```
//!
//! where G is the number of general-class clients (the data, txn, maintenance
//! and snapshot pools' members and the pub/sub client), K the general-class
//! clients a member replacement adds beside them (below), R the number of
//! stream readers that have connected, P the connections a general client
//! holds (1 on a standalone or sentinel store; on a cluster, the primary
//! count, as a cluster client connects to every primary), c is 1 on a cluster
//! and 0 otherwise, and:
//!
//! - **general connection**: the client's one warm connection, at
//!   `kept_queue_bytes + kept_write_bytes + kept_read_bytes` of
//!   [`GENERAL_LIMITS`].
//! - **idle connection**: [`IDLE_CONNECTION_BYTES`], every other connection
//!   of the client once drained: no queue or write-buffer allocation and an
//!   8 KiB read buffer. This is the only term that grows with the primary
//!   count.
//! - **reader connection**: `kept_queue_bytes + kept_write_bytes` of
//!   [`READER_LIMITS`]. A stream reader is a single-node client on every
//!   topology: it holds one connection, to its bound slot's owner, which is
//!   therefore always its warm one, and a reader that moves to another node
//!   retires its client, buffers and all. Its read buffer keeps less than
//!   `kept_read_bytes` of [`READER_LIMITS`], 256 KiB, which is within
//!   [`XREAD_REPLY_TARGET_BYTES`]: the per-reader reply memory the stream
//!   multiplexer already accounts for, so it is not counted again here.
//! - **backchannel**: [`BACKCHANNEL_BYTES`], tokio-util's two 8 KiB buffers.
//!   fred opens a backchannel connection lazily, beside a client's pipelined
//!   connections:
//!   - each reader's, for `CLIENT UNBLOCK`, opened the first time a command
//!     follows a client-side XREAD timeout (`Blocking::Interrupt`, readers
//!     only): the R term;
//!   - on a cluster, each general client's, for `CLUSTER SLOTS`, re-created
//!     at every topology sync of a healthy client: the c term.
//!
//!   Once a backchannel holds nothing, the router's next sweep returns both
//!   its buffers to fresh 8 KiB ones if it carried anything since its last
//!   cut, so this term is exact whatever the replies' sizes. The c term is
//!   exact only because the cluster routing table owns its strings
//!   (`fred::live_edit::retention::own_slot_ranges`): fred parses
//!   `CLUSTER SLOTS` without copying, and a routing string kept as a view
//!   would keep a whole read-buffer allocation alive past the sync, and with
//!   it every `Server` cloned from the routing, a stream reader's included.
//! - **retry buffer**: nothing. A client's retry buffer is returned to an
//!   unallocated one once its replay empties it.
//!
//! **K, member replacement** ([`PoolSizes::replacement_clients`]). A pool
//! member or `txn` slot whose client keeps failing is replaced
//! (`storage::member_replacement`): a fresh client, built from the same
//! configuration and so with the same [`GENERAL_LIMITS`], takes the member
//! over, and the old client is retired, still serving the commands already
//! handed to it through its grace. Each such client keeps what any general
//! client keeps, so K bounds what can exist at once beside the G
//! members:
//!
//! - **retiring clients**: `MEMBER_RETIRING_CAP` = 4. At most that many old
//!   clients retire at once, whatever their pool: a swap past the per-replica
//!   cap ends the oldest retirement other than its own early, a `maintenance`
//!   retirement inside its grace included.
//! - **connecting clients**: one per pool with members. A replacement holds
//!   its pool's slot from the trigger through the swap, so each pool connects
//!   one fresh client at a time beside the member it will replace.
//!
//! Both are transient: a retirement ends within its grace plus
//! `MEMBER_RETIRE_QUIT_BOUND`, and its router's end drops its pipelined
//! connections; a connect ends at the swap or fails within the connect
//! timeout. The bound counts K whether or not a replacement is in progress, so
//! it holds through one; with none in progress, K's clients do not exist.
//!
//! What K does not count, outside the bound:
//!
//! - **a retired client's backchannel, after its router ends.** A router's
//!   end does not drop the client's backchannel;
//!   `fred::live_edit::pool::retire` closes it on a detached task up to about
//!   6 s later (`RETIRE_ROUTER_WAIT` plus
//!   `RETIRE_BACKCHANNEL_CLOSE_TIMEOUT`), after the retirement has left the
//!   count.
//! - **an evicted retirement, while its end runs.** The per-replica cap
//!   removes an evicted retirement from the count at once
//!   (`Retiring::evict_over_cap`), while its router's abort and the release
//!   that follows are still running.
//! - **a backchannel left open in a spare cell.** When that close times out
//!   still waiting for the backchannel's lock (`fred::live_edit::pool::retire`
//!   documents when), the backchannel stays open on the retired client in its
//!   pool's spare cell until the pool drops, and no sweep cuts it again (its
//!   router has ended). At most one per spare cell.
//!
//! The terms count connection buffers only, which holds because nothing kept
//! past a call views a read buffer. fred parses replies without copying, and
//! keeps two values built from them: the cluster routing table and, on a
//! sentinel store, the primary `Server` a sentinel names, which every client
//! (readers included) keeps for its connection's life. The vendored fred
//! copies both out of the reply
//! (`fred::live_edit::retention::{own_slot_ranges, own_server}`), so neither
//! keeps a read buffer alive, and a sentinel store's P = 1 adds nothing per
//! client beyond its one connection.
//!
//! The per-holder bounds (`kept_*_bytes`) and their derivation are on
//! [`fred::live_edit::retention::RetentionLimits`].

use fred::live_edit::retention::{RetentionLimits, FRESH_BUFFER_BYTES};
use fred::types::Builder;

use crate::config::Config;
use crate::storage::client::{
    effective_maintenance_pool_size, effective_pool_size, effective_subscriber_pool_size,
};
use crate::storage::member_replacement::MEMBER_RETIRING_CAP;
use crate::storage::stream_mux::XREAD_REPLY_TARGET_BYTES;

/// Limits for the stream readers, whose one connection each carries a
/// reader's XREADs, one at a time, and the checks between them.
///
/// - `in_flight_commands` 64: fred's initial queue size; a reader has one
///   command in flight at a time.
/// - `write_buffer_bytes` 64 KiB: a reader writes only its XREADs and checks,
///   which are small next to its replies.
/// - `read_buffer_bytes` 64 KiB: a read buffer can keep just under four times
///   the larger of the largest frame it held and 64 KiB
///   (`RetentionLimits::kept_read_bytes`: the decoder leaves consumed replies
///   as an offset in the same allocation, growth doubles from there, and a
///   shared buffer's reallocation carries over up to 64 KiB of capacity), so
///   a drained reader's read buffer keeps under 256 KiB, a quarter of
///   [`XREAD_REPLY_TARGET_BYTES`], inside the per-reader reply memory the
///   stream multiplexer accounts for. Below 64 KiB that bound stays at
///   256 KiB, so a lower limit would only regrow more often. Stream replies
///   are small at realistic load (updates of a few hundred bytes); a reply
///   above 64 KiB costs a regrowth on a later reply above 8 KiB, never
///   retained memory.
pub const READER_LIMITS: RetentionLimits = RetentionLimits {
    in_flight_commands: 64,
    write_buffer_bytes: 64 * 1024,
    read_buffer_bytes: 64 * 1024,
};

// [`at_rest_bound_bytes`] leaves a reader's read buffer to the per-reader
// reply memory, which holds only while what the buffer keeps once drained
// stays within one reply target.
const _: () = assert!(READER_LIMITS.kept_read_bytes() <= XREAD_REPLY_TARGET_BYTES as usize);

/// Limits for every other connection: the multiplexed data, maintenance and
/// snapshot pools, the exclusive txn pool and the pub/sub client: 256
/// in-flight commands, 64 KiB written, 64 KiB per frame read.
///
/// Steady traffic's pipelines, commands and replies fit within them without a
/// reallocation per call. Measured at the chart's pools under the load
/// generator's realistic shape (200 subscribers, 32 writers, 1,000 writes/s of
/// 200 B updates, a burst of 250 subscriptions), these limits with
/// [`READER_LIMITS`] were
/// indistinguishable from 1,024 in-flight commands with a 256 KiB reader
/// read limit, and from no limits, in writes and deliveries per second and in
/// catch-up time. A pipeline above 256 commands, or a write or frame above
/// 64 KiB, costs one regrowth per call; that growth belongs to the call and
/// is let go at the connection's next drain. As for the readers, a read limit
/// below 64 KiB would keep the same `kept_read_bytes` and only regrow more
/// often.
pub const GENERAL_LIMITS: RetentionLimits = RetentionLimits {
    in_flight_commands: 256,
    write_buffer_bytes: 64 * 1024,
    read_buffer_bytes: 64 * 1024,
};

/// What one backchannel connection keeps: tokio-util's `Framed` read and
/// write buffers at their 8 KiB starting size, which the router's sweep
/// restores once the backchannel holds nothing after any use.
pub const BACKCHANNEL_BYTES: usize = 2 * FRESH_BUFFER_BYTES;

/// What a drained connection that is not its client's warm connection keeps:
/// the idle shape's 8 KiB read buffer, with no queue or write-buffer
/// allocation.
pub const IDLE_CONNECTION_BYTES: usize = FRESH_BUFFER_BYTES;

/// The one pub/sub client `connect` builds.
const PUBSUB_CLIENTS: usize = 1;

/// Set `limits` on every connection `builder` builds. A `ConnectionConfig`
/// setting, like the connect timeout: it changes how much a connection keeps
/// once drained, never what is sent or how a connection authenticates, so it
/// is not a divergence in the shared auth / TLS policy.
pub(crate) fn apply_limits(builder: &mut Builder, limits: RetentionLimits) {
    builder.with_connection_config(|connection| {
        connection.live_edit.retention = Some(limits);
    });
}

/// The client count of each pool `connect` builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolSizes {
    /// The data pool.
    pub data: usize,
    /// The txn pool.
    pub txn: usize,
    /// The maintenance pool.
    pub maintenance: usize,
    /// The snapshot pool.
    pub snapshot: usize,
    /// The stream readers: the most that can have connected (a reader
    /// connects the first time it serves a subscription).
    pub readers: usize,
}

impl PoolSizes {
    /// The pool sizes `connect` builds for `config`, with the same resolution
    /// of zero ("auto") values.
    pub fn from_config(config: &Config) -> Self {
        let pool_size = effective_pool_size(config.valkey_pool_size);
        Self {
            data: pool_size,
            txn: pool_size,
            maintenance: effective_maintenance_pool_size(config.maintenance_pool_size),
            snapshot: config.snapshot_pool_size as usize,
            readers: effective_subscriber_pool_size(config.subscriber_pool_size),
        }
    }

    /// The clients that carry [`GENERAL_LIMITS`]: every pool but the readers,
    /// and the pub/sub client.
    pub fn general_clients(&self) -> usize {
        self.data + self.txn + self.maintenance + self.snapshot + PUBSUB_CLIENTS
    }

    /// K: a bound on the general-class clients a member replacement
    /// (`storage::member_replacement`) can add beside
    /// [`Self::general_clients`] at once: the clients retiring plus one fresh
    /// client connecting per pool with members (module docs, *K, member
    /// replacement*).
    pub fn replacement_clients(&self) -> usize {
        self.retiring_clients() + self.connecting_clients()
    }

    /// The most old clients retiring at once: `MEMBER_RETIRING_CAP` = 4,
    /// whatever the pool sizes. The per-replica cap ends the oldest
    /// retirement other than the swap's own early, a `maintenance` one inside
    /// its grace included, so no pool's retirements can exceed it.
    fn retiring_clients(&self) -> usize {
        MEMBER_RETIRING_CAP
    }

    /// The most fresh clients connecting at once: one per replaceable pool
    /// (data, txn, maintenance, snapshot) with members, since a replacement
    /// holds its pool's slot from the trigger through the swap.
    fn connecting_clients(&self) -> usize {
        [self.data, self.txn, self.maintenance, self.snapshot]
            .into_iter()
            .filter(|&size| size > 0)
            .count()
    }
}

/// The store topology [`at_rest_bound_bytes`] is computed for: how many
/// connections a general client holds (P) and whether the store is a cluster
/// (c).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Topology {
    primaries: usize,
    clustered: bool,
}

impl Topology {
    /// A standalone or sentinel-managed store: every client holds one
    /// connection and no backchannel for topology syncs.
    pub const fn standalone() -> Self {
        Self {
            primaries: 1,
            clustered: false,
        }
    }

    /// A cluster with `primaries` primaries (0 is taken as 1): a general
    /// client holds a connection to each, plus a `CLUSTER SLOTS` backchannel.
    pub const fn cluster(primaries: usize) -> Self {
        Self {
            primaries: if primaries == 0 { 1 } else { primaries },
            clustered: true,
        }
    }

    /// The connections a general client holds: P.
    pub const fn primaries(&self) -> usize {
        self.primaries
    }

    /// Whether the store is a cluster: c.
    pub const fn clustered(&self) -> bool {
        self.clustered
    }
}

/// The most every Valkey connection of one replica keeps once its traffic
/// drains, in bytes, beyond the per-reader reply memory, with `pools` and on
/// `topology`: `(G + K) × (general connection + (P − 1) × idle connection + c
/// × backchannel) + R × (reader connection + backchannel)` (see the module
/// docs for the terms). K ([`PoolSizes::replacement_clients`]) counts the
/// retiring and connecting clients of member replacements, which exist only
/// while a replacement is in progress but are counted always, so the bound
/// holds through one. Not counted (module docs, *What K does not count*): a
/// retired client's backchannel for up to about 6 s after its router ends, an
/// evicted retirement while its abort and release run, and the rare
/// backchannel a timed-out close leaves open in a spare cell, at most one per
/// spare cell, until its pool drops. Each general client keeps one warm
/// connection at its class limits whatever P is, so the bound grows by only
/// (G + K) × 8 KiB per primary.
pub fn at_rest_bound_bytes(pools: PoolSizes, topology: Topology) -> u64 {
    let general_connection = GENERAL_LIMITS.kept_queue_bytes()
        + GENERAL_LIMITS.kept_write_bytes()
        + GENERAL_LIMITS.kept_read_bytes();
    let cluster_backchannel = if topology.clustered() {
        BACKCHANNEL_BYTES
    } else {
        0
    };
    let general_client = general_connection
        + (topology.primaries() - 1) * IDLE_CONNECTION_BYTES
        + cluster_backchannel;
    let reader_client =
        READER_LIMITS.kept_queue_bytes() + READER_LIMITS.kept_write_bytes() + BACKCHANNEL_BYTES;
    let general_clients = pools.general_clients() + pools.replacement_clients();
    (general_clients * general_client + pools.readers * reader_client) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use fred::bytes::Bytes;
    use fred::bytes_utils::Str;
    use fred::live_edit::pool::MIN_SPARE_CELLS;
    use fred::live_edit::retention::{
        action, cut_backchannel, elect, elect_streaming, own_server, own_slot_ranges, plan, Action,
        ConnectionView, IN_FLIGHT_SLOT_BYTES,
    };
    use fred::types::cluster::SlotRange;
    use fred::types::config::Server;
    use std::time::{Duration, Instant};

    /// The chart's pools (`helm/live-edit-service/values.yaml`): `pool_size`
    /// 64 for the data and txn pools, `maintenance_pool_size` 1, the snapshot
    /// lane the chart derives at that pool size (6), and
    /// `subscriber_pool_size` 256 stream readers. With the pub/sub client
    /// that is G = 136 general clients and R = 256 readers.
    const CHART_POOLS: PoolSizes = PoolSizes {
        data: 64,
        txn: 64,
        maintenance: 1,
        snapshot: 6,
        readers: 256,
    };

    /// The in-flight slot, one fred `Command` (`IN_FLIGHT_SLOT_BYTES`), the
    /// documented figures below are taken at: its size in the shipped build,
    /// with the `valkey-tls` feature on. Without that feature `Command` is
    /// smaller, so every figure that counts queue slots is smaller by the
    /// difference per slot: a build without it checks each figure moved by
    /// that ([`at_this_builds_slot`]), and the shipped build checks the
    /// figures exactly
    /// ([`the_documented_figures_are_at_the_shipped_builds_in_flight_slot`]).
    const DOCUMENTED_SLOT_BYTES: usize = 376;

    /// `documented`, a figure that counts `slots` in-flight slots at
    /// [`DOCUMENTED_SLOT_BYTES`] each, at this build's slot size: the figure
    /// itself in the shipped build.
    fn at_this_builds_slot(documented: u64, slots: usize) -> u64 {
        let per_slot = IN_FLIGHT_SLOT_BYTES as i64 - DOCUMENTED_SLOT_BYTES as i64;
        u64::try_from(documented as i64 + slots as i64 * per_slot)
            .expect("a figure at this build's slot size is not negative")
    }

    /// The in-flight slots `at_rest_bound_bytes` counts at `pools`: one
    /// warm queue per general client ([`GENERAL_LIMITS`]) and one per reader
    /// ([`READER_LIMITS`]), whatever the topology.
    fn slots_at(pools: PoolSizes) -> usize {
        (pools.general_clients() + pools.replacement_clients()) * GENERAL_LIMITS.in_flight_commands
            + pools.readers * READER_LIMITS.in_flight_commands
    }

    /// The documented figures are the shipped build's: with the
    /// `valkey-tls` feature on, one in-flight slot is
    /// [`DOCUMENTED_SLOT_BYTES`], so [`at_this_builds_slot`] leaves every
    /// figure as documented and the tests below compare them exactly.
    #[cfg(feature = "valkey-tls")]
    #[test]
    fn the_documented_figures_are_at_the_shipped_builds_in_flight_slot() {
        assert_eq!(
            IN_FLIGHT_SLOT_BYTES, DOCUMENTED_SLOT_BYTES,
            "fred's Command, one in-flight slot, in the shipped build (valkey-tls on): the \
             documented connection-buffer figures, and the memory-line record's C_conn, are \
             taken at this size"
        );
    }

    /// `at_rest_bound_bytes` at the chart pools, formula (B): `(G + K) ×
    /// (505,856 + (P − 1) × 8,192 + c × 16,384) + R × (171,520 + 16,384)`,
    /// with G = 136 general clients, K = 8 member-replacement clients and
    /// R = 256 readers. One warm general connection keeps 505,856 B at
    /// [`GENERAL_LIMITS`]: a 256-slot queue (96,256 B), the write bound
    /// (147,456 B) and the read bound (262,144 B). Each general client keeps
    /// one warm connection at its class limits, every other connection at the
    /// idle shape (an 8 KiB read buffer) and, on a cluster (`c`), its
    /// `CLUSTER SLOTS` backchannel's two 8 KiB buffers; each reader keeps its
    /// one connection and its `CLIENT UNBLOCK` backchannel. Dropping either
    /// backchannel term or K, going back to pass A's formula (every
    /// connection of a general client at its class limits), or going back to
    /// a 1,024-command general in-flight limit, changes these values.
    ///
    /// The `c` term is exact, not a floor: the cluster routing table owns its
    /// strings (`fred::live_edit::retention::own_slot_ranges`), so no value
    /// fred keeps past a sync holds a reply's read buffer alive beyond the
    /// backchannel's own two buffers.
    ///
    /// The figures are at the shipped build's in-flight slot
    /// ([`DOCUMENTED_SLOT_BYTES`]: 256 × 376 B is the 96,256 B queue); a
    /// build without `valkey-tls` checks them at its own slot size.
    #[test]
    fn at_rest_bound_bytes_at_the_chart_pools_is_the_documented_figure_per_topology() {
        assert_eq!(
            (
                CHART_POOLS.general_clients(),
                CHART_POOLS.replacement_clients()
            ),
            (136, 8),
            "fixture self-check: the chart's general clients (G) and member-replacement \
             clients (K)"
        );
        for (topology, want) in [
            (Topology::standalone(), 120_946_688),
            (Topology::cluster(1), 123_305_984),
            (Topology::cluster(3), 125_665_280),
            (Topology::cluster(10), 133_922_816),
        ] {
            assert_eq!(
                at_rest_bound_bytes(CHART_POOLS, topology),
                at_this_builds_slot(want, slots_at(CHART_POOLS)),
                "at_rest_bound_bytes at the chart pools on {topology:?} (documented {want} B at \
                 {DOCUMENTED_SLOT_BYTES} B per in-flight slot; this build's slot is \
                 {IN_FLIGHT_SLOT_BYTES} B)"
            );
        }
    }

    /// A cluster of 0 primaries is taken as 1: `Topology::cluster(0)` is
    /// `Topology::cluster(1)`, so `at_rest_bound_bytes`'s `(P − 1)` idle
    /// connections term cannot underflow on a routing table that names no
    /// primary.
    #[test]
    fn a_cluster_of_zero_primaries_is_taken_as_one() {
        assert_eq!(Topology::cluster(0), Topology::cluster(1));
        assert_eq!(
            at_rest_bound_bytes(CHART_POOLS, Topology::cluster(0)),
            at_this_builds_slot(123_305_984, slots_at(CHART_POOLS)),
            "the bound at the chart pools on a cluster of 0 primaries is the one-primary figure"
        );
    }

    /// The readers' term does not grow with the store's primaries: a stream
    /// reader is a single-node client, one connection and at most one
    /// backchannel on every topology. It is the bound with the chart's 256
    /// readers less the bound with none, the same on a standalone store and
    /// on a cluster of any size: 256 × (171,520 + 16,384), at the shipped
    /// build's in-flight slot ([`DOCUMENTED_SLOT_BYTES`]; a build without
    /// `valkey-tls` checks it at its own slot size).
    #[test]
    fn at_rest_bound_bytes_reader_term_is_the_same_at_every_primary_count() {
        let no_readers = PoolSizes {
            readers: 0,
            ..CHART_POOLS
        };
        for topology in [
            Topology::standalone(),
            Topology::cluster(1),
            Topology::cluster(3),
            Topology::cluster(10),
        ] {
            assert_eq!(
                at_rest_bound_bytes(CHART_POOLS, topology)
                    - at_rest_bound_bytes(no_readers, topology),
                at_this_builds_slot(
                    256 * (171_520 + 16_384),
                    256 * READER_LIMITS.in_flight_commands
                ),
                "the reader term on {topology:?}"
            );
        }
    }

    /// K, `PoolSizes::replacement_clients`: the retiring clients,
    /// `MEMBER_RETIRING_CAP` whatever the pool sizes, plus one connecting
    /// client per replaceable pool with members. The per-replica cap ends the
    /// oldest retirement other than a swap's own early, a `maintenance`
    /// retirement inside its grace included, so the `maintenance` pool's
    /// spare cells (`max(size, MIN_SPARE_CELLS)`) never raise the retiring
    /// term. The rows vary only what could move it, and the connecting count:
    ///
    /// - the chart pools: a `maintenance` pool of 1 and four pools connect:
    ///   K = `MEMBER_RETIRING_CAP` + 4 = 8, the figure `at_rest_bound_bytes`
    ///   counts at the chart;
    /// - a `maintenance` pool of 3, at or under the spare-cell floor, and one
    ///   above it: the retiring term is still the cap, not its spare cells
    ///   + 1;
    /// - no `maintenance` pool: the cap still binds, and only three pools
    ///   connect;
    /// - an empty snapshot pool: a pool with no members connects no client.
    ///
    /// Dropping the connecting term fails every row's K; deriving the
    /// retiring term from the `maintenance` pool's spare cells (`M + 1`, with
    /// or without the cap's `max`) fails the chart and both `maintenance`
    /// pool rows; counting a pool with no members fails the no-`maintenance`
    /// and empty-snapshot rows.
    #[test]
    fn replacement_clients_count_the_retiring_clients_and_one_connecting_client_per_pool_with_members(
    ) {
        let above_floor = MIN_SPARE_CELLS + 4;
        let rows = [
            ("the chart pools", CHART_POOLS, (MEMBER_RETIRING_CAP, 4)),
            (
                "a maintenance pool of 3, under the spare-cell floor",
                PoolSizes {
                    maintenance: 3,
                    ..CHART_POOLS
                },
                (MEMBER_RETIRING_CAP, 4),
            ),
            (
                "a maintenance pool above the spare-cell floor",
                PoolSizes {
                    maintenance: above_floor,
                    ..CHART_POOLS
                },
                (MEMBER_RETIRING_CAP, 4),
            ),
            (
                "no maintenance pool",
                PoolSizes {
                    maintenance: 0,
                    ..CHART_POOLS
                },
                (MEMBER_RETIRING_CAP, 3),
            ),
            (
                "an empty snapshot pool",
                PoolSizes {
                    snapshot: 0,
                    ..CHART_POOLS
                },
                (MEMBER_RETIRING_CAP, 3),
            ),
        ];
        let wrong: Vec<String> = rows
            .iter()
            .filter_map(|(name, pools, (retiring, connecting))| {
                let want = (*retiring, *connecting, retiring + connecting);
                let got = (
                    pools.retiring_clients(),
                    pools.connecting_clients(),
                    pools.replacement_clients(),
                );
                (got != want).then(|| {
                    format!(
                        "{name}: (retiring, connecting, K) = {got:?}, want {want:?} ({pools:?})"
                    )
                })
            })
            .collect();
        assert!(wrong.is_empty(), "replacement_clients: {wrong:#?}");
        assert_eq!(
            CHART_POOLS.replacement_clients(),
            8,
            "K at the chart pools, the figure at_rest_bound_bytes's chart figures count"
        );
    }

    // ── Test 20: the warm election, the per-connection decision and the
    // backchannel cut (`fred::live_edit::retention`'s pure decisions) ──────
    //
    // These are the only deterministic coverage of F4 (the backchannel cut):
    // a `CLUSTER SLOTS` reply big enough to grow a backchannel's buffer needs
    // a fragmented slot map, which the shared CI cluster cannot be put into
    // without disturbing other tests. They test the decisions alone; code
    // review checks that the sweep calls them (`retention::sweep`,
    // `sweep_backchannel`) and that the routing-table and sentinel hooks call
    // `own_slot_ranges` and `own_server`.

    /// A drained connection, not warm and not idle, with read progress `secs`
    /// seconds after `origin`, or none.
    fn conn(origin: Instant, secs: Option<u64>) -> ConnectionView {
        ConnectionView {
            last_progress: secs.map(|s| origin + Duration::from_secs(s)),
            ..ConnectionView::default()
        }
    }

    fn warm(view: ConnectionView) -> ConnectionView {
        ConnectionView { warm: true, ..view }
    }

    /// Each row whose result differs from what it expects, described; every
    /// row is checked, so one run shows every row a change broke.
    fn mismatches<I, O>(rows: &[(&str, I, O)], decide: impl Fn(&I) -> O) -> Vec<String>
    where
        O: PartialEq + std::fmt::Debug,
    {
        rows.iter()
            .filter_map(|(name, input, want)| {
                let got = decide(input);
                (got != *want).then(|| format!("{name}: want {want:?}, got {got:?}"))
            })
            .collect()
    }

    /// One `elect` row: its name, the connections, and the warm connection it
    /// wants.
    type ElectRow<'a> = (&'a str, Vec<ConnectionView>, Option<usize>);

    /// Test 20's `elect` rows, with read progress measured from `t`: a client
    /// with one connection is always warm; the warm connection W keeps warmth
    /// unless it is drained and another connection has strictly newer read
    /// progress; with no W the newest is elected, the first of a tie, and
    /// none when nothing has made progress.
    fn elect_rows(t: Instant) -> Vec<ElectRow<'static>> {
        let in_flight = |v: ConnectionView| ConnectionView { queued: 1, ..v };
        let partial = |v: ConnectionView| ConnectionView {
            read_buffered: 100,
            ..v
        };
        vec![
            ("no connection: none", vec![], None),
            (
                "one connection is warm, with no progress and no W",
                vec![conn(t, None)],
                Some(0),
            ),
            (
                "one connection is warm while in flight",
                vec![in_flight(conn(t, Some(1)))],
                Some(0),
            ),
            (
                "a drained W with a strictly newer N moves warmth to N",
                vec![warm(conn(t, Some(1))), conn(t, Some(2))],
                Some(1),
            ),
            (
                "a drained W that never made progress yields to one that has",
                vec![warm(conn(t, None)), conn(t, Some(1))],
                Some(1),
            ),
            (
                "an in-flight W keeps warmth against a newer N",
                vec![warm(in_flight(conn(t, Some(1)))), conn(t, Some(2))],
                Some(0),
            ),
            (
                "a W holding a partial frame keeps warmth against a newer N",
                vec![warm(partial(conn(t, Some(1)))), conn(t, Some(2))],
                Some(0),
            ),
            (
                "a tie keeps W",
                vec![warm(conn(t, Some(2))), conn(t, Some(2))],
                Some(0),
            ),
            (
                "a tie keeps W when the other connection comes first",
                vec![conn(t, Some(2)), warm(conn(t, Some(2)))],
                Some(1),
            ),
            (
                "an older N does not take warmth",
                vec![conn(t, Some(1)), warm(conn(t, Some(2)))],
                Some(1),
            ),
            (
                "no W: the newest is elected",
                vec![conn(t, Some(1)), conn(t, Some(3)), conn(t, Some(2))],
                Some(1),
            ),
            (
                "no W: the first of a tie for newest is elected",
                vec![conn(t, Some(1)), conn(t, Some(2)), conn(t, Some(2))],
                Some(1),
            ),
            (
                "no W and no progress at all: none",
                vec![conn(t, None), conn(t, None)],
                None,
            ),
        ]
    }

    /// Test 20, `elect`, over [`elect_rows`].
    #[test]
    fn elect_moves_warmth_only_from_a_drained_warm_connection_to_a_strictly_newer_one() {
        let wrong = mismatches(&elect_rows(Instant::now()), |views| elect(views));
        assert!(wrong.is_empty(), "elect: {wrong:?}");
    }

    /// Test 20, `action`: the drained warm connection gets the class rules; a
    /// drained connection that carried traffic since its last cut is cut to
    /// the idle shape; an already-idle connection is not cut again; and a
    /// connection with queued commands, a partial frame or an unflushed frame
    /// is never touched, warm or not.
    #[test]
    fn action_cuts_only_a_drained_connection_that_is_neither_warm_nor_already_idle() {
        let t = Instant::now();
        let busy = |queued, write_buffered, read_buffered| ConnectionView {
            queued,
            write_buffered,
            read_buffered,
            ..conn(t, Some(1))
        };
        let rows: Vec<(&str, ConnectionView, Action)> = vec![
            (
                "the drained warm connection gets the class rules",
                warm(conn(t, Some(1))),
                Action::ClassRules,
            ),
            (
                "a drained connection that carried traffic is cut",
                conn(t, Some(1)),
                Action::Cut,
            ),
            (
                "an already-idle connection is not cut",
                ConnectionView {
                    idle: true,
                    ..conn(t, Some(1))
                },
                Action::Keep,
            ),
            ("queued commands are never cut", busy(3, 0, 0), Action::Keep),
            (
                "a partial frame is never cut",
                busy(0, 0, 100),
                Action::Keep,
            ),
            (
                "an unflushed frame is never cut",
                busy(0, 100, 0),
                Action::Keep,
            ),
            (
                "an in-flight warm connection is left alone",
                warm(busy(1, 0, 0)),
                Action::Keep,
            ),
        ];
        let wrong = mismatches(&rows, action);
        assert!(wrong.is_empty(), "action: {wrong:?}");
    }

    /// One `plan` row: its name, the connections, and the warm connection and
    /// actions it wants.
    type PlanRow<'a> = (&'a str, Vec<ConnectionView>, (Option<usize>, Vec<Action>));

    /// Test 20's `plan` rows, with read progress measured from `t`: the
    /// election and the decision together, with each connection's warm flag
    /// taken from the election. When warmth moves, the old warm connection is
    /// cut in the same pass, so a client never has two connections at its
    /// class limits; an idle third connection is left as it is.
    fn plan_rows(t: Instant) -> Vec<PlanRow<'static>> {
        let idle = ConnectionView {
            idle: true,
            ..conn(t, None)
        };
        vec![
            (
                "warmth moves from a drained W to a newer N: W is cut, N gets the class rules",
                vec![warm(conn(t, Some(1))), conn(t, Some(2)), idle],
                (Some(1), vec![Action::Cut, Action::ClassRules, Action::Keep]),
            ),
            (
                "a tie keeps W: N is cut",
                vec![warm(conn(t, Some(2))), conn(t, Some(2)), idle],
                (Some(0), vec![Action::ClassRules, Action::Cut, Action::Keep]),
            ),
            (
                "a tie keeps W when the other connection comes first: it is cut",
                vec![conn(t, Some(2)), warm(conn(t, Some(2))), idle],
                (Some(1), vec![Action::Cut, Action::ClassRules, Action::Keep]),
            ),
        ]
    }

    /// Test 20, `plan`, over [`plan_rows`].
    #[test]
    fn plan_cuts_the_connection_warmth_moved_away_from_in_the_same_pass() {
        let wrong = mismatches(&plan_rows(Instant::now()), |views| plan(views));
        assert!(wrong.is_empty(), "plan: {wrong:?}");
    }

    /// A test-side model of the sweep's two walks over `views`
    /// (`retention::sweep`): the real election fold, [`elect_streaming`], the
    /// function the sweep's first walk runs over the router's connections;
    /// then, modelled here, the second walk: each connection's [`action`] with
    /// its warm flag set from that election. The sweep's own loop is not run.
    fn streaming_plan(views: &[ConnectionView]) -> (Option<usize>, Vec<Action>) {
        let warm = elect_streaming(views.iter().copied());
        let actions = views
            .iter()
            .enumerate()
            .map(|(i, view)| {
                action(&ConnectionView {
                    warm: warm == Some(i),
                    ..*view
                })
            })
            .collect();
        (warm, actions)
    }

    /// b.26e test 8's table: on every test 20 row, [`elect_streaming`] plus
    /// the model of the sweep's second walk ([`streaming_plan`]) makes the
    /// decisions [`plan`] makes and the row wants — on the `elect` rows the
    /// warm connection, and on the `plan` rows the warm connection and every
    /// action. The sweep collects nothing, so it does not call `plan`; this
    /// table pins the fold it runs and the decision its second walk is meant
    /// to make, not the sweep's own loop, which runs only on a live client:
    /// test 8 (`tests/upload_memory_envelope`, standalone and its cluster
    /// twin) pins that loop, as the one that allocates nothing on an idle pass
    /// (and tests 17 to 19 there, its decisions on a live client). Each row is checked against its own wanted
    /// election as well as `plan`'s, so a change to the fold that `plan`
    /// shares (its tie rule, say) fails here rather than agreeing with itself.
    #[test]
    fn elect_streaming_and_the_second_walk_model_decide_what_plan_decides_on_every_test_20_row() {
        let t = Instant::now();
        let mut wrong: Vec<String> = Vec::new();
        for (name, views, want) in elect_rows(t) {
            let (streaming, _) = streaming_plan(&views);
            let (planned, _) = plan(&views);
            if (streaming, planned) != (want, want) {
                wrong.push(format!(
                    "elect row {name}: want {want:?}, streaming {streaming:?}, plan {planned:?}"
                ));
            }
        }
        for (name, views, want) in plan_rows(t) {
            let streaming = streaming_plan(&views);
            let planned = plan(&views);
            if streaming != want || planned != want {
                wrong.push(format!(
                    "plan row {name}: want {want:?}, streaming {streaming:?}, plan {planned:?}"
                ));
            }
        }
        assert!(wrong.is_empty(), "streaming election vs plan: {wrong:#?}");
    }

    /// Test 20, `cut_backchannel(read_buffered, write_buffered, idle)`: a
    /// backchannel that holds nothing is cut to two fresh 8 KiB buffers once
    /// it carried anything since its last cut — whatever size it carried, so
    /// a 6,144 B `CLUSTER SLOTS` reply (one that can double an 8 KiB buffer
    /// behind its connection's setup replies) is cut too — and is left alone
    /// while it holds a frame either way, or is already idle.
    ///
    /// This is the only deterministic coverage of F4, the backchannel cut: no
    /// test drives a backchannel's buffers in place, because a `CLUSTER
    /// SLOTS` reply large enough to grow one needs a fragmented slot map that
    /// the shared CI cluster cannot be put into without disturbing other
    /// tests. Code review checks that `sweep_backchannel` calls this decision.
    #[test]
    fn cut_backchannel_cuts_an_empty_backchannel_that_carried_anything_since_its_last_cut() {
        let rows: Vec<(&str, (usize, usize, bool), bool)> = vec![
            (
                "empty, carried traffic (the 6,144 B reply case): cut",
                (0, 0, false),
                true,
            ),
            ("empty and already idle: kept", (0, 0, true), false),
            (
                "a partial frame in the read buffer: kept",
                (6_144, 0, false),
                false,
            ),
            (
                "an unflushed frame in the write buffer: kept",
                (0, 64, false),
                false,
            ),
        ];
        let wrong = mismatches(&rows, |&(read, write, idle)| {
            cut_backchannel(read, write, idle)
        });
        assert!(wrong.is_empty(), "cut_backchannel: {wrong:?}");
    }

    // ── Tests 24 and the sentinel copy: kept reply strings own their bytes ──

    /// Size of the stand-in read buffer: the strings fred keeps from a reply
    /// are views into the whole allocation the reply was read into.
    const REPLY_BUFFER_BYTES: usize = 64 * 1024;

    /// A stand-in for a reply's read-buffer allocation, holding each of
    /// `strings` at a distinct offset; returns the buffer and a `Str` view of
    /// each string, made as fred's zero-copy parse makes them
    /// (`Str::from_inner` over a slice of the buffer).
    fn reply_views(strings: &[&str]) -> (Bytes, Vec<Str>) {
        let mut raw = vec![b'.'; REPLY_BUFFER_BYTES];
        let mut at = Vec::with_capacity(strings.len());
        let mut offset = 1_000;
        for s in strings {
            raw[offset..offset + s.len()].copy_from_slice(s.as_bytes());
            at.push((offset, s.len()));
            offset += s.len() + 1_000;
        }
        let buffer = Bytes::from(raw);
        let views = at
            .iter()
            .map(|&(offset, len)| {
                Str::from_inner(buffer.slice(offset..offset + len)).expect("ASCII is UTF-8")
            })
            .collect();
        (buffer, views)
    }

    /// Whether `s`'s bytes lie inside `buffer`'s allocation.
    fn views_into(buffer: &Bytes, s: &str) -> bool {
        let start = buffer.as_ptr() as usize;
        let p = s.as_ptr() as usize;
        p >= start && p < start + buffer.len()
    }

    /// Each string, by name, that does not hold `want` or still views
    /// `buffer`.
    fn still_views(buffer: &Bytes, kept: &[(&str, &str, &str)]) -> Vec<String> {
        kept.iter()
            .filter(|(_, got, want)| got != want || views_into(buffer, got))
            .map(|(name, got, want)| {
                format!(
                    "{name}: {got:?} (want {want:?}), inside the reply buffer: {}",
                    views_into(buffer, got)
                )
            })
            .collect()
    }

    /// Test 24 (b.26e F4b): `own_slot_ranges` leaves every string a cluster
    /// routing table keeps — each range's `id`, its primary's `host` and,
    /// with TLS, its TLS server name — with the same contents in an
    /// allocation of its own, so no kept range holds the `CLUSTER SLOTS`
    /// reply's read buffer alive. Code review checks the call at the end of
    /// `ClusterRouting::rebuild`.
    #[test]
    fn own_slot_ranges_copies_every_kept_string_out_of_the_reply_buffer() {
        let names = [
            "07c37dfeb235213a872192d90877d0cd55635b91",
            "node-1.example.com",
            "node-1.tls.example.com",
            "e7d1eecce10fd6bb5eb35b9f99a514335d9ba9ca",
            "node-2.example.com",
            "node-2.tls.example.com",
        ];
        let (buffer, views) = reply_views(&names);
        let range = |start, end, id: &Str, host: &Str, _tls: &Str| SlotRange {
            start,
            end,
            id: id.clone(),
            primary: {
                #[allow(unused_mut)]
                let mut server = Server::new(host.clone(), 7000);
                #[cfg(feature = "valkey-tls")]
                {
                    server.tls_server_name = Some(_tls.clone());
                }
                server
            },
        };
        let mut ranges = vec![
            range(0, 8_191, &views[0], &views[1], &views[2]),
            range(8_192, 16_383, &views[3], &views[4], &views[5]),
        ];
        drop(views);
        assert!(
            ranges
                .iter()
                .all(|r| views_into(&buffer, &r.id) && views_into(&buffer, &r.primary.host)),
            "fixture self-check: before the copy, the ranges view the reply buffer"
        );

        own_slot_ranges(&mut ranges);

        let mut wrong = Vec::new();
        for (i, r) in ranges.iter().enumerate() {
            let checks: Vec<(&str, &str, &str)> = vec![
                ("id", &r.id, names[3 * i]),
                ("primary host", &r.primary.host, names[3 * i + 1]),
                #[cfg(feature = "valkey-tls")]
                (
                    "primary TLS name",
                    r.primary.tls_server_name.as_deref().unwrap_or(""),
                    names[3 * i + 2],
                ),
            ];
            wrong.extend(
                still_views(&buffer, &checks)
                    .into_iter()
                    .map(|w| format!("range {i} {w}")),
            );
        }
        assert!(
            wrong.is_empty(),
            "own_slot_ranges left a kept string viewing the reply buffer or changed it: {wrong:?}"
        );
    }

    /// The sentinel copy (b.26e): `own_server`
    /// leaves the primary `Server` a sentinel names with its `host` and, with
    /// TLS, its TLS server name holding the same contents in an allocation of
    /// its own, so the primary connection, its keys and the backchannel do
    /// not keep the sentinel connection's read buffer alive. Code review
    /// checks the call in `router/sentinel.rs` (`discover_primary_node`).
    #[test]
    fn own_server_copies_the_sentinel_primarys_strings_out_of_the_reply_buffer() {
        let names = ["10.0.0.7", "primary.tls.example.com"];
        let (buffer, views) = reply_views(&names);
        #[allow(unused_mut)]
        let mut server = Server::new(views[0].clone(), 6379);
        #[cfg(feature = "valkey-tls")]
        {
            server.tls_server_name = Some(views[1].clone());
        }
        drop(views);
        assert!(
            views_into(&buffer, &server.host),
            "fixture self-check: before the copy, the host views the reply buffer"
        );

        own_server(&mut server);

        let checks: Vec<(&str, &str, &str)> = vec![
            ("host", &server.host, names[0]),
            #[cfg(feature = "valkey-tls")]
            (
                "TLS name",
                server.tls_server_name.as_deref().unwrap_or(""),
                names[1],
            ),
        ];
        let wrong = still_views(&buffer, &checks);
        assert!(
            wrong.is_empty(),
            "own_server left a string viewing the reply buffer or changed it: {wrong:?}"
        );
        assert_eq!(server.port, 6379, "the port is unchanged");
    }
}
