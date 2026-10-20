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

//! b.yxt — cluster-gated regression coverage for the live-relay subscriber path.
//!
//! Everything load-bearing in the b.9uq wedged-connection fix, and the b.c7a
//! terminal-error contract layered on top of it, is **cluster-sensitive**:
//!
//! * a multiplexed XREAD may only name keys of one hash slot, so each stream
//!   reader is bound to one slot at a time and its wake key must hash to it;
//! * since b.azh a reader's connection is a single-node client to its bound
//!   slot's owner alone, opened when the reader first serves a subscription:
//!   it must target the owner the routing view names, move (retarget) when the
//!   reader binds a slot another primary owns, and its forced reconnect
//!   rebuilds that one connection while the owner is unchanged;
//! * fred's `"Failed to route command."` — the message b.c7a reclassified as
//!   transient — is emitted from `write_command` when the router's connection
//!   map is momentarily empty, which on a cluster means a topology rebuild.
//!
//! All of it shipped verified only against standalone Valkey. The dev
//! environment where b.9uq actually bit runs a *clustered*, proxied deployment.
//! This module is the closing guard: it re-runs the subscriber contracts
//! against a real 3-shard Valkey Cluster.
//!
//! ## Gating — how these tests activate
//!
//! Every test self-gates on `LIVE_EDIT_TEST_CLUSTER_URL` via
//! [`cluster_fixture`]: unset (or not a `redis-cluster://` URL) prints a `SKIP:`
//! line and returns. So the module compiles and passes everywhere, and CI's
//! `test-cluster:` job — which sets
//! `LIVE_EDIT_TEST_CLUSTER_URL=redis-cluster://127.0.0.1:7000` and runs the
//! unfiltered workspace suite — is what actually executes the bodies. Locally,
//! `bash scripts/dev-valkey.sh cluster-up` brings the same 3-shard cluster up
//! (Linux-only: it relies on Docker `--network host`).
//!
//! A3, B1 and B2 drive the stream multiplexer's fault seam and knobs, so they
//! also need `--features test-fault-injection`. That second condition is the
//! silent one: without the feature they are compiled out, with no `SKIP:`
//! line. CI's `test-cluster:` and `test-cluster-disruptive:` jobs pass it.
//!
//! Skipping is deliberately the *only* soft failure. Once the URL is set,
//! nothing here degrades quietly: [`cluster_fixture`] asserts `is_clustered()`,
//! [`object_ids_on_distinct_shards`] fails loudly if the cluster reports fewer
//! primaries than the scenario needs, and the connection-identity helpers
//! assert the connection they are about to disturb is the one under test.
//!
//! ## Two surfaces
//!
//! * **Release-gating** (`cluster_subscriber_*`) — plain `#[tokio::test]`s that
//!   are safe to run concurrently with the rest of the suite against a shared
//!   cluster. They disturb only connections this test owns.
//! * **Disruptive** (`cluster_subscriber_disruptive_*`) — additionally
//!   `#[ignore]`d, because they pause or reshard a whole cluster **node** and
//!   would perturb any other test process sharing it. CI runs them in a
//!   separate job, `test-cluster-disruptive:` (blocking, like
//!   `test-cluster:`), via
//!   `--run-ignored all -E 'test(/cluster_subscriber_disruptive/)'`. Both are
//!   written to be retry-safe (nextest `retries = 1`): the pause auto-expires
//!   and the migration is completed before the test asserts anything.
//!
//! ## Which b.yxt items each test closes
//!
//! | Test | Ticket item |
//! |---|---|
//! | `cluster_subscriber_delivers_live_updates_across_shards` | multi-shard delivery, one slot-bound reader per shard |
//! | `cluster_subscriber_a_real_non_owner_moved_is_classified_transient` | b.cv4 closing guard — a real non-owner `-MOVED` on a `max_attempts = 1` command vs. the classifier |
//! | `cluster_subscriber_forced_reconnect_rebuilds_the_clustered_connection_map` | a forced reconnect of a cluster reader's one connection, its slot's owner unchanged |
//! | `cluster_subscriber_cursor_survives_connection_loss` | durable-cursor resume across a real connection loss |
//! | `cluster_subscriber_a_client_unblock_on_a_slot_bound_reader_is_replanned_without_loss` | a `-UNBLOCKED` on a cluster reader's one connection takes the cheap re-plan path |
//! | `cluster_subscriber_a_bound_reader_connects_to_its_slot_owner_alone` | b.azh T4/T15 — one connection, to the routing view's `Server` for the slot, TLS name included |
//! | `cluster_subscriber_a_reader_rebound_to_another_primarys_slot_retargets_its_connection` | b.azh T5 — the retarget: the client moves, the old connection is gone from the old node, `cause="retarget"` ticks, delivery holds |
//! | `cluster_subscriber_a_slot_bound_reader_reauths_between_reads_without_interrupting_or_reconnecting` | b.19i — a live subscriber across several reader re-AUTHs: 0 unblocked retries, 0 `subscriber_pool` reconnects, every re-AUTH answered, delivery intact |
//! | `cluster_subscribe_rpc_relays_across_connection_loss` | b.c7a deferred item 3 — RPC-level relay across a routing/connection error |
//! | `cluster_subscriber_observed_errors_are_all_classified_transient` | b.c7a deferred item 1 — real (not hand-built) error text vs. the classifier |
//! | `cluster_subscriber_disruptive_node_pause_wedges_and_self_heals` | real `Timeout` → forced reconnect → self-heal |
//! | `cluster_subscriber_disruptive_ask_redirect_during_slot_migration` | does a LIVE slot migration produce the same budget-exhaustion shapes a pinned-`XLEN` control did? (datapoint reported, not asserted) |
//!
//! ## Wait budgets vs. nextest's hang-killer (b.cv4)
//!
//! `.config/nextest.toml` marks a test slow at 40 s and **hard-terminates it at
//! 120 s** (`terminate-after = 3`). A SIGKILLed test prints nothing: every
//! diagnostic panic message in this file — and they are the whole point of the
//! file — is lost exactly when a genuine failure needs reading. So each test's
//! worst case must finish *inside* that window, with room for the failure to be
//! formatted and reported.
//!
//! The budgets are the module constants below ([`PARK_BUDGET`],
//! [`DELIVERY_BUDGET`], [`CATCH_UP_BUDGET`], [`SELF_HEAL_DELIVERY_BUDGET`],
//! [`RECONNECT_BARRIER_BUDGET`]) rather than inline literals, so the arithmetic
//! in this table can be re-derived rather than re-guessed. Worst case = every
//! wait in the test consuming its whole budget:
//!
//! | Test | Waits | Worst case |
//! |---|---|---|
//! | A1 `…delivers_live_updates_across_shards` | 2 × delivery 15 s + 2 × quiet 0.75 s | **31.5 s** |
//! | A2 `…a_real_non_owner_moved_is_classified_transient` | two single-attempt commands, each bounded by the client's command timeout | **≈ 10 s** |
//! | A3 `…forced_reconnect_rebuilds…` | fault-drain 10 s¹ + delivery 15 s | **25 s** |
//! | A4 `…cursor_survives_connection_loss` | identify + park 15 s³ + 2 × delivery 15 s + quiet 0.75 s | **45.75 s** |
//! | A7 `…a_client_unblock_on_a_slot_bound_reader…` | identify + park 15 s³ + unblock retries 15 s + delivery 15 s | **45 s** |
//! | A8 `…a_bound_reader_connects_to_its_slot_owner_alone` | delivery 15 s | **15 s** |
//! | A9 `…rebound_to_another_primarys_slot_retargets…` | 2 × delivery 15 s + identify 15 s + old connection's close 15 s | **60 s** |
//! | A10 `…a_slot_bound_reader_reauths_between_reads…` | delivery 15 s + identify + park 15 s³ + re-AUTHs 15 s + delivery 15 s | **60 s** |
//! | A5 `cluster_subscribe_rpc_relays_across_connection_loss` | catch-up 20 s² + identify + park 15 s³ + delivery 15 s | **50 s** |
//! | A6 `…observed_errors_are_all_classified_transient` | identify + park 15 s³ + delivery 15 s | **30 s** |
//! | B1 `…disruptive_node_pause_wedges_and_self_heals` | identify + park 15 s³ + pause 9 s + barrier 30 s + self-heal delivery 30 s | **84 s** |
//! | B2 `…disruptive_ask_redirect_during_slot_migration` | delivery 15 s + observation 15 s + post-hand-over delivery 15 s | **45 s**⁴ |
//!
//! ¹ `await_faults_consumed`'s own deadline (`mux_test_helpers::SETTLE`) —
//!   shared with the single-node tests, so it is not re-tuned here.
//! ² A5's drain was the real defect: `drain_catch_up`'s 16 iterations each held
//!   a *separate* 30 s budget (16 × 30 s = 480 s, plus a fresh 30 s per body
//!   chunk). It is now ONE deadline spanning the whole catch-up.
//! ³ Finding the reader's connection and waiting for it to park in its
//!   `XREAD` share ONE [`PARK_BUDGET`] deadline ([`identify_parked_reader`]):
//!   as two budgets, each test's worst case would grow by 15 s, B1's to 99 s
//!   against the 120 s kill. A9 identifies the connection with nothing to park
//!   for, so its identify is its own 15 s.
//! ⁴ B2's *waits* are 45 s, but it also issues ~8 migration primitives each
//!   bounded by [`MIGRATION_COMMAND_TIMEOUT`] (20 s). Those only approach their
//!   budget on a cluster that has stopped answering — in which case the first
//!   one fails and the test panics rather than compounding. The one path that
//!   could genuinely compound was the completion drain (up to
//!   [`MIGRATION_DRAIN_MAX_BATCHES`] × 2 commands × 20 s ≈ 21 min); it now
//!   carries its own wall-clock bound, [`MIGRATION_DRAIN_BUDGET`].
//!
//! Budgets are *deadlines*, not sleeps: the healthy path returns in
//! milliseconds and none of this is paid. They are sized to be unreachable
//! under ordinary CI-runner load while still leaving the hang-killer as the
//! backstop it is meant to be rather than the primary failure surface.

use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{ClientInterface, ClientLike, ClusterInterface, StreamsInterface};
use fred::types::client::ClientKillFilter;
use fred::types::config::{Options, Server};
use fred::types::ClientUnblockFlag;

use live_edit::observability::metrics::{Metrics, CONSUMER_PHASE_XREAD};
use live_edit::storage::notifications::{
    consume_layer_stream, LayerNotification, LayerStreamConsumer, StreamCursor,
};
use live_edit::storage::stream_mux::router::Routing;
use live_edit::storage::test_support::TestValkey;

use super::mux_test_helpers::{install_metrics, read_permanent_exits, xadd_update};

// Only the fault-injected and knob-driven tests (A3, B1, B2) use these.
#[cfg(feature = "test-fault-injection")]
use fred::types::client::ClientPauseKind;
#[cfg(feature = "test-fault-injection")]
use live_edit::storage::stream_mux::StreamMuxKnobs;

#[cfg(feature = "test-fault-injection")]
use super::mux_test_helpers::{
    await_faults_consumed, fast_backoff_knobs, push_timeout_faults, read_forced_reconnects,
    read_series,
};

// ═════════════════════════════════════════════════════════════════════════════
// Wait budgets (b.cv4) — see the module-level table for the arithmetic
// ═════════════════════════════════════════════════════════════════════════════

/// How long a `CLIENT LIST`-polled barrier waits for a connection to report the
/// command it is supposed to be running ([`await_reserved_last_command`]).
///
/// This is a *server-side observation* barrier: the consumer has already been
/// constructed and its first command is one network round-trip away, polled
/// every 50 ms. 15 s is ~three orders of magnitude of headroom over the healthy
/// case; anything slower is a wedge, which is a finding, not a slow runner.
const PARK_BUDGET: Duration = Duration::from_secs(15);

/// How long any single notification / stream message may take to arrive once the
/// entry it carries has already been written.
///
/// The consumer is parked in `XREAD BLOCK` and Valkey pushes on `XADD`, so the
/// healthy path is a single round-trip. The budget only has to cover a
/// connection loss plus one reconnect and a re-issued read — the loop's own
/// backoff caps at `XREAD_TRANSIENT_BACKOFF_CAP`, well inside this.
const DELIVERY_BUDGET: Duration = Duration::from_secs(15);

/// A SINGLE deadline covering the WHOLE of [`drain_catch_up`] — all of its
/// iterations and every body chunk inside them.
///
/// This shape is the point (b.cv4). A per-message budget multiplies: 16
/// iterations × 30 s, each iteration itself able to spend a fresh 30 s per body
/// chunk, put A5's worst case in the minutes while the nextest hang-killer
/// terminates at 120 s — so a catch-up that stalled was SIGKILLed and A5's
/// diagnostic panics never printed. One shared deadline makes the worst case
/// equal to the budget, whatever the message count turns out to be.
const CATCH_UP_BUDGET: Duration = Duration::from_secs(20);

/// B1 only: delivery on a connection the consumer has just rebuilt after a
/// node-wide pause. Wider than [`DELIVERY_BUDGET`] because the path includes a
/// full forced reconnect — a check of the slot's owner in the routing view,
/// then a teardown and rebuild of the reader's connection — on a node that has
/// just come back from being frozen.
#[cfg(feature = "test-fault-injection")]
const SELF_HEAL_DELIVERY_BUDGET: Duration = Duration::from_secs(30);

/// B1 only: how long after the `CLIENT PAUSE` lapses the forced-reconnect
/// counter may take to move. Measured from the pause expiry, not from the poll
/// start, so the pause duration itself is not charged against it.
#[cfg(feature = "test-fault-injection")]
const RECONNECT_BARRIER_BUDGET: Duration = Duration::from_secs(30);

/// B2 only: how long the consumer is watched while the slot is half-migrated.
///
/// Unlike every other constant here this is a *sampling window*, not a deadline
/// — elapsing is one of the outcomes the experiment reports, not a failure. It
/// therefore costs its full value on the healthy path, which is why it is
/// counted in full in the module table.
#[cfg(feature = "test-fault-injection")]
const MIGRATION_OBSERVATION_WINDOW: Duration = Duration::from_secs(15);

// ═════════════════════════════════════════════════════════════════════════════
// Gating + cluster-topology helpers
// ═════════════════════════════════════════════════════════════════════════════

/// The gate. `None` means "not configured — the caller must return", and a
/// `SKIP:` line has already been printed explaining why.
///
/// Mirrors the established pattern at
/// `tests/integration/storage/client_test.rs` and
/// `tests/integration/checkpoint/e2e_cluster_recovery_test.rs`: read the env
/// var, insist on the `redis-cluster://` scheme, connect, and then assert
/// `is_clustered()` so a single-node URL smuggled in through the cluster gate
/// cannot silently turn every test in this file into a duplicate of its
/// single-node sibling.
///
/// `TestValkey::setup_with_url` panics if the cluster is unreachable, so a down
/// cluster fails loudly rather than skipping.
async fn cluster_fixture(valkey_pool_size: u32) -> Option<TestValkey> {
    cluster_fixture_with_readers(valkey_pool_size, 4).await
}

/// As [`cluster_fixture`], with `readers` stream-multiplexer reader connections.
async fn cluster_fixture_with_readers(valkey_pool_size: u32, readers: u32) -> Option<TestValkey> {
    cluster_fixture_inner(valkey_pool_size, readers, None).await
}

/// As [`cluster_fixture_with_readers`], with `provider` installed on every
/// connection (b.19i): the passwordless auth path against the cluster, whose
/// `default` user has no password, so any password authenticates.
async fn cluster_fixture_with_provider(
    valkey_pool_size: u32,
    readers: u32,
    provider: Arc<dyn fred::types::config::CredentialProvider>,
) -> Option<TestValkey> {
    cluster_fixture_inner(valkey_pool_size, readers, Some(provider)).await
}

async fn cluster_fixture_inner(
    valkey_pool_size: u32,
    readers: u32,
    provider: Option<Arc<dyn fred::types::config::CredentialProvider>>,
) -> Option<TestValkey> {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!(
            "SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured); \
             run `bash scripts/dev-valkey.sh cluster-up` or let CI's `test-cluster:` job run this"
        );
        return None;
    };
    if !cluster_url.starts_with("redis-cluster://") {
        eprintln!(
            "SKIP: LIVE_EDIT_TEST_CLUSTER_URL is not a plaintext redis-cluster:// URL \
             (got a URL with a different scheme)"
        );
        return None;
    }

    let tv = match provider {
        Some(provider) => {
            TestValkey::setup_with_url_readers_and_credential_provider(
                cluster_url,
                valkey_pool_size,
                readers,
                provider,
            )
            .await
        }
        None => {
            TestValkey::setup_with_url_and_readers(cluster_url, valkey_pool_size, readers).await
        }
    };
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment (use \
         `redis-cluster://...`). Every assertion in this module is about behaviour that only \
         exists in cluster mode — per-node connection maps, hash-slot routing, cross-shard \
         delivery — so a non-clustered fixture here would pass vacuously and report cluster \
         coverage the suite does not have"
    );
    assert_eq!(
        tv.client().stream_mux.snapshot().routing,
        Routing::SlotAffinity,
        "a clustered client's stream readers must bind to hash slots"
    );
    Some(tv)
}

/// The cluster hash slot a key routes to.
///
/// Routed through `fred::util::group_by_hash_slot` — the same helper
/// `src/storage/test_support.rs` uses for its cluster-aware key deletion — so
/// the test's notion of "which slot" is by construction the one fred's router
/// will compute for the very same key.
fn hash_slot_of(key: &str) -> u16 {
    let groups = fred::util::group_by_hash_slot([key])
        .unwrap_or_else(|e| panic!("group_by_hash_slot({key}) must succeed: {e}"));
    assert_eq!(
        groups.len(),
        1,
        "one key must hash to exactly one slot; got {} groups for {key}",
        groups.len()
    );
    *groups
        .keys()
        .next()
        .expect("a single-entry BTreeMap has a first key")
}

/// The primary that currently owns `slot`, from the client's cached
/// `CLUSTER SLOTS` view — i.e. exactly the node fred's router will send a
/// command for that slot to.
fn slot_owner(tv: &TestValkey, slot: u16) -> Server {
    let routing = tv
        .client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view");
    routing
        .get_server(slot)
        .unwrap_or_else(|| panic!("no primary owns slot {slot} in the cached routing table"))
        .clone()
}

/// Every primary in the cluster, from the cached routing table.
fn primaries(tv: &TestValkey) -> Vec<Server> {
    tv.client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view")
        .unique_primary_nodes()
}

/// `Options` that pin a command to one specific cluster node and take fred's
/// own retrying out of the picture, so what the test observes is what that node
/// actually replied.
fn pinned_to(node: &Server) -> Options {
    Options {
        cluster_node: Some(node.clone()),
        max_attempts: Some(1),
        ..Default::default()
    }
}

/// Pick `want` object ids whose `le:{object_id}:obj:stream` keys are owned by
/// `want` DISTINCT primaries.
///
/// Non-vacuity is the whole point: a scenario that means to exercise
/// cross-shard delivery must fail loudly, naming what it saw, if the cluster
/// cannot actually provide that many shards — otherwise "cross-shard delivery
/// works" silently degrades into "two consumers on one shard work", which the
/// single-node suite already covers.
fn object_ids_on_distinct_shards(tv: &TestValkey, want: usize) -> Vec<(u64, Server)> {
    let available = primaries(tv);
    assert!(
        available.len() >= want,
        "this scenario needs {want} distinct primaries but the cluster reports {} ({:?}). \
         `scripts/dev-valkey.sh cluster-up` provisions 3 shards; a smaller topology cannot \
         exercise cross-shard subscriber delivery at all",
        available.len(),
        available
    );

    let mut picked: Vec<(u64, Server)> = Vec::with_capacity(want);
    // The scan is DETERMINISTIC: `KeySchema::layer_stream` hash-tags on
    // `{object_id}` alone (the fixture's random prefix sits outside the braces),
    // so the slot — and therefore the owner — is a pure function of the object
    // id and this loop visits the same slots in the same order on every run.
    //
    // The bound is nonetheless sized well above what a 3-shard cluster needs (a
    // handful of candidates normally suffices) because it also has to hold for
    // topologies this test does not control: more primaries mean more distinct
    // owners to find, and an uneven slot assignment can put a long run of
    // consecutive object ids on already-picked owners. A generous ceiling costs
    // nothing on the happy path — the loop returns as soon as `want` is met —
    // and turns "the topology is unusual" into a panic with a diagnosis rather
    // than a silent short scan.
    for object_id in 950_000u64..960_000 {
        let owner = slot_owner(tv, hash_slot_of(&tv.key_schema().layer_stream(object_id)));
        if picked.iter().any(|(_, o)| *o == owner) {
            continue;
        }
        picked.push((object_id, owner));
        if picked.len() == want {
            return picked;
        }
    }
    panic!(
        "scanned 10000 candidate object ids without finding {want} distinct slot owners — the \
         cached routing table reports {} primaries ({:?}), so either the table is stale or the \
         key schema stopped hash-tagging on {{object_id}}",
        available.len(),
        available
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Connection identity — which socket is the reader parked on?
// ═════════════════════════════════════════════════════════════════════════════
//
// Several tests below have to disturb the *specific* connection a reader's
// `XREAD BLOCK` is parked on. Finding it by scanning `CLIENT LIST` for
// `cmd=xread` is not sound on a SHARED cluster: a sibling test process can have
// its own reader parked on the same node in the same instant.
//
// So identity is established from the reader itself, once the subscription
// exists (b.azh: a reader connects lazily, the first time it serves a
// subscription, so before that there is no connection to identify). The
// fixture has exactly ONE stream reader, so the subscription can only be
// served by it; it connects to the owning node alone (a reader is a
// single-node client to its bound slot's primary); and fred caches the
// `CLIENT ID` of the connection when it connects, so `connection_ids()` names
// the exact socket without sending a command — which, on a reader parked in
// `XREAD BLOCK`, would interrupt the read (`Blocking::Interrupt`). `CLIENT LIST
// ID` then confirms `cmd=xread` on that id — the "parked" barrier and the
// non-vacuity check at once.

/// The one stream reader's connection to `node`.
struct ReservedConsumerSlot {
    conn_id: i64,
    node: Server,
}

/// Identify the single reader's connection to `node`, the owner of the slot
/// its subscription is bound to. Runs once that subscription exists: waits for
/// the reader to connect, then reads the connection id fred cached (no command
/// is sent). Asserts the reader's one connection is to `node` alone. Bounded
/// by one [`PARK_BUDGET`] of its own; [`identify_parked_reader`] shares one
/// with the park that follows.
async fn identify_consumer_slot(tv: &TestValkey, node: &Server) -> ReservedConsumerSlot {
    identify_consumer_slot_by(tv, node, std::time::Instant::now() + PARK_BUDGET).await
}

/// [`identify_consumer_slot`] and then [`await_parked_in_xread`], under ONE
/// [`PARK_BUDGET`] deadline for both, so the pair costs one budget in the
/// module table, not two.
async fn identify_parked_reader(tv: &TestValkey, node: &Server) -> ReservedConsumerSlot {
    let deadline = std::time::Instant::now() + PARK_BUDGET;
    let reserved = identify_consumer_slot_by(tv, node, deadline).await;
    await_parked_in_xread(tv, &reserved, deadline).await;
    reserved
}

async fn identify_consumer_slot_by(
    tv: &TestValkey,
    node: &Server,
    deadline: std::time::Instant,
) -> ReservedConsumerSlot {
    let mux = &tv.client().stream_mux;
    assert_eq!(
        mux.snapshot().readers.len(),
        1,
        "fixture self-check: connection identity needs exactly one stream reader \
         (`cluster_fixture_with_readers(_, 1)`)"
    );
    let reader = loop {
        if let Some(reader) = mux.reader_client_for_test(0) {
            break reader;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the stream reader never connected to serve its subscription within {PARK_BUDGET:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    let ids = reader.connection_ids();
    assert_eq!(
        ids.keys().collect::<Vec<_>>(),
        vec![node],
        "a bound reader holds one connection, to its slot's owner {node} alone; got {ids:?}"
    );
    let conn_id = ids[node];
    assert!(
        conn_id > 0,
        "fred must have cached a positive connection id; got {conn_id}"
    );
    ReservedConsumerSlot {
        conn_id,
        node: node.clone(),
    }
}

/// The `CLIENT LIST ID <id>` line for one connection on one node, or `None`
/// when the server no longer knows that id (i.e. the connection was replaced).
async fn client_list_line(tv: &TestValkey, node: &Server, conn_id: i64) -> Option<String> {
    let listing: String = tv
        .client()
        .pool
        .next()
        .with_options(&pinned_to(node))
        .client_list::<String, String>(None, Some(vec![conn_id.to_string()]))
        .await
        .unwrap_or_else(|e| panic!("CLIENT LIST ID {conn_id} on {node} must succeed: {e}"));

    let line = listing.trim();
    (!line.is_empty()).then(|| line.to_owned())
}

/// Block until `CLIENT LIST ID <id>` on the owning node reports `cmd=<command>`
/// for the reserved connection — i.e. until that connection's LAST command is
/// the one named.
///
/// `CLIENT LIST`'s `cmd=` field is the last command the server processed on that
/// connection, which makes it the one server-side surface that says *which
/// socket ran which command*. Polling it is how this module turns "the consumer
/// is presumably parked / has presumably probed" into an observed fact.
///
/// `why` is appended to the failure message: what the caller was about to do
/// with the connection, so a timeout reads as a diagnosis rather than a bare
/// deadline. Polls until `deadline`, which the caller shares with the waits
/// before it ([`identify_parked_reader`]).
async fn await_reserved_last_command(
    tv: &TestValkey,
    reserved: &ReservedConsumerSlot,
    command: &str,
    why: &str,
    deadline: std::time::Instant,
) {
    let want = format!("cmd={command}");
    loop {
        let last_seen = client_list_line(tv, &reserved.node, reserved.conn_id).await;
        if last_seen
            .as_deref()
            .is_some_and(|line| line.contains(&want))
        {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the reserved stream reader connection (id={id} on {node}) never reported \
             `{want}` within its {PARK_BUDGET:?} deadline. Last CLIENT LIST line: \
             {last_seen:?}. {why}",
            id = reserved.conn_id,
            node = reserved.node
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Block until the reserved connection is sitting in a blocking `XREAD`.
///
/// Doubles as the non-vacuity check for the whole identity scheme: the id
/// fred cached for the reader's connection must still name a live connection,
/// and that connection must be the one running the read. If either is false
/// the disturbance the caller is about to stage would land somewhere harmless
/// and the test would pass without testing anything.
async fn await_parked_in_xread(
    tv: &TestValkey,
    reserved: &ReservedConsumerSlot,
    deadline: std::time::Instant,
) {
    await_reserved_last_command(
        tv,
        reserved,
        "xread",
        "Either the reader never issued its XREAD (check for an early terminal error), or the \
         reader replaced its connection after it was identified — in which case the disturbance \
         this test is about to stage would hit an unrelated socket and prove nothing",
        deadline,
    )
    .await;
}

/// `CLIENT KILL ID <id>` on the owning node — a real, server-side connection
/// loss on exactly the socket the consumer's `XREAD BLOCK` is parked on.
async fn kill_reserved_connection(tv: &TestValkey, reserved: &ReservedConsumerSlot) {
    let killed: u64 = tv
        .client()
        .pool
        .next()
        .with_options(&pinned_to(&reserved.node))
        .client_kill(vec![ClientKillFilter::ID(reserved.conn_id.to_string())])
        .await
        .unwrap_or_else(|e| {
            panic!(
                "CLIENT KILL ID {} on {} must succeed: {e}",
                reserved.conn_id, reserved.node
            )
        });
    assert_eq!(
        killed, 1,
        "CLIENT KILL must have closed exactly the one connection under test (id={} on {}); \
         a 0 means the connection had already gone away and the test never staged the \
         disturbance it asserts about",
        reserved.conn_id, reserved.node
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Assertion helpers
// ═════════════════════════════════════════════════════════════════════════════

/// Await one notification, failing with `context` on either a timeout or a
/// consumer exit — and, on an exit, quoting the terminal-error slot.
///
/// The slot is the b.c7a discriminator and no survival claim should be made
/// without reading it: `recv() == None` with an EMPTY slot is an affirmative
/// claim that the subscription ended *normally*, which the RPC layer turns into
/// a clean gRPC `OK` and a client never retries. A test that only checked
/// `recv() != None` would report "the consumer died" identically for a clean
/// close and a signalled tear-down, and those have opposite client-visible
/// consequences.
async fn expect_update(
    consumer: &LayerStreamConsumer,
    budget: Duration,
    context: &str,
) -> (String, u64) {
    match tokio::time::timeout(budget, consumer.recv()).await {
        Ok(Some((stream_id, LayerNotification::Update { sequence_num, .. }))) => {
            (stream_id, sequence_num)
        }
        Ok(Some(other)) => panic!("{context}: unexpected notification shape: {other:?}"),
        Ok(None) => panic!(
            "{context}: the subscription ended instead of delivering. terminal_error = {:?} \
             (b.c7a: with no terminal exit this would be an affirmative claim that the \
             subscription ended NORMALLY — the RPC layer would close the stream with a clean OK \
             and the client would never re-subscribe)",
            consumer.take_terminal_exit().map(|exit| exit.error)
        ),
        Err(_) => panic!("{context}: no notification within {budget:?}"),
    }
}

/// Assert nothing further arrives in `window` — and that the silence is a live
/// consumer waiting, not a dead one.
async fn expect_quiet(consumer: &LayerStreamConsumer, window: Duration, context: &str) {
    match tokio::time::timeout(window, consumer.recv()).await {
        Err(_elapsed) => {}
        Ok(Some(other)) => panic!("{context}: expected no further notification; got {other:?}"),
        Ok(None) => panic!(
            "{context}: the consumer exited during the quiet window. terminal_error = {:?}",
            consumer.take_terminal_exit().map(|exit| exit.error)
        ),
    }
}

/// The b.c7a survival claim, made explicitly: the consumer is still running and
/// has not stashed a tear-down signal.
fn assert_still_healthy(consumer: &LayerStreamConsumer, metrics: &Metrics, context: &str) {
    let terminal = consumer.take_terminal_exit().map(|exit| exit.error);
    assert!(
        terminal.is_none(),
        "{context}: the consumer stashed a terminal error while the test expected it to survive: \
         {terminal:?}"
    );
    assert_no_permanent_exits(metrics, context);
}

/// Transient disturbances must never be counted as permanent reader exits.
fn assert_no_permanent_exits(metrics: &Metrics, context: &str) {
    let exits = read_permanent_exits(metrics, CONSUMER_PHASE_XREAD);
    assert_eq!(
        exits, 0,
        "{context}: {exits} permanent reader exit(s) were counted, but this disturbance is \
         inside the transient taxonomy"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// A1 — cross-shard live delivery
// ═════════════════════════════════════════════════════════════════════════════

/// Two subscribers whose streams live on DIFFERENT shards each receive their
/// own updates, and only their own.
///
/// The single-node suite cannot distinguish "the consumer reads the stream it
/// was opened on" from "the consumer reads whatever the one connection has",
/// because there is only one server. On a cluster the two streams hash to
/// different slots, so they are read by two slot-bound readers on two different
/// primaries, and a regression that lost the per-key routing — a `ClusterHash`
/// override, a pinned `cluster_node`, two slots in one XREAD — shows up
/// immediately as cross-delivery, CROSSSLOT, or silence.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_subscriber_delivers_live_updates_across_shards() {
    let Some(tv) = cluster_fixture(8).await else {
        return;
    };
    let metrics = install_metrics(&tv).await;

    let shards = object_ids_on_distinct_shards(&tv, 2);
    let (oid_a, ref owner_a) = shards[0];
    let (oid_b, ref owner_b) = shards[1];
    assert_ne!(
        owner_a, owner_b,
        "fixture self-check: object_ids_on_distinct_shards must return distinct owners"
    );

    // Each consumer opens at the commit before its stream's first entry (`10-0`
    // below `11-0`, `21-0` below `22-0`), as a caller that read the layer at
    // that sequence number does (`grpc::subscribe::stream_cursor_at`), so both
    // are owed every entry appended after they open. A cursor further below a
    // stream's first entry reads as a trim past it and ends `stream_gap`.
    let consumer_a = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        oid_a,
        StreamCursor::From("10-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consumer A must start");
    let consumer_b = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        oid_b,
        StreamCursor::From("21-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consumer B must start");
    // Two slots → two slot-bound readers; one XREAD cannot span them.
    let bound: Vec<u16> = tv
        .client()
        .stream_mux
        .snapshot()
        .readers
        .iter()
        .filter_map(|r| r.slot)
        .collect();
    assert_eq!(
        bound.len(),
        2,
        "streams in two hash slots must be served by two slot-bound readers; bound {bound:?}"
    );

    // Distinct sequence numbers so a mis-delivered entry is unambiguous rather
    // than merely plausible.
    xadd_update(&tv, oid_a, 11).await;
    xadd_update(&tv, oid_b, 22).await;

    let (id_a, seq_a) = expect_update(
        &consumer_a,
        DELIVERY_BUDGET,
        &format!("consumer A on {owner_a} (object {oid_a})"),
    )
    .await;
    assert_eq!(
        (id_a.as_str(), seq_a),
        ("11-0", 11),
        "the consumer opened on object {oid_a} (shard {owner_a}) must receive that object's \
         entry — receiving object {oid_b}'s `22-0` instead would mean the XREAD was routed by \
         something other than the stream key's hash slot"
    );

    let (id_b, seq_b) = expect_update(
        &consumer_b,
        DELIVERY_BUDGET,
        &format!("consumer B on {owner_b} (object {oid_b})"),
    )
    .await;
    assert_eq!(
        (id_b.as_str(), seq_b),
        ("22-0", 22),
        "the consumer opened on object {oid_b} (shard {owner_b}) must receive that object's entry"
    );

    // Cross-delivery guard. Each consumer's stream received exactly one entry,
    // so a second notification on either side can only be the sibling's.
    expect_quiet(
        &consumer_a,
        Duration::from_millis(750),
        "consumer A after its own entry",
    )
    .await;
    expect_quiet(
        &consumer_b,
        Duration::from_millis(750),
        "consumer B after its own entry",
    )
    .await;

    assert_still_healthy(&consumer_a, &metrics, "cross-shard consumer A");
    assert_still_healthy(&consumer_b, &metrics, "cross-shard consumer B");

    drop(consumer_a);
    drop(consumer_b);
}

// ═════════════════════════════════════════════════════════════════════════════
// A2 — a real non-owner `-MOVED` is classified transient
// ═════════════════════════════════════════════════════════════════════════════

/// The closing guard on the transient classifier (b.cv4): a command on a
/// layer-stream key, pinned to a primary that does NOT own the key's slot,
/// with `max_attempts = 1` — the pin every stream reader's `XREAD` carries —
/// so fred cannot follow the real `-MOVED` and surfaces the refusal. That
/// error object, produced by an actual 3-shard cluster, must pass
/// `is_transient_fred_error`.
///
/// Every other test of the redirect-exhaustion shapes uses hand-built errors
/// (`storage::client::BUDGET_EXHAUSTED_SHAPES`), which restate the
/// classifier's assumption rather than check it: if fred reworded its message,
/// only this test would notice. The first real run of this guard (pipeline
/// 64932151) showed a shape nobody had predicted from reading fred's code.
///
/// A positive control first: the same command pinned to the slot OWNER
/// succeeds, so the refusal below is about the node, not the command.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_subscriber_a_real_non_owner_moved_is_classified_transient() {
    let Some(tv) = cluster_fixture(4).await else {
        return;
    };
    let object_id: u64 = 951_100;
    let stream_key = tv.key_schema().layer_stream(object_id);
    let slot = hash_slot_of(&stream_key);
    let owner = slot_owner(&tv, slot);
    xadd_update(&tv, object_id, 1).await;

    let len: u64 = tv
        .client()
        .pool
        .next()
        .with_options(&pinned_to(&owner))
        .xlen(stream_key.clone())
        .await
        .unwrap_or_else(|e| panic!("XLEN pinned to the slot owner {owner} must succeed: {e}"));
    assert_eq!(len, 1, "the owner serves the seeded entry");

    let non_owner = primaries(&tv)
        .into_iter()
        .find(|s| *s != owner)
        .expect("a 3-shard cluster must have a primary that does not own this slot");
    // `pinned_to` sets `max_attempts: Some(1)`: fred charges an attempt on the
    // re-write that follows a redirect (`write_command`,
    // fred-10.1.0/src/router/commands.rs:111), so it cannot complete the hop.
    let err = tv
        .client()
        .pool
        .next()
        .with_options(&pinned_to(&non_owner))
        .xlen::<u64, _>(stream_key.clone())
        .await
        .err()
        .unwrap_or_else(|| {
            panic!(
                "XLEN for slot {slot} (owned by {owner}) SUCCEEDED when pinned to the non-owner \
                 {non_owner}: either the cached routing table is stale or the deployment is not \
                 sharding by slot, and this guard proves nothing"
            )
        });
    let refusal = err.details().contains("MOVED")
        || matches!(err.kind(), fred::error::ErrorKind::Cluster)
        || live_edit::storage::client::is_router_retry_budget_exhausted_error(&err);
    assert!(
        refusal,
        "a non-owner must refuse with a redirect — a raw `-MOVED` (or kind `Cluster`) or one of \
         fred's budget-exhausted shapes; got kind={:?} details={:?}",
        err.kind(),
        err.details()
    );
    assert!(
        live_edit::storage::client::is_transient_fred_error(&err),
        "a real cluster's routing refusal must classify as TRANSIENT; got kind={:?} \
         details={:?}. Classified permanent, every resharding would end every live \
         subscription and fail every single-shot read INTERNAL. If fred introduced a new \
         exhaustion message, add it to `is_router_retry_budget_exhausted_error` — kind-gated \
         and prefix-anchored — rather than loosening the predicate",
        err.kind(),
        err.details()
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// A3 — a forced reconnect rebuilds a cluster reader's one connection
// ═════════════════════════════════════════════════════════════════════════════

/// A synthetic `Timeout` on a reader's XREAD forces a reconnect, and delivery
/// resumes from the durable cursor on the rebuilt connection.
///
/// Needs `test-fault-injection` (the injected `Timeout` and the knobs) as
/// well as `LIVE_EDIT_TEST_CLUSTER_URL`: without the feature it is compiled
/// out, not skipped, and prints no `SKIP:` line.
///
/// A cluster reader is a single-node client to its slot's owner (b.azh), so
/// its forced reconnect first re-resolves the bound slot's owner in the
/// readers' routing view: with the owner unchanged, as here, it rebuilds its
/// one connection with `force_reconnection()`; with the owner moved it would
/// retarget instead (pinned on the two-node facade by
/// `stream_mux_reader_connect_test::faults::a_forced_reconnect_after_the_slots_owner_moved_retargets_instead_of_reconnecting_the_old_node`).
/// The single-node twin cannot exercise
/// that check — standalone, there is no owner to re-resolve. Here a rebuild
/// that came back to the wrong node, or a check that wrongly retired the
/// connection, surfaces as the post-reconnect entry never arriving. (The name
/// predates b.azh, when the reader was a full cluster client and the rebuild
/// was of its per-node connection map.)
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_subscriber_forced_reconnect_rebuilds_the_clustered_connection_map() {
    let case = "cluster XREAD forced reconnect";
    let object_id: u64 = 951_200;
    let Some(tv) = cluster_fixture(4).await else {
        return;
    };
    let metrics = install_metrics(&tv).await;
    tv.client()
        .stream_mux
        .set_knobs_for_test(fast_backoff_knobs(tv.client()));

    // Seed at `5-0` and resume from `5-0` so the post-reconnect delivery also
    // proves the durable cursor survived the rebuild: a cursor reset to `0-0`
    // would replay `5-0`, and a lost cursor would deliver nothing.
    xadd_update(&tv, object_id, 5).await;
    // Consumed by the reader's first XREAD for this subscription.
    push_timeout_faults(tv.client(), 1);

    let consumer = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From("5-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consume_layer_stream must succeed");
    await_faults_consumed(tv.client()).await;

    // The faulted round delivered nothing, so a delivery can only come from an
    // XREAD issued after the forced reconnect, which counts before it
    // reconnects: the delivery is the happens-before edge for the metric.
    xadd_update(&tv, object_id, 6).await;
    let (stream_id, seq) = expect_update(&consumer, DELIVERY_BUDGET, case).await;
    assert_eq!(
        (stream_id.as_str(), seq),
        ("6-0", 6),
        "{case}: after the forced reconnect rebuilt the reader's connection, the reader must \
         resume from the caller's `5-0` cursor on a working connection to the slot owner. \
         Replaying `5-0` would mean the cursor was reset; delivering nothing would mean the \
         rebuilt connection cannot serve this slot"
    );
    assert_eq!(
        read_forced_reconnects(&metrics, CONSUMER_PHASE_XREAD),
        1,
        "{case}: a Timeout on the XREAD must force exactly one reconnect and record it"
    );
    assert_eq!(
        read_series(
            &metrics,
            "live_edit_stream_mux_reader_connects_total",
            &[("cause", "retarget"), ("result", "ok")],
        ),
        0,
        "{case}: the slot's owner did not move, so the forced reconnect rebuilt the reader's \
         connection rather than retargeting it"
    );

    assert_still_healthy(&consumer, &metrics, case);
    drop(consumer);
}

// ═════════════════════════════════════════════════════════════════════════════
// A4 — the durable cursor survives a REAL connection loss
// ═════════════════════════════════════════════════════════════════════════════

/// A consumer whose connection is killed mid-`XREAD BLOCK` resumes from its
/// caller-supplied cursor and delivers everything appended during the gap —
/// exactly once, in order.
///
/// This is the b.9uq durable-cursor contract driven by a genuine server-side
/// connection close on a cluster node, rather than by an injected error. Two
/// things only a real kill can cover:
///
///   * fred's cluster router has to rebuild a connection to *that node* and
///     re-issue on it. The injected-`Timeout` tests never actually lose a
///     socket.
///   * the entries written during the gap are written through a *different*
///     connection (the shared pool's), so they genuinely land while the
///     consumer has no reader — which is the state the old `"$"` cursor lost
///     data in.
///
/// Fixture: resume from `5-0` on a stream that already holds `5-0`, so a cursor
/// reset to `0-0` replays it and is caught, and a cursor re-pinned to the tail
/// delivers nothing and is caught.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_subscriber_cursor_survives_connection_loss() {
    let Some(tv) = cluster_fixture_with_readers(8, 1).await else {
        return;
    };
    let metrics = install_metrics(&tv).await;

    let object_id: u64 = 951_300;
    let stream_key = tv.key_schema().layer_stream(object_id);
    let owner = slot_owner(&tv, hash_slot_of(&stream_key));

    xadd_update(&tv, object_id, 5).await;

    // Production 30 s BLOCK: the reader is still parked when the kill lands,
    // so the loss hits an in-flight read rather than the gap between two.
    let consumer = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From("5-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consumer must start");

    // Which connection the reader parks on — see the "Connection identity"
    // note above.
    let reserved = identify_parked_reader(&tv, &owner).await;
    kill_reserved_connection(&tv, &reserved).await;

    // Written while the consumer has no reader on the owning node. Two entries,
    // so ordering and exactly-once are both observable.
    xadd_update(&tv, object_id, 6).await;
    xadd_update(&tv, object_id, 7).await;

    let (id6, seq6) = expect_update(
        &consumer,
        DELIVERY_BUDGET,
        "first entry appended during the connection gap",
    )
    .await;
    assert_eq!(
        (id6.as_str(), seq6),
        ("6-0", 6),
        "the resumed read must start strictly after the caller's `5-0` cursor. Replaying `5-0` \
         means the cursor was reset to a replay-from-the-beginning `0-0`; starting at `7-0` \
         means an entry written during the gap was skipped — the b.9uq silent-stall shape"
    );

    let (id7, seq7) = expect_update(
        &consumer,
        DELIVERY_BUDGET,
        "second entry appended during the connection gap",
    )
    .await;
    assert_eq!(
        (id7.as_str(), seq7),
        ("7-0", 7),
        "both gap entries must be delivered, in order"
    );

    // Exactly once: a re-issue that resumed from a stale cursor would replay.
    expect_quiet(
        &consumer,
        Duration::from_millis(750),
        "after both gap entries were delivered",
    )
    .await;

    assert_still_healthy(
        &consumer,
        &metrics,
        "a killed connection is an IO error — transient, so the consumer must self-heal rather \
         than tear the subscription down",
    );
    drop(consumer);
}

// ═════════════════════════════════════════════════════════════════════════════
// A7 — a per-node `CLIENT UNBLOCK` on a slot-bound reader is re-planned
// ═════════════════════════════════════════════════════════════════════════════

/// `CLIENT UNBLOCK <id> ERROR` against the exact connection a slot-bound reader
/// is parked on — the interrupt fred's `Blocking::Interrupt` sends — makes its
/// `XREAD` fail `-UNBLOCKED`. A cluster reader is a single-node client to its
/// slot's owner (b.azh), so the reply comes back on its one connection; the
/// reader must take its cheap re-plan path (counted on
/// `valkey_consumer_unblocked_retries_total`), keep the subscription, and
/// deliver what is appended next. (The two-node facade twin, which also pins
/// that a command queued behind the read interrupts it, runs without a
/// cluster in `stream_mux_reader_connect_test.rs`.)
///
/// The single-node twin is `stream_mux_test::a_client_unblock_on_a_reader_is_replanned_without_loss`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_subscriber_a_client_unblock_on_a_slot_bound_reader_is_replanned_without_loss() {
    let Some(tv) = cluster_fixture_with_readers(8, 1).await else {
        return;
    };
    let metrics = install_metrics(&tv).await;
    let object_id: u64 = 951_400;
    let stream_key = tv.key_schema().layer_stream(object_id);
    let owner = slot_owner(&tv, hash_slot_of(&stream_key));

    let consumer = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From("0-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consumer must start");
    let reserved = identify_parked_reader(&tv, &owner).await;

    // `cmd=xread` names the last command; retry until the server reports the
    // connection actually blocked in it.
    let deadline = std::time::Instant::now() + PARK_BUDGET;
    loop {
        let unblocked: i64 = tv
            .client()
            .pool
            .next()
            .with_options(&pinned_to(&owner))
            .client_unblock(reserved.conn_id, Some(ClientUnblockFlag::Error))
            .await
            .unwrap_or_else(|e| panic!("CLIENT UNBLOCK on {owner} must succeed: {e}"));
        if unblocked == 1 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the reserved connection (id={}) never blocked in the reader's XREAD",
            reserved.conn_id
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    xadd_update(&tv, object_id, 1).await;
    let (id, seq) = expect_update(
        &consumer,
        DELIVERY_BUDGET,
        "delivery after a per-node CLIENT UNBLOCK",
    )
    .await;
    assert_eq!((id.as_str(), seq), ("1-0", 1));
    assert_eq!(
        super::mux_test_helpers::read_unblocked_retries(&metrics),
        1,
        "the `-UNBLOCKED` reply must take the cheap re-plan path"
    );
    assert_still_healthy(&consumer, &metrics, "a per-node CLIENT UNBLOCK");
    drop(consumer);
}

// ═════════════════════════════════════════════════════════════════════════════
// A10 — b.19i: a slot-bound reader re-AUTHs between its reads
// ═════════════════════════════════════════════════════════════════════════════

/// b.19i on a real cluster (design item 6): a live subscriber's reader takes
/// several credential re-AUTH ticks while its read is parked on a quiet
/// layer. The first due credential wakes the parked read (a credential wake
/// is counted), and the reader sends each `AUTH` on its one connection
/// between reads, so across all of them no read is
/// interrupted (`valkey_consumer_unblocked_retries_total` stays 0), the
/// reader's connection is never rebuilt (`valkey_reconnects_total{pool=
/// "subscriber_pool"}` stays 0, and its server-side id is unchanged), every
/// re-AUTH is answered (`reauths_total{ok}` reaches the number granted), and
/// delivery holds.
///
/// Under a [`ScriptedProvider`] the pools start no fred refresh while their
/// routers keep running, so the cluster sees only the reader's own `AUTH`s;
/// the pooled clients'
/// `_AuthAllCluster` path is pinned on the two-node facade in
/// `stream_mux_reader_connect_test.rs`. Its single-node and facade twins are
/// `stream_mux_reauth_test::a_reauth_due_during_a_blocked_read_wakes_it_and_auths_between_reads`
/// and `a_cluster_readers_due_reauth_wakes_its_read_and_auths_between_reads`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_subscriber_a_slot_bound_reader_reauths_between_reads_without_interrupting_or_reconnecting(
) {
    use super::mux_test_helpers::{reauths, wakes, ScriptedProvider};
    use live_edit::observability::metrics::{
        REAUTH_RESULT_FAILED, REAUTH_RESULT_OK, WAKE_OUTCOME_OK,
    };

    /// Credentials granted: several re-AUTHs on one live subscription.
    const GRANTED: i64 = 3;

    let provider = ScriptedProvider::new("b19i-cluster-gated-password");
    let Some(tv) = cluster_fixture_with_provider(8, 1, provider.clone()).await else {
        return;
    };
    provider.start_ticking();
    let metrics = install_metrics(&tv).await;
    let object_id: u64 = 952_100;
    let owner = slot_owner(&tv, hash_slot_of(&tv.key_schema().layer_stream(object_id)));
    let consumer = subscribe_and_serve(&tv, object_id, 1).await;
    let reserved = identify_parked_reader(&tv, &owner).await;
    let reconnects = |metrics: &Metrics| {
        super::mux_test_helpers::read_series(
            metrics,
            "live_edit_valkey_reconnects_total",
            &[("pool", "subscriber_pool")],
        )
    };
    let wakes_before = wakes(&metrics, WAKE_OUTCOME_OK);

    provider.grant(GRANTED as usize);
    let deadline = std::time::Instant::now() + DELIVERY_BUDGET;
    while reauths(&metrics, REAUTH_RESULT_OK) < GRANTED {
        assert!(
            std::time::Instant::now() < deadline,
            "{GRANTED} granted re-AUTHs must each be answered well inside the 30 s BLOCK; {} were",
            reauths(&metrics, REAUTH_RESULT_OK)
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    assert_eq!(reauths(&metrics, REAUTH_RESULT_FAILED), 0);
    assert!(
        wakes(&metrics, WAKE_OUTCOME_OK) > wakes_before,
        "the first due credential found the read parked and woke it"
    );
    assert_eq!(
        super::mux_test_helpers::read_unblocked_retries(&metrics),
        0,
        "no re-AUTH interrupted a read: the reader sends its AUTH between reads"
    );
    assert_eq!(
        reconnects(&metrics),
        0,
        "no re-AUTH reconnected the reader's connection"
    );
    assert!(
        client_list_line(&tv, &reserved.node, reserved.conn_id)
            .await
            .is_some(),
        "the reader's connection (id={}) is still the one on {}",
        reserved.conn_id,
        reserved.node
    );
    xadd_update(&tv, object_id, 2).await;
    let (id, seq) = expect_update(
        &consumer,
        DELIVERY_BUDGET,
        "delivery after the reader's re-AUTHs",
    )
    .await;
    assert_eq!((id.as_str(), seq), ("2-0", 2));
    assert_still_healthy(&consumer, &metrics, "the reader's re-AUTHs");
    drop(consumer);
}

// ═════════════════════════════════════════════════════════════════════════════
// A8, A9 — b.azh: a reader connects to its slot's owner alone, and retargets
// ═════════════════════════════════════════════════════════════════════════════

/// Subscribes the fixture to `object_id` from `0-0`, writes `seq`, and returns
/// the subscription once that update was delivered.
async fn subscribe_and_serve(tv: &TestValkey, object_id: u64, seq: u64) -> LayerStreamConsumer {
    let consumer = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From("0-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consumer must start");
    xadd_update(tv, object_id, seq).await;
    let (_, delivered) =
        expect_update(&consumer, DELIVERY_BUDGET, "the subscription's update").await;
    assert_eq!(delivered, seq);
    consumer
}

/// b.azh T4 and T15: a bound reader's client is a single-node client with one
/// connection, to the primary that owns its slot — the very `Server` the
/// readers' routing view (the pinned data-pool client's cached map) names for
/// it. fred's `Server` equality ignores the TLS server name, so that field is
/// compared explicitly; it would mean something only against a TLS cluster,
/// and [`cluster_fixture`] runs this suite against plaintext
/// `redis-cluster://` URLs alone, where both sides are `None`. The proof that the `Server` is taken whole, never rebuilt from
/// host and port, is the source guard
/// `reader_connector_diverges_only_the_server_and_database_of_its_template` in
/// `src/storage/client.rs`. A cluster URL selects no database.
///
/// Before b.azh a reader was a full cluster client with a connection to every
/// primary.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_subscriber_a_bound_reader_connects_to_its_slot_owner_alone() {
    use fred::types::config::ServerConfig;

    let Some(tv) = cluster_fixture_with_readers(8, 1).await else {
        return;
    };
    let object_id: u64 = 951_700;
    let slot = hash_slot_of(&tv.key_schema().layer_stream(object_id));
    let consumer = subscribe_and_serve(&tv, object_id, 1).await;

    let owner = tv.client().pool.clients()[0]
        .cached_cluster_state()
        .expect("the readers' routing view is a cluster map")
        .get_server(slot)
        .cloned()
        .expect("the routing view names the slot's owner");
    let reader = tv
        .client()
        .stream_mux
        .reader_client_for_test(0)
        .expect("the bound reader holds a connection");
    assert!(!reader.is_clustered(), "a reader client is single-node");
    assert_eq!(
        reader.active_connections(),
        vec![owner.clone()],
        "one connection, to the slot's owner alone"
    );
    let config = reader.client_config();
    let server = match &config.server {
        ServerConfig::Centralized { server } => server.clone(),
        other => panic!("a reader's config targets one server; got {other:?}"),
    };
    assert_eq!(server, owner, "the routing view's Server for the slot");
    #[cfg(feature = "valkey-tls")]
    assert_eq!(
        server.tls_server_name, owner.tls_server_name,
        "the TLS server name the routing view carries rides along"
    );
    assert_eq!(
        config.database, None,
        "a cluster URL's database is never selected"
    );
    assert_eq!(
        tv.client().stream_mux.snapshot().readers[0].node,
        Some(owner.to_string())
    );
    drop(consumer);
}

/// b.azh T5: a one-reader fixture serves a slot on primary A, then a slot on
/// primary B. The reader retargets: its client now targets B alone, its old
/// connection is gone from A's `CLIENT LIST`, the retarget is counted
/// (`cause="retarget"`), and the subscription on B is served.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_subscriber_a_reader_rebound_to_another_primarys_slot_retargets_its_connection() {
    let Some(tv) = cluster_fixture_with_readers(8, 1).await else {
        return;
    };
    let metrics = install_metrics(&tv).await;
    let picked = object_ids_on_distinct_shards(&tv, 2);
    let ((on_a, node_a), (on_b, node_b)) = (picked[0].clone(), picked[1].clone());
    let connects = |cause: &str| {
        super::mux_test_helpers::read_series(
            &metrics,
            "live_edit_stream_mux_reader_connects_total",
            &[("cause", cause), ("result", "ok")],
        )
    };

    let first = subscribe_and_serve(&tv, on_a, 1).await;
    let on_node_a = identify_consumer_slot(&tv, &node_a).await;
    drop(first);
    let second = subscribe_and_serve(&tv, on_b, 1).await;

    let reader = tv
        .client()
        .stream_mux
        .reader_client_for_test(0)
        .expect("the reader holds a connection");
    assert_eq!(
        reader.active_connections(),
        vec![node_b.clone()],
        "the reader now targets the new slot's owner alone"
    );
    assert_eq!(
        (connects("bind"), connects("retarget")),
        (1, 1),
        "one bind, then one retarget"
    );
    let deadline = std::time::Instant::now() + PARK_BUDGET;
    while client_list_line(&tv, &node_a, on_node_a.conn_id)
        .await
        .is_some()
    {
        assert!(
            std::time::Instant::now() < deadline,
            "the reader's old connection (id={}) is still listed on {node_a} after the retarget",
            on_node_a.conn_id
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_still_healthy(&second, &metrics, "a retarget to another primary");
    drop(second);
}

// ═════════════════════════════════════════════════════════════════════════════
// A5 — the SubscribeReadLayer RPC survives a connection loss
// ═════════════════════════════════════════════════════════════════════════════

/// Drain subscribe-stream messages until the `total_size == 0` caught-up ack.
///
/// A brand-new catch-up (`object_id = 0, sequence_num = 0`) emits its full-state
/// content and then a zero-body ack header (b.rdr). Draining *to the ack*
/// rather than counting messages keeps this test independent of how many
/// content messages the catch-up happens to produce.
///
/// **One deadline for the whole drain** ([`CATCH_UP_BUDGET`], b.cv4). The 16
/// iterations and their body-chunk loops share it, so the worst case is the
/// budget rather than the budget times the message count — see the module-level
/// budget table for why that distinction decides whether A5's diagnostics ever
/// reach the job log.
async fn drain_catch_up<S>(stream: &mut S)
where
    S: futures::Stream<
            Item = Result<live_edit::grpc::proto::SubscribeReadLayerResponse, tonic::Status>,
        > + Unpin,
{
    let deadline = std::time::Instant::now() + CATCH_UP_BUDGET;
    for _ in 0..16 {
        let (header, _body) = recv_subscribe_message(stream, "catch-up", deadline).await;
        if header.total_size == 0 {
            return;
        }
    }
    panic!("catch-up did not terminate with a `total_size == 0` caught-up ack within 16 messages");
}

/// Read one header plus its body chunks off a subscribe stream, all of it bound
/// by the caller's `deadline`.
///
/// The deadline is passed in rather than derived per call so a caller draining
/// many messages ([`drain_catch_up`]) can hold ONE budget across all of them.
/// A per-call budget would let a slow-but-progressing stream run for
/// budget × messages, which is how A5 came to have a worst case measured in
/// minutes against a 120 s hang-killer (b.cv4).
async fn recv_subscribe_message<S>(
    stream: &mut S,
    context: &str,
    deadline: std::time::Instant,
) -> (live_edit::grpc::proto::SubscribeReadLayerHeader, Vec<u8>)
where
    S: futures::Stream<
            Item = Result<live_edit::grpc::proto::SubscribeReadLayerResponse, tonic::Status>,
        > + Unpin,
{
    use futures::StreamExt;
    use live_edit::grpc::proto::subscribe_read_layer_response;

    let first = tokio::time::timeout_at(deadline.into(), stream.next())
        .await
        .unwrap_or_else(|_| {
            panic!(
                "{context}: the shared budget for this read was exhausted while waiting for the \
                 next header"
            )
        })
        .unwrap_or_else(|| {
            panic!(
                "{context}: the gRPC stream CLOSED. b.c7a: a clean close is an affirmative claim \
                 that the subscription ended normally, and a client reading a normal stream end \
                 does not re-subscribe — a connection-level disturbance must surface as \
                 Code::Unavailable or not at all"
            )
        })
        .unwrap_or_else(|e| panic!("{context}: the gRPC stream yielded an error: {e}"));

    let header = match first.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => h,
        other => panic!("{context}: expected a Header, got {other:?}"),
    };

    let mut buf = Vec::new();
    while (buf.len() as u64) < header.total_size {
        let next = tokio::time::timeout_at(deadline.into(), stream.next())
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "{context}: the shared budget was exhausted waiting for a body chunk \
                     ({filled} of {total} bytes filled)",
                    filled = buf.len(),
                    total = header.total_size
                )
            })
            .unwrap_or_else(|| panic!("{context}: stream closed before total_size was reached"))
            .unwrap_or_else(|e| panic!("{context}: stream yielded an error mid-body: {e}"));
        match next.response_type {
            Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                buf.extend_from_slice(&c.data)
            }
            other => panic!("{context}: expected a Chunk while filling the body, got {other:?}"),
        }
    }
    (header, buf)
}

/// Apply one non-reset `DiffChangeSection` update through the real
/// `UpdateLayer` handler.
async fn apply_update(
    svc: &live_edit::grpc::service::LiveEditService,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    section_id: u64,
) {
    use crate::checkpoint::helpers::{
        make_streaming_token_request, make_update_data, make_update_header, streaming_update,
    };
    use crate::common::owned_delta::delta_to_ompf_bytes;
    use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, StructCommand};

    // `is_diff: false` — a `is_diff=true, base_version=0` delta triggers SDD
    // §9.1.3 reset semantics, which would wipe the catch-up state this test
    // just drained rather than layering on top of it.
    let delta = OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id,
            section_name: format!("cluster-relay-{section_id}"),
            section_type: 0,
            section_order: 0,
        })],
        ..Default::default()
    };
    let stream = streaming_update(vec![
        make_update_header(
            session_id,
            layer_name,
            object_id,
            section_id * 10,
            "cluster-writer",
        ),
        make_update_data(delta_to_ompf_bytes(&delta, &[])),
    ]);
    live_edit::grpc::layer::handle_update_layer(svc, make_streaming_token_request(stream))
        .await
        .unwrap_or_else(|e| panic!("UpdateLayer (section_id={section_id}) must succeed: {e}"));
}

/// `SubscribeReadLayer` keeps relaying across a real connection loss on the
/// shard owning the layer stream.
///
/// This is b.c7a's deferred item 3. The complementary half — a reader that
/// retries a transient routing miss — is pinned at the reader level through the
/// stream multiplexer's fault seam. Here the disturbance is real: a server-side
/// `CLIENT KILL` on the exact socket the RPC's reader is parked on, and the
/// claim is that the client-visible stream neither errors nor closes, and goes
/// on relaying the next commit.
///
/// The failure this guards against is specifically the *silent* one. A stream
/// that ends cleanly here would look, to a Kit client, exactly like a completed
/// subscription: no error, no retry, no further updates.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_subscribe_rpc_relays_across_connection_loss() {
    use crate::grpc::helpers::{build_service, create_session_and_layer, AllowAllPermissions};
    use live_edit::grpc::proto::SubscribeReadLayerRequest;
    use live_edit::grpc::subscribe::handle_subscribe_read_layer;
    use live_edit::grpc::BearerToken;

    let Some(tv) = cluster_fixture_with_readers(8, 1).await else {
        return;
    };
    let metrics = install_metrics(&tv).await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "cluster-relay-session", "cluster-relay-layer").await;

    let owner = slot_owner(&tv, hash_slot_of(&tv.key_schema().layer_stream(object_id)));

    let mut request = tonic::Request::new(SubscribeReadLayerRequest {
        session_id: session_id.clone(),
        layer_name: "cluster-relay-layer".to_owned(),
        object_id: 0,
        sequence_num: 0,
        client_id: "cluster-reader".to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));

    let response = handle_subscribe_read_layer(&svc, request)
        .await
        .expect("SubscribeReadLayer must succeed against the cluster");
    let mut stream = response.into_inner();

    drain_catch_up(&mut stream).await;

    // Now in steady state: the RPC's subscription can only be served by the
    // one reader, connected to the owning node; identify that connection, wait
    // for it to park in XREAD BLOCK, and kill it out from under the live
    // subscription.
    let reserved = identify_parked_reader(&tv, &owner).await;
    kill_reserved_connection(&tv, &reserved).await;

    // The commit that must still be relayed. It lands while the subscription's
    // read connection is gone.
    apply_update(&svc, &session_id, "cluster-relay-layer", object_id, 7_001).await;

    let (header, body) = recv_subscribe_message(
        &mut stream,
        "post-connection-loss relay: the update committed while the subscription's read \
         connection was closed",
        // A single message, so a fresh delivery budget rather than the shared
        // catch-up one.
        std::time::Instant::now() + DELIVERY_BUDGET,
    )
    .await;
    assert!(
        header.total_size > 0 && !body.is_empty(),
        "the relayed message must carry the update's content, not an empty body; got \
         total_size={} body_len={}",
        header.total_size,
        body.len()
    );
    assert_eq!(
        header.object_id, object_id,
        "the relayed message must belong to the subscribed object generation"
    );
    assert!(
        header.sequence_num > 0,
        "the relayed message must carry the committed sequence_num"
    );

    // The consumer's own counters corroborate the client-visible verdict: a
    // killed socket is an IO error, which is inside the transient taxonomy, so
    // nothing may be filed as a permanent exit.
    assert_no_permanent_exits(
        &metrics,
        "a CLIENT KILL under a live SubscribeReadLayer is a transient connection loss (b.c7a)",
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// A6 — every error a real cluster disturbance produces is classified transient
// ═════════════════════════════════════════════════════════════════════════════

/// Capture `tracing` output into a buffer.
#[derive(Clone)]
struct CapturingWriter(Arc<std::sync::Mutex<Vec<u8>>>);

impl CapturingWriter {
    fn new() -> (Self, Arc<std::sync::Mutex<Vec<u8>>>) {
        let buf = Arc::new(std::sync::Mutex::new(Vec::new()));
        (CapturingWriter(Arc::clone(&buf)), buf)
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturingWriter {
    type Writer = CapturingWriter;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

impl std::io::Write for CapturingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Reverse of `fred::error::ErrorKind::to_str`.
///
/// The consumer logs `kind = %e.kind().to_str()` and `error = %e` (whose
/// `Display` is `"{kind}: {details}"`, fred-10.1.0/src/error.rs:130-134), so a
/// captured event carries everything needed to rebuild the error — but only if
/// the kind string can be mapped back. An unmapped kind is deliberately a hard
/// failure rather than a skip: the harvest below would otherwise skip past the
/// one error nobody has classified, which is exactly the blind spot the test
/// exists to close.
///
/// Covers every `ErrorKind` variant this build compiles
/// (fred-10.1.0/src/error.rs:19-74). Two are feature-gated in fred:
///
/// * `Tls` ("TLS Error") is compiled in here — live-edit's default `valkey-tls`
///   feature routes `fred/enable-rustls-ring`, one of the three features gating
///   the variant — so it is mapped, under the same `cfg` so CI's
///   `no-default-features:build-lint-test` job (which compiles this target with
///   `--no-default-features`) still builds. A TLS error is deliberately NOT
///   transient (`is_transient_fred_error`, src/storage/client.rs:129-138): a
///   failed handshake is a certificate/configuration fault, not a blip, and
///   retrying it silently would hide a misconfigured deployment. Harvesting one
///   here must therefore FAIL the classification assertion below — which is the
///   correct outcome, and a very different message from "unmapped kind".
/// * `Replica` is `replicas`-gated and that feature is off, so the variant does
///   not exist in this build and cannot be named.
fn error_kind_from_str(kind: &str) -> Option<fred::error::ErrorKind> {
    use fred::error::ErrorKind as K;
    let mapped = match kind {
        "Authentication Error" => K::Auth,
        "IO Error" => K::IO,
        "Routing Error" => K::Routing,
        "Invalid Argument" => K::InvalidArgument,
        "Invalid Command" => K::InvalidCommand,
        "Url Error" => K::Url,
        "Protocol Error" => K::Protocol,
        "Unknown Error" => K::Unknown,
        "Canceled" => K::Canceled,
        "Cluster Error" => K::Cluster,
        "Timeout Error" => K::Timeout,
        #[cfg(feature = "valkey-tls")]
        "TLS Error" => K::Tls,
        "Config Error" => K::Config,
        "Parse Error" => K::Parse,
        "Sentinel Error" => K::Sentinel,
        "Not Found" => K::NotFound,
        "Backpressure" => K::Backpressure,
        _ => return None,
    };
    Some(mapped)
}

/// Every fred error the consumer surfaced during a real cluster disturbance is
/// one the classifier accepts as retryable.
///
/// **This is the anti-self-reference guard for b.c7a.** The b.c7a regression
/// tests hand-build `Error::new(ErrorKind::Unknown, "Failed to route command.")`
/// and push it through the fault-injection seam, which restates the very
/// assumption `is_router_no_connection_error` encodes — if fred ever changes the
/// kind or the wording, those tests stay green while production regresses to
/// tearing subscriptions down. Only a REAL cluster can produce the real text,
/// and only a cluster can produce a real router rebuild at all.
///
/// So this test provokes a genuine connection loss and then harvests, from the
/// consumer's own log events, the `(kind, error)` pairs it actually observed —
/// reconstructing each into a `fred::error::Error` and running it back through
/// the production predicates. Anything the classifier does not accept is a
/// subscription the consumer would have torn down, and the failure quotes the
/// unrecognised text verbatim so the fix is a one-line predicate edit rather
/// than an investigation.
///
/// **What is asserted vs. what is reported.** The guard is "everything observed
/// is classified transient" — that holds on every run, whatever the disturbance
/// happened to produce. Whether this particular run also exercised a
/// *message-matched* carve-out is **run-dependent**: a `CLIENT KILL` usually
/// surfaces as an `IO`/`Canceled` error, which the classifier accepts by KIND
/// and which therefore says nothing about any carve-out's wording. Asserting a
/// carve-out was hit would make this test flaky; asserting it was NOT hit would
/// be worse. So the test classifies each observed error as "transient by kind"
/// vs. "accepted ONLY by a message match" and `eprintln!`s the verdict either
/// way, so the CI log states plainly whether real-text coverage was exercised on
/// this run. Read that line before concluding it exists.
///
/// **Every rescue predicate is checked, and the bucket names which one fired
/// (b.cv4).** There is more than one way past the kind check:
///
/// * `is_router_no_connection_error` — b.c7a, `"Failed to route command."` from
///   a momentarily-empty router map;
/// * `is_router_retry_budget_exhausted_error` — b.cv4, the redirect
///   budget-exhaustion family. **Reachable from this test's own disturbance
///   without any redirect**: `Router::retry_buffer`
///   (fred-10.1.0/src/router/mod.rs:373-390) re-dispatches a dropped
///   connection's in-flight commands straight back into `write_command`,
///   filtering only on `timed_out`, and `write_command` charges
///   `decr_check_attempted` at `:111`. The consumer pins `max_attempts: 1`, so
///   its re-dispatched `XREAD` arrives with a spent budget and dies
///   `Unknown` / `"Too many failed write attempts."` — after a `CLIENT KILL`,
///   which is exactly what this test does;
/// * `is_store_not_serving_reply` — b.xd1, the store's `-LOADING` / `-BUSY`;
///   a cluster node restarting or running a long script answers them, and
///   this test's `CLIENT KILL` does not, so it is listed for completeness of
///   the chain;
/// * `is_tryagain_reply` — b.vn4, a cluster node's `-TRYAGAIN` to a multi-key
///   command during a slot migration; a `CLIENT KILL` does not produce it
///   either, so it too is listed for completeness;
/// * `is_transaction_router_exhausted_error` — b.cyu, fred's `Transaction`
///   router out of redirections or attempts; the readers send no
///   transaction, so it is listed for completeness as well;
/// * `is_client_unblocked_error` — b.9yf, `-UNBLOCKED`; not part of
///   `is_transient_fred_error` at all, but it still means the kind alone did
///   *not* suffice.
///
/// Checking only one of them would not fail the assertion — it would
/// misfile a rescued error as "transient by kind" and make the empty-bucket
/// branch print a conclusion nothing had verified, in the one line this test
/// exists to produce.
///
/// **On the test-writing guide's "don't assert on exact log messages" rule.**
/// This test deliberately reads log output, but it is not pinning wording: it
/// harvests real error text in order to check its CLASSIFICATION. A reworded
/// consumer log message changes nothing here; a reworded *fred* error is
/// precisely what must fail. The `error`/`kind` fields are the only place the
/// consumer exposes the errors it swallowed — they are, by design, not on any
/// metric (a counter would record that something happened, never what).
///
/// **Why several predicates.** The reader's own acceptance set is
/// `is_transient_fred_error(e) || is_client_unblocked_error(e)`. `is_transient_fred_error` — the b.c7a
/// predicate — is the target, but a `-UNBLOCKED` that overflows the cheap
/// allowance also reaches the WARN path, and failing on that would report a
/// b.9yf-shaped interrupt (since b.19i only an outside `CLIENT UNBLOCK` of a
/// live read produces one) as a b.c7a classification bug.
/// Asserting the union keeps the verdict exactly "the consumer would have
/// retried this".
///
/// **Runtime flavour is load-bearing.** `#[tokio::test]` defaults to the
/// current-thread runtime, so the reader tasks are polled on this
/// test's own thread and `tracing::subscriber::set_default`'s thread-local
/// dispatcher reaches it. A multi-thread runtime would migrate the task to a
/// worker with no dispatcher installed and the capture would come back empty —
/// which is why the emptiness check below is an assertion, not a convenience.
#[tokio::test]
async fn cluster_subscriber_observed_errors_are_all_classified_transient() {
    use live_edit::storage::client::{
        is_client_unblocked_error, is_router_no_connection_error,
        is_router_retry_budget_exhausted_error, is_store_not_serving_reply,
        is_transaction_router_exhausted_error, is_transient_fred_error, is_tryagain_reply,
    };

    let Some(tv) = cluster_fixture_with_readers(8, 1).await else {
        return;
    };
    let metrics = install_metrics(&tv).await;

    let object_id: u64 = 951_400;
    let owner = slot_owner(&tv, hash_slot_of(&tv.key_schema().layer_stream(object_id)));
    xadd_update(&tv, object_id, 5).await;

    let (writer, captured) = CapturingWriter::new();
    let subscriber = tracing_subscriber::fmt()
        .json()
        .with_writer(writer)
        // WARN is where the reader reports every error it did NOT take the
        // cheap path on — i.e. exactly the set whose classification decides
        // whether its subscriptions survive.
        .with_max_level(tracing::Level::WARN)
        .with_ansi(false)
        .finish();
    let dispatch_guard = tracing::subscriber::set_default(subscriber);
    // Callsite interest is cached process-wide. With exactly one scoped
    // dispatcher alive, tracing-core computes a callsite's interest from the
    // registering thread's default only, so a sibling test (no subscriber)
    // hitting the reader's WARN callsite first caches it as "never" and this
    // capture stays empty under an in-process `cargo test`. A second live
    // dispatcher disables that shortcut; the rebuild re-evaluates callsites
    // already cached.
    let _interest_anchor = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    tracing::callsite::rebuild_interest_cache();

    let consumer = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From("5-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consumer must start");

    let reserved = identify_parked_reader(&tv, &owner).await;
    kill_reserved_connection(&tv, &reserved).await;
    xadd_update(&tv, object_id, 6).await;

    let (stream_id, seq) = expect_update(
        &consumer,
        DELIVERY_BUDGET,
        "delivery after a real cluster connection loss",
    )
    .await;
    assert_eq!(
        (stream_id.as_str(), seq),
        ("6-0", 6),
        "the consumer must ride out the connection loss and resume from its durable cursor"
    );

    assert_still_healthy(
        &consumer,
        &metrics,
        "a real CLIENT KILL on a cluster node is a transient connection loss",
    );
    drop(consumer);
    drop(dispatch_guard);

    // ── Harvest ──────────────────────────────────────────────────────────────
    let text = String::from_utf8_lossy(&captured.lock().unwrap()).into_owned();
    let mut observed: Vec<(String, String)> = Vec::new();
    for line in text.lines() {
        let Ok(event) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        // Only the stream readers' own events. fred and the rest of the storage
        // layer log through the same dispatcher, and their errors are not
        // governed by this classifier.
        let target = event.get("target").and_then(|t| t.as_str()).unwrap_or("");
        if !target.starts_with("live_edit::storage::stream_mux") {
            continue;
        }
        let Some(fields) = event.get("fields") else {
            continue;
        };
        let (Some(kind), Some(error)) = (
            fields.get("kind").and_then(|k| k.as_str()),
            fields.get("error").and_then(|e| e.as_str()),
        ) else {
            continue;
        };
        observed.push((kind.to_owned(), error.to_owned()));
    }

    assert!(
        !observed.is_empty(),
        "no consumer error events were captured, so this test asserted nothing. The kill above \
         MUST have produced at least one fred error on the consumer's read. Either the capture \
         never reached the consumer's spawned task (see the runtime-flavour note on this test) \
         or the consumer swallowed the loss without reporting it — the latter would itself be a \
         regression: an error that heals silently is one no operator can ever see. Captured \
         output was {} bytes.",
        text.len()
    );

    // Which of the observed errors would have been rejected by KIND and were
    // rescued only by a message match, each tagged with the predicate that
    // rescued it. Reported, never asserted on — see the doc-comment.
    let mut carve_out_only: Vec<String> = Vec::new();

    for (kind_str, display) in &observed {
        let kind = error_kind_from_str(kind_str).unwrap_or_else(|| {
            panic!(
                "fred reported error kind {kind_str:?}, which is unmapped in this test's reverse \
                 table (`error_kind_from_str`) — either a fred variant gated off in the build \
                 this table was written against, or one fred has since added. Full error: \
                 {display:?}. Add it to `error_kind_from_str` AND decide, deliberately, which \
                 side of `is_transient_fred_error` it belongs on"
            )
        });
        // `Display` is `"{kind}: {details}"`, so stripping the prefix recovers
        // exactly what `Error::details()` held.
        let details = display
            .strip_prefix(&format!("{kind_str}: "))
            .unwrap_or(display)
            .to_owned();
        let rebuilt = fred::error::Error::new(kind.clone(), details.clone());

        // Is this error transient by KIND alone? Re-classify the same kind
        // under a message no carve-out can match, rather than restating
        // `is_transient_fred_error`'s kind list here — a second copy of that
        // list would drift.
        let transient_by_kind = is_transient_fred_error(&fred::error::Error::new(
            kind,
            "b.yxt kind-only classification probe (matches no message carve-out)",
        ));
        // EVERY message-matched rescue path, not just b.c7a's (b.cv4). Missing
        // one does not fail the assertion above — it silently misfiles the error
        // as "transient by kind", and the empty-bucket branch below then prints
        // a claim nothing checked. Each rescuer is NAMED so the datapoint says
        // which predicate did the work rather than merely that one did.
        if !transient_by_kind {
            let rescued_by = if is_router_no_connection_error(&rebuilt) {
                Some("is_router_no_connection_error — b.c7a, \"Failed to route command.\"")
            } else if is_router_retry_budget_exhausted_error(&rebuilt) {
                Some("is_router_retry_budget_exhausted_error — b.cv4, router retry-budget exhaustion")
            } else if is_store_not_serving_reply(&rebuilt) {
                Some("is_store_not_serving_reply — b.xd1, the store's `-LOADING` / `-BUSY`")
            } else if is_tryagain_reply(&rebuilt) {
                Some("is_tryagain_reply — b.vn4, a multi-key command's `-TRYAGAIN` during a slot migration")
            } else if is_transaction_router_exhausted_error(&rebuilt) {
                Some("is_transaction_router_exhausted_error — b.cyu, fred's `Transaction` router out of redirections or attempts")
            } else if is_client_unblocked_error(&rebuilt) {
                // Not part of `is_transient_fred_error` at all — the reader
                // retries it on its own. Counted here so the empty bucket really
                // does mean "kind alone sufficed".
                Some("is_client_unblocked_error — b.9yf, `-UNBLOCKED` (reader-local, NOT in the shared classifier)")
            } else {
                None
            };
            if let Some(rescued_by) = rescued_by {
                carve_out_only.push(format!("[{rescued_by}] {kind_str}: {details}"));
            }
        }

        assert!(
            is_transient_fred_error(&rebuilt) || is_client_unblocked_error(&rebuilt),
            "a reader observed a fred error during an ORDINARY cluster connection loss that \
             no classifier predicate accepts, so it would take its permanent exit and end its \
             subscriptions with Code::Unavailable — on a disturbance the client cannot do \
             anything about and the connection heals from on its own.\n\
             \n  kind:    {kind_str}\n  details: {details}\n\
             \n\
             This is exactly the b.c7a failure mode, and exactly the blind spot the hand-built \
             `Error::new(ErrorKind::Unknown, \"Failed to route command.\")` fault-injected \
             fixtures cannot see: they restate the predicate's own assumption \
             about fred's wording. Widen `is_transient_fred_error` / \
             `is_router_no_connection_error` in src/storage/client.rs to cover the message above \
             — quoting it verbatim in the doc-comment, as the existing carve-out does."
        );
    }

    // ── Carve-out visibility ─────────────────────────────────────────────────
    // A datapoint for the CI log, not an assertion: whether this run happened to
    // produce an error that only a MESSAGE match rescued, or whether every error
    // it saw was already transient by kind. Without this line a green run is
    // silent about which of the two it was, and the real-text coverage item
    // would look covered on runs that never touched it.
    //
    // The bucket is filled against EVERY rescue predicate (b.cv4, b.xd1, b.vn4, b.cyu), so
    // the two branches below are genuinely exhaustive of what was observed.
    if carve_out_only.is_empty() {
        eprintln!(
            "b.yxt A6 DATAPOINT: no message-matched carve-out was exercised on this run — each of \
             the {} observed consumer error(s) was transient by KIND alone, checked against every \
             rescue predicate (`is_router_no_connection_error`, \
             `is_router_retry_budget_exhausted_error`, `is_store_not_serving_reply`, \
             `is_tryagain_reply`, `is_transaction_router_exhausted_error`, \
             `is_client_unblocked_error`). Kinds seen: \
             {:?}. This is expected and not a failure — the carve-outs need conditions a CLIENT \
             KILL does not reliably produce (an empty router connection map; a command re-entering \
             `write_command` with its attempt budget already spent) — but it does mean this run \
             demonstrates NO real-text coverage of any carve-out message.",
            observed.len(),
            observed.iter().map(|(k, _)| k).collect::<Vec<_>>()
        );
    } else {
        eprintln!(
            "b.yxt A6 DATAPOINT: a message-matched carve-out WAS exercised on this run — {} of {} \
             observed consumer error(s) were rejected by KIND and accepted ONLY by a message \
             match, so fred's real wording still matches the predicate that rescued it. Each entry \
             is tagged with the predicate responsible: {carve_out_only:?}",
            carve_out_only.len(),
            observed.len()
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// B1 — DISRUPTIVE: a paused node wedges the read, and the consumer self-heals
// ═════════════════════════════════════════════════════════════════════════════

/// A node that stops answering long enough for the consumer's command timeout
/// to fire produces a REAL `Timeout`, which must drive the b.9uq forced
/// reconnect and then resume delivery once the node comes back.
///
/// Every other `Timeout` test in the suite injects the error through the fault
/// queue. That covers the handling but not the premise — that a wedged backend
/// actually presents as `ErrorKind::Timeout` on the consumer's `XREAD`, on a
/// cluster, through fred's per-command timeout. `CLIENT PAUSE ... ALL` is the
/// closest reproducible analogue of the managed-proxy invalidation b.9uq was
/// filed for: the node accepts the connection and never answers, so fred's
/// timeout fires with no FIN, no RST, and no server error — the same signature.
///
/// Timing (block 1 s via the test knobs → fred per-command timeout
/// `block_ms + 5 s` = 6 s; pause 9 s): the parked `XREAD` cannot be answered
/// before the timeout, so the
/// `Timeout` is guaranteed rather than raced. Verified on Valkey 9.0.3 that
/// `CLIENT PAUSE` holds back a blocked client's BLOCK-expiry reply. (`DEBUG
/// SLEEP` would be the more direct instrument but is not available on the
/// harness cluster.)
///
/// 9 s is also deliberately under the default 15 s `cluster-node-timeout`, so
/// the paused primary is not marked failing and no failover is triggered.
///
/// **`#[ignore]` + retry safety.** The pause suspends *every* client of that
/// node, including other test processes sharing the cluster, which is why this
/// is not in the release-gating set. It is retry-safe under nextest's
/// `retries = 1` because a `CLIENT PAUSE` expires on its own after `timeout`
/// milliseconds — a run that panics mid-test leaves nothing to clean up, and the
/// next attempt starts from an unpaused node.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, CLIENT PAUSE suspends a whole cluster node"]
async fn cluster_subscriber_disruptive_node_pause_wedges_and_self_heals() {
    const BLOCK_MS: u64 = 1_000;
    const PAUSE_MS: i64 = 9_000;

    let Some(tv) = cluster_fixture_with_readers(8, 1).await else {
        return;
    };
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        block_ms: BLOCK_MS,
        ..tv.client().stream_mux.knobs()
    });

    let object_id: u64 = 951_500;
    let owner = slot_owner(&tv, hash_slot_of(&tv.key_schema().layer_stream(object_id)));
    xadd_update(&tv, object_id, 5).await;

    let consumer = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From("5-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consumer must start");

    // The pause below must catch the read parked on the owner.
    identify_parked_reader(&tv, &owner).await;
    assert_eq!(
        read_forced_reconnects(&metrics, CONSUMER_PHASE_XREAD),
        0,
        "baseline: nothing has wedged yet"
    );

    tv.client()
        .pool
        .next()
        .with_options(&pinned_to(&owner))
        .client_pause(PAUSE_MS, Some(ClientPauseKind::All))
        .await
        .unwrap_or_else(|e| panic!("CLIENT PAUSE on {owner} must succeed: {e}"));
    let pause_expiry = std::time::Instant::now() + Duration::from_millis(PAUSE_MS as u64);

    // Barrier, not a settle sleep: poll until the forced reconnect is OBSERVED
    // on the counter, with two bounds.
    //
    //   * Floor at the pause expiry. Nothing may be asserted — and nothing may
    //     be written — before the pause lapses: the XADD below would itself be
    //     held by the paused node, so an early exit would time out the delivery
    //     assertion for a reason unrelated to the contract.
    //   * Deadline above it, so a self-heal that never happens fails with a
    //     diagnosis instead of hanging until nextest's hang-killer fires.
    //
    // The old fixed `PAUSE_MS + 3 s` sleep encoded a guess at how long after the
    // pause the reconnect would land; this waits for the thing itself.
    let barrier_deadline = pause_expiry + RECONNECT_BARRIER_BUDGET;
    let forced = loop {
        let forced = read_forced_reconnects(&metrics, CONSUMER_PHASE_XREAD);
        let now = std::time::Instant::now();
        if forced >= 1 && now >= pause_expiry {
            break forced;
        }
        assert!(
            now < barrier_deadline,
            "no forced reconnect was recorded under phase=\"xread\" within \
             {RECONNECT_BARRIER_BUDGET:?} of the \
             {PAUSE_MS} ms pause lapsing (observed {forced}). The pause outlasts the consumer's \
             {BLOCK_MS} ms BLOCK plus its 5 s command budget, so the XREAD MUST have timed out \
             by now. A zero here means either the pause did not hold back the blocked read \
             (check the Valkey version's CLIENT PAUSE semantics — this test is only meaningful \
             if it does) or the Timeout was not read as a wedge verdict, which is the b.9uq \
             regression"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    };

    // The self-heal must be visible to operators, attributed to the read arm.
    // This counter is the only direct signal that subscriber connections are
    // being silently invalidated upstream — `valkey_reconnects_total` counts
    // fred's own reconnects and would not distinguish them.
    assert!(
        forced >= 1,
        "a {PAUSE_MS} ms node pause outlasts the consumer's {BLOCK_MS} ms BLOCK plus its 5 s \
         command budget, so the XREAD MUST have timed out and forced at least one reconnect \
         under phase=\"xread\"; got {forced}. A zero here means either the pause did not hold \
         back the blocked read (check the Valkey version's CLIENT PAUSE semantics — this test \
         is only meaningful if it does) or the Timeout was not read as a wedge verdict, which \
         is the b.9uq regression"
    );

    // Delivery resumes on the healed connection, from the durable cursor.
    xadd_update(&tv, object_id, 6).await;
    let (stream_id, seq) = expect_update(
        &consumer,
        SELF_HEAL_DELIVERY_BUDGET,
        "delivery after the node pause lapsed",
    )
    .await;
    assert_eq!(
        (stream_id.as_str(), seq),
        ("6-0", 6),
        "after the pause lapses the consumer must resume from its `5-0` cursor on the rebuilt \
         connection — a wedge the self-heal never recovered from looks exactly like the b.9uq \
         silent stall"
    );

    assert_still_healthy(
        &consumer,
        &metrics,
        "a node pause is a Timeout — inside the transient taxonomy, so no permanent exit may be \
         recorded",
    );
    drop(consumer);
}

// ═════════════════════════════════════════════════════════════════════════════
// B2 — DISRUPTIVE: an ASK redirect during a live slot migration
// ═════════════════════════════════════════════════════════════════════════════

/// Per-command budget for the migration primitives.
///
/// The fixture's fred client is built with `valkey_command_timeout_ms = 5000`
/// (`TestValkey::setup_with_url`) and `MIGRATE` carries its OWN 5000 ms
/// server-side timeout argument — so at the default budget fred can give up at
/// the exact moment the server is still legitimately working, surfacing a
/// `Timeout` that says nothing about whether the key moved and leaving the slot
/// half-staged. Every command in this section is issued with a budget
/// comfortably above the server-side one so a failure means the *server*
/// refused.
#[cfg(feature = "test-fault-injection")]
const MIGRATION_COMMAND_TIMEOUT: Duration = Duration::from_secs(20);

/// How many `GETKEYSINSLOT` batches the pre-completion drain may take before it
/// gives up. At 64 keys per batch this is 2048 keys — orders of magnitude above
/// anything this fixture creates, so hitting the cap means the drain is not
/// converging (a key being rewritten into the slot as fast as it is moved, or a
/// `MIGRATE` that silently no-ops) rather than that the slot is merely busy.
/// Without the cap that case is an infinite loop inside a `#[ignore]`d job.
#[cfg(feature = "test-fault-injection")]
const MIGRATION_DRAIN_MAX_BATCHES: usize = 32;

/// Wall-clock bound on the whole completion drain (b.cv4).
///
/// [`MIGRATION_DRAIN_MAX_BATCHES`] bounds the drain in *batches*, which was not
/// enough: 32 batches × two commands × [`MIGRATION_COMMAND_TIMEOUT`] is ~21
/// minutes, so a drain that converges slowly (rather than not at all) blew
/// straight past nextest's 120 s hard-kill and B2 was SIGKILLed with no
/// message. This is the arithmetic-free backstop — whichever cap trips first
/// wins, and both produce the same diagnostic Err rather than a dead worker.
///
/// On the normal path the drain finds the slot already empty and exits after one
/// `GETKEYSINSLOT`, so none of this is paid.
#[cfg(feature = "test-fault-injection")]
const MIGRATION_DRAIN_BUDGET: Duration = Duration::from_secs(30);

/// `Options` for the migration primitives: pinned to one node, fred's own
/// retrying out of the picture, and the wider [`MIGRATION_COMMAND_TIMEOUT`].
#[cfg(feature = "test-fault-injection")]
fn pinned_for_migration(node: &Server) -> Options {
    Options {
        timeout: Some(MIGRATION_COMMAND_TIMEOUT),
        ..pinned_to(node)
    }
}

/// Run one node-targeted command built from raw arguments, surfacing the error
/// instead of panicking.
///
/// fred models a custom command's name as a single token, so multi-word
/// commands (`CLUSTER SETSLOT`) pass their subcommand as the first argument.
/// `cluster_node` pins the node; it takes precedence over any hashing policy,
/// which is what lets a key-bearing command like `MIGRATE` be aimed at the
/// source shard explicitly.
///
/// The `Result` shape exists for [`SlotMigration`]'s `Drop` guard: it runs
/// during unwind, where a panic would abort the process instead of failing the
/// test.
#[cfg(feature = "test-fault-injection")]
async fn try_raw_command_on(
    tv: &TestValkey,
    node: &Server,
    cmd: &'static str,
    args: &[String],
) -> Result<(), String> {
    use fred::types::CustomCommand;

    let command = CustomCommand::new_static(cmd, None::<u16>, false);
    tv.client()
        .pool
        .next()
        .with_options(&pinned_for_migration(node))
        .custom::<fred::types::Value, _>(command, args.to_vec())
        .await
        .map(|_| ())
        .map_err(|e| format!("`{cmd} {}` on {node} failed: {e}", args.join(" ")))
}

/// Panicking wrapper on [`try_raw_command_on`] for the happy path.
#[cfg(feature = "test-fault-injection")]
async fn raw_command_on(tv: &TestValkey, node: &Server, cmd: &'static str, args: &[String]) {
    if let Err(msg) = try_raw_command_on(tv, node, cmd, args).await {
        panic!("{msg}");
    }
}

/// The keys a node still holds in `slot` (`CLUSTER GETKEYSINSLOT`), capped at a
/// batch that comfortably exceeds anything this fixture creates. Non-panicking
/// for the same reason as [`try_raw_command_on`].
#[cfg(feature = "test-fault-injection")]
pub(crate) async fn try_keys_in_slot(
    tv: &TestValkey,
    node: &Server,
    slot: u16,
) -> Result<Vec<String>, String> {
    use fred::types::CustomCommand;

    let command = CustomCommand::new_static("CLUSTER", None::<u16>, false);
    tv.client()
        .pool
        .next()
        .with_options(&pinned_for_migration(node))
        .custom::<Vec<String>, _>(
            command,
            vec![
                "GETKEYSINSLOT".to_string(),
                slot.to_string(),
                "64".to_string(),
            ],
        )
        .await
        .map_err(|e| format!("CLUSTER GETKEYSINSLOT {slot} on {node} failed: {e}"))
}

/// A node's own cluster id.
#[cfg(feature = "test-fault-injection")]
async fn node_id(tv: &TestValkey, node: &Server) -> String {
    tv.client()
        .pool
        .next()
        .with_options(&pinned_for_migration(node))
        .cluster_myid::<String>()
        .await
        .unwrap_or_else(|e| panic!("CLUSTER MYID on {node} must succeed: {e}"))
}

/// How far a hand-driven slot migration has got. The unwind guard reads this to
/// decide which end state is the CONSISTENT one.
#[cfg(feature = "test-fault-injection")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MigrationStage {
    /// `CLUSTER SETSLOT <slot> IMPORTING` accepted on the destination; the
    /// source still owns and serves the slot.
    Importing,
    /// Plus `CLUSTER SETSLOT <slot> MIGRATING` on the source. Both flags set,
    /// but no data has moved: the source still owns and serves every key.
    Migrating,
    /// At least one `MIGRATE` was ISSUED, so data may already live on the
    /// destination.
    KeyMoved,
    /// `CLUSTER SETSLOT <slot> NODE <dst>` applied on both nodes — the
    /// destination owns the slot and the routing table has been resynced.
    Completed,
}

/// A hand-driven slot migration that always leaves the cluster in a consistent
/// end state — including when the test panics part-way through.
///
/// **Why a guard rather than a completion block.** The B2 experiment stages a
/// migration, observes a live consumer, and then completes it. Any panic before
/// the completion — a `MIGRATE` the source refuses, an `expect_update` deadline,
/// a fixture self-check — used to leave the slot MIGRATING on the source and
/// IMPORTING on the destination. Every subsequent access to a key in that slot
/// then answers `-ASK`, and nextest's `retries = 1` re-runs this very test
/// straight into the wreckage: the retry derives a *stale* owner from a routing
/// table that no longer describes reality, and its own migration is staged on
/// top of a half-finished one. The first failure is a finding; the second is
/// noise generated by the harness.
///
/// **Which end state, and why it depends on the stage.** The rule is "never
/// leave a key where nobody will look for it":
///
/// * before any `MIGRATE` (`Importing` / `Migrating`) the source still holds
///   every key and still owns the slot, so the correct undo is
///   `CLUSTER SETSLOT <slot> STABLE` on BOTH nodes — it clears the two flags and
///   restores exactly the pre-test topology. Completing forward here would be a
///   gratuitous topology change on a shared fixture.
/// * once a `MIGRATE` has been ISSUED (`KeyMoved`) the data may already sit on
///   the destination, which does NOT own the slot — a `STABLE` rollback would
///   strand it there, reachable by nobody and invisible until some later reader
///   found the key missing. The only consistent end state is forward:
///   drain the remainder, then `SETSLOT NODE <dst>` on both.
///
/// The stage is advanced BEFORE the command that reaches it is issued wherever
/// the command's failure could still have taken effect server-side (`MIGRATE`
/// can move the key and then fail on the reply path), so the guard errs toward
/// completing forward rather than stranding data.
///
/// Shared with `snapshot_read_cluster_test.rs`'s disruptive window-script
/// fallback test, `b_vn4_tryagain_test.rs`'s `slot_migration_disruptive`
/// test, which stages `MIGRATING` and moves no key, and
/// `b_cyu_slot_migration_test.rs`'s, some of which move every key in the slot
/// (with [`try_keys_in_slot`]) before completing.
#[cfg(feature = "test-fault-injection")]
pub(crate) struct SlotMigration<'a> {
    tv: &'a TestValkey,
    slot: u16,
    src: Server,
    dst: Server,
    dst_id: String,
    stage: MigrationStage,
}

#[cfg(feature = "test-fault-injection")]
impl<'a> SlotMigration<'a> {
    /// Set `IMPORTING` on the destination and `MIGRATING` on the source. From
    /// here on the slot is half-staged and the guard is armed.
    pub(crate) async fn begin(
        tv: &'a TestValkey,
        slot: u16,
        src: Server,
        dst: Server,
    ) -> SlotMigration<'a> {
        let src_id = node_id(tv, &src).await;
        let dst_id = node_id(tv, &dst).await;

        raw_command_on(
            tv,
            &dst,
            "CLUSTER",
            &[
                "SETSLOT".into(),
                slot.to_string(),
                "IMPORTING".into(),
                src_id,
            ],
        )
        .await;
        // Armed from here: `migration` is a local, so an unwind out of the
        // MIGRATING command below drops it and runs the STABLE rollback.
        let mut migration = SlotMigration {
            tv,
            slot,
            src,
            dst,
            dst_id: dst_id.clone(),
            stage: MigrationStage::Importing,
        };
        raw_command_on(
            tv,
            &migration.src,
            "CLUSTER",
            &[
                "SETSLOT".into(),
                migration.slot.to_string(),
                "MIGRATING".into(),
                dst_id,
            ],
        )
        .await;
        migration.stage = MigrationStage::Migrating;
        migration
    }

    /// Move one key to the destination. With the slot MIGRATING on the source
    /// and the key gone from it, the source answers `-ASK <slot> <dst>` for
    /// every subsequent access — which is the redirect B2 exists to observe.
    pub(crate) async fn move_key(&mut self, key: &str) {
        // Advance the stage FIRST: a MIGRATE that fails on the reply path may
        // still have moved the key, and the guard must not strand it.
        self.stage = MigrationStage::KeyMoved;
        raw_command_on(
            self.tv,
            &self.src,
            "MIGRATE",
            &[
                self.dst.host.to_string(),
                self.dst.port.to_string(),
                key.to_string(),
                "0".into(),
                "5000".into(),
                // REPLACE: a retry (nextest `retries = 1`, or the guard
                // completing a migration whose first MIGRATE already landed)
                // would otherwise fail with `BUSYKEY` on a destination that
                // already holds the key, turning a recoverable state into a
                // second failure.
                "REPLACE".into(),
            ],
        )
        .await;
    }

    /// Transfer ownership of the whole slot to the destination, draining
    /// anything still on the source first. Panics on failure — this is the
    /// happy path.
    pub(crate) async fn complete(&mut self) {
        if let Err(msg) = self.complete_inner().await {
            panic!("completing the slot migration failed: {msg}");
        }
        self.stage = MigrationStage::Completed;
    }

    /// Non-panicking body of [`Self::complete`], shared with the `Drop` guard.
    async fn complete_inner(&self) -> Result<(), String> {
        // `CLUSTER SETSLOT <slot> NODE <dst>` transfers ownership of the WHOLE
        // slot, so anything still living on the source in that slot would be
        // stranded — reachable by nobody, and invisible until some later test
        // read it back as missing. Drain the remainder first. In this job's
        // isolated cluster the only occupant is normally this fixture's own key
        // (already moved), but the loop keeps the invariant true rather than
        // assumed.
        let drain_deadline = std::time::Instant::now() + MIGRATION_DRAIN_BUDGET;
        for batch in 0..MIGRATION_DRAIN_MAX_BATCHES {
            let remaining = try_keys_in_slot(self.tv, &self.src, self.slot).await?;
            if remaining.is_empty() {
                break;
            }
            // Whichever cap trips first. The batch count catches a drain that is
            // not converging; this one catches a drain that is converging too
            // slowly to finish inside nextest's 120 s terminate (b.cv4).
            if batch + 1 == MIGRATION_DRAIN_MAX_BATCHES
                || std::time::Instant::now() >= drain_deadline
            {
                return Err(format!(
                    "slot {slot} still holds {n} key(s) on {src} after {batches} drain batches / \
                     {MIGRATION_DRAIN_BUDGET:?} ({remaining:?}) — the drain is not converging fast \
                     enough, so the slot is left MIGRATING rather than looped on until nextest \
                     SIGKILLs this test",
                    slot = self.slot,
                    batches = batch + 1,
                    n = remaining.len(),
                    src = self.src
                ));
            }
            let mut args: Vec<String> = vec![
                self.dst.host.to_string(),
                self.dst.port.to_string(),
                // Empty key + `KEYS ...` is the multi-key MIGRATE form.
                String::new(),
                "0".into(),
                "5000".into(),
                "REPLACE".into(),
                "KEYS".into(),
            ];
            args.extend(remaining);
            try_raw_command_on(self.tv, &self.src, "MIGRATE", &args).await?;
        }

        for node in [&self.dst, &self.src] {
            try_raw_command_on(
                self.tv,
                node,
                "CLUSTER",
                &[
                    "SETSLOT".into(),
                    self.slot.to_string(),
                    "NODE".into(),
                    self.dst_id.clone(),
                ],
            )
            .await?;
        }

        // Refresh our own cached routing table so the assertions that follow —
        // and any subsequent test in this process — see the new owner.
        self.tv
            .client()
            .pool
            .next()
            .sync_cluster()
            .await
            .map_err(|e| format!("CLUSTER SLOTS resync after completing the migration failed: {e}"))
    }

    /// Clear both staging flags, restoring the pre-test topology. Only correct
    /// while no key has moved — see the type's doc-comment.
    async fn rollback_inner(&self) -> Result<(), String> {
        for node in [&self.src, &self.dst] {
            try_raw_command_on(
                self.tv,
                node,
                "CLUSTER",
                &["SETSLOT".into(), self.slot.to_string(), "STABLE".into()],
            )
            .await?;
        }
        Ok(())
    }
}

#[cfg(feature = "test-fault-injection")]
impl Drop for SlotMigration<'_> {
    fn drop(&mut self) {
        if self.stage == MigrationStage::Completed {
            return;
        }
        let stage = self.stage;

        // `block_in_place` panics outside a multi-thread runtime, and panicking
        // in `Drop` during an unwind aborts the process. B2 is
        // `flavor = "multi_thread"`; say so loudly rather than crash if that
        // ever changes.
        let multi_thread = tokio::runtime::Handle::try_current()
            .is_ok_and(|h| h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread);
        if !multi_thread {
            eprintln!(
                "b.yxt B2 WARNING: slot {slot} was left mid-migration at stage {stage:?} and the \
                 unwind guard cannot recover it from a non-multi-thread runtime (recovery needs \
                 `block_in_place`). Give the test `#[tokio::test(flavor = \"multi_thread\")]`.",
                slot = self.slot
            );
            return;
        }

        // Async work in `Drop`: `block_in_place` hands this worker back to the
        // multi-thread runtime (B2 is `flavor = "multi_thread"`) so fred's IO
        // tasks keep being polled while `block_on` drives the recovery here.
        //
        // Everything is wrapped in `catch_unwind` because this normally runs
        // DURING an unwind, where a second panic aborts the process — turning a
        // failed test into a dead nextest worker and losing the original
        // failure message, which is the one worth reading.
        let recovered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    match stage {
                        MigrationStage::Importing | MigrationStage::Migrating => {
                            self.rollback_inner().await
                        }
                        // Data may already be on the destination: only forward
                        // is consistent.
                        MigrationStage::KeyMoved => self.complete_inner().await,
                        MigrationStage::Completed => Ok(()),
                    }
                })
            })
        }));

        match recovered {
            Ok(Ok(())) => eprintln!(
                "b.yxt B2: slot {slot} left mid-migration at stage {stage:?}; the unwind guard \
                 drove it to a consistent end state ({end}), so a nextest retry starts from a \
                 clean topology.",
                slot = self.slot,
                end = match stage {
                    MigrationStage::KeyMoved => "migration completed forward onto the destination",
                    _ => "SETSLOT STABLE rollback on both nodes",
                }
            ),
            Ok(Err(msg)) => eprintln!(
                "b.yxt B2 WARNING: slot {slot} was left mid-migration at stage {stage:?} and the \
                 unwind guard could NOT recover it: {msg}. The cluster in this job is now \
                 half-migrated — expect the nextest retry of this test, and any other test \
                 touching slot {slot}, to fail for that reason rather than on their own merits.",
                slot = self.slot
            ),
            Err(_) => eprintln!(
                "b.yxt B2 WARNING: the unwind guard for slot {slot} (stage {stage:?}) itself \
                 panicked; the panic was caught so the original failure survives, but the slot is \
                 still half-migrated.",
                slot = self.slot
            ),
        }
    }
}

/// A live subscription must never end CLEANLY because of a slot migration — and
/// this test samples what a LIVE migration actually surfaces to the consumer.
///
/// **What is still open here.** The classification question this test was
/// originally built to settle is **closed**: the redirect budget-exhaustion
/// family is now transient (`is_router_retry_budget_exhausted_error`,
/// `src/storage/client.rs`). It was not closed by this test but by a
/// negative control that mis-pinned an `XLEN` to a non-owner primary, drew a
/// `MOVED` the router had no attempt budget left to follow, and surfaced
/// `Unknown` / `"Too many failed write attempts."`: a shape nobody predicted
/// from reading fred's source.
///
/// **Since b.azh the reader is a single-node client** to its slot's owner, so
/// fred's cluster router no longer handles its redirects at all: during the
/// migration the source's `-ASK` reaches the reader as a raw `Cluster`-kind
/// error (transient: it backs off and retries the source), and once the slot
/// is handed over the source answers `-MOVED`, which makes the reader
/// re-resolve the slot's owner and retarget its connection to the destination
/// (`is_moved_redirect`; pinned deterministically on the two-node facade in
/// `stream_mux_reader_connect_test.rs`). What remains unmeasured is what a
/// **real slot migration** surfaces along the way, and whether the retarget
/// follows it: this test records both. A migration that produces something
/// the reader does not survive is the finding worth having.
///
/// **One hard assertion, one reported datapoint — and the split is deliberate.**
///
/// 1. *Asserted*: the b.c7a contract, which holds whatever the classification
///    turns out to be. The subscription either keeps delivering or surfaces
///    `Code::Unavailable`; it must NEVER end with a clean `Ok(())`. That is
///    wrong for a migration under every classification of the ASK redirect, so
///    it is a real regression guard and it fails the job when it trips.
/// 2. *Reported, not asserted*: WHICH of the two the consumer did — rode the
///    migration out, or gave up with `Unavailable` (and with which fred kind and
///    message) — and, after the migration completes, whether an update written
///    then was delivered, how many times the reader retargeted
///    (`live_edit_stream_mux_reader_connects_total{cause="retarget"}`) and which
///    node it reads from. The expectation is the ride-out and one retarget to
///    the destination: the reader backs off on the `-ASK`s and re-issues from
///    its durable cursor, then follows the `-MOVED`. A give-up therefore
///    **contradicts** the current classification and is the single most
///    interesting thing this test can print — which is precisely why it must not
///    be an assertion. Pinning it would make the arm that *reports the finding*
///    the arm that turns the job red. The datapoint is `eprintln!`ed with a
///    `B2 DATAPOINT:` prefix and
///    read out of the `test-cluster-disruptive:` job log, which is legible only
///    because `.config/nextest.toml` carries a `success-output` override for
///    this test — see that file. The line also carries
///    `valkey_router_restarts_total{pool="subscriber_pool"}` (b.v6d): on stock
///    fred an `-ASK` the one-attempt `XREAD` could not follow ended the
///    reader's router task, which the router supervisor now restarts and
///    counts; with the vendored fred patch it should read 0. Reported, not
///    asserted, for the same reason.
///
/// **Why the measurement is not an assertion.** B2 shares the
/// `test-cluster-disruptive:` job with B1
/// (`cluster_subscriber_disruptive_node_pause_wedges_and_self_heals`), which is
/// a genuine regression test with no open question in it, and the job blocks
/// the pipeline. An arm that can legitimately fail — because fred's real
/// behaviour differs from what the source reading predicted — in the same job
/// masks B1's signal: a red job becomes ambiguous ("did fred's migration
/// behaviour change, or did the b.9uq self-heal break?"), and it would block a
/// pipeline on a finding rather than a regression. With the measurement
/// demoted to a log line, a red `test-cluster-disruptive:` job means a
/// regression in one of its tests — B1, B2's b.c7a contract, or a sibling's
/// — and nothing else.
///
/// **`#[ignore]` + retry safety.** Resharding perturbs every client of both
/// shards, so this is not release-gating. The migration is driven through
/// [`SlotMigration`], whose `Drop` guard leaves the slot in a consistent end
/// state even when the test panics part-way through staging — see that type's
/// doc-comment. A nextest retry therefore always starts from a coherent
/// topology, re-deriving the (possibly now different) owner from the refreshed
/// routing table.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, migrates a hash slot between shards"]
async fn cluster_subscriber_disruptive_ask_redirect_during_slot_migration() {
    let Some(tv) = cluster_fixture(8).await else {
        return;
    };
    let metrics = install_metrics(&tv).await;
    // Short BLOCK so the reader re-issues promptly: a slot migration does not
    // necessarily unblock an already-parked XREAD, so the redirect is observed
    // on the NEXT read.
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        block_ms: 500,
        ..tv.client().stream_mux.knobs()
    });

    let object_id: u64 = 951_600;
    let stream_key = tv.key_schema().layer_stream(object_id);
    let slot = hash_slot_of(&stream_key);
    let src = slot_owner(&tv, slot);
    let dst = primaries(&tv)
        .into_iter()
        .find(|s| *s != src)
        .expect("a 3-shard cluster must have a primary to migrate the slot TO");

    xadd_update(&tv, object_id, 5).await;

    let consumer = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From("5-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consumer must start");

    // Let the consumer establish itself on the source shard before the topology
    // moves under it: one real delivery proves it is parked on `src` and
    // reading, so anything that happens afterwards is attributable to the
    // migration rather than to a consumer that never started.
    xadd_update(&tv, object_id, 6).await;
    let (stream_id, _) = expect_update(
        &consumer,
        DELIVERY_BUDGET,
        "pre-migration delivery (establishes the consumer on the source shard)",
    )
    .await;
    assert_eq!(
        stream_id, "6-0",
        "fixture self-check: the pre-migration entry must be the one just written"
    );

    // ── Stage the migration ──────────────────────────────────────────────────
    // Everything from here to `complete()` is covered by the unwind guard: a
    // panic in the observation window below leaves the slot consistent rather
    // than MIGRATING/IMPORTING for the nextest retry to trip over.
    let mut migration = SlotMigration::begin(&tv, slot, src.clone(), dst.clone()).await;
    migration.move_key(&stream_key).await;

    // ── Observe ──────────────────────────────────────────────────────────────
    // Give the consumer several BLOCK cycles to re-issue into the migrating
    // slot, then record what it did. Nothing is asserted yet: the migration is
    // completed first, so the cluster is back to a coherent topology before any
    // assertion or datapoint report can fail the test.
    let outcome = tokio::time::timeout(MIGRATION_OBSERVATION_WINDOW, consumer.recv()).await;
    let observed_terminal = consumer.take_terminal_exit().map(|exit| exit.error);
    let forced_xread = read_forced_reconnects(&metrics, CONSUMER_PHASE_XREAD);
    let permanent_xread = read_permanent_exits(&metrics, CONSUMER_PHASE_XREAD);
    // b.v6d T40: a stream reader's router task ending (an `-ASK` its one-attempt
    // XREAD could not follow, on stock fred) is restarted and counted here; the
    // vendored fred patch should keep this at 0 through a live migration.
    let reader_router_restarts = read_series(
        &metrics,
        "live_edit_valkey_router_restarts_total",
        &[("pool", "subscriber_pool")],
    );

    // ── Complete the migration ───────────────────────────────────────────────
    migration.complete().await;

    // ── After the hand-over (b.azh): the reader's retarget ───────────────────
    // The source now answers the reader's XREAD `-MOVED`; the reader should
    // re-resolve the owner and retarget to `dst`. Sampled, not asserted (see
    // the doc-comment): an update written now, waited for one delivery budget.
    let post_migration = if observed_terminal.is_none() {
        xadd_update(&tv, object_id, 7).await;
        let delivered = tokio::time::timeout(DELIVERY_BUDGET, consumer.recv()).await;
        format!("{delivered:?}")
    } else {
        "not sampled: the consumer had already ended".to_owned()
    };
    let retargets = read_series(
        &metrics,
        "live_edit_stream_mux_reader_connects_total",
        &[("cause", "retarget"), ("result", "ok")],
    );
    // The reader serving the subscription: this fixture has several readers,
    // and the slot's binding, not the index, decides which one serves it.
    let reader_node = tv
        .client()
        .stream_mux
        .snapshot()
        .readers
        .into_iter()
        .find(|reader| reader.subscriptions > 0)
        .map_or_else(
            || "no reader (none holds the subscription)".to_owned(),
            |reader| format!("{:?}", reader.node),
        );

    // ── Assert (one thing only) ──────────────────────────────────────────────
    // The b.c7a contract, independent of whatever shape the migration produces.
    // This is the whole hard-failure surface of this test — see the doc-comment
    // for why the measurement is reported rather than asserted.
    let ended_cleanly = matches!(outcome, Ok(None)) && observed_terminal.is_none();
    assert!(
        !ended_cleanly,
        "the subscription ended with a CLEAN close during a slot migration: `recv()` returned \
         None and the terminal-error slot was empty. b.c7a: an empty slot is an affirmative \
         claim that the subscription ended NORMALLY — `run_steady_state` turns it into a clean \
         gRPC OK, the client's retry policy never fires, and the subscriber silently stops \
         receiving updates. A resharding window must surface as Code::Unavailable (client \
         re-subscribes onto the new topology) or not at all (the consumer rides it out); \
         `Ok(())` is wrong under every classification of the ASK redirect"
    );

    // ── Report the measurement (never fails the job) ─────────────────────────
    // What a LIVE slot migration surfaces to the consumer, in enough detail to
    // compare against the pinned-`XLEN` control — the only member of
    // `BUDGET_EXHAUSTED_SHAPES` (src/storage/client.rs) actually observed so
    // far. The classification itself is settled: the family is transient, so the
    // expected arm here is the ride-out.
    let behaviour = match (&outcome, &observed_terminal) {
        (Ok(None), Some(live_edit::error::ServiceError::Unavailable(msg))) => {
            let code =
                tonic::Status::from(live_edit::error::ServiceError::Unavailable(msg.clone()))
                    .code();
            format!(
                "surfaced Unavailable and GAVE UP (wire code {code:?}) — this CONTRADICTS the \
                 current classification, which makes the redirect budget-exhaustion family \
                 TRANSIENT and expects the consumer to back off and re-issue from its durable \
                 cursor. Read the terminal message below: if its fred kind + text are outside \
                 `is_router_retry_budget_exhausted_error`'s four shapes, a live migration \
                 produces something the pinned-`XLEN` control never showed and the predicate \
                 needs that shape added (kind-gated, prefix-anchored, with its fred file:line). \
                 If they are INSIDE the four, the give-up is not a classification gap and the \
                 fault is downstream — the consumer exhausted its transient budget, or the \
                 migration outlasted it. Terminal message: {msg:?}"
            )
        }
        (Ok(None), Some(other)) => format!(
            "exited with an UNEXPECTED terminal error {other:?} — neither the expected ride-out \
             nor the ServiceError::Unavailable give-up"
        ),
        (Ok(None), None) => {
            "ended cleanly (the hard assertion above has already failed)".to_string()
        }
        (Ok(Some(n)), terminal) => format!(
            "RODE THE MIGRATION OUT and kept delivering ({n:?}, terminal slot {terminal:?}) — the \
             expected result post-b.cv4: the redirect was classified transient and the loop \
             re-issued from its durable cursor onto the re-resolved topology. Note this arm does \
             NOT confirm which fred shape appeared, only that whatever appeared was survivable; \
             the shape itself is visible in the consumer's WARN lines"
        ),
        (Err(_elapsed), terminal) => format!(
            "stayed alive and idle for the whole {MIGRATION_OBSERVATION_WINDOW:?} observation \
             window (terminal slot {terminal:?}) — it neither gave up nor delivered, so the \
             migrating slot produced no consumer-visible event at all"
        ),
    };
    eprintln!(
        "b.yxt B2 DATAPOINT: consumer {behaviour}.\n\
         Counters during the migration window: \
         forced_reconnects{{phase=\"xread\"}}={forced_xread}, \
         permanent_exits{{phase=\"xread\"}}={permanent_xread}, \
         valkey_router_restarts_total{{pool=\"subscriber_pool\"}}={reader_router_restarts}.\n\
         After the hand-over (b.azh retarget): update 7-0 written then: {post_migration}; \
         reader_connects_total{{cause=\"retarget\",result=\"ok\"}}={retargets}; the reader \
         reads from {reader_node} (source {src}, destination {dst}).\n\
         This line is the measurement — REPORTED, not asserted, so this job's red/green stays a \
         verdict on the b.c7a contract (and on B1, which shares the job) rather than on fred's \
         real migration behaviour. The open question it samples: does a LIVE migration produce \
         the same budget-exhaustion shapes the pinned-`XLEN` control did? Record the answer on \
         the follow-up read-out ticket."
    );

    drop(consumer);
}

// ═════════════════════════════════════════════════════════════════════════════
// b.f4f — the new checkout on a real cluster
// ═════════════════════════════════════════════════════════════════════════════

/// A pinned `txn_pool` checkout still routes its probe, its `WATCH` and its
/// `EXEC` to the key's own shard, and still returns its permit, after b.f4f
/// replaced `ExclusivePool::acquire` with `ExclusiveCheckout` (b.f4f).
///
/// Why a cluster test at all when the checkout is pure in-process scheduling:
/// the *value* it hands out changed shape. `acquire_txn_for` used to return
/// `(OwnedMutexGuard<Client>, PoolUseToken)` and now returns one
/// `CheckedOutClient` that `WatchGuard::new` takes whole and moves into its
/// spawned UNWATCH body. Everything downstream of that value is keyless
/// connection state — `DISCARD` and `UNWATCH` carry no key, so fred's
/// `ClusterHash::FirstKey` policy has nothing to hash and falls through to an
/// ARBITRARY primary unless the call site pins it. On one node that is invisible:
/// every routing decision resolves to the same socket, so a lost pin passes the
/// whole single-node suite and fails in production as the `CROSSSLOT` /
/// `DEADLINE_EXCEEDED` cascade b.7td was filed for. SDD §15.3.
///
/// Two distinct shards, because a single-shard run would prove nothing the
/// single-node suite does not already cover — the fixture asserts the two keys
/// really hash to different slots before it relies on them.
///
/// Skips cleanly without `LIVE_EDIT_TEST_CLUSTER_URL`; CI's `test-cluster:` job
/// runs it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_pinned_txn_checkout_commits_and_returns_its_permit_on_each_shard() {
    use fred::interfaces::{KeysInterface, TransactionInterface};
    use fred::types::Value as FredValue;
    use live_edit::storage::watch_guard::WatchGuard;
    use std::ops::Deref;

    /// Unreachable on the healthy path; a deadline, not a measurement.
    const SETTLE_DEADLINE: Duration = Duration::from_secs(20);

    let Some(tv) = cluster_fixture(4).await else {
        return;
    };

    let shards = object_ids_on_distinct_shards(&tv, 2);
    let keys: Vec<String> = shards
        .iter()
        .map(|(oid, _)| tv.key_schema().obj_header(&oid.to_string()))
        .collect();
    assert_ne!(
        hash_slot_of(&keys[0]),
        hash_slot_of(&keys[1]),
        "fixture self-check: the two transaction keys must live on different hash slots, or \
         this test degrades to the single-node case"
    );

    let permits_before = tv.client().txn_pool.available_permits();
    assert!(
        permits_before >= 1,
        "fixture self-check: the txn pool must have at least one permit to hand out; got \
         {permits_before}"
    );

    for (key, (_oid, owner)) in keys.iter().zip(shards.iter()) {
        let _: () = tv
            .client()
            .pool
            .set(key, "initial", None, None, false)
            .await
            .unwrap_or_else(|e| panic!("seed SET {key} (owner {owner}) must succeed: {e}"));

        // The checkout's probe (`DISCARD` then `UNWATCH`, both keyless) is
        // pinned to this key's slot by `acquire_txn_for`. A lost pin surfaces
        // here as a MOVED/CROSSSLOT error out of the probe or the EXEC below.
        let mut guard = WatchGuard::new(
            tv.client().acquire_txn_for(key).await,
            tv.client().watch_unwatch_tracker(),
        );
        guard.deref().watch::<&str>(key).await.unwrap_or_else(|e| {
            panic!("WATCH {key} on its owner shard ({owner}) must succeed: {e}")
        });
        guard.mark_watched(key.as_str(), &[]);

        let trx = guard.deref().multi();
        let _: () = trx
            .set(key, "committed-under-watch", None, None, false)
            .await
            .unwrap_or_else(|e| panic!("MULTI queue SET {key} must succeed: {e}"));
        let reply: FredValue = trx.exec(true).await.unwrap_or_else(|e| {
            panic!("EXEC on {key}'s owner shard ({owner}) must not error: {e}")
        });
        assert!(
            !reply.is_null(),
            "EXEC on {key} (owner {owner}) returned nil — the WATCH was tripped, which on an \
             otherwise-idle key means the transaction ran on a connection that did not hold the \
             WATCH: a routing pin was lost somewhere between `acquire_txn_for` and `WatchedTxn`"
        );

        // `exec()` clears the watch set, so this Drop takes the inline arm; the
        // spawned-UNWATCH arm's permit accounting is covered on a single node by
        // `watch_leak_test::watch_guard_spawned_unwatch_returns_its_permit_as_well_as_its_slot`.
        drop(guard);

        let stored: Option<String> = tv
            .client()
            .pool
            .get(key)
            .await
            .unwrap_or_else(|e| panic!("GET {key} must succeed: {e}"));
        assert_eq!(
            stored.as_deref(),
            Some("committed-under-watch"),
            "the committed value must be readable from {key}'s owner shard ({owner})"
        );
    }

    // Every permit is back. A checkout that stranded one on a cluster would show
    // up only as a pool that is quietly one slot smaller per shard visited.
    let deadline = std::time::Instant::now() + SETTLE_DEADLINE;
    while tv.client().watch_unwatch_inflight() > 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "a WatchGuard teardown never finished"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        tv.client().txn_pool.available_permits(),
        permits_before,
        "the txn pool lost a permit across two pinned cluster transactions"
    );
    assert_eq!(
        tv.client().txn_pool.waiters(),
        0,
        "a `WaiterSlot` guard leaked on the txn pool"
    );
}
