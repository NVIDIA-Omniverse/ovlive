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

//! b.ji8 — `WatchGuard` teardown must be routed back to the cluster node that
//! is actually holding the WATCH.
//!
//! # The defect
//!
//! `UNWATCH` carries no arguments. fred builds it via `ok_cmd!(unwatch,
//! Unwatch)` with an empty argument list, so the default
//! `ClusterHash::FirstKey` policy finds no key to hash,
//! `Command::cluster_hash()` yields `None`, and the router falls through to
//! `writers.values_mut().next()` (`fred-10.1.0/src/router/mod.rs`) — an
//! ARBITRARY node. On a three-primary cluster roughly two teardowns in three
//! therefore land on a node that never saw the WATCH, and the WATCH stands.
//!
//! The connection is then returned to the shared `txn_pool` carrying state the
//! service believes it cleared, and the NEXT borrower pays: a spurious
//! `EXEC=nil` → in-process `Aborted` on a plain cluster (a wasted retry, or a
//! client-visible `UNAVAILABLE` where no retry loop wraps the transaction), or
//! — on a proxied deployment
//! (Redis Enterprise / Azure Managed Redis) — an `INTERNAL` carrying a
//! `CROSSSLOT` that names a key the failing request never mentioned. That last
//! shape is what was observed in the field: `SessionStore::create_once`'s
//! leaked `le:path:{hex(base_path)}:names` WATCH reported against an unrelated
//! `{object_id}` command.
//!
//! **A single-node store can never reproduce any of this.** One connection
//! means every route is the right route, which is exactly why the pre-existing
//! `watch_leak_test.rs` (b.jov) and `b_k1r_tests.rs` (b.k1r) coverage stayed
//! green through the whole defect. That is the coverage hole this file closes.
//!
//! # What lives here
//!
//! | Test | Needs a cluster? | What it pins |
//! |---|---|---|
//! | [`watch_guard_pinned_unwatch_clears_the_watch_on_every_shard`] | yes | the deterministic regression — *both* teardown sites (the **explicit** `unwatch_pinned()` and the `Drop`-time one) reach the watching node for tags spread across every primary, and an unpinned teardown demonstrably does not |
//! | [`session_duplicate_then_structural_update_commits_without_a_retry`] | yes | the field shape — the two real call sites, back to back, on one pooled connection |
//! | [`pinned_unwatch_clears_the_watch_on_a_centralized_store`] | no | the non-regression the fix could plausibly have broken: a `cluster_hash`-pinned UNWATCH must still work against a non-clustered server |
//! | [`mark_watched_object_keys_share_one_slot_derived_from_the_hash_tag`] | no | the key-schema co-location `mark_watched`'s `debug_assert` encodes |
//! | [`mark_watched_cross_slot_keys_trip_the_debug_assert`] | no | that the `debug_assert` actually fires |
//! | [`unwatch_failure_counts_both_sites_with_retry_only_at_drop`] | no | the asymmetric retry policy → `error!` → metric failure path: one attempt at the explicit site, attempt-plus-retry at `Drop`, a failure-counter tick at each, and one `valkey_errors_total{pool="txn", kind="command_timeout"}` tick per expired attempt |
//! | [`no_bare_unwatch_call_survives_outside_watch_guard`] | no | the source-shape fence: every teardown goes through the pinned funnel, and that funnel's `Options` carries both the slot and the per-attempt deadline |
//!
//! The two timeout couplings — `UNWATCH_ATTEMPT_TIMEOUT × 2 <=
//! UNWATCH_FUNNEL_TIMEOUT`, and `UNWATCH_FUNNEL_TIMEOUT + reserve <=
//! ValkeyClient::quit`'s `UNWATCH_DRAIN_TIMEOUT` — used to be a source-parsing
//! test here. They are now `const _: () = assert!(...)` blocks in
//! `src/storage/watch_guard.rs` (b.ji8 round 3), which no filter can skip and no
//! declaration reshape can silently defeat, so the test was removed rather than
//! kept as a weaker duplicate.
//!
//! The metric's pre-initialisation (`{site="explicit"}` and `{site="drop"}`
//! present at zero on a fresh registry) is pinned next to its peers in
//! `tests/integration/observability_metrics_test.rs`.
//!
//! # Gating
//!
//! The two cluster tests self-gate on `LIVE_EDIT_TEST_CLUSTER_URL` via
//! [`cluster_fixture`] and print a `SKIP:` line when it is unset, following the
//! inline-gate idiom in `client_test.rs` and `cluster_subscriber_test.rs`. So
//! this module compiles and passes everywhere; CI's `test-cluster:` job — which
//! sets `LIVE_EDIT_TEST_CLUSTER_URL=redis-cluster://127.0.0.1:7000` and runs
//! the unfiltered workspace suite under nextest — is what actually executes
//! those two bodies. Neither carries `#[ignore]` and neither sits behind a
//! cargo feature, so there is no CI-side filter to keep in sync and no
//! `.gitlab-ci.yml` change is needed for pickup. Locally, `bash
//! scripts/dev-valkey.sh cluster-up` brings up the same 3-shard cluster
//! (Linux-only — it relies on Docker `--network host`).
//!
//! Skipping is the only soft failure. Once the URL is set nothing degrades
//! quietly: [`cluster_fixture`] asserts `is_clustered()`, and
//! [`watch_guard_pinned_unwatch_clears_the_watch_on_every_shard`] fails loudly
//! if its tags did not actually spread across the primaries it needs.
//!
//! # Wait budgets vs. nextest's hang-killer
//!
//! `.config/nextest.toml` marks a test slow at 40 s and hard-terminates it at
//! 120 s, and a SIGKILLed test prints nothing — so every diagnostic panic below
//! would be lost exactly when it is needed. Each test that can block holds ONE
//! deadline spanning its whole body rather than a per-iteration budget that
//! multiplies:
//!
//! | Test | Waits | Worst case |
//! |---|---|---|
//! | `…clears_the_watch_on_every_shard` | 1 × [`CLUSTER_SCENARIO_BUDGET`] | **45 s** |
//! | `…commits_without_a_retry` | 1 × [`CLUSTER_SCENARIO_BUDGET`] | **45 s** |
//! | `…on_a_centralized_store` | 1 × [`CLUSTER_SCENARIO_BUDGET`] | **45 s** |
//! | `…with_retry_only_at_drop` | 2 × [`DEAD_CONNECTION_BUDGET`] + 1 × [`TRACKER_DRAIN_BUDGET`] | **45 s** |
//!
//! The healthy path returns in milliseconds and pays none of it.

use std::collections::BTreeSet;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fred::interfaces::{
    ClientLike, ClusterInterface, KeysInterface, SetsInterface, TransactionInterface,
};
use fred::types::config::Server;
use fred::types::{ClusterHash, Value as FredValue};
use tracing_subscriber::fmt::MakeWriter;

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{histogram_bucket, render, sample_value};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use live_edit::storage::watch_guard::WatchGuard;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, StructCommand};

// ═════════════════════════════════════════════════════════════════════════════
// Budgets — see the module-level table for the arithmetic
// ═════════════════════════════════════════════════════════════════════════════

/// One deadline covering a whole cluster scenario body (every WATCH, UNWATCH,
/// mutation and EXEC in it). The healthy path is a few hundred round trips at
/// worst — [`watch_guard_pinned_unwatch_clears_the_watch_on_every_shard`] is
/// the big one at three arms × [`TAG_COUNT`] tags — and finishes in well under
/// a second against a local cluster; 45 s leaves roughly two orders of
/// magnitude of headroom over that, and keeps the 120 s hang-killer as a
/// backstop rather than the primary failure surface.
const CLUSTER_SCENARIO_BUDGET: Duration = Duration::from_secs(45);

/// Per-command deadline on the client
/// [`unwatch_failure_counts_both_sites_with_retry_only_at_drop`] builds, for
/// every command that does **not** go through the pinned UNWATCH funnel.
///
/// It deliberately does not size that test's wall clock, and cannot: since
/// b.ji8 round 3 every UNWATCH the funnel issues carries its own
/// `Options::timeout` (`UNWATCH_ATTEMPT_TIMEOUT`, 650 ms,
/// `src/storage/watch_guard.rs`), and fred's per-command option *overrides*
/// `default_command_timeout` outright rather than capping it. So the three
/// expiries that test observes — one at the explicit site, two at `Drop`
/// (attempt plus retry) — cost 650 ms each whatever this constant says, and the
/// test runs ~1.98 s. Raising or lowering this value moves neither number.
///
/// What it still buys is a bound on the test's *other* traffic against the
/// same client: `connect_test_client`'s handshake, the healthy `WATCH` that
/// arms the seam, and the `quit()` that closes the connection. Those inherit
/// the client default, so without this line they would inherit the `TestValkey`
/// fixture's hardcoded 5 000 ms and a wedged local Valkey would park the
/// gating suite there instead of failing promptly. 500 ms is still two orders
/// of magnitude above a healthy local round trip, so the healthy WATCH cannot
/// flake on it.
const NON_FUNNEL_COMMAND_TIMEOUT_MS: u64 = 500;

/// How long one teardown against the deliberately-closed connection may take
/// before the test calls it a hang rather than a failure.
///
/// The funnel bounds a whole teardown at `UNWATCH_FUNNEL_TIMEOUT` (1 500 ms,
/// `src/storage/watch_guard.rs`) and the explicit site's single attempt at
/// `UNWATCH_ATTEMPT_TIMEOUT` (650 ms), so 15 s is an order of magnitude of
/// headroom over the worst of them for CI-runner load. Anything that actually
/// reaches this budget is an unbounded funnel, not a slow runner.
const DEAD_CONNECTION_BUDGET: Duration = Duration::from_secs(15);

/// How long the `Drop`-spawned UNWATCH tracker may take to drain — one more
/// teardown, i.e. an attempt plus its retry under the same 1 500 ms funnel
/// bound, plus task scheduling.
const TRACKER_DRAIN_BUDGET: Duration = Duration::from_secs(15);

/// Hash tags used by the deterministic cluster regression. Twelve is chosen so
/// the set straddles all three primaries of the `scripts/dev-valkey.sh
/// cluster-up` topology several times over rather than by luck — the slots are
/// a pure function of these literals, so the spread is identical on every run.
/// [`watch_guard_pinned_unwatch_clears_the_watch_on_every_shard`] re-derives
/// and asserts the spread instead of trusting this comment.
const TAG_COUNT: usize = 12;

// ═════════════════════════════════════════════════════════════════════════════
// Fixtures + helpers
// ═════════════════════════════════════════════════════════════════════════════

/// The cluster gate. `None` means "not configured — the caller must return",
/// and a `SKIP:` line has already been printed saying why.
///
/// `TestValkey::setup_with_url` panics if the cluster is unreachable, so a
/// configured-but-down cluster fails loudly rather than skipping.
async fn cluster_fixture(valkey_pool_size: u32) -> Option<TestValkey> {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!(
            "SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured); \
             run `bash scripts/dev-valkey.sh cluster-up` or let CI's `test-cluster:` job run \
             this. b.ji8's defect is invisible on a single connection, so there is nothing \
             this test can assert without a real cluster"
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

    let tv = TestValkey::setup_with_url(cluster_url, valkey_pool_size).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment (use \
         `redis-cluster://...`). Every assertion in the cluster tests here is about \
         behaviour that exists only in cluster mode — a per-node connection map and \
         hash-slot routing — so a single-node fixture smuggled in through this gate would \
         pass vacuously and report cluster coverage the suite does not have"
    );
    Some(tv)
}

/// The hash slot `WatchGuard::mark_watched` derives from `key`.
///
/// Deliberately computed the same way the production code does
/// (`ClusterHash::from(&str)` → `Custom(redis_keyslot(key))`) rather than by a
/// hand-rolled CRC16: the claim under test is about the slot the *guard*
/// records, so re-implementing the hash here would let the two drift and still
/// agree.
fn slot_of(key: &str) -> u16 {
    match ClusterHash::from(key) {
        ClusterHash::Custom(slot) => slot,
        other => panic!(
            "ClusterHash::from(&str) must yield a Custom(slot) — `WatchGuard::mark_watched` \
             relies on it to pin teardown; got {other:?}"
        ),
    }
}

/// Every primary in the cluster, from the client's cached `CLUSTER SLOTS` view.
fn primaries(tv: &TestValkey) -> Vec<Server> {
    tv.client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view")
        .unique_primary_nodes()
}

/// The primary that owns `slot`, from the cached routing table.
fn slot_owner(tv: &TestValkey, slot: u16) -> Server {
    tv.client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view")
        .get_server(slot)
        .unwrap_or_else(|| panic!("no primary owns slot {slot} in the cached routing table"))
        .clone()
}

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

/// The smallest object a `create_layer` will accept — this file never inspects
/// layer contents, only which connection the surrounding commands landed on.
fn minimal_object_state() -> ObjectState {
    ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 1,
        ..Default::default()
    }
}

/// Read `live_edit_storage_watch_unwatch_failures_total{site}`.
fn read_unwatch_failures(metrics: &live_edit::observability::metrics::Metrics, site: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_storage_watch_unwatch_failures_total",
        &[("site", site)],
    )
    .unwrap_or(0.0) as u64
}

/// `path` label of a commit on the WATCH slow path, where a structural delta
/// commits under the `lua-watch-structural` and `txn` backends.
const WATCH_PATH: &str = "watch";

/// Read `live_edit_update_layer_retries_total{path,outcome}`.
fn read_update_layer_retries(
    metrics: &live_edit::observability::metrics::Metrics,
    path: &str,
    outcome: &str,
) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_update_layer_retries_total",
        &[("path", path), ("outcome", outcome)],
    )
    .unwrap_or(0.0) as u64
}

/// The funnel's retry log line, verbatim from
/// `src/storage/watch_guard.rs::unwatch_pinned_on`. The retry-once has no
/// counter and no return value of its own — a `Result` looks identical whether
/// the funnel tried once or twice — so this line is the only observable that
/// distinguishes them, which is why
/// [`unwatch_failure_counts_both_sites_with_retry_only_at_drop`] asserts on it
/// despite the "don't assert on log strings" default in the test-writing
/// guide. Deleting the retry, moving it to the explicit site, or making it a
/// loop all change which site emits it and how often.
const UNWATCH_RETRY_LOG_LINE: &str = "WatchGuard: pinned UNWATCH failed; retrying once";

/// The funnel's terminal `error!`, verbatim prefix from the same function. It
/// fires exactly once per given-up teardown, at every site and regardless of
/// retry policy, so it is what pins "the explicit site *did* run a teardown and
/// gave up on it" — without which "no retry line at the explicit site" would
/// also be satisfied by a site that never attempted anything at all.
const UNWATCH_GAVE_UP_LOG_LINE: &str = "WatchGuard: pinned UNWATCH gave up";

/// The funnel's `first_error` field as the default `fmt` layer renders it when
/// no retry happened. `unwatch_pinned_on` populates `first_error` only on the
/// `Err(e, Some(first_err))` arm, which is reachable only after a second
/// attempt — so `None` here is a second, field-level witness (independent of
/// the absent retry line) that the explicit site made exactly one attempt.
const NO_FIRST_ERROR_FIELD: &str = "first_error=None";

/// The funnel's `funnel_elapsed` field (a `bool`, so the `fmt` layer renders it
/// unquoted) in its "an attempt returned an error" state.
///
/// This is the third field-level witness on a give-up event, and it pins
/// something the counter and the retry line cannot: *which* of the two bounds
/// ended the teardown. `true` means `UNWATCH_FUNNEL_TIMEOUT` elapsed with a
/// command still in flight; `false` means an attempt came back with an error,
/// which is what this test's closed connection produces. The distinction is
/// exactly the b.ji8 round-3 regression surface — before the per-attempt
/// `Options::timeout` existed, an attempt inherited the 5 000 ms client default,
/// the 1 500 ms funnel deadline always fired mid-attempt-one, and the `Drop`
/// site's retry was unreachable. That state emits `funnel_elapsed=true` and no
/// retry line, so asserting `false` here fails loudly if the per-attempt bound
/// is ever removed, raised past the funnel, or stops being applied.
const NOT_FUNNEL_ELAPSED_FIELD: &str = "funnel_elapsed=false";

/// Formatted event lines carrying both `message` and `site="…"`.
///
/// The default `fmt` layer writes one line per event with fields rendered as
/// `key=value`, and `site` is a `&str` field, so it renders quoted. Matching on
/// the pair is what lets a single capture buffer — which necessarily holds both
/// teardowns plus fred's own chatter — attribute each line to its site.
fn lines_at_site<'a>(logged: &'a str, message: &str, site: &str) -> Vec<&'a str> {
    let site_field = format!("site=\"{site}\"");
    logged
        .lines()
        .filter(|line| line.contains(message) && line.contains(site_field.as_str()))
        .collect()
}

/// Count of [`lines_at_site`], for the assertions that only care how many.
fn events_at_site(logged: &str, message: &str, site: &str) -> usize {
    lines_at_site(logged, message, site).len()
}

/// `MakeWriter` that captures every byte written to it into a shared buffer.
///
/// Mirrors the per-module helper in `tests/misc/service_token_startup_warns.rs`
/// (itself mirroring `src/grpc/auth_interceptor.rs::tests::CapturingWriter`) —
/// the established project pattern is to duplicate these six lines rather than
/// hoist them into `tests/common`.
#[derive(Clone)]
struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

impl CapturingWriter {
    fn new() -> (Self, Arc<Mutex<Vec<u8>>>) {
        let buf = Arc::new(Mutex::new(Vec::new()));
        (Self(buf.clone()), buf)
    }
}

impl<'a> MakeWriter<'a> for CapturingWriter {
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

/// WATCH `key` on `guard`, record the slot, then tear the WATCH down through
/// the pinned funnel — the exact `watch → mark_watched → unwatch_pinned →
/// mark_unwatched` sequence every production call site runs.
async fn watch_then_pinned_unwatch(guard: &mut WatchGuard, key: &str) {
    guard
        .deref()
        .watch::<&str>(key)
        .await
        .unwrap_or_else(|e| panic!("WATCH {key} must succeed: {e}"));
    guard.mark_watched(key, &[]);
    guard
        .unwatch_pinned()
        .await
        .unwrap_or_else(|e| panic!("pinned UNWATCH for {key} must succeed: {e}"));
    guard.mark_unwatched();
}

/// Queue `SET key` inside a `MULTI` on `guard`'s connection and run `EXEC`,
/// returning the raw reply.
///
/// Decoded as `fred::types::Value`, never `()`: fred's `FromValue for ()`
/// silently accepts `Value::Null`, and a nil EXEC — the WATCH-canceled reply —
/// is precisely the signal this whole file exists to detect.
async fn exec_set(guard: &WatchGuard, key: &str, value: &str) -> FredValue {
    let trx = guard.deref().multi();
    let _: () = trx
        .set(key, value, None, None, false)
        .await
        .unwrap_or_else(|e| panic!("MULTI queue of SET {key} must succeed: {e}"));
    trx.exec(true)
        .await
        .unwrap_or_else(|e| panic!("EXEC for {key} must not error at the fred layer: {e}"))
}

// ═════════════════════════════════════════════════════════════════════════════
// T1 — the deterministic cluster regression
// ═════════════════════════════════════════════════════════════════════════════

/// The test that would have caught b.ji8.
///
/// For each of [`TAG_COUNT`] distinct hash tags — chosen so the slots spread
/// across every primary the cluster reports — on a `pool_size = 1` `txn_pool`
/// (so every borrow provably lands on the same fred `Client`, and therefore on
/// the same per-node connection map), three arms run back to back. Each is the
/// same four-step probe over a different teardown site:
///
/// 1. WATCH the tag's `obj:hdr` key through a `WatchGuard`, recording the slot
///    via `mark_watched`.
/// 2. Tear the WATCH down.
/// 3. Modify that same key from the *data* pool — a different connection, so
///    this is a genuine third-party write and trips any surviving WATCH.
/// 4. Run `MULTI { SET key } EXEC` on the guard and inspect the reply.
///
/// A non-nil EXEC at step 4 is only possible if step 2's UNWATCH actually
/// reached the node holding the WATCH. Pre-fix, the keyless UNWATCH went to
/// `writers.values_mut().next()`, so on a three-primary cluster roughly
/// two-thirds of the tags kept their WATCH, step 3 tripped it, and EXEC came
/// back `Value::Null`.
///
/// | Arm | Step 2 is | Expected reply |
/// |---|---|---|
/// | explicit | `unwatch_pinned()` — the early-return path both production call sites run | non-nil, every tag |
/// | drop | dropping an *armed* guard — `Drop`'s own pinned UNWATCH | non-nil, every tag |
/// | control | a bare, unpinned `Client::unwatch()` — the pre-fix shape | nil for at least one tag |
///
/// The `drop` arm is not redundant with the `explicit` one. `Drop::drop` reads
/// `watched_hash` out through its own `take()` and hands it to the funnel from
/// a spawned task, which is a separate routing decision from
/// `unwatch_pinned()`'s; and it is the terminal one — a teardown that misses
/// there releases the pooled connection dirty with no further chance to fix it.
/// Every *other* `drop(guard)` in this file follows a `mark_unwatched()` and so
/// takes `Drop`'s synchronous no-UNWATCH branch, meaning without this arm the
/// drop-site routing has zero cluster coverage.
///
/// The `control` arm is the detector's positive control, and the reason it can
/// assert anything deterministically is that fred's fallback picks *one* node
/// (`writers.values_mut().next()`), not a random one per call: every tag whose
/// slot is owned by some other primary keeps its WATCH. Without it, an arming
/// step that silently stopped arming (a WATCH that never landed, a third-party
/// write on the wrong key) would leave all three arms passing vacuously.
///
/// The per-shard coverage assertion at the end is the other non-vacuity guard:
/// a topology (or a future edit to the tag literals) that put every tag on one
/// primary would reduce this to a slow re-run of the single-node `watch_leak`
/// test while still reporting cluster coverage.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_guard_pinned_unwatch_clears_the_watch_on_every_shard() {
    let Some(tv) = cluster_fixture(1).await else {
        return;
    };

    let available = primaries(&tv);
    assert!(
        available.len() >= 2,
        "b.ji8 is a *routing* defect: with a single primary ({:?}) there is only one node an \
         UNWATCH can reach, so a broken teardown and a correct one are indistinguishable. \
         `scripts/dev-valkey.sh cluster-up` provisions 3 shards",
        available
    );

    let client = tv.client();
    let keys = tv.key_schema();

    // ONE deadline for the whole scenario (see the module budget table): a
    // per-tag timeout would multiply by TAG_COUNT and blow past the 120 s
    // hang-killer, at which point every panic message below is lost to SIGKILL.
    let (covered, control_cancellations) = tokio::time::timeout(CLUSTER_SCENARIO_BUDGET, async {
        let mut covered: BTreeSet<Server> = BTreeSet::new();
        // How many tags the `control` arm's unpinned teardown failed to clear
        // — i.e. how many EXECs the third-party write canceled. See the
        // post-loop assertion.
        let mut control_cancellations = 0usize;

        for i in 0..TAG_COUNT {
            let key = keys.obj_header(&format!("ji8-{i}"));
            covered.insert(slot_owner(&tv, slot_of(&key)));

            // Seed so the WATCH has an existing key to latch onto.
            let _: () = client
                .pool
                .set(&key, "seed", None, None, false)
                .await
                .unwrap_or_else(|e| panic!("seed SET {key} must succeed: {e}"));

            // ── Arm `explicit`: the early-return teardown both call sites run. ──
            let raw = client.acquire_txn().await;
            let mut guard = WatchGuard::new(raw, client.watch_unwatch_tracker());
            watch_then_pinned_unwatch(&mut guard, &key).await;

            // Third-party write on the multiplexed data pool. If the UNWATCH
            // above went to the wrong node, the WATCH is still armed on the
            // owning node and this is what trips it.
            let _: () = client
                .pool
                .set(&key, "modified-externally", None, None, false)
                .await
                .unwrap_or_else(|e| panic!("external SET {key} must succeed: {e}"));

            let reply = exec_set(&guard, &key, "committed").await;
            assert!(
                !reply.is_null(),
                "EXEC on hash tag `ji8-{i}` (key `{key}`, slot {slot}, owned by {owner}) \
                 returned nil. The preceding `unwatch_pinned()` reported success but the \
                 WATCH was still standing on the owning node when the external SET landed — \
                 i.e. the UNWATCH was routed somewhere else. This is b.ji8: UNWATCH carries \
                 no key, so without an explicit `Options::cluster_hash` fred sends it to an \
                 arbitrary node and the pooled txn connection is released with stale WATCH \
                 state. Got: {reply:?}",
                slot = slot_of(&key),
                owner = slot_owner(&tv, slot_of(&key)),
            );

            // `mark_unwatched()` ran inside `watch_then_pinned_unwatch`, so
            // this drop takes `Drop`'s synchronous no-UNWATCH branch — which
            // is exactly why the next arm exists.
            drop(guard);

            // ── Arm `drop`: the same probe over the `Drop`-time teardown. ──
            let raw = client.acquire_txn().await;
            let mut armed = WatchGuard::new(raw, client.watch_unwatch_tracker());
            armed
                .deref()
                .watch::<&str>(&key)
                .await
                .unwrap_or_else(|e| panic!("WATCH {key} (drop arm) must succeed: {e}"));
            armed.mark_watched(key.as_str(), &[]);
            // Deliberately NO `unwatch_pinned()` / `mark_unwatched()`: the
            // guard must drop still ARMED, because `has_watched = true` is the
            // only way into `Drop`'s spawned, slot-pinned UNWATCH.
            drop(armed);

            // The barrier — no sleep, no polling. `Drop` moves the pool guard
            // INTO the spawned UNWATCH task, so the `ExclusivePool` mutex stays
            // held until that UNWATCH resolves; on a `pool_size = 1` pool this
            // re-acquire therefore cannot return before teardown finished, and
            // it hands back the very same connection.
            let raw = client.acquire_txn().await;
            let guard = WatchGuard::new(raw, client.watch_unwatch_tracker());

            let _: () = client
                .pool
                .set(&key, "modified-after-drop", None, None, false)
                .await
                .unwrap_or_else(|e| panic!("external SET {key} (drop arm) must succeed: {e}"));

            let reply = exec_set(&guard, &key, "committed-after-drop").await;
            assert!(
                !reply.is_null(),
                "EXEC on hash tag `ji8-{i}` (key `{key}`, slot {slot}, owned by {owner}) \
                 returned nil after a `Drop`-time teardown. The guard dropped armed, so \
                 `WatchGuard::drop` spawned its UNWATCH and held the pool mutex across it — \
                 the connection this EXEC ran on is the one that UNWATCH was issued on. A nil \
                 reply means the UNWATCH did not reach the node holding the WATCH, i.e. \
                 `Drop` is passing the recorded slot to the funnel incorrectly (or not at \
                 all). This site is terminal: nothing gets a second chance at the connection \
                 after it. Got: {reply:?}",
                slot = slot_of(&key),
                owner = slot_owner(&tv, slot_of(&key)),
            );
            drop(guard);

            // ── Arm `control`: the pre-fix shape, to prove the probe detects. ──
            let raw = client.acquire_txn().await;
            let guard = WatchGuard::new(raw, client.watch_unwatch_tracker());
            guard
                .deref()
                .watch::<&str>(&key)
                .await
                .unwrap_or_else(|e| panic!("WATCH {key} (control arm) must succeed: {e}"));
            // The bare, keyless teardown b.ji8 replaced — reached through
            // `Deref` on purpose. `no_bare_unwatch_call_survives_outside_watch_guard`
            // below forbids this in `src/`; the scan is deliberately scoped to
            // production code so the pre-fix shape stays available here as a
            // control. `mark_watched` is skipped so the guard drops unarmed and
            // `Drop`'s pinned funnel does not quietly repair what this arm is
            // measuring.
            //
            // `clippy.toml`'s `disallowed-methods` entry mechanizes that same
            // prohibition repo-wide, so the control arm needs an explicit
            // waiver: reproducing the pre-fix call IS the measurement here.
            #[allow(clippy::disallowed_methods)]
            guard
                .deref()
                .unwatch()
                .await
                .unwrap_or_else(|e| panic!("bare UNWATCH for {key} must not error: {e}"));

            let _: () = client
                .pool
                .set(&key, "modified-after-bare-unwatch", None, None, false)
                .await
                .unwrap_or_else(|e| panic!("external SET {key} (control arm) must succeed: {e}"));

            // EXEC clears WATCH state on the node it runs on whether it commits
            // or aborts, so this arm leaves no residue for the next iteration.
            if exec_set(&guard, &key, "committed-after-bare-unwatch")
                .await
                .is_null()
            {
                control_cancellations += 1;
            }
            drop(guard);
        }

        (covered, control_cancellations)
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "the {TAG_COUNT}-tag WATCH/UNWATCH sweep did not finish within \
             {CLUSTER_SCENARIO_BUDGET:?}; the healthy path is a few dozen round trips, so \
             this is a wedge, not a slow runner"
        )
    });

    // Positive control. fred's keyless fallback resolves to ONE node
    // (`writers.values_mut().next()` over a connection map that is not
    // re-shuffled between commands), so every tag owned by a *different*
    // primary must have kept its WATCH and had its EXEC canceled by the
    // third-party SET. With the >= 2-primary precondition asserted above and
    // the spread re-derived below, at least one such tag is guaranteed.
    assert!(
        control_cancellations > 0,
        "not one of the {TAG_COUNT} tags torn down with a BARE, unpinned `Client::unwatch()` \
         came back with a canceled EXEC. That is the pre-fix shape this whole file exists to \
         detect, so the detector itself is not detecting: most likely the WATCH is not being \
         armed (a WATCH that never landed, or a third-party SET on a key nothing is watching), \
         in which case the two arms above are passing vacuously and would keep passing with \
         the b.ji8 fix reverted. The alternative — fred started routing keyless commands to \
         the slot's owner — would make the fix unnecessary and is worth confirming before \
         weakening this check"
    );

    let want = available.len().min(3);
    assert!(
        covered.len() >= want,
        "the {TAG_COUNT} hash tags landed on only {} of the cluster's {} primaries ({covered:?}); \
         at least {want} distinct primaries are needed for this test to distinguish \
         `UNWATCH routed to the watching node` from `UNWATCH routed to whichever node came \
         first`. Either the topology changed or the tag literals were edited — widen \
         TAG_COUNT rather than deleting this check",
        covered.len(),
        available.len(),
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T2 — the field shape
// ═════════════════════════════════════════════════════════════════════════════

/// The two real call sites, back to back, on one pooled connection — the exact
/// sequence the field report described.
///
/// Borrow 1 drives `SessionStore::create` into its duplicate-name arm, which is
/// the early return that UNWATCHes a `le:path:{hex(base_path)}:names` key and,
/// pre-fix, leaked it. The names index is then mutated from a different
/// connection so any surviving WATCH is armed. Borrow 2 runs a
/// `DiffChangeSection` `update_layer` — a structural command, which the store's
/// `lua-watch-structural` backend routes down the WATCH/MULTI/EXEC slow path
/// (the default `lua` backend would plan it on the structural cache, with no
/// WATCH for a leaked one to trip) — against an `{object_id}`-tagged key set
/// that has nothing to do with `{hex(base_path)}`.
///
/// The assertion is that the update commits on its FIRST attempt, read from
/// `live_edit_update_layer_attempts_histogram{path="watch"}`'s `le="1"` bucket
/// and from
/// `live_edit_update_layer_retries_total{path="watch",outcome="committed"}`
/// (the store retries internally, so a bare `Ok` cannot tell a clean commit
/// from a recovered one), plus zero teardown failures at either site. The histogram
/// read is deliberately the *positive* one: the other three are zero-reads,
/// which a store holding no metrics handle would satisfy just as well.
///
/// **This one is probabilistic against an OSS cluster, and deliberately so.**
/// A leaked `{hex(base_path)}` WATCH only bites the update when the two slots
/// happen to co-reside on one primary, so pre-fix this test fails some of the
/// time rather than every time — which is why
/// [`watch_guard_pinned_unwatch_clears_the_watch_on_every_shard`] above exists
/// and is the deterministic guard. Post-fix it is deterministically green,
/// so it carries no flake risk in the gating job. Its value is composition:
/// it is the only place the two production call sites are exercised in cluster
/// mode through their real APIs, and on a *proxied* cluster (the deployment
/// b.ji8 was reported from) the pre-fix residue is not an abort at all but a
/// hard `CROSSSLOT`, which this test would surface as an outright failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_duplicate_then_structural_update_commits_without_a_retry() {
    let Some(tv) = cluster_fixture(1).await else {
        return;
    };

    // Metrics have to be wired on BOTH sides: on the client so every
    // `WatchGuard` picks them up through `metrics_snapshot()`, and on the
    // `LayerStore` so the retry counter is populated at all.
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    let sessions = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    // Pinned to `lua-watch-structural`: the leak under test only bites a
    // WATCH, and the default `lua` backend commits this structural delta on
    // the structural route, which takes none.
    let layers = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::LuaWatchStructural)
        .with_metrics(metrics.clone());

    const SESSION_NAME: &str = "ji8-session";
    const BASE_PATH: &str = "/ji8-stage";
    const LAYER_NAME: &str = "ji8-layer";

    tokio::time::timeout(CLUSTER_SCENARIO_BUDGET, async {
        let session_id = sessions
            .create(SESSION_NAME, BASE_PATH, "ji8-user")
            .await
            .expect("first create_session must succeed");
        let object_id = layers
            .create_layer(
                &session_id,
                LAYER_NAME,
                "ji8-user",
                &minimal_object_state(),
                mock_ext_store(),
            )
            .await
            .expect("create_layer must succeed");

        // ── Borrow 1: the duplicate-name early return. ──
        let err = sessions
            .create(SESSION_NAME, BASE_PATH, "ji8-user")
            .await
            .expect_err("a duplicate (name, base_path) must be rejected");
        assert!(
            matches!(err, ServiceError::AlreadyExists(_)),
            "the duplicate-name arm is the leak site under test; a different error means \
             this test never reached it. Got: {err:?}"
        );

        // Arm whatever WATCH borrow 1 may have left behind: mutate the names
        // index from the multiplexed data pool (a different connection).
        let names_key = tv.key_schema().path_names_index(BASE_PATH);
        let _: () = tv
            .client()
            .pool
            .sadd(
                &names_key,
                "ji8-probe\u{0}00000000-0000-0000-0000-000000000000",
            )
            .await
            .expect("third-party SADD on the names index must succeed");

        // ── Borrow 2: a structural update on an unrelated key family. ──
        let delta = OwnedDelta {
            struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 0,
                section_id: 1,
                section_name: "Ji8Section".to_string(),
                section_type: 1,
                section_order: 0,
            })],
            ..Default::default()
        };
        layers
            .update_layer_relayed(
                &session_id,
                LAYER_NAME,
                object_id,
                Some(delta),
                "ji8-writer",
                0,
            )
            .await
            .expect(
                "the structural (slow-path) update must commit. A CROSSSLOT here naming \
                 `le:path:...:names` — a key this call never mentions — is b.ji8 as reported \
                 from the field: the previous borrower's WATCH was never torn down on the \
                 node holding it",
            );
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "the duplicate-create + structural-update sequence did not finish within \
             {CLUSTER_SCENARIO_BUDGET:?}"
        )
    });

    // Positive read FIRST. Every other metric assertion in this test is a
    // zero-read, and a zero-read is satisfied identically by a `LayerStore`
    // that never received the metrics handle at all — in which case the three
    // checks below would pass while observing nothing. `update_layer` records
    // exactly one observation into this histogram per successful commit
    // (`LayerStore::update_layer`, on the `Ok` arm), and it is the only writer
    // of the family, so the single structural update above must land in the
    // `le="1"` bucket — a real boundary in `UPDATE_LAYER_ATTEMPTS_BUCKETS`, and
    // one that doubles as the "committed on the first attempt" claim the
    // retry-counter assertion makes from the other direction.
    assert_eq!(
        histogram_bucket(
            &render(&metrics),
            "live_edit_update_layer_attempts_histogram",
            "1",
            &[("path", WATCH_PATH)],
        ),
        Some(1.0),
        "the one `update_layer` above must be recorded in the attempts histogram's `le=\"1\"` \
         bucket. `None` means the family is absent from the exposition entirely — the store \
         has no metrics handle, so every zero-read below is vacuous. A count > 1 in that \
         bucket means something else in this test recorded attempts; a count of 0 means the \
         commit took more than one attempt, which is the b.ji8 symptom the retry assertion \
         below describes"
    );
    assert_eq!(
        read_update_layer_retries(&metrics, WATCH_PATH, "committed"),
        0,
        "the structural update must commit on its FIRST attempt. A retry here means its EXEC \
         came back nil, which on an uncontended object can only be the previous borrower's \
         leaked `{{hex(base_path)}}` WATCH being tripped by the third-party SADD — the \
         same-node case of b.ji8"
    );
    for site in ["explicit", "drop"] {
        assert_eq!(
            read_unwatch_failures(&metrics, site),
            0,
            "no `WatchGuard` teardown may fail against a healthy cluster; \
             site=\"{site}\" ticked"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// T3 — single-node coverage (runs in the default `test:` job)
// ═════════════════════════════════════════════════════════════════════════════

/// The non-regression the fix could plausibly have broken.
///
/// Every production teardown now goes out as
/// `client.with_options(Options { cluster_hash: Some(Custom(slot)), .. }).unwatch()`,
/// including on centralized deployments where there is no routing decision to
/// make. If fred rejected or mis-handled a `cluster_hash` on a non-clustered
/// client, WATCH teardown would break for *every* single-node operator — a far
/// worse outcome than the defect being fixed — and no cluster-gated test would
/// see it. So this runs on the ordinary single-node fixture.
///
/// Shape mirrors `watch_leak_test.rs` (b.jov): `pool_size = 1` pins both
/// borrows to one connection, and the assertion is that an unrelated later
/// transaction is not spuriously canceled.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_unwatch_clears_the_watch_on_a_centralized_store() {
    let tv = TestValkey::setup_with_pool_size(1).await;
    let client = tv.client();
    let keys = tv.key_schema();

    let watched = keys.obj_header("ji8-central-watched");
    let unrelated = keys.obj_header("ji8-central-unrelated");

    tokio::time::timeout(CLUSTER_SCENARIO_BUDGET, async {
        let _: () = client
            .pool
            .set(&watched, "seed", None, None, false)
            .await
            .expect("seed SET must succeed");

        let raw = client.acquire_txn().await;
        let mut guard = WatchGuard::new(raw, client.watch_unwatch_tracker());
        watch_then_pinned_unwatch(&mut guard, &watched).await;

        // Trip the WATCH if it survived.
        let _: () = client
            .pool
            .set(&watched, "modified-externally", None, None, false)
            .await
            .expect("external SET must succeed");

        let reply = exec_set(&guard, &unrelated, "committed").await;
        assert!(
            !reply.is_null(),
            "EXEC of an unrelated `MULTI {{ SET }}` returned nil on a CENTRALIZED store — the \
             slot-pinned `unwatch_pinned()` reported success but did not actually clear the \
             WATCH. That would mean fred's `Options::cluster_hash` breaks UNWATCH on a \
             non-clustered client, regressing every single-node deployment. Got: {reply:?}"
        );

        // Belt and suspenders: a nil EXEC discards the queued SET, so the
        // value's presence is independent evidence the transaction committed.
        let stored: Option<String> = client
            .pool
            .get(&unrelated)
            .await
            .expect("GET on the unrelated key must succeed");
        assert_eq!(
            stored.as_deref(),
            Some("committed"),
            "the unrelated key must hold the value the transaction wrote"
        );
    })
    .await
    .unwrap_or_else(|_| {
        panic!("the centralized pinned-UNWATCH round trip did not finish within {CLUSTER_SCENARIO_BUDGET:?}")
    });
}

/// `mark_watched` records ONE slot for the whole watched key set, so the
/// three-key WATCH (header, stream, snapshot fence) in
/// `LayerStore::update_layer_once_watch_attempt` is only tearable-down because
/// its keys co-locate.
///
/// That co-location is a property of `KeySchema` — all three keys carry the
/// `{object_id}` hash tag, and the per-test prefix sits outside the braces —
/// and `mark_watched`'s `debug_assert` turns it into an executable invariant.
/// This test pins the same property from the outside, on keys built by the real
/// `KeySchema` rather than string literals, so a schema change that moved
/// any of them off the tag fails here with an explanation instead of failing in
/// production as an untearable-down WATCH.
///
/// The last assertion is the non-vacuity control: a *different* tag must land
/// on a different slot. Without it, a `slot_of` that returned a constant would
/// satisfy the others.
#[test]
fn mark_watched_object_keys_share_one_slot_derived_from_the_hash_tag() {
    let keys = live_edit::storage::keys::KeySchema::new("test_b_ji8:");
    let object_id: u64 = 123;

    let header = keys.obj_header(&object_id.to_string());
    let stream = keys.layer_stream(object_id);
    let fence = keys.obj_snapshot_fence(&object_id.to_string());

    for watched in [&stream, &fence] {
        assert_eq!(
            slot_of(&header),
            slot_of(watched),
            "`{header}` and `{watched}` must hash to one slot: \
             `update_layer_once_watch_attempt` WATCHes the header, the stream and the snapshot \
             fence in a single command and `mark_watched` records ONE slot to route the \
             teardown back to. If they diverged, the WATCH would not be expressible as one \
             cluster command and no single-slot UNWATCH could undo it"
        );
    }
    assert_eq!(
        slot_of(&header),
        slot_of(&object_id.to_string()),
        "the recorded slot must be the slot of the bare `{{object_id}}` hash tag — that is \
         what makes a keyless UNWATCH pinnable at all. A prefix or suffix leaking inside the \
         braces would move it"
    );
    assert_ne!(
        slot_of(&header),
        slot_of(&keys.obj_header("456")),
        "non-vacuity: two different `{{object_id}}` tags must land on different slots, or the \
         two assertions above would hold for a slot function that ignored its input"
    );
}

/// `mark_watched` must reject a key set that spans slots.
///
/// A cross-slot WATCH is not expressible as one command in cluster mode, and no
/// single-slot UNWATCH could undo it — so a call site that passed one has a bug
/// the guard cannot paper over. `#[cfg(debug_assertions)]` because the check is
/// a `debug_assert!`: in a release build there is deliberately nothing to
/// observe, and an unconditional `#[should_panic]` would fail there.
#[cfg(debug_assertions)]
#[tokio::test]
#[should_panic(expected = "span multiple cluster slots")]
async fn mark_watched_cross_slot_keys_trip_the_debug_assert() {
    // A real guard, not a synthetic one: `WatchGuard` can only be built from a
    // pool-acquired client, and the assertion is about the production
    // constructor's behaviour. Nothing is written to Valkey — the panic fires
    // before any command is issued — and `has_watched` is still false when the
    // guard drops during the unwind, so `Drop` takes its synchronous no-UNWATCH
    // path.
    //
    // If Valkey is unreachable, `TestValkey::setup_*` panics with a *different*
    // message and the `expected =` filter turns that into a loud failure rather
    // than a vacuous pass.
    let tv = TestValkey::setup_with_pool_size(1).await;
    let client = tv.client();
    let keys = tv.key_schema();

    let a = keys.obj_header("ji8-slot-a");
    let b = keys.obj_header("ji8-slot-b");
    assert_ne!(
        slot_of(&a),
        slot_of(&b),
        "fixture precondition: the two keys must genuinely differ in slot, or the \
         debug_assert under test would have nothing to fire on"
    );

    let raw = client.acquire_txn().await;
    let mut guard = WatchGuard::new(raw, client.watch_unwatch_tracker());
    guard.mark_watched(a.as_str(), &[b.as_str()]);
}

/// A doubly-failed teardown is counted at both sites, and the retry lives at
/// exactly one of them.
///
/// # The asymmetry under test
///
/// The funnel's retry policy is deliberately per-site (b.ji8 round 2), and this
/// test is what pins it:
///
/// | Site | Attempts | Why |
/// |---|---|---|
/// | `unwatch_pinned()` (explicit) | **one** | it runs inline on the RPC request path, and it stays armed on `Err`, so `Drop` — which still holds the pool mutex — *is* its retry. A second inline attempt would only add a per-command deadline to a request that is already returning an error. |
/// | `Drop` | **two** (attempt + one retry) | this is the last moment the process holds the connection; there is no later site to defer to. |
///
/// So "retried" is not a property of the funnel, it is a property of the site,
/// and a regression in either direction is a real defect: retrying at the
/// explicit site doubles request-path latency on every failed teardown, and
/// dropping the retry at the `Drop` site loses the last chance to clear a WATCH
/// whose connection may only have been momentarily unresponsive. Both are
/// invisible to the counter (one tick per given-up teardown either way) and to
/// the `Result` (identical whether the funnel tried once or twice), which is why
/// the funnel's own `debug!` line is the observable — see
/// [`UNWATCH_RETRY_LOG_LINE`].
///
/// Only `ErrorKind::Canceled` / `ErrorKind::IO` and fred's routing miss ("Failed to route
/// command.", the router holding no connection) suppress the `Drop` retry ("the WATCH died
/// with the connection"); the seam below produces `Timeout`, which is precisely
/// the *unknown-state* kind the policy retries on.
///
/// # The seam
///
/// The connection itself: `ClientLike::quit()` on the guarded `txn_pool` client
/// (reached through `Deref`) closes exactly that connection. Every subsequent
/// UNWATCH on it fails, which is the only condition under which the funnel's
/// retry, `error!`, and counter are reachable — there is no fault-injection
/// seam on this path.
///
/// A closed fred connection does not reject commands; it lets them expire
/// against a deadline. For the UNWATCHes that deadline is the funnel's own
/// `Options::timeout` (`UNWATCH_ATTEMPT_TIMEOUT`, 650 ms), which overrides the
/// client default — so the three expiries this test observes (one explicit, two
/// at `Drop`) cost ~1.98 s and are unaffected by client configuration. The test
/// still builds its own client rather than using `TestValkey`, to bound
/// everything *outside* the funnel at [`NON_FUNNEL_COMMAND_TIMEOUT_MS`] instead
/// of the fixture's hardcoded 5 000 ms. It needs no `TestValkey` cleanup
/// either, because `WATCH` creates no key and nothing else writes.
///
/// # Why the whole body is under one capture, on a current-thread runtime
///
/// The `Drop`-site events are emitted from a task `WatchGuard::drop` spawns on
/// the shared `TaskTracker`, and nothing in production attaches a dispatcher to
/// it — `tokio::spawn` does not propagate one. Scoping the capture to a future
/// (`WithSubscriber`) therefore reaches the explicit teardown and nothing else,
/// so the retry that matters would be emitted after the scope closed and go
/// unrecorded. `tracing::subscriber::set_default` is thread-local instead, and
/// on the default current-thread runtime *every* task — the test body and the
/// `Drop`-spawned UNWATCH alike — is polled on this one thread, so one guard
/// covers both windows. It is still a scoped default, not a global one, so
/// nothing here can steal or duplicate a sibling test's output. Both sites land
/// in one buffer and are told apart by their `site` field via
/// [`events_at_site`].
///
/// # Why both counter cells
///
/// They carry different meanings: `explicit` is recoverable (the call site
/// leaves the guard armed, so `Drop` gets its own go under the pool mutex)
/// while `drop` is terminal. A regression that wired one label into both sites
/// would still tick a counter, and only checking one cell would miss it.
///
/// # The per-attempt timeout count (b.21r)
///
/// The same three expiries are also counted per ATTEMPT on
/// `live_edit_valkey_errors_total{pool="txn", kind="command_timeout"}` — one at
/// the explicit site, two at `Drop` — unlike the failure counter above, which
/// ticks once per given-up teardown. The whole row is compared, so a timeout
/// counted on another pool fails too.
#[tokio::test]
async fn unwatch_failure_counts_both_sites_with_retry_only_at_drop() {
    let mut cfg = crate::common::test_config();
    cfg.valkey_pool_size = 1;
    cfg.subscriber_pool_size = 1;
    cfg.valkey_command_timeout_ms = NON_FUNNEL_COMMAND_TIMEOUT_MS;
    let Ok(client) = live_edit::storage::test_support::connect_test_client(&cfg).await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    client.install_observability_hooks(metrics.clone()).await;

    // WATCH creates nothing, so this key never has to be cleaned up. The
    // `{...}` hash tag keeps it single-slot if the fixture URL ever points at a
    // cluster.
    let key = format!("test_b_ji8_dead:{{{}}}:obj:hdr", uuid::Uuid::new_v4());

    // `ValkeyClient::metrics_snapshot` (what production call sites pass) is
    // `pub(crate)`, so the handle is supplied directly here. That the two real
    // call sites actually chain `.with_metrics(...)` — without which this
    // counter would never tick in production, silently — is pinned by
    // [`watch_guard_construction_sites_attach_metrics`] below.
    let raw = client.acquire_txn().await;
    let mut guard =
        WatchGuard::new(raw, client.watch_unwatch_tracker()).with_metrics(Some(metrics.clone()));

    guard
        .deref()
        .watch::<&str>(&key)
        .await
        .expect("WATCH must succeed while the connection is still healthy");
    guard.mark_watched(key.as_str(), &[]);

    // Kill the connection out from under the guard.
    tokio::time::timeout(DEAD_CONNECTION_BUDGET, guard.deref().quit())
        .await
        .expect("quitting the guarded txn client must not hang")
        .expect("quit must not error");

    // Capture the funnel's own log output across BOTH teardown windows — the
    // explicit call below and, further down, the `Drop`-spawned one. See the
    // doc comment for why this is a thread-local scoped default on a
    // current-thread runtime rather than a per-future `WithSubscriber`.
    let (writer, captured) = CapturingWriter::new();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(writer)
        // The retry line is `debug!`; the fmt default of INFO would drop it.
        .with_max_level(tracing::Level::DEBUG)
        // ANSI escapes would otherwise land in the buffer and make the
        // substring match depend on whether colour was auto-enabled.
        .with_ansi(false)
        .finish();
    let capture = tracing::subscriber::set_default(subscriber);

    let explicit = tokio::time::timeout(DEAD_CONNECTION_BUDGET, guard.unwatch_pinned())
        .await
        .expect(
            "the explicit teardown must resolve against a closed connection; hanging here \
             would mean the funnel has no bound",
        );
    assert!(
        explicit.is_err(),
        "UNWATCH on a closed connection must surface as an error — this test's whole seam is \
         that failure, and an Ok here makes every assertion below vacuous"
    );
    assert_eq!(
        read_unwatch_failures(&metrics, "explicit"),
        1,
        "a given-up explicit teardown must tick \
         live_edit_storage_watch_unwatch_failures_total{{site=\"explicit\"}} exactly once \
         (one increment per teardown, not per attempt)"
    );

    // Production pattern: do NOT `mark_unwatched()` after a failure, so the
    // guard stays armed and `Drop` makes its own attempt. That attempt, and the
    // retry it is entitled to, also fail here — ticking the terminal `drop`
    // cell.
    drop(guard);

    let tracker = client.watch_unwatch_tracker();
    tracker.close();
    tokio::time::timeout(TRACKER_DRAIN_BUDGET, tracker.wait())
        .await
        .expect("the Drop-spawned UNWATCH must finish within the drain budget");

    assert_eq!(
        read_unwatch_failures(&metrics, "drop"),
        1,
        "leaving the guard armed after a failed explicit teardown is what gives `Drop` its \
         attempt; a zero here means either `Drop` never attempted the UNWATCH (the guard was \
         disarmed despite the failure) or the drop site is reporting under the wrong `site` \
         label"
    );
    // The Drop teardown has drained (tracker wait above), so all three attempts
    // have reported.
    assert_eq!(
        live_edit::storage::test_support::command_timeout_row(&metrics),
        live_edit::storage::test_support::command_timeouts_only_on("txn", 3),
        "each UNWATCH attempt that expired must count once on the txn pool: one at the \
         explicit site plus the attempt and its retry at `Drop`"
    );

    // Both teardowns are now in the buffer. Stop capturing before asserting, so
    // a panic's own output is not swallowed by this subscriber.
    let logged = {
        let buffer = captured
            .lock()
            .expect("the log-capture mutex is never poisoned");
        String::from_utf8_lossy(&buffer).into_owned()
    };
    drop(capture);

    // ── The explicit site: exactly one attempt ───────────────────────────────
    //
    // The counter above proves the teardown was given up on; these two prove it
    // was given up on WITHOUT a second attempt. The txn `command_timeout` row
    // counts attempts but not where they ran: if the retry moved from `Drop` to
    // here — which reads like a robustness improvement — the total would still
    // be 3 and every counter assertion would stay green, while each failing RPC
    // paid a second per-command deadline inline, on a path whose retry already
    // exists one frame later. Only the log lines say which site made which
    // attempt.
    assert_eq!(
        events_at_site(&logged, UNWATCH_RETRY_LOG_LINE, "explicit"),
        0,
        "the explicit teardown must NOT retry: it runs inline on the request path and stays \
         armed on Err, so `Drop` (still holding the pool mutex) is its retry. A \
         `{UNWATCH_RETRY_LOG_LINE}` event at site=\"explicit\" means the single-attempt policy \
         regressed. Captured output: {logged:?}"
    );
    let explicit_gave_up = lines_at_site(&logged, UNWATCH_GAVE_UP_LOG_LINE, "explicit");
    assert_eq!(
        explicit_gave_up.len(),
        1,
        "the explicit teardown must still run and still give up loudly — without this, \
         `zero retry events` above would also be satisfied by a site that never attempted \
         anything at all. Captured output: {logged:?}"
    );
    assert!(
        explicit_gave_up[0].contains(NO_FIRST_ERROR_FIELD),
        "the explicit site's give-up event must carry `{NO_FIRST_ERROR_FIELD}`: the funnel \
         populates `first_error` only on the retried arm, so a `Some(..)` here is a second, \
         field-level witness that a second attempt happened. Event: {event:?}",
        event = explicit_gave_up[0]
    );
    assert!(
        explicit_gave_up[0].contains(NOT_FUNNEL_ELAPSED_FIELD),
        "the explicit site's give-up event must carry `{NOT_FUNNEL_ELAPSED_FIELD}`: its single \
         attempt is bounded at 650 ms inside a 1 500 ms funnel, so the observed failure has to \
         be the attempt's own error, never the funnel deadline. Event: {event:?}",
        event = explicit_gave_up[0]
    );

    // ── The `Drop` site: attempt plus exactly one retry ──────────────────────
    //
    // `Drop` is the last moment the process holds this connection, so the retry
    // belongs here and nowhere else. `== 1` is what makes it retry-ONCE rather
    // than "retries at least once": a loop here runs inside `quit()`'s 2 s
    // drain budget and, on a proxied cluster, can outlive the request that owns
    // the connection.
    assert_eq!(
        events_at_site(&logged, UNWATCH_RETRY_LOG_LINE, "drop"),
        1,
        "expected exactly one `{UNWATCH_RETRY_LOG_LINE}` event at site=\"drop\". Zero means \
         the `Drop` funnel gave up after a single attempt — either the retry was removed, or \
         its message was reworded (update `UNWATCH_RETRY_LOG_LINE`), or the seam stopped \
         producing the unknown-connection-state error kind the retry is gated on \
         (`Canceled`/`IO` and fred's routing miss suppress it by design; this seam yields \
         `Timeout`, which must not). \
         More than one means the single retry became a loop. Captured output: {logged:?}"
    );
    let drop_gave_up = lines_at_site(&logged, UNWATCH_GAVE_UP_LOG_LINE, "drop");
    assert_eq!(
        drop_gave_up.len(),
        1,
        "the `Drop` teardown must give up exactly once — one terminal event per teardown, not \
         one per attempt. Captured output: {logged:?}"
    );
    // The retry-line count above says a second attempt happened; this says the
    // teardown was *allowed to finish* it. Both must hold: a funnel deadline
    // firing mid-retry would still leave one retry line in the buffer while the
    // second attempt's own verdict was discarded, and the give-up event would
    // then report the funnel's synthetic Timeout instead of the connection's.
    assert!(
        drop_gave_up[0].contains(NOT_FUNNEL_ELAPSED_FIELD),
        "the `Drop` site's give-up event must carry `{NOT_FUNNEL_ELAPSED_FIELD}`: attempt plus \
         retry are bounded at 2 × 650 ms = 1 300 ms, which the `const` assertion in \
         `src/storage/watch_guard.rs` keeps inside the 1 500 ms funnel. A `true` here means \
         the pair no longer fits and the funnel deadline is pre-empting the retry — the exact \
         defect the per-attempt `Options::timeout` was added to remove. Event: {event:?}",
        event = drop_gave_up[0]
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T4 — source-shape fence
// ═════════════════════════════════════════════════════════════════════════════

/// The needle. `.unwatch(` matches fred's keyless teardown call and cannot
/// cross-match `unwatch_pinned` / `unwatch_pinned_on` / `watch_unwatch_tracker`
/// (no `.` immediately before `unwatch`, or a `_` immediately after it).
const BARE_UNWATCH_NEEDLE: &str = ".unwatch(";

/// The single file allowed to issue a bare `UNWATCH` — the funnel itself, whose
/// lone `with_options(...)` call carries both the recorded `cluster_hash` (when
/// there is one) and the per-attempt `Options::timeout`.
const UNWATCH_FUNNEL: &str = "src/storage/watch_guard.rs";

/// How many source lines above a funnel `.unwatch(` call the `Options { .. }`
/// it is issued with may be constructed.
///
/// `unwatch_attempt`'s options literal opens nine lines above its call at
/// current formatting; 25 leaves room for rustfmt and for comments inside the
/// literal (which are stripped from the window, but still consume lines here)
/// without reaching into a neighbouring function.
const UNWATCH_OPTIONS_LOOKBACK_LINES: usize = 25;

/// The `Options` field initialisers every UNWATCH the funnel issues must carry,
/// spelled as they appear in the struct literal.
///
/// Both are load-bearing and neither is visible in the funnel's behaviour on a
/// single-node store, so both are pinned as a shape. `cluster_hash` is the b.ji8
/// fix itself — without it fred routes the keyless UNWATCH to
/// `writers.values_mut().next()`, an arbitrary node. `timeout` is the round-3
/// completion of it: without a per-attempt bound the attempt inherits
/// `default_command_timeout` (5 000 ms by default), the 1 500 ms funnel deadline
/// fires during attempt one, and the `Drop` site's retry becomes unreachable.
const REQUIRED_UNWATCH_OPTION_FIELDS: [&str; 2] = ["cluster_hash:", "timeout:"];

/// Collect every `.rs` file under `dir`, recursively, as a repo-root-relative
/// path.
fn rust_sources_under(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {} must succeed: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("directory entry must be readable").path();
        if path.is_dir() {
            rust_sources_under(root, &path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(
                path.strip_prefix(root)
                    .expect("scanned paths are under the manifest dir")
                    .to_path_buf(),
            );
        }
    }
}

/// How many lines after a `WatchGuard::new(` opener the `.with_metrics(` chain
/// may appear. rustfmt wraps the constructor across two or three lines at
/// current indentation; four gives headroom without letting an unrelated later
/// statement satisfy the match.
const WITH_METRICS_LOOKAHEAD_LINES: usize = 4;

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// Every production `WatchGuard` must be handed the process metrics handle.
///
/// Without the `.with_metrics(...)` chain the guard still retries and still
/// logs at `error!`, so nothing about a dropped teardown looks broken in a
/// test — but `live_edit_storage_watch_unwatch_failures_total` stays flat
/// forever and the operator alert built on it can never fire. That is a silent
/// failure of the observability half of b.ji8, invisible to any behavioural
/// test that does not itself force a teardown failure, which is why it is
/// pinned as a shape.
#[test]
fn watch_guard_construction_sites_attach_metrics() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_sources_under(&root, &root.join("src"), &mut files);
    files.sort();

    let mut sites = 0usize;
    let mut violations: Vec<String> = Vec::new();

    for file in &files {
        let source = std::fs::read_to_string(root.join(file))
            .unwrap_or_else(|e| panic!("reading {} must succeed: {e}", file.display()));
        let lines: Vec<&str> = source.lines().collect();
        for (idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("//") || !trimmed.contains("WatchGuard::new(") {
                continue;
            }
            sites += 1;
            let end = (idx + 1 + WITH_METRICS_LOOKAHEAD_LINES).min(lines.len());
            if !lines[idx..end].join("\n").contains(".with_metrics(") {
                violations.push(format!("  {}:{}\n    {trimmed}", file.display(), idx + 1));
            }
        }
    }

    // Non-vacuity: the four production call sites are
    // `LayerStore::update_layer_once_watch_attempt`, `SessionStore::create_once`,
    // `checkpoint::residue::wipe_restore_residue` (b.f4t — the DR restore's
    // pre-write residue wipe, and the only one whose committed effect is an
    // irreversible `UNLINK`), and `LayerStore::write_restored_object` (b.h4m —
    // the DR restore's own materialization, fenced on `obj:hdr` being absent).
    // A scan that found none would pass while asserting nothing.
    assert_eq!(
        sites, 4,
        "expected exactly the four production `WatchGuard::new(` sites \
         (update_layer_once_watch_attempt, create_once, checkpoint::residue::wipe_restore_residue, \
         LayerStore::write_restored_object) but found {sites}. A new WATCH site is fine — update \
         this count and make sure it chains `.with_metrics(...)` — but a count of 0 means the \
         needle stopped matching and this guard is asserting nothing"
    );
    assert!(
        violations.is_empty(),
        "`WatchGuard::new(...)` site(s) not chained with `.with_metrics(...)`:\n\n{}\n\n\
         Pass `self.client.metrics_snapshot()` so a doubly-failed teardown ticks \
         `live_edit_storage_watch_unwatch_failures_total{{site}}`. Without it the guard fails \
         silently as far as monitoring is concerned (b.ji8).",
        violations.join("\n")
    );
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// Every teardown in production code must go through `WatchGuard`'s pinned
/// funnel; no module may reach past `Deref` to fred's keyless
/// `Client::unwatch()`.
///
/// This is a shape guard, in the idiom of the `connect` source guards in
/// `src/storage/client.rs` and the workspace-wide scan in
/// `mapper_routing_test.rs`. It exists because b.ji8's blast radius is
/// *deferred and misattributed*: a bare `.unwatch()` added tomorrow does not
/// fail the request that issues it — it fails some unrelated request that
/// borrows the connection next, with an error naming a key that request never
/// touched. Nothing in a behavioural test suite points at the line responsible.
///
/// The scan walks the directory tree rather than an enumerated file list on
/// purpose: a NEW module under `src/` that reaches for `Client::unwatch()` is
/// exactly the regression worth catching, and an `include_str!` table would
/// have to be remembered and updated to see it.
///
/// The funnel file is exempt from the ban but not from scrutiny: the calls
/// found there are held to a shape of their own — each must be issued through
/// `with_options(...)` and each must have both entries of
/// [`REQUIRED_UNWATCH_OPTION_FIELDS`] initialised on the `Options` it carries.
/// That is what keeps the exemption from being a hole; "the teardown lives in
/// the right file" is worth nothing if the teardown in it is unpinned or
/// unbounded.
#[test]
fn no_bare_unwatch_call_survives_outside_watch_guard() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_sources_under(&root, &root.join("src"), &mut files);
    files.sort();

    // Non-vacuity, part 1: the scan must actually have visited the two files
    // b.ji8 converted. A `src/` layout change that silently narrowed the walk
    // would otherwise leave this test green and blind.
    for expected in [
        "src/storage/layer.rs",
        "src/storage/session.rs",
        UNWATCH_FUNNEL,
    ] {
        assert!(
            files.iter().any(|f| f == Path::new(expected)),
            "the source scan did not visit `{expected}`; it walked {} files. Either the file \
             moved (update this list) or the walk is not reaching where it should",
            files.len()
        );
    }

    let mut violations: Vec<String> = Vec::new();
    // Every funnel call site, as (1-based line, that line, the window above it
    // in which its `Options` must be built).
    let mut funnel_sites: Vec<(usize, String, String)> = Vec::new();

    for file in &files {
        let source = std::fs::read_to_string(root.join(file))
            .unwrap_or_else(|e| panic!("reading {} must succeed: {e}", file.display()));
        let lines: Vec<&str> = source.lines().collect();
        for (idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            // Comments are part of the file and legitimately discuss the
            // pre-fix call (the module docs in the funnel name it repeatedly).
            if trimmed.starts_with("//") {
                continue;
            }
            if !trimmed.contains(BARE_UNWATCH_NEEDLE) {
                continue;
            }
            if file == Path::new(UNWATCH_FUNNEL) {
                // Comment lines are dropped from the window before it is
                // matched. The funnel's prose names `default_command_timeout`
                // and discusses `cluster_hash` at length, so a substring match
                // over raw source would be satisfied by the *explanation* of
                // the fields long after the fields themselves were deleted.
                let start = idx.saturating_sub(UNWATCH_OPTIONS_LOOKBACK_LINES);
                let window = lines[start..=idx]
                    .iter()
                    .map(|l| l.trim())
                    .filter(|l| !l.starts_with("//"))
                    .collect::<Vec<_>>()
                    .join("\n");
                funnel_sites.push((idx + 1, trimmed.to_string(), window));
                continue;
            }
            violations.push(format!("  {}:{}\n    {trimmed}", file.display(), idx + 1));
        }
    }

    // Non-vacuity, part 2: the needle must still match something. If a refactor
    // renamed fred's method, every `.unwatch(` disappears, the scan finds
    // nothing anywhere, and this test would pass while asserting nothing.
    assert!(
        !funnel_sites.is_empty(),
        "expected at least one bare `{BARE_UNWATCH_NEEDLE}` call site inside `{UNWATCH_FUNNEL}` \
         (the pinned, per-attempt-bounded `with_options(...).unwatch()`) but found none. The \
         needle no longer matches the code it is meant to police — update BARE_UNWATCH_NEEDLE \
         rather than deleting this guard"
    );

    // Shape, not count. b.ji8 round 3 collapsed `unwatch_attempt` to a single
    // `with_options(...).unwatch()` — `cluster_hash: hash.cloned()` expresses
    // "pin when we have a slot" without a second `Some`/`None` call site — so a
    // hit count stopped carrying information about whether the surviving call is
    // still the *pinned, bounded* one. Asserting on what the call site is made
    // of instead is both stricter than the old `>= 2` and indifferent to how
    // many call sites the funnel grows: each one must independently be
    // options-bearing and carry both fields, so a second, unpinned `.unwatch()`
    // added inside the funnel file fails here rather than merely satisfying a
    // count.
    for (line_no, call, window) in &funnel_sites {
        assert!(
            call.contains(".with_options("),
            "`{UNWATCH_FUNNEL}:{line_no}` issues an UNWATCH that is not options-bearing:\n    \
             {call}\nEvery UNWATCH in the funnel must go out as \
             `client.with_options(&options).unwatch()`. A bare call here is the pre-fix shape \
             b.ji8 removed, reintroduced in the one file the scan above exempts."
        );
        for field in REQUIRED_UNWATCH_OPTION_FIELDS {
            assert!(
                window.contains(field),
                "the `Options` built for the UNWATCH at `{UNWATCH_FUNNEL}:{line_no}` is missing \
                 a `{field}` initialiser within {UNWATCH_OPTIONS_LOOKBACK_LINES} lines above \
                 the call. Both fields in {REQUIRED_UNWATCH_OPTION_FIELDS:?} are required: \
                 dropping `cluster_hash` restores b.ji8's arbitrary teardown routing, and \
                 dropping `timeout` makes the attempt inherit `default_command_timeout` so the \
                 funnel deadline pre-empts the `Drop` site's retry. Neither loss is visible to \
                 any behavioural test on a single-node store, which is why it is pinned here. \
                 If the literal merely moved further from the call, widen \
                 UNWATCH_OPTIONS_LOOKBACK_LINES. Window scanned:\n{window}"
            );
        }
    }

    assert!(
        violations.is_empty(),
        "found bare `Client::unwatch()` call(s) outside `{UNWATCH_FUNNEL}`:\n\n{}\n\n\
         UNWATCH carries no key, so fred routes it to an arbitrary cluster node and the WATCH \
         survives on the node that actually holds it (b.ji8). The connection then goes back \
         to the shared txn pool carrying state the service believes it cleared, and the NEXT \
         borrower fails — with a spurious EXEC=nil on a plain cluster, or a CROSSSLOT naming \
         a key it never touched on a proxied one. Replace the call with \
         `WatchGuard::unwatch_pinned()` (and only `mark_unwatched()` on Ok, so a failed \
         teardown stays armed for Drop to retry).",
        violations.join("\n")
    );
}
