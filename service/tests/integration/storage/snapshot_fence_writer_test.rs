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

//! Writers honour the snapshot fence (b.y3m Epic 3 Task 3; Spec SDD SR-5.4,
//! SR-8.3; structural-edits SR-8.1): every `update_layer` commit path against
//! a live store with a fence granted by the Task 2 wrapper
//! (`snapshot_fence::acquire_fence`).
//!
//! # The commit paths
//!
//! [`Path`] names each one and how a test reaches it: the update script's fast
//! mode (`fast`); its structural mode on a cached object (`structural`), on a
//! cold start (`structural_full`) and for a reset (`reset`) — the three
//! script paths that honour the fence at step 1d and register no intent
//! (structural-edits SR-8.2); and the WATCH path, under
//! `lua-watch-structural`, under `txn` (field-only and structural) and as the
//! `lua` fallback for an object the structural cache cannot hold — the paths
//! that probe the fence beside their `WATCH`ed header read and register writer
//! intents when they read a snapshot.
//!
//! # How a test knows a writer met the fence
//!
//! A refused commit writes nothing, so the keyspace cannot say whether the
//! writer ran at all. A [`MonitorTap`] can: the store's own command feed shows
//! each attempt's `EVALSHA` (the update script, either mode, or the WATCH
//! path's `snapshot_fence_probe.lua`), and a test waits for the N-th one
//! before it checks that nothing was written. The fence a test holds is made
//! "permanently renewed" by staging its lease's deadline far ahead of the
//! store's clock, so a refusal is certain whenever the attempt runs and the
//! test, not a timer, decides when the fence ends (by releasing the lease —
//! whose `VERIFIED` answer is itself proof that no commit landed under it).
//! Nothing is asserted about how long anything took; the only real waits are
//! the writer's own fence sleeps (about one `poll`, ~1.1 s, each), in the
//! killed-reader tests one lease (~2.2 s), and in the held-attempt test one
//! intent renewal period (~2.2 s). A renewal is a store round trip and can
//! land late on a loaded store, up to the client's command timeout. So the
//! aborted-attempt test, whose count of live intents an intent lapsing that
//! way would change, gives the lingering fence key a stored size whose
//! `intent_ttl(L)` exceeds [`TEST_BUDGET`]; the held-attempt test, whose
//! subject is the intent's renewal, sizes it so a renewal has more than that
//! timeout to land, and waits for the renewal itself, never for the store's
//! clock.
//!
//! A one-attempt retry budget (`with_update_layer_retry_max_attempts(1)`) is
//! the proof that a fence wait is uncounted: a counted wait would exhaust it
//! and fail the call `Aborted`.
//!
//! # What lives here
//!
//! | Section | Pins |
//! |---|---|
//! | 1. Every path, a fence briefly held | a fenced commit waits, writes nothing, commits after release on attempt 1 with no retry; one `snapshot_fence_wait` conflict per refusal on the refused attempt's path; one fence-wait observation under the backend of the path that first met the fence (`lua` for the script paths, `txn` for the WATCH path); intents on the WATCH paths that read a snapshot (refused while the reader is live, registered once it ends, cleared on commit) and none on the script paths; step 1d on an unfenced layer costs two `TYPE`s and no fence read, in both script modes |
//! | 2. WATCH path | registration before the snapshot read on a recently fenced layer, on the `lua-watch-structural` knob and the uncacheable fallback alike; no intent and no register command on a never-fenced layer, for an attempt that reads no snapshot, or on the cached structural route; intents cleared on error; a refused before-`WATCH` registration falls through to `SUPERSEDED`; a fenced txn writer does not hold the txn slot while it waits |
//! | 3. Killed reader | a lease nobody renews or releases ends, and the fast path, the cached structural route and the WATCH path commit without spending a retry |
//! | 4. Fence-wait cap and the per-call record | on every path: `UNAVAILABLE` with the `SNAPSHOT_FENCE_TIMEOUT` retry hint, naming the wait, the cap and the path, nothing written, the timeout counter, the wait histogram, no WARN (the handler boundary's line is the one log line) — at `LIVE_EDIT_RPC_DEADLINE_MS` (the recorded span, the refused attempt included, under the cap; the path alone named); and at the RPC's remaining deadline (what was left of it named too), on the fast path and the txn structural path; one histogram observation and one `update_layer fence wait` event per call that met a fence, on every exit — returned (the cap, a commit: `cancelled` false; `committed` true only for a commit of the call's own — one its re-run finds already landed under its own token included (in section 6) — false for one overtaken, answered `ALREADY_COMMITTED` or refused `ALREADY_EXISTS` after its wait) or dropped mid-call (`cancelled` true, with its `attempt`) — and none for a call that never met one |
//! | 5. Path parity | a wrong-typed fence or intent key is refused with one byte-identical `WRONGTYPE_KEY` on every path, both script modes and the WATCH path; a wrong-typed commit release record (`obj:wrel`, b.yuo) under a live fence is waited out like any fenced commit on the fast, cached structural and both `txn` paths, then refused `WRONGTYPE_KEY` after the release, or answered the cap's `UNAVAILABLE` with its `SNAPSHOT_FENCE_TIMEOUT` retry hint on the fast and `txn` paths when the fence outlasts the cap |
//! | 6. Seams (`test-fault-injection`) | a fence granted between `WATCH` and `EXEC` is an uncounted wait (field-only on `txn`, structural on the knob and on the uncacheable fallback; verdict `fence_live`); a reader's whole epoch between the probe and the `EXEC` — on a lingering key (on the knob), and on a never-fenced layer (field-only on `txn`) — is an uncounted wait too (verdict `fence_changed`), its recorded span covering the attempt the epoch failed; a peer's conflict on a layer inside a gap stays counted (verdict `conflict`); intents cleared on abort and on drop; a held attempt keeps renewing its intent past its first deadline, and a renewal never recreates a removed one; registration before `WATCH` after tallies past the ceiling; a refused before-`WATCH` registration still reports `ALREADY_LANDED`; a fenced call whose re-run finds its own landed commit (pinned token) reports it and records its fence wait `committed` true; on every path the step-1b guards (`ALREADY_LANDED`, `OVERTAKEN`, `SUPERSEDED`) answer ahead of a live fence |
//!
//! Elsewhere: the script's `FENCED` reply and its guard order against a
//! hand-built invocation in both modes, and the `KEYS` / `ARGV` slot
//! constants, in `update_layer_lua_test.rs`, `structural_commit_test.rs` and
//! `structural_guard_parity_test.rs`; the fence-key state across both backends
//! in `tests/proptests.rs`; `FenceWaitClock` / `WriterFenceClass` in
//! `src/storage/snapshot_fence/writer.rs`; the epoch-marker comparison the
//! conflict classifier reads (`FenceProbe::changed_since`) and the probe
//! reply's decoding in `src/storage/snapshot_fence/mod.rs`; the uncounted-wait
//! retry loop in
//! `src/storage/retry.rs`; the wrappers in `snapshot_fence_test.rs`; the
//! writer counters' zero-seeding in `observability_metrics_test.rs`; the
//! single log line a capped call's RPC emits in
//! `grpc/rpc_failure_log_test.rs`; readers completing under writers on every
//! writer path in `grpc/subscribe_fenced_catch_up_test.rs`.
//!
//! Unpinned: step 2b's registration answering `FENCED` (a fence granted) or
//! `NOT_FOUND` (the header removed) between the attempt's probe and that
//! registration. No seam reaches that window — the pre-dispatch hook fires
//! after it, the watch-window hook before the probe. A wrong "carry on" there
//! still commits nothing under a fence or on a removed object, because both
//! the fence key and the header are `WATCH`ed: the grant or the removal fails
//! the attempt's `EXEC`, the conflict re-probe makes the fence — a live
//! lease, or a fence key changed since the attempt's own probe — an uncounted
//! wait (pinned by the `WATCH`-to-`EXEC` and probe-to-`EXEC` tests in section
//! 6), and
//! the next attempt's header guard answers `NotFound`. What stays unpinned is
//! only that step 2b itself answers first, one round trip earlier.
//!
//! # Hang budget
//!
//! Every test body runs under one deadline, [`TEST_BUDGET`] (90 s), warm-up
//! included, so no test can reach nextest's 120 s hang-killer, which prints
//! nothing. Inside it the individual waits keep their own budgets, so a stuck
//! wait still panics with its own message whenever the waits before it fit:
//! a writer's call [`CALL_BUDGET`] (20 s; the slowest healthy call, the killed
//! reader's, takes about one lease plus one `poll`, ~3.3 s), a `MONITOR` drain
//! `MONITOR_DRAIN_BUDGET` (20 s), opening a `MONITOR` tap — whose start waits
//! up to that same 20 s for the node's acknowledgement — the dropped intent's
//! clean-up `DROP_POLL_BUDGET` (10 s). Single round trips are bounded by the
//! client's 5 s command timeout and are not tabulated. `warm_scripts` is three
//! calls (60 s). One `drain_until_nth(n, …)` is one drain budget whatever `n`
//! is.
//! Every wait summed at its full budget:
//!
//! | Test | Waits in series | Sum | Worst case |
//! |---|---|---|---|
//! | every path's fenced commit (8 tests) | warm-up + tap + 4 drains + 1 call | 180 s | 90 s |
//! | step 1d's cost (2 tests); attempts needing no intent | warm-up + 2 or 3 × (tap + 1 call + 1 drain) | 180–240 s | 90 s |
//! | recently fenced registration (2 tests); refused registration → `SUPERSEDED` | warm-up + tap + 1 call + 1 drain | 120 s | 90 s |
//! | fenced txn writer and the txn slot | warm-up + tap + 2 drains + 2 calls | 160 s | 90 s |
//! | error clears the intent; the three killed-reader tests; the eight cap tests; the two remaining-deadline tests | warm-up + 1 call | 80 s | 80 s |
//! | path parity | 24 fixtures, each up to 2 calls | 960 s | 90 s |
//! | wrong-typed release record waits out the fence (4 tests) | warm-up + tap + 4 drains + 1 call | 180 s | 90 s |
//! | wrong-typed release record past the cap (2 tests) | warm-up + 1 call | 80 s | 80 s |
//! | dropped mid-fence-wait call | warm-up + tap + 1 drain (the call runs inside it) + a 50 ms timeout | 100 s | 90 s |
//! | fenced call committed after release | warm-up + tap + 1 call (the drain and the release run beside it) | 100 s | 90 s |
//! | fenced call ended by a staged record (3 tests; the seams' own landed commit) | warm-up + tap + 1 call (the drain and the staging run beside it) | 100 s | 90 s |
//! | calls that never met a fence | warm-up + one poll of a call + 1 call | 80 s | 80 s |
//! | seams: each fence between `WATCH` and `EXEC` | warm-up + tap + 1 call (the drain and the release run beside it) | 100 s | 90 s |
//! | seams: each epoch between the probe and the `EXEC`; peer conflict inside a gap | warm-up + 1 call (the epoch's 200 ms hold, or the competitor, runs inside it) | 80 s | 80 s |
//! | seams: aborted attempt | warm-up + 1 call (the competitor runs inside it) | 80 s | 80 s |
//! | seams: dropped attempt | warm-up + the dispatch wait + the drop poll | 90 s | 90 s |
//! | seams: held attempt renews | warm-up + tap + the dispatch wait + 1 drain + 1 call | 140 s | 90 s |
//! | seams: removed intent not recreated | warm-up + the dispatch wait + tap + 2 drains + 1 call | 160 s | 90 s |
//! | seams: tallies; already landed | warm-up + tap + 1 call + 1 drain | 120 s | 90 s |
//! | seams: step-1b guards ahead of a live fence | 18 fixtures, each up to 2 calls | 720 s | 90 s |
//!
//! The healthy path takes a few seconds.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface, StreamsInterface};
use fred::types::scripts::Script;

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::ObjectState;
use live_edit::error::{refusal_reason, ServiceError, CAPACITY_RETRY_DELAY};
use live_edit::observability::metrics::test_util::{render, sample_sum, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::lua::{UPDATE_SCRIPT_KEY_FENCE, UPDATE_SCRIPT_KEY_HEADER};
use live_edit::storage::session::SessionStore;
use live_edit::storage::snapshot_fence::{
    self as fence, ms_to_duration, AcquireVerdict, FenceLease, LayerSize, ReleaseVerdict,
};
use live_edit::storage::structural_cache::StructuralCache;
use live_edit::storage::test_support::{connect_test_client, TestValkey};
use oodf_ffi::OwnedDelta;

use serde_json::Value;

use crate::common::owned_delta::{
    delta_to_ompf_bytes, delta_with_diff_change_section, delta_with_set_field_inline,
};
use crate::common::refusal_detail::assert_refusal_error;
use crate::grpc::log_capture::{capture_debug, Captured};
use crate::storage::snapshot_read_atomic_test::{create_cmd, structural, unmark, ROOT};
use crate::storage::snapshot_read_support::{
    on_current_thread, test_valkey_addr, MonitorTap, MonitoredCommand,
};

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

const LAYER: &str = "fence-writer";

const CLIENT_ID: &str = "fence-writer-client";

/// The size the test's reader brings to its grant: 4× the shipped ceiling,
/// the report's checkpoint reference layer.
const SIZE: LayerSize = LayerSize {
    items: 150_216,
    bytes: 1_799_752,
};

/// The message of `update_layer`'s per-call fence-wait DEBUG event
/// (`storage::layer`'s `FenceWaitRecord`), spelled out: the validation tooling
/// greps for it, so a rename must fail here.
const FENCE_WAIT_EVENT_MSG: &str = "update_layer fence wait";

/// How far ahead of the store's clock a held lease's deadline is staged: past
/// any wait a test makes, so the fence is live until the test releases it.
const FAR_MS: i64 = 600_000;

/// Bound on one `update_layer` call. A healthy fenced call takes about one
/// `poll` per refusal; see the module's hang budget.
const CALL_BUDGET: Duration = Duration::from_secs(20);

/// The one deadline over a whole test — warm-up, every call, drain and poll —
/// so no test can reach nextest's 120 s hang-killer, which prints nothing.
/// The inner budgets above still fire first wherever their waits alone fit
/// under it, with their own diagnostic; this one is the backstop that names
/// the test when they would sum past it. See the module's hang budget.
const TEST_BUDGET: Duration = Duration::from_secs(90);

/// The structural cache's budget in the fixture's shared cache: far above
/// anything these one-layer tests hold.
const CACHE_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

/// Run a test's body under [`TEST_BUDGET`].
///
/// A plain `fn` that boxes the body at once, not an `async fn`: the body's
/// future (a fixture plus whole commit paths) then lives on the heap, not in
/// the test thread's stack frame, which it can overflow in an unoptimised
/// build at std's default 2 MiB (`.cargo/config.toml` raises debug test
/// threads to 8 MiB; the box keeps the body within the default).
fn within_test_budget<'a>(
    body: impl std::future::Future<Output = ()> + 'a,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>> {
    until_test_deadline(tokio::time::Instant::now() + TEST_BUDGET, body)
}

/// Run `body` until `deadline`, the end of a test's [`TEST_BUDGET`] — for a
/// test whose budget spans more than one runtime — boxed as
/// [`within_test_budget`] does.
fn until_test_deadline<'a, T: 'a>(
    deadline: tokio::time::Instant,
    body: impl std::future::Future<Output = T> + 'a,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = T> + 'a>> {
    let body = Box::pin(body);
    Box::pin(async move {
        tokio::time::timeout_at(deadline, body)
            .await
            .unwrap_or_else(|_| {
                panic!("the test did not finish within TEST_BUDGET ({TEST_BUDGET:?})")
            })
    })
}

/// The store clock, as the fence scripts read it: `TIME` in whole ms.
static CLOCK_SCRIPT: std::sync::LazyLock<Script> = std::sync::LazyLock::new(|| {
    Script::from_lua(
        "local t = redis.call('TIME') \
         return tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)",
    )
});

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// The three delta shapes, which take different commit paths per backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// One inline `SetField`: the fast path under `lua` and
    /// `lua-watch-structural`, the WATCH path's no-snapshot branch under
    /// `txn`.
    FieldOnly,
    /// One `CreateSection` under the root: the structural route under `lua`,
    /// the WATCH path with a snapshot read under the other two backends (and
    /// under `lua` for an object the structural cache cannot hold).
    Structural,
    /// A reset (`is_diff`, `base_version = 0`) carrying one
    /// `DiffChangeSection`: the structural route's `reset` under `lua`.
    Reset,
}

/// The section a structural or reset delta writes, distinct per call so each
/// is a real change.
fn delta(shape: Shape, n: u64) -> OwnedDelta {
    match shape {
        Shape::FieldOnly => delta_with_set_field_inline(1, "color", vec![n as u8], 0),
        Shape::Structural => structural(vec![create_cmd(ROOT, 100 + n, "child")]),
        Shape::Reset => delta_with_diff_change_section(0, 100 + n, "child", 0, 0),
    }
}

/// The writer's raw OMPF, relayed verbatim to subscribers: `update_layer`
/// refuses a delta without it.
fn relay(delta: &OwnedDelta) -> Option<Bytes> {
    Some(Bytes::from(delta_to_ompf_bytes(delta, &[])))
}

/// Every `update_layer` commit path, and how a test reaches it (see the
/// module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Path {
    /// `lua`, field-only: the update script's fast mode.
    Fast,
    /// `lua`, structural, on an object the store's structural cache holds:
    /// the cached structural route, `path="structural"`.
    Structural,
    /// `lua`, structural, on a store whose cache does not hold the object
    /// yet: the first attempt's read is a cold start, `path="structural_full"`
    /// (a re-run after a fence wait is a hit, `structural`).
    StructuralFull,
    /// `lua`, a reset delta: the structural route's `reset`.
    Reset,
    /// `lua-watch-structural`, structural: the WATCH path.
    WatchKnob,
    /// `txn`, field-only: the WATCH path's no-snapshot branch.
    TxnFieldOnly,
    /// `txn`, structural.
    TxnStructural,
    /// `lua`, structural, on an object the structural cache cannot hold (no
    /// `chs_complete` mark): the WATCH fallback.
    UncacheableFallback,
}

impl Path {
    const ALL: [Path; 8] = [
        Path::Fast,
        Path::Structural,
        Path::StructuralFull,
        Path::Reset,
        Path::WatchKnob,
        Path::TxnFieldOnly,
        Path::TxnStructural,
        Path::UncacheableFallback,
    ];

    fn backend(self) -> UpdateLayerBackend {
        match self {
            Path::WatchKnob => UpdateLayerBackend::LuaWatchStructural,
            Path::TxnFieldOnly | Path::TxnStructural => UpdateLayerBackend::Txn,
            _ => UpdateLayerBackend::Lua,
        }
    }

    fn shape(self) -> Shape {
        match self {
            Path::Fast | Path::TxnFieldOnly => Shape::FieldOnly,
            Path::Reset => Shape::Reset,
            _ => Shape::Structural,
        }
    }

    /// Whether the path is the WATCH path (probe beside the header read)
    /// rather than the update script.
    fn on_watch(self) -> bool {
        matches!(
            self,
            Path::WatchKnob | Path::TxnFieldOnly | Path::TxnStructural | Path::UncacheableFallback
        )
    }

    /// Whether the path's attempts read a snapshot on the WATCH path, the
    /// only attempts that register writer intents (SR-5.4, structural-edits
    /// SR-8.2).
    fn registers_intents(self) -> bool {
        matches!(
            self,
            Path::WatchKnob | Path::TxnStructural | Path::UncacheableFallback
        )
    }

    /// The `backend` label the call's fence wait is observed under.
    fn fence_backend(self) -> &'static str {
        if self.on_watch() {
            "txn"
        } else {
            "lua"
        }
    }

    /// The `path` label of the attempt that first meets the fence.
    fn first_label(self) -> &'static str {
        match self {
            Path::Fast => "fast",
            Path::Structural => "structural",
            Path::StructuralFull => "structural_full",
            Path::Reset => "reset",
            _ => "watch",
        }
    }

    /// Whether the store under test starts with a structural cache that does
    /// not hold the object (a cold start, or the uncacheable verdict).
    fn fresh_cache(self) -> bool {
        matches!(self, Path::StructuralFull | Path::UncacheableFallback)
    }
}

/// One layer on a fresh store, with the client its `LayerStore` and the fence
/// wrappers run on (the fixture's own, or one with lowered atomic-read
/// limits), a metrics registry every store built here reports to, and the
/// structural cache every store built here shares (so a warm-up's cold start
/// caches the object for the store under test).
struct Wx {
    tv: TestValkey,
    client: ValkeyClient,
    metrics: Arc<Metrics>,
    cache: StructuralCache,
    session_id: String,
    oid: u64,
}

/// What a refused commit must leave untouched: the full header (`seq`,
/// `last_server_ts`, everything), the stream length, the commit-token record
/// and the writer-intent key.
#[derive(Debug, PartialEq, Eq)]
struct ObjectImage {
    header: BTreeMap<String, String>,
    stream_len: u64,
    commit_tokens: BTreeMap<String, String>,
    intents: BTreeMap<String, String>,
}

impl Wx {
    async fn new() -> Self {
        let tv = TestValkey::setup().await;
        let client = tv.client().clone();
        Self::create(tv, client).await
    }

    /// A store whose atomic-read ceilings are `items` / `bytes`.
    #[cfg(feature = "test-fault-injection")]
    async fn with_limits(items: u64, bytes: u64) -> Self {
        let tv = TestValkey::setup().await;
        let client = crate::storage::snapshot_read_support::client_with_limits(items, bytes).await;
        Self::create(tv, client).await
    }

    /// A store whose pools — the exclusive txn pool included — hold
    /// `pool_size` connections each.
    async fn with_pool_size(pool_size: u32) -> Self {
        let tv = TestValkey::setup_with_pool_size(pool_size).await;
        let client = tv.client().clone();
        Self::create(tv, client).await
    }

    /// A second layer `name` in this fixture's session; its object id.
    async fn create_other_layer(&self, name: &str) -> u64 {
        self.bare_store()
            .create_layer(
                &self.session_id,
                name,
                "fence-user",
                &ObjectState {
                    sequence_num: 1,
                    first_version: 1,
                    source_format: "usda".to_string(),
                    next_set_order: 1,
                    ..Default::default()
                },
                Arc::new(MockExternalValueStore::new()),
            )
            .await
            .expect("create_layer must succeed")
    }

    async fn create(tv: TestValkey, client: ValkeyClient) -> Self {
        let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .create(LAYER, &format!("/{LAYER}"), "fence-user")
            .await
            .expect("session creation must succeed");
        let state = ObjectState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_string(),
            next_set_order: 1,
            ..Default::default()
        };
        let oid = LayerStore::new(client.clone(), tv.key_schema().clone(), 4096)
            .create_layer(
                &session_id,
                LAYER,
                "fence-user",
                &state,
                Arc::new(MockExternalValueStore::new()),
            )
            .await
            .expect("create_layer must succeed");
        Wx {
            tv,
            client,
            metrics: Arc::new(crate::common::metrics_scrape_only("fence-writer")),
            cache: StructuralCache::new(CACHE_BUDGET_BYTES),
            session_id,
            oid,
        }
    }

    /// A store on this layer's client, metrics and shared structural cache,
    /// with a retry budget of `max_attempts`.
    fn store(&self, backend: UpdateLayerBackend, max_attempts: u32) -> LayerStore {
        LayerStore::new(self.client.clone(), self.tv.key_schema().clone(), 4096)
            .with_update_layer_backend(backend)
            .with_update_layer_retry_max_attempts(max_attempts)
            .with_structural_cache(self.cache.clone())
            .with_metrics(Arc::clone(&self.metrics))
    }

    /// The store `path` is reached on: [`Self::store`], or — for a path whose
    /// first read must be a cold start — one with a structural cache of its
    /// own, which does not hold the object.
    fn store_for(&self, path: Path, max_attempts: u32) -> LayerStore {
        let store = self.store(path.backend(), max_attempts);
        if path.fresh_cache() {
            store.with_structural_cache(StructuralCache::new(CACHE_BUDGET_BYTES))
        } else {
            store
        }
    }

    /// Put the object in the state `path` needs before its call: cached in
    /// the shared structural cache (`Structural`, through a structural commit
    /// of the bare store), or unmarked so the structural cache cannot hold it
    /// (`UncacheableFallback`). The other paths need nothing. Commits made
    /// here are the bare store's: they show in no metric the tests read.
    async fn prepare(&self, path: Path) {
        match path {
            Path::Structural => {
                self.commit(&self.bare_store(), delta(Shape::Structural, 240))
                    .await
                    .expect("the caching structural commit");
            }
            Path::UncacheableFallback => unmark(&self.tv, self.oid).await,
            _ => {}
        }
    }

    /// A Lua-backend store reporting to no registry, for commits that are not
    /// the one under test (warm-ups, competitors): they must not show in the
    /// metrics the tests read. It shares the fixture's structural cache.
    fn bare_store(&self) -> LayerStore {
        LayerStore::new(self.client.clone(), self.tv.key_schema().clone(), 4096)
            .with_update_layer_backend(UpdateLayerBackend::Lua)
            .with_structural_cache(self.cache.clone())
    }

    fn oid_str(&self) -> String {
        self.oid.to_string()
    }

    fn header_key(&self) -> String {
        self.tv.key_schema().obj_header(&self.oid_str())
    }

    fn stream_key(&self) -> String {
        self.tv.key_schema().layer_stream(self.oid)
    }

    fn fence_key(&self) -> String {
        self.tv.key_schema().obj_snapshot_fence(&self.oid_str())
    }

    fn intent_key(&self) -> String {
        self.tv.key_schema().obj_snapshot_intent(&self.oid_str())
    }

    fn wtok_key(&self) -> String {
        self.tv.key_schema().obj_write_token(&self.oid_str())
    }

    /// `update_layer` on `store`, bounded by [`CALL_BUDGET`].
    async fn commit(
        &self,
        store: &LayerStore,
        delta: OwnedDelta,
    ) -> Result<(u64, u64), ServiceError> {
        let relay = relay(&delta);
        tokio::time::timeout(
            CALL_BUDGET,
            store.update_layer(
                &self.session_id,
                LAYER,
                self.oid,
                Some(delta),
                relay,
                CLIENT_ID,
                1_000,
            ),
        )
        .await
        .expect("update_layer did not settle within the call budget")
    }

    /// `update_layer` on `store` as a background task.
    fn spawn_commit(
        &self,
        store: &LayerStore,
        delta: OwnedDelta,
    ) -> tokio::task::JoinHandle<Result<(u64, u64), ServiceError>> {
        let store = store.clone();
        let session_id = self.session_id.clone();
        let oid = self.oid;
        tokio::spawn(async move {
            let relay = relay(&delta);
            store
                .update_layer(
                    &session_id,
                    LAYER,
                    oid,
                    Some(delta),
                    relay,
                    CLIENT_ID,
                    1_000,
                )
                .await
        })
    }

    /// Load every script the tests watch for, so no attempt's `EVALSHA` shows
    /// in a `MONITOR` record twice (a `NOSCRIPT` miss and its retry): the
    /// update script (both modes are one script), the structural cache's
    /// cold-start scripts, a probe, and an intent registration, renewed once
    /// and cleared. (A test that waits for the first renewal `EVALSHA` after
    /// some event would otherwise read the keyspace after a `NOSCRIPT` miss,
    /// before the script actually ran.) Its structural commit caches the
    /// object in the shared structural cache.
    async fn warm_scripts(&self) {
        let store = self.bare_store();
        self.commit(&store, delta(Shape::FieldOnly, 250))
            .await
            .expect("warm-up fast-path commit");
        self.commit(&store, delta(Shape::Structural, 250))
            .await
            .expect("warm-up structural commit");
        self.commit(&store, delta(Shape::Reset, 251))
            .await
            .expect("warm-up reset commit");
        fence::probe_fence(&self.client, self.tv.key_schema(), self.oid)
            .await
            .expect("warm-up probe");
        match fence::register_intent(&self.client, self.tv.key_schema(), self.oid, SIZE)
            .await
            .expect("warm-up registration")
        {
            fence::IntentVerdict::Registered(intent) => {
                assert_eq!(
                    intent.renew().await.expect("warm-up renewal"),
                    fence::IntentRenewVerdict::Renewed
                );
                intent.clear().await.expect("warm-up intent clear")
            }
            other => panic!("warm-up: the intent must register; got {other:?}"),
        }
    }

    async fn now(&self) -> i64 {
        CLOCK_SCRIPT
            .evalsha_with_reload::<i64, _, _>(
                self.tv.client().pool.next(),
                vec![self.fence_key()],
                Vec::<String>::new(),
            )
            .await
            .expect("TIME must answer")
    }

    async fn fields(&self, key: &str) -> BTreeMap<String, String> {
        let raw: std::collections::HashMap<String, String> = self
            .tv
            .client()
            .pool
            .hgetall(key)
            .await
            .expect("HGETALL must succeed");
        raw.into_iter().collect()
    }

    async fn seq(&self) -> u64 {
        self.fields(&self.header_key())
            .await
            .get("seq")
            .expect("the header has a seq")
            .parse()
            .expect("seq is numeric")
    }

    async fn stream_len(&self) -> u64 {
        self.tv
            .client()
            .pool
            .xlen(self.stream_key())
            .await
            .expect("XLEN must succeed")
    }

    async fn image(&self) -> ObjectImage {
        ObjectImage {
            header: self.fields(&self.header_key()).await,
            stream_len: self.stream_len().await,
            commit_tokens: self.fields(&self.wtok_key()).await,
            intents: self.fields(&self.intent_key()).await,
        }
    }

    /// The writer-intent fields (`w:<token>` → deadline).
    async fn intents(&self) -> BTreeMap<String, String> {
        self.fields(&self.intent_key())
            .await
            .into_iter()
            .filter(|(k, _)| k.starts_with("w:"))
            .collect()
    }

    /// Grant a fence through the Task 2 wrapper.
    async fn grant(&self) -> Box<FenceLease> {
        self.grant_at(SIZE).await
    }

    /// Grant a fence to a reader that brings `size`.
    async fn grant_at(&self, size: LayerSize) -> Box<FenceLease> {
        match fence::acquire_fence(&self.client, self.tv.key_schema(), self.oid, size)
            .await
            .expect("acquire_fence must answer")
        {
            AcquireVerdict::Granted(lease) => lease,
            other => panic!("the fence must be granted; got {other:?}"),
        }
    }

    /// Grant a fence and stage its lease's deadline [`FAR_MS`] ahead of the
    /// store's clock — a reader that keeps renewing — so it is live until the
    /// test releases it.
    async fn hold_fence(&self) -> Box<FenceLease> {
        let lease = self.grant().await;
        let readers: Vec<String> = self
            .fields(&self.fence_key())
            .await
            .into_keys()
            .filter(|k| k.starts_with("r:"))
            .collect();
        assert_eq!(readers.len(), 1, "one reader holds the fence: {readers:?}");
        let far = (self.now().await + FAR_MS).to_string();
        let pool = &self.tv.client().pool;
        let _: () = pool
            .hset(self.fence_key(), (readers[0].clone(), far))
            .await
            .expect("staging the lease deadline must succeed");
        let _: i64 = pool
            .pexpire(self.fence_key(), FAR_MS, None)
            .await
            .expect("staging the key expiry must succeed");
        lease
    }

    /// Grant a fence and release it at once: the key lingers, not live, with
    /// the stored size — a layer "fenced within its last `key_linger(L)`".
    async fn fence_and_release(&self) {
        self.fence_and_release_at(SIZE).await;
    }

    /// [`Self::fence_and_release`] with `size` as the stored size, which
    /// sizes the `intent_ttl(L)` of every intent registered on the layer
    /// afterwards.
    async fn fence_and_release_at(&self, size: LayerSize) {
        let lease = self.grant_at(size).await;
        assert_eq!(
            lease
                .release(Some(size))
                .await
                .expect("release must answer"),
            ReleaseVerdict::Verified
        );
    }

    /// The `size_e` / `size_b` the fence key stores.
    #[cfg(feature = "test-fault-injection")]
    async fn stored_size(&self) -> LayerSize {
        let fields = self.fields(&self.fence_key()).await;
        let count = |name: &str| -> u64 {
            fields
                .get(name)
                .unwrap_or_else(|| panic!("the fence key stores {name}: {fields:?}"))
                .parse()
                .expect("a stored size is an integer")
        };
        LayerSize {
            items: count("size_e"),
            bytes: count("size_b"),
        }
    }

    /// `intent_ttl(L)` at `size`, from this store's coefficients.
    #[cfg(feature = "test-fault-injection")]
    fn intent_ttl(&self, size: LayerSize) -> Duration {
        ms_to_duration(fence::coefficients(&self.client).timing(size).intent_ttl_ms)
    }

    async fn release(&self, lease: Box<FenceLease>) {
        assert_eq!(
            lease
                .release(Some(SIZE))
                .await
                .expect("release must answer"),
            ReleaseVerdict::Verified,
            "the release must verify the cut: no commit may have landed under the fence"
        );
    }

    /// `poll`, the step a refused writer re-checks at (`min(retry_after,
    /// poll)` + up to `rtt99`), from this store's coefficients.
    fn poll(&self) -> Duration {
        ms_to_duration(fence::coefficients(&self.client).poll_ms())
    }

    fn metric(&self, name: &str, labels: &[(&str, &str)]) -> f64 {
        sample_value(&render(&self.metrics), name, labels).unwrap_or(0.0)
    }

    /// The sum of every series of `name` matching `labels` (a `sum` over the
    /// labels left out).
    fn metric_sum(&self, name: &str, labels: &[(&str, &str)]) -> f64 {
        sample_sum(&render(&self.metrics), name, labels).unwrap_or(0.0)
    }

    /// `(count, sum seconds)` of `live_edit_update_layer_fence_wait_seconds`.
    fn fence_waits(&self, backend: &str) -> (u64, f64) {
        (
            self.metric(
                "live_edit_update_layer_fence_wait_seconds_count",
                &[("backend", backend)],
            ) as u64,
            self.metric(
                "live_edit_update_layer_fence_wait_seconds_sum",
                &[("backend", backend)],
            ),
        )
    }

    fn intent_registrations(&self, result: &str) -> u64 {
        self.metric(
            "live_edit_update_layer_fence_intent_total",
            &[("result", result)],
        ) as u64
    }

    fn fence_timeouts(&self) -> u64 {
        self.metric("live_edit_update_layer_fence_timeout_total", &[]) as u64
    }

    /// `live_edit_update_layer_conflicts_total{kind="snapshot_fence_wait"}` on
    /// `path`, or over every path when `path` is `None`.
    fn fence_conflicts(&self, path: Option<&str>) -> u64 {
        let mut labels = vec![("kind", "snapshot_fence_wait")];
        if let Some(path) = path {
            labels.push(("path", path));
        }
        self.metric_sum("live_edit_update_layer_conflicts_total", &labels) as u64
    }

    /// Assert the call spent exactly one attempt and no retry: a fence wait
    /// is not an attempt. Summed over every `path` series, so the assertion
    /// holds whichever path the committing attempt took.
    fn assert_one_attempt(&self, what: &str) {
        assert_eq!(
            (
                self.metric_sum("live_edit_update_layer_attempts_histogram_count", &[]),
                self.metric_sum("live_edit_update_layer_attempts_histogram_sum", &[]),
            ),
            (1.0, 1.0),
            "{what}: one commit, on attempt 1 — fence waits are not attempts"
        );
        assert_eq!(
            self.metric_sum("live_edit_update_layer_retries_total", &[]),
            0.0,
            "{what}: fence waits never tick the retry counter"
        );
    }

    /// Plant a string on this object's commit release record
    /// (`le:{oid}:obj:wrel`, b.yuo); return the key.
    async fn plant_string_wrel(&self) -> String {
        let wrel_key = self.tv.key_schema().obj_write_releases(&self.oid_str());
        let _: () = self
            .tv
            .client()
            .pool
            .set(&wrel_key, "not-a-hash", None, None, false)
            .await
            .expect("planting a string must succeed");
        wrel_key
    }

    /// Wait for a fenced writer on `path` to run its first attempt and then
    /// its re-run after the uncounted wait, and require each to have written
    /// nothing: the object still matches `before`. An update-script attempt
    /// is over once its script ran. A WATCH-path attempt is over at the next
    /// `UNWATCH` (it gave up) or `EXEC` (it committed) its WATCHing connection
    /// — the one that ran the probe — sends after the probe.
    async fn await_two_fenced_attempts(
        &self,
        tap: &mut MonitorTap,
        path: Path,
        before: &ObjectImage,
        what: &str,
    ) {
        for run in ["first", "re-run"] {
            if path.on_watch() {
                let probed = tap
                    .drain_until_nth(1, "the WATCH path's fence probe", self.is_probe_eval())
                    .await;
                let conn = probed.last().expect("the probe line").client.clone();
                tap.drain_until_nth(1, "the attempt's UNWATCH or EXEC", |c| {
                    c.client == conn && matches!(c.name().as_str(), "UNWATCH" | "EXEC")
                })
                .await;
            } else {
                tap.drain_until_nth(
                    1,
                    "the update script's EVALSHA",
                    self.is_update_script_eval(),
                )
                .await;
            }
            assert_eq!(
                &self.image().await,
                before,
                "{what}: the {run} attempt ran under the live fence and must have written \
                 nothing — no header bump, no stream entry, no commit-token record, no intent"
            );
        }
    }

    // ── `MONITOR` record ───────────────────────────────────────────────────

    /// Everything `tap` recorded up to a marker command issued now, after the
    /// operation under test returned: the record is then complete without a
    /// sleep.
    async fn drain_to_marker(&self, tap: &mut MonitorTap) -> Vec<MonitoredCommand> {
        let marker = crate::storage::snapshot_read_support::monitor_fence(self.tv.prefix());
        let _: i64 = self.tv.client().pool.exists(&marker).await.unwrap();
        tap.drain_until(&marker).await
    }

    /// An update-script `EVALSHA` on this object, in either mode: its shared
    /// `KEYS` prefix puts the header at `UPDATE_SCRIPT_KEY_HEADER` and the
    /// fence at `UPDATE_SCRIPT_KEY_FENCE` (`args[0]` is the command, `[1]` the
    /// SHA, `[2]` the key count).
    fn is_update_script_eval(&self) -> impl Fn(&MonitoredCommand) -> bool {
        let header = self.header_key();
        let fence = self.fence_key();
        move |c| {
            c.name() == "EVALSHA"
                && c.args.get(2 + UPDATE_SCRIPT_KEY_HEADER) == Some(&header)
                && c.args.get(2 + UPDATE_SCRIPT_KEY_FENCE) == Some(&fence)
        }
    }

    /// A `snapshot_fence_probe.lua` `EVALSHA` on this object: `KEYS` =
    /// fence, intent.
    fn is_probe_eval(&self) -> impl Fn(&MonitoredCommand) -> bool {
        let fence = self.fence_key();
        let intent = self.intent_key();
        move |c| {
            c.name() == "EVALSHA"
                && c.args.get(2).map(String::as_str) == Some("2")
                && c.args.get(3) == Some(&fence)
                && c.args.get(4) == Some(&intent)
        }
    }

    /// A `snapshot_intent_register.lua` `EVALSHA` on this object: `KEYS` =
    /// intent, fence, header.
    fn is_register_eval(&self) -> impl Fn(&MonitoredCommand) -> bool {
        let intent = self.intent_key();
        let fence = self.fence_key();
        move |c| {
            c.name() == "EVALSHA" && c.args.get(3) == Some(&intent) && c.args.get(4) == Some(&fence)
        }
    }

    /// A `snapshot_intent_renew.lua` `EVALSHA` on this object: one key, the
    /// intent key.
    #[cfg(feature = "test-fault-injection")]
    fn is_renew_eval(&self) -> impl Fn(&MonitoredCommand) -> bool {
        let intent = self.intent_key();
        move |c| {
            c.name() == "EVALSHA"
                && c.args.get(2).map(String::as_str) == Some("1")
                && c.args.get(3) == Some(&intent)
        }
    }

    /// A client command (not a script's) whose first key is `key`.
    fn names(key: String, command: &'static str) -> impl Fn(&MonitoredCommand) -> bool {
        move |c| !c.issued_by_script() && c.name() == command && c.args.get(1) == Some(&key)
    }
}

/// The commands a script issued for the `EVALSHA` at `at`: the `lua` lines
/// straight after it. A script runs atomically, so nothing interleaves.
fn script_commands(seen: &[MonitoredCommand], at: usize) -> Vec<MonitoredCommand> {
    seen.iter()
        .skip(at + 1)
        .take_while(|c| c.issued_by_script())
        .cloned()
        .collect()
}

fn position(seen: &[MonitoredCommand], what: &str, p: impl Fn(&MonitoredCommand) -> bool) -> usize {
    seen.iter()
        .position(p)
        .unwrap_or_else(|| panic!("the MONITOR record holds no {what}: {seen:#?}"))
}

/// Await a spawned commit within [`CALL_BUDGET`].
async fn join(
    writer: tokio::task::JoinHandle<Result<(u64, u64), ServiceError>>,
) -> Result<(u64, u64), ServiceError> {
    tokio::time::timeout(CALL_BUDGET, writer)
        .await
        .expect("the writer did not settle within the call budget")
        .expect("the writer task must not panic")
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. Every path, a fence briefly held
// ═════════════════════════════════════════════════════════════════════════════

/// A writer on `path` meeting a held fence is refused on every attempt, writes
/// nothing, and commits once the reader releases — on a one-attempt retry
/// budget, so the waits spent no attempt. Each attempt shows in the `MONITOR`
/// feed by its fence-checking command (the update script's `EVALSHA`, or the
/// WATCH path's probe); two of them prove the writer was refused, slept,
/// re-ran and was refused again. The call's wait is recorded once, under the
/// backend of the path that met the fence (`lua` for the update script's
/// paths, `txn` for the WATCH path), and every refusal is a
/// `snapshot_fence_wait` conflict on the refused attempt's path.
async fn assert_fenced_commit_waits_then_commits(path: Path) {
    let wx = Wx::new().await;
    wx.warm_scripts().await;
    wx.prepare(path).await;
    let store = wx.store_for(path, 1);
    let lease = wx.hold_fence().await;
    let before = wx.image().await;
    let seq_before = wx.seq().await;
    let what = format!("{path:?}");
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    let writer = wx.spawn_commit(&store, delta(path.shape(), 1));
    wx.await_two_fenced_attempts(&mut tap, path, &before, &what)
        .await;
    if path.registers_intents() {
        assert!(
            wx.intent_registrations("refused_fenced") >= 1,
            "{what}: the re-run, classified above the ceiling by its first probe, must have \
             tried to register an intent before WATCH and been refused by the live reader"
        );
        assert_eq!(wx.intent_registrations("registered"), 0, "{what}");
    }

    wx.release(lease).await;
    let (seq, _) = join(writer)
        .await
        .unwrap_or_else(|e| panic!("{what}: the writer must commit after release; got {e:?}"));

    assert_eq!(seq, seq_before + 1, "{what}: exactly one commit");
    assert_eq!(wx.stream_len().await, before.stream_len + 1, "{what}");
    wx.assert_one_attempt(&what);
    assert_eq!(
        wx.fence_waits(path.fence_backend()).0,
        1,
        "{what}: one call met the fence and is observed once, under the backend of the path \
         that met it"
    );
    let other_backend = if path.fence_backend() == "lua" {
        "txn"
    } else {
        "lua"
    };
    assert_eq!(wx.fence_waits(other_backend).0, 0, "{what}");
    assert_eq!(wx.fence_timeouts(), 0, "{what}");
    assert!(
        wx.fence_conflicts(Some(path.first_label())) >= 1,
        "{what}: the first refusal is a snapshot_fence_wait conflict on path {}",
        path.first_label()
    );
    assert!(
        wx.fence_conflicts(None) >= 2,
        "{what}: both refused attempts (the tap saw two) are snapshot_fence_wait conflicts"
    );
    if path.registers_intents() {
        assert_eq!(
            wx.intent_registrations("registered"),
            1,
            "{what}: once the reader released, the attempt registered its intent"
        );
    } else {
        assert_eq!(
            (
                wx.intent_registrations("registered"),
                wx.intent_registrations("refused_fenced")
            ),
            (0, 0),
            "{what}: this path registers no writer intent (structural-edits SR-8.2)"
        );
    }
    assert!(
        wx.intents().await.is_empty(),
        "{what}: no intent outlives the call"
    );
}

/// The fast path's script refuses `FENCED` while a reader holds the fence:
/// the writer re-runs it after an uncounted wait, nothing lands, and it
/// commits once the reader releases.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fast_path_commit_under_a_live_fence_waits_writes_nothing_and_commits_after_release() {
    within_test_budget(assert_fenced_commit_waits_then_commits(Path::Fast)).await;
}

/// The cached structural route: the update script's structural mode refuses
/// `FENCED` at step 1d, the route puts its entry back and waits uncounted, and
/// the re-run commits from the same entry after the release. No intent.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cached_structural_commit_under_a_live_fence_waits_writes_nothing_and_commits_after_release(
) {
    within_test_budget(assert_fenced_commit_waits_then_commits(Path::Structural)).await;
}

/// A structural call whose first read is a cold start (`structural_full`):
/// refused after its cold start, its re-runs read the entry the cold start put
/// back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cold_start_structural_commit_under_a_live_fence_waits_writes_nothing_and_commits_after_release(
) {
    within_test_budget(assert_fenced_commit_waits_then_commits(
        Path::StructuralFull,
    ))
    .await;
}

/// A reset on the structural route.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reset_commit_under_a_live_fence_waits_writes_nothing_and_commits_after_release() {
    within_test_budget(assert_fenced_commit_waits_then_commits(Path::Reset)).await;
}

/// A structural commit under `lua-watch-structural` (the WATCH path): the
/// probe beside its `WATCH`ed header read sees the live fence, and the writer
/// intent is refused while the reader is live (nothing written to the intent
/// key), registered once it released, cleared on commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn watch_knob_structural_commit_under_a_live_fence_waits_writes_nothing_and_commits_after_release(
) {
    within_test_budget(assert_fenced_commit_waits_then_commits(Path::WatchKnob)).await;
}

/// The txn backend's field-only commit (the rollback knob): the probe beside
/// its `WATCH`ed header read sees the live fence, the attempt waits uncounted,
/// and it commits after release. It reads no snapshot, so no intent.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn txn_field_only_commit_under_a_live_fence_waits_writes_nothing_and_commits_after_release() {
    within_test_budget(assert_fenced_commit_waits_then_commits(Path::TxnFieldOnly)).await;
}

/// The same structural commit under the txn backend.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn txn_structural_commit_under_a_live_fence_waits_writes_nothing_and_commits_after_release() {
    within_test_budget(assert_fenced_commit_waits_then_commits(Path::TxnStructural)).await;
}

/// A structural commit under `lua` on an object the structural cache cannot
/// hold: the WATCH fallback honours the fence like the knob does.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn uncacheable_fallback_commit_under_a_live_fence_waits_writes_nothing_and_commits_after_release(
) {
    within_test_budget(assert_fenced_commit_waits_then_commits(
        Path::UncacheableFallback,
    ))
    .await;
}

/// The update script's step 1d (review round 1 fix 2), in `path`'s mode: on a
/// layer with no fence key it costs exactly two `TYPE`s — fence, then intent —
/// and neither the fence `HGETALL` nor `TIME`. The control half runs the same
/// commit on a layer whose fence key lingers (released, not live): there the
/// script reads the key and the clock and still commits, so the record can
/// see the commands the first half asserts absent.
async fn step_1d_on_an_unfenced_layer_costs_two_types_and_no_fence_read(path: Path) {
    let wx = Wx::new().await;
    wx.warm_scripts().await;
    wx.prepare(path).await;
    let store = wx.store_for(path, 1);
    let fence = wx.fence_key();
    let intent = wx.intent_key();

    for lingering in [false, true] {
        if lingering {
            wx.fence_and_release().await;
        }
        let mut tap = MonitorTap::start(&test_valkey_addr()).await;
        wx.commit(&store, delta(path.shape(), u64::from(lingering)))
            .await
            .expect("an unfenced commit must commit");
        let record = wx.drain_to_marker(&mut tap).await;
        let eval = position(
            &record,
            "the update script's EVALSHA",
            wx.is_update_script_eval(),
        );
        let script = script_commands(&record, eval);
        let names: Vec<(String, Option<String>)> = script
            .iter()
            .map(|c| (c.name(), c.args.get(1).cloned()))
            .collect();

        let types: Vec<&String> = script
            .iter()
            .filter(|c| c.name() == "TYPE" && (c.args[1] == fence || c.args[1] == intent))
            .map(|c| &c.args[1])
            .collect();
        assert_eq!(
            types,
            vec![&fence, &intent],
            "{path:?}, lingering={lingering}: step 1d TYPE-probes the fence, then the intent \
             key; script: {names:?}"
        );
        let fence_read = script
            .iter()
            .any(|c| c.name() == "HGETALL" && c.args.get(1) == Some(&fence));
        let clock_read = script.iter().any(|c| c.name() == "TIME");
        assert_eq!(
            (fence_read, clock_read),
            (lingering, lingering),
            "{path:?}, lingering={lingering}: the fence HGETALL and TIME run only when the fence \
             key exists; script: {names:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fast_path_step_1d_on_an_unfenced_layer_costs_two_types_and_no_fence_read() {
    within_test_budget(step_1d_on_an_unfenced_layer_costs_two_types_and_no_fence_read(Path::Fast))
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_mode_step_1d_on_an_unfenced_layer_costs_two_types_and_no_fence_read() {
    within_test_budget(
        step_1d_on_an_unfenced_layer_costs_two_types_and_no_fence_read(Path::Structural),
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. WATCH path
// ═════════════════════════════════════════════════════════════════════════════

/// On a layer fenced within its last `key_linger(L)` (the key lingers, no
/// reader live), a structural call's FIRST WATCH-path attempt — not yet
/// classified — learns the layer is above the ceiling from its own probe and
/// registers its intent there, before its snapshot read: `WATCH` (header,
/// stream and fence) → probe → registration → the snapshot read's first
/// command → `EXEC` → the intent's `HDEL`. Exactly one registration, cleared
/// on commit.
async fn recently_fenced_layer_registers_the_intent_before_the_snapshot_read(path: Path) {
    let wx = Wx::new().await;
    wx.warm_scripts().await;
    wx.prepare(path).await;
    wx.fence_and_release().await;
    let store = wx.store_for(path, 1);
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    wx.commit(&store, delta(Shape::Structural, 1))
        .await
        .expect("the structural commit must commit");
    let seen = tap
        .drain_until_nth(1, "the intent's HDEL", Wx::names(wx.intent_key(), "HDEL"))
        .await;

    let header = wx.header_key();
    let watch = position(&seen, "WATCH", |c| {
        !c.issued_by_script() && c.name() == "WATCH" && c.args.get(1) == Some(&header)
    });
    assert_eq!(
        seen[watch].args[1..],
        [wx.header_key(), wx.stream_key(), wx.fence_key()],
        "{path:?}: the WATCH path WATCHes the header, the stream and the snapshot fence in one \
         command"
    );
    let watcher = seen[watch].client.clone();
    let probe = position(&seen, "the fence probe", wx.is_probe_eval());
    let register = position(&seen, "the intent registration", wx.is_register_eval());
    // The snapshot read's first round reads the sections index; nothing else
    // in the attempt touches that key, outside a script, before its EXEC.
    let sections = wx.tv.key_schema().obj_sections_index(&wx.oid_str());
    let snapshot_read = watch
        + 1
        + position(
            &seen[watch + 1..],
            "the snapshot read's sections-index read",
            |c| !c.issued_by_script() && c.args.get(1) == Some(&sections),
        );
    // The attempt's EXEC: the first one its WATCHing connection sends after
    // the WATCH (a pooled txn connection may carry earlier traffic).
    let exec = watch
        + 1
        + position(&seen[watch + 1..], "the EXEC", |c| {
            c.name() == "EXEC" && c.client == watcher
        });
    let hdel = seen.len() - 1;
    assert!(
        watch < probe
            && probe < register
            && register < snapshot_read
            && snapshot_read < exec
            && exec < hdel,
        "{path:?}: expected WATCH {watch} < probe {probe} < registration {register} < snapshot \
         read {snapshot_read} < EXEC {exec} < HDEL {hdel}"
    );
    assert_eq!(
        seen.iter().filter(|c| wx.is_register_eval()(c)).count(),
        1,
        "{path:?}: one registration: the attempt was not classified before WATCH"
    );
    assert_eq!(wx.intent_registrations("registered"), 1, "{path:?}");
    assert_eq!(wx.intent_registrations("refused_fenced"), 0, "{path:?}");
    assert!(wx.intents().await.is_empty(), "{path:?}: cleared on commit");
}

/// T12's WATCH half on the `lua-watch-structural` knob.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_knob_attempt_on_a_recently_fenced_layer_registers_its_intent_before_its_snapshot_read(
) {
    within_test_budget(
        recently_fenced_layer_registers_the_intent_before_the_snapshot_read(Path::WatchKnob),
    )
    .await;
}

/// T12's WATCH half on the `lua` fallback for an uncacheable object: the
/// fallback registers its intent exactly as the knob does.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn uncacheable_fallback_attempt_on_a_recently_fenced_layer_registers_its_intent_before_its_snapshot_read(
) {
    within_test_budget(
        recently_fenced_layer_registers_the_intent_before_the_snapshot_read(
            Path::UncacheableFallback,
        ),
    )
    .await;
}

/// SR-5.4's "an attempt on a never-fenced layer registers none": with no
/// fence key a below-ceiling structural WATCH-path commit issues no
/// registration and no command on the intent key beyond the probe's own
/// `TYPE`. An attempt that reads no snapshot — the txn knob's field-only
/// commit — registers none even on a recently fenced layer, and neither does
/// the cached structural route (structural-edits SR-8.2), whose script only
/// `TYPE`-probes the key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn attempts_that_need_no_intent_issue_no_intent_command() {
    within_test_budget(async {
        let wx = Wx::new().await;
        wx.warm_scripts().await;

        let mut fenced_once = false;
        let mut row = 0;
        for (path, fenced) in [
            (Path::WatchKnob, false),
            (Path::TxnFieldOnly, true),
            (Path::Structural, true),
        ] {
            // One epoch for both fenced rows: a second grant straight after
            // the first would meet the writers' gap the first one left.
            if fenced && !fenced_once {
                wx.fence_and_release().await;
                fenced_once = true;
            }
            let store = wx.store_for(path, 1);
            let mut tap = MonitorTap::start(&test_valkey_addr()).await;
            row += 1;
            wx.commit(&store, delta(path.shape(), 7 + row))
                .await
                .expect("the commit must commit");
            let seen = wx.drain_to_marker(&mut tap).await;

            // A command ON the intent key: one naming it first (`HDEL`, …) or a
            // script taking it as `KEYS[1]` (registration, renewal). The fence
            // probe, the update script and the pre-`MULTI` keyspace type probe
            // name it further in, for its type alone.
            let intent = wx.intent_key();
            let intent_commands: Vec<&MonitoredCommand> = seen
                .iter()
                .filter(|c| {
                    !c.issued_by_script()
                        && (c.args.get(1) == Some(&intent)
                            || (c.name() == "EVALSHA" && c.args.get(3) == Some(&intent)))
                })
                .collect();
            assert!(
                intent_commands.is_empty(),
                "{path:?} (fence key lingering: {fenced}): no registration, renewal or HDEL may \
                 be issued; got {intent_commands:#?}"
            );
            // Vacuity guard: the attempt did check the fence.
            let checked = if path.on_watch() {
                seen.iter().any(wx.is_probe_eval())
            } else {
                seen.iter().any(wx.is_update_script_eval())
            };
            assert!(checked, "{path:?}: the attempt checked the fence");
        }
        assert_eq!(wx.intent_registrations("registered"), 0);
        assert_eq!(wx.intent_registrations("refused_fenced"), 0);
    })
    .await;
}

/// An attempt that registered its intent and then failed — a wrong-typed key
/// the commit would write, refused by the pre-`MULTI` keyspace probe after
/// the snapshot read — clears its intent on the way out.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_attempt_that_errors_after_registering_clears_its_intent() {
    within_test_budget(async {
        let wx = Wx::new().await;
        wx.warm_scripts().await;
        wx.fence_and_release().await;
        let section_key = wx.tv.key_schema().obj_section(&wx.oid_str(), "101");
        let _: () = wx
            .tv
            .client()
            .pool
            .set(&section_key, "not-a-hash", None, None, false)
            .await
            .expect("planting a string must succeed");
        let store = wx.store_for(Path::WatchKnob, 1);

        let outcome = wx.commit(&store, delta(Shape::Structural, 1)).await;

        match outcome {
            Err(ServiceError::Internal(msg)) => assert!(
                msg.contains("WRONGTYPE_KEY") && msg.contains(&section_key),
                "the keyspace probe refuses the planted key; got {msg:?}"
            ),
            other => panic!("the commit must be refused Internal; got {other:?}"),
        }
        assert_eq!(
            wx.intent_registrations("registered"),
            1,
            "precondition: the attempt registered before it failed"
        );
        assert!(
            wx.intents().await.is_empty(),
            "an attempt that errors clears its intent"
        );
    })
    .await;
}

/// Review round 1 fix 1, the `SUPERSEDED` half: a call classified above the
/// ceiling (its first run's probe found a live fence) whose before-`WATCH`
/// registration is refused `FENCED` falls through to its guards, and the
/// object having been superseded meanwhile, it is refused `SUPERSEDED` at
/// once — under a fence that is still live, without waiting it out. The cap
/// is set so the pre-fix behaviour (treat the refused registration as a fence
/// wait) would end `UNAVAILABLE` instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_registration_before_watch_falls_through_to_superseded_without_waiting() {
    within_test_budget(async {
        let wx = Wx::new().await;
        wx.warm_scripts().await;
        let poll = wx.poll();
        let store = wx
            .store_for(Path::WatchKnob, 1)
            .with_fence_wait_deadline(poll * 5 / 2, Arc::new(|| None));
        let lease = wx.hold_fence().await;
        let mut tap = MonitorTap::start(&test_valkey_addr()).await;

        let writer = wx.spawn_commit(&store, delta(Shape::Structural, 1));
        tap.drain_until_nth(1, "the first run's probe", wx.is_probe_eval())
            .await;
        let _: () = wx
            .tv
            .client()
            .pool
            .hset(wx.header_key(), ("superseded", "1"))
            .await
            .expect("staging superseded=1 must succeed");
        let outcome = join(writer).await;

        match outcome {
            Err(ServiceError::FailedPrecondition(msg)) => assert!(
                msg.contains("superseded"),
                "the superseded guard answers; got {msg:?}"
            ),
            other => panic!(
                "a refused before-WATCH registration must fall through to the superseded guard \
                 rather than wait on the fence; got {other:?}"
            ),
        }
        assert!(
            wx.intent_registrations("refused_fenced") >= 1,
            "vacuity guard: the before-WATCH registration ran and was refused"
        );
        assert_eq!(wx.fence_timeouts(), 0);
        assert!(wx.intents().await.is_empty());
        assert!(
            fence::probe_fence(&wx.client, wx.tv.key_schema(), wx.oid)
                .await
                .map(|p| matches!(
                    p,
                    fence::FenceProbe::Present {
                        live_retry_after: Some(_),
                        ..
                    }
                ))
                .unwrap_or(false),
            "the fence was still live when the call ended"
        );
        drop(lease);
    })
    .await;
}

/// The Task's contract for siblings: the txn slot is held by one attempt
/// only, never across a fence wait. On a store whose txn pool has ONE
/// connection, a txn-backend writer parked on a fenced object — its attempt
/// over (the `UNWATCH` after its probe), its fence sleep running — leaves that
/// connection free: a txn commit on another object completes while the first
/// writer still waits. Held across the sleep, the slot would keep the second
/// commit queued until the fence ends, which here is never before the test
/// releases it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_fenced_txn_writer_does_not_hold_the_txn_slot_while_it_waits() {
    within_test_budget(async {
        const OTHER_LAYER: &str = "fence-writer-other";
        let wx = Wx::with_pool_size(1).await;
        wx.warm_scripts().await;
        let other = wx.create_other_layer(OTHER_LAYER).await;
        let store = wx.store(UpdateLayerBackend::Txn, 1);
        let lease = wx.hold_fence().await;
        let before = wx.image().await;
        let mut tap = MonitorTap::start(&test_valkey_addr()).await;

        let writer = wx.spawn_commit(&store, delta(Shape::FieldOnly, 1));
        let probed = tap
            .drain_until_nth(1, "the fenced writer's probe", wx.is_probe_eval())
            .await;
        let conn = probed.last().expect("the probe line").client.clone();
        tap.drain_until_nth(1, "the fenced attempt's UNWATCH", |c| {
            c.client == conn && c.name() == "UNWATCH"
        })
        .await;

        let other_delta = delta(Shape::FieldOnly, 2);
        let other_relay = relay(&other_delta);
        let (other_seq, _) = tokio::time::timeout(
            CALL_BUDGET,
            store.update_layer(
                &wx.session_id,
                OTHER_LAYER,
                other,
                Some(other_delta),
                other_relay,
                "other-writer",
                1,
            ),
        )
        .await
        .expect(
            "a txn commit on another object must not queue behind a fenced writer: the pool's only \
             txn connection was held across the fence wait",
        )
        .expect("the other object's commit must commit");

        assert_eq!(other_seq, 2, "the other object took its first commit");
        assert!(
            !writer.is_finished(),
            "the fenced writer is still waiting (its fence is live until released)"
        );
        assert_eq!(
            wx.image().await,
            before,
            "nothing landed on the fenced object"
        );
        wx.release(lease).await;
        join(writer)
            .await
            .expect("the fenced writer commits after the release");
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. Killed reader
// ═════════════════════════════════════════════════════════════════════════════

/// A reader that is granted the fence and then dies — never renewing, never
/// releasing, no destructor — holds writers off only until its lease ends.
/// The writer on `path` starts under the live lease and commits on a
/// one-attempt budget, and the dead reader's field is still on the key,
/// expired by the store's own clock.
async fn killed_reader(path: Path) {
    let wx = Wx::new().await;
    wx.warm_scripts().await;
    wx.prepare(path).await;
    let store = wx.store_for(path, 1);
    let seq_before = wx.seq().await;
    let lease = wx.grant().await;
    let reader: (String, String) = wx
        .fields(&wx.fence_key())
        .await
        .into_iter()
        .find(|(k, _)| k.starts_with("r:"))
        .expect("the lease's field");
    // A crash runs no destructor: no release, no renewal.
    std::mem::forget(lease);
    let what = format!("{path:?}");

    let (seq, _) = wx
        .commit(&store, delta(path.shape(), 1))
        .await
        .unwrap_or_else(|e| panic!("{what}: the writer must commit after the lease; got {e:?}"));

    assert_eq!(seq, seq_before + 1, "{what}");
    assert_eq!(
        wx.fence_waits(path.fence_backend()).0,
        1,
        "{what}: the writer met the live lease (vacuity guard: it did not simply run after it)"
    );
    wx.assert_one_attempt(&what);
    let deadline: i64 = wx
        .fields(&wx.fence_key())
        .await
        .get(&reader.0)
        .unwrap_or_else(|| panic!("{what}: the dead reader's field is never released"))
        .parse()
        .expect("a deadline");
    assert!(
        deadline <= wx.now().await,
        "{what}: the commit landed only after the lease ended on the store's clock"
    );
    assert!(wx.intents().await.is_empty(), "{what}");
}

// One test per commit path, each on its own fixture. A single test joining
// several fixtures on one task once overran its call budget (and its cap
// variant ran ~6 s where each path alone takes ~1.3 s); the cause is
// unconfirmed.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_killed_readers_lease_ends_and_the_fast_path_resumes_without_spending_a_retry() {
    within_test_budget(killed_reader(Path::Fast)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_killed_readers_lease_ends_and_the_cached_structural_route_resumes_without_spending_a_retry(
) {
    within_test_budget(killed_reader(Path::Structural)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_killed_readers_lease_ends_and_the_watch_path_resumes_without_spending_a_retry() {
    within_test_budget(killed_reader(Path::WatchKnob)).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. The fence-wait cap
// ═════════════════════════════════════════════════════════════════════════════

/// A fence renewed for ever: the call's fence wait reaches its cap. With the
/// cap at 1.5 × `poll` exactly one sleep fits — the refused attempt plus a
/// sleep of `poll` to `poll + rtt99` ends well under it — and the second would
/// end past `2 × poll`, so the call fails `UNAVAILABLE` — its message naming
/// the waited span, the cap, the object and the path of the attempt the cap
/// stopped (structural-edits B5) — logs no WARN of its own (the handler
/// boundary's line is the call's one log line, SDD §17.2.1), writes nothing,
/// ticks the timeout counter once, and records one wait under the backend of
/// the path that met the fence. That wait spans the refused first attempt and the one sleep taken
/// (Spec SDD SR-8.3: from the start of the first attempt a fence refused to
/// the end of the last fence sleep), so it is no shorter than the sleep,
/// strictly under the cap that admitted the sleep, and inside the time the
/// call ran.
async fn capped(path: Path) {
    let wx = Wx::new().await;
    wx.warm_scripts().await;
    wx.prepare(path).await;
    let poll = wx.poll();
    let cap = poll * 3 / 2;
    let store = wx
        .store_for(path, 1)
        .with_fence_wait_deadline(cap, Arc::new(|| None));
    let _lease = wx.hold_fence().await;
    let before = wx.image().await;
    let what = format!("{path:?}");

    let started = tokio::time::Instant::now();
    let (outcome, captured) = capture_debug(wx.commit(&store, delta(path.shape(), 1))).await;
    let ran_for = started.elapsed();

    assert_no_warn_logged(&captured, &what);
    let err = outcome.as_ref().err().unwrap_or_else(|| {
        panic!("{what}: the cap fails the call UNAVAILABLE (retryable); got {outcome:?}")
    });
    let message = assert_refusal_error(
        err,
        refusal_reason::SNAPSHOT_FENCE_TIMEOUT,
        &[],
        CAPACITY_RETRY_DELAY,
    );
    // The attempt the cap stopped is a re-run: on a cold-start path that is
    // already a hit, so its path is the hit's. This store has no remaining
    // deadline, so the text names the path alone.
    let stopped_label = match path {
        Path::StructuralFull => "structural",
        _ => path.first_label(),
    };
    assert_cap_message(message, cap, wx.oid, stopped_label, &what);
    let event = only_fence_wait_event(&wx, &captured, path.fence_backend());
    assert_fence_wait_event(event, false, false, 1, &what);
    assert_event_matches_histogram(event, wx.fence_waits(path.fence_backend()).1, &what);
    assert_eq!(wx.image().await, before, "{what}: nothing written");
    assert_eq!(wx.fence_timeouts(), 1, "{what}");
    let (count, sum) = wx.fence_waits(path.fence_backend());
    assert_eq!(count, 1, "{what}: one observation for the call");
    // The histogram holds f64 seconds: allow a microsecond of rounding.
    let waited = Duration::from_secs_f64(sum);
    let slack = Duration::from_micros(1);
    assert!(
        waited + slack >= poll,
        "{what}: the recorded wait holds the one sleep taken (at least poll {poll:?}); got \
         {waited:?}"
    );
    assert!(
        waited < cap + slack,
        "{what}: the cap admitted that sleep only because the span to its end — the refused \
         attempt and the sleep — stayed under the cap ({cap:?}); got {waited:?}"
    );
    assert!(
        waited <= ran_for + slack,
        "{what}: the span ends with the last sleep, inside the call's {ran_for:?}; got {waited:?}"
    );
    assert!(wx.intents().await.is_empty(), "{what}");
}

// One test per commit path, as for the killed-reader tests above: the joined
// form was the one seen to overrun.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_permanently_renewed_fence_fails_a_fast_path_commit_unavailable_at_the_cap() {
    within_test_budget(capped(Path::Fast)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_permanently_renewed_fence_fails_a_cached_structural_commit_unavailable_at_the_cap() {
    within_test_budget(capped(Path::Structural)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_permanently_renewed_fence_fails_a_cold_start_structural_commit_unavailable_at_the_cap() {
    within_test_budget(capped(Path::StructuralFull)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_permanently_renewed_fence_fails_a_reset_commit_unavailable_at_the_cap() {
    within_test_budget(capped(Path::Reset)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_permanently_renewed_fence_fails_a_watch_knob_commit_unavailable_at_the_cap() {
    within_test_budget(capped(Path::WatchKnob)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_permanently_renewed_fence_fails_a_txn_field_only_commit_unavailable_at_the_cap() {
    within_test_budget(capped(Path::TxnFieldOnly)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_permanently_renewed_fence_fails_a_txn_structural_commit_unavailable_at_the_cap() {
    within_test_budget(capped(Path::TxnStructural)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_permanently_renewed_fence_fails_an_uncacheable_fallback_commit_unavailable_at_the_cap() {
    within_test_budget(capped(Path::UncacheableFallback)).await;
}

/// The fence-wait cap on a fast-path call of two attempts, after attempt 1's
/// `EVALSHA` timed out when `after_a_timed_out_attempt` (a lost reply, whose
/// commit may still land), else on attempt 1. Attempt 1's cap carries the
/// `SNAPSHOT_FENCE_TIMEOUT` hint; attempt 2's answers plain `UNAVAILABLE`,
/// outcome unknown — no hint, no nothing-written claim (SDD §18.1). Either
/// way the timeout is counted once and nothing is written.
#[cfg(feature = "test-fault-injection")]
async fn capped_on_attempt(after_a_timed_out_attempt: bool) {
    let wx = Wx::new().await;
    wx.warm_scripts().await;
    let cap = wx.poll() * 3 / 2;
    let timed_out: Vec<fred::error::Error> = if after_a_timed_out_attempt {
        vec![fred::error::Error::new(
            fred::error::ErrorKind::Timeout,
            "synthetic lost reply",
        )]
    } else {
        Vec::new()
    };
    let store = wx
        .store_for(Path::Fast, 2)
        .with_fence_wait_deadline(cap, Arc::new(|| None))
        .with_lua_fred_error_injector_for_test(Arc::new(std::sync::Mutex::new(
            timed_out.into_iter().collect(),
        )));
    let _lease = wx.hold_fence().await;
    let before = wx.image().await;
    let what = format!("after a timed-out attempt: {after_a_timed_out_attempt}");

    let outcome = wx.commit(&store, delta(Path::Fast.shape(), 1)).await;

    let err = outcome
        .as_ref()
        .err()
        .unwrap_or_else(|| panic!("{what}: the cap fails the call; got {outcome:?}"));
    if after_a_timed_out_attempt {
        let ServiceError::Unavailable(message) = err else {
            panic!("{what}: plain UNAVAILABLE, without the retry hint; got {err:?}");
        };
        let prefix = "UpdateLayer outcome unknown: an earlier attempt of this call may have \
                      committed before attempt 2 was refused (update_layer waited ";
        let suffix = format!(
            "snapshot-fence wait cap on object {} ({}))",
            wx.oid,
            Path::Fast.first_label()
        );
        assert!(
            message.starts_with(prefix) && message.ends_with(&suffix),
            "{what}: {message}"
        );
        assert!(!message.contains("nothing was"), "{what}: {message}");
    } else {
        let message = assert_refusal_error(
            err,
            refusal_reason::SNAPSHOT_FENCE_TIMEOUT,
            &[],
            CAPACITY_RETRY_DELAY,
        );
        assert_cap_message(message, cap, wx.oid, Path::Fast.first_label(), &what);
    }
    assert_eq!(wx.image().await, before, "{what}: nothing written");
    assert_eq!(wx.fence_timeouts(), 1, "{what}");
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_fence_wait_cap_on_attempt_one_keeps_its_retry_hint_in_a_call_allowed_a_retry() {
    within_test_budget(capped_on_attempt(false)).await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_fence_wait_cap_after_a_timed_out_attempt_is_plain_unavailable_outcome_unknown() {
    within_test_budget(capped_on_attempt(true)).await;
}

/// What a [`deadline_bound`] store has left of its call's deadline: under one
/// `poll`, so the first fence sleep (`poll` or more) already would not end
/// before it. Not 1 ms, so the value the status text names is one no other
/// figure in it could be.
const LEFT_OF_DEADLINE: Duration = Duration::from_millis(250);

/// The RPC's remaining deadline is the second bound: with
/// [`LEFT_OF_DEADLINE`] left, the first sleep already would not end before
/// it, so the call fails `UNAVAILABLE` after its first refusal, having waited
/// nothing — and still counts as a call that met the fence (a zero
/// observation). As at the cap, the commit logs no WARN; the status text
/// names the path and what was left of the call's deadline beside the cap.
async fn deadline_bound(path: Path) {
    let wx = Wx::new().await;
    wx.warm_scripts().await;
    wx.prepare(path).await;
    let cap = Duration::from_secs(30);
    assert!(
        LEFT_OF_DEADLINE < wx.poll(),
        "fixture: the remaining deadline must refuse the first sleep"
    );
    let store = wx
        .store_for(path, 1)
        .with_fence_wait_deadline(cap, Arc::new(|| Some(LEFT_OF_DEADLINE)));
    let _lease = wx.hold_fence().await;
    let before = wx.image().await;
    let what = format!("{path:?}, deadline-bound");

    let (outcome, captured) = capture_debug(wx.commit(&store, delta(path.shape(), 1))).await;

    assert_no_warn_logged(&captured, &what);
    let err = outcome.as_ref().err().unwrap_or_else(|| {
        panic!("{what}: the deadline fails the call UNAVAILABLE (retryable); got {outcome:?}")
    });
    let message = assert_refusal_error(
        err,
        refusal_reason::SNAPSHOT_FENCE_TIMEOUT,
        &[],
        CAPACITY_RETRY_DELAY,
    );
    assert_cap_message(
        message,
        cap,
        wx.oid,
        &format!(
            "{}; {} ms left of the call's deadline",
            path.first_label(),
            LEFT_OF_DEADLINE.as_millis()
        ),
        &what,
    );
    assert_eq!(wx.image().await, before, "{what}: nothing written");
    assert_eq!(wx.fence_timeouts(), 1, "{what}");
    assert_eq!(wx.fence_waits(path.fence_backend()), (1, 0.0), "{what}");
    let event = only_fence_wait_event(&wx, &captured, path.fence_backend());
    assert_fence_wait_event(event, false, false, 1, &what);
    assert_eq!(
        event["fields"]["waited_us"], 0,
        "{what}: it waited nothing: {event}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_remaining_rpc_deadline_bounds_the_fence_wait() {
    within_test_budget(deadline_bound(Path::Fast)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_remaining_rpc_deadline_bounds_a_structural_commits_fence_wait() {
    within_test_budget(deadline_bound(Path::TxnStructural)).await;
}

/// The message of the WARN `update_layer` logged at its fence-wait cap before
/// that failure was left to the handler boundary's one completion line (SDD
/// §17.2.1). Matched as a prefix, so no rewording of the old line passes.
const RETIRED_FENCE_CAP_WARN: &str = "UpdateLayer gave up waiting for a snapshot fence";

/// Assert the commit logged no WARN-or-above event — in particular not the
/// retired fence-cap WARN: the failed call's one log line is the handler
/// boundary's, which a store-level call never reaches.
fn assert_no_warn_logged(captured: &Captured, what: &str) {
    let retired: Vec<&Value> = captured
        .events
        .iter()
        .filter(|e| {
            e["fields"]["message"]
                .as_str()
                .is_some_and(|m| m.starts_with(RETIRED_FENCE_CAP_WARN))
        })
        .collect();
    assert!(
        retired.is_empty(),
        "{what}: the fence-wait cap logs no line of its own; got {retired:?}"
    );
    let loud: Vec<&Value> = captured
        .events
        .iter()
        .filter(|e| matches!(e["level"].as_str(), Some("WARN" | "ERROR")))
        .collect();
    assert!(
        loud.is_empty(),
        "{what}: the commit logs no WARN or ERROR; full capture:\n{}",
        captured.text
    );
}

/// Assert a fence-wait-cap `UNAVAILABLE` text is exactly `update_layer
/// waited {N} ms of its {cap} ms snapshot-fence wait cap on object {oid}
/// ({paren}); nothing was written, retry`, `N` being any whole number of ms
/// (the wait varies run to run) and `paren` the path label, followed by what
/// was left of the call's deadline when one is in scope.
fn assert_cap_message(message: &str, cap: Duration, oid: u64, paren: &str, what: &str) {
    let waited_ms = message
        .strip_prefix("update_layer waited ")
        .and_then(|rest| rest.split_once(" ms of its"))
        .and_then(|(n, _)| n.parse::<u64>().ok())
        .unwrap_or_else(|| panic!("{what}: the text starts with the wait in ms; got {message:?}"));
    assert_eq!(
        message,
        format!(
            "update_layer waited {waited_ms} ms of its {} ms snapshot-fence wait cap on object \
             {oid} ({paren}); nothing was written, retry",
            cap.as_millis()
        ),
        "{what}: the client-visible text names the wait, the cap, the object, the path and, \
         when one is in scope, what was left of the call's deadline"
    );
}

// ── Every exit records once: the cancelled call (Spec SDD SR-8.3) ──────────
//
// A call that met a fence records its fence wait — one histogram observation
// and one `update_layer fence wait` DEBUG event — exactly once on every exit:
// where it returns (commit, the cap's `UNAVAILABLE`, any error), and when its
// future is dropped mid-call (the RPC deadline's tower layer drops the handler
// future; a disconnect or shutdown does the same). The event is the
// validation tooling's per-call record (`committed`, `cancelled`, `attempt`),
// so its fields are the contract here. A call that never met a fence records
// nothing.

/// The single `update_layer fence wait` event in `captured`, checked against
/// this fixture's object and the call's backend label.
fn only_fence_wait_event<'c>(wx: &Wx, captured: &'c Captured, backend: &str) -> &'c Value {
    let event = captured.only(FENCE_WAIT_EVENT_MSG);
    assert_eq!(event["level"], "DEBUG", "{event}");
    assert_eq!(event["fields"]["object_id"], wx.oid, "{event}");
    assert_eq!(event["fields"]["backend"], backend, "{event}");
    event
}

/// Assert an `update_layer fence wait` event's call-ending fields.
fn assert_fence_wait_event(
    event: &Value,
    committed: bool,
    cancelled: bool,
    attempt: u64,
    what: &str,
) {
    assert_eq!(
        event["fields"]["committed"], committed,
        "{what}: committed; event: {event}"
    );
    assert_eq!(
        event["fields"]["cancelled"], cancelled,
        "{what}: cancelled; event: {event}"
    );
    assert_eq!(
        event["fields"]["attempt"], attempt,
        "{what}: the attempt the call reached; event: {event}"
    );
}

/// The event's `waited_us` against the histogram's sum for the same single
/// observation (f64 seconds; a microsecond of rounding allowed).
fn assert_event_matches_histogram(event: &Value, histogram_sum_s: f64, what: &str) {
    let waited_us = event["fields"]["waited_us"]
        .as_u64()
        .unwrap_or_else(|| panic!("{what}: waited_us must be an integer; event: {event}"));
    let observed_us = histogram_sum_s * 1e6;
    assert!(
        (waited_us as f64 - observed_us).abs() <= 1.0,
        "{what}: the event's waited_us ({waited_us}) must be the histogram's observation \
         ({observed_us} µs)"
    );
}

/// A call meets a held fence, sleeps once, re-runs (refused again) and is
/// then dropped — a `tokio::time::timeout` elapsing during its second fence
/// sleep, as the RPC deadline's tower layer drops the handler future. It
/// records exactly one histogram observation and one event, marked
/// `cancelled` and not `committed`, at attempt 1 (fence waits re-run the same
/// attempt), spanning at least its one completed sleep (`poll`, the hint a
/// far deadline gives) and at most the time the call ran. Without the drop
/// record, a dropped call records nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_call_dropped_mid_fence_wait_records_its_wait_once_marked_cancelled() {
    within_test_budget(async {
        let wx = Wx::new().await;
        wx.warm_scripts().await;
        let store = wx.store(UpdateLayerBackend::Lua, 1);
        let _lease = wx.hold_fence().await;
        let before = wx.image().await;
        let poll = wx.poll();
        let mut tap = MonitorTap::start(&test_valkey_addr()).await;

        let fenced = delta(Shape::FieldOnly, 1);
        let fenced_relay = relay(&fenced);
        let (ran_for, captured) = capture_debug(async {
            let started = tokio::time::Instant::now();
            let mut call = Box::pin(store.update_layer(
                &wx.session_id,
                LAYER,
                wx.oid,
                Some(fenced),
                fenced_relay,
                CLIENT_ID,
                1_000,
            ));
            // Drive the call until its re-run — after one whole fence sleep —
            // has reached the store.
            tokio::select! {
                outcome = call.as_mut() => {
                    panic!("a call under a held fence must not return; got {outcome:?}")
                }
                _ = tap.drain_until_nth(2, "the re-run's fast-path EVALSHA", wx.is_update_script_eval()) => {}
            }
            // The deadline elapses during the re-run or its fence sleep.
            let timed_out = tokio::time::timeout(Duration::from_millis(50), call.as_mut()).await;
            assert!(
                timed_out.is_err(),
                "precondition: the call is still waiting on the fence; got {timed_out:?}"
            );
            drop(call);
            started.elapsed()
        })
        .await;

        assert_eq!(wx.image().await, before, "nothing written");
        let (count, sum) = wx.fence_waits("lua");
        assert_eq!(count, 1, "one observation for the dropped call");
        let waited = Duration::from_secs_f64(sum);
        let slack = Duration::from_micros(1);
        assert!(
            waited + slack >= poll && waited <= ran_for + slack,
            "the recorded span covers the one completed sleep (poll {poll:?}) and ends at the \
             drop, inside the call's {ran_for:?}; got {waited:?}"
        );
        let event = only_fence_wait_event(&wx, &captured, "lua");
        assert_fence_wait_event(event, false, true, 1, "the dropped call");
        assert_event_matches_histogram(event, sum, "the dropped call");
        assert_eq!(wx.fence_timeouts(), 0, "a dropped call is not a cap timeout");
    })
    .await;
}

/// A call that met a fence and returned a commit records once, with
/// `cancelled` false and `committed` true: the return site's record disarms
/// the drop's, so the call's future going out of scope adds nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_call_that_met_a_fence_and_committed_records_its_wait_once_not_cancelled() {
    within_test_budget(async {
        let wx = Wx::new().await;
        wx.warm_scripts().await;
        let store = wx.store(UpdateLayerBackend::Lua, 1);
        let lease = wx.hold_fence().await;
        let mut tap = MonitorTap::start(&test_valkey_addr()).await;

        let fenced = delta(Shape::FieldOnly, 1);
        let fenced_relay = relay(&fenced);
        let (outcome, captured) = capture_debug(async {
            let call = store.update_layer(
                &wx.session_id,
                LAYER,
                wx.oid,
                Some(fenced),
                fenced_relay,
                CLIENT_ID,
                1_000,
            );
            let release_after_the_first_refusal = async {
                tap.drain_until_nth(
                    1,
                    "the refused fast-path EVALSHA",
                    wx.is_update_script_eval(),
                )
                .await;
                wx.release(lease).await;
            };
            let (outcome, ()) = tokio::join!(
                tokio::time::timeout(CALL_BUDGET, call),
                release_after_the_first_refusal
            );
            outcome.expect("update_layer did not settle within the call budget")
        })
        .await;

        outcome.unwrap_or_else(|e| panic!("the call must commit after the release; got {e:?}"));
        let (count, sum) = wx.fence_waits("lua");
        assert_eq!(count, 1, "one observation for the call");
        let event = only_fence_wait_event(&wx, &captured, "lua");
        assert_fence_wait_event(event, true, false, 1, "the committed call");
        assert_event_matches_histogram(event, sum, "the committed call");
    })
    .await;
}

/// The step-1b verdict a staged commit record ends a fenced call with
/// ([`a_fenced_call_ended_by_a_staged_record`]).
#[derive(Debug, Clone, Copy)]
enum FencedEnding {
    /// The call's `client_id`'s commit record moved while it waited:
    /// `UNAVAILABLE` (overtaken). No commit of its own.
    Overtaken,
    /// An earlier call's commit of the same operation and request: reported
    /// `ALREADY_COMMITTED`, at that call's `seq`. No commit of its own.
    EarlierCall,
    /// An earlier call's commit of the same operation for a different
    /// request: `ALREADY_EXISTS`. No commit of its own.
    Conflict,
    /// A commit under the call's own token — as an earlier attempt of it
    /// whose reply was lost would leave: reported as the call's own commit.
    /// Needs the call's token pinned (`with_commit_tokens_for_test`).
    #[cfg(feature = "test-fault-injection")]
    ThisCall,
}

/// The token [`FencedEnding::ThisCall`] pins for the call.
#[cfg(feature = "test-fault-injection")]
const FENCED_ENDING_OWN_TOKEN: u128 = 0x0b3a_8e3a_0000_0000_0000_0000_0000_0003;

/// A call that met a held fence and is then ended, still under the fence, by
/// a step-1b verdict read from a staged commit record records its wait once
/// with `cancelled` false and `committed` true only when the verdict is a
/// commit of its own: the fast path's first attempt is refused `FENCED`;
/// while the call sleeps the test stages `wtok[client_id]` for `ending`; the
/// re-run's step 1b answers ahead of the still-live fence (`OVERTAKEN`, or
/// `ALREADY_LANDED` with the record). Every ending returns through an `Ok`
/// attempt verdict, so a record keyed on "the attempt loop returned `Ok`"
/// would log `committed` true for the three that committed nothing, and one
/// keyed on "this attempt committed" would log it false for
/// [`FencedEnding::ThisCall`]. Nothing is written.
async fn a_fenced_call_ended_by_a_staged_record(ending: FencedEnding) {
    use live_edit::storage::layer::{
        operation_request_digest, Operation, OperationOutcome, OPERATION_CONFLICT_MESSAGE,
        OPERATION_OVERTAKEN_MESSAGE,
    };

    let wx = Wx::new().await;
    wx.warm_scripts().await;
    let store = wx.store(UpdateLayerBackend::Lua, 1);
    #[cfg(feature = "test-fault-injection")]
    let store = match ending {
        FencedEnding::ThisCall => store.with_commit_tokens_for_test(Arc::new(
            std::sync::Mutex::new(std::collections::VecDeque::from([FENCED_ENDING_OWN_TOKEN])),
        )),
        _ => store,
    };
    let lease = wx.hold_fence().await;
    let seq_before = wx.seq().await;
    let stream_len_before = wx.stream_len().await;
    let what = format!("{ending:?}");
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    let fenced = delta(Shape::FieldOnly, 1);
    let fenced_relay = relay(&fenced);
    let digest = operation_request_digest(
        LAYER,
        fenced_relay
            .as_deref()
            .expect("relay() always carries bytes"),
    );
    // A commit record no attempt of this call wrote: another operation's
    // (overtaken); this operation's under another call's token (the call's
    // own is random unless pinned); or, for `ThisCall`, under the call's
    // pinned token — each at a `seq` and `server_ts` of the test's choosing.
    let landed_seq = seq_before;
    let landed_ts = 1_700_000_000_000_000_u64;
    let staged = match ending {
        FencedEnding::Overtaken => "op.other-op:2:1700000000000000:tok:digest".to_owned(),
        FencedEnding::EarlierCall => {
            format!(
                "op.fenced-op:{landed_seq}:{landed_ts}:{:032x}:{digest}",
                u128::MAX
            )
        }
        FencedEnding::Conflict => format!(
            "op.fenced-op:{landed_seq}:{landed_ts}:{:032x}:{}",
            u128::MAX,
            operation_request_digest(LAYER, b"a different request")
        ),
        #[cfg(feature = "test-fault-injection")]
        FencedEnding::ThisCall => {
            format!("op.fenced-op:{landed_seq}:{landed_ts}:{FENCED_ENDING_OWN_TOKEN:032x}:{digest}")
        }
    };
    // No start record: `update_layer_with_operation` reads `wtok[client_id]`
    // itself at call start, as a caller that passes none gets, so step 1b
    // answers "apply" until the staging below.
    let operation = Operation::new("fenced-op", digest.clone());

    let (outcome, captured) = capture_debug(async {
        let call = store.update_layer_with_operation(
            &wx.session_id,
            LAYER,
            wx.oid,
            Arc::new(Some(fenced)),
            fenced_relay,
            CLIENT_ID,
            1_000,
            Some(operation),
        );
        // Staged after the first refusal, so the call has met the fence (its
        // fence-wait record is armed) before step 1b can answer anything but
        // "apply". A re-run that ran before the staging is refused again and
        // re-runs once more; the fence stays live throughout.
        let stage_after_the_first_refusal = async {
            tap.drain_until_nth(
                1,
                "the refused fast-path EVALSHA",
                wx.is_update_script_eval(),
            )
            .await;
            let _: () = wx
                .tv
                .client()
                .pool
                .hset(wx.wtok_key(), (CLIENT_ID, staged.as_str()))
                .await
                .expect("staging the commit record must succeed");
        };
        let (outcome, ()) = tokio::join!(
            tokio::time::timeout(CALL_BUDGET, call),
            stage_after_the_first_refusal
        );
        outcome.expect("update_layer did not settle within the call budget")
    })
    .await;

    let own_commit = match (ending, outcome) {
        (FencedEnding::Overtaken, Err(ServiceError::Unavailable(msg))) => {
            assert_eq!(
                msg, OPERATION_OVERTAKEN_MESSAGE,
                "{what}: the overtaken verdict, not the fence-wait cap"
            );
            false
        }
        (FencedEnding::EarlierCall, Ok(commit)) => {
            assert_eq!(
                (commit.seq, commit.server_ts, commit.outcome),
                (landed_seq, landed_ts, OperationOutcome::AlreadyCommitted),
                "{what}: the earlier call's commit is reported ALREADY_COMMITTED"
            );
            false
        }
        (FencedEnding::Conflict, Err(ServiceError::AlreadyExists(msg))) => {
            assert_eq!(msg, OPERATION_CONFLICT_MESSAGE, "{what}");
            false
        }
        #[cfg(feature = "test-fault-injection")]
        (FencedEnding::ThisCall, Ok(commit)) => {
            assert_eq!(
                (commit.seq, commit.server_ts, commit.outcome),
                (landed_seq, landed_ts, OperationOutcome::Applied),
                "{what}: the call's own landed commit is reported as its commit"
            );
            true
        }
        (_, other) => panic!("{what}: unexpected outcome {other:?}"),
    };
    assert_eq!(wx.seq().await, seq_before, "{what}: nothing committed");
    assert_eq!(
        wx.stream_len().await,
        stream_len_before,
        "{what}: no stream entry"
    );
    assert!(
        wx.fence_conflicts(Some("fast")) >= 1,
        "vacuity guard ({what}): the call met the fence before its verdict"
    );
    assert_eq!(
        wx.fence_timeouts(),
        0,
        "{what}: no fence wait reached the cap"
    );
    let (count, sum) = wx.fence_waits("lua");
    assert_eq!(count, 1, "{what}: one observation for the call");
    let event = only_fence_wait_event(&wx, &captured, "lua");
    let returned = if own_commit {
        "a commit of its own"
    } else {
        "no commit of its own"
    };
    assert_fence_wait_event(
        event,
        own_commit,
        false,
        1,
        &format!("{what}: the call returned {returned}"),
    );
    assert_event_matches_histogram(event, sum, &what);
    wx.release(lease).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_fenced_call_overtaken_after_its_wait_records_its_wait_not_committed() {
    within_test_budget(a_fenced_call_ended_by_a_staged_record(
        FencedEnding::Overtaken,
    ))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_fenced_resend_answered_already_committed_after_its_wait_records_its_wait_not_committed()
{
    within_test_budget(a_fenced_call_ended_by_a_staged_record(
        FencedEnding::EarlierCall,
    ))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_fenced_operation_conflict_after_its_wait_records_its_wait_not_committed() {
    within_test_budget(a_fenced_call_ended_by_a_staged_record(
        FencedEnding::Conflict,
    ))
    .await;
}

/// A call that never met a fence records nothing, whether it returns (a
/// commit on a never-fenced layer) or is dropped mid-call (polled once —
/// past its uncontended admission permit into its first attempt, which waits
/// on the store — and dropped there, before any fence answer): the drop
/// record arms only when a fence first refuses or fails one of the call's
/// attempts (its first fence wait).
///
/// The dropped call runs on a current-thread runtime of its own
/// (`snapshot_read_support`'s [`on_current_thread`]) on a client connected
/// there, so its one poll cannot finish it: no reply can be delivered inside
/// the poll that sent its request. On the test's multi-thread runtime a
/// worker could deliver every reply while the polling thread is descheduled,
/// and the call would then commit in that one poll.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_call_that_never_met_a_fence_records_no_fence_wait_however_it_ends() {
    let deadline = tokio::time::Instant::now() + TEST_BUDGET;
    let wx = until_test_deadline(deadline, async {
        let wx = Wx::new().await;
        wx.warm_scripts().await;
        Arc::new(wx)
    })
    .await;

    let theirs = Arc::clone(&wx);
    on_current_thread(move || {
        Box::pin(async move {
            let polled = async {
                let client = connect_test_client(&crate::common::test_config())
                    .await
                    .expect("the dropped call's client must connect");
                let store = LayerStore::new(client, theirs.tv.key_schema().clone(), 4096)
                    .with_update_layer_backend(UpdateLayerBackend::Lua)
                    .with_update_layer_retry_max_attempts(1)
                    .with_structural_cache(theirs.cache.clone())
                    .with_metrics(Arc::clone(&theirs.metrics));
                let unfenced = delta(Shape::FieldOnly, 1);
                let unfenced_relay = relay(&unfenced);
                capture_debug(async {
                    futures::FutureExt::now_or_never(store.update_layer(
                        &theirs.session_id,
                        LAYER,
                        theirs.oid,
                        Some(unfenced),
                        unfenced_relay,
                        CLIENT_ID,
                        1_000,
                    ))
                })
                .await
            };
            let (dropped, captured_dropped) = until_test_deadline(deadline, polled).await;
            assert!(
                dropped.is_none(),
                "precondition: one poll on a current-thread runtime cannot finish a call that \
                 waits on the store, so it is dropped mid-flight; got {dropped:?}"
            );
            assert!(
                captured_dropped
                    .with_message(FENCE_WAIT_EVENT_MSG)
                    .is_empty(),
                "a call that never met a fence and was dropped mid-call logs no fence-wait \
                 event:\n{}",
                captured_dropped.text
            );
        })
    })
    .await;

    until_test_deadline(deadline, async {
        let store = wx.store(UpdateLayerBackend::Lua, 1);
        let (committed, captured_committed) =
            capture_debug(wx.commit(&store, delta(Shape::FieldOnly, 2))).await;
        committed.unwrap_or_else(|e| panic!("an unfenced commit must succeed; got {e:?}"));

        assert!(
            captured_committed
                .with_message(FENCE_WAIT_EVENT_MSG)
                .is_empty(),
            "a call that never met a fence and returned a commit logs no fence-wait event:\n{}",
            captured_committed.text
        );
        assert_eq!(
            wx.fence_waits("lua").0,
            0,
            "no fence-wait observation for calls that never met a fence"
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. Path parity on a wrong-typed fence, intent key or commit release record
//    (SDD §7.5)
// ═════════════════════════════════════════════════════════════════════════════

/// Plant a string on the fence key and/or the intent key (`plant`), commit on
/// `path`, and return the refusal with the two key names replaced by
/// `<fence>` / `<intent>` (they carry the per-fixture prefix and object id).
/// Asserts nothing committed and the planted keys are left as they were.
async fn wrongtype_refusal(path: Path, plant: [bool; 2]) -> String {
    let wx = Wx::new().await;
    wx.prepare(path).await;
    let planted: Vec<String> = [wx.fence_key(), wx.intent_key()]
        .into_iter()
        .zip(plant)
        .filter(|(_, p)| *p)
        .map(|(k, _)| k)
        .collect();
    for key in &planted {
        let _: () = wx
            .tv
            .client()
            .pool
            .set(key, "not-a-hash", None, None, false)
            .await
            .expect("planting a string must succeed");
    }
    let before = (
        wx.fields(&wx.header_key()).await,
        wx.stream_len().await,
        wx.fields(&wx.wtok_key()).await,
    );

    let outcome = wx
        .commit(&wx.store_for(path, 8), delta(path.shape(), 1))
        .await;

    let what = format!("{path:?} / planted {planted:?}");
    assert_eq!(
        (
            wx.fields(&wx.header_key()).await,
            wx.stream_len().await,
            wx.fields(&wx.wtok_key()).await,
        ),
        before,
        "{what}: nothing may commit"
    );
    for key in &planted {
        let still: String = wx.tv.client().pool.get(key).await.unwrap();
        assert_eq!(still, "not-a-hash", "{what}: the key is left as found");
    }
    match outcome {
        Err(ServiceError::Internal(msg)) => msg
            .replace(&wx.fence_key(), "<fence>")
            .replace(&wx.intent_key(), "<intent>"),
        other => panic!("{what}: a wrong-typed key must refuse the commit Internal; got {other:?}"),
    }
}

/// A wrong-typed fence or intent key is refused with ONE refusal, byte for
/// byte, on every path — the update script's two modes (fast; structural on a
/// cached object, on a cold start, and for a reset) and the WATCH path (the
/// knob, `txn` field-only and structural, the uncacheable fallback) — the
/// fence named first when both are wrong, closing both halves of SDD §7.5's
/// parity (decider ruling of 2026-10-03).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_wrong_typed_fence_or_intent_key_is_refused_identically_on_every_path() {
    within_test_budget(async {
        for (plant, named) in [
            ([true, false], "<fence>"),
            ([false, true], "<intent>"),
            ([true, true], "<fence>"),
        ] {
            let expected = format!(
                "update_layer keyspace type error: WRONGTYPE_KEY: {named} holds a string, expected \
                 hash"
            );
            for path in Path::ALL {
                assert_eq!(
                    wrongtype_refusal(path, plant).await,
                    expected,
                    "{path:?}, planted [fence, intent] = {plant:?}: every path gives the fast \
                     path's refusal, byte for byte"
                );
            }
        }
    })
    .await;
}

// ── A wrong-typed commit release record under a live fence (b.yuo) ──────────
//
// The commit release record `le:{oid}:obj:wrel` is TYPE-checked after the
// fence on both backends: the update script's step 1e after its step-1d fence
// check, the WATCH path's step 2c after its step-2b fence wait. So a
// wrong-typed record on a fenced object is not refused at once: the writer
// waits the fence out like any other, writing nothing, and only then is
// refused `WRONGTYPE_KEY`, with one text on every path — or, when the fence
// outlasts the wait cap, answers the cap's retryable `UNAVAILABLE`, the same
// on both backends.

/// Plant a string on `path`'s object's commit release record under a held
/// fence and commit on `path` with a one-attempt budget. The writer must wait
/// the fence out — two fence-checking attempts seen in the `MONITOR` feed,
/// each writing nothing, the call still running — and once the reader
/// releases (verifying no commit landed under the fence) it must be refused
/// `WRONGTYPE_KEY` naming the record, after one recorded fence wait and no
/// timeout. Returns the refusal with the record's key replaced by `<wrel>`.
async fn wrong_typed_release_record_waits_out_the_fence(path: Path) -> String {
    let wx = Wx::new().await;
    wx.warm_scripts().await;
    wx.prepare(path).await;
    let wrel_key = wx.plant_string_wrel().await;
    let store = wx.store_for(path, 1);
    let lease = wx.hold_fence().await;
    let before = wx.image().await;
    let what = format!("{path:?} / wrong-typed obj:wrel");
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    let writer = wx.spawn_commit(&store, delta(path.shape(), 1));
    wx.await_two_fenced_attempts(&mut tap, path, &before, &what)
        .await;
    assert!(
        !writer.is_finished(),
        "{what}: the writer waits the fence out rather than refusing the record at once"
    );

    wx.release(lease).await;
    let outcome = join(writer).await;
    assert_eq!(
        wx.image().await,
        before,
        "{what}: nothing may commit after the fence either"
    );
    let still: String = wx.tv.client().pool.get(&wrel_key).await.unwrap();
    assert_eq!(still, "not-a-hash", "{what}: the key is left as found");
    assert_eq!(
        wx.fence_waits(path.fence_backend()).0,
        1,
        "{what}: the call met the fence and recorded one wait"
    );
    assert_eq!(wx.fence_timeouts(), 0, "{what}");
    assert!(
        wx.intents().await.is_empty(),
        "{what}: no intent outlives the call"
    );
    match outcome {
        Err(ServiceError::Internal(msg)) => msg.replace(&wrel_key, "<wrel>"),
        other => panic!("{what}: the record must refuse the commit Internal; got {other:?}"),
    }
}

/// The refusal of a wrong-typed release record once the fence has ended,
/// exactly, on every path: the update script's step 1e and the WATCH path's
/// step 2c give the same text, byte for byte, the key's qualifier
/// ` (commit release record)` included.
const WREL_WRONGTYPE_REFUSAL: &str = "update_layer keyspace type error: WRONGTYPE_KEY: <wrel> \
                                      holds a string, expected hash (commit release record)";

async fn assert_wrong_typed_release_record_waits_out_the_fence(path: Path) {
    let refusal = wrong_typed_release_record_waits_out_the_fence(path).await;
    assert_eq!(
        refusal, WREL_WRONGTYPE_REFUSAL,
        "{path:?}: after the fence, the wrong-typed release record is refused WRONGTYPE_KEY, \
         with the one text every path gives"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_wrong_typed_release_record_waits_out_the_fence_on_the_fast_path() {
    within_test_budget(assert_wrong_typed_release_record_waits_out_the_fence(
        Path::Fast,
    ))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_wrong_typed_release_record_waits_out_the_fence_on_the_cached_structural_route() {
    within_test_budget(assert_wrong_typed_release_record_waits_out_the_fence(
        Path::Structural,
    ))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_wrong_typed_release_record_waits_out_the_fence_on_the_txn_field_only_path() {
    within_test_budget(assert_wrong_typed_release_record_waits_out_the_fence(
        Path::TxnFieldOnly,
    ))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_wrong_typed_release_record_waits_out_the_fence_on_the_txn_structural_path() {
    within_test_budget(assert_wrong_typed_release_record_waits_out_the_fence(
        Path::TxnStructural,
    ))
    .await;
}

/// A wrong-typed release record under a fence that outlasts the wait cap
/// (1.5 × `poll`, as in [`capped`]): the call answers the cap's retryable
/// `UNAVAILABLE` with its `SNAPSHOT_FENCE_TIMEOUT` retry hint — "nothing was
/// written, retry" — not `WRONGTYPE_KEY`, writes nothing, and ticks the
/// timeout counter once, on `path` as on any other.
async fn wrong_typed_release_record_under_a_fence_past_the_cap(path: Path) {
    let wx = Wx::new().await;
    wx.warm_scripts().await;
    wx.prepare(path).await;
    let wrel_key = wx.plant_string_wrel().await;
    let cap = wx.poll() * 3 / 2;
    let store = wx
        .store_for(path, 1)
        .with_fence_wait_deadline(cap, Arc::new(|| None));
    let _lease = wx.hold_fence().await;
    let before = wx.image().await;
    let what = format!("{path:?} / wrong-typed obj:wrel past the cap");

    let outcome = wx.commit(&store, delta(path.shape(), 1)).await;
    let err = outcome.as_ref().err().unwrap_or_else(|| {
        panic!(
            "{what}: a fence past the cap fails the call UNAVAILABLE (retryable) before the \
             record is probed; got {outcome:?}"
        )
    });
    let message = assert_refusal_error(
        err,
        refusal_reason::SNAPSHOT_FENCE_TIMEOUT,
        &[],
        CAPACITY_RETRY_DELAY,
    );
    assert_cap_message(message, cap, wx.oid, path.first_label(), &what);
    assert_eq!(wx.image().await, before, "{what}: nothing written");
    assert_eq!(wx.fence_timeouts(), 1, "{what}");
    let still: String = wx.tv.client().pool.get(&wrel_key).await.unwrap();
    assert_eq!(still, "not-a-hash", "{what}: the key is left as found");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_wrong_typed_release_record_under_a_fence_past_the_cap_fails_the_fast_path_unavailable() {
    within_test_budget(wrong_typed_release_record_under_a_fence_past_the_cap(
        Path::Fast,
    ))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_wrong_typed_release_record_under_a_fence_past_the_cap_fails_the_txn_path_unavailable() {
    within_test_budget(wrong_typed_release_record_under_a_fence_past_the_cap(
        Path::TxnFieldOnly,
    ))
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 6. Through the commit seams (`test-fault-injection`)
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod seams {
    //! Scenarios only a commit-window seam can stage: a write between the
    //! WATCH path's `WATCH` and its `EXEC` (`with_pre_dispatch_hook_for_test`,
    //! after the probes, before the pipeline), a call's fixed commit token
    //! (`with_commit_tokens_for_test`), and an attempt dropped mid-flight.
    //! Compiled out without `--features test-fault-injection`.

    use super::*;

    use std::sync::Mutex;

    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use fred::interfaces::ClientLike;
    use futures::FutureExt;
    use live_edit::storage::layer::{
        operation_request_digest, Operation, StartRecord, OPERATION_OVERTAKEN_MESSAGE,
    };
    use live_edit::storage::snapshot_fence::FenceProbe;
    use live_edit::storage::watched_txn::WatchWindowHook;
    use tokio::sync::Notify;

    /// A pre-dispatch hook running `body` with the 1-based count of its own
    /// firings — one per attempt, or fence-wait re-run of one, that reaches
    /// dispatch — and that count, for the test to read.
    fn hook<F, Fut>(body: F) -> (WatchWindowHook, Arc<AtomicUsize>)
    where
        F: Fn(usize) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let body = Arc::new(body);
        let hook: WatchWindowHook = Arc::new(move |_conn| {
            let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
            let body = Arc::clone(&body);
            async move { body(n).await }.boxed()
        });
        (hook, calls)
    }

    /// A competing fast-path commit on the same object from another store —
    /// it bumps the `WATCH`ed header — run as its own task so no
    /// `update_layer` is polled inside another.
    async fn compete(store: LayerStore, session_id: String, oid: u64, n: u64) {
        tokio::spawn(async move {
            let competing = delta(Shape::FieldOnly, 50 + n);
            let competing_relay = relay(&competing);
            store
                .update_layer(
                    &session_id,
                    LAYER,
                    oid,
                    Some(competing),
                    competing_relay,
                    "competitor",
                    1,
                )
                .await
        })
        .await
        .expect("the competitor must not panic")
        .expect("the competing commit must commit");
    }

    /// The message of the slow path's per-conflict DEBUG event
    /// (`LayerStore::classify_exec_conflict`), spelled out: the validation
    /// tooling greps for it and for its `verdict` words, so a rename must fail
    /// here.
    const EXEC_CONFLICT_EVENT_MSG: &str = "update_layer exec conflict";

    /// The single `update_layer exec conflict` event in `captured`: DEBUG, on
    /// `storage::layer`'s target, for this fixture's object.
    fn only_exec_conflict_event<'c>(wx: &Wx, captured: &'c Captured) -> &'c Value {
        let event = captured.only(EXEC_CONFLICT_EVENT_MSG);
        assert_eq!(event["level"], "DEBUG", "{event}");
        assert_eq!(event["target"], "live_edit::storage::layer", "{event}");
        assert_eq!(event["fields"]["object_id"], wx.oid, "{event}");
        event
    }

    /// A fence granted after the attempt's `WATCH` and probe — from the
    /// pre-dispatch hook, between the probe and the `EXEC` — fails the `EXEC`;
    /// the re-probe finds the live fence (verdict `fence_live`, the live
    /// lease's remaining time as the hint) and the attempt waits, uncounted,
    /// on a one-attempt budget, and commits once the reader releases (whose
    /// `VERIFIED` shows the failed `EXEC` wrote nothing). The fence key in the
    /// `WATCH` set is what fails that `EXEC` (B6): nothing else the hook
    /// writes is watched.
    async fn fence_between_watch_and_exec(path: Path) {
        let wx = Arc::new(Wx::new().await);
        wx.warm_scripts().await;
        wx.prepare(path).await;
        let seq_before = wx.seq().await;
        let held: Arc<Mutex<Option<Box<FenceLease>>>> = Arc::new(Mutex::new(None));
        let (pre_dispatch, calls) = {
            let wx = Arc::clone(&wx);
            let held = Arc::clone(&held);
            hook(move |n| {
                let wx = Arc::clone(&wx);
                let held = Arc::clone(&held);
                async move {
                    if n == 1 {
                        let lease = wx.hold_fence().await;
                        *held.lock().unwrap() = Some(lease);
                    }
                }
            })
        };
        let store = wx
            .store_for(path, 1)
            .with_pre_dispatch_hook_for_test(pre_dispatch);
        let what = format!("{path:?}");
        let mut tap = MonitorTap::start(&test_valkey_addr()).await;

        // The call and the release run in one captured future, so the call's
        // DEBUG events are recorded.
        let (outcome, captured) = capture_debug(async {
            let release_after_the_reprobe = async {
                // The attempt's own probe, then the re-probe after the failed
                // EXEC.
                tap.drain_until_nth(
                    2,
                    "the attempt's probe and the conflict re-probe",
                    wx.is_probe_eval(),
                )
                .await;
                let lease = held
                    .lock()
                    .unwrap()
                    .take()
                    .expect("the hook granted the fence");
                wx.release(lease).await;
            };
            let (outcome, ()) = tokio::join!(
                wx.commit(&store, delta(path.shape(), 1)),
                release_after_the_reprobe
            );
            outcome
        })
        .await;
        let (seq, _) =
            outcome.unwrap_or_else(|e| panic!("{what}: must commit after the release; got {e:?}"));

        assert_eq!(seq, seq_before + 1, "{what}");
        assert!(
            calls.load(Ordering::SeqCst) >= 2,
            "{what}: the attempt reached dispatch again after the wait"
        );
        wx.assert_one_attempt(&what);
        assert_eq!(wx.fence_waits("txn").0, 1, "{what}");
        let event = only_exec_conflict_event(&wx, &captured);
        assert_eq!(event["fields"]["verdict"], "fence_live", "{what}: {event}");
        assert!(
            event["fields"]["retry_after_ms"]
                .as_u64()
                .is_some_and(|ms| ms > 0),
            "{what}: a live lease's hint is its remaining time; event: {event}"
        );
    }

    /// How long the reader's epoch in [`epoch_between_probe_and_exec`] holds
    /// the fence: past any one round trip, so a recorded span that starts
    /// after the failed `EXEC` (only the jitter-length sleep) cannot reach it.
    const EPOCH: Duration = Duration::from_millis(200);

    /// The g6 shape (design revision 5 § A): a reader's whole epoch — grant,
    /// a 200 ms hold, release — lands between the attempt's probe and its
    /// `EXEC` (the pre-dispatch hook's first firing), so the `EXEC` fails on
    /// the `WATCH`ed fence key while the re-probe finds no live lease. The
    /// fence key changed since the attempt's own probe, so the conflict is
    /// the fence's: an uncounted wait (verdict `fence_changed`, a zero hint)
    /// on a one-attempt budget, and the re-run commits on attempt 1. Its one
    /// `txn` fence-wait observation spans the attempt the epoch failed —
    /// started before the hook's hold — so it is no shorter than [`EPOCH`].
    ///
    /// `lingering`: the key is left from an earlier epoch (present, not live,
    /// its gap and intent window long past, as at g6's 4× and 200k runs) and
    /// the change is its marker; otherwise the layer was never fenced and the
    /// change is `Absent` → `Present`.
    async fn epoch_between_probe_and_exec(path: Path, lingering: bool) {
        let wx = Arc::new(Wx::new().await);
        wx.warm_scripts().await;
        wx.prepare(path).await;
        if lingering {
            wx.fence_and_release().await;
            // The gap and the intent window ended a minute ago: the attempt
            // starts outside both, so the reader's grant does not wait for it.
            let past = wx.now().await - 60_000;
            let pool = &wx.tv.client().pool;
            let _: () = pool
                .hset(wx.fence_key(), ("gap_until", past.to_string()))
                .await
                .expect("staging gap_until must succeed");
            let _: i64 = pool
                .pexpire(wx.fence_key(), FAR_MS, None)
                .await
                .expect("staging the key expiry must succeed");
        }
        let watched = fence::probe_fence(&wx.client, wx.tv.key_schema(), wx.oid)
            .await
            .expect("the precondition probe");
        assert_eq!(
            matches!(
                watched,
                FenceProbe::Present {
                    live_retry_after: None,
                    ..
                }
            ),
            lingering,
            "precondition: the key lingers, not live, exactly when staged so; got {watched:?}"
        );
        let seq_before = wx.seq().await;
        let (pre_dispatch, calls) = {
            let wx = Arc::clone(&wx);
            hook(move |n| {
                let wx = Arc::clone(&wx);
                async move {
                    if n == 1 {
                        let lease = wx.grant().await;
                        tokio::time::sleep(EPOCH).await;
                        wx.release(lease).await;
                    }
                }
            })
        };
        let store = wx
            .store_for(path, 1)
            .with_pre_dispatch_hook_for_test(pre_dispatch);
        let what = format!(
            "{path:?} / {}",
            if lingering {
                "lingering key"
            } else {
                "never fenced"
            }
        );

        let (outcome, captured) = capture_debug(wx.commit(&store, delta(path.shape(), 1))).await;

        let (seq, _) = outcome.unwrap_or_else(|e| {
            panic!(
                "{what}: an epoch that came and went between the attempt's probe and its EXEC is \
                 the fence's, an uncounted wait, so the call commits on its one attempt; got {e:?}"
            )
        });
        assert_eq!(seq, seq_before + 1, "{what}");
        assert!(
            calls.load(Ordering::SeqCst) >= 2,
            "{what}: the attempt reached dispatch again after the wait"
        );
        wx.assert_one_attempt(&what);
        let event = only_exec_conflict_event(&wx, &captured);
        assert_eq!(
            event["fields"]["verdict"], "fence_changed",
            "{what}: {event}"
        );
        assert_eq!(
            event["fields"]["retry_after_ms"], 0,
            "{what}: no lease is live, so the wait is the jitter alone; event: {event}"
        );
        let (count, sum) = wx.fence_waits("txn");
        assert_eq!(count, 1, "{what}: one fence-wait observation for the call");
        let waited = Duration::from_secs_f64(sum);
        assert!(
            waited + Duration::from_micros(1) >= EPOCH,
            "{what}: the span starts at the attempt the epoch failed, which held the reader's \
             {EPOCH:?} epoch; got {waited:?}"
        );
        let wait_event = only_fence_wait_event(&wx, &captured, "txn");
        assert_fence_wait_event(wait_event, true, false, 1, &what);
        assert_event_matches_histogram(wait_event, sum, &what);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn an_epoch_between_probe_and_exec_on_a_lingering_key_is_an_uncounted_wait() {
        within_test_budget(epoch_between_probe_and_exec(Path::WatchKnob, true)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn an_epoch_between_probe_and_exec_on_a_never_fenced_layer_is_an_uncounted_wait() {
        within_test_budget(epoch_between_probe_and_exec(Path::TxnFieldOnly, false)).await;
    }

    /// The other side of the marker rule: a peer's commit between the probe
    /// and the `EXEC` on a layer whose fence key lingers inside a gap (staged
    /// [`FAR_MS`] ahead, so the whole attempt runs in it) leaves the key
    /// unchanged, so the conflict stays counted — verdict `conflict`, the
    /// call commits on attempt 2 with one retry and no fence wait. A rule
    /// that took "a gap in progress" for the fence would make it uncounted.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_peer_conflict_on_a_layer_inside_a_gap_stays_counted() {
        within_test_budget(async {
            let wx = Arc::new(Wx::new().await);
            wx.warm_scripts().await;
            wx.fence_and_release().await;
            let far = wx.now().await + FAR_MS;
            let pool = &wx.tv.client().pool;
            let _: () = pool
                .hset(wx.fence_key(), ("gap_until", far.to_string()))
                .await
                .expect("staging gap_until must succeed");
            let _: i64 = pool
                .pexpire(wx.fence_key(), FAR_MS, None)
                .await
                .expect("staging the key expiry must succeed");
            let fence_before = wx.fields(&wx.fence_key()).await;
            let (pre_dispatch, calls) = {
                let wx = Arc::clone(&wx);
                let competitor = wx.bare_store();
                hook(move |n| {
                    let wx = Arc::clone(&wx);
                    let competitor = competitor.clone();
                    async move {
                        if n == 1 {
                            compete(competitor, wx.session_id.clone(), wx.oid, n as u64).await;
                        }
                    }
                })
            };
            let store = wx
                .store_for(Path::WatchKnob, 2)
                .with_pre_dispatch_hook_for_test(pre_dispatch);

            let (outcome, captured) =
                capture_debug(wx.commit(&store, delta(Path::WatchKnob.shape(), 1))).await;

            outcome.unwrap_or_else(|e| panic!("the second attempt must commit; got {e:?}"));
            assert_eq!(
                calls.load(Ordering::SeqCst),
                2,
                "two attempts reached dispatch"
            );
            assert_eq!(
                wx.fields(&wx.fence_key()).await,
                fence_before,
                "precondition: the peer's commit left the fence key as it was"
            );
            let event = only_exec_conflict_event(&wx, &captured);
            assert_eq!(event["fields"]["verdict"], "conflict", "{event}");
            assert_eq!(event["fields"]["retry_after_ms"], 0, "{event}");
            // Every read names the committing attempt's `path` (the WATCH
            // path's `watch`). `retries_total` is zero-seeded over every
            // `path` × `outcome` cell, and `metric` returns the first series
            // its labels match in exposition order, which follows the SDK's
            // per-process randomised attribute-set map: an `outcome`-only read
            // lands on an arbitrary path's seeded zero.
            assert_eq!(
                (
                    wx.metric(
                        "live_edit_update_layer_attempts_histogram_count",
                        &[("path", "watch")]
                    ),
                    wx.metric(
                        "live_edit_update_layer_attempts_histogram_sum",
                        &[("path", "watch")]
                    ),
                ),
                (1.0, 2.0),
                "one commit, on attempt 2: the peer's conflict spent an attempt"
            );
            assert_eq!(
                wx.metric(
                    "live_edit_update_layer_retries_total",
                    &[("path", "watch"), ("outcome", "committed")]
                ),
                1.0,
                "the counted conflict ticks the retry counter once"
            );
            assert_eq!(wx.fence_waits("txn").0, 0, "no fence wait");
            assert!(
                captured.with_message(FENCE_WAIT_EVENT_MSG).is_empty(),
                "a call that never waited on a fence logs no fence-wait event:\n{}",
                captured.text
            );
        })
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_fence_granted_between_watch_and_exec_is_an_uncounted_wait_for_a_txn_field_commit() {
        within_test_budget(fence_between_watch_and_exec(Path::TxnFieldOnly)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_fence_granted_between_watch_and_exec_is_an_uncounted_wait_for_a_watch_knob_structural_commit(
    ) {
        within_test_budget(fence_between_watch_and_exec(Path::WatchKnob)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_fence_granted_between_watch_and_exec_is_an_uncounted_wait_for_an_uncacheable_fallback_commit(
    ) {
        within_test_budget(fence_between_watch_and_exec(Path::UncacheableFallback)).await;
    }

    /// An attempt aborted by an ordinary conflict clears its intent before
    /// the next attempt registers its own: at the second attempt's dispatch
    /// exactly one intent is live. The first attempt registered at step 2b
    /// (its probe found the lingering key); the second, classified by then,
    /// registered before its `WATCH`.
    ///
    /// The lingering key stores [`UNLAPSING_SIZE`], so each intent's
    /// `intent_ttl(L)` exceeds [`TEST_BUDGET`]: the count at each dispatch is
    /// of intents registered and not cleared, never of one that lapsed while
    /// its renewals were late on a loaded store. Renewal is not this test's
    /// subject.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn an_aborted_attempt_clears_its_intent_before_the_next_registers() {
        within_test_budget(async {
            let wx = Arc::new(Wx::new().await);
            wx.warm_scripts().await;
            assert!(
                wx.intent_ttl(UNLAPSING_SIZE) > TEST_BUDGET,
                "fixture precondition: no intent on the layer can lapse inside the test"
            );
            wx.fence_and_release_at(UNLAPSING_SIZE).await;
            assert_eq!(
                wx.stored_size().await,
                UNLAPSING_SIZE,
                "fixture precondition: the intents' ttl comes from the lingering key's stored size"
            );
            let live_intents: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
            let (pre_dispatch, _) = {
                let wx = Arc::clone(&wx);
                let live_intents = Arc::clone(&live_intents);
                let competitor = wx.bare_store();
                hook(move |n| {
                    let wx = Arc::clone(&wx);
                    let live_intents = Arc::clone(&live_intents);
                    let competitor = competitor.clone();
                    async move {
                        let live = wx.intents().await.len();
                        live_intents.lock().unwrap().push(live);
                        if n == 1 {
                            compete(competitor, wx.session_id.clone(), wx.oid, n as u64).await;
                        }
                    }
                })
            };
            let store = wx
                .store_for(Path::WatchKnob, 2)
                .with_pre_dispatch_hook_for_test(pre_dispatch);

            wx.commit(&store, delta(Shape::Structural, 1))
                .await
                .expect("the second attempt must commit");

            assert_eq!(
                *live_intents.lock().unwrap(),
                vec![1, 1],
                "each attempt holds its own intent at dispatch, and only its own: the aborted \
                 attempt's was cleared"
            );
            assert_eq!(wx.intent_registrations("registered"), 2);
            assert!(wx.intents().await.is_empty(), "cleared on commit");
        })
        .await;
    }

    /// An attempt dropped mid-flight (its task aborted while it holds an
    /// intent) leaves the intent's clean-up to the renewal task, which still
    /// removes it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_dropped_attempt_still_clears_its_intent() {
        within_test_budget(async {
            let wx = Arc::new(Wx::new().await);
            wx.warm_scripts().await;
            wx.fence_and_release().await;
            let reached = Arc::new(Notify::new());
            let (pre_dispatch, _) = {
                let reached = Arc::clone(&reached);
                hook(move |_| {
                    let reached = Arc::clone(&reached);
                    async move {
                        reached.notify_one();
                        std::future::pending::<()>().await;
                    }
                })
            };
            let store = wx
                .store_for(Path::WatchKnob, 1)
                .with_pre_dispatch_hook_for_test(pre_dispatch);

            let writer = wx.spawn_commit(&store, delta(Shape::Structural, 1));
            tokio::time::timeout(CALL_BUDGET, reached.notified())
                .await
                .expect("the attempt must reach dispatch");
            assert_eq!(
                wx.intents().await.len(),
                1,
                "precondition: the attempt holds a live intent"
            );

            writer.abort();
            let _ = writer.await;

            let wx = &wx;
            tokio::time::timeout(DROP_POLL_BUDGET, async move {
                while !wx.intents().await.is_empty() {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .expect("the dropped attempt's intent must be cleared");
        })
        .await;
    }

    /// Bound on waiting for a dropped intent's background `HDEL`.
    const DROP_POLL_BUDGET: Duration = Duration::from_secs(10);

    /// The per-command timeout `client`'s data pool runs with (fred's
    /// `PerformanceConfig::default_command_timeout`, set from
    /// `valkey_command_timeout_ms` at connect): the longest one renewal on
    /// that client can take to reach the store and answer.
    fn command_timeout(client: &ValkeyClient) -> Duration {
        client.pool.next().perf_config().default_command_timeout
    }

    /// A stored size whose `intent_ttl(L)` (about 8.7 s on the recorded
    /// coefficients) gives a renewal sent at its first period,
    /// `intent_ttl/4`, `0.75·intent_ttl` (about 6.5 s) to land before the
    /// deadline it extends: more than the writer's [`command_timeout`] (5 s
    /// on the fixture), so a renewal that reaches the store at all reaches it
    /// before that deadline.
    const RENEWED_SIZE: LayerSize = LayerSize {
        items: 1_500_000,
        bytes: 18_000_000,
    };

    /// A stored size whose `intent_ttl(L)` (about 116 s on the recorded
    /// coefficients) exceeds [`TEST_BUDGET`]: no intent registered on it can
    /// lapse inside a test.
    const UNLAPSING_SIZE: LayerSize = LayerSize {
        items: 20_000_000,
        bytes: 240_000_000,
    };

    /// A structural WATCH-path writer on a layer recently fenced by a reader
    /// that stored `size` (so its first attempt registers an intent at step
    /// 2b, its `intent_ttl(L)` from `size`), parked at its pre-dispatch hook —
    /// holding its intent, renewal running — until the returned `Notify` lets
    /// it go on to commit.
    async fn park_registered_attempt(
        wx: &Arc<Wx>,
        size: LayerSize,
    ) -> (
        tokio::task::JoinHandle<Result<(u64, u64), ServiceError>>,
        Arc<Notify>,
    ) {
        wx.fence_and_release_at(size).await;
        let reached = Arc::new(Notify::new());
        let proceed = Arc::new(Notify::new());
        let (pre_dispatch, _) = {
            let reached = Arc::clone(&reached);
            let proceed = Arc::clone(&proceed);
            hook(move |_| {
                let reached = Arc::clone(&reached);
                let proceed = Arc::clone(&proceed);
                async move {
                    reached.notify_one();
                    proceed.notified().await;
                }
            })
        };
        let store = wx
            .store_for(Path::WatchKnob, 1)
            .with_pre_dispatch_hook_for_test(pre_dispatch);
        let writer = wx.spawn_commit(&store, delta(Shape::Structural, 1));
        tokio::time::timeout(CALL_BUDGET, reached.notified())
            .await
            .expect("the attempt must reach dispatch");
        (writer, proceed)
    }

    /// The single `w:` field of the parked attempt's intent, and its deadline.
    async fn only_intent(wx: &Wx) -> (String, i64) {
        let intents = wx.intents().await;
        assert_eq!(intents.len(), 1, "one live intent: {intents:?}");
        let (field, deadline) = intents.into_iter().next().expect("one intent");
        (field, deadline.parse().expect("a deadline"))
    }

    /// A held attempt's observed renewal extends its intent's deadline past
    /// the deadline it registered with: the renewal task renews the intent
    /// every `intent_ttl/4`, and its first renewal — waited for in the
    /// store's `MONITOR` feed, never on a clock — leaves the same field with a
    /// later deadline. The intent's `intent_ttl(L)` is sized
    /// ([`RENEWED_SIZE`]) so a renewal delayed up to the client's command
    /// timeout still lands before the deadline it extends.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_held_attempt_keeps_renewing_its_intent_past_its_first_deadline() {
        within_test_budget(async {
            let wx = Arc::new(Wx::new().await);
            wx.warm_scripts().await;
            let ttl = wx.intent_ttl(RENEWED_SIZE);
            let command_timeout = command_timeout(&wx.client);
            assert!(
                ttl.mul_f64(0.75) > command_timeout,
                "fixture precondition: a renewal sent at intent_ttl/4 has more than the writer's \
                 command timeout ({command_timeout:?}) to land before the deadline it extends \
                 (intent_ttl {ttl:?})"
            );
            let mut tap = MonitorTap::start(&test_valkey_addr()).await;
            let (writer, proceed) = park_registered_attempt(&wx, RENEWED_SIZE).await;
            assert_eq!(
                wx.stored_size().await,
                RENEWED_SIZE,
                "fixture precondition: the intent's ttl comes from the lingering key's stored size"
            );
            let (field, registered_deadline) = only_intent(&wx).await;

            tap.drain_until_nth(1, "the held attempt's first renewal", wx.is_renew_eval())
                .await;

            let (renewed_field, renewed_deadline) = only_intent(&wx).await;
            assert_eq!(renewed_field, field, "the same intent, renewed in place");
            assert!(
                renewed_deadline > registered_deadline,
                "the held attempt's renewal must extend its intent past its first deadline \
                 ({registered_deadline}): deadline {renewed_deadline}"
            );

            proceed.notify_one();
            join(writer).await.expect("the released attempt commits");
            assert!(wx.intents().await.is_empty(), "cleared on commit");
        })
        .await;
    }

    /// A renewal never revives an intent that is gone: with the field removed
    /// under a held attempt, the next renewal (`LOST`) creates nothing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_renewal_after_the_intent_was_removed_does_not_recreate_it() {
        within_test_budget(async {
            let wx = Arc::new(Wx::new().await);
            wx.warm_scripts().await;
            let (writer, proceed) = park_registered_attempt(&wx, SIZE).await;
            let (field, _) = only_intent(&wx).await;
            let mut tap = MonitorTap::start(&test_valkey_addr()).await;

            let _: i64 = wx
                .tv
                .client()
                .pool
                .hdel(wx.intent_key(), field.clone())
                .await
                .expect("removing the intent field must succeed");
            tap.drain_until_nth(1, "the removal's HDEL", Wx::names(wx.intent_key(), "HDEL"))
                .await;
            tap.drain_until_nth(1, "a renewal after the removal", wx.is_renew_eval())
                .await;

            assert!(
                wx.intents().await.is_empty(),
                "the renewal after the removal must not recreate the intent"
            );

            proceed.notify_one();
            join(writer).await.expect("the released attempt commits");
            assert!(wx.intents().await.is_empty());
        })
        .await;
    }

    /// The tallies classification point: a structural WATCH-path attempt
    /// whose snapshot read tallied past the atomic-read ceilings (a client
    /// with a 1-item ceiling; no fence key anywhere) aborts on an ordinary
    /// conflict, and the call's next attempt registers its intent BEFORE its
    /// `WATCH`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn an_attempt_after_one_that_tallied_past_the_ceiling_registers_before_watch() {
        within_test_budget(async {
            let wx = Arc::new(Wx::with_limits(1, 1).await);
            wx.warm_scripts().await;
            // The warm-up's structural commit cold-starts the object through
            // the structural cache, which above the ceilings (1 item here)
            // takes the fenced read: its fence key lingers. This test is about
            // a layer the fence never touched, so remove that key; nothing
            // holds a lease or an intent once the warm-up has returned.
            let _: i64 = wx
                .tv
                .client()
                .pool
                .del(wx.fence_key())
                .await
                .expect("DEL the warm-up's fence key");
            let (pre_dispatch, _) = {
                let wx = Arc::clone(&wx);
                let competitor = wx.bare_store();
                hook(move |n| {
                    let wx = Arc::clone(&wx);
                    let competitor = competitor.clone();
                    async move {
                        if n == 1 {
                            compete(competitor, wx.session_id.clone(), wx.oid, n as u64).await;
                        }
                    }
                })
            };
            let store = wx
                .store_for(Path::WatchKnob, 2)
                .with_pre_dispatch_hook_for_test(pre_dispatch);
            assert!(
                !wx.tv
                    .client()
                    .pool
                    .exists::<bool, _>(wx.fence_key())
                    .await
                    .unwrap(),
                "precondition: the layer has never been fenced"
            );
            let mut tap = MonitorTap::start(&test_valkey_addr()).await;

            wx.commit(&store, delta(Shape::Structural, 1))
                .await
                .expect("the second attempt must commit");
            let seen = tap
                .drain_until_nth(1, "the intent's HDEL", Wx::names(wx.intent_key(), "HDEL"))
                .await;

            let header = wx.header_key();
            let watches: Vec<usize> = seen
                .iter()
                .enumerate()
                .filter(|(_, c)| {
                    !c.issued_by_script() && c.name() == "WATCH" && c.args.get(1) == Some(&header)
                })
                .map(|(i, _)| i)
                .collect();
            let registers: Vec<usize> = seen
                .iter()
                .enumerate()
                .filter(|(_, c)| wx.is_register_eval()(c))
                .map(|(i, _)| i)
                .collect();
            assert_eq!(watches.len(), 2, "two attempts: {watches:?}");
            assert_eq!(
                registers.len(),
                1,
                "only the second attempt registers; the first knew nothing yet"
            );
            assert!(
                watches[0] < registers[0] && registers[0] < watches[1],
                "the registration comes after the first attempt and before the second's WATCH: \
                 WATCHes {watches:?}, registration {registers:?}"
            );
            assert_eq!(wx.intent_registrations("registered"), 1);
            assert!(wx.intents().await.is_empty());
        })
        .await;
    }

    /// Review round 1 fix 1, the `ALREADY_LANDED` half: a call classified
    /// above the ceiling whose before-`WATCH` registration is refused
    /// `FENCED` falls through to its commit-token check, and a commit this
    /// call already landed is reported — under a fence that is still live,
    /// without waiting it out. The landed record is staged with the call's own
    /// (fixed) token while its first run sleeps; the cap is set so the
    /// pre-fix behaviour would end `UNAVAILABLE` instead.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn refused_registration_before_watch_still_reports_an_already_landed_commit() {
        within_test_budget(async {
            /// This call's fixed commit token.
            const TOKEN: u128 = 0x0b3a_8e3a_0000_0000_0000_0000_0000_0001;
            let wx = Wx::new().await;
            wx.warm_scripts().await;
            let poll = wx.poll();
            let store = wx
                .store_for(Path::WatchKnob, 1)
                .with_commit_tokens_for_test(Arc::new(Mutex::new(VecDeque::from([TOKEN]))))
                .with_fence_wait_deadline(poll * 5 / 2, Arc::new(|| None));
            let lease = wx.hold_fence().await;
            let mut tap = MonitorTap::start(&test_valkey_addr()).await;

            let writer = wx.spawn_commit(&store, delta(Shape::Structural, 1));
            tap.drain_until_nth(1, "the first run's probe", wx.is_probe_eval())
                .await;
            let landed_seq = wx.seq().await;
            let landed_ts = 1_700_000_000_000_000_u64;
            let _: () = wx
                .tv
                .client()
                .pool
                .hset(
                    wx.wtok_key(),
                    (CLIENT_ID, format!("{TOKEN:032x}:{landed_seq}:{landed_ts}")),
                )
                .await
                .expect("staging the landed record must succeed");
            let before = wx.image().await;
            let outcome = join(writer).await;

            assert_eq!(
                outcome.expect("the landed commit is reported, not waited on"),
                (landed_seq, landed_ts)
            );
            assert!(
                wx.intent_registrations("refused_fenced") >= 1,
                "vacuity guard: the before-WATCH registration ran and was refused"
            );
            assert_eq!(
                wx.metric(
                    "live_edit_update_layer_commit_already_landed_total",
                    &[("backend", "txn")]
                ),
                1.0
            );
            assert_eq!(wx.image().await, before, "nothing written");
            assert_eq!(wx.fence_timeouts(), 0);
            wx.release(lease).await;
        })
        .await;
    }

    /// A fenced call whose re-run finds a commit under its own (pinned) token
    /// — what an earlier attempt of it whose reply was lost leaves — reports
    /// that commit as its own and records its wait `committed` true, not
    /// cancelled: the own-commit counterpart of the three verdicts in section
    /// 4 that record `committed` false
    /// ([`super::a_fenced_call_ended_by_a_staged_record`]).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_fenced_call_finding_its_own_landed_commit_after_its_wait_records_its_wait_committed()
    {
        within_test_budget(a_fenced_call_ended_by_a_staged_record(
            FencedEnding::ThisCall,
        ))
        .await;
    }

    /// The answer a step-1b guard gives ahead of a live fence (T2, every
    /// path).
    #[derive(Debug, Clone, Copy)]
    enum Ahead {
        /// A commit this call already landed: reported, not waited on.
        AlreadyLanded,
        /// The object superseded: refused `FAILED_PRECONDITION` at once.
        Superseded,
        /// The call's operation overtaken (its `client_id`'s record moved
        /// since the call started): refused `UNAVAILABLE` with the overtaken
        /// message, not the fence-wait cap's.
        Overtaken,
    }

    /// T2 through `update_layer`, on every path: under a live fence the
    /// step-1b guards still answer first — own-key match (`ALREADY_LANDED`),
    /// the start record (`OVERTAKEN`), `SUPERSEDED` — on the update script's
    /// paths (both modes) and the WATCH path alike, so a commit this call
    /// already landed is reported and a call that can never apply is refused
    /// without waiting for the reader. The store's fence-wait cap is 1 ms, so
    /// a path that put the fence first would end `UNAVAILABLE` with the cap's
    /// message instead. Nothing is written in any case. The script-level rows
    /// (and `FENCED` ahead of `STALE` / `CACHE_DIVERGED`) are in
    /// `structural_guard_parity_test.rs`.
    async fn guard_ahead_of_a_live_fence(path: Path, guard: Ahead) {
        /// This call's fixed commit token.
        const TOKEN: u128 = 0x0b3a_8e3a_0000_0000_0000_0000_0000_0002;
        let wx = Wx::new().await;
        wx.prepare(path).await;
        let store = wx
            .store_for(path, 1)
            .with_commit_tokens_for_test(Arc::new(Mutex::new(VecDeque::from([TOKEN]))))
            .with_fence_wait_deadline(Duration::from_millis(1), Arc::new(|| None));
        let _lease = wx.hold_fence().await;
        let landed_seq = wx.seq().await;
        let landed_ts = 1_700_000_000_000_000_u64;
        let pool = &wx.tv.client().pool;
        let mut operation = None;
        match guard {
            Ahead::AlreadyLanded => {
                let _: () = pool
                    .hset(
                        wx.wtok_key(),
                        (CLIENT_ID, format!("{TOKEN:032x}:{landed_seq}:{landed_ts}")),
                    )
                    .await
                    .expect("staging the landed record must succeed");
            }
            Ahead::Superseded => {
                let _: () = pool
                    .hset(wx.header_key(), ("superseded", "1"))
                    .await
                    .expect("staging superseded=1 must succeed");
            }
            Ahead::Overtaken => {
                // Another operation's record, moved in after this call
                // "started" with the field absent.
                let _: () = pool
                    .hset(
                        wx.wtok_key(),
                        (CLIENT_ID, "op.other-op:2:1700000000000000:tok:digest"),
                    )
                    .await
                    .expect("staging the moved record must succeed");
                operation = Some(
                    Operation::new("w1-op", operation_request_digest(LAYER, b"w1"))
                        .with_start_record(StartRecord::Absent),
                );
            }
        }
        let before = wx.image().await;
        let what = format!("{path:?} / {guard:?}");

        let delta = delta(path.shape(), 1);
        let relay = relay(&delta);
        let outcome = tokio::time::timeout(
            CALL_BUDGET,
            store.update_layer_with_operation(
                &wx.session_id,
                LAYER,
                wx.oid,
                Arc::new(Some(delta)),
                relay,
                CLIENT_ID,
                1_000,
                operation,
            ),
        )
        .await
        .expect("update_layer did not settle within the call budget");

        match (guard, outcome) {
            (Ahead::AlreadyLanded, Ok(commit)) => assert_eq!(
                (commit.seq, commit.server_ts),
                (landed_seq, landed_ts),
                "{what}: the landed commit is reported"
            ),
            (Ahead::Superseded, Err(ServiceError::FailedPrecondition(msg))) => {
                assert!(msg.contains("superseded"), "{what}: got {msg:?}")
            }
            (Ahead::Overtaken, Err(ServiceError::Unavailable(msg))) => assert_eq!(
                msg, OPERATION_OVERTAKEN_MESSAGE,
                "{what}: the overtaken verdict, not the fence-wait cap"
            ),
            (_, other) => panic!("{what}: the guard must answer ahead of the fence; got {other:?}"),
        }
        assert_eq!(wx.image().await, before, "{what}: nothing written");
        assert_eq!(
            wx.fence_timeouts(),
            0,
            "{what}: no fence wait reached the cap"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_step_1b_guards_answer_ahead_of_a_live_fence_on_every_path() {
        within_test_budget(async {
            for path in [
                Path::Fast,
                Path::Structural,
                Path::Reset,
                Path::WatchKnob,
                Path::TxnFieldOnly,
                Path::UncacheableFallback,
            ] {
                for guard in [Ahead::AlreadyLanded, Ahead::Superseded, Ahead::Overtaken] {
                    guard_ahead_of_a_live_fence(path, guard).await;
                }
            }
        })
        .await;
    }
}
