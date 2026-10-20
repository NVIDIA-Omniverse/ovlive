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

//! The structural cache's cold start —
//! `storage::structural_cache_cold_start::cold_start`, the read side's probed
//! `cold_start_probed`, and the fenced route's capacity hook
//! (`storage::structural_cache_fenced`) (b.dkt Epic 4 Task 3, Epic 6 Task 11,
//! C1 the fenced cold start; structural-edits SDD SR-4.3 steps 1, 3 and 4,
//! SR-3.2's dangling seeding, SR-2.5's metrics) — and the snapshot script's
//! `return_dangling` flag it reads with.
//!
//! | Area | What it proves |
//! |---|---|
//! | Route `atomic` | within the shared ceilings the entry is the stored cut: the compact projection of the normalized fresh state (from the independent fan-out reader, `compact_snapshot`), the store's dangling `obj:chs` ids, the header's `seq` and `commit_chain`, and the call's commit-token record |
//! | Route `fenced` | above the plain item or byte ceiling — however far above, the raised item ceiling included — the plain read answers `TOO_LARGE` and b.y3m's fenced read loads the same cut, its dangling ids from that read's structural facts; size alone is never an uncacheable verdict |
//! | Route `uncacheable` | over the weight cap: `OverWeight`; an unmarked object: `NoChsComplete`, with no fenced read when it is too large |
//! | The probed route | the probe's cardinalities give the script's own item bound, marked or not; the predictor prices the bound at `STRUCTURAL_COLD_START_ROUTE_BYTES_PER_ITEM` (read from the constant, never restated), inclusively, saturating; T5 — a fenced entry equals the atomic read's entry at the same cut, and the next structural edits hit it with no new read; T7 — a layer over the plain byte ceiling and under the item bound costs one full read, on `path="fenced"` (two without the predictor, the unprobed cold start); a predictor miss reads twice and counts the miss as the atomic `too_large`; T8 — a lower weight bound over the per-entry cap records `OverWeight` at that bound with no read, no fence acquire, no `P_f` wait and no memory grant, from the probe and (unprobed) from the `TOO_LARGE` answer's cardinalities, while a cap at the bound reads; an unmarked object answers `NoChsComplete` with no full read |
//! | No entry, no verdict | a superseded header answers `Superseded` with the token record; an absent object `NotFound`; a disabled cache reads nothing |
//! | The one-at-a-time permit | two fenced cold starts of different objects run one at a time (the second sends no acquire meanwhile); the plain route and the decisions taken before the fenced read (an unmarked `TOO_LARGE`, the probe's unmarked verdict, the `TOO_LARGE` weight check) do not take it, while a fenced cold start under a held permit stays blocked; the permit is released when the fenced read fails and when its holder (and a waiter) is cancelled, and is not held through the admission and the token `HGET` |
//! | The plain permits | three plain cold starts of three objects run at most two reads at once and all load; three fenced ones run one fenced read at a time, and the two queued behind it hold no plain permit; T6(b) — a plain cold start loads while one fenced read is in flight and three fenced cold starts are queued (the decider's pin); a plain cold start dropped holding its permit, or waiting for one, releases it to the next waiter; a fenced cold start dropped while waiting for its plain permit, the one-at-a-time permit already taken, releases that permit; neither route holds a permit through its token `HGET` |
//! | The fenced read's capacity | T6(a) — with `P_f` held elsewhere a fenced cold start holds its pair and no memory, and sends no acquire, until `P_f` frees; T6(c) — through a `RETRY_AFTER` sleep (a forged live lease) it holds no pair, no `P_f` and no grant, and another fenced cold start takes the one-at-a-time permit and loads; T6(d) — cancelled mid-read, or failed by three broken attempts, it gives back the pair, `P_f`, the grant and the lease; T16 — `(SNAPSHOT_MEMORY_FACTOR + 1) × b` of `T` (read at run time) is reserved across the acquire, beside `P_f` and the pair, and nothing is held at the test hook before `admit`; T17 — a deadline that leaves no time for the read and one WATCH attempt refuses before any acquire, a `RETRY_AFTER` past the cap refuses at once, and with no deadline the cap is `CappedFenceWait`'s; each such failure is `fenced_read_failed` for that call only: no negative entry, no cold-start tick, a DEBUG line, and the next call caches |
//! | The deadline check (M9) | `fenced_read_over_deadline` is true exactly when the configured deadline is at or below the fenced read's reserve (a zero waiting cap with the whole deadline left), at every size, and so monotone in the deadline; a predictor miss whose `TOO_LARGE` size cannot fit the request's configured deadline (`with_rpc_deadline`) is recorded `OverDeadline` after the plain read — judged at the answer's size, not the probe's estimate — with no fenced read, acquire, `P_f` wait or grant, one tick, one `uncacheable` route tick and one INFO line carrying the size and the deadline, and the next call is routed by its negative entry with no read. The predictor's route, the per-call condition and the plain route's exemption run end to end in `structural_fenced_cold_start_dispatch_test` |
//! | No fan-out outside the fence; plain ceilings only (T10) | `MONITOR`: the atomic route issues the snapshot script and the token `HGET` only; the fenced route's fan-out commands all run between its acquire and its release; every snapshot-script call passes the client's plain ceilings, on every route, a predictor miss's included |
//! | Script flag | `ARGV[12] = '1'` appends exactly the dangling ids to an `OK` reply (`{}` when there are none, and unmarked); absent and `'0'` leave the nine-element reply unchanged; `TOO_LARGE` gains nothing; another value is an argument error. `ARGV[13]` (`slim`, b.mr6) unset — absent or `'0'`, beside either `ARGV[12]` — leaves every reply byte-identical; set, it changes only the per-row elements of `flds` and `tss`, to each row's shape (the tallies stay the unset read's), and no `TOO_LARGE`; another value is an argument error |
//! | Retirement during the read | a `DeleteLayer` between the read and the admission answers `Invalidated`, stores and records nothing, reads no token (`MONITOR`), ticks nothing |
//! | Task 2's deferred items | an entry mini-moka refuses is served from the side store as a hit and never ticks `uncacheable`; an over-weight object under a busy full cache gets one read, then routes `Uncacheable` from its negative entry until that entry's absolute TTL |
//! | Metrics | one `live_edit_structural_cold_start_reads_total{route}` tick (and one duration) per cold start that ends in an entry or a recorded verdict; nothing for superseded, absent, disabled; every read on `live_edit_snapshot_read_*{consumer="structural_cache"}`, the fenced ones on `path="fenced"`; no `atomic_raised` series; a token `HGET` failing after the admission is still counted |
//! | Reporting a verdict | each recorded verdict ticks `live_edit_structural_cache_uncacheable_total` under its reason once and logs one INFO line with the bounds it compared (the weight against the per-entry cap, none for `no_chs_complete`); `fenced_read_failed` logs at DEBUG with its cause, attempts, size, wait and cap; `over_deadline` at INFO with its size, the configured deadline and the read's reserve (M9's row), fields no other verdict's line carries; no line carries the retired raised read's bounds |
//! | A refused plain read's hold allowance | `plain_cold_start_hold_allowance_bytes` is the ceiling × `r` × (1 + `b`), floored, saturating, 5,535,891 B at the shipped ceiling; under a charge refusing CR0, a hold (CR0's need: the decoded state plus the index build's peak) exactly at the allowance loads, stores the entry and ticks `atomic`, and one byte over loads nothing — nothing stored, no route tick, no read permit held, one DEBUG line with the hold, the allowance and the target — both answering `NeedsCharge` with CR0's target; the allowance judges the plain route only: a fenced cold start under the same refusal and a 1 B allowance still loads, stores its entry and ticks `fenced` |
//!
//! Most rows drive the unprobed `cold_start`, whose plain read always runs
//! first: the routes, permits and verdicts it pins are the probed route's too
//! once that route reaches the same read. The probed rows drive
//! `cold_start_probed` with the read side's own probe cardinalities; the
//! deadline rows (T17), the failure rows that check the next call and the
//! hold-allowance row drive the read side's entry point,
//! `read_for_structural_edit`, the only path a deadline or a commit charge
//! reaches the cold start by. T5's cache hits run end to end through
//! `LayerStore::update_layer`.
//!
//! The hook's own events — the pair, `P_f` and the grant taken in that
//! order, the grant adjusted to the measured weight on a verified read, a
//! broken attempt keeping everything, the waiting cap's arithmetic — are unit
//! tests in `src/storage/structural_cache_fenced.rs`; the cache's own ticket
//! and epoch rules are inline in `src/storage/structural_cache.rs`; the
//! zero-seeded series are in `observability_metrics_test.rs`. The fixtures
//! are `snapshot_read_atomic_test`'s; the instruments ([`MonitorTap`],
//! [`ScriptReplyProxy`] and its [`ReplyGate`]) are in `snapshot_read_support`.
//! The plain permits' tests use no proxy: the test task polls each cold start
//! itself, which parks a read with its permits held (section 2).
//!
//! **No figure the cost model or configuration owns is restated.** The route
//! predictor's bytes per item `K`, the memory factor, the budget `T`, the
//! `P_f` limit and every fence timing are read at run time from the
//! constants, the client and `snapshot_fence::coefficients`.
//!
//! **Hang budget.** Every wait is bounded by [`WAIT_BUDGET`] (20 s) or a
//! shorter named constant; a healthy fenced read takes milliseconds plus, at
//! most, a writers' gap or a `RETRY_AFTER` step of about one second. The
//! longest test, the decider's pin, waits six budgets in series (120 s summed
//! at full budget, every one of them unreachable on a healthy store); the
//! fenced-capacity tests wait at most four (80 s).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex as StdMutex};
use std::task::Poll;
use std::time::{Duration, Instant};

use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface, LuaInterface};
use fred::types::scripts::Script;
use fred::types::Value;

use live_edit::config::{
    UpdateLayerBackend, RPC_DEADLINE_MS_DEFAULT, SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT,
    SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
};
use live_edit::domain::delta::MutationAllowance;
use live_edit::domain::replay::normalize_state;
use live_edit::domain::types::{
    compact, CompactState, Field, FieldValue, ObjectState, SampleRows, Section,
};
use live_edit::error::ServiceError;
use live_edit::grpc::upload_admission::{
    object_state_charge_for_test, object_state_rewrite_allowance_for_test,
};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{
    Metrics, CATCHUP_BUILD_ADMISSION_RESULTS, SNAPSHOT_FENCE_ACQUIRE_RESULTS,
    SNAPSHOT_READ_OUTCOMES, SNAPSHOT_READ_PATHS,
};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::commit_charge::{ObjectStateNeed, SharedCommitCharge};
use live_edit::storage::diff_read::{
    read_object_snapshot, read_object_snapshot_slim_with_ceilings, snapshot_item_bound,
    SlimSnapshotReadConsumer, SnapshotCardinalities, SnapshotCeilings, SnapshotRead,
    SnapshotReadConsumer, SnapshotReadOf, SNAPSHOT_FIELD_HASH_WIDTH, SNAPSHOT_SECTION_HASH_WIDTH,
    SNAPSHOT_TIME_SAMPLE_HASH_WIDTH, SNAPSHOT_TOMBSTONE_HASH_WIDTH,
};
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::{
    LayerStore, CHS_COMPLETE_MARKED, HEADER_FIELD_CHS_COMPLETE, HEADER_FIELD_COMMIT_CHAIN,
};
use live_edit::storage::memory_budget::{kib_for_bytes, SNAPSHOT_MEMORY_FACTOR};
use live_edit::storage::snapshot_fence::read::CappedFenceWait;
use live_edit::storage::snapshot_fence::{
    coefficients, ms_to_duration, FenceCoefficients, LayerSize,
};
use live_edit::storage::structural_cache::{
    compact_state, estimate_state_bytes, state_weight_lower_bound, CacheEntry, CacheLookup,
    ColdStartReadsInFlight, StructuralCache, UncacheableReason,
};
use live_edit::storage::structural_cache_cold_start::{
    cold_start, cold_start_probed, fenced_read_over_deadline,
    plain_cold_start_hold_allowance_bytes, plain_read_predicted_to_fit, raised_cold_start_ceilings,
    ColdStart, ColdStartLoaded, ColdStartRoute, STRUCTURAL_COLD_START_BYTES_FACTOR,
    STRUCTURAL_COLD_START_ITEMS_FACTOR, STRUCTURAL_COLD_START_ROUTE_BYTES_PER_ITEM,
};
use live_edit::storage::structural_cache_fenced::{wait_cap, wait_reserve};
use live_edit::storage::structural_cache_read::{
    read_for_structural_edit, StructuralRead, StructuralReadRequest,
};
use live_edit::storage::structural_cache_sync::{
    read_structural_cache_sync, StructuralSyncRequest, SyncChunk, SyncHeader,
};
use live_edit::storage::test_support::{compact_snapshot, read_object_state_for_test, TestValkey};

use crate::common::owned_delta::{delta_to_ompf_bytes, delta_with_set_field_inline};
use crate::grpc::log_capture::{
    assert_field, assert_field_absent, capture, capture_debug, Captured,
};
use crate::storage::snapshot_read_atomic_test::{
    commit, create, create_cmd, create_session, every_shape_state, indexed_parents, layer_store,
    mock_ext_store, structural, unmark, CUBE, DANGLING_PARENT, ROOT, WORLD,
};
use crate::storage::snapshot_read_support::{
    client_through, client_through_with_limits, client_with_limits, error_reply_script,
    monitor_fence, on_current_thread, read_too_large, test_valkey_addr, MonitorTap,
    MonitoredCommand, ProxyAction, ProxyRule, ReplyGate, ScriptReplyProxy,
};

/// A budget no single fixture entry comes near.
const BUDGET_BYTES: u64 = 64 * 1024 * 1024;
/// The structural call's `client_id`, whose commit-token record the cold
/// start reads.
const CLIENT_ID: &str = "cold-start-client";
/// The commit-token record [`seeded_object`] stores for [`CLIENT_ID`].
const TOKEN_RECORD: &str = "cold-start-token-record";
/// The `client_id` of a read-side request ([`read_for_structural_edit`]):
/// one with no commit-token record, so the token check never answers in the
/// cold start's place.
const READ_CLIENT_ID: &str = "cold-start-read-side-client";
/// That request's commit token.
const READ_TOKEN: &str = "cold-start-read-side-token";
/// A byte ceiling no fixture comes near, for the clients whose item ceiling
/// is the one under test.
const NO_BYTE_LIMIT: u64 = 1 << 40;
/// `K`, the route predictor's bytes per bound item. Always read from the
/// constant: it is re-derived from measurement (policy decision P11), so no
/// test restates its value.
const K: u64 = STRUCTURAL_COLD_START_ROUTE_BYTES_PER_ITEM;
/// Bound on a wait a correct cold start never comes near; well inside
/// nextest's 120 s hang-killer.
const WAIT_BUDGET: Duration = Duration::from_secs(20);
/// How long a test watches a cold start that must stay parked on a
/// cold-start permit (or on `P_f`). A correct implementation never leaves the
/// wait while the permit is held, so the window only bounds how long the test
/// looks; an implementation that took no permit finishes its read in
/// milliseconds and is caught.
const BLOCKED_WINDOW: Duration = Duration::from_millis(500);
/// Pause between the test's own polls of a cold start it drives (the plain
/// permits' tests), and between samples of a condition it waits for.
const POLL_PAUSE: Duration = Duration::from_millis(5);
/// A TTL no test comes near, for the expiry a test is not exercising.
const LONG_TTL: Duration = Duration::from_secs(3_600);
/// The snapshot script, `EVAL`ed directly by the flag tests.
const SNAPSHOT_SCRIPT: &str = include_str!("../../../src/storage/scripts/object_snapshot_read.lua");
/// A reader deadline far past any test's end (ms since the Unix epoch, in
/// the year 2286). A forged `r:*` field holding it is a live lease to every
/// acquire, which answers `RETRY_AFTER` `epoch` with a hint of one `poll`.
const FORGED_LEASE_DEADLINE_MS: i64 = 9_999_999_999_999;
/// The forged lease's field in the fence key.
const FORGED_LEASE_FIELD: &str = "r:forged-by-the-cold-start-test";

// ═════════════════════════════════════════════════════════════════════════════
// Fixtures
// ═════════════════════════════════════════════════════════════════════════════

/// A marked `every_shape_state` object after one fast-path commit — so its
/// header `seq` and `commit_chain` are a commit's, not the creation's — with
/// [`CLIENT_ID`]'s commit-token record set to [`TOKEN_RECORD`]. It keeps the
/// list under `DANGLING_PARENT`, which no live section names, and its time
/// samples.
async fn seeded_object(tv: &TestValkey, ls: &LayerStore, layer: &str) -> u64 {
    let session_id = create_session(tv, layer).await;
    let oid = create(ls, &session_id, layer, &every_shape_state()).await;
    commit(
        ls,
        &session_id,
        layer,
        oid,
        delta_with_set_field_inline(WORLD, "label", vec![0x2A], 5),
    )
    .await;
    let _: i64 = tv
        .client()
        .pool
        .hset(
            tv.key_schema().obj_write_token(&oid.to_string()),
            [(CLIENT_ID, TOKEN_RECORD)],
        )
        .await
        .expect("HSET obj:wtok must succeed");
    oid
}

pub(crate) async fn header(tv: &TestValkey, oid: u64) -> HashMap<String, String> {
    tv.client()
        .pool
        .hgetall(tv.key_schema().obj_header(&oid.to_string()))
        .await
        .expect("HGETALL obj:hdr must succeed")
}

/// The fresh state a cold-started entry must equal (SR-3.2's cache contract):
/// the compact projection (`compact_snapshot`) of the fan-out reader's rows —
/// a reader independent of the atomic script — normalized. Each time-sample
/// row is keyed by its stored bits, so a `0.0` and a `-0.0` row stay two rows
/// as they do in the entry. Shared with `structural_cache_read_test`,
/// `structural_dispatch_test` and `structural_size_gate_test`.
pub(crate) async fn normalized_fresh_state(tv: &TestValkey, oid: u64) -> CompactState {
    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("the reference read must succeed");
    let mut state = compact_snapshot(&snapshot);
    normalize_state(&mut state);
    state
}

/// The store's dangling children-list parents: every `obj:chs` member that is
/// neither the root nor a live section (the fan-out reader's sections).
/// Shared with `structural_cache_read_test`.
pub(crate) async fn stored_dangling_parents(tv: &TestValkey, oid: u64) -> HashSet<u64> {
    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("the reference read must succeed");
    indexed_parents(tv, oid)
        .await
        .into_iter()
        .filter(|p| *p != ROOT && !snapshot.sections.contains_key(p))
        .collect()
}

/// The object's step-2 item bound, from a read under a one-item ceiling.
async fn item_bound(tv: &TestValkey, oid: u64) -> u64 {
    read_too_large(
        &client_with_limits(1, NO_BYTE_LIMIT).await,
        tv.key_schema(),
        oid,
    )
    .await
    .item_bound
}

/// A client whose plain item ceiling is one under `oid`'s step-2 bound: every
/// cold start of `oid` on it takes the fenced read.
async fn fenced_route_client(tv: &TestValkey, oid: u64) -> ValkeyClient {
    client_with_limits(item_bound(tv, oid).await - 1, NO_BYTE_LIMIT).await
}

/// The client's plain atomic-read ceilings.
fn plain_ceilings(client: &ValkeyClient) -> SnapshotCeilings {
    SnapshotCeilings::of_client(client)
}

/// The bytes a whole read of the object tallies.
async fn full_reply_bytes(tv: &TestValkey, oid: u64) -> u64 {
    match read_object_snapshot(
        tv.client(),
        tv.key_schema(),
        &oid.to_string(),
        SnapshotReadConsumer::Catchup,
    )
    .await
    {
        Ok(SnapshotRead::Snapshot(snapshot)) => snapshot.reply_bytes,
        Ok(SnapshotRead::TooLarge(too_large)) => {
            panic!("fixture: the whole read of object {oid} must succeed; got {too_large:?}")
        }
        Err(e) => panic!("fixture: the whole read of object {oid} must succeed; got {e:?}"),
    }
}

/// The weight the cache charges a cold-started entry of `state`: the estimate
/// of its compacted form, which is what `admit` stores. The estimate charges
/// each children list's capacity, and a cold start trims every list to its
/// length before it admits the entry (`structural_cache_cold_start::load`),
/// so the lists are weighed exact ([`as_cold_started`]).
fn charged_weight(state: &CompactState) -> u64 {
    let mut state = as_cold_started(state);
    compact_state(&mut state);
    estimate_state_bytes(&state)
}

/// `state` as a cold start holds it: each children list trimmed to its
/// length (see [`charged_weight`]).
fn as_cold_started(state: &CompactState) -> CompactState {
    let mut state = state.clone();
    for list in state.children.values_mut() {
        list.shrink_to_fit();
    }
    state
}

/// Load the snapshot script into the store's script cache, so a test's first
/// `EVALSHA` runs at once rather than after a `NOSCRIPT` round trip (a
/// `MONITOR` record or a held reply would otherwise see the retry).
async fn warm_snapshot_script(tv: &TestValkey, oid: u64) {
    full_reply_bytes(tv, oid).await;
}

/// Load the fence scripts (acquire, release) into the store's script cache:
/// one fenced cold start of an object of its own. For the tests whose proxy
/// rule or `MONITOR` record must see each fence script's `EVALSHA` once.
async fn warm_fence_scripts(tv: &TestValkey, ls: &LayerStore, layer: &str) {
    let oid = seeded_object(tv, ls, &format!("{layer}-warm")).await;
    let client = fenced_route_client(tv, oid).await;
    let cache = StructuralCache::new(BUDGET_BYTES);
    let loaded = expect_loaded(
        cold_start(&client, tv.key_schema(), &cache, oid, CLIENT_ID).await,
        "the fence scripts' warm-up",
    );
    assert_eq!(
        loaded.route,
        ColdStartRoute::Fenced,
        "fixture: the warm-up took the fenced read"
    );
}

fn expect_loaded(answer: Result<ColdStart, ServiceError>, context: &str) -> ColdStartLoaded {
    match answer {
        Ok(ColdStart::Loaded(loaded)) => loaded,
        other => panic!("{context}: the cold start must load an entry; got {other:?}"),
    }
}

/// The reason and the size an `Uncacheable` answer carries.
fn expect_uncacheable_sized(
    answer: Result<ColdStart, ServiceError>,
    context: &str,
) -> (UncacheableReason, Option<LayerSize>) {
    match answer {
        Ok(ColdStart::Uncacheable {
            reason,
            above_ceiling,
        }) => (reason, above_ceiling),
        other => panic!("{context}: the cold start must answer Uncacheable; got {other:?}"),
    }
}

fn expect_uncacheable(answer: Result<ColdStart, ServiceError>, context: &str) -> UncacheableReason {
    expect_uncacheable_sized(answer, context).0
}

/// Assert `entry` is the object's stored cut: the normalized fresh state, the
/// store's dangling ids, and the header's `seq` and `commit_chain` as `hdr`
/// read them before the cold start.
async fn assert_entry_is_the_stored_cut(
    tv: &TestValkey,
    oid: u64,
    entry: &CacheEntry,
    hdr: &HashMap<String, String>,
    context: &str,
) {
    let dangling = stored_dangling_parents(tv, oid).await;
    assert_eq!(
        dangling,
        HashSet::from([DANGLING_PARENT]),
        "{context}: fixture precondition — the object must own a list under a parent that names \
         no section, or the dangling half of this check is vacuous"
    );
    let seq: u64 = hdr["seq"].parse().expect("the header seq is decimal");
    let commit_chain = &hdr[HEADER_FIELD_COMMIT_CHAIN];
    assert!(
        seq > 1 && !commit_chain.is_empty(),
        "{context}: fixture precondition — the header must carry a commit's seq and \
         commit_chain; got seq {seq}, commit_chain {commit_chain:?}"
    );
    assert_eq!(entry.object_id, oid, "{context}: object_id");
    assert_eq!(
        (entry.seq, &entry.commit_chain),
        (seq, commit_chain),
        "{context}: the entry's (seq, commit_chain) cut must be the header's"
    );
    assert_eq!(
        entry.dangling_parents, dangling,
        "{context}: dangling_parents must be exactly the store's dangling obj:chs ids, seeded by \
         the read (SR-3.2)"
    );
    assert_eq!(
        entry.state,
        normalized_fresh_state(tv, oid).await,
        "{context}: the entry must equal the normalized fresh state's compact projection \
         (SR-3.2's cache contract)"
    );
}

/// Spawn a cold start of `oid` on `cache` through `client`: probed with
/// `probed`'s cardinalities, or unprobed.
fn spawn_cold_start(
    client: &ValkeyClient,
    keys: &KeySchema,
    cache: &StructuralCache,
    oid: u64,
    probed: Option<SnapshotCardinalities>,
) -> tokio::task::JoinHandle<Result<ColdStart, ServiceError>> {
    let (client, keys, cache) = (client.clone(), keys.clone(), cache.clone());
    tokio::spawn(async move {
        match probed {
            Some(cards) => cold_start_probed(&client, &keys, &cache, oid, CLIENT_ID, cards).await,
            None => cold_start(&client, &keys, &cache, oid, CLIENT_ID).await,
        }
    })
}

/// Join a spawned cold start within [`WAIT_BUDGET`].
async fn join_cold_start(
    task: tokio::task::JoinHandle<Result<ColdStart, ServiceError>>,
    what: &str,
) -> Result<ColdStart, ServiceError> {
    tokio::time::timeout(WAIT_BUDGET, task)
        .await
        .unwrap_or_else(|_| panic!("{what}: the cold start must finish within the budget"))
        .expect("the cold-start task must not panic")
}

// ── Metrics ──────────────────────────────────────────────────────────────────

/// A fresh registry, installed on every client in `clients`.
async fn hooked_metrics(label: &str, clients: &[&ValkeyClient]) -> Arc<Metrics> {
    let metrics = Arc::new(crate::common::metrics_scrape_only(label));
    for client in clients {
        client
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
    }
    metrics
}

/// `live_edit_structural_cold_start_reads_total{route}`. Panics on an absent
/// cell: every route is zero-seeded.
fn cold_starts(metrics: &Metrics, route: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_structural_cold_start_reads_total",
        &[("route", route)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_structural_cold_start_reads_total{{route=\"{route}\"}} is absent — it is \
             zero-seeded"
        )
    }) as u64
}

/// Every cold-start route tick, `(atomic, fenced, uncacheable)`.
fn cold_starts_by_route(metrics: &Metrics) -> (u64, u64, u64) {
    (
        cold_starts(metrics, "atomic"),
        cold_starts(metrics, "fenced"),
        cold_starts(metrics, "uncacheable"),
    )
}

/// The `_count` of `live_edit_structural_cold_start_read_duration_seconds`
/// for `route`; `None` before its first observation.
fn cold_start_durations(metrics: &Metrics, route: &str) -> Option<u64> {
    sample_value(
        &render(metrics),
        "live_edit_structural_cold_start_read_duration_seconds_count",
        &[("route", route)],
    )
    .map(|v| v as u64)
}

/// `live_edit_snapshot_read_total{consumer="structural_cache",path,outcome}`.
/// Panics on an absent cell: every atomic and fenced cell of this consumer is
/// zero-seeded.
fn structural_reads_on(metrics: &Metrics, path: &str, outcome: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_snapshot_read_total",
        &[
            ("consumer", "structural_cache"),
            ("path", path),
            ("outcome", outcome),
        ],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_snapshot_read_total{{consumer=\"structural_cache\",path=\"{path}\",\
             outcome=\"{outcome}\"}} is absent — it is zero-seeded"
        )
    }) as u64
}

/// The atomic `structural_cache` reads of `outcome`.
fn structural_reads(metrics: &Metrics, outcome: &str) -> u64 {
    structural_reads_on(metrics, "atomic", outcome)
}

/// The fenced `structural_cache` reads of `outcome`.
fn fenced_reads(metrics: &Metrics, outcome: &str) -> u64 {
    structural_reads_on(metrics, "fenced", outcome)
}

/// `(too_large, ok)` atomic structural-cache snapshot reads.
fn too_large_and_ok_reads(metrics: &Metrics) -> (u64, u64) {
    (
        structural_reads(metrics, "too_large"),
        structural_reads(metrics, "ok"),
    )
}

/// Every `live_edit_snapshot_read_total{consumer="structural_cache"}` tick,
/// over every path and outcome: the full reads (and fenced attempts) the cold
/// start issued.
fn all_structural_reads(metrics: &Metrics) -> u64 {
    let rendered = render(metrics);
    let mut total = 0.0;
    for path in SNAPSHOT_READ_PATHS {
        for outcome in SNAPSHOT_READ_OUTCOMES {
            total += sample_value(
                &rendered,
                "live_edit_snapshot_read_total",
                &[
                    ("consumer", "structural_cache"),
                    ("path", path),
                    ("outcome", outcome),
                ],
            )
            .unwrap_or(0.0);
        }
    }
    total as u64
}

/// `live_edit_snapshot_fence_acquire_total{result}`; an absent cell reads 0.
fn fence_acquires_of(metrics: &Metrics, result: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_snapshot_fence_acquire_total",
        &[("result", result)],
    )
    .unwrap_or(0.0) as u64
}

/// Every fence acquire reply, over every result: the acquires sent.
fn fence_acquires(metrics: &Metrics) -> u64 {
    SNAPSHOT_FENCE_ACQUIRE_RESULTS
        .iter()
        .map(|result| fence_acquires_of(metrics, result))
        .sum()
}

/// The fenced cold start's `P_f` waits:
/// `live_edit_catchup_snapshot_read_wait_seconds_count{consumer="structural_cache",class="fenced"}`,
/// absent as 0.
fn fenced_permit_waits(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_catchup_snapshot_read_wait_seconds_count",
        &[("consumer", "structural_cache"), ("class", "fenced")],
    )
    .unwrap_or(0.0) as u64
}

/// Every reservation the catch-up memory budget answered
/// (`live_edit_catchup_build_admission_total`, every result), absent as 0.
fn budget_reservations(metrics: &Metrics) -> u64 {
    let rendered = render(metrics);
    CATCHUP_BUILD_ADMISSION_RESULTS
        .iter()
        .map(|result| {
            sample_value(
                &rendered,
                "live_edit_catchup_build_admission_total",
                &[("result", result)],
            )
            .unwrap_or(0.0) as u64
        })
        .sum()
}

/// The `structural_cache_lookups_total{outcome}` cell, absent as 0.
fn lookups(metrics: &Metrics, outcome: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_structural_cache_lookups_total",
        &[("outcome", outcome)],
    )
    .unwrap_or(0.0) as u64
}

// ── The fenced route's capacity ──────────────────────────────────────────────

/// What a fenced cold start can hold, at one instant: the cache's cold-start
/// reads in flight (its pair counts as one `fenced`), the client's `P_f`
/// permits held, and the client's catch-up memory budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FencedCapacity {
    in_flight: ColdStartReadsInFlight,
    /// `P_f` permits held: the gate's limit less those available.
    p_f_held: usize,
    /// Bytes held by grants.
    reserved_bytes: u64,
    /// Bytes held by top-ups.
    overcommit_bytes: u64,
}

/// Nothing held.
const NOTHING_HELD: FencedCapacity = FencedCapacity {
    in_flight: NONE_IN_FLIGHT,
    p_f_held: 0,
    reserved_bytes: 0,
    overcommit_bytes: 0,
};

fn fenced_capacity(client: &ValkeyClient, cache: &StructuralCache) -> FencedCapacity {
    let gate = &client.catchup_fenced_read_gate;
    FencedCapacity {
        in_flight: cache.cold_start_reads_in_flight_for_test(),
        p_f_held: gate.limit() - gate.available(),
        reserved_bytes: client.catchup_build_budget.reserved_bytes(),
        overcommit_bytes: client.catchup_build_budget.overcommit_bytes(),
    }
}

/// Wait, up to [`WAIT_BUDGET`], until `client` and `cache` hold exactly
/// `want`; panics with the last state seen.
async fn wait_for_capacity(
    client: &ValkeyClient,
    cache: &StructuralCache,
    want: FencedCapacity,
    what: &str,
) {
    let deadline = Instant::now() + WAIT_BUDGET;
    loop {
        let now = fenced_capacity(client, cache);
        if now == want {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "{what}: the capacity held never became {want:?}; last seen {now:?}"
        );
        tokio::time::sleep(POLL_PAUSE).await;
    }
}

/// The memory a fenced cold start of a layer estimated at `b` reply bytes
/// reserves before its acquire, as `client`'s budget takes it:
/// `(SNAPSHOT_MEMORY_FACTOR + 1) × b`, the snapshot plus one window, rounded
/// up to whole KiB and clamped to the configured `T` — read from the client
/// at run time, never restated (the decider's ruling).
fn expected_grant_bytes(client: &ValkeyClient, b: u64) -> u64 {
    let total_kib = client.catchup_build_budget.total_bytes() / 1024;
    kib_for_bytes(b.saturating_mul(SNAPSHOT_MEMORY_FACTOR + 1)).min(total_kib) * 1024
}

/// The fenced route's estimate on a probed cold start: the probe's bound, and
/// that bound priced at `K` bytes per item.
fn probed_estimate(cards: &SnapshotCardinalities) -> LayerSize {
    let bound = snapshot_item_bound(cards);
    LayerSize {
        items: bound,
        bytes: bound * K,
    }
}

/// `hold(L) + max(S(L), 0)` at `size` under `client`'s fence coefficients:
/// what the waiting cap keeps of the deadline for the read and one WATCH
/// attempt.
fn read_and_watch_reserve(client: &ValkeyClient, size: LayerSize) -> Duration {
    let timing = coefficients(client).timing(size);
    ms_to_duration(timing.hold_ms.max(0.0) + timing.structural_attempt_ms.max(0.0))
}

/// The object's fence key.
fn fence_key(tv: &TestValkey, oid: u64) -> String {
    tv.key_schema().obj_snapshot_fence(&oid.to_string())
}

/// The reader-lease fields (`r:*`) of the object's fence key.
#[cfg(feature = "test-fault-injection")]
async fn lease_fields(tv: &TestValkey, oid: u64) -> std::collections::BTreeSet<String> {
    let fields: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(fence_key(tv, oid))
        .await
        .expect("HGETALL obj:snapfence must succeed");
    fields.into_keys().filter(|k| k.starts_with("r:")).collect()
}

/// Wait, up to [`WAIT_BUDGET`], until the object's fence key holds no reader
/// lease: every lease the cold start took was released.
#[cfg(feature = "test-fault-injection")]
async fn wait_until_no_lease(tv: &TestValkey, oid: u64, what: &str) {
    let deadline = Instant::now() + WAIT_BUDGET;
    loop {
        let leases = lease_fields(tv, oid).await;
        if leases.is_empty() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "{what}: a lease was never released: {leases:?}"
        );
        tokio::time::sleep(POLL_PAUSE).await;
    }
}

/// Write a live reader lease nobody holds into the object's fence key, with
/// no `epoch_start`: every acquire then answers `RETRY_AFTER` `epoch`, and
/// writes nothing, until [`remove_forged_lease`].
async fn forge_live_lease(tv: &TestValkey, oid: u64) {
    let _: i64 = tv
        .client()
        .pool
        .hset(
            fence_key(tv, oid),
            [(FORGED_LEASE_FIELD, FORGED_LEASE_DEADLINE_MS.to_string())],
        )
        .await
        .expect("HSET of the forged lease must succeed");
}

async fn remove_forged_lease(tv: &TestValkey, oid: u64) {
    let _: i64 = tv
        .client()
        .pool
        .hdel(fence_key(tv, oid), FORGED_LEASE_FIELD)
        .await
        .expect("HDEL of the forged lease must succeed");
}

/// The SHA of a script, for recognising its `EVALSHA` on a proxied
/// connection and in a `MONITOR` feed.
fn script_sha(text: &str) -> String {
    let script = Script::from_lua(text);
    let sha: &str = script.sha1();
    sha.to_owned()
}

fn snapshot_sha() -> String {
    script_sha(SNAPSHOT_SCRIPT)
}

fn acquire_sha() -> String {
    script_sha(include_str!(
        "../../../src/storage/scripts/snapshot_fence_acquire.lua"
    ))
}

fn release_sha() -> String {
    script_sha(include_str!(
        "../../../src/storage/scripts/snapshot_fence_release.lua"
    ))
}

/// Apply `action` to the fence acquire whose fence key (`KEYS[1]`, frame
/// argument 3) is `fence_key`.
fn fence_acquire_rule(fence_key: String, action: ProxyAction) -> ProxyRule {
    let sha = acquire_sha();
    Arc::new(move |args: &[Vec<u8>]| {
        let name = args.first()?.to_ascii_uppercase();
        (name == b"EVALSHA"
            && args.get(1).map(Vec::as_slice) == Some(sha.as_bytes())
            && args.get(3).map(Vec::as_slice) == Some(fence_key.as_bytes()))
        .then(|| action.clone())
    })
}

// ── Proxy rules ──────────────────────────────────────────────────────────────

/// Apply `action` to the snapshot script's read of `header_key` at the item
/// ceiling `max_items` (the script's `ARGV[4]`, frame argument 7 after
/// `EVALSHA sha numkeys key prefix oid mode`).
fn snapshot_read_rule(header_key: String, max_items: u64, action: ProxyAction) -> ProxyRule {
    let items = max_items.to_string();
    Arc::new(move |args: &[Vec<u8>]| {
        let name = args.first()?.to_ascii_uppercase();
        if name != b"EVALSHA" && name != b"EVAL" {
            return None;
        }
        let ours = args.get(3).map(Vec::as_slice) == Some(header_key.as_bytes())
            && args.get(7).map(Vec::as_slice) == Some(items.as_bytes());
        ours.then(|| action.clone())
    })
}

/// Hold the replies to the snapshot script's read of `header_key` at the item
/// ceiling `max_items`.
fn hold_snapshot_read(header_key: String, max_items: u64, gate: ReplyGate) -> ProxyRule {
    snapshot_read_rule(header_key, max_items, ProxyAction::HoldReplies(gate))
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. Routes
// ═════════════════════════════════════════════════════════════════════════════

/// The decider's factors (items ×6, bytes ×8), applied per call, saturating.
/// No production read takes them now (neither a cold start nor the sampled
/// audit); the measurement tool's raised read still does.
#[test]
fn raised_cold_start_ceilings_multiply_by_the_decided_factors_and_saturate() {
    assert_eq!(
        (
            STRUCTURAL_COLD_START_ITEMS_FACTOR,
            STRUCTURAL_COLD_START_BYTES_FACTOR
        ),
        (6, 8),
        "the decider's ruling sets the raised read at items ×6 and bytes ×8"
    );
    assert_eq!(
        raised_cold_start_ceilings(SnapshotCeilings {
            max_items: 37_554,
            max_bytes: 449_938,
        }),
        SnapshotCeilings {
            max_items: 225_324,
            max_bytes: 3_599_504,
        },
        "the ceilings in force when the factors were decided (the byte limit then 449,938; \
         now 604,619), raised"
    );
    assert_eq!(
        raised_cold_start_ceilings(SnapshotCeilings {
            max_items: u64::MAX / 2,
            max_bytes: u64::MAX / 4,
        }),
        SnapshotCeilings {
            max_items: u64::MAX,
            max_bytes: u64::MAX,
        },
        "a product past u64 saturates rather than wrapping to a small ceiling"
    );
}

/// The predictor (P5, P11): `K` bytes per bound item; the bound within the
/// plain item ceiling and the priced bound within the plain byte ceiling,
/// both inclusive, as the script's own tests are; a bound too large to price
/// is never predicted to fit. `K` is read from its constant: it is
/// re-derived from measurement, and this test pins the rule, not the value.
#[test]
fn plain_read_predictor_prices_the_bound_at_the_route_constant_inclusively() {
    const { assert!(K > 0, "a zero K would predict every layer to fit on bytes") };
    let plain = SnapshotCeilings {
        max_items: 1_000,
        max_bytes: 500 * K,
    };
    assert!(
        plain_read_predicted_to_fit(500, plain),
        "exactly at the bytes"
    );
    assert!(
        !plain_read_predicted_to_fit(501, plain),
        "one item over the bytes"
    );
    let items_bind = SnapshotCeilings {
        max_items: 100,
        max_bytes: 1 << 40,
    };
    assert!(
        plain_read_predicted_to_fit(100, items_bind),
        "exactly at the items"
    );
    assert!(
        !plain_read_predicted_to_fit(101, items_bind),
        "one over the items"
    );
    assert!(
        !plain_read_predicted_to_fit(
            u64::MAX,
            SnapshotCeilings {
                max_items: u64::MAX,
                max_bytes: u64::MAX - 1,
            }
        ),
        "a priced bound past u64 saturates, never wraps to a small one"
    );
}

/// The deadline check's predicate (M9, `fenced_read_over_deadline`): a
/// fenced-bound layer is over the configured RPC deadline exactly when the
/// fenced read's waiting cap with that whole deadline left is zero — at
/// every size, when the deadline is at or below the read's reserve
/// (`wait_reserve`: its expected time plus one WATCH attempt's), and never
/// above it. That equality, checked over deadlines from zero through the
/// reserve's neighbours to an hour past it, implies the rest: a zero deadline
/// is always over, a generous one never is, and the answer is monotone in the
/// deadline (once a deadline fits, every longer one fits). The sizes span a
/// small layer to one whose reserve exceeds the default
/// `LIVE_EDIT_RPC_DEADLINE_MS`, so the default deadline is checked on both
/// sides; no coefficient is restated.
#[test]
fn fenced_read_over_deadline_is_a_zero_waiting_cap_at_the_whole_deadline_and_monotone_in_it() {
    let coefficients = FenceCoefficients::recorded(SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT);
    let default_deadline = Duration::from_millis(RPC_DEADLINE_MS_DEFAULT);
    let huge = LayerSize {
        items: 20_000_000,
        bytes: 400_000_000,
    };
    assert!(
        wait_reserve(&coefficients, huge) > default_deadline,
        "fixture: the largest size's fenced read cannot fit the default deadline"
    );
    for size in [
        LayerSize { items: 0, bytes: 0 },
        LayerSize {
            items: 100,
            bytes: 100 * K,
        },
        LayerSize {
            items: 40_000,
            bytes: 800_000,
        },
        huge,
    ] {
        let reserve = wait_reserve(&coefficients, size);
        assert!(reserve > Duration::ZERO, "fixture: {size:?} has a reserve");
        let one_ns = Duration::from_nanos(1);
        let deadlines = [
            Duration::ZERO,
            reserve - one_ns,
            reserve,
            reserve + one_ns,
            reserve * 2,
            default_deadline,
            reserve + Duration::from_secs(3_600),
        ];
        for deadline in deadlines {
            assert_eq!(
                fenced_read_over_deadline(&coefficients, deadline, size),
                deadline <= reserve,
                "size {size:?}, deadline {deadline:?}, reserve {reserve:?}: over exactly when \
                 the deadline leaves the read no waiting time"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_within_the_plain_ceilings_loads_the_stored_cut_on_the_atomic_route() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-atomic").await;
    let hdr = header(&tv, oid).await;
    let cache = StructuralCache::new(BUDGET_BYTES);

    let loaded = expect_loaded(
        cold_start(tv.client(), tv.key_schema(), &cache, oid, CLIENT_ID).await,
        "within the plain ceilings",
    );

    assert_eq!(loaded.route, ColdStartRoute::Atomic);
    assert_entry_is_the_stored_cut(&tv, oid, loaded.entry.entry(), &hdr, "atomic route").await;
}

/// The commit-token record is the call's own field of `obj:wtok`, read after
/// the read: the seeded record for [`CLIENT_ID`], and nothing for a client
/// that has none.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_reads_the_calls_own_commit_token_record() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-token").await;

    let mut tokens = Vec::new();
    for client_id in [CLIENT_ID, "a-client-with-no-record"] {
        let cache = StructuralCache::new(BUDGET_BYTES);
        let loaded = expect_loaded(
            cold_start(tv.client(), tv.key_schema(), &cache, oid, client_id).await,
            client_id,
        );
        tokens.push(loaded.commit_token);
    }

    assert_eq!(tokens, vec![Some(TOKEN_RECORD.to_owned()), None]);
}

/// Above the shared ceilings, unprobed: the plain read answers `TOO_LARGE`,
/// then b.y3m's fenced read loads the same cut, its dangling ids from that
/// fenced read's structural facts. Exactly those two reads, and one `fenced`
/// cold start.
async fn loads_on_the_fenced_route(
    tv: &TestValkey,
    oid: u64,
    client: &ValkeyClient,
    context: &str,
) {
    let metrics = hooked_metrics(context, &[client]).await;
    let hdr = header(tv, oid).await;
    let cache = StructuralCache::new(BUDGET_BYTES);

    let loaded = expect_loaded(
        cold_start(client, tv.key_schema(), &cache, oid, CLIENT_ID).await,
        context,
    );

    assert_eq!(
        loaded.route,
        ColdStartRoute::Fenced,
        "{context}: above the plain ceiling, the fenced read loads"
    );
    assert_eq!(
        (
            too_large_and_ok_reads(&metrics),
            fenced_reads(&metrics, "ok"),
            all_structural_reads(&metrics)
        ),
        ((1, 0), 1, 2),
        "{context}: the plain read answers TOO_LARGE, then one fenced read loads ((atomic \
         too_large, atomic ok), fenced ok, every read)"
    );
    assert_eq!(
        cold_starts_by_route(&metrics),
        (0, 1, 0),
        "{context}: one fenced cold start (atomic, fenced, uncacheable)"
    );
    assert_entry_is_the_stored_cut(tv, oid, loaded.entry.entry(), &hdr, context).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_above_the_plain_item_ceiling_loads_the_stored_cut_on_the_fenced_route() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-fenced-items").await;
    let client = fenced_route_client(&tv, oid).await;

    loads_on_the_fenced_route(&tv, oid, &client, "one item under the bound").await;
}

/// A mid-read `TOO_LARGE` on bytes: the fenced read loads, sized from the
/// answer's tallies.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_above_the_plain_byte_ceiling_loads_the_stored_cut_on_the_fenced_route() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-fenced-bytes").await;
    let bytes = full_reply_bytes(&tv, oid).await;
    let plain_bytes = bytes / 2;
    assert!(
        plain_bytes > 0 && item_bound(&tv, oid).await <= SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
        "fixture: the object must be over the plain byte ceiling and within the item ceiling"
    );
    let client = client_with_limits(SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT, plain_bytes).await;

    loads_on_the_fenced_route(&tv, oid, &client, "half of the object's bytes").await;
}

/// The size of a layer is never an uncacheable verdict any more: a bound
/// above even the raised item ceiling (six times the plain one) —
/// formerly the item early-out's `AboveColdStartCeiling` — loads on the fenced
/// route, records no negative entry and reports no verdict.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_far_above_the_plain_item_ceiling_loads_on_the_fenced_route_and_records_no_verdict(
) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-far-above").await;
    let bound = item_bound(&tv, oid).await;
    let plain_items = (bound - 1) / STRUCTURAL_COLD_START_ITEMS_FACTOR;
    assert!(
        plain_items > 0 && plain_items * STRUCTURAL_COLD_START_ITEMS_FACTOR < bound,
        "fixture: the bound must be over the raised item ceiling too"
    );
    let client = client_with_limits(plain_items, NO_BYTE_LIMIT).await;
    let metrics = hooked_metrics("cs-far-above", &[&client]).await;
    let cache = StructuralCache::new(BUDGET_BYTES);

    let loaded = expect_loaded(
        cold_start(&client, tv.key_schema(), &cache, oid, CLIENT_ID).await,
        "far above the plain item ceiling",
    );

    assert_eq!(loaded.route, ColdStartRoute::Fenced);
    drop(loaded);
    assert_eq!(
        uncacheable_verdicts(&metrics),
        BTreeMap::new(),
        "no verdict for size"
    );
    assert!(
        matches!(cache.lookup(oid), CacheLookup::Hit(_)),
        "the entry is cached, and no negative entry routes the object"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_over_the_weight_cap_records_over_weight_with_the_entrys_weight() {
    /// A budget whose quarter, the per-entry cap, no real object fits.
    const TINY_BUDGET: u64 = 64;
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-overweight").await;
    let weight = charged_weight(&normalized_fresh_state(&tv, oid).await);
    let cache = StructuralCache::with_ttls(TINY_BUDGET, LONG_TTL, LONG_TTL);
    assert!(
        weight > cache.max_entry_bytes(),
        "fixture: the entry must be over the cap"
    );

    let (reason, above_ceiling) = expect_uncacheable_sized(
        cold_start(tv.client(), tv.key_schema(), &cache, oid, CLIENT_ID).await,
        "over the weight cap",
    );

    assert_eq!(reason, UncacheableReason::OverWeight(weight));
    assert_eq!(
        above_ceiling, None,
        "a layer read within the plain ceilings carries no size"
    );
    assert!(
        matches!(
            cache.lookup(oid),
            CacheLookup::Uncacheable {
                reason: UncacheableReason::OverWeight(w)
            } if w == weight
        ),
        "the over-weight verdict is recorded as a negative entry"
    );
}

/// A cold-started entry's children lists carry no spare capacity, on the
/// atomic route and on the fenced one: each list's capacity is its length.
/// The readers' decoder collects a list from a filtered iterator, which
/// leaves a short list four slots wide; the entry is the decoded state
/// itself, so without the cold start's trim the entry would hold, and the
/// estimate would charge, those slots. The fixture's root list holds two
/// children and `WORLD`'s one, both short enough to show it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_entry_children_lists_carry_no_spare_capacity() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-exact-lists").await;
    let fenced_client = fenced_route_client(&tv, oid).await;

    for (client, route) in [
        (tv.client(), ColdStartRoute::Atomic),
        (&fenced_client, ColdStartRoute::Fenced),
    ] {
        let cache = StructuralCache::new(BUDGET_BYTES);
        let loaded = expect_loaded(
            cold_start(client, tv.key_schema(), &cache, oid, CLIENT_ID).await,
            "a cached cold start",
        );
        assert_eq!(loaded.route, route, "fixture: the route");
        let children = &loaded.entry.entry().state.children;
        assert!(
            [ROOT, WORLD].iter().all(|parent| children
                .get(parent)
                .is_some_and(|l| (1..=2).contains(&l.len()))),
            "{route:?}: fixture: the root and WORLD hold short lists; got {children:?}"
        );
        for (parent, list) in children {
            assert_eq!(
                list.capacity(),
                list.len(),
                "{route:?}: the list under {parent} holds {} slots for {} children",
                list.capacity(),
                list.len()
            );
        }
    }
}

/// An unmarked object cannot enumerate its dangling lists: `NoChsComplete`,
/// whether its read loads or answers `TOO_LARGE` — and in the second case
/// with no fenced read, since no read would help.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_of_an_unmarked_object_records_no_chs_complete_without_a_fenced_read() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-unmarked").await;
    unmark(&tv, oid).await;
    let bound = item_bound(&tv, oid).await;

    for (context, max_items, expected_reads) in [
        (
            "within the ceilings",
            SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
            (0, 1),
        ),
        ("above the plain item ceiling", bound - 1, (1, 0)),
    ] {
        let client = client_with_limits(max_items, NO_BYTE_LIMIT).await;
        let metrics = hooked_metrics(context, &[&client]).await;
        let cache = StructuralCache::new(BUDGET_BYTES);

        let reason = expect_uncacheable(
            cold_start(&client, tv.key_schema(), &cache, oid, CLIENT_ID).await,
            context,
        );

        assert_eq!(reason, UncacheableReason::NoChsComplete, "{context}");
        assert_eq!(
            (
                too_large_and_ok_reads(&metrics),
                all_structural_reads(&metrics),
                fence_acquires(&metrics)
            ),
            (expected_reads, 1, 0),
            "{context}: one plain read and no fenced read ((too_large, ok), every read, fence \
             acquires)"
        );
        assert!(
            matches!(
                cache.lookup(oid),
                CacheLookup::Uncacheable {
                    reason: UncacheableReason::NoChsComplete
                }
            ),
            "{context}: the verdict is recorded"
        );
    }
}

// ── The probed route (step 0's cardinalities, the predictor, the weight check)

/// The object's index cardinalities as the read side's probe reports them
/// (`structural_cache_sync` in probe mode), in the form the cold start
/// routes on.
async fn probed_cardinalities(tv: &TestValkey, oid: u64) -> SnapshotCardinalities {
    let probe = read_structural_cache_sync(
        tv.client(),
        tv.key_schema(),
        &StructuralSyncRequest::probe(oid, CLIENT_ID),
    )
    .await
    .expect("the probe must succeed");
    let SyncHeader::Present(facts) = probe.header else {
        panic!("fixture: object {oid} must have a header");
    };
    let SyncChunk::Probe(Some(cards)) = probe.chunk else {
        panic!(
            "a probe of a well-formed object answers a PROBE chunk with cardinalities; got {:?}",
            probe.chunk
        );
    };
    cards.snapshot_cardinalities(facts.chs_complete_marked())
}

/// The probe's cardinalities give the atomic read's own step-2 item bound:
/// what the route is decided on before any full read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_probe_cardinalities_give_the_snapshot_scripts_own_item_bound() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-probe-bound").await;

    let cards = probed_cardinalities(&tv, oid).await;

    assert!(
        cards.children_index.is_some(),
        "a marked object counts its children-list index"
    );
    assert_eq!(
        snapshot_item_bound(&cards),
        item_bound(&tv, oid).await,
        "the bound computed from the probe must be the script's own early-out bound"
    );
    unmark(&tv, oid).await;
    let unmarked = probed_cardinalities(&tv, oid).await;
    assert_eq!(unmarked.children_index, None, "unmarked: not counted");
    assert_eq!(
        snapshot_item_bound(&unmarked),
        item_bound(&tv, oid).await,
        "unmarked, the bound still matches the script's"
    );
}

/// T5 — a fenced entry equals the atomic entry. The plain item ceiling is
/// lowered under the probe's bound, so the probe routes straight to the
/// fenced read; the layer is marked, owns a dangling list and has time
/// samples. The reference is the atomic read for the same consumer at
/// ceilings no layer reaches, decoded, normalized and its dangling set built
/// exactly as a cold start builds it, at the same (quiescent) cut. The
/// entry's `(seq, commit_chain)`, dangling set and state must all be the
/// reference's — and the read must be exactly one, fenced, `ok`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_probed_fenced_entry_equals_the_atomic_entry_at_the_same_cut() {
    /// Ceilings no fixture comes near: the reference read is never refused.
    const HUGE: SnapshotCeilings = SnapshotCeilings {
        max_items: 1 << 40,
        max_bytes: 1 << 40,
    };
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-t5").await;
    let hdr = header(&tv, oid).await;
    let cards = probed_cardinalities(&tv, oid).await;
    let client = fenced_route_client(&tv, oid).await;
    assert!(
        !plain_read_predicted_to_fit(snapshot_item_bound(&cards), plain_ceilings(&client)),
        "fixture: the probe must route the layer to the fenced read"
    );
    let metrics = hooked_metrics("cs-t5", &[&client]).await;
    let cache = StructuralCache::new(BUDGET_BYTES);

    let loaded = expect_loaded(
        cold_start_probed(&client, tv.key_schema(), &cache, oid, CLIENT_ID, cards).await,
        "probed, routed fenced",
    );

    assert_eq!(loaded.route, ColdStartRoute::Fenced);
    assert_eq!(
        (
            fenced_reads(&metrics, "ok"),
            all_structural_reads(&metrics),
            cold_starts_by_route(&metrics)
        ),
        (1, 1, (0, 1, 0)),
        "one path=\"fenced\" outcome=\"ok\" read and no atomic read; one fenced cold start"
    );
    let reference = match read_object_snapshot_slim_with_ceilings(
        tv.client(),
        tv.key_schema(),
        &oid.to_string(),
        SlimSnapshotReadConsumer::StructuralCache,
        HUGE,
    )
    .await
    {
        Ok(SnapshotReadOf::Snapshot(snapshot)) => *snapshot,
        Ok(SnapshotReadOf::TooLarge(too_large)) => {
            panic!("the reference atomic read must answer with the state; got {too_large:?}")
        }
        Err(e) => panic!("the reference atomic read must succeed; got {e:?}"),
    };
    let facts = reference
        .structural
        .clone()
        .expect("the structural_cache consumer's atomic read carries the structural facts");
    let mut state = reference.state.clone();
    let mut dangling: HashSet<u64> = facts.dangling_parents.iter().copied().collect();
    dangling.extend(normalize_state(&mut state));
    let entry = loaded.entry.entry();
    assert!(
        dangling.contains(&DANGLING_PARENT) && !state.time_samples.is_empty(),
        "fixture: the reference must hold the dangling list's parent and time samples, or the \
         comparison below covers neither"
    );
    let header_seq: u64 = hdr["seq"].parse().expect("the header seq is decimal");
    assert_eq!(
        (entry.seq, &entry.commit_chain),
        (reference.cut, &facts.commit_chain),
        "the fenced entry's cut must be the atomic read's"
    );
    assert_eq!(
        (entry.seq, &entry.commit_chain),
        (header_seq, &hdr[HEADER_FIELD_COMMIT_CHAIN]),
        "and the header's (seq, commit_chain)"
    );
    assert_eq!(
        entry.dangling_parents, dangling,
        "the fenced entry's dangling parents must be the atomic entry's (the fan-out's \
         structural facts carry them)"
    );
    assert_eq!(
        entry.state, state,
        "the fenced entry's state must be the atomic entry's, decoded and normalized alike"
    );
}

/// T5's second half, end to end: on a layer above the plain item ceiling,
/// the first structural `UpdateLayer` cold-starts the object on the fenced
/// route, and the next two are served by the cached entry — no second cold
/// start and no new read of any kind on the `structural_cache` consumer.
/// No uncacheable verdict is reached. (Formerly this layer, above the raised
/// item ceiling, was an uncacheable verdict reported once; size alone no
/// longer is one.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_edits_above_the_plain_ceilings_cold_start_fenced_once_then_hit_the_cache() {
    const LAYER: &str = "cs-t5-e2e";
    let tv = TestValkey::setup().await;
    let session_id = create_session(&tv, LAYER).await;
    let oid = create(&layer_store(&tv), &session_id, LAYER, &every_shape_state()).await;
    let client = fenced_route_client(&tv, oid).await;
    let metrics = hooked_metrics(LAYER, &[&client]).await;
    let store = LayerStore::new(client.clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua)
        .with_structural_cache(StructuralCache::with_ttls(BUDGET_BYTES, LONG_TTL, LONG_TTL))
        .with_metrics(Arc::clone(&metrics));
    let edit = |section_id: u64, name: &'static str| {
        let delta = structural(vec![create_cmd(ROOT, section_id, name)]);
        let relay = Some(Bytes::from(delta_to_ompf_bytes(&delta, &[])));
        let (store, session_id) = (&store, session_id.as_str());
        async move {
            tokio::time::timeout(
                WAIT_BUDGET,
                store.update_layer(session_id, LAYER, oid, Some(delta), relay, CLIENT_ID, 1),
            )
            .await
            .expect("the edit must settle within the budget")
        }
    };

    let (first, first_log) = capture(edit(70, "First")).await;
    first.expect("the first edit commits");
    let reads_after_first = all_structural_reads(&metrics);
    edit(71, "Second").await.expect("the second edit commits");
    edit(72, "Third").await.expect("the third edit commits");

    assert_eq!(
        (reads_after_first, fenced_reads(&metrics, "ok")),
        (1, 1),
        "the first edit's cold start read the layer once, fenced"
    );
    assert_eq!(
        cold_starts_by_route(&metrics),
        (0, 1, 0),
        "one fenced cold start, and nothing else (atomic, fenced, uncacheable)"
    );
    assert_eq!(
        all_structural_reads(&metrics),
        reads_after_first,
        "the next two edits are served by the cached entry: no new read"
    );
    assert_eq!(
        (
            lookups(&metrics, "cold"),
            lookups(&metrics, "hit") + lookups(&metrics, "repaired"),
            lookups(&metrics, "reload")
        ),
        (1, 2, 0),
        "one cold lookup, then two lookups that found the entry (cold, hit or repaired, reload)"
    );
    assert_eq!(uncacheable_verdicts(&metrics), BTreeMap::new());
    assert!(
        first_log.with_message(UNCACHEABLE_VERDICT_MSG).is_empty(),
        "a fenced cold start that loads reaches no verdict:\n{}",
        first_log.text
    );
}

/// T7 — one full read. A layer over the plain byte ceiling but under the item
/// bound: the predictor sends the probed cold start straight to the fenced
/// read, which loads it — exactly one `structural_cache` snapshot read, on
/// `path="fenced"`, route `fenced`. Without the predictor (the unprobed cold
/// start, plain read first) the same layer costs two: the plain read refuses
/// on bytes and the fenced read repeats it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_probed_over_the_plain_byte_ceiling_and_under_the_item_bound_reads_once() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-one-read").await;
    let bytes = full_reply_bytes(&tv, oid).await;
    let cards = probed_cardinalities(&tv, oid).await;
    let bound = snapshot_item_bound(&cards);
    let plain = SnapshotCeilings {
        max_items: SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
        max_bytes: bytes.min(bound * K) - 1,
    };
    assert!(
        bound <= plain.max_items && plain.max_bytes < bytes,
        "fixture: under the plain item ceiling, over the plain byte ceiling (bound {bound}, \
         bytes {bytes}, plain {plain:?})"
    );
    assert!(
        !plain_read_predicted_to_fit(bound, plain),
        "fixture: the predictor must route this layer past the plain read ({bound} bound items \
         × {K} > {} bytes)",
        plain.max_bytes
    );
    let client = client_with_limits(plain.max_items, plain.max_bytes).await;
    let hdr = header(&tv, oid).await;

    let metrics = hooked_metrics("cs-one-read-probed", &[&client]).await;
    let cache = StructuralCache::new(BUDGET_BYTES);
    let loaded = expect_loaded(
        cold_start_probed(&client, tv.key_schema(), &cache, oid, CLIENT_ID, cards).await,
        "probed",
    );
    assert_eq!(loaded.route, ColdStartRoute::Fenced);
    assert_eq!(
        (
            all_structural_reads(&metrics),
            too_large_and_ok_reads(&metrics),
            fenced_reads(&metrics, "ok")
        ),
        (1, (0, 0), 1),
        "probed: exactly one full read, the fenced read, and no plain read refusing on bytes"
    );
    assert_eq!(cold_starts(&metrics, "fenced"), 1);
    assert_entry_is_the_stored_cut(&tv, oid, loaded.entry.entry(), &hdr, "probed").await;
    drop(loaded);

    // The predictor disabled: the unprobed cold start reads plain first.
    let metrics = hooked_metrics("cs-one-read-unprobed", &[&client]).await;
    let cache = StructuralCache::new(BUDGET_BYTES);
    let loaded = expect_loaded(
        cold_start(&client, tv.key_schema(), &cache, oid, CLIENT_ID).await,
        "unprobed",
    );
    assert_eq!(loaded.route, ColdStartRoute::Fenced);
    assert_eq!(
        (
            all_structural_reads(&metrics),
            too_large_and_ok_reads(&metrics),
            fenced_reads(&metrics, "ok")
        ),
        (2, (1, 0), 1),
        "without the predictor the same layer reads twice: the plain read refuses on bytes, \
         then the fenced read loads"
    );
}

/// A seeded object made byte-dense by one large inline value, so its bytes
/// per bound item are far above `K`, and the plain ceilings that predict it
/// to fit while its bytes are over them: the predictor-miss fixture.
async fn dense_object(tv: &TestValkey, ls: &LayerStore, layer: &str) -> (u64, SnapshotCeilings) {
    let session_id = create_session(tv, layer).await;
    let oid = create(ls, &session_id, layer, &every_shape_state()).await;
    // Twice K bytes per bound item over the whole layer, and then some: the
    // one value outweighs the layer's priced bound by a wide margin, at any K.
    let dense_len = ((item_bound(tv, oid).await + 64) * K * 2) as usize;
    commit(
        ls,
        &session_id,
        layer,
        oid,
        delta_with_set_field_inline(WORLD, "dense", vec![0x5A; dense_len], 5),
    )
    .await;
    let bytes = full_reply_bytes(tv, oid).await;
    let bound = item_bound(tv, oid).await;
    let plain = SnapshotCeilings {
        max_items: SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
        max_bytes: bound * K,
    };
    assert!(
        plain_read_predicted_to_fit(bound, plain) && plain.max_bytes < bytes,
        "fixture: predicted to fit ({bound} bound items × {K} <= {} bytes), yet over the plain \
         byte ceiling ({bytes} bytes)",
        plain.max_bytes
    );
    (oid, plain)
}

/// T7 — the predictor's miss is counted. A layer denser than `K` bytes per
/// bound item (one large inline value) is predicted to fit the plain read
/// and refuses on bytes: two full reads, the miss on the atomic `too_large`
/// count, and the fenced read loads it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_probed_predictor_miss_reads_twice_and_counts_the_miss() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let (oid, plain) = dense_object(&tv, &ls, "cs-predictor-miss").await;
    let cards = probed_cardinalities(&tv, oid).await;
    let client = client_with_limits(plain.max_items, plain.max_bytes).await;
    let metrics = hooked_metrics("cs-predictor-miss", &[&client]).await;
    let cache = StructuralCache::new(BUDGET_BYTES);

    let loaded = expect_loaded(
        cold_start_probed(&client, tv.key_schema(), &cache, oid, CLIENT_ID, cards).await,
        "a predictor miss",
    );

    assert_eq!(loaded.route, ColdStartRoute::Fenced);
    assert_eq!(
        (
            all_structural_reads(&metrics),
            too_large_and_ok_reads(&metrics),
            fenced_reads(&metrics, "ok")
        ),
        (2, (1, 0), 1),
        "a miss reads twice, and the plain read's refusal is the counted miss (every read, \
         (atomic too_large, atomic ok), fenced ok)"
    );
    assert_eq!(cold_starts_by_route(&metrics), (0, 1, 0));
}

/// T8 — the weight check before any read. When the lower weight bound from
/// the probe's cardinalities is above the per-entry cap, the cold start
/// records `OverWeight` carrying that bound with no full read; at a cap equal
/// to the bound it reads, and the admission judges the real weight.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_probed_lower_weight_bound_over_the_cap_records_over_weight_with_no_read() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-weight-check").await;
    let cards = probed_cardinalities(&tv, oid).await;
    let lower = state_weight_lower_bound(&cards);
    let charged = charged_weight(&normalized_fresh_state(&tv, oid).await);
    assert!(
        lower <= charged,
        "the bound is a lower bound: {lower} <= {charged}"
    );

    // A cap one byte under the bound: refused before any read.
    let metrics = hooked_metrics("cs-weight-check-refused", &[tv.client()]).await;
    let cache = StructuralCache::with_ttls((lower - 1) * 4, LONG_TTL, LONG_TTL);
    assert_eq!(cache.max_entry_bytes(), lower - 1, "fixture: the cap");
    let reason = expect_uncacheable(
        cold_start_probed(tv.client(), tv.key_schema(), &cache, oid, CLIENT_ID, cards).await,
        "lower bound over the cap",
    );
    assert_eq!(reason, UncacheableReason::OverWeight(lower));
    assert_eq!(
        all_structural_reads(&metrics),
        0,
        "the weight check answers before any full read"
    );
    assert_eq!(cold_starts(&metrics, "uncacheable"), 1);
    assert!(
        matches!(
            cache.lookup(oid),
            CacheLookup::Uncacheable {
                reason: UncacheableReason::OverWeight(w)
            } if w == lower
        ),
        "the verdict is recorded as a negative entry"
    );

    // A cap at the bound: the check lets the read run.
    let metrics = hooked_metrics("cs-weight-check-read", &[tv.client()]).await;
    let cache = StructuralCache::with_ttls(lower * 4, LONG_TTL, LONG_TTL);
    let answer =
        cold_start_probed(tv.client(), tv.key_schema(), &cache, oid, CLIENT_ID, cards).await;
    assert_eq!(
        all_structural_reads(&metrics),
        1,
        "a bound within the cap reads; got {answer:?}"
    );
    if charged > lower {
        assert_eq!(
            expect_uncacheable(answer, "the admission judges the real weight"),
            UncacheableReason::OverWeight(charged),
            "over the cap only by its real weight: the admission's verdict, at that weight"
        );
    }
}

/// T8, above the ceilings: the probe routes the layer to the fenced read, and
/// its lower weight bound is over the cap. `OverWeight` at the bound, carrying
/// the probe's size (its bound is over the item ceiling), with no read on any
/// path, no fence acquire, no `P_f` wait and no memory reservation. The
/// control — the same layer under a cap at the bound — takes the fenced read,
/// so the check is what stopped it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_probed_lower_weight_bound_over_the_cap_above_the_ceilings_takes_no_fence_p_f_or_grant(
) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-weight-fenced").await;
    let cards = probed_cardinalities(&tv, oid).await;
    let lower = state_weight_lower_bound(&cards);
    let client = fenced_route_client(&tv, oid).await;
    let metrics = hooked_metrics("cs-weight-fenced", &[&client]).await;
    let cache = StructuralCache::with_ttls((lower - 1) * 4, LONG_TTL, LONG_TTL);

    let (reason, above_ceiling) = expect_uncacheable_sized(
        cold_start_probed(&client, tv.key_schema(), &cache, oid, CLIENT_ID, cards).await,
        "lower bound over the cap, above the ceilings",
    );

    assert_eq!(reason, UncacheableReason::OverWeight(lower));
    assert_eq!(
        above_ceiling,
        Some(probed_estimate(&cards)),
        "the probe's bound is over the item ceiling: the answer carries the probe's size"
    );
    assert_eq!(
        (
            all_structural_reads(&metrics),
            fence_acquires(&metrics),
            fenced_permit_waits(&metrics),
            budget_reservations(&metrics)
        ),
        (0, 0, 0, 0),
        "no read on any path, no fence acquire, no P_f wait, no memory reservation (reads, \
         acquires, P_f waits, reservations)"
    );
    assert_eq!(fenced_capacity(&client, &cache), NOTHING_HELD);

    // Control: a cap at the bound takes the fenced read.
    let cache = StructuralCache::with_ttls(lower * 4, LONG_TTL, LONG_TTL);
    let answer = cold_start_probed(&client, tv.key_schema(), &cache, oid, CLIENT_ID, cards).await;
    assert!(
        fence_acquires(&metrics) >= 1 && fenced_reads(&metrics, "ok") == 1,
        "control: with the bound within the cap the fenced read runs; got {answer:?}"
    );
}

/// T8, new on the unprobed path: the plain read answers `TOO_LARGE`, and the
/// lower weight bound on that answer's own cardinalities is over the cap.
/// `OverWeight` at that bound, carrying the answer's size, before any fenced
/// read: one atomic `too_large`, no fence acquire, no `P_f` wait and no
/// memory reservation. The control — a cap at the bound — takes the fenced
/// read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_unprobed_too_large_lower_weight_bound_over_the_cap_records_over_weight_before_any_fenced_read(
) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-weight-too-large").await;
    let client = fenced_route_client(&tv, oid).await;
    let too_large = read_too_large(&client, tv.key_schema(), oid).await;
    let lower = state_weight_lower_bound(&too_large.cardinalities);
    let size = LayerSize::from_too_large(&too_large, None);
    let metrics = hooked_metrics("cs-weight-too-large", &[&client]).await;
    let cache = StructuralCache::with_ttls((lower - 1) * 4, LONG_TTL, LONG_TTL);
    assert_eq!(cache.max_entry_bytes(), lower - 1, "fixture: the cap");

    let (reason, above_ceiling) = expect_uncacheable_sized(
        cold_start(&client, tv.key_schema(), &cache, oid, CLIENT_ID).await,
        "TOO_LARGE, lower bound over the cap",
    );

    assert_eq!(reason, UncacheableReason::OverWeight(lower));
    assert_eq!(
        above_ceiling,
        Some(size),
        "the answer carries the TOO_LARGE answer's size"
    );
    assert_eq!(
        (
            too_large_and_ok_reads(&metrics),
            all_structural_reads(&metrics),
            fence_acquires(&metrics),
            fenced_permit_waits(&metrics),
            budget_reservations(&metrics)
        ),
        ((1, 0), 1, 0, 0, 0),
        "the plain read only: no fenced read, no acquire, no P_f wait, no memory reservation \
         ((too_large, ok), reads, acquires, P_f waits, reservations)"
    );
    assert_eq!(cold_starts_by_route(&metrics), (0, 0, 1));
    assert!(
        matches!(
            cache.lookup(oid),
            CacheLookup::Uncacheable {
                reason: UncacheableReason::OverWeight(w)
            } if w == lower
        ),
        "the verdict is recorded as a negative entry"
    );

    // Control: a cap at the bound takes the fenced read.
    let cache = StructuralCache::with_ttls(lower * 4, LONG_TTL, LONG_TTL);
    let answer = cold_start(&client, tv.key_schema(), &cache, oid, CLIENT_ID).await;
    assert!(
        fence_acquires(&metrics) >= 1 && fenced_reads(&metrics, "ok") == 1,
        "control: with the bound within the cap the fenced read runs; got {answer:?}"
    );
}

/// Step 1 on the probe: an unmarked object answers `NoChsComplete` with no
/// full read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_probed_unmarked_object_records_no_chs_complete_with_no_read() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-probed-unmarked").await;
    unmark(&tv, oid).await;
    let cards = probed_cardinalities(&tv, oid).await;
    let metrics = hooked_metrics("cs-probed-unmarked", &[tv.client()]).await;
    let cache = StructuralCache::new(BUDGET_BYTES);

    let reason = expect_uncacheable(
        cold_start_probed(tv.client(), tv.key_schema(), &cache, oid, CLIENT_ID, cards).await,
        "unmarked",
    );

    assert_eq!(reason, UncacheableReason::NoChsComplete);
    assert_eq!(all_structural_reads(&metrics), 0, "no full read");
}

/// A superseded header (the overwrite's `superseded` column) answers
/// `Superseded` with the call's token record, for the caller's token check,
/// and stores and records nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_of_a_superseded_object_answers_superseded_with_the_token_and_stores_nothing() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-superseded").await;
    let _: i64 = tv
        .client()
        .pool
        .hset(
            tv.key_schema().obj_header(&oid.to_string()),
            [("superseded", "1")],
        )
        .await
        .expect("HSET obj:hdr superseded must succeed");
    let cache = StructuralCache::new(BUDGET_BYTES);

    let answer = cold_start(tv.client(), tv.key_schema(), &cache, oid, CLIENT_ID).await;

    assert!(
        matches!(&answer, Ok(ColdStart::Superseded { commit_token: Some(t) }) if t == TOKEN_RECORD),
        "a superseded header answers Superseded with the token record; got {answer:?}"
    );
    assert!(
        matches!(cache.lookup(oid), CacheLookup::Miss),
        "nothing is stored or recorded for a superseded object"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_of_an_absent_object_is_not_found() {
    let tv = TestValkey::setup().await;
    let cache = StructuralCache::new(BUDGET_BYTES);

    let answer = cold_start(tv.client(), tv.key_schema(), &cache, 987_654_321, CLIENT_ID).await;

    assert!(
        matches!(answer, Err(ServiceError::NotFound(_))),
        "no header: NotFound, for the caller's re-probe; got {answer:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_on_a_disabled_cache_reads_nothing_and_answers_disabled() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-disabled").await;
    let metrics = hooked_metrics("cs-disabled", &[tv.client()]).await;

    let reason = expect_uncacheable(
        cold_start(
            tv.client(),
            tv.key_schema(),
            &StructuralCache::new(0),
            oid,
            CLIENT_ID,
        )
        .await,
        "a disabled cache",
    );

    assert_eq!(reason, UncacheableReason::Disabled);
    assert_eq!(
        (
            too_large_and_ok_reads(&metrics),
            structural_reads(&metrics, "not_found")
        ),
        ((0, 0), 0),
        "a disabled cache answers before any read"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. The cold-start permits
// ═════════════════════════════════════════════════════════════════════════════
//
// A fenced cold start holds the one-at-a-time cold-start permit and then a
// plain permit (its pair); a plain read holds a plain permit only. The tests
// up to the plain permits' block pin the one-at-a-time permit.

/// Two fenced cold starts of different objects on one cache. A's fenced read
/// has sent its acquire and the reply is held at the proxy, so A holds its
/// pair with its read in flight. B (its own client, its own registry) then
/// finishes its plain read and must not send its acquire until A's read
/// returns. Once A's reply is released, both load on the fenced route.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cold_start_fenced_reads_of_two_objects_run_one_at_a_time() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let a = seeded_object(&tv, &ls, "cs-permit-a").await;
    let b = seeded_object(&tv, &ls, "cs-permit-b").await;
    warm_snapshot_script(&tv, a).await;
    warm_fence_scripts(&tv, &ls, "cs-permit").await;
    let plain_items = item_bound(&tv, a).await.max(item_bound(&tv, b).await) - 1;
    let gate = ReplyGate::new();
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        fence_acquire_rule(fence_key(&tv, a), ProxyAction::HoldReplies(gate.clone())),
    )
    .await;
    let client_a = client_through_with_limits(&proxy, plain_items, NO_BYTE_LIMIT).await;
    let client_b = client_with_limits(plain_items, NO_BYTE_LIMIT).await;
    let metrics_b = hooked_metrics("cs-permit-b", &[&client_b]).await;
    let cache = StructuralCache::new(BUDGET_BYTES);

    let a_task = spawn_cold_start(&client_a, tv.key_schema(), &cache, a, None);
    gate.reply_held().await;
    let mut b_task = spawn_cold_start(&client_b, tv.key_schema(), &cache, b, None);
    assert!(
        wait_until(|| structural_reads(&metrics_b, "too_large") == 1).await,
        "B's plain read must finish within the budget"
    );

    assert!(
        tokio::time::timeout(BLOCKED_WINDOW, &mut b_task)
            .await
            .is_err(),
        "B must wait for the one-at-a-time permit while A's fenced read is in flight"
    );
    assert_eq!(
        (fence_acquires(&metrics_b), fenced_reads(&metrics_b, "ok")),
        (0, 0),
        "B's fenced read must not send its acquire while A holds the permit"
    );
    gate.open();
    for (name, task) in [("A", a_task), ("B", b_task)] {
        assert_eq!(
            expect_loaded(join_cold_start(task, name).await, name).route,
            ColdStartRoute::Fenced,
            "{name}"
        );
    }
    assert_eq!(
        (fenced_reads(&metrics_b, "ok"), proxy.hits()),
        (1, 1),
        "B's fenced read ran after A's; the proxy held exactly A's acquire"
    );
}

/// With the one-at-a-time permit held elsewhere, the plain route and every
/// decision taken before the fenced read still finish: the probe's unmarked
/// verdict, an unmarked `TOO_LARGE`, and the weight check on a `TOO_LARGE`
/// answer (on a second cache, whose own permit is held too). None takes the
/// permit (the plain read's plain permits are free). The control — a fenced
/// cold start under the same held permit stays parked until it is released —
/// shows the held permit does block, so the others finishing is not vacuous.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_plain_route_and_decisions_before_the_fenced_read_finish_while_the_permit_is_held(
) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-no-permit").await;
    let unmarked = seeded_object(&tv, &ls, "cs-no-permit-unmarked").await;
    unmark(&tv, unmarked).await;
    let unmarked_cards = probed_cardinalities(&tv, unmarked).await;
    let unmarked_client = fenced_route_client(&tv, unmarked).await;
    let fenced_client = fenced_route_client(&tv, oid).await;
    let lower = state_weight_lower_bound(
        &read_too_large(&fenced_client, tv.key_schema(), oid)
            .await
            .cardinalities,
    );
    let cache = StructuralCache::new(BUDGET_BYTES);
    let small_cache = StructuralCache::with_ttls((lower - 1) * 4, LONG_TTL, LONG_TTL);
    let keys = tv.key_schema();
    let permit = cache.acquire_cold_start_permit().await;
    let small_permit = small_cache.acquire_cold_start_permit().await;

    let plain = tokio::time::timeout(
        WAIT_BUDGET,
        cold_start(tv.client(), keys, &cache, oid, CLIENT_ID),
    )
    .await
    .expect(
        "the plain route does not take the one-at-a-time permit, so it finishes while that \
         permit is held",
    );
    let probed_unmarked = tokio::time::timeout(
        WAIT_BUDGET,
        cold_start_probed(
            &unmarked_client,
            keys,
            &cache,
            unmarked,
            CLIENT_ID,
            unmarked_cards,
        ),
    )
    .await
    .expect("the probe's unmarked verdict is decided before the permit");
    let unmarked_answer = tokio::time::timeout(
        WAIT_BUDGET,
        cold_start(&unmarked_client, keys, &cache, unmarked, CLIENT_ID),
    )
    .await
    .expect("an unmarked TOO_LARGE is decided before the permit, so it finishes while it is held");
    let over_weight = tokio::time::timeout(
        WAIT_BUDGET,
        cold_start(&fenced_client, keys, &small_cache, oid, CLIENT_ID),
    )
    .await
    .expect("the TOO_LARGE weight check is decided before the permit");

    assert_eq!(expect_loaded(plain, "plain").route, ColdStartRoute::Atomic);
    assert_eq!(
        expect_uncacheable(probed_unmarked, "probed unmarked"),
        UncacheableReason::NoChsComplete
    );
    assert_eq!(
        expect_uncacheable(unmarked_answer, "unmarked"),
        UncacheableReason::NoChsComplete
    );
    assert_eq!(
        expect_uncacheable(over_weight, "TOO_LARGE over the cap"),
        UncacheableReason::OverWeight(lower)
    );
    drop(small_permit);
    let fenced = cold_start(&fenced_client, keys, &cache, oid, CLIENT_ID);
    tokio::pin!(fenced);
    assert!(
        tokio::time::timeout(BLOCKED_WINDOW, fenced.as_mut())
            .await
            .is_err(),
        "control: a fenced cold start must stay parked while the permit is held"
    );
    drop(permit);
    let fenced = tokio::time::timeout(WAIT_BUDGET, fenced)
        .await
        .expect("control: the fenced cold start finishes once the permit is released");
    assert_eq!(
        expect_loaded(fenced, "fenced control").route,
        ColdStartRoute::Fenced
    );
}

// ── The one-at-a-time permit is released on every exit, and held only across
// the read ──
//
// The spec SDD's Mechanism lifecycle: the one-at-a-time cold-start permit is
// held only across one fenced read and released by RAII on every exit. Each
// test below takes A's fenced cold start out by one exit — a failed read, a
// cancelled task, or past the read into its token `HGET` — and then requires
// B's fenced cold start on the same cache to get the permit.

/// Two seeded objects of one shape, and a plain item ceiling one under their
/// step-2 bound, so that a cold start of either at that ceiling takes the
/// fenced route.
async fn fenced_pair(tv: &TestValkey, ls: &LayerStore, layer: &str) -> (u64, u64, u64) {
    let a = seeded_object(tv, ls, &format!("{layer}-a")).await;
    let b = seeded_object(tv, ls, &format!("{layer}-b")).await;
    let plain_items = item_bound(tv, a).await.max(item_bound(tv, b).await) - 1;
    (a, b, plain_items)
}

/// A fenced cold start of `oid` on `cache`, through its own client, must get
/// the permit and load within [`WAIT_BUDGET`].
async fn assert_fenced_cold_start_finishes(
    tv: &TestValkey,
    cache: &StructuralCache,
    oid: u64,
    plain_items: u64,
    context: &str,
) {
    let client = client_with_limits(plain_items, NO_BYTE_LIMIT).await;
    let answer = tokio::time::timeout(
        WAIT_BUDGET,
        cold_start(&client, tv.key_schema(), cache, oid, CLIENT_ID),
    )
    .await
    .unwrap_or_else(|_| {
        panic!("{context}: B's fenced cold start must get the permit — it was never released")
    });
    assert_eq!(
        expect_loaded(answer, context).route,
        ColdStartRoute::Fenced,
        "{context}"
    );
}

/// A's fenced read fails (the proxy answers its acquire with an error), so
/// A's cold start returns the error: the fenced read's capacity — its pair,
/// `P_f` and its memory grant — is released on that exit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_permit_is_released_when_the_fenced_read_fails() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let (a, b, plain_items) = fenced_pair(&tv, &ls, "cs-permit-error").await;
    warm_fence_scripts(&tv, &ls, "cs-permit-error").await;
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        fence_acquire_rule(
            fence_key(&tv, a),
            ProxyAction::Eval(error_reply_script("ERR injected fence-acquire failure")),
        ),
    )
    .await;
    let client_a = client_through_with_limits(&proxy, plain_items, NO_BYTE_LIMIT).await;
    let cache = StructuralCache::new(BUDGET_BYTES);

    let failed = tokio::time::timeout(
        WAIT_BUDGET,
        cold_start(&client_a, tv.key_schema(), &cache, a, CLIENT_ID),
    )
    .await
    .expect("A's cold start must return its read's error");

    assert!(
        matches!(failed, Err(ServiceError::Internal(_))),
        "fixture: A's fenced read fails, and the cold start with it; got {failed:?}"
    );
    assert_eq!(proxy.hits(), 1, "fixture: the proxy failed A's acquire");
    assert_eq!(
        fenced_capacity(&client_a, &cache),
        NOTHING_HELD,
        "after A's fenced read failed, it holds nothing"
    );
    assert_cold_start_permits_free(&cache, "after A's fenced read failed").await;
    assert_fenced_cold_start_finishes(&tv, &cache, b, plain_items, "after A's read failed").await;
}

/// A's fenced read is in flight (its acquire's reply held) and C waits for
/// the permit behind it; both tasks are aborted. Dropping A's future must
/// release everything it holds — the pair, `P_f` and the grant — and dropping
/// C's must leave no claim on them.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cold_start_permit_is_released_when_its_holder_and_a_waiter_are_cancelled() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let (a, b, plain_items) = fenced_pair(&tv, &ls, "cs-permit-cancel").await;
    let c = seeded_object(&tv, &ls, "cs-permit-cancel-c").await;
    warm_snapshot_script(&tv, a).await;
    warm_fence_scripts(&tv, &ls, "cs-permit-cancel").await;
    let gate = ReplyGate::new();
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        fence_acquire_rule(fence_key(&tv, a), ProxyAction::HoldReplies(gate.clone())),
    )
    .await;
    let client_a = client_through_with_limits(&proxy, plain_items, NO_BYTE_LIMIT).await;
    let client_c = client_with_limits(plain_items, NO_BYTE_LIMIT).await;
    let metrics_c = hooked_metrics("cs-permit-cancel-c", &[&client_c]).await;
    let cache = StructuralCache::new(BUDGET_BYTES);
    let a_task = spawn_cold_start(&client_a, tv.key_schema(), &cache, a, None);
    gate.reply_held().await;
    let mut c_task = spawn_cold_start(&client_c, tv.key_schema(), &cache, c, None);
    assert!(
        wait_until(|| structural_reads(&metrics_c, "too_large") == 1).await,
        "C's plain read must finish within the budget"
    );
    assert!(
        tokio::time::timeout(BLOCKED_WINDOW, &mut c_task)
            .await
            .is_err(),
        "fixture: C must be parked on the permit A holds"
    );
    assert_eq!(
        fenced_capacity(&client_a, &cache).p_f_held,
        1,
        "fixture: A holds its P_f across its acquire"
    );

    a_task.abort();
    c_task.abort();
    for (name, task) in [("A", a_task), ("C", c_task)] {
        let joined = task.await;
        assert!(
            joined.as_ref().is_err_and(|e| e.is_cancelled()),
            "fixture: {name} must have been cancelled; got {joined:?}"
        );
    }

    assert_eq!(fenced_capacity(&client_a, &cache), NOTHING_HELD, "A's");
    assert_eq!(fenced_capacity(&client_c, &cache), NOTHING_HELD, "C's");
    assert_cold_start_permits_free(&cache, "after A and C were cancelled").await;
    assert_fenced_cold_start_finishes(&tv, &cache, b, plain_items, "after A and C were cancelled")
        .await;
    gate.open();
}

/// A's fenced cold start has read and been admitted, and its token `HGET` is
/// held at the proxy: B's fenced cold start must get the permit while A is
/// still in that `HGET`. A permit kept through the admission and the token
/// read would park B until A's `HGET` returned — at the latest at the client's
/// command timeout (5 s in the test config), which is why the check is that A
/// is still unfinished when B loads, not merely that B loads.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_permit_is_held_only_across_the_fenced_read_not_the_token_read() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let (a, b, plain_items) = fenced_pair(&tv, &ls, "cs-permit-scope").await;
    let gate = ReplyGate::new();
    let wtok = tv.key_schema().obj_write_token(&a.to_string());
    let hold_token_read: ProxyRule = {
        let gate = gate.clone();
        Arc::new(move |args: &[Vec<u8>]| {
            let name = args.first()?.to_ascii_uppercase();
            (name == b"HGET" && args.get(1).map(Vec::as_slice) == Some(wtok.as_bytes()))
                .then(|| ProxyAction::HoldReplies(gate.clone()))
        })
    };
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), hold_token_read).await;
    let client_a = client_through_with_limits(&proxy, plain_items, NO_BYTE_LIMIT).await;
    let cache = StructuralCache::new(BUDGET_BYTES);
    let a_task = spawn_cold_start(&client_a, tv.key_schema(), &cache, a, None);
    gate.reply_held().await;

    assert_fenced_cold_start_finishes(&tv, &cache, b, plain_items, "while A's HGET is held").await;
    assert!(
        !a_task.is_finished(),
        "A must still be in its held token read when B loads; A had already ended, so B got the \
         permit only once A's cold start ended — it was held past the fenced read"
    );

    gate.open();
    let a_loaded = expect_loaded(join_cold_start(a_task, "A").await, "A");
    assert_eq!(
        (
            a_loaded.route,
            a_loaded.commit_token.as_deref(),
            proxy.hits()
        ),
        (ColdStartRoute::Fenced, Some(TOKEN_RECORD), 1),
        "fixture: A loaded on the fenced route, and the held HGET was its token read"
    );
}

// ── The plain permits ────────────────────────────────────────────────────────
//
// Every whole-layer cold-start read holds one of the cache's two plain
// permits across that read; a fenced read takes the one-at-a-time permit
// first, then a plain permit. These tests park reads without a proxy: the
// test task polls each cold start itself, so a cold start polled until it has
// issued its read keeps its permits, its reply unread, until the test polls
// it again. Polls run under `tokio::task::coop::unconstrained`, so an
// exhausted coop budget never answers `Pending` in place of a permit wait,
// and on a current-thread runtime of their own (`snapshot_read_support`'s
// [`on_current_thread`], with each scenario's clients connected inside it),
// so no poll completes the read it issues.
//
// Two premises, from `cold_start`: nothing it does before its plain permit
// awaits (the ticket, the ceilings and the metrics handle are synchronous),
// so its first poll takes that permit or queues for it; and the poll that
// sees its plain read's reply records that read's `structural_cache` outcome
// and runs on, with no await between (the `TOO_LARGE` checks and the hook's
// construction are synchronous), to its fenced read's pair waits — so a
// fenced cold start polled until its plain read's `too_large` tick is
// waiting for, or holding, those permits. Since a permit is held across a
// read, every poll ends with each permit it took still held: sampling the
// in-flight seam after every poll sees each peak.
//
// Every wait is bounded by [`WAIT_BUDGET`] (20 s). Summed at full budget, the
// longest test, the decider's pin, waits six of them (120 s), each
// unreachable on a healthy store; a healthy run pays milliseconds.

/// A cold start the test polls itself.
type Polled<'a> = Pin<Box<dyn Future<Output = Result<ColdStart, ServiceError>> + Send + 'a>>;

/// No cold-start read in flight.
const NONE_IN_FLIGHT: ColdStartReadsInFlight = ColdStartReadsInFlight {
    plain: 0,
    fenced: 0,
    raised: 0,
};
/// One fenced read in flight (its pair), and no plain one.
const ONE_FENCED: ColdStartReadsInFlight = ColdStartReadsInFlight {
    plain: 0,
    fenced: 1,
    raised: 0,
};
/// One plain read in flight, and no fenced one.
const ONE_PLAIN: ColdStartReadsInFlight = ColdStartReadsInFlight {
    plain: 1,
    fenced: 0,
    raised: 0,
};
/// Two plain reads in flight — both plain permits — and no fenced one.
const TWO_PLAIN: ColdStartReadsInFlight = ColdStartReadsInFlight {
    plain: 2,
    fenced: 0,
    raised: 0,
};

/// A cold start of `oid` on `cache` through `client`, not yet polled.
fn polled<'a>(
    client: &'a ValkeyClient,
    keys: &'a KeySchema,
    cache: &'a StructuralCache,
    oid: u64,
) -> Polled<'a> {
    Box::pin(cold_start(client, keys, cache, oid, CLIENT_ID))
}

/// Poll `fut` once, under `unconstrained`.
async fn poll_once<F: Future + Unpin>(fut: &mut F) -> Poll<F::Output> {
    tokio::task::coop::unconstrained(std::future::poll_fn(|cx| {
        Poll::Ready(Pin::new(&mut *fut).poll(cx))
    }))
    .await
}

/// The cache's cold-start reads in flight now.
fn in_flight(cache: &StructuralCache) -> ColdStartReadsInFlight {
    cache.cold_start_reads_in_flight_for_test()
}

/// Assert that no cold-start read holds a permit of `cache` now: none is
/// counted in flight, and both plain permits and the one-at-a-time permit can
/// be taken at once (each first poll answers, the permits held until all
/// three have). Shared with `structural_cache_audit_test`.
pub(crate) async fn assert_cold_start_permits_free(cache: &StructuralCache, context: &str) {
    assert_eq!(
        in_flight(cache),
        NONE_IN_FLIGHT,
        "{context}: no cold-start read may be counted in flight"
    );
    let mut first_plain = Box::pin(cache.acquire_plain_cold_start_read_permit());
    let mut second_plain = Box::pin(cache.acquire_plain_cold_start_read_permit());
    let mut one_at_a_time = Box::pin(cache.acquire_cold_start_permit());
    let first_plain = poll_once(&mut first_plain).await;
    let second_plain = poll_once(&mut second_plain).await;
    let one_at_a_time = poll_once(&mut one_at_a_time).await;
    assert_eq!(
        (
            first_plain.is_ready(),
            second_plain.is_ready(),
            one_at_a_time.is_ready()
        ),
        (true, true, true),
        "{context}: both plain permits and the one-at-a-time permit must be free \
         (plain, plain, one-at-a-time)"
    );
}

/// Poll `start` until `reached` holds after a poll, pausing [`POLL_PAUSE`]
/// between polls. Panics, naming `what`, if the cold start finishes first or
/// [`WAIT_BUDGET`] runs out. Asserts first that it runs on a current-thread
/// runtime ([`on_current_thread`]), the premise of every stop it makes.
async fn drive_until(start: &mut Polled<'_>, what: &str, reached: impl Fn() -> bool) {
    assert_eq!(
        tokio::runtime::Handle::current().runtime_flavor(),
        tokio::runtime::RuntimeFlavor::CurrentThread,
        "fixture: a hand-polled cold start stops between two store awaits only inside \
         `on_current_thread`, with its client connected there; on a multi-thread runtime a store \
         reply can land inside the poll that sent its request, and the cold start runs past the \
         point named {what:?}"
    );
    let deadline = Instant::now() + WAIT_BUDGET;
    loop {
        if let Poll::Ready(answer) = poll_once(start).await {
            panic!("{what}: the cold start finished first; got {answer:?}");
        }
        if reached() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "{what}: not reached within the budget"
        );
        tokio::time::sleep(POLL_PAUSE).await;
    }
}

/// Wait, up to [`WAIT_BUDGET`], until `reached` holds; whether it did.
async fn wait_until(reached: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + WAIT_BUDGET;
    while !reached() {
        if Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(POLL_PAUSE).await;
    }
    true
}

/// Poll every cold start in `starts` until all have finished, requiring after
/// every poll that the cache's reads in flight stay within the bound: at most
/// two (the decider's constant), at most one of them fenced, and no raised
/// read (no cold start takes it). Returns the answers in `starts`' order and
/// the most fenced reads seen in flight at once after one of its polls.
async fn drive_all_within_the_bound(
    cache: &StructuralCache,
    starts: Vec<Polled<'_>>,
) -> (Vec<Result<ColdStart, ServiceError>>, usize) {
    let mut starts: Vec<Option<Polled<'_>>> = starts.into_iter().map(Some).collect();
    let mut answers: Vec<Option<Result<ColdStart, ServiceError>>> =
        starts.iter().map(|_| None).collect();
    let mut most_fenced = 0;
    let deadline = Instant::now() + WAIT_BUDGET;
    while starts.iter().any(Option::is_some) {
        for (slot, answer) in starts.iter_mut().zip(answers.iter_mut()) {
            let Some(start) = slot.as_mut() else {
                continue;
            };
            let polled = poll_once(start).await;
            let now = in_flight(cache);
            assert!(
                now.plain + now.fenced + now.raised <= 2 && now.fenced <= 1 && now.raised == 0,
                "at most two cold-start reads run at once, at most one of them fenced, and none \
                 raised; got {now:?}"
            );
            most_fenced = most_fenced.max(now.fenced);
            if let Poll::Ready(done) = polled {
                *answer = Some(done);
                *slot = None;
            }
        }
        assert!(
            Instant::now() < deadline,
            "every cold start must finish within the budget"
        );
        tokio::time::sleep(POLL_PAUSE).await;
    }
    let answers = answers
        .into_iter()
        .map(|answer| answer.expect("every slot emptied holds its answer"))
        .collect();
    (answers, most_fenced)
}

/// `n` seeded objects of one shape, and a plain item ceiling one under their
/// step-2 bound, so that a cold start of any of them at that ceiling takes
/// the fenced route.
async fn fenced_objects(
    tv: &TestValkey,
    ls: &LayerStore,
    layer: &str,
    n: usize,
) -> (Vec<u64>, u64) {
    let mut oids = Vec::with_capacity(n);
    let mut bound = 0;
    for i in 0..n {
        let oid = seeded_object(tv, ls, &format!("{layer}-{i}")).await;
        bound = bound.max(item_bound(tv, oid).await);
        oids.push(oid);
    }
    (oids, bound - 1)
}

/// A client at the shared item ceiling, for within-ceiling objects.
async fn default_ceiling_client() -> ValkeyClient {
    client_with_limits(SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT, NO_BYTE_LIMIT).await
}

/// Three plain cold starts of three objects at once on one cache: the first
/// two take the two plain permits and issue their reads; the third waits,
/// and stays parked while those two reads are held unread. Driven to the end
/// together, no more than two reads are ever in flight, and all three load.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_of_three_objects_runs_at_most_two_reads_at_once_and_all_complete() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let mut oids = Vec::new();
    for i in 0..3 {
        oids.push(seeded_object(&tv, &ls, &format!("cs-plain-bound-{i}")).await);
    }
    warm_snapshot_script(&tv, oids[0]).await;
    let keys = tv.key_schema().clone();

    on_current_thread(move || async move {
        let client = default_ceiling_client().await;
        let cache = StructuralCache::new(BUDGET_BYTES);
        let mut starts: Vec<Polled<'_>> = oids
            .iter()
            .map(|&oid| polled(&client, &keys, &cache, oid))
            .collect();

        for (i, start) in starts.iter_mut().enumerate() {
            assert!(
                poll_once(start).await.is_pending(),
                "fixture: cold start {i} must be waiting for its read or a permit"
            );
        }
        let after_first_polls = in_flight(&cache);
        let third_while_parked = tokio::time::timeout(BLOCKED_WINDOW, &mut starts[2]).await;

        assert!(
            third_while_parked.is_err(),
            "the third cold start must wait for a plain permit while the first two hold both \
             across their reads; it finished with {third_while_parked:?}"
        );
        assert_eq!(
            after_first_polls, TWO_PLAIN,
            "the first two cold starts hold the two plain permits across their reads"
        );
        let (answers, _) = drive_all_within_the_bound(&cache, starts).await;
        for (i, answer) in answers.into_iter().enumerate() {
            assert_eq!(
                expect_loaded(answer, &format!("cold start {i}")).route,
                ColdStartRoute::Atomic,
                "cold start {i}"
            );
        }
        assert_eq!(
            in_flight(&cache),
            NONE_IN_FLIGHT,
            "every permit is released once all three have loaded"
        );
    })
    .await;
}

/// Three fenced cold starts of three objects. A's fenced read is issued and
/// parked: one fenced read in flight, holding the one-at-a-time permit and a
/// plain permit. B then finishes its plain read, and C its own on the plain
/// permit B's plain read released; each then waits for the one-at-a-time
/// permit holding no plain permit, so A's stays the only read in flight.
/// Driven to the end together, no second fenced read ever runs, and all
/// three load on the fenced route.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_fenced_reads_of_three_objects_run_one_at_a_time_and_the_queued_hold_no_plain_permit(
) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let (oids, plain_items) = fenced_objects(&tv, &ls, "cs-fenced-bound", 3).await;
    warm_snapshot_script(&tv, oids[0]).await;
    warm_fence_scripts(&tv, &ls, "cs-fenced-bound").await;
    let keys = tv.key_schema().clone();

    on_current_thread(move || async move {
        let client = client_with_limits(plain_items, NO_BYTE_LIMIT).await;
        let metrics = hooked_metrics("cs-fenced-bound", &[&client]).await;
        let plain_reads_done = || structural_reads(&metrics, "too_large");
        let cache = StructuralCache::new(BUDGET_BYTES);
        let mut a = polled(&client, &keys, &cache, oids[0]);
        let mut b = polled(&client, &keys, &cache, oids[1]);
        let mut c = polled(&client, &keys, &cache, oids[2]);

        drive_until(&mut a, "A's fenced read in flight", || {
            in_flight(&cache).fenced == 1
        })
        .await;
        drive_until(&mut b, "B's plain read", || plain_reads_done() == 2).await;
        let with_b_queued = in_flight(&cache);
        drive_until(
            &mut c,
            "C's plain read, which needs the plain permit A's fenced read does not hold, while \
             B waits for the one-at-a-time permit — a queued fenced cold start must hold no \
             plain permit",
            || plain_reads_done() == 3,
        )
        .await;
        let with_c_queued = in_flight(&cache);

        assert_eq!(
            with_b_queued, ONE_FENCED,
            "B waits for the one-at-a-time permit A holds: A's is the only fenced read in \
             flight, and B released its plain permit"
        );
        assert_eq!(
            with_c_queued, ONE_FENCED,
            "C waits behind A too, holding no plain permit either"
        );
        let (answers, most_fenced) = drive_all_within_the_bound(&cache, vec![a, b, c]).await;
        for (name, answer) in ["A", "B", "C"].into_iter().zip(answers) {
            assert_eq!(
                expect_loaded(answer, name).route,
                ColdStartRoute::Fenced,
                "{name}"
            );
        }
        assert_eq!(
            (most_fenced, fenced_reads(&metrics, "ok")),
            (1, 3),
            "three fenced reads, one at a time"
        );
        assert_eq!(in_flight(&cache), NONE_IN_FLIGHT);
    })
    .await;
}

/// T6(b), the decider's pin (amended Q2 ruling): a plain cold start proceeds
/// while fenced cold starts are queued. A's fenced read is in flight, parked,
/// holding the one-at-a-time permit and a plain permit; B has finished its
/// plain read and waits for the one-at-a-time permit; C and E then do the
/// same — three queued. D, a within-ceiling object, must load on the plain
/// route while all four stay where they are. Were a fenced cold start to take
/// its plain permit before the one-at-a-time permit, B would hold the second
/// plain permit while it waited, and C's and E's plain reads and D would wait
/// behind it until A's read ended — which, parked, it never does.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_plain_proceeds_while_three_fenced_cold_starts_are_queued() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let (oids, plain_items) = fenced_objects(&tv, &ls, "cs-plain-past-fenced", 4).await;
    let d = seeded_object(&tv, &ls, "cs-plain-past-fenced-d").await;
    warm_snapshot_script(&tv, d).await;
    warm_fence_scripts(&tv, &ls, "cs-plain-past-fenced").await;
    let keys = tv.key_schema().clone();

    on_current_thread(move || async move {
        let client = client_with_limits(plain_items, NO_BYTE_LIMIT).await;
        let d_client = default_ceiling_client().await;
        let metrics = hooked_metrics("cs-plain-past-fenced", &[&client]).await;
        let plain_reads_done = || structural_reads(&metrics, "too_large");
        let cache = StructuralCache::new(BUDGET_BYTES);
        let mut a = polled(&client, &keys, &cache, oids[0]);
        let mut b = polled(&client, &keys, &cache, oids[1]);
        drive_until(&mut a, "A's fenced read in flight", || {
            in_flight(&cache).fenced == 1
        })
        .await;
        drive_until(&mut b, "B's plain read", || plain_reads_done() == 2).await;
        let queued: Vec<_> = [oids[2], oids[3]]
            .into_iter()
            .map(|oid| spawn_cold_start(&client, &keys, &cache, oid, None))
            .collect();
        // Not asserted here: were B holding a plain permit, C's and E's plain
        // reads could not run, and the check on D below names the defect.
        let c_and_e_queued = wait_until(|| plain_reads_done() == 4).await;

        let d_answer = tokio::time::timeout(
            WAIT_BUDGET,
            cold_start(&d_client, &keys, &cache, d, CLIENT_ID),
        )
        .await;

        let d_answer = d_answer.unwrap_or_else(|_| {
            panic!(
                "D's plain cold start must load while A's fenced read is in flight and B, C and \
                 E are queued for the one-at-a-time permit; it waited behind them, so a queued \
                 fenced cold start holds a plain permit (it must take the one-at-a-time permit \
                 first)"
            )
        });
        assert_eq!(expect_loaded(d_answer, "D").route, ColdStartRoute::Atomic);
        assert!(
            c_and_e_queued
                && queued.iter().all(|t| !t.is_finished())
                && poll_once(&mut b).await.is_pending(),
            "fixture: B, C and E, past their plain reads, wait behind A's fenced read"
        );
        assert_eq!(
            in_flight(&cache),
            ONE_FENCED,
            "fixture: A's fenced read is the one read in flight"
        );
        let (a_answer, b_answer, queued_answers) = tokio::time::timeout(WAIT_BUDGET, async {
            let a_answer = (&mut a).await;
            let b_answer = (&mut b).await;
            let mut queued_answers = Vec::new();
            for task in queued {
                queued_answers.push(task.await.expect("the cold-start task must not panic"));
            }
            (a_answer, b_answer, queued_answers)
        })
        .await
        .expect("A, then B, C and E, must finish once A is polled again");
        let mut answers = vec![("A", a_answer), ("B", b_answer)];
        answers.extend(["C", "E"].into_iter().zip(queued_answers));
        for (name, answer) in answers {
            assert_eq!(
                expect_loaded(answer, name).route,
                ColdStartRoute::Fenced,
                "{name}"
            );
        }
    })
    .await;
}

/// Four plain cold starts of four objects: P1 and P2 hold the two plain
/// permits, their reads parked; P3 and P4 wait, in that order. P3 is dropped
/// while it waits, and P1 while it holds its permit mid-read. P1's permit
/// must be released and passed to P4, not lost to P3's abandoned place in the
/// queue, so P4 loads while P2 still holds the other permit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_dropped_holding_or_waiting_for_a_plain_permit_releases_it_to_the_next_waiter() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let mut oids = Vec::new();
    for i in 1..=4 {
        oids.push(seeded_object(&tv, &ls, &format!("cs-plain-drop-{i}")).await);
    }
    warm_snapshot_script(&tv, oids[0]).await;
    let keys = tv.key_schema().clone();

    on_current_thread(move || async move {
        let client = default_ceiling_client().await;
        let cache = StructuralCache::new(BUDGET_BYTES);
        let mut starts: Vec<Polled<'_>> = oids
            .iter()
            .map(|&oid| polled(&client, &keys, &cache, oid))
            .collect();
        for (i, start) in starts.iter_mut().enumerate() {
            assert!(
                poll_once(start).await.is_pending(),
                "fixture: P{} must be waiting for its read or a permit",
                i + 1
            );
        }
        assert_eq!(
            in_flight(&cache),
            TWO_PLAIN,
            "fixture: P1 and P2 hold the two plain permits"
        );
        let mut starts = starts.into_iter();
        let (p1, mut p2, p3, mut p4) = (
            starts.next().unwrap(),
            starts.next().unwrap(),
            starts.next().unwrap(),
            starts.next().unwrap(),
        );

        drop(p3);
        drop(p1);
        let after_drops = in_flight(&cache);
        let p4_answer = tokio::time::timeout(WAIT_BUDGET, &mut p4).await;

        let p4_answer = p4_answer.unwrap_or_else(|_| {
            panic!(
                "P4 must get the plain permit P1 held when it was dropped mid-read, while P2 \
                 holds the other; a dropped holder or waiter kept its claim"
            )
        });
        assert_eq!(expect_loaded(p4_answer, "P4").route, ColdStartRoute::Atomic);
        assert_eq!(
            after_drops, ONE_PLAIN,
            "dropping P1 mid-read released its read: only P2's is in flight"
        );
        let p2_answer = tokio::time::timeout(WAIT_BUDGET, &mut p2)
            .await
            .expect("P2 must finish once polled");
        assert_eq!(expect_loaded(p2_answer, "P2").route, ColdStartRoute::Atomic);
        assert_eq!(in_flight(&cache), NONE_IN_FLIGHT);
    })
    .await;
}

/// A fenced cold start R dropped between its two permit waits: holding the
/// one-at-a-time permit, waiting for a plain permit. P's plain read holds one
/// plain permit and R's plain read the other; the test's own claim queues for
/// a plain permit first, so when R's plain read returns its permit goes to
/// that claim, and R takes the one-at-a-time permit and waits. R must hold
/// that permit while it waits (it is taken first), and dropping R must
/// release it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_fenced_dropped_waiting_for_its_plain_permit_releases_the_one_at_a_time_permit()
{
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let p = seeded_object(&tv, &ls, "cs-fenced-drop-p").await;
    let r = seeded_object(&tv, &ls, "cs-fenced-drop-r").await;
    warm_snapshot_script(&tv, r).await;
    let plain_items = item_bound(&tv, r).await - 1;
    let keys = tv.key_schema().clone();

    on_current_thread(move || async move {
        let p_client = default_ceiling_client().await;
        let r_client = client_with_limits(plain_items, NO_BYTE_LIMIT).await;
        let metrics = hooked_metrics("cs-fenced-drop", &[&r_client]).await;
        let cache = StructuralCache::new(BUDGET_BYTES);
        let mut p_start = polled(&p_client, &keys, &cache, p);
        let mut r_start = polled(&r_client, &keys, &cache, r);
        assert!(poll_once(&mut p_start).await.is_pending());
        assert!(poll_once(&mut r_start).await.is_pending());
        assert_eq!(
            in_flight(&cache),
            TWO_PLAIN,
            "fixture: P's read and R's plain read hold the two plain permits"
        );
        let mut claim = Box::pin(cache.acquire_plain_cold_start_read_permit());
        assert!(
            poll_once(&mut claim).await.is_pending(),
            "fixture: the test's claim queues first for a plain permit"
        );
        drive_until(&mut r_start, "R's plain read", || {
            structural_reads(&metrics, "too_large") == 1
        })
        .await;
        let mut one_at_a_time = Box::pin(cache.acquire_cold_start_permit());

        assert!(
            poll_once(&mut one_at_a_time).await.is_pending(),
            "R, waiting for a plain permit after its plain read, must hold the one-at-a-time \
             permit, which a fenced read takes first"
        );
        drop(r_start);
        assert!(
            tokio::time::timeout(WAIT_BUDGET, &mut one_at_a_time)
                .await
                .is_ok(),
            "dropping R while it waited for its plain permit must release the one-at-a-time \
             permit it held"
        );
        let claimed = tokio::time::timeout(WAIT_BUDGET, &mut claim)
            .await
            .expect("fixture: the test's claim got R's plain permit");
        drop(claimed);
        let p_answer = tokio::time::timeout(WAIT_BUDGET, &mut p_start)
            .await
            .expect("P must finish once polled");
        assert_eq!(expect_loaded(p_answer, "P").route, ColdStartRoute::Atomic);
        assert_eq!(
            in_flight(&cache),
            NONE_IN_FLIGHT,
            "R left no read counted in flight"
        );
    })
    .await;
}

/// A cold start's permits are held across its read only, not through the
/// admission and the token `HGET` after it, on either route. A plain cold
/// start is polled until its read has returned (its `ok` tick), and a fenced
/// one until its fenced read has; the poll that sees that reply runs on to
/// the token `HGET`, whose reply cannot arrive within it on this runtime. With
/// each still pending there, every cold-start permit, `P_f` and the memory
/// grant must be free.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_releases_its_read_permits_before_the_token_read_on_both_routes() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-permit-scope-polled").await;
    warm_snapshot_script(&tv, oid).await;
    warm_fence_scripts(&tv, &ls, "cs-permit-scope-polled").await;
    let fenced_route_items = item_bound(&tv, oid).await - 1;
    let keys = tv.key_schema().clone();

    on_current_thread(move || async move {
        for (route, max_items, path, expected) in [
            (
                "plain",
                SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
                "atomic",
                ColdStartRoute::Atomic,
            ),
            (
                "fenced",
                fenced_route_items,
                "fenced",
                ColdStartRoute::Fenced,
            ),
        ] {
            let client = client_with_limits(max_items, NO_BYTE_LIMIT).await;
            let metrics = hooked_metrics(route, &[&client]).await;
            let cache = StructuralCache::new(BUDGET_BYTES);
            let mut start = polled(&client, &keys, &cache, oid);

            drive_until(
                &mut start,
                &format!("the {route} route's loading read"),
                || structural_reads_on(&metrics, path, "ok") == 1,
            )
            .await;
            assert_cold_start_permits_free(
                &cache,
                &format!("the {route} route, pending in its token HGET"),
            )
            .await;
            assert_eq!(
                fenced_capacity(&client, &cache),
                NOTHING_HELD,
                "the {route} route, pending in its token HGET, holds no P_f and no memory"
            );

            let answer = tokio::time::timeout(WAIT_BUDGET, &mut start)
                .await
                .unwrap_or_else(|_| panic!("the {route} route's cold start must finish"));
            let loaded = expect_loaded(answer, route);
            assert_eq!(
                (loaded.route, loaded.commit_token.as_deref()),
                (expected, Some(TOKEN_RECORD)),
                "{route}: fixture: the route, and the token record the pending step read"
            );
        }
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. The fenced read's capacity (T6, T16, T17)
// ═════════════════════════════════════════════════════════════════════════════
//
// A fenced cold start takes, in order, the cache's pair, a fenced-class
// catch-up read permit (`P_f`, the client's `catchup_fenced_read_gate`) and a
// grant from the client's catch-up memory budget `T`; it gives all of them
// back during every `RETRY_AFTER` sleep, and its waiting is capped against
// the call's deadline. Every client here is its own, so its gate and budget
// are the test's alone. The order inside one hook event (the grant adjusted
// before `P_f` is given back) has no await between its steps, so no test here
// can observe it; the hook's unit tests pin the end state instead.

/// T6(a) — with every `P_f` permit held by the test, a fenced cold start
/// takes its pair and waits for `P_f` holding no memory, and sends no
/// acquire, until `P_f` frees; it then loads, its `P_f` wait observed under
/// `consumer="structural_cache",class="fenced"`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_cold_start_waiting_for_p_f_holds_its_pair_and_no_memory_until_p_f_frees() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-p-f-held").await;
    let cards = probed_cardinalities(&tv, oid).await;
    let client = fenced_route_client(&tv, oid).await;
    let metrics = hooked_metrics("cs-p-f-held", &[&client]).await;
    let cache = StructuralCache::new(BUDGET_BYTES);
    let gate = &client.catchup_fenced_read_gate;
    let mut held = Vec::new();
    for _ in 0..gate.limit() {
        held.push(
            gate.acquire(
                tokio::time::Instant::now() + LONG_TTL,
                SnapshotReadConsumer::Catchup,
            )
            .await
            .expect("fixture: the test takes every P_f permit"),
        );
    }
    let waiting = FencedCapacity {
        in_flight: ONE_FENCED,
        p_f_held: gate.limit(),
        reserved_bytes: 0,
        overcommit_bytes: 0,
    };

    let task = spawn_cold_start(&client, tv.key_schema(), &cache, oid, Some(cards));
    assert!(
        wait_until(|| in_flight(&cache).fenced == 1).await,
        "the fenced cold start must take its pair"
    );
    let watch_until = Instant::now() + BLOCKED_WINDOW;
    while Instant::now() < watch_until {
        assert_eq!(
            fenced_capacity(&client, &cache),
            waiting,
            "waiting for P_f, the cold start holds its pair and no memory (a grant is never \
             taken before P_f)"
        );
        assert!(!task.is_finished(), "it must wait for P_f");
        tokio::time::sleep(POLL_PAUSE).await;
    }
    assert_eq!(fence_acquires(&metrics), 0, "no acquire before P_f");
    drop(held);

    let loaded = expect_loaded(join_cold_start(task, "once P_f frees").await, "P_f freed");
    assert_eq!(loaded.route, ColdStartRoute::Fenced);
    drop(loaded);
    assert_eq!(
        fenced_permit_waits(&metrics),
        1,
        "the P_f wait is observed once, under the structural_cache consumer's fenced class"
    );
    assert_eq!(fenced_capacity(&client, &cache), NOTHING_HELD);
}

/// T6(c) — during a `RETRY_AFTER` sleep the cold start holds nothing. A live
/// reader lease nobody holds is forged into A's fence key, so every acquire
/// of A answers `RETRY_AFTER` (`epoch`, a hint of one `poll`). Between
/// acquires, A must hold no pair, no `P_f` and no grant, and B — a fenced
/// cold start of another object on the same cache and client — must take the
/// one-at-a-time permit, `P_f` and a grant, and load while A still waits.
/// Once the forged lease is gone, A loads too.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_cold_start_holds_no_pair_p_f_or_grant_through_a_retry_after_sleep() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let a = seeded_object(&tv, &ls, "cs-retry-a").await;
    let b = seeded_object(&tv, &ls, "cs-retry-b").await;
    let (a_cards, b_cards) = (
        probed_cardinalities(&tv, a).await,
        probed_cardinalities(&tv, b).await,
    );
    let plain_items = item_bound(&tv, a).await.max(item_bound(&tv, b).await) - 1;
    let client = client_with_limits(plain_items, NO_BYTE_LIMIT).await;
    let metrics = hooked_metrics("cs-retry", &[&client]).await;
    let cache = StructuralCache::new(BUDGET_BYTES);
    forge_live_lease(&tv, a).await;

    let a_task = spawn_cold_start(&client, tv.key_schema(), &cache, a, Some(a_cards));
    assert!(
        wait_until(|| fence_acquires_of(&metrics, "retry_after_epoch") >= 1).await,
        "fixture: A's acquire must meet the forged lease"
    );
    wait_for_capacity(
        &client,
        &cache,
        NOTHING_HELD,
        "A, sleeping out its RETRY_AFTER, must give back its pair, P_f and grant",
    )
    .await;
    let b_answer = tokio::time::timeout(
        WAIT_BUDGET,
        cold_start_probed(&client, tv.key_schema(), &cache, b, CLIENT_ID, b_cards),
    )
    .await
    .unwrap_or_else(|_| {
        panic!(
            "B's fenced cold start must take the one-at-a-time permit, P_f and a grant while A \
             sleeps; it waited behind A, so A holds them through its sleep"
        )
    });

    assert_eq!(expect_loaded(b_answer, "B").route, ColdStartRoute::Fenced);
    assert!(
        !a_task.is_finished(),
        "fixture: A still waits on the forged lease"
    );
    remove_forged_lease(&tv, a).await;
    let a_loaded = expect_loaded(join_cold_start(a_task, "A").await, "A");
    assert_eq!(a_loaded.route, ColdStartRoute::Fenced);
    drop(a_loaded);
    assert_eq!(fenced_capacity(&client, &cache), NOTHING_HELD);
}

/// T6(d) — cancelled mid-read. A fenced cold start's fan-out is held back
/// (the fault seam delays its first command) while it holds its lease, the
/// pair, `P_f` and its grant; the task is then aborted. Everything must be
/// given back — the pair, `P_f`, the grant — and the lease released (the
/// lease guard's background release), and a second fenced cold start on the
/// same cache and client then loads.
///
/// Needs `test-fault-injection` (the read's fault seam).
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_cold_start_cancelled_mid_read_gives_back_its_pair_p_f_grant_and_lease() {
    use live_edit::storage::diff_read::{
        ObjectSnapshotReadFaultInjector, ObjectSnapshotReadFaultTarget, ObjectSnapshotReader,
    };
    /// Longer than the test: the read is cancelled inside it.
    const FAN_OUT_DELAY: Duration = Duration::from_secs(60);
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let (a, b, plain_items) = fenced_pair(&tv, &ls, "cs-cancel-mid-read").await;
    let a_cards = probed_cardinalities(&tv, a).await;
    let client = client_with_limits(plain_items, NO_BYTE_LIMIT).await;
    let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
    injector.push_delay_at(
        ObjectSnapshotReadFaultTarget::nth_command(ObjectSnapshotReader::FencedFanout, 0),
        FAN_OUT_DELAY,
    );
    client.install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));
    let cache = StructuralCache::new(BUDGET_BYTES);
    let reading = FencedCapacity {
        in_flight: ONE_FENCED,
        p_f_held: 1,
        reserved_bytes: expected_grant_bytes(&client, probed_estimate(&a_cards).bytes),
        overcommit_bytes: 0,
    };

    let task = spawn_cold_start(&client, tv.key_schema(), &cache, a, Some(a_cards));
    let deadline = Instant::now() + WAIT_BUDGET;
    while lease_fields(&tv, a).await.is_empty() || injector.delays_remaining() > 0 {
        assert!(
            Instant::now() < deadline,
            "fixture: A must hold its lease, its fan-out held back"
        );
        tokio::time::sleep(POLL_PAUSE).await;
    }
    assert_eq!(
        fenced_capacity(&client, &cache),
        reading,
        "fixture: mid-read, A holds its pair, P_f and its grant"
    );
    task.abort();
    let joined = task.await;
    assert!(
        joined.as_ref().is_err_and(|e| e.is_cancelled()),
        "fixture: A must have been cancelled mid-read; got {joined:?}"
    );

    assert_eq!(
        fenced_capacity(&client, &cache),
        NOTHING_HELD,
        "the cancelled cold start gave back its pair, P_f and grant"
    );
    wait_until_no_lease(&tv, a, "the cancelled cold start's lease").await;
    assert_fenced_cold_start_finishes(&tv, &cache, b, plain_items, "after A was cancelled").await;
}

/// T6(d) and T9's broken half — three broken attempts. Under each of the
/// fenced read's leases the header `seq` moves (a raw `HINCRBY` while the
/// fault seam holds the fan-out's first command back), so every release
/// answers `BROKEN`. After `FENCE_READ_ATTEMPTS` the read fails for this call
/// only: `Uncacheable { FencedReadFailed }` with the layer's size, one
/// `fenced_read_failed` tick and its DEBUG line (cause `broken`), no
/// cold-start tick, no negative entry, every lease released and the pair,
/// `P_f` and the grant given back. The next cold start loads.
///
/// Needs `test-fault-injection` (the read's fault seam).
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_cold_start_broken_three_times_fails_for_this_call_only_and_gives_everything_back() {
    use live_edit::storage::diff_read::{
        ObjectSnapshotReadFaultInjector, ObjectSnapshotReadFaultTarget, ObjectSnapshotReader,
    };
    use live_edit::storage::snapshot_fence::FENCE_READ_ATTEMPTS;
    use std::sync::atomic::{AtomicBool, Ordering};
    /// Long enough for the watcher, polling every [`POLL_PAUSE`], to move
    /// the header under the lease before the fan-out reads it.
    const FAN_OUT_DELAY: Duration = Duration::from_millis(300);
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-broken-thrice").await;
    let cards = probed_cardinalities(&tv, oid).await;
    let client = fenced_route_client(&tv, oid).await;
    let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
    for _ in 0..FENCE_READ_ATTEMPTS {
        injector.push_delay_at(
            ObjectSnapshotReadFaultTarget::nth_command(ObjectSnapshotReader::FencedFanout, 0),
            FAN_OUT_DELAY,
        );
    }
    client.install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));
    let metrics = hooked_metrics("cs-broken-thrice", &[&client]).await;
    let cache = StructuralCache::new(BUDGET_BYTES);
    // The watcher: one header move per lease it sees.
    let stop = Arc::new(AtomicBool::new(false));
    let moved = Arc::new(StdMutex::new(std::collections::BTreeSet::<String>::new()));
    let watcher = {
        let (data, fence, hdr) = (
            tv.client().clone(),
            fence_key(&tv, oid),
            tv.key_schema().obj_header(&oid.to_string()),
        );
        let (stop, moved) = (Arc::clone(&stop), Arc::clone(&moved));
        tokio::spawn(async move {
            while !stop.load(Ordering::SeqCst) {
                let fields: HashMap<String, String> =
                    data.pool.hgetall(&fence).await.expect("HGETALL fence");
                for lease in fields.into_keys().filter(|k| k.starts_with("r:")) {
                    if moved.lock().expect("moved").insert(lease) {
                        let _: i64 = data
                            .pool
                            .hincrby(&hdr, "seq", 1)
                            .await
                            .expect("HINCRBY seq");
                    }
                }
                tokio::time::sleep(POLL_PAUSE).await;
            }
        })
    };

    let (answer, captured) = capture_debug(tokio::time::timeout(
        WAIT_BUDGET,
        cold_start_probed(&client, tv.key_schema(), &cache, oid, CLIENT_ID, cards),
    ))
    .await;
    stop.store(true, Ordering::SeqCst);
    watcher.await.expect("the watcher must not panic");
    let answer = answer.expect("the cold start must finish within the budget");

    let (reason, above_ceiling) = expect_uncacheable_sized(answer, "broken three times");
    assert_eq!(reason, UncacheableReason::FencedReadFailed);
    let size = above_ceiling.expect("a layer known to be above the ceilings carries its size");
    assert!(
        size.items > plain_ceilings(&client).max_items,
        "the size is above the item ceiling: {size:?}"
    );
    assert_eq!(
        (
            moved.lock().expect("moved").len(),
            fenced_reads(&metrics, "fence_broken"),
            fenced_reads(&metrics, "ok")
        ),
        (FENCE_READ_ATTEMPTS as usize, FENCE_READ_ATTEMPTS as u64, 0),
        "fixture: the header moved under each of the attempts' leases, and each attempt broke \
         (leases moved under, fence_broken, ok)"
    );
    assert_eq!(
        uncacheable_verdicts(&metrics),
        BTreeMap::from([("fenced_read_failed".to_owned(), 1)]),
        "one fenced_read_failed verdict for the call"
    );
    assert_eq!(
        cold_starts_by_route(&metrics),
        (0, 0, 0),
        "a fenced read that failed for its call ticks no cold-start route"
    );
    let mut absent = WEIGHT_FIELDS.to_vec();
    absent.extend(LAYER_FIELDS);
    absent.extend(DEADLINE_FIELDS);
    absent.extend(CEILING_FIELDS);
    assert_one_verdict_line_at(
        &captured,
        "DEBUG",
        oid,
        "fenced_read_failed",
        &[
            ("cause", "broken".into()),
            ("attempts", FENCE_READ_ATTEMPTS.into()),
            ("size_items", size.items.into()),
            ("size_bytes", size.bytes.into()),
        ],
        &absent,
    );
    assert_eq!(fenced_capacity(&client, &cache), NOTHING_HELD);
    wait_until_no_lease(&tv, oid, "the broken attempts' leases").await;
    assert!(
        matches!(cache.lookup(oid), CacheLookup::Miss),
        "no negative entry: the verdict is this call's only"
    );
    let next = expect_loaded(
        cold_start_probed(
            &client,
            tv.key_schema(),
            &cache,
            oid,
            CLIENT_ID,
            probed_cardinalities(&tv, oid).await,
        )
        .await,
        "the next cold start",
    );
    assert_eq!(next.route, ColdStartRoute::Fenced);
}

/// T16 — the grant is the snapshot plus one window, reserved before the
/// acquire. The cold start's acquire reply is held at the proxy: across it
/// the cold start holds its pair, one `P_f` permit and exactly
/// `(SNAPSHOT_MEMORY_FACTOR + 1) × b` of `T` (`b` the probe's bound priced at
/// `K`; the clamp to `T` read from the client at run time). Released, it
/// loads and gives everything back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_cold_start_reserves_the_snapshot_plus_one_window_before_its_acquire() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-grant").await;
    warm_fence_scripts(&tv, &ls, "cs-grant").await;
    let cards = probed_cardinalities(&tv, oid).await;
    let gate = ReplyGate::new();
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        fence_acquire_rule(fence_key(&tv, oid), ProxyAction::HoldReplies(gate.clone())),
    )
    .await;
    let client =
        client_through_with_limits(&proxy, item_bound(&tv, oid).await - 1, NO_BYTE_LIMIT).await;
    let cache = StructuralCache::new(BUDGET_BYTES);
    let expected = FencedCapacity {
        in_flight: ONE_FENCED,
        p_f_held: 1,
        reserved_bytes: expected_grant_bytes(&client, probed_estimate(&cards).bytes),
        overcommit_bytes: 0,
    };
    assert!(
        expected.reserved_bytes > 0,
        "fixture: the expected grant is not empty"
    );

    let task = spawn_cold_start(&client, tv.key_schema(), &cache, oid, Some(cards));
    gate.reply_held().await;
    let across_acquire = fenced_capacity(&client, &cache);
    gate.open();
    let loaded = expect_loaded(join_cold_start(task, "the grant test").await, "grant");

    assert_eq!(
        across_acquire, expected,
        "across its acquire the cold start holds its pair, one P_f permit and (factor + 1) × b \
         of T"
    );
    assert_eq!(
        (loaded.route, proxy.hits()),
        (ColdStartRoute::Fenced, 1),
        "fixture: the held reply was the cold start's one acquire"
    );
    drop(loaded);
    assert_eq!(fenced_capacity(&client, &cache), NOTHING_HELD);
}

/// T16 — the memory grant counted for the read's reply lives only as long as
/// the read. The read decodes straight into the entry's compact state, which
/// the cold start moves out of the snapshot and which lives on as the entry;
/// the grant is given back once that state is normalized and its dangling set
/// built (Seam B, `structural_cache_cold_start`'s module docs). So at the
/// cache's before-admit test hook — after the decode and before `admit` (and
/// so before the token `HGET`) — the cold start holds nothing: no grant, no
/// top-up, no `P_f`, no pair. Fires exactly once, for the object.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_cold_start_holds_no_memory_p_f_or_permit_at_the_before_admit_hook() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-before-admit").await;
    let cards = probed_cardinalities(&tv, oid).await;
    let client = fenced_route_client(&tv, oid).await;
    let cache = StructuralCache::new(BUDGET_BYTES);
    let seen: Arc<StdMutex<Vec<(u64, FencedCapacity)>>> = Arc::default();
    cache.set_before_admit_hook_for_test(Some({
        let (client, cache, seen) = (client.clone(), cache.clone(), Arc::clone(&seen));
        Arc::new(move |object_id| {
            let held = fenced_capacity(&client, &cache);
            seen.lock().expect("seen").push((object_id, held));
        })
    }));

    let loaded = expect_loaded(
        cold_start_probed(&client, tv.key_schema(), &cache, oid, CLIENT_ID, cards).await,
        "fenced",
    );
    // The hook holds a clone of the cache: remove it, so the two drop.
    cache.set_before_admit_hook_for_test(None);

    assert_eq!(loaded.route, ColdStartRoute::Fenced);
    assert_eq!(
        *seen.lock().expect("seen"),
        vec![(oid, NOTHING_HELD)],
        "before admit, the snapshot's grant, P_f and the pair are all given back"
    );
}

/// T17 — a deadline that leaves no time. Through the read side's entry point
/// with the calling RPC's deadline left equal to `hold(L) + S(L)` — the time
/// this read and one WATCH attempt need — the waiting cap is zero, and the
/// fenced read refuses at its first attempt before any permit is taken or
/// any acquire sent. The call answers `Uncacheable { FencedReadFailed }` with
/// the probe's size: one `fenced_read_failed` tick and its DEBUG line (cause
/// `cap`), no cold-start tick, no negative entry. The next call, with no
/// deadline, caches.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_read_with_no_deadline_left_past_the_read_and_a_watch_attempt_fails_before_any_acquire(
) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-cap-zero").await;
    let cards = probed_cardinalities(&tv, oid).await;
    let client = fenced_route_client(&tv, oid).await;
    let size = probed_estimate(&cards);
    let remaining = read_and_watch_reserve(&client, size);
    let coefficients = coefficients(&client);
    assert_eq!(
        wait_cap(&coefficients, Some(remaining), size),
        Duration::ZERO,
        "fixture: the deadline left is exactly hold + S, so the cap is zero"
    );
    assert!(
        wait_cap(
            &coefficients,
            Some(remaining + Duration::from_secs(1)),
            size
        ) > Duration::ZERO,
        "fixture: a second more would leave a cap"
    );
    let metrics = hooked_metrics("cs-cap-zero", &[&client]).await;
    let cache = StructuralCache::new(BUDGET_BYTES);
    let request = StructuralReadRequest::new(oid, READ_CLIENT_ID, READ_TOKEN)
        .with_remaining_deadline(Arc::new(move || Some(remaining)));

    let (answer, captured) = capture_debug(read_for_structural_edit(
        &client,
        tv.key_schema(),
        &cache,
        &request,
    ))
    .await;

    assert!(
        matches!(
            &answer,
            Ok(StructuralRead::Uncacheable {
                reason: UncacheableReason::FencedReadFailed,
                above_ceiling: Some(s),
            }) if *s == size
        ),
        "no time for the read: FencedReadFailed, with the probe's size; got {answer:?}"
    );
    assert_eq!(
        (
            fence_acquires(&metrics),
            all_structural_reads(&metrics),
            fenced_permit_waits(&metrics),
            budget_reservations(&metrics)
        ),
        (0, 0, 0, 0),
        "refused before any acquire, read, P_f wait or memory reservation (acquires, reads, P_f \
         waits, reservations)"
    );
    assert_fenced_read_failed_for_this_call_only(&tv, &client, &cache, &metrics, oid).await;
    let mut absent = WEIGHT_FIELDS.to_vec();
    absent.extend(LAYER_FIELDS);
    absent.extend(DEADLINE_FIELDS);
    absent.extend(CEILING_FIELDS);
    assert_one_verdict_line_at(
        &captured,
        "DEBUG",
        oid,
        "fenced_read_failed",
        &[
            ("cause", "cap".into()),
            ("attempts", 1.into()),
            ("size_items", size.items.into()),
            ("size_bytes", size.bytes.into()),
            ("cap_ms", 0.into()),
        ],
        &absent,
    );
}

/// T17 — a `RETRY_AFTER` past the cap. A forged live lease makes the
/// acquire answer `RETRY_AFTER` with a hint of one `poll` (about a second),
/// and the deadline left caps the wait at a fraction of that: the hook
/// refuses at once instead of sleeping, after exactly one acquire, giving
/// its capacity back. `Uncacheable { FencedReadFailed }` for this call only;
/// once the lease is gone the next call caches.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_read_whose_retry_after_would_sleep_past_the_cap_fails_the_fenced_read_at_once()
{
    /// The waiting cap the deadline leaves: far under one `poll`.
    const CAP: Duration = Duration::from_millis(300);
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-cap-retry").await;
    let cards = probed_cardinalities(&tv, oid).await;
    let client = fenced_route_client(&tv, oid).await;
    let size = probed_estimate(&cards);
    let remaining = read_and_watch_reserve(&client, size) + CAP;
    let coefficients = coefficients(&client);
    assert_eq!(
        wait_cap(&coefficients, Some(remaining), size),
        CAP,
        "fixture: the deadline left caps the wait at CAP"
    );
    assert!(
        ms_to_duration(coefficients.poll_ms()) > CAP,
        "fixture: the RETRY_AFTER hint, one poll, is past the cap"
    );
    forge_live_lease(&tv, oid).await;
    let metrics = hooked_metrics("cs-cap-retry", &[&client]).await;
    let cache = StructuralCache::new(BUDGET_BYTES);
    let request = StructuralReadRequest::new(oid, READ_CLIENT_ID, READ_TOKEN)
        .with_remaining_deadline(Arc::new(move || Some(remaining)));

    let (answer, captured) = capture_debug(read_for_structural_edit(
        &client,
        tv.key_schema(),
        &cache,
        &request,
    ))
    .await;

    assert!(
        matches!(
            &answer,
            Ok(StructuralRead::Uncacheable {
                reason: UncacheableReason::FencedReadFailed,
                above_ceiling: Some(s),
            }) if *s == size
        ),
        "a RETRY_AFTER past the cap: FencedReadFailed, with the layer's size; got {answer:?}"
    );
    assert_eq!(
        (
            fence_acquires_of(&metrics, "retry_after_epoch"),
            fence_acquires(&metrics),
            all_structural_reads(&metrics)
        ),
        (1, 1, 0),
        "exactly one acquire, answered RETRY_AFTER, and no read: the hook refused rather than \
         sleep (retry_after_epoch, every acquire, reads)"
    );
    let mut absent = WEIGHT_FIELDS.to_vec();
    absent.extend(LAYER_FIELDS);
    absent.extend(DEADLINE_FIELDS);
    absent.extend(CEILING_FIELDS);
    assert_one_verdict_line_at(
        &captured,
        "DEBUG",
        oid,
        "fenced_read_failed",
        &[
            ("cause", "cap".into()),
            ("attempts", 1.into()),
            ("cap_ms", (CAP.as_millis() as u64).into()),
        ],
        &absent,
    );
    remove_forged_lease(&tv, oid).await;
    assert_fenced_read_failed_for_this_call_only(&tv, &client, &cache, &metrics, oid).await;
}

/// The per-call verdict's aftermath (T9's shape, for T17's failures): one
/// `fenced_read_failed` tick, no cold-start tick, the capacity given back, no
/// negative entry — and the next call, with no deadline, cold-starts the
/// object on the fenced route and plans on it.
async fn assert_fenced_read_failed_for_this_call_only(
    tv: &TestValkey,
    client: &ValkeyClient,
    cache: &StructuralCache,
    metrics: &Metrics,
    oid: u64,
) {
    assert_eq!(
        uncacheable_verdicts(metrics),
        BTreeMap::from([("fenced_read_failed".to_owned(), 1)]),
        "one fenced_read_failed verdict for the call"
    );
    assert_eq!(
        cold_starts_by_route(metrics),
        (0, 0, 0),
        "a fenced read that failed for its call ticks no cold-start route"
    );
    assert_eq!(
        fenced_capacity(client, cache),
        NOTHING_HELD,
        "the failed read gave its capacity back"
    );
    assert!(
        matches!(cache.lookup(oid), CacheLookup::Miss),
        "no negative entry: the verdict is this call's only"
    );
    let next = tokio::time::timeout(
        WAIT_BUDGET,
        read_for_structural_edit(
            client,
            tv.key_schema(),
            cache,
            &StructuralReadRequest::new(oid, READ_CLIENT_ID, READ_TOKEN),
        ),
    )
    .await
    .expect("the next call must finish within the budget");
    assert!(
        matches!(next, Ok(StructuralRead::Ready(_))),
        "the next call, with no deadline, caches the object; got {next:?}"
    );
    assert_eq!(
        cold_starts_by_route(metrics),
        (0, 1, 0),
        "the next call's cold start took the fenced route"
    );
}

/// T17 — with no deadline in scope the cap is the cost model's
/// `reader_wait_max(L)` alone: exactly `CappedFenceWait`'s cap, on the same
/// client's coefficients, at every size and at two item ceilings (the
/// ceiling feeds `min_gap` and `Q`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_cold_start_cap_with_no_deadline_is_capped_fence_waits_cap() {
    for max_items in [SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT, 64] {
        let client = client_with_limits(max_items, NO_BYTE_LIMIT).await;
        let capped = CappedFenceWait::new(&client, 1, SlimSnapshotReadConsumer::StructuralCache);
        for size in [
            LayerSize { items: 0, bytes: 0 },
            LayerSize {
                items: 100,
                bytes: 100 * K,
            },
            LayerSize {
                items: 40_000,
                bytes: 800_000,
            },
            LayerSize {
                items: 2_000_000,
                bytes: 40_000_000,
            },
        ] {
            assert_eq!(
                wait_cap(&coefficients(&client), None, size),
                capped.cap(size),
                "item ceiling {max_items}, size {size:?}: with no deadline the cap is \
                 reader_wait_max(L), CappedFenceWait's"
            );
        }
    }
}

/// M9 — the deadline check on a plain read's `TOO_LARGE` answer. A predictor
/// miss (the byte-dense fixture: predicted to fit, over the plain byte
/// ceiling) reads plain first, and the answer's size
/// (`LayerSize::from_too_large`, measured here by the same slim read on a
/// second client) is judged against the configured RPC deadline the request
/// carries (`with_rpc_deadline`): exactly the fenced read's reserve at that
/// size, so its waiting cap would be zero with the whole deadline left. The
/// same deadline at the probe's own estimate would leave a cap, so it is the
/// answer's size that is judged. The read side answers
/// `Uncacheable { OverDeadline }` carrying that size after one plain read
/// (the counted miss) and no fenced read, acquire, `P_f` wait or memory
/// reservation; the verdict is recorded on the `uncacheable` route, ticked
/// and logged once at INFO with the size and the deadline; and the next
/// call is routed by the negative entry with no lookup miss, read or tick.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_read_predictor_miss_over_the_configured_deadline_records_over_deadline_with_no_fenced_read(
) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let (oid, plain) = dense_object(&tv, &ls, "cs-over-deadline-miss").await;
    let cards = probed_cardinalities(&tv, oid).await;
    let measure = client_with_limits(plain.max_items, plain.max_bytes).await;
    let size = match read_object_snapshot_slim_with_ceilings(
        &measure,
        tv.key_schema(),
        &oid.to_string(),
        SlimSnapshotReadConsumer::StructuralCache,
        plain,
    )
    .await
    .expect("fixture: the measuring read must answer")
    {
        SnapshotReadOf::TooLarge(too_large) => LayerSize::from_too_large(&too_large, None),
        SnapshotReadOf::Snapshot(_) => {
            panic!("fixture: the dense object must be over the plain byte ceiling")
        }
    };
    let client = client_with_limits(plain.max_items, plain.max_bytes).await;
    let coefficients = coefficients(&client);
    let rpc_deadline = wait_reserve(&coefficients, size);
    assert!(
        fenced_read_over_deadline(&coefficients, rpc_deadline, size)
            && !fenced_read_over_deadline(&coefficients, rpc_deadline, probed_estimate(&cards)),
        "fixture: {rpc_deadline:?} fits a fenced read at the probe's estimate {:?} but not at \
         the answer's size {size:?}",
        probed_estimate(&cards)
    );
    let metrics = hooked_metrics("cs-over-deadline-miss", &[&client]).await;
    let cache = StructuralCache::new(BUDGET_BYTES);
    let request =
        StructuralReadRequest::new(oid, READ_CLIENT_ID, READ_TOKEN).with_rpc_deadline(rpc_deadline);

    let (answer, captured) = capture(read_for_structural_edit(
        &client,
        tv.key_schema(),
        &cache,
        &request,
    ))
    .await;

    assert!(
        matches!(
            &answer,
            Ok(StructuralRead::Uncacheable {
                reason: UncacheableReason::OverDeadline,
                above_ceiling: Some(s),
            }) if *s == size
        ),
        "over the configured deadline: OverDeadline, with the answer's size; got {answer:?}"
    );
    assert_eq!(
        (
            too_large_and_ok_reads(&metrics),
            all_structural_reads(&metrics)
        ),
        ((1, 0), 1),
        "one plain read, refused TOO_LARGE, and nothing else ((atomic too_large, atomic ok), \
         every read)"
    );
    assert_eq!(
        (
            fence_acquires(&metrics),
            fenced_permit_waits(&metrics),
            budget_reservations(&metrics)
        ),
        (0, 0, 0),
        "no fenced read began: no acquire, P_f wait or memory reservation"
    );
    assert_eq!(fenced_capacity(&client, &cache), NOTHING_HELD);
    assert_eq!(
        uncacheable_verdicts(&metrics),
        BTreeMap::from([("over_deadline".to_owned(), 1)])
    );
    assert_eq!(
        cold_starts_by_route(&metrics),
        (0, 0, 1),
        "a recorded verdict ends the cold start on the uncacheable route"
    );
    assert!(
        matches!(
            cache.lookup(oid),
            CacheLookup::Uncacheable {
                reason: UncacheableReason::OverDeadline
            }
        ),
        "the verdict is recorded as a negative entry"
    );
    let mut absent = WEIGHT_FIELDS.to_vec();
    absent.extend(LAYER_FIELDS);
    absent.extend(CEILING_FIELDS);
    absent.extend(["cause", "attempts", "waited_ms", "cap_ms"]);
    assert_one_verdict_line(
        &captured,
        oid,
        "over_deadline",
        &[
            ("size_items", size.items.into()),
            ("size_bytes", size.bytes.into()),
            ("rpc_deadline_ms", (rpc_deadline.as_millis() as u64).into()),
            // The deadline is the reserve at the judged size.
            ("reserve_ms", (rpc_deadline.as_millis() as u64).into()),
        ],
        &absent,
    );
    assert_eq!(
        lookups(&metrics, "cold"),
        1,
        "fixture: the first call's miss"
    );

    let next = read_for_structural_edit(&client, tv.key_schema(), &cache, &request).await;

    assert!(
        matches!(
            next,
            Ok(StructuralRead::Uncacheable {
                reason: UncacheableReason::OverDeadline,
                above_ceiling: None,
            })
        ),
        "the next call is routed by the negative entry, which holds no size; got {next:?}"
    );
    assert_eq!(
        (
            lookups(&metrics, "cold"),
            all_structural_reads(&metrics),
            cold_starts_by_route(&metrics)
        ),
        (1, 1, (0, 0, 1)),
        "the next call missed nothing, read nothing and cold-started nothing (cold lookups, \
         reads, routes)"
    );
    assert_eq!(
        uncacheable_verdicts(&metrics),
        BTreeMap::from([("over_deadline".to_owned(), 1)]),
        "one tick per verdict, not per call"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. No fan-out outside the fence, and the plain ceilings only (MONITOR; T10)
// ═════════════════════════════════════════════════════════════════════════════

/// Every command a client (not a script) issued against the object's keys
/// while a cold start ran on `client` — probed with `probed`, or unprobed —
/// from the store's own command feed.
async fn object_commands_during_cold_start(
    tv: &TestValkey,
    client: &ValkeyClient,
    oid: u64,
    probed: Option<SnapshotCardinalities>,
) -> (Result<ColdStart, ServiceError>, Vec<MonitoredCommand>) {
    let tag = format!("{}{{{oid}}}:obj:", tv.prefix());
    let cache = StructuralCache::new(BUDGET_BYTES);
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let answer = match probed {
        Some(cards) => {
            cold_start_probed(client, tv.key_schema(), &cache, oid, CLIENT_ID, cards).await
        }
        None => cold_start(client, tv.key_schema(), &cache, oid, CLIENT_ID).await,
    };
    let fence = monitor_fence(tv.prefix());
    let _: i64 = tv.client().pool.exists(&fence).await.unwrap();
    let commands = tap
        .drain_until(&fence)
        .await
        .into_iter()
        .filter(|c| !c.issued_by_script() && c.args.iter().any(|a| a.contains(&tag)))
        .collect();
    (answer, commands)
}

/// Whether `command` is an `EVALSHA` of the script `sha`.
fn is_evalsha_of(command: &MonitoredCommand, sha: &str) -> bool {
    command.name() == "EVALSHA" && command.args.get(1).map(String::as_str) == Some(sha)
}

/// The structure of one cold start's commands against its object: the
/// snapshot script's `EVALSHA`s (each passing exactly `plain`'s ceilings, T10),
/// one granted fence acquire (after any the writers' gap refused) and one
/// release with every other command — the fenced fan-out — strictly between
/// them, and the token `HGET` last. `metrics` is the registry of this cold
/// start alone: its `live_edit_snapshot_fence_acquire_total{result}` ticks
/// must account for every acquire seen — one grant, and each earlier acquire
/// a `RETRY_AFTER` refusal.
/// Returns `(snapshot reads, fan-out commands under the lease)`.
fn assert_cold_start_commands(
    route: &str,
    commands: &[MonitoredCommand],
    plain: SnapshotCeilings,
    wtok: &str,
    metrics: &Metrics,
) -> (usize, usize) {
    let (snapshot, acquire, release) = (snapshot_sha(), acquire_sha(), release_sha());
    let snapshot_reads: Vec<&MonitoredCommand> = commands
        .iter()
        .filter(|c| is_evalsha_of(c, &snapshot))
        .collect();
    for read in &snapshot_reads {
        assert_eq!(
            (
                read.args.get(7).map(String::as_str),
                read.args.get(8).map(String::as_str)
            ),
            (
                Some(plain.max_items.to_string().as_str()),
                Some(plain.max_bytes.to_string().as_str())
            ),
            "{route}: every structural_cache snapshot read passes the client's plain ceilings \
             (ARGV[4], ARGV[5]) — no raised read; read: {read:?}"
        );
    }
    let position = |sha: &str| -> Vec<usize> {
        commands
            .iter()
            .enumerate()
            .filter(|(_, c)| is_evalsha_of(c, sha))
            .map(|(i, _)| i)
            .collect()
    };
    let (acquires, releases) = (position(&acquire), position(&release));
    let token_reads: Vec<usize> = commands
        .iter()
        .enumerate()
        .filter(|(_, c)| c.name() == "HGET" && c.args.get(1).map(String::as_str) == Some(wtok))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        token_reads,
        vec![commands.len() - 1],
        "{route}: one token HGET, the last command against the object; commands: {commands:?}"
    );
    assert_eq!(
        commands[commands.len() - 1].args.get(2).map(String::as_str),
        Some(CLIENT_ID),
        "{route}: the HGET reads the call's commit-token record"
    );
    let fan_out: Vec<usize> = commands
        .iter()
        .enumerate()
        .filter(|(i, c)| c.name() != "EVALSHA" && !token_reads.contains(i))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        fence_acquires(metrics),
        acquires.len() as u64,
        "{route}: one fence_acquire_total tick per acquire seen; commands: {commands:?}"
    );
    match (acquires.as_slice(), releases.as_slice()) {
        ([], []) => assert!(
            fan_out.is_empty(),
            "{route}: with no fence, no fan-out command may run; commands: {commands:?}"
        ),
        // An acquire inside the writers' gap after the previous case's
        // release on the same object answers `RETRY_AFTER` and grants
        // nothing, so the read acquires again: any acquires before the last
        // are refused ones, with no fan-out after them. The acquire ticks
        // check that tolerance: exactly one grant, and every other acquire a
        // gap or epoch refusal (never an intent one: no writer runs here).
        ([first, .., granted], [released]) | ([first @ granted], [released]) => {
            assert_eq!(
                fence_acquires_of(metrics, "retry_after_gap")
                    + fence_acquires_of(metrics, "retry_after_epoch"),
                acquires.len() as u64 - 1,
                "{route}: every acquire before the granted one was a gap or epoch refusal; \
                 commands: {commands:?}"
            );
            assert_eq!(
                fence_acquires_of(metrics, "started_epoch")
                    + fence_acquires_of(metrics, "joined_epoch"),
                1,
                "{route}: exactly one acquire was granted; commands: {commands:?}"
            );
            assert!(
                fan_out.iter().all(|i| granted < i && i < released),
                "{route}: every fan-out command runs between the granted fence acquire and its \
                 release — under the lease; commands: {commands:?}"
            );
            assert!(
                snapshot_reads
                    .iter()
                    .all(|r| commands.iter().position(|c| std::ptr::eq(c, *r)) < Some(*first)),
                "{route}: a plain read comes before the fenced read, never after"
            );
        }
        other => panic!(
            "{route}: one granted acquire and one release (no fenced read broke here); got \
             {other:?}; commands: {commands:?}"
        ),
    }
    (snapshot_reads.len(), fan_out.len())
}

/// On every route the cold start's reads are the snapshot script — at the
/// client's plain ceilings, never raised — and, above the ceilings, the
/// fenced read's fan-out strictly inside its lease, never the unfenced
/// fan-out; plus the one token `HGET` after a load. The routes: the atomic
/// read (one `EVALSHA`); the unprobed fenced route (the plain `TOO_LARGE`,
/// then the fenced read); the probed fenced route (no plain read); and a
/// predictor miss (the plain read refusing mid-read on bytes, then fenced).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_reads_only_the_plain_snapshot_script_and_the_fenced_fan_out_under_its_lease() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-monitor").await;
    let (dense, dense_plain) = dense_object(&tv, &ls, "cs-monitor-dense").await;
    warm_snapshot_script(&tv, oid).await;
    warm_fence_scripts(&tv, &ls, "cs-monitor").await;
    let cards = probed_cardinalities(&tv, oid).await;
    let dense_cards = probed_cardinalities(&tv, dense).await;
    let atomic_client = default_ceiling_client().await;
    let fenced_client = fenced_route_client(&tv, oid).await;
    let dense_client = client_with_limits(dense_plain.max_items, dense_plain.max_bytes).await;

    for (route, client, object, probed, expected_route, expected_snapshot_reads) in [
        (
            "atomic",
            &atomic_client,
            oid,
            None,
            ColdStartRoute::Atomic,
            1,
        ),
        (
            "fenced, unprobed",
            &fenced_client,
            oid,
            None,
            ColdStartRoute::Fenced,
            1,
        ),
        (
            "fenced, probed",
            &fenced_client,
            oid,
            Some(cards),
            ColdStartRoute::Fenced,
            0,
        ),
        (
            "fenced after a predictor miss",
            &dense_client,
            dense,
            Some(dense_cards),
            ColdStartRoute::Fenced,
            1,
        ),
    ] {
        let wtok = tv.key_schema().obj_write_token(&object.to_string());
        // A registry of this case's own, so its acquire ticks are its alone.
        let metrics = hooked_metrics("cs-monitor", &[client]).await;
        let (answer, commands) =
            object_commands_during_cold_start(&tv, client, object, probed).await;
        let loaded = expect_loaded(answer, route);
        assert_eq!(loaded.route, expected_route, "{route}");
        drop(loaded);

        let (snapshot_reads, fan_out) =
            assert_cold_start_commands(route, &commands, plain_ceilings(client), &wtok, &metrics);

        assert_eq!(
            snapshot_reads, expected_snapshot_reads,
            "{route}: the snapshot script's reads; commands: {commands:?}"
        );
        assert_eq!(
            fan_out > 0,
            expected_route == ColdStartRoute::Fenced,
            "{route}: the fenced route reads through its fan-out, the atomic route never does; \
             commands: {commands:?}"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. The script's `return_dangling` (ARGV[12]) and `slim` (ARGV[13]) flags
// ═════════════════════════════════════════════════════════════════════════════

/// The script's arguments, as `read_object_snapshot` builds them, plus the
/// optional flag.
fn script_argv(keys: &KeySchema, oid: u64, max_items: u64, flag: Option<&str>) -> Vec<String> {
    script_argv_with(keys, oid, max_items, flag.as_slice())
}

/// The script's arguments, as `read_object_snapshot` builds them, plus
/// `flags` as `ARGV[12]` onwards (`return_dangling`, then `slim`).
fn script_argv_with(keys: &KeySchema, oid: u64, max_items: u64, flags: &[&str]) -> Vec<String> {
    let mut argv = vec![
        keys.prefix().to_owned(),
        oid.to_string(),
        "read".to_owned(),
        max_items.to_string(),
        NO_BYTE_LIMIT.to_string(),
        SNAPSHOT_SECTION_HASH_WIDTH.to_string(),
        SNAPSHOT_TOMBSTONE_HASH_WIDTH.to_string(),
        SNAPSHOT_FIELD_HASH_WIDTH.to_string(),
        SNAPSHOT_TIME_SAMPLE_HASH_WIDTH.to_string(),
        HEADER_FIELD_CHS_COMPLETE.to_owned(),
        CHS_COMPLETE_MARKED.to_owned(),
    ];
    argv.extend(flags.iter().map(|flag| (*flag).to_owned()));
    argv
}

/// `EVAL` the script against `oid`'s header.
async fn eval_snapshot_script(
    tv: &TestValkey,
    oid: u64,
    max_items: u64,
    flag: Option<&str>,
) -> Result<Value, fred::error::Error> {
    tv.client()
        .pool
        .eval(
            SNAPSHOT_SCRIPT,
            vec![tv.key_schema().obj_header(&oid.to_string())],
            script_argv(tv.key_schema(), oid, max_items, flag),
        )
        .await
}

/// The reply's elements, which must be an array.
async fn reply_parts(tv: &TestValkey, oid: u64, max_items: u64, flag: Option<&str>) -> Vec<Value> {
    match eval_snapshot_script(tv, oid, max_items, flag).await {
        Ok(Value::Array(parts)) => parts,
        other => panic!("the script must reply with an array (flag {flag:?}); got {other:?}"),
    }
}

/// An array reply element's members, as strings.
fn member_strings(value: &Value) -> Vec<String> {
    match value {
        Value::Array(members) => members
            .iter()
            .map(|m| m.as_string().expect("a dangling member is a string"))
            .collect(),
        other => panic!("the dangling element must be an array; got {other:?}"),
    }
}

/// With the flag set, an `OK` reply gains exactly one trailing element — the
/// dangling ids — and its first nine are the unflagged reply's: on a marked
/// object with a dangling list, `{"77"}`; on a marked object without one, and
/// on an unmarked object (whose `obj:chs` may be partial), `{}`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn object_snapshot_read_return_dangling_appends_exactly_the_dangling_ids_to_ok() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv, "cs-flag").await;
    let marked = create(&ls, &session_id, "cs-flag-marked", &every_shape_state()).await;
    let mut clean_state = every_shape_state();
    clean_state.children.remove(&DANGLING_PARENT);
    clean_state
        .sections
        .retain(|_, s| s.parent_id != DANGLING_PARENT);
    let clean = create(&ls, &session_id, "cs-flag-clean", &clean_state).await;
    let unmarked = create(&ls, &session_id, "cs-flag-unmarked", &every_shape_state()).await;
    unmark(&tv, unmarked).await;
    assert!(
        indexed_parents(&tv, marked)
            .await
            .contains(&DANGLING_PARENT)
            && !indexed_parents(&tv, clean).await.contains(&DANGLING_PARENT),
        "fixture: only the marked object's index names the dangling parent"
    );

    for (what, oid, expected) in [
        (
            "marked, with a dangling list",
            marked,
            vec![DANGLING_PARENT.to_string()],
        ),
        ("marked, without one", clean, vec![]),
        ("unmarked, with a dangling list", unmarked, vec![]),
    ] {
        let flagged =
            reply_parts(&tv, oid, SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT, Some("1")).await;
        let unflagged = reply_parts(&tv, oid, SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT, None).await;

        assert_eq!(
            (flagged.len(), flagged[1].as_string().as_deref()),
            (10, Some("OK")),
            "{what}: a flagged OK reply has ten elements"
        );
        assert_eq!(
            &flagged[..9],
            &unflagged[..],
            "{what}: the flag only appends; the nine elements before it are the unflagged reply"
        );
        assert_eq!(
            member_strings(&flagged[9]),
            expected,
            "{what}: the trailing element is exactly the dangling ids"
        );
    }
}

/// Unset — absent, or `'0'` — the reply is the nine-element reply every other
/// consumer decodes; the flag never touches a `TOO_LARGE` reply; any other
/// value is an argument error, never a silent unset.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn object_snapshot_read_return_dangling_unset_changes_nothing_and_rejects_other_values() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv, "cs-flag-unset").await;
    let oid = create(&ls, &session_id, "cs-flag-unset", &every_shape_state()).await;

    let absent = reply_parts(&tv, oid, SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT, None).await;
    let zero = reply_parts(&tv, oid, SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT, Some("0")).await;
    let too_large = reply_parts(&tv, oid, 1, Some("1")).await;
    let bad =
        eval_snapshot_script(&tv, oid, SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT, Some("2")).await;

    assert_eq!(
        absent.len(),
        9,
        "eleven arguments: the original nine-element OK reply"
    );
    assert_eq!(zero, absent, "'0' is the flag unset: the same reply");
    assert_eq!(
        (too_large.len(), too_large[1].as_string().as_deref()),
        (6, Some("TOO_LARGE")),
        "the flag adds its element to OK only"
    );
    let err = bad.expect_err("a flag other than 0 or 1 must be refused");
    assert!(
        err.details().contains("SNAPSHOT_READ_ARGS"),
        "the refusal is the script's argument error; got {err:?}"
    );
}

// ── The script's `slim` flag (ARGV[13], b.mr6) ──────────────────────────────

/// `EVAL` the script against `oid`'s header with `flags` from `ARGV[12]` on.
async fn eval_snapshot_script_with(
    tv: &TestValkey,
    oid: u64,
    max_items: u64,
    flags: &[&str],
) -> Result<Value, fred::error::Error> {
    tv.client()
        .pool
        .eval(
            SNAPSHOT_SCRIPT,
            vec![tv.key_schema().obj_header(&oid.to_string())],
            script_argv_with(tv.key_schema(), oid, max_items, flags),
        )
        .await
}

/// The reply's elements with `flags`, which must be an array.
async fn reply_parts_with(tv: &TestValkey, oid: u64, max_items: u64, flags: &[&str]) -> Vec<Value> {
    match eval_snapshot_script_with(tv, oid, max_items, flags).await {
        Ok(Value::Array(parts)) => parts,
        other => panic!("the script must reply with an array (flags {flags:?}); got {other:?}"),
    }
}

/// The `OK` reply's elements: `hdr, secs, dels, flds, tss, children,
/// tallies` after `fmt, 'OK'`.
const FLDS: usize = 5;
const TSS: usize = 6;
const TALLIES: usize = 8;

/// A slim value-row element of an `OK` reply: member → `(shape, hash)`.
fn slim_rows(value: &Value) -> BTreeMap<String, (String, Option<String>)> {
    let Value::Array(rows) = value else {
        panic!("a value-row element must be an array; got {value:?}");
    };
    rows.iter()
        .map(|row| {
            let Value::Array(parts) = row else {
                panic!("a slim row must be an array; got {row:?}");
            };
            let strings: Vec<String> = parts
                .iter()
                .map(|p| p.as_string().expect("a slim row's parts are strings"))
                .collect();
            match strings.as_slice() {
                [member, shape] => (member.clone(), (shape.clone(), None)),
                [member, shape, hash] => (member.clone(), (shape.clone(), Some(hash.clone()))),
                other => panic!("a slim row is {{member, shape[, hash]}}; got {other:?}"),
            }
        })
        .collect()
}

/// Unset — absent, or `'0'`, with `return_dangling` set or not — the reply is
/// byte for byte the reply without the argument; set, it never touches a
/// `TOO_LARGE` reply; any other value is an argument error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn object_snapshot_read_slim_unset_changes_nothing_and_rejects_other_values() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv, "cs-slim-unset").await;
    let marked = create(
        &ls,
        &session_id,
        "cs-slim-unset-marked",
        &every_shape_state(),
    )
    .await;
    let unmarked = create(
        &ls,
        &session_id,
        "cs-slim-unset-unmarked",
        &every_shape_state(),
    )
    .await;
    unmark(&tv, unmarked).await;
    const MAX: u64 = SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT;

    for (what, oid) in [("marked", marked), ("unmarked", unmarked)] {
        for return_dangling in ["0", "1"] {
            let without = reply_parts_with(&tv, oid, MAX, &[return_dangling]).await;
            let unset = reply_parts_with(&tv, oid, MAX, &[return_dangling, "0"]).await;
            assert_eq!(
                unset, without,
                "{what}, return_dangling {return_dangling}: `slim` '0' is the reply without it"
            );
        }
        assert_eq!(
            reply_parts_with(&tv, oid, MAX, &["0", "0"]).await,
            reply_parts_with(&tv, oid, MAX, &[]).await,
            "{what}: both flags unset is the eleven-argument reply"
        );
        let too_large_set = reply_parts_with(&tv, oid, 1, &["0", "1"]).await;
        let too_large_unset = reply_parts_with(&tv, oid, 1, &[]).await;
        assert_eq!(
            (too_large_set.len(), too_large_set[1].as_string().as_deref()),
            (6, Some("TOO_LARGE")),
            "{what}: fixture: over a one-item ceiling"
        );
        assert_eq!(
            too_large_set, too_large_unset,
            "{what}: the flag changes no TOO_LARGE reply"
        );
    }
    let err = eval_snapshot_script_with(&tv, marked, MAX, &["0", "2"])
        .await
        .expect_err("a slim flag other than 0 or 1 must be refused");
    assert!(
        err.details().contains("SNAPSHOT_READ_ARGS") && err.details().contains("slim"),
        "the refusal is the script's argument error naming the flag; got {err:?}"
    );
}

/// Set, the flag changes only the per-row elements of `flds` and `tss`: each
/// row is its shape — `i` inline, `x` with the raw `hash` column — and every
/// other element (`hdr`, sections, tombstones, children, the `(items, bytes,
/// entries)` tallies, and with `return_dangling` the dangling ids) is the
/// unset read's on the same object.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn object_snapshot_read_slim_set_sends_each_rows_shape_and_the_unset_reads_tallies() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv, "cs-slim-set").await;
    let oid = create(&ls, &session_id, "cs-slim-set", &every_shape_state()).await;
    const MAX: u64 = SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT;
    let hex = |tag: u8| -> String { format!("{tag:02x}").repeat(32) };

    for return_dangling in ["0", "1"] {
        let set = reply_parts_with(&tv, oid, MAX, &[return_dangling, "1"]).await;
        let unset = reply_parts_with(&tv, oid, MAX, &[return_dangling]).await;
        assert_eq!(
            (set.len(), set[1].as_string().as_deref()),
            (unset.len(), Some("OK")),
            "return_dangling {return_dangling}: one OK reply shape"
        );
        for (i, (set_part, unset_part)) in set.iter().zip(&unset).enumerate() {
            if i != FLDS && i != TSS {
                assert_eq!(
                    set_part, unset_part,
                    "return_dangling {return_dangling}: element {i} is the unset read's"
                );
            }
        }
        assert_eq!(
            set[TALLIES], unset[TALLIES],
            "return_dangling {return_dangling}: the tallies are a full read's"
        );
        assert_eq!(
            slim_rows(&set[FLDS]),
            BTreeMap::from([
                (
                    KeySchema::field_member(&WORLD.to_string(), "color"),
                    ("i".to_owned(), None)
                ),
                (
                    KeySchema::field_member(&CUBE.to_string(), "mesh"),
                    ("x".to_owned(), Some(hex(0xAB)))
                ),
            ]),
            "return_dangling {return_dangling}: the fields' shapes"
        );
        assert_eq!(
            slim_rows(&set[TSS]),
            BTreeMap::from([
                (
                    KeySchema::time_sample_member(&WORLD.to_string(), 0.5),
                    ("i".to_owned(), None)
                ),
                (
                    KeySchema::time_sample_member(&CUBE.to_string(), 1.5),
                    ("x".to_owned(), Some(hex(0xCD)))
                ),
            ]),
            "return_dangling {return_dangling}: the time samples' shapes"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 6. Retirement during the read
// ═════════════════════════════════════════════════════════════════════════════

/// The cold start's read runs and its reply is held; `DeleteLayer` then
/// retires the object (its `SemaphoreMapGuard` invalidates it in the store's
/// cache); the reply is released. The read's cut predates the retirement, so
/// `admit` refuses it: `Invalidated`, nothing stored or recorded, no token
/// read (the store's command feed shows the read's `EVALSHA` and no `HGET`
/// of `obj:wtok`; `DeleteLayer` never reads that key), no cold-start tick.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_of_an_object_deleted_during_its_read_answers_invalidated_and_ticks_nothing() {
    let tv = TestValkey::setup().await;
    let cache = StructuralCache::new(BUDGET_BYTES);
    let ls = layer_store(&tv).with_structural_cache(cache.clone());
    let layer = "cs-retired";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    warm_snapshot_script(&tv, oid).await;
    let gate = ReplyGate::new();
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        hold_snapshot_read(
            tv.key_schema().obj_header(&oid.to_string()),
            SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
            gate.clone(),
        ),
    )
    .await;
    let client = client_through(&proxy).await;
    let metrics = hooked_metrics(layer, &[&client]).await;
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let wtok = tv.key_schema().obj_write_token(&oid.to_string());
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let task = spawn_cold_start(&client, tv.key_schema(), &cache, oid, None);

    gate.reply_held().await;
    ls.delete_layer(&session_id, layer, mock_ext_store())
        .await
        .expect("delete_layer must succeed");
    gate.open();
    let answer = join_cold_start(task, "once its reply is released").await;
    let fence = monitor_fence(tv.prefix());
    let _: i64 = tv.client().pool.exists(&fence).await.unwrap();
    let commands = tap.drain_until(&fence).await;

    assert!(
        matches!(answer, Ok(ColdStart::Invalidated)),
        "a retirement between the read and the admission refuses the entry; got {answer:?}"
    );
    assert!(
        commands
            .iter()
            .any(|c| c.name() == "EVALSHA" && c.args.get(3) == Some(&header_key)),
        "fixture: the MONITOR window must hold the cold start's read, or the HGET check below is \
         vacuous"
    );
    assert!(
        !commands
            .iter()
            .any(|c| c.name() == "HGET" && c.args.get(1) == Some(&wtok)),
        "an invalidated cold start reads no commit token; commands: {commands:?}"
    );
    assert_eq!(
        (proxy.hits(), structural_reads(&metrics, "ok")),
        (1, 1),
        "fixture: the held read is the cold start's one read, and it returned the object's state"
    );
    assert!(
        matches!(cache.lookup(oid), CacheLookup::Miss) && cache.entry_count() == 0,
        "nothing is stored and no negative entry is recorded for the retired object"
    );
    assert_eq!(
        cold_starts_by_route(&metrics),
        (0, 0, 0),
        "an invalidated cold start is neither an entry nor a verdict (atomic, fenced, \
         uncacheable)"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 7. Task 2's deferred items, end to end
// ═════════════════════════════════════════════════════════════════════════════

/// Objects that fill a busy full cache; far from any id the store allocates
/// in these tests.
const RESIDENTS: [u64; 3] = [900_001, 900_002, 900_003];
/// How often each resident is read, so its frequency in mini-moka's sketch
/// beats a new key's by a wide margin (W-TinyLFU admits a new key over a
/// victim only if it is read strictly more often).
const RESIDENT_READS: usize = 8;

/// Store `state` for `object_id` as a cold start does — `admit` under a
/// ticket, then a put-back — and return whether it is stored. It is stored
/// as a cold start holds it ([`as_cold_started`]), so it weighs
/// [`charged_weight`] of `state`.
fn store_entry(cache: &StructuralCache, object_id: u64, state: &CompactState) -> bool {
    let entry = CacheEntry::new(
        object_id,
        "resident-commit-chain".to_owned(),
        1,
        HashSet::new(),
        as_cold_started(state),
    );
    let guard = cache
        .admit(cache.begin_cold_start(), entry.clone())
        .unwrap_or_else(|refused| panic!("fixture: {object_id} must be admissible: {refused:?}"));
    guard.put_back(entry).is_stored()
}

/// A cache sized for four `state` entries, each weighed whole — its state
/// and plan indexes — so mini-moka holds three, the side store one, and the
/// per-entry cap admits one; with [`RESIDENTS`] filling mini-moka, each read
/// [`RESIDENT_READS`] times. The budget is four state weights, or more when
/// the entry's indexes weigh more than φ of its state (a small layer's
/// fixed overhead), so the resident budget holds four whole entries.
fn full_cache(state: &CompactState, negative_ttl: Duration) -> StructuralCache {
    let weight = charged_weight(state);
    let whole =
        CacheEntry::new(0, String::new(), 1, HashSet::new(), as_cold_started(state)).weight_bytes();
    // The smallest budget whose resident budget, × (1 + φ), holds four
    // whole entries, with room for the weigher's 4-byte rounding.
    let raised = 100 + live_edit::storage::structural_cache::PLAN_INDEX_BUDGET_CENTI;
    let budget = (4 * weight).max((4 * whole * 100).div_ceil(raised) + 16);
    let cache = StructuralCache::with_ttls(budget, LONG_TTL, negative_ttl);
    assert!(
        cache.max_entry_bytes() >= weight,
        "fixture: the cap admits one resident"
    );
    for object_id in RESIDENTS {
        assert!(
            store_entry(&cache, object_id, state),
            "fixture: resident {object_id}"
        );
    }
    for _ in 0..RESIDENT_READS {
        for object_id in RESIDENTS {
            match cache.lookup(object_id) {
                CacheLookup::Hit(guard) => drop(guard),
                other => panic!("fixture: resident {object_id} must hit; got {other:?}"),
            }
        }
    }
    assert_eq!(
        cache.entry_count(),
        RESIDENTS.len() as u64,
        "fixture: mini-moka holds the three residents"
    );
    cache
}

/// A filler state of `sections` sections under the root, each with one
/// inline field, in the compact form a cache entry holds.
fn filler_state(sections: u64) -> CompactState {
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_owned(),
        ..ObjectState::default()
    };
    for id in 1..=sections {
        state.sections.insert(
            id,
            Section {
                section_id: id,
                parent_id: ROOT,
                name: format!("s{id}"),
                section_type: 6,
                section_order: id as u32,
                first_version: 1,
                last_version: 1,
            },
        );
        state.fields.insert(
            (id, "default".to_owned()),
            Field {
                section_id: id,
                key_name: "default".to_owned(),
                value: FieldValue::Inline(vec![7; 24]),
                set_order: id,
                authoring_set_order: id,
                version: 1,
            },
        );
    }
    state.children.insert(ROOT, (1..=sections).collect());
    compact(&state)
}

/// Deferred from Task 2 (PM D1): a cold-started entry that mini-moka's
/// admission refuses is kept in the side store and served as a `Hit` — a
/// loaded entry, never an uncacheable verdict, and never a second cold start.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_entry_refused_by_mini_moka_is_served_as_a_hit_and_never_ticks_uncacheable() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-side-store").await;
    let state = normalized_fresh_state(&tv, oid).await;
    let cache = full_cache(&state, LONG_TTL);
    let metrics = hooked_metrics("cs-side-store", &[tv.client()]).await;
    assert!(
        matches!(cache.lookup(oid), CacheLookup::Miss),
        "the caller's lookup misses"
    );

    let loaded = expect_loaded(
        cold_start(tv.client(), tv.key_schema(), &cache, oid, CLIENT_ID).await,
        "into a full cache",
    );
    assert_eq!(
        loaded.entry.entry().state_weight_bytes,
        charged_weight(&state),
        "fixture: the candidate weighs exactly one resident"
    );
    let entry = loaded.entry.entry().clone();
    assert!(
        loaded.entry.put_back(entry).is_stored(),
        "a refused admission still stores the entry"
    );
    assert_eq!(
        cache.entry_count(),
        4,
        "fixture: mini-moka must have refused the candidate, which sits in the side store beside \
         the three residents (admitted, it would have evicted one)"
    );

    for round in 1..=3 {
        match cache.lookup(oid) {
            CacheLookup::Hit(guard) => drop(guard),
            other => panic!("lookup {round} after a refused admission must hit; got {other:?}"),
        }
    }
    assert_eq!(
        (
            cold_starts(&metrics, "atomic"),
            cold_starts(&metrics, "uncacheable"),
            structural_reads(&metrics, "ok"),
        ),
        (1, 0, 1),
        "one cold start on the atomic route and one read: the refused admission is a loaded \
         entry, never an uncacheable tick, and costs no second cold start"
    );
}

/// Deferred from Task 2 (PM D1): an object over the per-entry cap, under a
/// busy full cache, gets exactly one read; from then on the caller's lookups
/// route it `Uncacheable { OverWeight }` from its negative entry — which
/// lives outside mini-moka, so the churn cannot evict it — until the entry's
/// absolute TTL, which the lookups do not extend. (The verdict this test
/// once drove, `AboveColdStartCeiling`, no longer exists: an object above the
/// ceilings now loads on the fenced route.)
///
/// The caller is modelled as production's is: look up, and cold-start only
/// on a `Miss`. Timing-safe both ways: lookups run until three quarters of
/// the TTL, measured from an instant before the cold start (so before the
/// entry was recorded), so each must find the entry live; the final lookup
/// runs at the TTL plus slack after the cold start returned (so after the
/// entry was recorded). A lookup that extended the entry's life would keep it
/// past that final check. Slow (about 4 s) by design: the negative TTL must
/// be real time, since the cache reads the real clock, and long enough that
/// the cold start's read fits well inside three quarters of it on a loaded
/// runner.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_over_the_weight_cap_under_a_busy_full_cache_reads_once_then_routes_uncacheable_until_its_ttl(
) {
    const NEGATIVE_TTL: Duration = Duration::from_secs(3);
    const SLACK: Duration = Duration::from_millis(500);
    /// Pacing between the caller's lookups.
    const PACE: Duration = Duration::from_millis(25);
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-busy-ttl").await;
    let weight = charged_weight(&normalized_fresh_state(&tv, oid).await);
    let client = default_ceiling_client().await;
    let metrics = hooked_metrics("cs-busy-ttl", &[&client]).await;
    // One section: a resident, and so the cap, far lighter than the object.
    let filler = filler_state(1);
    let cache = full_cache(&filler, NEGATIVE_TTL);
    assert!(
        weight > cache.max_entry_bytes(),
        "fixture: the object ({weight} bytes) must be over the cap of one filler resident ({})",
        cache.max_entry_bytes()
    );
    let mut churn = 950_000;
    let mut cold_starts_issued = 0;
    let mut uncacheable_lookups = 0;
    let mut recorded = None;

    let before = Instant::now();
    while before.elapsed() < NEGATIVE_TTL * 3 / 4 {
        match cache.lookup(oid) {
            CacheLookup::Miss => {
                cold_starts_issued += 1;
                let reason = expect_uncacheable(
                    cold_start(&client, tv.key_schema(), &cache, oid, CLIENT_ID).await,
                    "over the weight cap",
                );
                recorded = Some(Instant::now());
                assert_eq!(reason, UncacheableReason::OverWeight(weight));
            }
            CacheLookup::Uncacheable {
                reason: UncacheableReason::OverWeight(w),
            } if w == weight => uncacheable_lookups += 1,
            other => panic!("the object must miss or route uncacheable; got {other:?}"),
        }
        // Keep the cache full and busy: a new key stored, a resident read.
        assert!(store_entry(&cache, churn, &filler));
        churn += 1;
        drop(cache.lookup(RESIDENTS[churn as usize % RESIDENTS.len()]));
        tokio::time::sleep(PACE).await;
    }
    assert!(
        before.elapsed() < NEGATIVE_TTL,
        "fixture: every lookup above must have run inside the TTL"
    );

    assert_eq!(
        cold_starts_issued, 1,
        "inside the negative TTL the object is cold-started once"
    );
    assert!(
        uncacheable_lookups >= 2,
        "the negative entry must route the object repeatedly; got {uncacheable_lookups}"
    );
    assert_eq!(
        (
            too_large_and_ok_reads(&metrics),
            all_structural_reads(&metrics),
            cold_starts(&metrics, "uncacheable")
        ),
        ((0, 1), 1, 1),
        "exactly one read, and one uncacheable verdict"
    );
    let expiry = recorded.expect("the first lookup cold-started the object") + NEGATIVE_TTL + SLACK;
    tokio::time::sleep(expiry.saturating_duration_since(Instant::now())).await;
    assert!(
        matches!(cache.lookup(oid), CacheLookup::Miss),
        "after its absolute TTL the negative entry is gone, despite the lookups, and the object \
         is cold-started again"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 8. Metrics
// ═════════════════════════════════════════════════════════════════════════════

/// One registry across every route and every no-verdict answer: each cold
/// start that ends in an entry or a verdict ticks its route once and records
/// one duration; superseded, absent and disabled cold starts tick nothing;
/// every read shows under `consumer="structural_cache"`, the fenced ones on
/// `path="fenced"`; and no `atomic_raised` series exists (T10).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_metrics_tick_once_per_route_and_report_the_structural_cache_consumer() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-metrics").await;
    let superseded = seeded_object(&tv, &ls, "cs-metrics-superseded").await;
    let _: i64 = tv
        .client()
        .pool
        .hset(
            tv.key_schema().obj_header(&superseded.to_string()),
            [("superseded", "1")],
        )
        .await
        .unwrap();
    let bound = item_bound(&tv, oid).await;
    let fenced_client = client_with_limits(bound - 1, NO_BYTE_LIMIT).await;
    let far_client = client_with_limits(
        (bound - 1) / STRUCTURAL_COLD_START_ITEMS_FACTOR,
        NO_BYTE_LIMIT,
    )
    .await;
    let metrics = hooked_metrics("cs-metrics", &[tv.client(), &fenced_client, &far_client]).await;
    let keys = tv.key_schema();
    let fresh = || StructuralCache::new(BUDGET_BYTES);

    // Ticked: atomic, fenced (twice: just above, and above the raised
    // ceiling), uncacheable (over the weight cap).
    expect_loaded(
        cold_start(tv.client(), keys, &fresh(), oid, CLIENT_ID).await,
        "atomic",
    );
    expect_loaded(
        cold_start(&fenced_client, keys, &fresh(), oid, CLIENT_ID).await,
        "fenced",
    );
    expect_loaded(
        cold_start(&far_client, keys, &fresh(), oid, CLIENT_ID).await,
        "fenced, far above",
    );
    expect_uncacheable(
        cold_start(
            tv.client(),
            keys,
            &StructuralCache::with_ttls(64, LONG_TTL, LONG_TTL),
            oid,
            CLIENT_ID,
        )
        .await,
        "over weight",
    );
    // Not ticked: superseded, absent, disabled.
    let answer = cold_start(tv.client(), keys, &fresh(), superseded, CLIENT_ID).await;
    assert!(
        matches!(answer, Ok(ColdStart::Superseded { .. })),
        "{answer:?}"
    );
    let answer = cold_start(tv.client(), keys, &fresh(), 987_654_321, CLIENT_ID).await;
    assert!(
        matches!(answer, Err(ServiceError::NotFound(_))),
        "{answer:?}"
    );
    expect_uncacheable(
        cold_start(tv.client(), keys, &StructuralCache::new(0), oid, CLIENT_ID).await,
        "disabled",
    );

    for (route, ticks) in [("atomic", 1), ("fenced", 2), ("uncacheable", 1)] {
        assert_eq!(
            (
                cold_starts(&metrics, route),
                cold_start_durations(&metrics, route)
            ),
            (ticks, (ticks > 0).then_some(ticks)),
            "route={route}: one tick and one duration per cold start that route ended, none for \
             a superseded, absent or disabled cold start"
        );
    }
    assert_eq!(
        uncacheable_verdicts(&metrics),
        BTreeMap::from([("over_weight".to_owned(), 1)]),
        "each uncacheable route tick is one verdict under its reason; a disabled cache is no \
         verdict"
    );
    // Atomic reads: atomic 1 ok; each fenced cold start 1 too_large; over
    // weight 1 ok; superseded 1 ok; absent 1 not_found; disabled none.
    // Fenced reads: one ok per fenced cold start.
    assert_eq!(
        (
            structural_reads(&metrics, "ok"),
            structural_reads(&metrics, "too_large"),
            structural_reads(&metrics, "not_found"),
            fenced_reads(&metrics, "ok"),
        ),
        (3, 2, 1, 2),
        "every read the cold starts issued is counted under consumer=\"structural_cache\" \
         (atomic ok, atomic too_large, atomic not_found, fenced ok)"
    );
    let rendered = render(&metrics);
    for (path, count) in [("atomic", 6.0), ("fenced", 2.0)] {
        assert_eq!(
            sample_value(
                &rendered,
                "live_edit_snapshot_read_duration_seconds_count",
                &[("consumer", "structural_cache"), ("path", path)],
            ),
            Some(count),
            "path={path}: every read records its wall time under that consumer"
        );
    }
    assert!(
        !rendered.contains("atomic_raised"),
        "no atomic_raised series exists: no cold start takes the raised read"
    );
}

/// The cold-start tick precedes the token `HGET`: a load whose `HGET` then
/// fails is still one `atomic` cold start — the read and the admission
/// happened — and the loaded entry is put back for the retry to repair.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_token_hget_failure_after_admission_is_counted_and_puts_the_entry_back() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-hget-fails").await;
    let wtok = tv.key_schema().obj_write_token(&oid.to_string());
    let rule: ProxyRule = Arc::new(move |args: &[Vec<u8>]| {
        let name = args.first()?.to_ascii_uppercase();
        (name == b"HGET" && args.get(1).map(Vec::as_slice) == Some(wtok.as_bytes())).then(|| {
            ProxyAction::Eval(error_reply_script("ERR injected commit-token read failure"))
        })
    });
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let client = client_through(&proxy).await;
    let metrics = hooked_metrics("cs-hget-fails", &[&client]).await;
    let cache = StructuralCache::new(BUDGET_BYTES);

    let answer = cold_start(&client, tv.key_schema(), &cache, oid, CLIENT_ID).await;

    assert!(
        matches!(answer, Err(ServiceError::Internal(_))),
        "a permanent token-read failure fails the cold start Internal; got {answer:?}"
    );
    assert_eq!(proxy.hits(), 1, "fixture: the proxy failed the token HGET");
    assert_eq!(
        (
            cold_starts(&metrics, "atomic"),
            cold_start_durations(&metrics, "atomic")
        ),
        (1, Some(1)),
        "the cold start that loaded is counted, though its token read then failed"
    );
    assert!(
        matches!(cache.lookup(oid), CacheLookup::Hit(_)),
        "the loaded entry is put back, so the retry repairs it rather than reading it again"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 9. Reporting a verdict
// ═════════════════════════════════════════════════════════════════════════════
//
// Every uncacheable verdict ticks `live_edit_structural_cache_uncacheable_total`
// under its reason once and logs one line — INFO, or DEBUG for the per-call
// `reload_limit` and `fenced_read_failed` — naming the object (and, from a
// structural `UpdateLayer` call, its session and layer) with what the verdict
// compared. The line is the deliverable, so it is pinned field by field under
// a JSON capture (test-writing guide, "when the log line is the
// deliverable"). The cold start runs inside the awaited future, so the
// `WithSubscriber` scope sees it on a multi-thread runtime. The
// `fenced_read_failed` lines are pinned beside their failures in section 3.

/// The verdict line's message. Restated from
/// `structural_cache_cold_start::UNCACHEABLE_VERDICT_MSG` (crate-private) on
/// purpose: operators search for the string, so a rename must break a test.
pub(crate) const UNCACHEABLE_VERDICT_MSG: &str =
    "structural cache found the object uncacheable; its structural edits take the WATCH path";

/// The nonzero cells of `live_edit_structural_cache_uncacheable_total`, by
/// reason. Panics on an absent cell: every reason is zero-seeded.
pub(crate) fn uncacheable_verdicts(metrics: &Metrics) -> BTreeMap<String, u64> {
    let text = render(metrics);
    [
        "over_weight",
        "no_chs_complete",
        "reload_limit",
        "fenced_read_failed",
        "over_deadline",
    ]
    .into_iter()
    .filter_map(|reason| {
        let value = sample_value(
            &text,
            "live_edit_structural_cache_uncacheable_total",
            &[("reason", reason)],
        )
        .unwrap_or_else(|| {
            panic!(
                "live_edit_structural_cache_uncacheable_total{{reason=\"{reason}\"}} is absent — \
                 it is zero-seeded"
            )
        }) as u64;
        (value != 0).then(|| (reason.to_owned(), value))
    })
    .collect()
}

/// Assert `captured` holds exactly one verdict line, at INFO, for `oid` and
/// `reason`, carrying every `(field, value)` in `fields` and none of `absent`
/// — an absent field is a missing key, not a `null` (the capture is JSON).
/// Returns the line.
pub(crate) fn assert_one_verdict_line<'c>(
    captured: &'c Captured,
    oid: u64,
    reason: &str,
    fields: &[(&str, serde_json::Value)],
    absent: &[&str],
) -> &'c serde_json::Value {
    assert_one_verdict_line_at(captured, "INFO", oid, reason, fields, absent)
}

/// [`assert_one_verdict_line`] at `level`: `DEBUG` for `reload_limit` and
/// `fenced_read_failed`, whose rates nothing bounds (the capture must record
/// DEBUG for it to be seen).
pub(crate) fn assert_one_verdict_line_at<'c>(
    captured: &'c Captured,
    level: &str,
    oid: u64,
    reason: &str,
    fields: &[(&str, serde_json::Value)],
    absent: &[&str],
) -> &'c serde_json::Value {
    let line = captured.only(UNCACHEABLE_VERDICT_MSG);
    assert_eq!(line["level"], level, "the verdict line is {level}: {line}");
    assert_field(line, "object_id", oid.into());
    assert_field(line, "reason", reason.into());
    for (field, want) in fields {
        assert_field(line, field, want.clone());
    }
    for field in absent {
        assert_field_absent(line, field, "this verdict does not compare it");
    }
    line
}

/// The bound fields the retired raised-ceiling read's verdict line carried.
/// No verdict line carries them any more; tests list them as absent so none
/// regains them.
pub(crate) const CEILING_FIELDS: [&str; 4] = [
    "item_bound",
    "raised_max_items",
    "byte_tally",
    "raised_max_bytes",
];
/// The fields a `fenced_read_failed` line carries. Of them, an
/// `over_deadline` line carries `size_items` and `size_bytes` too (the size
/// it was judged at); no other verdict's line carries any.
pub(crate) const FENCED_FIELDS: [&str; 6] = [
    "cause",
    "attempts",
    "size_items",
    "size_bytes",
    "waited_ms",
    "cap_ms",
];
/// The fields an `over_weight` line carries.
pub(crate) const WEIGHT_FIELDS: [&str; 2] = ["weight_bytes", "max_entry_bytes"];
/// The fields only an `over_deadline` line carries: the configured RPC
/// deadline it was judged against, and what the fenced read would keep back
/// of it at the judged size.
pub(crate) const DEADLINE_FIELDS: [&str; 2] = ["rpc_deadline_ms", "reserve_ms"];
/// The fields only a caller that names its layer adds (the read side's
/// `with_layer`); a bare cold start names the object only.
pub(crate) const LAYER_FIELDS: [&str; 2] = ["session_id", "layer_name"];

/// A cold start's verdict on `client`: the answer's reason, the verdicts
/// ticked on a registry of `client`'s own, and the captured log.
async fn reported_cold_start(
    tv: &TestValkey,
    client: &ValkeyClient,
    cache: &StructuralCache,
    oid: u64,
    context: &str,
) -> (UncacheableReason, BTreeMap<String, u64>, Captured) {
    let metrics = hooked_metrics(context, &[client]).await;
    let (answer, captured) =
        capture(cold_start(client, tv.key_schema(), cache, oid, CLIENT_ID)).await;
    (
        expect_uncacheable(answer, context),
        uncacheable_verdicts(&metrics),
        captured,
    )
}

/// An unmarked object: one `no_chs_complete` tick and one line naming the
/// object and reason, with no bound fields.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_no_chs_complete_verdict_reports_once_with_no_bounds() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-report-unmarked").await;
    unmark(&tv, oid).await;
    let client = client_with_limits(SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT, NO_BYTE_LIMIT).await;
    let cache = StructuralCache::new(BUDGET_BYTES);

    let (reason, verdicts, captured) =
        reported_cold_start(&tv, &client, &cache, oid, "cs-report-unmarked").await;

    assert_eq!(reason, UncacheableReason::NoChsComplete);
    assert_eq!(
        verdicts,
        BTreeMap::from([("no_chs_complete".to_owned(), 1)])
    );
    let mut absent = CEILING_FIELDS.to_vec();
    absent.extend(FENCED_FIELDS);
    absent.extend(WEIGHT_FIELDS);
    absent.extend(LAYER_FIELDS);
    absent.extend(DEADLINE_FIELDS);
    assert_one_verdict_line(&captured, oid, "no_chs_complete", &[], &absent);
}

/// Over the per-entry cap: one `over_weight` tick and one line naming the
/// entry's estimated weight against the cap.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_start_over_weight_verdict_reports_the_weight_against_the_entry_cap_once() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-report-overweight").await;
    let weight = charged_weight(&normalized_fresh_state(&tv, oid).await);
    let client = client_with_limits(SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT, NO_BYTE_LIMIT).await;
    let cache = StructuralCache::with_ttls(64, LONG_TTL, LONG_TTL);
    assert!(weight > cache.max_entry_bytes(), "fixture: over the cap");

    let (reason, verdicts, captured) =
        reported_cold_start(&tv, &client, &cache, oid, "cs-report-overweight").await;

    assert_eq!(reason, UncacheableReason::OverWeight(weight));
    assert_eq!(verdicts, BTreeMap::from([("over_weight".to_owned(), 1)]));
    let mut absent = CEILING_FIELDS.to_vec();
    absent.extend(FENCED_FIELDS);
    absent.extend(LAYER_FIELDS);
    absent.extend(DEADLINE_FIELDS);
    assert_one_verdict_line(
        &captured,
        oid,
        "over_weight",
        &[
            ("weight_bytes", weight.into()),
            ("max_entry_bytes", cache.max_entry_bytes().into()),
        ],
        &absent,
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 10. A refused plain cold start's hold allowance (seam A)
// ═════════════════════════════════════════════════════════════════════════════

/// The allowance a refused plain cold start's decoded state and index build
/// are held to: the shipped plain byte ceiling's is the SDD's 5,535,891 B;
/// the product of the ceiling, `r` and `1 + b` is floored, exact in integers
/// (a ceiling of 10,000 B gives exactly 91,560 B) and saturates rather than
/// wraps.
#[test]
fn plain_cold_start_hold_allowance_is_the_ceiling_times_r_times_one_plus_b_floored() {
    assert_eq!(
        plain_cold_start_hold_allowance_bytes(SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT),
        5_535_891,
        "at the shipped byte ceiling"
    );
    assert_eq!(plain_cold_start_hold_allowance_bytes(0), 0);
    assert_eq!(
        plain_cold_start_hold_allowance_bytes(1),
        9,
        "2.8 x 3.27 = 9.156, floored"
    );
    assert_eq!(
        plain_cold_start_hold_allowance_bytes(10_000),
        91_560,
        "exact in integers"
    );
    assert_eq!(
        plain_cold_start_hold_allowance_bytes(u64::MAX),
        u64::MAX,
        "saturates"
    );
}

/// A structural `UpdateLayer`'s commit charge that refuses every hold — the
/// cold start's CR0 at seam A included — and records each need it prices.
#[derive(Default)]
struct RefusingCharge {
    needs: StdMutex<Vec<ObjectStateNeed>>,
}

impl SharedCommitCharge for RefusingCharge {
    fn held(&self) -> u64 {
        0
    }

    fn target(&self, need: &ObjectStateNeed) -> u64 {
        self.needs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(*need);
        object_state_charge_for_test(need)
    }

    fn rewrite_allowance(&self, need: &ObjectStateNeed) -> MutationAllowance {
        object_state_rewrite_allowance_for_test(0, need)
    }

    fn try_hold(&self, _target: u64) -> bool {
        false
    }

    fn try_hold_at_least(&self, _target: u64) -> bool {
        false
    }
}

/// One read-side call through `client` on `cache` under a
/// [`RefusingCharge`], a plain cold start's hold judged against `allowance`
/// when set (else the production allowance of the client's plain byte
/// ceiling), captured at DEBUG. Returns the answer, the one need CR0 priced
/// and the capture.
async fn read_under_a_refused_charge(
    client: &ValkeyClient,
    tv: &TestValkey,
    cache: &StructuralCache,
    oid: u64,
    allowance: Option<u64>,
) -> (
    Result<StructuralRead, ServiceError>,
    ObjectStateNeed,
    Captured,
) {
    let charge = RefusingCharge::default();
    let request =
        StructuralReadRequest::new(oid, READ_CLIENT_ID, READ_TOKEN).with_commit_charge(&charge);
    let request = match allowance {
        Some(allowance) => request.with_plain_cold_start_hold_allowance_for_test(allowance),
        None => request,
    };
    let (answer, captured) = capture_debug(read_for_structural_edit(
        client,
        tv.key_schema(),
        cache,
        &request,
    ))
    .await;
    let needs = charge
        .needs
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(needs.len(), 1, "one charge point priced: CR0; {needs:?}");
    (answer, needs[0], captured)
}

/// The DEBUG lines a refused plain cold start over its allowance logs: the
/// events carrying a `hold_bytes` field.
fn hold_over_allowance_lines(captured: &Captured) -> Vec<&serde_json::Value> {
    captured
        .events
        .iter()
        .filter(|event| event["fields"].get("hold_bytes").is_some())
        .collect()
}

/// W4-T5: the boundary of a refused plain cold start's hold allowance. On
/// the same stored cut, under a charge that refuses CR0: with the
/// allowance exactly the hold — the decoded state's estimate plus the index
/// build's peak, CR0's own need — the cold start loads, stores its entry
/// and ticks the `atomic` route; one byte below the hold it loads nothing —
/// nothing stored, no route tick, no read permit held — and logs one DEBUG
/// line naming the object, the hold, the allowance and the target. Both
/// answer `NeedsCharge` with CR0's target, the call's wait.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn refused_plain_cold_start_loads_at_its_hold_allowance_and_nothing_one_byte_over_it_ug2() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-hold-allowance").await;
    let metrics = hooked_metrics("cs-hold-allowance", &[tv.client()]).await;
    let needs_charge = |answer: &Result<StructuralRead, ServiceError>, context: &str| match answer {
        Ok(StructuralRead::NeedsCharge { target, .. }) => *target,
        other => panic!("{context}: a refused CR0 answers NeedsCharge; got {other:?}"),
    };

    // The production allowance, far above this object's hold.
    let calibration = StructuralCache::new(BUDGET_BYTES);
    let (answer, need, captured) =
        read_under_a_refused_charge(tv.client(), &tv, &calibration, oid, None).await;
    let target = needs_charge(&answer, "the production allowance");
    let ObjectStateNeed::Cached { held, growth, .. } = need else {
        panic!("CR0 prices the cached route's need; got {need:?}");
    };
    let hold = held + growth;
    assert!(
        hold > 1
            && hold <= plain_cold_start_hold_allowance_bytes(plain_ceilings(tv.client()).max_bytes),
        "fixture: the object's hold ({hold} B) is within the production allowance"
    );
    assert!(
        calibration.weight_hint(oid).is_some() && hold_over_allowance_lines(&captured).is_empty(),
        "fixture: within the production allowance the refused cold start stored its entry"
    );

    // At the allowance: loads.
    let at = StructuralCache::new(BUDGET_BYTES);
    let routes = cold_starts_by_route(&metrics);
    let (answer, need_at, captured) =
        read_under_a_refused_charge(tv.client(), &tv, &at, oid, Some(hold)).await;
    assert_eq!(
        need_at, need,
        "fixture: the same cut decoded to the same hold"
    );
    assert_eq!(needs_charge(&answer, "at the allowance"), target);
    assert!(
        at.weight_hint(oid).is_some(),
        "a hold exactly at the allowance loads and stores the entry"
    );
    assert!(
        hold_over_allowance_lines(&captured).is_empty(),
        "and logs no line"
    );
    assert_eq!(
        cold_starts_by_route(&metrics),
        (routes.0 + 1, routes.1, routes.2),
        "one atomic route tick (atomic, fenced, uncacheable)"
    );

    // One byte over: loads nothing.
    let over = StructuralCache::new(BUDGET_BYTES);
    let (routes, reads) = (
        cold_starts_by_route(&metrics),
        structural_reads(&metrics, "ok"),
    );
    let (answer, need_over, captured) =
        read_under_a_refused_charge(tv.client(), &tv, &over, oid, Some(hold - 1)).await;
    assert_eq!(
        need_over, need,
        "fixture: the same cut decoded to the same hold"
    );
    assert_eq!(
        needs_charge(&answer, "one byte over the allowance"),
        target,
        "the call waits for CR0's target"
    );
    assert_eq!(
        structural_reads(&metrics, "ok"),
        reads + 1,
        "fixture: the cold start read the object"
    );
    assert!(
        over.weight_hint(oid).is_none() && over.entry_count() == 0,
        "a hold one byte over the allowance stores nothing"
    );
    assert_eq!(
        cold_starts_by_route(&metrics),
        routes,
        "and ticks no route: neither an entry nor a verdict"
    );
    assert_eq!(in_flight(&over), NONE_IN_FLIGHT, "and holds no read permit");
    let lines = hold_over_allowance_lines(&captured);
    assert_eq!(lines.len(), 1, "one line; {}", captured.text);
    let line = lines[0];
    assert_eq!(line["level"], "DEBUG", "{line}");
    assert_field(line, "object_id", oid.into());
    assert_field(line, "hold_bytes", hold.into());
    assert_field(line, "allowance_bytes", (hold - 1).into());
    assert_field(line, "target_bytes", target.into());
    assert!(matches!(over.lookup(oid), CacheLookup::Miss));
}

/// W4-T5's route condition: the hold allowance judges a refused **plain**
/// cold start only. On a client whose plain item ceiling routes the object
/// to the fenced read, under a charge that refuses CR0 and an allowance of
/// 1 B (under any object's hold), the cold start still loads: it stores its
/// entry, ticks the `fenced` route, logs no over-allowance line and, once
/// the entry is put back, holds no read permit. The fenced read's refused
/// hold is the existing outside-the-line item, held under its grant until
/// the put-back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn refused_fenced_cold_start_over_the_plain_hold_allowance_still_loads_ug2() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let oid = seeded_object(&tv, &ls, "cs-hold-allowance-fenced").await;
    let client = fenced_route_client(&tv, oid).await;
    let metrics = hooked_metrics("cs-hold-allowance-fenced", &[&client]).await;
    let cache = StructuralCache::new(BUDGET_BYTES);
    let routes = cold_starts_by_route(&metrics);

    let (answer, need, captured) =
        read_under_a_refused_charge(&client, &tv, &cache, oid, Some(1)).await;

    let ObjectStateNeed::Cached { held, growth, .. } = need else {
        panic!("CR0 prices the cached route's need; got {need:?}");
    };
    assert!(
        held + growth > 1,
        "fixture: the hold ({} B) is over the 1 B allowance",
        held + growth
    );
    assert!(
        matches!(answer, Ok(StructuralRead::NeedsCharge { .. })),
        "a refused CR0 answers NeedsCharge; got {answer:?}"
    );
    assert_eq!(
        cold_starts_by_route(&metrics),
        (routes.0, routes.1 + 1, routes.2),
        "one fenced route tick (atomic, fenced, uncacheable): the fenced read loaded"
    );
    assert!(
        cache.weight_hint(oid).is_some(),
        "the fenced cold start stored its entry, the plain allowance notwithstanding"
    );
    assert!(
        hold_over_allowance_lines(&captured).is_empty(),
        "and logs no over-allowance line"
    );
    assert_eq!(
        in_flight(&cache),
        NONE_IN_FLIGHT,
        "and holds no read permit"
    );
}
