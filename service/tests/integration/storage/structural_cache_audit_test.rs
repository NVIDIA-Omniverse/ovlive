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

//! The structural cache's sampled audit —
//! `storage::structural_cache_audit::StructuralCacheAudit` and its
//! `LayerStore` wiring (b.dkt Epic 4 Task 6; structural-edits SDD SR-3.6,
//! SR-2.4, SR-3.4) — and, above the plain atomic-read ceilings, its fenced
//! read under the cold start's hook in its audit mode
//! (`storage::structural_cache_fenced::ColdStartFenceHook::for_audit`).
//!
//! Every object is seeded through the real writers — `create_layer`, then
//! `update_layer` commits carrying their delta's OMPF as relay bytes — and
//! every entry an audit is handed comes from the read side
//! (`read_for_structural_edit`), as the committed structural call's would.
//! Audits run at rate 1.0, so each trigger samples. An object is put above
//! the plain ceilings by a client whose item (or byte) ceiling is under it.
//!
//! | Area | What it proves |
//! |---|---|
//! | Match | an unchanged object audits `match`, with the entry behind the header (the repair replays) and current alike, and ticks only `match` |
//! | Same cut | commits landing between the audit's read and its repair (the read's reply held at a proxy) still audit `match`: the copy is repaired to the read's cut, not the header; a cut behind the entry (a rolled-back header) audits `skipped` (`regressed`) and keeps the entry |
//! | Diverged | after `rewrite_out_of_band` of a section row, a children list or a dangling list, or an out-of-band rewrite of a field row's external hash, the audit answers `diverged` at the entry's cut naming that key, drops the entry and leaves the keyspace byte-identical (`DUMP`); a checkout taken before the audit then has its put-back refused (and one taken before a `match` has it accepted); one WARN carries `object_id`, `seq` and `first_difference`; an inline value rewritten out of band is a `match` (the compact entry holds no inline value) |
//! | Skipped | a repair past its budget (`too_far`), a trimmed stream (`gap`), an unmarked object (`NoChsComplete`), a superseded and a deleted object (`Gone`) each tick `skipped` only; so does a store error on the snapshot read or the repair's read (`ReadFailed`, injected at a proxy), and an object deleted between the read and the repair (`Gone`) |
//! | Repair bounds | the audit repairs under the production bounds, its 64 KiB read and replay caps retired: a pending entry past 64 KiB, and five records whose delta parts sum past 64 KiB, are `match` with the repair held on the upload budget and given back; a cut one entry past the repair's 1,024-entry budget is still `skipped` (`too_far`) |
//! | Charge on `B` | with the audit's real charge on an upload gate (`UploadBudgetAuditCharges`): refused at its first read on a budget an upload fills, or between the two records of one chunk (the second record's RP2, on a budget with room for every earlier point), the audit is `skipped` (`ChargeRefused`) with the budget back to the uploads', no slot taken, nothing queued and the cache's entry unchanged; a read over the per-audit cap is `skipped` (`ChargeOverCap`) having held only the first read's RP0; granted, its peak is the record's RP2 as priced on the index-less copy (the per-record estimate with no plan indexes), on `match` and on `diverged` alike, given back after; with the budget full, a processing step queued or an admission queued, the audit never waits and the waiter stays queued; cancelled mid repair read, or failed on it (`ReadFailed`), the budget is given back |
//! | Above the ceiling | with the switch off, `skipped` (`AboveCeiling`) from the plain read alone, with no fenced tick; with it on, `match` through the fenced read above the plain item ceiling, far above it (past where the retired raised read's item ceiling ended) and above the plain byte ceiling alone (past the raised byte ceiling); `diverged` there too after an out-of-band rewrite, the entry dropped |
//! | The fenced read | one `path="fenced"` `ok` read and one fence hold under `consumer="structural_cache_audit"`, none under `structural_cache`, one `result="read"` tick, one acquire; `MONITOR`: one snapshot-script `EVALSHA` — the plain read, at the client's plain ceilings, never raised — then one acquire and one release, every fan-out command between them |
//! | Pacing | a second fenced audit within the interval (from the first read's lease grant), on a clone of the audit, is `skipped` (`Paced`) with a `paced` tick and no acquire; a clone at a zero interval sharing the same slot reads again; an unmarked object above the ceiling is `skipped` (`NoChsComplete`) before any capacity (it ends while the test holds the cold-start permit), so the next marked audit reads |
//! | Pacing, continued | a busy audit, an object gone during the fenced read and an unmarked object are granted no lease, so they use up no interval: the same audit reads next |
//! | Busy | with the one-at-a-time cold-start permit, both plain permits, every `P_f` permit or the whole catch-up budget `T` held by the test, the audit ends `skipped` (`Busy`) within half its waiting cap while they are still held, ticks `busy`, sends no acquire, counts no `refused_timeout` or `waited` admission, and holds nothing afterwards (and the one-at-a-time permit it may have taken is free again) |
//! | Gone, closed | an object whose header goes during the fenced read is `skipped` (`Gone`) with no fenced result; a `P_f` gate closed at shutdown ends the audit `skipped` (`ReadFailed`) with no fenced result and no acquire |
//! | Memory | across its acquire (the reply held at a proxy) the audit holds its pair, one `P_f` permit and `(SNAPSHOT_MEMORY_FACTOR + 1) × b` of `T` — `b` from the plain read's `TOO_LARGE` answer, `T` read at run time; at its repair's read (held too), after the decode, it holds nothing |
//! | Live lease | a forged live reader lease makes the fenced read sleep out `RETRY_AFTER` holding no pair, no `P_f` and no grant; once the lease is gone the audit compares, `match`, and releases its own lease |
//! | Plain read | takes no cold-start permit: it compares with every one held elsewhere |
//! | Isolation | a `match` by repair, a fenced `match` and a `diverged` leave the lookups, reloads, divergence, replayed-entries, repair-bytes, cold-start and every `consumer="structural_cache"` snapshot-read, fence and read-permit series unchanged; the reads show under `consumer="structural_cache_audit"` |
//! | One at a time | while an audit runs, a trigger on the store or a clone of it starts none; the flag clears when the audit ends and when it panics |
//! | Shutdown | the token ends an audit held before its read, and one mid fenced read (its fan-out held back by the read's fault seam), which gives back its pair, `P_f`, grant and lease; neither ticks anything, and no audit starts after it |
//! | Defaults | `LayerStore::new`'s audit never starts; `StructuralCacheAudit::new` refuses a rate outside `[0, 1]` |
//!
//! The WARN test runs on a current-thread runtime with a thread-local
//! subscriber (test-writing guide, "A line emitted from a spawned task"), so
//! `TestValkey`'s `Drop` cleanup is skipped there and the test removes its
//! keys itself. Every wait is bounded by [`AUDIT_BUDGET`] (20 s; the proxy's
//! `reply_held` and the `MONITOR` drain by their own 20 s). The longest test,
//! `audit_that_cannot_compare_is_skipped_with_its_reason`, waits on five
//! audits in sequence: 100 s at full budget plus five fixture setups, under
//! nextest's 120 s hang-killer; so does the busy test (four busy audits and
//! one fenced read); the other fenced-read tests wait at most four budgets
//! (80 s). A healthy run pays none of it: milliseconds per audit,
//! plus about a second for a fenced read that meets a fence's `RETRY_AFTER`.
//!
//! **No figure the cost model or configuration owns is restated.** The memory
//! factor, the budget `T`, the `P_f` limit and the waiting cap are read at
//! run time from the constants, the client and `snapshot_fence::coefficients`.
//! The fenced read's shutdown test needs `test-fault-injection` (the read's
//! fault seam).
//!
//! The knob parsing and the chart pin of the switch's default are in
//! `src/config.rs`'s tests; `first_difference`'s naming of object columns and
//! time samples, and its walk order, inline in
//! `src/storage/structural_cache_audit.rs`; the hook's own events in
//! `src/storage/structural_cache_fenced.rs`; the zero-seeded audit series and
//! the `structural_cache_audit` snapshot-read cells in
//! `observability_metrics_test.rs`; the repair to a target itself in
//! `structural_cache_read_test`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use fred::interfaces::{ClientLike, HashesInterface, KeysInterface, StreamsInterface};
use fred::types::scan::ScanType;
use fred::types::scripts::Script;
use fred::types::streams::XCap;
use fred::types::Key;
use futures::{FutureExt, Stream, StreamExt};
use oodf_ffi::OwnedDelta;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use live_edit::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{Metrics, SNAPSHOT_FENCE_ACQUIRE_RESULTS};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::diff_read::{SnapshotCeilings, SnapshotReadConsumer};
use live_edit::storage::layer::{
    LayerStore, CHS_COMPLETE_MARKED, HEADER_FIELD_CHS_COMPLETE, HEADER_FIELD_COMMIT_CHAIN,
};
use live_edit::storage::memory_budget::{kib_for_bytes, SNAPSHOT_MEMORY_FACTOR};
use live_edit::storage::snapshot_fence::{coefficients, LayerSize};
use live_edit::storage::structural_cache::{
    CacheEntry, CacheLookup, ColdStartReadsInFlight, PutBack, StructuralCache,
};
use live_edit::storage::structural_cache_audit::{AuditOutcome, AuditSkip, StructuralCacheAudit};
use live_edit::storage::structural_cache_cold_start::STRUCTURAL_COLD_START_ITEMS_FACTOR;
use live_edit::storage::structural_cache_fenced::wait_cap;
use live_edit::storage::structural_cache_read::{
    read_for_structural_edit, ReadyEntry, StructuralRead, StructuralReadRequest,
};
use live_edit::storage::structural_cache_repair::{
    ReloadReason, RepairBounds, TestHook, STRUCTURAL_REPAIR_MAX_TOTAL,
};
use live_edit::storage::structural_cache_sync::{
    STRUCTURAL_REPAIR_INITIAL_BYTES, STRUCTURAL_REPAIR_INITIAL_ENTRIES,
    STRUCTURAL_REPAIR_MAX_ENTRIES,
};
use live_edit::storage::test_support::{rewrite_out_of_band, OutOfBandChange, TestValkey};

use live_edit::domain::delta::MutationAllowance;
use live_edit::domain::replay::decode_stream_record;
use live_edit::grpc::upload_admission::{
    admission_charge, object_state_charge_for_test, object_state_rewrite_allowance_for_test,
    UploadAdmissionConfig, UploadAdmissionGate, UploadBudgetAuditCharges, UploadReservation,
    UploadRpc, SAMPLED_AUDIT_CHARGE_CAP_BYTES,
};
use live_edit::storage::commit_charge::{
    AuditCharge, AuditChargeSource, ObjectStateNeed, RepairTransient, SharedCommitCharge,
    UnboundedAuditCharges,
};
use live_edit::storage::notifications::RELAY_OMPF_FIELD;
use live_edit::storage::structural_cache::estimate_replay_record_growth;

use crate::common::owned_delta::{delta_with_set_field_external, delta_with_set_field_inline};
use crate::storage::snapshot_read_atomic_test::{
    create, create_cmd, create_session, every_shape_state, ext_hash, layer_store, mock_ext_store,
    structural, unmark, DANGLING_PARENT, ROOT, SIBLING, WORLD,
};
use crate::storage::snapshot_read_support::{
    client_through, client_through_with_limits, client_with_limits, error_reply_script,
    monitor_fence, read_atomic, read_too_large, script_frames_naming, test_valkey_addr, MonitorTap,
    MonitoredCommand, ProxyAction, ProxyRule, ReplyGate, ScriptReplyProxy,
};
use crate::storage::structural_cache_cold_start_test::header;
use crate::storage::structural_cache_read_test::{commit_to, stream_entries};

/// A budget no fixture entry comes near.
const BUDGET_BYTES: u64 = 64 * 1024 * 1024;
/// The `client_id` every seeded commit is made under.
const WRITER: &str = "structural-audit-writer";
/// The `client_id` of the structural call whose entry is audited.
const CLIENT_ID: &str = "structural-audit-client";
/// That call's commit token; no commit records it.
const TOKEN: &str = "0123456789abcdef0123456789abcdef";
const CLIENT_TS: u64 = 7_000;
/// A byte ceiling no fixture comes near, for the clients whose item ceiling
/// is the one under test.
const NO_BYTE_LIMIT: u64 = 1 << 40;
/// Bound on one wait for an audit, a gate or a permit; a correct audit ends
/// in milliseconds.
const AUDIT_BUDGET: Duration = Duration::from_secs(20);
/// How long a test watches for a permit it expects to be free (taken at
/// once) or held (never taken while it watches).
const BLOCKED_WINDOW: Duration = Duration::from_millis(500);
/// Pause between samples of a condition a test waits for.
const POLL_PAUSE: Duration = Duration::from_millis(5);
/// A deadline no test comes near, for the `P_f` permits a test holds.
const LONG_TTL: Duration = Duration::from_secs(3_600);
/// The header's `seq` column (`HEADER_FIELD_SEQ` is crate-private).
const SEQ_COLUMN: &str = "seq";
/// The header column a `CreateLayer` overwrite sets; no public constant.
const SUPERSEDED_COLUMN: &str = "superseded";
const AUDIT_TOTAL: &str = "live_edit_structural_cache_audit_total";
const AUDIT_FENCED_TOTAL: &str = "live_edit_structural_cache_audit_fenced_total";
/// The audit's snapshot-read consumer label.
const AUDIT_CONSUMER: &str = "structural_cache_audit";
/// A reader deadline far past any test's end (ms since the Unix epoch, in
/// the year 2286). A forged `r:*` field holding it is a live lease to every
/// acquire, which answers `RETRY_AFTER` `epoch` and writes nothing.
const FORGED_LEASE_DEADLINE_MS: i64 = 9_999_999_999_999;
/// The forged lease's field in the fence key.
const FORGED_LEASE_FIELD: &str = "r:forged-by-the-audit-test";

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// One marked `every_shape_state` object — with a list under
/// `DANGLING_PARENT`, so its entries carry a dangling parent — after one
/// fast-path commit; a store and a cache over it; a metrics registry hooked
/// onto the store client.
struct Fx {
    tv: TestValkey,
    ls: LayerStore,
    session_id: String,
    layer: String,
    oid: u64,
    cache: StructuralCache,
    metrics: Arc<Metrics>,
}

impl Fx {
    async fn new(layer: &str) -> Self {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only(layer));
        tv.client()
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let cache = StructuralCache::new(BUDGET_BYTES);
        let ls = layer_store(&tv).with_structural_cache(cache.clone());
        let session_id = create_session(&tv, layer).await;
        let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
        let fx = Self {
            tv,
            ls,
            session_id,
            layer: layer.to_owned(),
            oid,
            cache,
            metrics,
        };
        fx.commit(delta_with_set_field_inline(WORLD, "label", vec![0x2A], 5))
            .await;
        fx
    }

    fn header_key(&self) -> String {
        self.tv.key_schema().obj_header(&self.oid.to_string())
    }

    /// Commit `delta` as [`WRITER`], with its OMPF as relay bytes.
    async fn commit(&self, delta: OwnedDelta) {
        commit_to(
            &self.ls,
            &self.session_id,
            &self.layer,
            self.oid,
            WRITER,
            Some(delta),
            &[],
            CLIENT_TS,
        )
        .await;
    }

    /// `n` fast-path writes of `WORLD`'s fields `<tag>0`, `<tag>1`, …: each a
    /// new field, so each moves the state.
    async fn commit_fast(&self, tag: &str, n: u64) {
        for i in 0..n {
            self.commit(delta_with_set_field_inline(
                WORLD,
                &format!("{tag}{i}"),
                vec![i as u8],
                100 + i,
            ))
            .await;
        }
    }

    /// The header's `(seq, commit_chain)`.
    async fn header_cut(&self) -> (u64, String) {
        let hdr = header(&self.tv, self.oid).await;
        (
            hdr[SEQ_COLUMN].parse().expect("the header seq is decimal"),
            hdr[HEADER_FIELD_COMMIT_CHAIN].clone(),
        )
    }

    async fn read_ready(&self) -> ReadyEntry {
        let request = StructuralReadRequest::new(self.oid, CLIENT_ID, TOKEN);
        match read_for_structural_edit(
            self.tv.client(),
            self.tv.key_schema(),
            &self.cache,
            &request,
        )
        .await
        {
            Ok(StructuralRead::Ready(ready)) => ready,
            other => panic!("fixture: the read side must answer Ready; got {other:?}"),
        }
    }

    /// The object's entry, current at the header, from the read side — left
    /// in the cache — and a copy of it to hand an audit.
    async fn entry(&self) -> CacheEntry {
        let ready = self.read_ready().await;
        let entry = ready.entry.entry().clone();
        assert!(
            ready.entry.put_back_held().is_stored(),
            "fixture: the entry is stored"
        );
        entry
    }

    /// The cached entry, if the object has one (a hit, put back at once).
    fn cached(&self) -> Option<CacheEntry> {
        match self.cache.lookup(self.oid) {
            CacheLookup::Hit(guard) => Some(guard.entry().clone()),
            CacheLookup::Miss => None,
            other => panic!("expected a hit or a miss; got {other:?}"),
        }
    }

    /// An audit at rate 1.0 with a token nothing cancels.
    fn audit(&self, above_ceiling: bool) -> StructuralCacheAudit {
        StructuralCacheAudit::new(
            1.0,
            above_ceiling,
            Arc::new(UnboundedAuditCharges),
            CancellationToken::new(),
        )
    }

    /// An audit at rate 1.0, below the ceilings only, whose repair holds its
    /// charge from `charges`, with a token nothing cancels, and the most each
    /// audit's charge held at once, in order (its outcome observer).
    fn audit_charged(
        &self,
        charges: Arc<dyn AuditChargeSource>,
    ) -> (StructuralCacheAudit, Arc<StdMutex<Vec<u64>>>) {
        let peaks = Arc::new(StdMutex::new(Vec::new()));
        let sink = Arc::clone(&peaks);
        let audit = StructuralCacheAudit::new(1.0, false, charges, CancellationToken::new())
            .with_outcome_observer_for_test(Arc::new(move |_outcome, peak| {
                sink.lock().unwrap().push(peak);
            }));
        (audit, peaks)
    }

    /// Spawn `audit` of `entry` on `client` and wait for its outcome.
    async fn run_on(
        &self,
        audit: &StructuralCacheAudit,
        client: &ValkeyClient,
        entry: &CacheEntry,
    ) -> AuditOutcome {
        let handle = audit
            .maybe_spawn(client, self.tv.key_schema(), &self.cache, entry)
            .expect("an audit at rate 1.0 with none running starts");
        finished(handle)
            .await
            .expect("an audit no token cancelled answers an outcome")
    }

    /// [`Self::run_on`] on the store client.
    async fn run(&self, audit: &StructuralCacheAudit, entry: &CacheEntry) -> AuditOutcome {
        self.run_on(audit, self.tv.client(), entry).await
    }

    /// A client with the shared atomic-read item ceiling one under the
    /// object's item bound, so its plain read answers `TOO_LARGE` and, with
    /// the switch on, the audit takes the fenced read; hooked onto this
    /// fixture's registry.
    async fn client_one_item_under_the_bound(&self) -> (ValkeyClient, u64) {
        let plain_items = self.item_bound().await - 1;
        let client = client_with_limits(plain_items, NO_BYTE_LIMIT).await;
        client
            .install_observability_hooks(Arc::clone(&self.metrics))
            .await;
        (client, plain_items)
    }

    /// The object's item bound, from a read under a one-item ceiling.
    async fn item_bound(&self) -> u64 {
        read_too_large(
            &client_with_limits(1, NO_BYTE_LIMIT).await,
            self.tv.key_schema(),
            self.oid,
        )
        .await
        .item_bound
    }

    /// The size the audit's fenced read starts from on a client whose plain
    /// item ceiling is `plain_items`: `LayerSize::from_too_large` of that
    /// ceiling's `TOO_LARGE` answer, with no recorded bytes per item — as the
    /// audit sizes it. Read on a client reporting nowhere.
    async fn estimate(&self, plain_items: u64) -> LayerSize {
        let answer = read_too_large(
            &client_with_limits(plain_items, NO_BYTE_LIMIT).await,
            self.tv.key_schema(),
            self.oid,
        )
        .await;
        LayerSize::from_too_large(&answer, None)
    }

    /// `live_edit_structural_cache_audit_total` as `(match, diverged,
    /// skipped)`. Panics on an absent cell: all three are zero-seeded.
    fn audits(&self) -> (u64, u64, u64) {
        let text = render(&self.metrics);
        let cell = |outcome: &str| {
            sample_value(&text, AUDIT_TOTAL, &[("outcome", outcome)])
                .unwrap_or_else(|| panic!("{AUDIT_TOTAL}{{outcome=\"{outcome}\"}} is zero-seeded"))
                as u64
        };
        (cell("match"), cell("diverged"), cell("skipped"))
    }

    /// `live_edit_structural_cache_audit_fenced_total` as `(read, paced,
    /// busy, failed)`. Panics on an absent cell: all four are zero-seeded.
    fn audit_fenced(&self) -> (u64, u64, u64, u64) {
        let text = render(&self.metrics);
        let cell = |result: &str| {
            sample_value(&text, AUDIT_FENCED_TOTAL, &[("result", result)]).unwrap_or_else(|| {
                panic!("{AUDIT_FENCED_TOTAL}{{result=\"{result}\"}} is zero-seeded")
            }) as u64
        };
        (cell("read"), cell("paced"), cell("busy"), cell("failed"))
    }

    /// `live_edit_snapshot_read_total{consumer, path, outcome}`. Panics on an
    /// absent cell: every atomic and fenced cell of the structural cache's
    /// two consumers is zero-seeded.
    fn reads_of(&self, consumer: &str, path: &str, outcome: &str) -> u64 {
        sample_value(
            &render(&self.metrics),
            "live_edit_snapshot_read_total",
            &[("consumer", consumer), ("path", path), ("outcome", outcome)],
        )
        .unwrap_or_else(|| {
            panic!("the {consumer} {path} {outcome} snapshot-read cell is zero-seeded")
        }) as u64
    }

    /// `live_edit_snapshot_read_total{consumer="structural_cache_audit",
    /// path="atomic", outcome}`.
    fn audit_reads(&self, outcome: &str) -> u64 {
        self.reads_of(AUDIT_CONSUMER, "atomic", outcome)
    }

    /// `live_edit_snapshot_read_total{consumer="structural_cache_audit",
    /// path="fenced", outcome}`.
    fn fenced_reads(&self, outcome: &str) -> u64 {
        self.reads_of(AUDIT_CONSUMER, "fenced", outcome)
    }

    /// `live_edit_snapshot_fence_acquire_total{result}`; absent reads 0.
    fn fence_acquires_of(&self, result: &str) -> u64 {
        sample_value(
            &render(&self.metrics),
            "live_edit_snapshot_fence_acquire_total",
            &[("result", result)],
        )
        .unwrap_or(0.0) as u64
    }

    /// Every fence acquire reply, over every result: the acquires sent.
    fn fence_acquires(&self) -> u64 {
        SNAPSHOT_FENCE_ACQUIRE_RESULTS
            .iter()
            .map(|result| self.fence_acquires_of(result))
            .sum()
    }

    /// The fence holds observed under `consumer`
    /// (`live_edit_snapshot_fence_hold_seconds_count`); absent reads 0.
    fn fence_holds(&self, consumer: &str) -> u64 {
        sample_value(
            &render(&self.metrics),
            "live_edit_snapshot_fence_hold_seconds_count",
            &[("consumer", consumer)],
        )
        .unwrap_or(0.0) as u64
    }

    /// `live_edit_catchup_build_admission_total{result}`; absent reads 0.
    fn admissions(&self, result: &str) -> u64 {
        sample_value(
            &render(&self.metrics),
            "live_edit_catchup_build_admission_total",
            &[("result", result)],
        )
        .unwrap_or(0.0) as u64
    }

    /// Mark the object's children-list index complete again (undo
    /// [`unmark`]).
    async fn remark(&self) {
        let _: i64 = self
            .tv
            .client()
            .pool
            .hset(
                self.header_key(),
                [(HEADER_FIELD_CHS_COMPLETE, CHS_COMPLETE_MARKED)],
            )
            .await
            .expect("HSET obj:hdr chs_complete must succeed");
    }
}

/// Load the fence scripts (acquire, release) into the store's script cache
/// with one fenced audit of an object of its own, under its own prefix and on
/// a client reporting nowhere: a later acquire is then one `EVALSHA` that
/// runs, not a `NOSCRIPT` round trip a proxy rule or a `MONITOR` record would
/// see twice, and this test's own object has no fence key yet.
async fn warm_fence_scripts(layer: &str) {
    let warm = Fx::new(&format!("{layer}-warm")).await;
    let entry = warm.entry().await;
    let client = client_with_limits(warm.item_bound().await - 1, NO_BYTE_LIMIT).await;
    let outcome = warm.run_on(&warm.audit(true), &client, &entry).await;
    assert_eq!(
        outcome,
        AuditOutcome::Match,
        "fixture: the fence scripts' warm-up audit reads fenced and matches"
    );
}

// ── The fenced read's capacity ──────────────────────────────────────────────

/// What a fenced read can hold, at one instant: the cache's cold-start reads
/// in flight (its pair counts as one `fenced`), the client's `P_f` permits
/// held, and the client's catch-up memory budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Capacity {
    in_flight: ColdStartReadsInFlight,
    /// `P_f` permits held: the gate's limit less those available.
    p_f_held: usize,
    /// Bytes held by grants.
    reserved_bytes: u64,
    /// Bytes held by top-ups.
    overcommit_bytes: u64,
}

const NONE_IN_FLIGHT: ColdStartReadsInFlight = ColdStartReadsInFlight {
    plain: 0,
    fenced: 0,
    raised: 0,
};

const ONE_FENCED: ColdStartReadsInFlight = ColdStartReadsInFlight {
    plain: 0,
    fenced: 1,
    raised: 0,
};

/// Nothing held.
const NOTHING_HELD: Capacity = Capacity {
    in_flight: NONE_IN_FLIGHT,
    p_f_held: 0,
    reserved_bytes: 0,
    overcommit_bytes: 0,
};

fn capacity(client: &ValkeyClient, cache: &StructuralCache) -> Capacity {
    let gate = &client.catchup_fenced_read_gate;
    Capacity {
        in_flight: cache.cold_start_reads_in_flight_for_test(),
        p_f_held: gate.limit() - gate.available(),
        reserved_bytes: client.catchup_build_budget.reserved_bytes(),
        overcommit_bytes: client.catchup_build_budget.overcommit_bytes(),
    }
}

/// Wait, up to [`AUDIT_BUDGET`], until `client` and `cache` hold exactly
/// `want`; panics with the last state seen.
async fn wait_for_capacity(
    client: &ValkeyClient,
    cache: &StructuralCache,
    want: Capacity,
    what: &str,
) {
    let deadline = Instant::now() + AUDIT_BUDGET;
    loop {
        let now = capacity(client, cache);
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

/// Wait, up to [`AUDIT_BUDGET`], until `reached` holds; panics naming `what`.
async fn wait_until(what: &str, reached: impl Fn() -> bool) {
    let deadline = Instant::now() + AUDIT_BUDGET;
    while !reached() {
        assert!(Instant::now() < deadline, "{what} within the budget");
        tokio::time::sleep(POLL_PAUSE).await;
    }
}

/// The memory the audit's fenced read reserves before its acquire, for a
/// layer estimated at `b` reply bytes, as `client`'s budget takes it:
/// `(SNAPSHOT_MEMORY_FACTOR + 1) × b`, the snapshot plus one window, in whole
/// KiB and clamped to the configured `T` — read from the client, never
/// restated.
fn expected_grant_bytes(client: &ValkeyClient, b: u64) -> u64 {
    let total_kib = client.catchup_build_budget.total_bytes() / 1024;
    kib_for_bytes(b.saturating_mul(SNAPSHOT_MEMORY_FACTOR + 1)).min(total_kib) * 1024
}

// ── The fence key ────────────────────────────────────────────────────────────

fn fence_key(tv: &TestValkey, oid: u64) -> String {
    tv.key_schema().obj_snapshot_fence(&oid.to_string())
}

/// The reader-lease fields (`r:*`) of the object's fence key.
async fn lease_fields(tv: &TestValkey, oid: u64) -> BTreeSet<String> {
    let fields: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(fence_key(tv, oid))
        .await
        .expect("HGETALL obj:snapfence must succeed");
    fields.into_keys().filter(|k| k.starts_with("r:")).collect()
}

/// Wait, up to [`AUDIT_BUDGET`], until the object's fence key holds no reader
/// lease: every lease taken was released.
async fn wait_until_no_lease(tv: &TestValkey, oid: u64, what: &str) {
    let deadline = Instant::now() + AUDIT_BUDGET;
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

// ── Scripts ──────────────────────────────────────────────────────────────────

/// The SHA of a script, for recognising its `EVALSHA` on a proxied
/// connection and in a `MONITOR` feed.
fn script_sha(text: &str) -> String {
    let script = Script::from_lua(text);
    let sha: &str = script.sha1();
    sha.to_owned()
}

fn snapshot_sha() -> String {
    script_sha(include_str!(
        "../../../src/storage/scripts/object_snapshot_read.lua"
    ))
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

/// Whether `command` is an `EVALSHA` of the script `sha`.
fn is_evalsha_of(command: &MonitoredCommand, sha: &str) -> bool {
    command.name() == "EVALSHA" && command.args.get(1).map(String::as_str) == Some(sha)
}

/// Wait for an audit task, within [`AUDIT_BUDGET`]; a panic in it fails the
/// test.
async fn finished(handle: JoinHandle<Option<AuditOutcome>>) -> Option<AuditOutcome> {
    tokio::time::timeout(AUDIT_BUDGET, handle)
        .await
        .expect("the audit must end within the budget")
        .expect("the audit task must not panic")
}

/// A before-read hook that reports each entry on `entered` and then waits
/// until `open` reads `true`. Once opened it stays open.
fn parking_hook() -> (TestHook, watch::Receiver<u64>, watch::Sender<bool>) {
    let (entered_tx, entered_rx) = watch::channel(0_u64);
    let (open_tx, open_rx) = watch::channel(false);
    let entered_tx = Arc::new(entered_tx);
    let hook: TestHook = Arc::new(move || {
        let (entered, mut open) = (Arc::clone(&entered_tx), open_rx.clone());
        async move {
            entered.send_modify(|n| *n += 1);
            let _ = open.wait_for(|&o| o).await;
        }
        .boxed()
    });
    (hook, entered_rx, open_tx)
}

/// Wait until the hook has been entered `n` times in all.
async fn entered(rx: &mut watch::Receiver<u64>, n: u64) {
    tokio::time::timeout(AUDIT_BUDGET, rx.wait_for(|&k| k >= n))
        .await
        .expect("the audit must reach its before-read hook within the budget")
        .expect("the hook's sender lives as long as the audit");
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. Match, and the same cut
// ═════════════════════════════════════════════════════════════════════════════

/// An unchanged object audits `match`, whether the entry is current or three
/// commits behind (the copy is repaired forward to the read's cut); the
/// cache's own entry is left as it was.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_of_an_unchanged_object_is_match() {
    let fx = Fx::new("audit-match").await;
    let earlier = fx.entry().await;
    fx.commit_fast("behind", 3).await;
    let audit = fx.audit(false);

    let at_the_header = fx.run(&audit, &fx.entry().await).await;
    let behind = fx.run(&audit, &earlier).await;

    assert_eq!(at_the_header, AuditOutcome::Match, "a current entry");
    assert_eq!(behind, AuditOutcome::Match, "an entry three commits behind");
    assert_eq!(fx.audits(), (2, 0, 0), "two audits, both match");
    assert!(
        fx.cached().is_some(),
        "a match leaves the cached entry in place"
    );
}

/// SR-3.6's same cut: the entry is two commits behind; the audit's read runs
/// on the store, and its reply is held at a proxy while two more commits
/// land; then the audit goes on. Its copy must be repaired to the read's cut
/// (two commits), not the header's (four), or it would hold the later
/// commits' fields and diverge.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_with_commits_landing_between_its_read_and_its_repair_is_match() {
    let fx = Fx::new("audit-same-cut").await;
    // The warm-up's cold start loads the snapshot script, so the audit's
    // read is one EVALSHA that runs, not a NOSCRIPT round trip.
    let entry = fx.entry().await;
    fx.commit_fast("before-read", 2).await;
    let (read_seq, _) = fx.header_cut().await;
    let gate = ReplyGate::new();
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        script_frames_naming(fx.header_key(), ProxyAction::HoldReplies(gate.clone())),
    )
    .await;
    let client = client_through(&proxy).await;
    client
        .install_observability_hooks(Arc::clone(&fx.metrics))
        .await;
    let handle = fx
        .audit(false)
        .maybe_spawn(&client, fx.tv.key_schema(), &fx.cache, &entry)
        .expect("an audit at rate 1.0 starts");

    gate.reply_held().await;
    fx.commit_fast("after-read", 2).await;
    gate.open();
    let outcome = finished(handle).await;

    assert_eq!(
        outcome,
        Some(AuditOutcome::Match),
        "the copy is compared with the read at the read's cut"
    );
    assert!(proxy.hits() >= 1, "fixture: the audit's read was held");
    assert_eq!(
        fx.header_cut().await.0,
        read_seq + 2,
        "fixture: two commits landed after the read's cut"
    );
    assert_eq!(fx.audits(), (1, 0, 0));
}

/// A cut behind the entry — the header rolled back one commit, `seq` and
/// `commit_chain` both — cannot be reached by replaying forward: `skipped`
/// (`regressed`), and the entry stays cached.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_of_a_cut_behind_the_entry_is_skipped_regressed() {
    let fx = Fx::new("audit-behind").await;
    let (rolled_back_seq, rolled_back_chain) = fx.header_cut().await;
    fx.commit_fast("rolled-back", 1).await;
    let entry = fx.entry().await;
    let _: () = fx
        .tv
        .client()
        .pool
        .hset(
            fx.header_key(),
            [
                (SEQ_COLUMN, rolled_back_seq.to_string()),
                (HEADER_FIELD_COMMIT_CHAIN, rolled_back_chain),
            ],
        )
        .await
        .expect("HSET obj:hdr");

    let outcome = fx.run(&fx.audit(false), &entry).await;

    assert_eq!(
        outcome,
        AuditOutcome::Skipped(AuditSkip::Unreachable(ReloadReason::Regressed))
    );
    assert_eq!(fx.audits(), (0, 0, 1));
    assert_eq!(
        fx.cached().map(|e| e.seq),
        Some(entry.seq),
        "a skipped audit drops nothing"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. Diverged
// ═════════════════════════════════════════════════════════════════════════════

/// The out-of-band changes the audit must catch, with the key it must name.
fn out_of_band_changes() -> Vec<(OutOfBandChange, &'static str)> {
    vec![
        (
            OutOfBandChange::SectionName {
                section_id: WORLD,
                name: "Renamed out of band".to_owned(),
            },
            "section 1",
        ),
        (
            OutOfBandChange::Children {
                parent_id: ROOT,
                children: vec![SIBLING, WORLD],
            },
            "children list of 0",
        ),
        (
            // The list under a parent naming no section: absent from the
            // state either way, so only the dangling parents differ.
            OutOfBandChange::Children {
                parent_id: DANGLING_PARENT,
                children: Vec::new(),
            },
            "dangling parent 77",
        ),
    ]
}

/// After each out-of-band change the audit answers `diverged` at the entry's
/// cut, naming the changed key; it drops the entry, ticks only `diverged`,
/// and writes nothing to the store.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_after_an_out_of_band_rewrite_is_diverged_drops_the_entry_and_writes_nothing() {
    for (change, key) in out_of_band_changes() {
        let fx = Fx::new("audit-diverged").await;
        let entry = fx.entry().await;
        rewrite_out_of_band(&fx.tv, fx.oid, change.clone()).await;
        let before = dump_keyspace(&fx.tv).await;
        assert!(
            before.contains_key(&fx.header_key()),
            "fixture: the dump holds the object, or its equality is vacuous"
        );

        let outcome = fx.run(&fx.audit(false), &entry).await;

        assert_eq!(
            outcome,
            AuditOutcome::Diverged {
                seq: entry.seq,
                first_difference: key.to_owned(),
            },
            "{change:?}"
        );
        assert!(fx.cached().is_none(), "{change:?}: the entry is dropped");
        assert_eq!(fx.audits(), (0, 1, 0), "{change:?}");
        assert_eq!(
            dump_keyspace(&fx.tv).await,
            before,
            "{change:?}: the audit writes nothing to the store"
        );
    }
}

/// An out-of-band rewrite of a value row's external hash — `color` on
/// `WORLD` turned into an external row of a hash no row held — changes what
/// the compact entry holds of the row: the audit answers `diverged`, naming
/// the field, drops the entry and writes nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_after_an_out_of_band_hash_rewrite_is_diverged_naming_the_field() {
    let fx = Fx::new("audit-diverged-hash").await;
    let entry = fx.entry().await;
    let row = fx
        .tv
        .key_schema()
        .obj_field(&fx.oid.to_string(), &WORLD.to_string(), "color");
    let _: i64 = fx
        .tv
        .client()
        .pool
        .hdel(&row, "value")
        .await
        .expect("HDEL the row's inline value");
    let _: () = fx
        .tv
        .client()
        .pool
        .hset(&row, [("hash", "ab".repeat(32)), ("size", "64".to_owned())])
        .await
        .expect("HSET the row's external hash");
    let before = dump_keyspace(&fx.tv).await;

    let outcome = fx.run(&fx.audit(false), &entry).await;

    assert_eq!(
        outcome,
        AuditOutcome::Diverged {
            seq: entry.seq,
            first_difference: "field \"color\" of section 1".to_owned(),
        }
    );
    assert!(fx.cached().is_none(), "the entry is dropped");
    assert_eq!(fx.audits(), (0, 1, 0));
    assert_eq!(
        dump_keyspace(&fx.tv).await,
        before,
        "the audit writes nothing to the store"
    );
}

/// What the compact entry does not hold is not a divergence (b.mr6): an
/// inline value rewritten out of band leaves the row inline, which is all
/// the entry holds of it, so the audit answers `match` and keeps the entry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_after_an_out_of_band_inline_value_rewrite_is_a_match() {
    let fx = Fx::new("audit-inline-match").await;
    let entry = fx.entry().await;
    rewrite_out_of_band(
        &fx.tv,
        fx.oid,
        OutOfBandChange::InlineFieldValue {
            section_id: WORLD,
            key_name: "color".to_owned(),
            value: vec![0x09, 0x09],
        },
    )
    .await;

    let outcome = fx.run(&fx.audit(false), &entry).await;

    assert_eq!(outcome, AuditOutcome::Match);
    assert_eq!(fx.audits(), (1, 0, 0));
    assert_eq!(
        fx.cached().map(|e| e.seq),
        Some(entry.seq),
        "a matching audit drops nothing"
    );
}

/// SR-3.4 through the audit: a checkout taken before the audit — the
/// committed call's successor, about to be put back — has its put-back
/// refused once the audit answers `diverged`, and the object stays uncached.
/// The control: a checkout held across a `match` puts back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_diverged_refuses_the_put_back_of_a_checkout_taken_before_it() {
    let fx = Fx::new("audit-put-back").await;
    fx.entry().await;
    let audit = fx.audit(false);

    let held = fx.read_ready().await;
    let matched = fx.run(&audit, held.entry.entry()).await;
    let accepted = held.entry.put_back_held().is_stored();
    rewrite_out_of_band(&fx.tv, fx.oid, out_of_band_changes().remove(0).0).await;
    let held = fx.read_ready().await;
    let diverged = fx.run(&audit, held.entry.entry()).await;
    let refused = held.entry.put_back_held() == PutBack::Invalidated;

    assert_eq!(matched, AuditOutcome::Match, "control");
    assert!(accepted, "control: a put-back across a match is accepted");
    assert!(
        matches!(diverged, AuditOutcome::Diverged { .. }),
        "got {diverged:?}"
    );
    assert!(
        refused,
        "a put-back by a checkout taken before the diverged audit is refused"
    );
    assert!(fx.cached().is_none(), "the object stays uncached");
}

/// Every key under the test's prefix, `DUMP`ed: type-agnostic and
/// byte-exact, so one before/after equality covers "no key changed", "no key
/// was created or deleted" and "no stream entry was appended" — for the
/// object's keys and every other key an audit could touch.
async fn dump_keyspace(tv: &TestValkey) -> BTreeMap<String, Vec<u8>> {
    let client = tv.client().pool.next();
    let pattern = format!("{}*", tv.prefix());
    let mut scan: Pin<Box<dyn Stream<Item = Result<Key, fred::error::Error>> + Send>> =
        if client.is_clustered() {
            Box::pin(client.scan_cluster_buffered(pattern, Some(1_000), None::<ScanType>))
        } else {
            Box::pin(client.scan_buffered(pattern, Some(1_000), None::<ScanType>))
        };
    let mut keys = Vec::new();
    while let Some(key) = scan.next().await {
        let key = key.expect("SCAN must succeed");
        keys.push(key.as_str().expect("test keys are UTF-8").to_owned());
    }
    let mut dumped = BTreeMap::new();
    for key in keys {
        let bytes: Option<Vec<u8>> = tv
            .client()
            .pool
            .dump(&key)
            .await
            .expect("DUMP must succeed");
        let bytes = bytes.unwrap_or_else(|| panic!("{key} vanished between SCAN and DUMP"));
        dumped.insert(key, bytes);
    }
    dumped
}

/// Remove every key under the test's prefix: the WARN test's runtime is
/// current-thread, where `TestValkey`'s `Drop` skips its cleanup.
async fn unlink_keyspace(tv: &TestValkey) {
    for key in dump_keyspace(tv).await.into_keys() {
        let _: i64 = tv.client().pool.del(&key).await.expect("DEL must succeed");
    }
}

/// A `MakeWriter` into a shared buffer.
struct SharedBuf(Arc<StdMutex<Vec<u8>>>);

impl std::io::Write for SharedBuf {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The `diverged` WARN is the operator's pointer to the object and key
/// (SR-3.6): exactly one, from the audit, carrying `object_id`, the cut's
/// `seq` and `first_difference`; a `match` logs none.
///
/// The audit logs from its spawned task, which inherits no dispatcher, so the
/// capture is a thread-local default on a current-thread runtime, where the
/// task is polled on this thread (test-writing guide).
#[tokio::test(flavor = "current_thread")]
async fn audit_diverged_logs_one_warn_naming_the_object_cut_and_key() {
    let fx = Fx::new("audit-warn").await;
    let entry = fx.entry().await;
    let buf = Arc::new(StdMutex::new(Vec::<u8>::new()));
    let sink = Arc::clone(&buf);
    let _capture = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .json()
            .with_max_level(tracing::Level::WARN)
            .with_writer(move || SharedBuf(Arc::clone(&sink)))
            .finish(),
    );
    let audit = fx.audit(false);
    let events = || -> Vec<serde_json::Value> {
        String::from_utf8(buf.lock().unwrap().clone())
            .expect("JSON log is UTF-8")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("log line {l:?}: {e}")))
            .filter(|e: &serde_json::Value| {
                e["target"]
                    .as_str()
                    .is_some_and(|t| t.ends_with("structural_cache_audit"))
            })
            .collect()
    };

    let matched = fx.run(&audit, &entry).await;
    let after_match = events();
    rewrite_out_of_band(&fx.tv, fx.oid, out_of_band_changes().remove(0).0).await;
    let diverged = fx.run(&audit, &entry).await;
    let logged = events();
    unlink_keyspace(&fx.tv).await;

    assert_eq!(matched, AuditOutcome::Match, "fixture");
    assert!(
        after_match.is_empty(),
        "a match logs nothing: {after_match:?}"
    );
    assert!(
        matches!(diverged, AuditOutcome::Diverged { .. }),
        "fixture: got {diverged:?}"
    );
    assert_eq!(logged.len(), 1, "one WARN per diverged audit: {logged:?}");
    let warn = &logged[0];
    assert_eq!(warn["level"], "WARN");
    assert_eq!(warn["fields"]["object_id"], fx.oid);
    assert_eq!(warn["fields"]["seq"], entry.seq);
    assert_eq!(warn["fields"]["first_difference"], "section 1");
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. Skipped
// ═════════════════════════════════════════════════════════════════════════════

/// Each way an audit cannot compare counts `skipped` once and nothing else.
/// `too_far` under an injected budget of two entries, three commits ahead;
/// `gap` on a stream trimmed past the entry; an unmarked object; a superseded
/// header; and an object deleted through another store (so this cache was
/// not invalidated by the retirement).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_that_cannot_compare_is_skipped_with_its_reason() {
    for case in ["too_far", "gap", "unmarked", "superseded", "deleted"] {
        let fx = Fx::new("audit-skipped").await;
        let entry = fx.entry().await;
        let mut audit = fx.audit(false);
        let expected = match case {
            "too_far" => {
                fx.commit_fast("ahead", 3).await;
                audit = audit.with_repair_bounds(RepairBounds {
                    max_total: 2,
                    ..RepairBounds::PRODUCTION
                });
                AuditSkip::Unreachable(ReloadReason::TooFar)
            }
            "gap" => {
                fx.commit_fast("ahead", 3).await;
                let _: () = fx
                    .tv
                    .client()
                    .pool
                    .xtrim(
                        fx.tv.key_schema().layer_stream(fx.oid),
                        XCap::try_from(("MAXLEN", "=", 1_i64)).expect("MAXLEN cap"),
                    )
                    .await
                    .expect("XTRIM must succeed");
                AuditSkip::Unreachable(ReloadReason::Gap)
            }
            "unmarked" => {
                unmark(&fx.tv, fx.oid).await;
                AuditSkip::NoChsComplete
            }
            "superseded" => {
                let _: () = fx
                    .tv
                    .client()
                    .pool
                    .hset(fx.header_key(), [(SUPERSEDED_COLUMN, "1")])
                    .await
                    .expect("HSET obj:hdr superseded");
                AuditSkip::Gone
            }
            _ => {
                layer_store(&fx.tv)
                    .delete_layer(&fx.session_id, &fx.layer, mock_ext_store())
                    .await
                    .expect("delete_layer must succeed");
                AuditSkip::Gone
            }
        };

        let outcome = fx.run(&audit, &entry).await;

        assert_eq!(outcome, AuditOutcome::Skipped(expected), "{case}");
        assert_eq!(fx.audits(), (0, 0, 1), "{case}: one skipped tick only");
    }
}

/// A fast-path record setting `WORLD`'s field `key` to `len` inline bytes.
fn inline_record(key: &str, len: usize, set_order: u64) -> OwnedDelta {
    delta_with_set_field_inline(WORLD, key, vec![0x6B; len], set_order)
}

/// The object's stream entries after `after` through `through` (the shared
/// [`stream_entries`]), each as `(its size as a repair read's byte bound
/// counts it, its OMPF's delta part in bytes)`: every audited commit relays
/// an OMPF.
async fn relayed_entries(fx: &Fx, after: u64, through: u64) -> Vec<(u64, u64)> {
    stream_entries(&fx.tv, fx.oid, after, through)
        .await
        .into_iter()
        .map(|(size, delta)| (size, delta.expect("a relayed commit's OMPF parses")))
        .collect()
}

/// An upload gate at the production defaults (`B` = 384 MiB) and the
/// sampled audit's charges on it, as `main.rs` wires them
/// (`UploadBudgetAuditCharges`).
fn production_audit_charges() -> (Arc<UploadAdmissionGate>, Arc<dyn AuditChargeSource>) {
    let gate = Arc::new(UploadAdmissionGate::new(
        UploadAdmissionConfig::default(),
        None,
    ));
    let charges: Arc<dyn AuditChargeSource> =
        Arc::new(UploadBudgetAuditCharges::new(Arc::clone(&gate)));
    (gate, charges)
}

/// The target a charge on the upload budget holds for `need` on the audit's
/// copy of weight `base`: its production price less the copy's own weight,
/// which the line's audit term funds (`UploadBudgetAuditCharges`).
fn net_target(need: &ObjectStateNeed, base: u64) -> u64 {
    object_state_charge_for_test(need).saturating_sub(base)
}

/// What the audit's first repair read holds on the upload budget at its RP0
/// under the production bounds, on a copy of weight `base` with nothing
/// replayed: the read's reply at the first read's entry bound
/// (`STRUCTURAL_REPAIR_INITIAL_ENTRIES`) and byte bound
/// (`STRUCTURAL_REPAIR_INITIAL_BYTES`), net of `base`.
fn first_read_rp0(base: u64) -> u64 {
    net_target(
        &ObjectStateNeed::cached_repair(
            base,
            RepairTransient {
                reply_entry_count: STRUCTURAL_REPAIR_INITIAL_ENTRIES
                    .min(STRUCTURAL_REPAIR_MAX_ENTRIES),
                reply_bytes: STRUCTURAL_REPAIR_INITIAL_BYTES,
                ..RepairTransient::default()
            },
        ),
        base,
    )
}

/// The one audit observed by an [`Fx::audit_charged`] observer: the most
/// its charge held at once.
fn only_peak(peaks: &StdMutex<Vec<u64>>) -> u64 {
    let peaks = peaks.lock().unwrap();
    assert_eq!(peaks.len(), 1, "one audit observed: {peaks:?}");
    peaks[0]
}

/// The audit repairs under the production repair bounds, holding its repair
/// on the upload budget: the 64 KiB read bound it had while its repair held
/// no charge is retired (PD-A). A pending entry past that bound — which it
/// used to skip `too_large` — is a `match`, its repair held on the budget
/// while it ran (its charge's peak at least the chunk's RP1, net of the
/// copy's own weight, and within the per-audit cap), all of it given back
/// at the end and none of it ever on the overflow slot.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_repair_of_a_pending_entry_past_the_retired_64_kib_read_bound_is_match_holding_its_repair_on_the_upload_budget(
) {
    /// The read bound the audit no longer has.
    const RETIRED_READ_BOUND: u64 = 64 * 1024;
    let fx = Fx::new("audit-read-bound").await;
    let entry = fx.entry().await;
    let (before, _) = fx.header_cut().await;
    fx.commit(inline_record("bound", RETIRED_READ_BOUND as usize, 61))
        .await;
    let [(bytes, delta)] = relayed_entries(&fx, before, before + 1).await[..] else {
        panic!("fixture: the pending record's one entry");
    };
    assert!(
        RETIRED_READ_BOUND < bytes && bytes <= STRUCTURAL_REPAIR_INITIAL_BYTES,
        "fixture: the pending entry ({bytes} B) is past the retired 64 KiB bound and within the \
         first read's own bound"
    );
    let (gate, charges) = production_audit_charges();
    let (audit, peaks) = fx.audit_charged(charges);

    let outcome = fx.run(&audit, &entry).await;

    assert_eq!(
        outcome,
        AuditOutcome::Match,
        "a pending entry of {bytes} B, past the retired 64 KiB read bound"
    );
    let rp1 = net_target(
        &ObjectStateNeed::cached_repair(
            entry.state_weight_bytes,
            RepairTransient {
                reply_entry_count: 1,
                reply_bytes: bytes,
                parse_delta_bytes: delta,
                replay_delta_bytes: delta,
            },
        ),
        entry.state_weight_bytes,
    );
    let peak = only_peak(&peaks);
    assert!(
        rp1 <= peak && peak <= SAMPLED_AUDIT_CHARGE_CAP_BYTES,
        "the repair held at least its chunk's RP1 ({rp1} B) on the upload budget, within the cap: \
         peak {peak} B"
    );
    assert_eq!(
        (
            gate.reserved_bytes(),
            gate.overflow_bytes(),
            gate.overflow_high_water_bytes()
        ),
        (0, 0, 0),
        "given back at the end, and never on the overflow slot"
    );
    assert_eq!(
        fx.cached().map(|e| e.seq),
        Some(entry.seq),
        "the audit drops nothing"
    );
    assert_eq!(fx.audits(), (1, 0, 0));
}

/// The audit's repair replays any amount of delta parts: the 64 KiB cap on
/// them it had while its repair held no charge is retired (PD-A), and its
/// copy's growth is held on the upload budget record by record instead. Five
/// pending records whose delta parts sum past 64 KiB — which it used to skip
/// `too_far` — are a `match`, the budget given back after. The repair's own
/// entry budget still binds: a cut one entry past it
/// (`STRUCTURAL_REPAIR_MAX_TOTAL` + 1, under the production bounds) is
/// `skipped` (`too_far`), with nothing replayed and the budget given back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_repair_whose_replay_crosses_the_retired_64_kib_cap_is_match_and_one_past_the_entry_budget_is_skipped_too_far(
) {
    /// The replayed-bytes cap the audit no longer has.
    const RETIRED_REPLAY_CAP: u64 = 64 * 1024;
    const RECORD_LEN: usize = 14 * 1024;
    const RECORDS: usize = 5;
    let fx = Fx::new("audit-replay-cap").await;
    let entry = fx.entry().await;
    let (before, _) = fx.header_cut().await;
    for i in 0..RECORDS {
        fx.commit(inline_record(&format!("r{i}"), RECORD_LEN, 70 + i as u64))
            .await;
    }
    let pending = relayed_entries(&fx, before, before + RECORDS as u64).await;
    let delta: u64 = pending.iter().map(|(_, delta)| delta).sum();
    assert!(
        pending.len() == RECORDS && delta > RETIRED_REPLAY_CAP,
        "fixture: {RECORDS} pending records whose delta parts ({delta} B) sum past the retired \
         64 KiB cap"
    );
    let (gate, charges) = production_audit_charges();
    let (audit, peaks) = fx.audit_charged(charges);

    let crossing = fx.run(&audit, &entry).await;

    assert_eq!(
        crossing,
        AuditOutcome::Match,
        "{delta} B of delta parts replayed, past the retired cap"
    );
    assert!(
        only_peak(&peaks) > 0,
        "the repair held its replay on the upload budget"
    );
    assert_eq!(
        (gate.reserved_bytes(), gate.overflow_high_water_bytes()),
        (0, 0),
        "the replay's budget given back, and never on the slot"
    );

    let fx = Fx::new("audit-entry-budget").await;
    let entry = fx.entry().await;
    fx.commit_fast("ahead", STRUCTURAL_REPAIR_MAX_TOTAL + 1)
        .await;
    let (gate, charges) = production_audit_charges();
    let (audit, _) = fx.audit_charged(charges);

    let past_the_budget = fx.run(&audit, &entry).await;

    assert_eq!(
        past_the_budget,
        AuditOutcome::Skipped(AuditSkip::Unreachable(ReloadReason::TooFar)),
        "a cut {} entries ahead, one past the repair's budget",
        STRUCTURAL_REPAIR_MAX_TOTAL + 1
    );
    assert_eq!(gate.reserved_bytes(), 0, "the budget given back");
    assert_eq!(fx.audits(), (0, 0, 1));
}

/// A store error on either of the audit's reads is `skipped` (`ReadFailed`):
/// its snapshot read (a frame naming the header key), or — with that read
/// answered — the repair's read (the only frame naming the stream key). The
/// proxy answers the matched script with an error reply in its place.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_with_a_store_error_on_either_read_is_skipped_read_failed() {
    for (read, snapshot_reads_ok) in [("snapshot", 0), ("repair", 1)] {
        let fx = Fx::new("audit-read-failed").await;
        let entry = fx.entry().await;
        fx.commit_fast("behind", 1).await;
        let needle = match read {
            "snapshot" => fx.header_key(),
            _ => fx.tv.key_schema().layer_stream(fx.oid),
        };
        let proxy = ScriptReplyProxy::start(
            test_valkey_addr(),
            script_frames_naming(
                needle,
                ProxyAction::Eval(error_reply_script("ERR injected audit read failure")),
            ),
        )
        .await;
        let client = client_through(&proxy).await;
        client
            .install_observability_hooks(Arc::clone(&fx.metrics))
            .await;

        let outcome = fx.run_on(&fx.audit(false), &client, &entry).await;

        assert_eq!(
            outcome,
            AuditOutcome::Skipped(AuditSkip::ReadFailed),
            "an error on the {read} read"
        );
        assert!(proxy.hits() >= 1, "{read}: fixture: the error was injected");
        assert_eq!(
            fx.audit_reads("ok"),
            snapshot_reads_ok,
            "{read}: the snapshot read answered only when the repair's read failed"
        );
        assert_eq!(fx.audits(), (0, 0, 1), "{read}: one skipped tick only");
    }
}

/// The object deleted after the audit's read and before its repair: the read
/// answered with a cut (its reply held at a proxy while another store deletes
/// the layer), and the repair's header check finds the object gone —
/// `skipped` (`Gone`), not a comparison with a cut the store no longer has.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_of_an_object_deleted_between_its_read_and_its_repair_is_skipped_gone() {
    let fx = Fx::new("audit-gone-mid").await;
    let entry = fx.entry().await;
    let gate = ReplyGate::new();
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        script_frames_naming(fx.header_key(), ProxyAction::HoldReplies(gate.clone())),
    )
    .await;
    let client = client_through(&proxy).await;
    client
        .install_observability_hooks(Arc::clone(&fx.metrics))
        .await;
    let handle = fx
        .audit(false)
        .maybe_spawn(&client, fx.tv.key_schema(), &fx.cache, &entry)
        .expect("an audit at rate 1.0 starts");

    gate.reply_held().await;
    layer_store(&fx.tv)
        .delete_layer(&fx.session_id, &fx.layer, mock_ext_store())
        .await
        .expect("delete_layer must succeed");
    gate.open();
    let outcome = finished(handle).await;

    assert_eq!(outcome, Some(AuditOutcome::Skipped(AuditSkip::Gone)));
    assert_eq!(
        fx.audit_reads("ok"),
        1,
        "fixture: the read answered before the delete"
    );
    assert_eq!(fx.audits(), (0, 0, 1));
}

// ═════════════════════════════════════════════════════════════════════════════
// 3a. The repair's charge on the upload budget
// ═════════════════════════════════════════════════════════════════════════════
//
// The audit holds its repair of the entry's copy on the upload budget `B`
// through its own charge (`storage::commit_charge::AuditCharge`; on the
// replica `grpc::upload_admission::UploadBudgetAuditCharges`): each read,
// round and record priced as a structural call's repair is, less the copy's
// own weight; never waiting, never on the overflow slot, refused while an
// upload is queued or when it would leave less than one admission's room,
// and capped per audit. The gate-side rules are unit-tested in
// `grpc::upload_admission`'s tests; these drive them through real audits.

/// A section id [`commit_field_write_then_creation`] creates: clear of the
/// seeded tree's.
const AUDIT_NEW_SECTION: u64 = 900;

/// The `UpdateLayer` budget the queue tests' gate holds: room for three
/// admissions (`F + A` each) at once and not four.
const QUEUE_BUDGET: u64 = 32 * 1024 * 1024;

/// One need an audit's repair priced, and its target net of the copy's
/// weight ([`RecordingAuditCharges`]).
type PricedNeed = (ObjectStateNeed, u64);

/// An [`AuditChargeSource`] whose charges price every need as the upload
/// budget's do — production pricing less the copy's weight ([`net_target`])
/// — grant every growth, and record each need priced with its target, in
/// order: what a real charge would ask for, with no budget to refuse it.
#[derive(Default, Clone)]
struct RecordingAuditCharges {
    priced: Arc<StdMutex<Vec<PricedNeed>>>,
}

impl RecordingAuditCharges {
    fn priced(&self) -> Vec<PricedNeed> {
        self.priced.lock().unwrap().clone()
    }
}

impl AuditChargeSource for RecordingAuditCharges {
    fn open(&self, base: u64) -> Box<dyn AuditCharge> {
        Box::new(RecordingAuditCharge {
            base,
            priced: Arc::clone(&self.priced),
            held: AtomicU64::new(0),
            peak: AtomicU64::new(0),
        })
    }
}

/// [`RecordingAuditCharges`]' charge.
struct RecordingAuditCharge {
    base: u64,
    priced: Arc<StdMutex<Vec<PricedNeed>>>,
    held: AtomicU64,
    peak: AtomicU64,
}

impl SharedCommitCharge for RecordingAuditCharge {
    fn held(&self) -> u64 {
        self.held.load(Ordering::SeqCst)
    }

    fn target(&self, need: &ObjectStateNeed) -> u64 {
        let target = net_target(need, self.base);
        self.priced.lock().unwrap().push((*need, target));
        target
    }

    fn rewrite_allowance(&self, need: &ObjectStateNeed) -> MutationAllowance {
        object_state_rewrite_allowance_for_test(self.held().saturating_add(self.base), need)
    }

    fn try_hold(&self, target: u64) -> bool {
        self.held.store(target, Ordering::SeqCst);
        self.peak.fetch_max(target, Ordering::SeqCst);
        true
    }

    fn try_hold_at_least(&self, target: u64) -> bool {
        if target > self.held() {
            self.try_hold(target);
        }
        true
    }
}

impl AuditCharge for RecordingAuditCharge {
    fn cap(&self) -> u64 {
        u64::MAX
    }

    fn peak(&self) -> u64 {
        self.peak.load(Ordering::SeqCst)
    }
}

/// An upload gate of `budget` bytes (no size cap of its own, so any budget
/// from one admission up is valid) and the audit charges on it.
fn audit_gate(budget: u64) -> (Arc<UploadAdmissionGate>, Arc<dyn AuditChargeSource>) {
    let gate = Arc::new(UploadAdmissionGate::new(
        UploadAdmissionConfig {
            budget_bytes: budget,
            wait: AUDIT_BUDGET,
            max_waiters: 16,
            max_upload_bytes: 0,
        },
        None,
    ));
    let charges: Arc<dyn AuditChargeSource> =
        Arc::new(UploadBudgetAuditCharges::new(Arc::clone(&gate)));
    (gate, charges)
}

/// An `UpdateLayer` admission on `gate` for `session`, granted at once.
async fn admitted(
    gate: &Arc<UploadAdmissionGate>,
    session: &str,
    shutdown: &CancellationToken,
) -> UploadReservation {
    tokio::time::timeout(
        AUDIT_BUDGET,
        gate.admit(UploadRpc::UpdateLayer, session, shutdown),
    )
    .await
    .unwrap_or_else(|_| panic!("{session}: admitted at once"))
    .unwrap_or_else(|e| panic!("{session}: admitted; got {e:?}"))
}

/// Commit two records behind the audited entry, which one read takes
/// together: a field write on `WORLD`, then the creation of
/// [`AUDIT_NEW_SECTION`] under the root, named so its delta part is the
/// larger. So the second record's RP2 holds its round's RP1 at least — the
/// same reply and replayed delta parts, its own parse at the largest — plus
/// what it grows the copy by, which a creation always prices (its parent's
/// list, at its length on the copy, reallocates). Returns each pending
/// entry's (size, delta part).
async fn commit_field_write_then_creation(fx: &Fx) -> Vec<(u64, u64)> {
    let (before, _) = fx.header_cut().await;
    fx.commit(inline_record("before-the-creation", 8, 80)).await;
    fx.commit(structural(vec![create_cmd(
        ROOT,
        AUDIT_NEW_SECTION,
        &"n".repeat(256),
    )]))
    .await;
    let pending = relayed_entries(fx, before, before + 2).await;
    assert!(
        pending.len() == 2 && pending[1].1 >= pending[0].1,
        "fixture: two pending records, the creation's delta part the larger: {pending:?}"
    );
    pending
}

/// Repair bounds whose first read takes exactly `pending` in one chunk, and
/// prices that read at its size: its entry bound the records' count and its
/// byte bound their bytes. So the read's RP0 holds no more than the chunk's
/// RP1, and a record's RP2 can be the repair's largest point.
fn one_chunk_bounds(pending: &[(u64, u64)]) -> RepairBounds {
    RepairBounds {
        initial_entries: pending.len() as u64,
        initial_bytes: pending.iter().map(|(size, _)| size).sum(),
        ..RepairBounds::PRODUCTION
    }
}

/// The cached entry's cut and state, for an unchanged-entry check.
fn cut_of(entry: &CacheEntry) -> (u64, String, live_edit::domain::types::CompactState) {
    (entry.seq, entry.commit_chain.clone(), entry.state.clone())
}

/// The audit's charge refused is a `skipped` (`ChargeRefused`) audit that
/// gives back everything it held and never touches the overflow slot or a
/// queue, and leaves the cache's entry as it was: refused at its first
/// charge point (RP0) on a budget an upload fills, and refused between the
/// two records of one chunk (the second record's RP2) on a budget whose
/// room, less the one admission the audit always leaves free, holds every
/// point before it and not that one. The targets come from a first audit of
/// the same cut under a granting charge that records them
/// ([`RecordingAuditCharges`]), whose premise is checked: an RP0, an RP1 and
/// an RP2 per record, the second RP2 above every point before it.
///
/// Fails if the hold were granted regardless of the budget (`match`), or
/// went through an upload's processing step, which takes the free overflow
/// slot (`overflow_high_water_bytes > 0`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_whose_repair_charge_is_refused_is_skipped_and_gives_everything_back() {
    for case in ["at its first read", "between two records"] {
        let fx = Fx::new("audit-charge-refused").await;
        let entry = fx.entry().await;
        let pending = commit_field_write_then_creation(&fx).await;
        let bounds = one_chunk_bounds(&pending);
        let recording = RecordingAuditCharges::default();
        let (audit, _) = fx.audit_charged(Arc::new(recording.clone()));
        assert_eq!(
            fx.run(&audit.with_repair_bounds(bounds), &entry).await,
            AuditOutcome::Match,
            "{case}: fixture: granted, the audit matches"
        );
        let targets: Vec<u64> = recording.priced().iter().map(|(_, t)| *t).collect();
        let (&last, earlier) = targets.split_last().expect("fixture: the repair priced");
        let before_last = earlier.iter().copied().max().unwrap_or(0);
        assert!(
            targets.len() == 2 + pending.len() && last > before_last,
            "{case}: fixture: RP0, RP1 and an RP2 per record, the last RP2 above every point \
             before it: {targets:?}"
        );

        let shutdown = CancellationToken::new();
        let (gate, charges, filler) = if case == "at its first read" {
            let (gate, charges) = audit_gate(QUEUE_BUDGET);
            let mut filler = admitted(&gate, "filler", &shutdown).await;
            filler
                .try_grow_in_budget(QUEUE_BUDGET)
                .expect("the filler fills the budget");
            (gate, charges, Some(filler))
        } else {
            let (gate, charges) = audit_gate(admission_charge() + before_last);
            (gate, charges, None)
        };
        let uploads = gate.reserved_bytes();
        let cached = cut_of(&fx.cached().expect("fixture: the object is cached"));
        let (audit, peaks) = fx.audit_charged(charges);

        let outcome = fx.run(&audit.with_repair_bounds(bounds), &entry).await;

        assert_eq!(
            outcome,
            AuditOutcome::Skipped(AuditSkip::ChargeRefused),
            "{case}"
        );
        let held = if filler.is_some() { 0 } else { before_last };
        assert_eq!(
            only_peak(&peaks),
            held,
            "{case}: it held every point before the refused one, and no more"
        );
        assert_eq!(
            (
                gate.reserved_bytes(),
                gate.overflow_bytes(),
                gate.overflow_high_water_bytes()
            ),
            (uploads, 0, 0),
            "{case}: the budget is back to what the uploads hold, and the slot was never taken"
        );
        assert_eq!(
            (gate.waiters(), gate.processing_waiters()),
            (0, 0),
            "{case}: nothing queued"
        );
        assert_eq!(
            fx.cached().as_ref().map(cut_of),
            Some(cached),
            "{case}: the cache's entry is left as it was"
        );
        assert_eq!(
            fx.audits(),
            (1, 0, 1),
            "{case}: the recording audit's match and this skip"
        );
        drop(filler);
    }
}

/// A repair needing more of the upload budget than one audit may hold
/// (`SAMPLED_AUDIT_CHARGE_CAP_BYTES`, one upload admission) is `skipped`
/// (`ChargeOverCap`) on a budget with room for it: a pending record whose
/// stream entry is about 3 MiB (an external value it relays), whose first
/// read (256 KiB) answers `TOO_LARGE`, and whose read at its size prices
/// over the cap. The charge never held more than the cap —
/// its peak is the first read's RP0, net of the copy's weight — and gave it
/// back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_whose_repair_needs_more_than_its_cap_is_skipped_over_cap_holding_at_most_the_cap() {
    let fx = Fx::new("audit-over-cap").await;
    let entry = fx.entry().await;
    let (before, _) = fx.header_cut().await;
    // A 3 MiB external value: the stream entry relays it in the OMPF, while
    // the object holds only its hash, so the audit's read stays within the
    // plain ceilings.
    let blob = vec![0x3C; 3 * 1024 * 1024];
    commit_to(
        &fx.ls,
        &fx.session_id,
        &fx.layer,
        fx.oid,
        WRITER,
        Some(delta_with_set_field_external(
            WORLD,
            "over-cap",
            ext_hash(0x3C).to_vec(),
            blob.len() as u64,
            1,
            63,
        )),
        &[&blob],
        CLIENT_TS,
    )
    .await;
    let [(bytes, _)] = relayed_entries(&fx, before, before + 1).await[..] else {
        panic!("fixture: the pending record's one entry");
    };
    let base = entry.state_weight_bytes;
    let first_read = first_read_rp0(base);
    let at_its_size = net_target(
        &ObjectStateNeed::cached_repair(
            base,
            RepairTransient {
                reply_entry_count: STRUCTURAL_REPAIR_MAX_ENTRIES,
                reply_bytes: bytes,
                ..RepairTransient::default()
            },
        ),
        base,
    );
    let (gate, charges) = production_audit_charges();
    assert!(
        STRUCTURAL_REPAIR_INITIAL_BYTES < bytes
            && first_read <= SAMPLED_AUDIT_CHARGE_CAP_BYTES
            && at_its_size > SAMPLED_AUDIT_CHARGE_CAP_BYTES
            && admission_charge() + at_its_size <= gate.budget_bytes(),
        "fixture: the entry ({bytes} B) is over the first read's bound, which prices within the \
         cap ({first_read} B), its read at its size over it ({at_its_size} B), and the budget has \
         room for that"
    );
    let (audit, peaks) = fx.audit_charged(charges);

    let outcome = fx.run(&audit, &entry).await;

    assert_eq!(outcome, AuditOutcome::Skipped(AuditSkip::ChargeOverCap));
    assert_eq!(
        only_peak(&peaks),
        first_read,
        "the charge held the first read's RP0 and never more: the over-cap read was refused"
    );
    assert_eq!(
        (gate.reserved_bytes(), gate.overflow_high_water_bytes()),
        (0, 0),
        "given back, and never on the slot"
    );
    assert_eq!(fx.audits(), (0, 0, 1));
}

/// The pending stream entry at `seq` of the fixture's object: its OMPF.
async fn pending_ompf(fx: &Fx, seq: u64) -> Vec<u8> {
    let entries: Vec<(String, HashMap<String, bytes::Bytes>)> = fx
        .tv
        .client()
        .pool
        .xrange_values(
            fx.tv.key_schema().layer_stream(fx.oid),
            format!("{seq}-0"),
            format!("{seq}-0"),
            None,
        )
        .await
        .expect("XRANGE must succeed");
    let [(_, fields)] = &entries[..] else {
        panic!("fixture: one stream entry at {seq}: {entries:?}");
    };
    fields
        .get(RELAY_OMPF_FIELD)
        .expect("fixture: the entry relays its OMPF")
        .to_vec()
}

/// Granted, the audit's charge holds each point's need as priced on its
/// copy, which holds no plan indexes. With one structural record pending,
/// read in one entry at its size, its peak is that record's RP2: the copy's
/// stored weight `W0` with the growth, the scratch and the index-less
/// rewrite the per-record estimate prices for the record on the copy
/// (`estimate_replay_record_growth` with no plan indexes — the name and
/// adoption indexes a replay builds per record, and the full depth walk;
/// the entry's dangling parent puts the record through the list-delete
/// sink, whose ids, mutation in flight and `SetChildren` copy are scratch),
/// and the chunk's reply, the record's parse and its growth per delta byte,
/// net of `W0`: computed here from the entry and the record's decoded
/// delta, independently of the repair, and above its RP0 and RP1. The audit
/// is a `match` and the budget is back to nothing after. Diverged the same
/// way (a section renamed out of band), the charge is given back too, before
/// the entry is dropped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_whose_repair_charge_is_granted_peaks_at_its_records_priced_need_and_gives_it_back() {
    for diverged in [false, true] {
        let fx = Fx::new("audit-charge-granted").await;
        let entry = fx.entry().await;
        let (before, _) = fx.header_cut().await;
        fx.commit(structural(vec![create_cmd(
            ROOT,
            AUDIT_NEW_SECTION,
            "Granted",
        )]))
        .await;
        let [(bytes, delta)] = relayed_entries(&fx, before, before + 1).await[..] else {
            panic!("fixture: the pending record's one entry");
        };
        let record = decode_stream_record(
            before,
            Some(&pending_ompf(&fx, before + 1).await),
            before + 1,
        )
        .expect("fixture: the pending record decodes");
        let estimate = estimate_replay_record_growth(
            &entry.state,
            None,
            record.delta(),
            !entry.dangling_parents.is_empty(),
        );
        let base = entry.state_weight_bytes;
        let chunk = RepairTransient {
            reply_entry_count: 1,
            reply_bytes: bytes,
            parse_delta_bytes: delta,
            replay_delta_bytes: delta,
        };
        let rp0 = net_target(
            &ObjectStateNeed::cached_repair(
                base,
                RepairTransient {
                    reply_entry_count: 1,
                    reply_bytes: bytes,
                    ..RepairTransient::default()
                },
            ),
            base,
        );
        let rp1 = net_target(&ObjectStateNeed::cached_repair(base, chunk), base);
        // RP2 holds the record's growth and the scratch its apply frees
        // again (`ReplayRecordGrowth::scratch`: on this entry, which holds a
        // dangling parent, the list-delete sink's ids, its one mutation in
        // flight and the dirty parent's `SetChildren` copy).
        let rp2 = net_target(
            &ObjectStateNeed::cached_repair_record(
                base,
                estimate.growth + estimate.scratch,
                estimate.rewrite,
                chunk,
            ),
            base,
        );
        assert!(
            estimate.growth > 0
                && estimate.scratch > 0
                && estimate.rewrite.indexed_sections > 0
                && estimate.rewrite.adoption_entries > 0
                && rp0 <= rp1
                && rp1 < rp2
                && rp2 <= SAMPLED_AUDIT_CHARGE_CAP_BYTES,
            "fixture: the creation prices growth, scratch and the index-less rewrite on the copy, \
             its RP2 above its RP0 ({rp0} B) and RP1 ({rp1} B) and within the cap: {estimate:?}, \
             RP2 {rp2} B"
        );
        if diverged {
            rewrite_out_of_band(&fx.tv, fx.oid, out_of_band_changes().remove(0).0).await;
        }
        let (gate, charges) = production_audit_charges();
        let (audit, peaks) = fx.audit_charged(charges);
        let bounds = RepairBounds {
            initial_entries: 1,
            initial_bytes: bytes,
            ..RepairBounds::PRODUCTION
        };

        let outcome = fx.run(&audit.with_repair_bounds(bounds), &entry).await;

        if diverged {
            assert!(
                matches!(outcome, AuditOutcome::Diverged { .. }),
                "fixture: got {outcome:?}"
            );
            assert!(fx.cached().is_none(), "the diverged entry is dropped");
        } else {
            assert_eq!(outcome, AuditOutcome::Match);
        }
        assert_eq!(
            only_peak(&peaks),
            rp2,
            "diverged {diverged}: the charge peaked at the record's RP2, priced on the copy"
        );
        assert_eq!(
            (gate.reserved_bytes(), gate.overflow_high_water_bytes()),
            (0, 0),
            "diverged {diverged}: given back, and never on the slot"
        );
    }
}

/// The audit never waits for the upload budget: with the budget full and
/// the overflow slot free, with a processing step queued for the slot, and
/// with an admission queued, each audit ends `skipped` (`ChargeRefused`)
/// within its budget (the run's own [`AUDIT_BUDGET`] bound), the queued
/// waiter still queued — the audit neither waited behind it nor went ahead
/// of it — and the budget back to what the uploads hold. With the processing
/// step queued the budget has room for the audit beside the uploads, so only
/// the queue refuses it. The waiter is then served once the test gives the
/// room back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_never_waits_for_the_upload_budget_and_leaves_every_queued_upload_queued() {
    for case in [
        "the budget full, the slot free",
        "a processing step queued",
        "an admission queued",
    ] {
        let fx = Fx::new("audit-never-waits").await;
        let entry = fx.entry().await;
        fx.commit_fast("behind", 1).await;
        let (gate, charges) = audit_gate(QUEUE_BUDGET);
        let shutdown = CancellationToken::new();
        let mut held = Vec::new();
        let mut waiter: std::pin::Pin<Box<dyn std::future::Future<Output = bool>>> = match case {
            "the budget full, the slot free" => {
                let mut filler = admitted(&gate, "filler", &shutdown).await;
                filler
                    .try_grow_in_budget(QUEUE_BUDGET)
                    .expect("the filler fills the budget");
                held.push(filler);
                Box::pin(async { true })
            }
            "a processing step queued" => {
                let mut holder = admitted(&gate, "holder", &shutdown).await;
                holder.end_receive();
                let grant = holder
                    .reserve_processing(QUEUE_BUDGET + 1, &shutdown)
                    .await
                    .expect("the slot is free");
                assert!(grant.entered_overflow, "fixture: the holder takes the slot");
                held.push(holder);
                let mut queued = admitted(&gate, "queued", &shutdown).await;
                queued.end_receive();
                let token = shutdown.clone();
                Box::pin(async move {
                    queued
                        .reserve_processing(QUEUE_BUDGET + 1, &token)
                        .await
                        .is_ok()
                })
            }
            _ => {
                for session in ["a", "b", "c"] {
                    held.push(admitted(&gate, session, &shutdown).await);
                }
                let (gate, token) = (Arc::clone(&gate), shutdown.clone());
                Box::pin(async move {
                    gate.admit(UploadRpc::UpdateLayer, "d", &token)
                        .await
                        .is_ok()
                })
            }
        };
        let queued = case != "the budget full, the slot free";
        if queued {
            assert!(
                futures::poll!(waiter.as_mut()).is_pending(),
                "{case}: fixture: the waiter queues"
            );
            assert_eq!(
                gate.waiters() + gate.processing_waiters(),
                1,
                "{case}: fixture: one waiter queued"
            );
        }
        let uploads = (gate.reserved_bytes(), gate.overflow_bytes());
        let (audit, _) = fx.audit_charged(charges);

        let outcome = fx.run(&audit, &entry).await;

        assert_eq!(
            outcome,
            AuditOutcome::Skipped(AuditSkip::ChargeRefused),
            "{case}: refused, never waiting"
        );
        assert_eq!(
            (gate.reserved_bytes(), gate.overflow_bytes()),
            uploads,
            "{case}: the budget and the slot are back to what the uploads hold"
        );
        if queued {
            assert!(
                futures::poll!(waiter.as_mut()).is_pending(),
                "{case}: the waiter is still queued"
            );
            assert_eq!(
                gate.waiters() + gate.processing_waiters(),
                1,
                "{case}: still one waiter queued"
            );
            drop(held);
            assert!(
                tokio::time::timeout(AUDIT_BUDGET, waiter)
                    .await
                    .expect("the freed room serves the waiter"),
                "{case}: the waiter is granted"
            );
        }
    }
}

/// Shutdown while the audit's repair read is in flight — its reply held at
/// a proxy, after its first charge point (RP0) took the upload budget — ends
/// the audit with no outcome and no tick, and the charge's drop gives the
/// budget back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_cancelled_during_its_repair_read_gives_its_upload_budget_back() {
    let fx = Fx::new("audit-cancel-held-read").await;
    let entry = fx.entry().await;
    fx.commit_fast("behind", 1).await;
    let reply = ReplyGate::new();
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        script_frames_naming(
            fx.tv.key_schema().layer_stream(fx.oid),
            ProxyAction::HoldReplies(reply.clone()),
        ),
    )
    .await;
    let client = client_through(&proxy).await;
    client
        .install_observability_hooks(Arc::clone(&fx.metrics))
        .await;
    let (gate, charges) = production_audit_charges();
    let shutdown = CancellationToken::new();
    let audit = StructuralCacheAudit::new(1.0, false, charges, shutdown.clone());
    let rp0 = first_read_rp0(entry.state_weight_bytes);

    let handle = audit
        .maybe_spawn(&client, fx.tv.key_schema(), &fx.cache, &entry)
        .expect("the audit starts");
    reply.reply_held().await;
    let mid_read = gate.reserved_bytes();
    shutdown.cancel();
    let outcome = finished(handle).await;
    reply.open();

    assert_eq!(
        mid_read, rp0,
        "fixture: mid-read the audit holds its first read's RP0 on the budget"
    );
    assert_eq!(outcome, None, "a cancelled audit answers no outcome");
    assert_eq!(
        (gate.reserved_bytes(), gate.overflow_high_water_bytes()),
        (0, 0),
        "the cancelled audit's charge gave the budget back"
    );
    assert_eq!(fx.audits(), (0, 0, 0), "a cancelled audit ticks nothing");
}

/// A store error on the repair's read — after its first charge point (RP0)
/// took the upload budget — ends the audit `skipped` (`ReadFailed`), the
/// budget given back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_whose_repair_read_fails_is_skipped_read_failed_and_gives_its_upload_budget_back() {
    let fx = Fx::new("audit-failed-read-budget").await;
    let entry = fx.entry().await;
    fx.commit_fast("behind", 1).await;
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        script_frames_naming(
            fx.tv.key_schema().layer_stream(fx.oid),
            ProxyAction::Eval(error_reply_script("ERR injected audit repair read failure")),
        ),
    )
    .await;
    let client = client_through(&proxy).await;
    client
        .install_observability_hooks(Arc::clone(&fx.metrics))
        .await;
    let (gate, charges) = production_audit_charges();
    let (audit, peaks) = fx.audit_charged(charges);

    let outcome = fx.run_on(&audit, &client, &entry).await;

    assert_eq!(outcome, AuditOutcome::Skipped(AuditSkip::ReadFailed));
    assert!(proxy.hits() >= 1, "fixture: the error was injected");
    assert_eq!(
        only_peak(&peaks),
        first_read_rp0(entry.state_weight_bytes),
        "the repair held its first read's RP0 before the read, and no more"
    );
    assert_eq!(
        (gate.reserved_bytes(), gate.overflow_high_water_bytes()),
        (0, 0),
        "the failed read's charge gave the budget back"
    );
    assert_eq!(fx.audits(), (0, 0, 1));
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. Above the shared ceilings: the fenced read
// ═════════════════════════════════════════════════════════════════════════════

/// Above the shared item ceiling: with the switch off the plain read's
/// `TOO_LARGE` is `skipped` (`AboveCeiling`) and the fenced counter is not
/// ticked; with it on the fenced read compares, `match`. Far above the plain
/// item ceiling — past the item ceiling of the retired raised read (the
/// plain one × `STRUCTURAL_COLD_START_ITEMS_FACTOR`), which skipped such an
/// object — the fenced read compares too. No second atomic read in either.
///
/// Fails if: `read_fresh` answered `AboveCeiling` with the switch on (e.g.
/// `let Some(pacing) = above_ceiling else` inverted), or the raised read's
/// item early-out were restored.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_above_the_plain_ceiling_is_skipped_with_the_switch_off_and_read_fenced_with_it_on() {
    let fx = Fx::new("audit-above").await;
    let entry = fx.entry().await;
    let (client, plain_items) = fx.client_one_item_under_the_bound().await;
    let far_above = client_with_limits(
        plain_items / STRUCTURAL_COLD_START_ITEMS_FACTOR,
        NO_BYTE_LIMIT,
    )
    .await;
    far_above
        .install_observability_hooks(Arc::clone(&fx.metrics))
        .await;

    let off = fx.run_on(&fx.audit(false), &client, &entry).await;
    let reads_off = (
        fx.audit_reads("too_large"),
        fx.audit_reads("ok"),
        fx.fenced_reads("ok"),
    );
    let fenced_off = fx.audit_fenced();
    let on = fx.run_on(&fx.audit(true), &client, &entry).await;
    let reads_on = (
        fx.audit_reads("too_large"),
        fx.audit_reads("ok"),
        fx.fenced_reads("ok"),
    );
    let far = fx.run_on(&fx.audit(true), &far_above, &entry).await;
    let reads_far = (
        fx.audit_reads("too_large"),
        fx.audit_reads("ok"),
        fx.fenced_reads("ok"),
    );

    assert_eq!(off, AuditOutcome::Skipped(AuditSkip::AboveCeiling), "off");
    assert_eq!(reads_off, (1, 0, 0), "off: the plain read only");
    assert_eq!(
        fenced_off,
        (0, 0, 0, 0),
        "off: the switch off ticks no fenced result"
    );
    assert_eq!(on, AuditOutcome::Match, "on");
    assert_eq!(
        reads_on,
        (2, 0, 1),
        "on: the plain read's TOO_LARGE, then one fenced read (atomic too_large, atomic ok, \
         fenced ok)"
    );
    assert_eq!(
        far,
        AuditOutcome::Match,
        "on, far above the plain item ceiling: no raised ceiling skips it"
    );
    assert_eq!(reads_far, (3, 0, 2), "far above: one more fenced read");
    assert_eq!(fx.audit_fenced(), (2, 0, 0, 0), "two fenced reads");
    assert_eq!(fx.audits(), (2, 0, 1));
}

/// Above the plain byte ceiling alone — within the item ceiling, so the
/// plain read answers `TOO_LARGE` mid-read, sized from its tallies — the
/// fenced read compares, `match`. The byte ceiling is a sixteenth of a whole
/// read's bytes, under the retired raised read's byte ceiling (×8) too.
///
/// Fails if: the audit's fenced read were taken on an item early-out only, or
/// a mid-read `TOO_LARGE` were answered `AboveCeiling` with the switch on.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_above_the_plain_byte_ceiling_alone_reads_fenced_and_is_match() {
    let fx = Fx::new("audit-above-bytes").await;
    let entry = fx.entry().await;
    let whole_read_bytes = read_atomic(fx.tv.client(), fx.tv.key_schema(), fx.oid)
        .await
        .reply_bytes;
    assert!(
        whole_read_bytes / 16 > 0
            && fx.item_bound().await <= SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
        "fixture: the object must be over the byte ceiling and within the item ceiling"
    );
    let client = client_with_limits(
        SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
        whole_read_bytes / 16,
    )
    .await;
    client
        .install_observability_hooks(Arc::clone(&fx.metrics))
        .await;

    let outcome = fx.run_on(&fx.audit(true), &client, &entry).await;

    assert_eq!(outcome, AuditOutcome::Match);
    assert_eq!(
        (
            fx.audit_reads("too_large"),
            fx.audit_reads("ok"),
            fx.fenced_reads("ok")
        ),
        (1, 0, 1),
        "the plain read answers TOO_LARGE on bytes, then one fenced read compares"
    );
    assert_eq!(fx.audit_fenced(), (1, 0, 0, 0));
    assert_eq!(fx.audits(), (1, 0, 0));
}

/// The fenced read of an above-ceiling object, end to end. It answers
/// `match`; it is one `path="fenced"` `ok` read and one fence hold under the
/// audit's own consumer — none under the cold start's — one fence acquire
/// and one `result="read"` tick; it leaves the entry cached and holds
/// nothing once done. The store's command feed (`MONITOR`, the object's keys,
/// commands a client sent) shows the plain read's one snapshot-script
/// `EVALSHA`, at the client's plain ceilings — no atomic read at raised
/// ceilings — then one acquire and one release with every fan-out command
/// between them, under the lease.
///
/// Fails if: the audit's fenced read were observed under the cold start's
/// consumer (`ColdStartFenceHook::for_audit` passing
/// `SlimSnapshotReadConsumer::StructuralCache`, or `read_fresh` passing it to
/// `read_object_snapshot_fenced`); the `read` tick were dropped; or the
/// raised read were restored after the `TOO_LARGE`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_above_the_plain_ceiling_reads_fenced_under_its_own_consumer_and_is_match() {
    let fx = Fx::new("audit-fenced").await;
    warm_fence_scripts("audit-fenced").await;
    let entry = fx.entry().await;
    let (client, _) = fx.client_one_item_under_the_bound().await;
    let plain = SnapshotCeilings::of_client(&client);
    let tag = format!("{}{{{}}}:obj:", fx.tv.prefix(), fx.oid);

    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let outcome = fx.run_on(&fx.audit(true), &client, &entry).await;
    let fence = monitor_fence(fx.tv.prefix());
    let _: i64 = fx.tv.client().pool.exists(&fence).await.unwrap();
    let commands: Vec<MonitoredCommand> = tap
        .drain_until(&fence)
        .await
        .into_iter()
        .filter(|c| !c.issued_by_script() && c.args.iter().any(|a| a.contains(&tag)))
        .collect();

    assert_eq!(outcome, AuditOutcome::Match);
    assert_eq!(fx.audits(), (1, 0, 0));
    assert_eq!(
        (
            fx.audit_reads("too_large"),
            fx.audit_reads("ok"),
            fx.fenced_reads("ok"),
            fx.fenced_reads("fence_broken"),
        ),
        (1, 0, 1, 0),
        "the plain read's TOO_LARGE, then one fenced ok read under structural_cache_audit"
    );
    assert_eq!(
        (
            fx.reads_of("structural_cache", "fenced", "ok"),
            fx.fence_holds("structural_cache")
        ),
        (0, 0),
        "nothing under the cold start's consumer"
    );
    assert_eq!(
        fx.fence_holds(AUDIT_CONSUMER),
        1,
        "the writers' pause is observed as one fence hold under structural_cache_audit"
    );
    assert_eq!(fx.fence_acquires(), 1, "one acquire");
    assert_eq!(
        fx.audit_fenced(),
        (1, 0, 0, 0),
        "one result=\"read\" tick (read, paced, busy, failed)"
    );
    assert!(
        fx.cached().is_some(),
        "a match leaves the cached entry in place"
    );
    assert_eq!(
        capacity(&client, &fx.cache),
        NOTHING_HELD,
        "nothing held once done"
    );

    let (snapshot, acquire, release) = (snapshot_sha(), acquire_sha(), release_sha());
    let positions = |sha: &str| -> Vec<usize> {
        commands
            .iter()
            .enumerate()
            .filter(|(_, c)| is_evalsha_of(c, sha))
            .map(|(i, _)| i)
            .collect()
    };
    let snapshot_reads = positions(&snapshot);
    assert_eq!(
        snapshot_reads.len(),
        1,
        "one snapshot-script read, the plain one: no atomic read after the TOO_LARGE; \
         commands: {commands:?}"
    );
    let read = &commands[snapshot_reads[0]];
    assert_eq!(
        (
            read.args.get(7).map(String::as_str),
            read.args.get(8).map(String::as_str)
        ),
        (
            Some(plain.max_items.to_string().as_str()),
            Some(plain.max_bytes.to_string().as_str())
        ),
        "the snapshot read passes the client's plain ceilings (ARGV[4], ARGV[5]), never raised \
         ones; read: {read:?}"
    );
    let (acquires, releases) = (positions(&acquire), positions(&release));
    let ([acquired], [released]) = (acquires.as_slice(), releases.as_slice()) else {
        panic!(
            "one acquire and one release; got {acquires:?} and {releases:?}; commands: \
             {commands:?}"
        );
    };
    assert!(
        snapshot_reads[0] < *acquired,
        "the plain read comes before the fenced read"
    );
    let fan_out: Vec<usize> = commands
        .iter()
        .enumerate()
        .filter(|(_, c)| c.name() != "EVALSHA")
        .map(|(i, _)| i)
        .collect();
    assert!(
        !fan_out.is_empty() && fan_out.iter().all(|i| acquired < i && i < released),
        "the fan-out runs, and only between the acquire and the release — under the lease; \
         commands: {commands:?}"
    );
}

/// Above the ceiling, an out-of-band rewrite is caught by the fenced read:
/// `diverged` at the entry's cut, naming the changed key, with the entry
/// dropped.
///
/// Fails if: the fenced snapshot were not the one compared (e.g. `read_fresh`
/// answering the entry's own state), or the fenced read's structural facts
/// were not taken from that read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_above_the_plain_ceiling_after_an_out_of_band_rewrite_is_diverged_and_drops_the_entry(
) {
    let fx = Fx::new("audit-fenced-diverged").await;
    let entry = fx.entry().await;
    let (client, _) = fx.client_one_item_under_the_bound().await;
    let (change, key) = out_of_band_changes().remove(0);
    rewrite_out_of_band(&fx.tv, fx.oid, change).await;

    let outcome = fx.run_on(&fx.audit(true), &client, &entry).await;

    assert_eq!(
        outcome,
        AuditOutcome::Diverged {
            seq: entry.seq,
            first_difference: key.to_owned(),
        }
    );
    assert!(fx.cached().is_none(), "the entry is dropped");
    assert_eq!(fx.fenced_reads("ok"), 1, "compared through the fenced read");
    assert_eq!(fx.audit_fenced(), (1, 0, 0, 0));
    assert_eq!(fx.audits(), (0, 1, 0));
}

/// Pacing: one audit fenced read per replica per interval, the interval
/// running from the read's first lease grant. A second above-ceiling audit
/// within the (production) interval — on a clone of the audit, which shares
/// the slot — is `skipped` (`Paced`): it ticks `paced`, sends no fence
/// acquire and reads nothing fenced. A clone whose interval is zero, sharing
/// the same slot, reads again.
///
/// Fails if: `FencedPacing::due` always answered `true`, or `record` were
/// never called after a granted read (the second audit reads); the clones
/// did not share `last_fenced` (a fresh `Arc` per `maybe_spawn`); or
/// `with_fenced_min_interval` were ignored (the third audit is `Paced`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_fenced_reads_are_paced_to_one_per_interval_across_clones_of_the_audit() {
    let fx = Fx::new("audit-paced").await;
    let entry = fx.entry().await;
    let (client, _) = fx.client_one_item_under_the_bound().await;
    let audit = fx.audit(true);

    let first = fx.run_on(&audit, &client, &entry).await;
    let after_first = (
        fx.fence_acquires(),
        fx.audit_fenced(),
        fx.fenced_reads("ok"),
    );
    let second = fx.run_on(&audit.clone(), &client, &entry).await;
    let after_second = (
        fx.fence_acquires(),
        fx.audit_fenced(),
        fx.fenced_reads("ok"),
    );
    let unpaced = audit.clone().with_fenced_min_interval(Duration::ZERO);
    let third = fx.run_on(&unpaced, &client, &entry).await;

    assert_eq!(first, AuditOutcome::Match, "the first reads fenced");
    assert_eq!(after_first.1, (1, 0, 0, 0), "fixture: one read tick");
    assert!(after_first.0 >= 1, "fixture: the first sent an acquire");
    assert_eq!(
        second,
        AuditOutcome::Skipped(AuditSkip::Paced),
        "a second within the interval"
    );
    assert_eq!(
        after_second,
        (after_first.0, (1, 1, 0, 0), after_first.2),
        "the paced audit ticks paced, sends no acquire and reads nothing fenced (acquires, \
         (read, paced, busy, failed), fenced ok)"
    );
    assert_eq!(
        third,
        AuditOutcome::Match,
        "at a zero interval, the same slot reads again"
    );
    assert_eq!(fx.audit_fenced(), (2, 1, 0, 0));
    assert_eq!(fx.fenced_reads("ok"), 2);
    assert_eq!(fx.audits(), (2, 0, 1));
}

/// With the switch on, an unmarked object above the plain ceiling is
/// `skipped` (`NoChsComplete`) from the plain read's `TOO_LARGE` alone:
/// before any capacity — it ends while the test holds the one-at-a-time
/// cold-start permit, where a fenced read would be `Busy` — before the
/// pacing slot, and with no fenced read, acquire or fenced tick. The slot
/// being untaken, the same audit then reads the re-marked object fenced.
///
/// Fails if: the unmarked check moved after the fenced read's capacity (the
/// first audit is `Busy`), or after `pacing.due()` with the slot recorded
/// there (the second is `Paced`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_above_the_ceiling_of_an_unmarked_object_is_skipped_before_the_capacity_and_the_pacing_slot(
) {
    let fx = Fx::new("audit-above-unmarked").await;
    let entry = fx.entry().await;
    unmark(&fx.tv, fx.oid).await;
    let (client, _) = fx.client_one_item_under_the_bound().await;
    let audit = fx.audit(true);
    let permit = fx.cache.acquire_cold_start_permit().await;

    let unmarked = fx.run_on(&audit, &client, &entry).await;
    let after_unmarked = (
        fx.audit_reads("too_large"),
        fx.audit_reads("ok"),
        fx.fenced_reads("ok"),
        fx.fence_acquires(),
        fx.audit_fenced(),
    );
    drop(permit);
    fx.remark().await;
    let marked = fx.run_on(&audit, &client, &entry).await;

    assert_eq!(unmarked, AuditOutcome::Skipped(AuditSkip::NoChsComplete));
    assert_eq!(
        after_unmarked,
        (1, 0, 0, 0, (0, 0, 0, 0)),
        "the plain read only: no fenced read, no acquire, no fenced tick"
    );
    assert_eq!(
        marked,
        AuditOutcome::Match,
        "the unmarked audit took no pacing slot: the next one reads fenced"
    );
    assert_eq!(fx.audit_fenced(), (1, 0, 0, 0));
    assert_eq!(fx.audits(), (1, 0, 1));
}

/// What a busy-test case holds of the fenced read's capacity while the audit
/// runs; dropped at the case's end.
type Held = Vec<Box<dyn Send>>;

/// The audit never waits for capacity: with what one of its fenced read's
/// waits needs held by the test — the one-at-a-time cold-start permit, both
/// plain permits (the pair's second half), every `P_f` permit, or the whole
/// catch-up budget `T` — the audit ends `skipped` (`Busy`) while it is still
/// held, ticks `busy` (and `skipped`), sends no fence acquire and holds
/// nothing afterwards; the one-at-a-time permit it may have taken first is
/// free again. A declined grant counts nothing on the catch-up budget's
/// admission series (no `refused_timeout`, no `waited`): nothing waited for
/// memory.
///
/// One audit (one pacing slot) runs every case and then, with nothing held,
/// reads fenced: a busy audit was granted no lease, so it used up no
/// interval — each case after the first would otherwise be `Paced`.
///
/// "Promptly" is a deadline, not a timing: each wait, had it waited, would
/// have ended only at the waiting cap, `reader_wait_max(L)` with no deadline
/// in scope (read from the cost model at run time); the audit must end
/// within half of it (and within [`AUDIT_BUDGET`]) while the test still
/// holds what it waits for.
///
/// Fails if: the audit's hook waited (in `ColdStartFenceHook::take_capacity`,
/// `CapacityWait::None`'s pair taken with
/// `acquire_fenced_cold_start_read_permits` under the cap's deadline, or its
/// grant with `reserve(bytes, Some(started + wait_cap))`); a
/// `CapacityTimeout` were mapped to `FencedReadFailed` instead of `Busy`; the
/// grant were asked with `reserve(bytes, Some(now))` (ticks
/// `refused_timeout`); or the pacing slot were recorded before the read
/// rather than at its first grant (the second case is `Paced`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_above_the_ceiling_whose_capacity_is_not_free_at_once_is_skipped_busy_without_waiting(
) {
    let fx = Fx::new("audit-busy").await;
    let entry = fx.entry().await;
    let (client, plain_items) = fx.client_one_item_under_the_bound().await;
    let cap = wait_cap(&coefficients(&client), None, fx.estimate(plain_items).await);
    let bound = AUDIT_BUDGET.min(cap / 2);
    assert!(
        bound > Duration::ZERO,
        "fixture: the waiting cap {cap:?} leaves a bound"
    );
    let audit = fx.audit(true);

    for case in [
        "the one-at-a-time cold-start permit",
        "both plain cold-start permits",
        "every P_f permit",
        "the whole catch-up budget",
    ] {
        let mut held: Held = Vec::new();
        match case {
            "the one-at-a-time cold-start permit" => {
                held.push(Box::new(fx.cache.acquire_cold_start_permit().await));
            }
            "both plain cold-start permits" => {
                held.push(Box::new(
                    fx.cache.acquire_plain_cold_start_read_permit().await,
                ));
                held.push(Box::new(
                    fx.cache.acquire_plain_cold_start_read_permit().await,
                ));
            }
            "every P_f permit" => {
                let gate = &client.catchup_fenced_read_gate;
                for _ in 0..gate.limit() {
                    held.push(Box::new(
                        gate.acquire(
                            tokio::time::Instant::now() + LONG_TTL,
                            SnapshotReadConsumer::Catchup,
                        )
                        .await
                        .expect("fixture: the test takes every P_f permit"),
                    ));
                }
            }
            _ => {
                let budget = &client.catchup_build_budget;
                held.push(Box::new(
                    budget
                        .reserve(budget.total_bytes(), None)
                        .await
                        .expect("fixture: the test takes the whole budget"),
                ));
            }
        }
        let holding = capacity(&client, &fx.cache);
        let before = (fx.fence_acquires(), fx.audit_fenced(), fx.audits());
        let admissions_before = (fx.admissions("refused_timeout"), fx.admissions("waited"));
        let handle = audit
            .maybe_spawn(&client, fx.tv.key_schema(), &fx.cache, &entry)
            .expect("an audit at rate 1.0 starts");

        let outcome = tokio::time::timeout(bound, handle)
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "{case}: the audit must end within {bound:?} — half its waiting cap — while \
                     the test holds it; it waited for it"
                )
            })
            .expect("the audit task must not panic");
        let after = capacity(&client, &fx.cache);
        let one_at_a_time_free = case == "the one-at-a-time cold-start permit"
            || tokio::time::timeout(BLOCKED_WINDOW, fx.cache.acquire_cold_start_permit())
                .await
                .is_ok();
        let (acquires, fenced, audits) = (fx.fence_acquires(), fx.audit_fenced(), fx.audits());
        let admissions_after = (fx.admissions("refused_timeout"), fx.admissions("waited"));
        drop(held);

        assert_eq!(
            outcome,
            Some(AuditOutcome::Skipped(AuditSkip::Busy)),
            "{case}"
        );
        assert_eq!(
            after, holding,
            "{case}: the audit holds nothing afterwards: only the test's holding remains"
        );
        assert!(
            one_at_a_time_free,
            "{case}: the one-at-a-time permit the audit took first is free again"
        );
        assert_eq!(acquires, before.0, "{case}: no fence acquire is sent");
        assert_eq!(
            fenced,
            (before.1 .0, before.1 .1, before.1 .2 + 1, before.1 .3),
            "{case}: one busy tick (read, paced, busy, failed)"
        );
        assert_eq!(
            audits,
            (before.2 .0, before.2 .1, before.2 .2 + 1),
            "{case}: one skipped tick"
        );
        assert_eq!(
            admissions_after, admissions_before,
            "{case}: no refused_timeout or waited admission: nothing waited for memory"
        );
        assert_eq!(
            capacity(&client, &fx.cache),
            NOTHING_HELD,
            "{case}: the test's holding released"
        );
    }
    assert_eq!(fx.fenced_reads("ok"), 0, "no fenced read ran");

    let free = fx.run_on(&audit, &client, &entry).await;

    assert_eq!(
        free,
        AuditOutcome::Match,
        "the busy audits used up no pacing interval: the same audit reads fenced at once"
    );
    assert_eq!(fx.audit_fenced(), (1, 0, 4, 0));
    assert_eq!(fx.fenced_reads("ok"), 1);
}

/// An object gone during the audit's fenced read — its header removed while
/// the read sleeps out a forged live lease's `RETRY_AFTER`, so the next
/// acquire answers `NOT_FOUND` — is `skipped` (`Gone`), as within the plain
/// ceilings, and ticks no fenced result; it was granted no lease, so it uses
/// up no pacing interval: the same audit then reads another object fenced.
///
/// Fails if: a `NotFound` from the fenced read were ticked `failed` (or
/// answered `ReadFailed`), or the pacing slot were recorded before the read
/// rather than at its first grant (the second audit is `Paced`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_of_an_object_gone_during_its_fenced_read_is_skipped_gone_and_uses_no_interval() {
    let fx = Fx::new("audit-fenced-gone").await;
    let other = Fx::new("audit-fenced-gone-other").await;
    let entry = fx.entry().await;
    let other_entry = other.entry().await;
    let (client, _) = fx.client_one_item_under_the_bound().await;
    let (other_client, _) = other.client_one_item_under_the_bound().await;
    let audit = fx.audit(true);
    forge_live_lease(&fx.tv, fx.oid).await;

    let handle = audit
        .maybe_spawn(&client, fx.tv.key_schema(), &fx.cache, &entry)
        .expect("an audit at rate 1.0 starts");
    wait_until(
        "fixture: the audit's acquire must meet the forged lease",
        || fx.fence_acquires_of("retry_after_epoch") >= 1,
    )
    .await;
    let _: i64 = fx
        .tv
        .client()
        .pool
        .del(fx.header_key())
        .await
        .expect("DEL obj:hdr must succeed");
    remove_forged_lease(&fx.tv, fx.oid).await;
    let gone = finished(handle).await;
    let next = other.run_on(&audit, &other_client, &other_entry).await;

    assert_eq!(gone, Some(AuditOutcome::Skipped(AuditSkip::Gone)));
    assert_eq!(
        fx.audit_fenced(),
        (0, 0, 0, 0),
        "an object gone during the fenced read is no fenced result"
    );
    assert_eq!(fx.audits(), (0, 0, 1));
    assert_eq!(capacity(&client, &fx.cache), NOTHING_HELD);
    assert_eq!(
        next,
        AuditOutcome::Match,
        "the gone audit was granted no lease: the same audit reads at once"
    );
    assert_eq!(other.audit_fenced(), (1, 0, 0, 0));
}

/// The fenced read's capacity closed at shutdown — the client's `P_f` gate
/// closed, as `ValkeyClient::quit` closes it — ends the audit `skipped`
/// (`ReadFailed`) with no fenced result ticked (not `busy`, not `failed`),
/// no acquire sent, and nothing held.
///
/// Fails if: `FencedFailure::Closed` fell through to the `failed` tick in
/// `read_fresh`, or `CatchUpRefusal::Closed` were classified
/// `CapacityTimeout` (`busy`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_whose_fenced_capacity_is_closed_at_shutdown_ticks_no_fenced_result() {
    let fx = Fx::new("audit-fenced-closed").await;
    let entry = fx.entry().await;
    let (client, _) = fx.client_one_item_under_the_bound().await;
    client.catchup_fenced_read_gate.close();

    let outcome = fx.run_on(&fx.audit(true), &client, &entry).await;

    assert_eq!(outcome, AuditOutcome::Skipped(AuditSkip::ReadFailed));
    assert_eq!(fx.audit_fenced(), (0, 0, 0, 0), "no fenced result");
    assert_eq!(fx.fence_acquires(), 0, "no acquire");
    assert_eq!(fx.audits(), (0, 0, 1));
    let held = capacity(&client, &fx.cache);
    assert_eq!(
        (held.in_flight, held.reserved_bytes, held.overcommit_bytes),
        (NONE_IN_FLIGHT, 0, 0),
        "nothing held"
    );
}

/// The fenced read's memory is charged to the catch-up budget like a cold
/// start's, and freed once the snapshot is decoded. The audit's acquire reply
/// is held at a proxy: across it the audit holds its pair, one `P_f` permit
/// and exactly `(SNAPSHOT_MEMORY_FACTOR + 1) × b` of `T` — `b` the estimate
/// `LayerSize::from_too_large` makes of the plain read's `TOO_LARGE` answer,
/// the clamp to `T` read from the client. Then the reply of its repair's read
/// (the one script naming the object's stream) is held: by then — after the
/// decode, before the repair — it holds nothing.
///
/// Fails if: the grant were not taken in the audit's mode (no `reserve` in
/// `take_capacity`, or a zero-byte grant), or the snapshot's memory were held
/// across the repair (`drop(memory)` moved after `repair_to_target_with`, or the
/// hold's `read_permits` kept to the end of the audit).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_fenced_read_charges_its_grant_to_the_catch_up_budget_and_frees_it_before_the_repair()
{
    let fx = Fx::new("audit-fenced-memory").await;
    warm_fence_scripts("audit-fenced-memory").await;
    let entry = fx.entry().await;
    fx.commit_fast("behind", 1).await;
    let plain_items = fx.item_bound().await - 1;
    let estimate = fx.estimate(plain_items).await;
    let (acquire_gate, repair_gate) = (ReplyGate::new(), ReplyGate::new());
    let rule: ProxyRule = {
        let (acquire, fence, stream) = (
            acquire_sha(),
            fence_key(&fx.tv, fx.oid),
            fx.tv.key_schema().layer_stream(fx.oid),
        );
        let (acquire_gate, repair_gate) = (acquire_gate.clone(), repair_gate.clone());
        Arc::new(move |args: &[Vec<u8>]| {
            let name = args.first()?.to_ascii_uppercase();
            if name != b"EVALSHA" && name != b"EVAL" {
                return None;
            }
            if args.get(1).map(Vec::as_slice) == Some(acquire.as_bytes())
                && args.get(3).map(Vec::as_slice) == Some(fence.as_bytes())
            {
                return Some(ProxyAction::HoldReplies(acquire_gate.clone()));
            }
            args.iter()
                .any(|a| String::from_utf8_lossy(a).contains(stream.as_str()))
                .then(|| ProxyAction::HoldReplies(repair_gate.clone()))
        })
    };
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let client = client_through_with_limits(&proxy, plain_items, NO_BYTE_LIMIT).await;
    let grant = expected_grant_bytes(&client, estimate.bytes);
    assert!(
        grant > 0 && grant <= client.catchup_build_budget.total_bytes(),
        "fixture: the expected grant {grant} is not empty and within T"
    );

    let handle = fx
        .audit(true)
        .maybe_spawn(&client, fx.tv.key_schema(), &fx.cache, &entry)
        .expect("an audit at rate 1.0 starts");
    acquire_gate.reply_held().await;
    let across_acquire = capacity(&client, &fx.cache);
    acquire_gate.open();
    repair_gate.reply_held().await;
    let at_the_repair = capacity(&client, &fx.cache);
    repair_gate.open();
    let outcome = finished(handle).await;

    assert_eq!(
        across_acquire,
        Capacity {
            in_flight: ONE_FENCED,
            p_f_held: 1,
            reserved_bytes: grant,
            overcommit_bytes: 0,
        },
        "across its acquire the audit holds its pair, one P_f permit and (factor + 1) × b of T"
    );
    assert_eq!(
        at_the_repair, NOTHING_HELD,
        "at its repair's read the snapshot is decoded and its pair, P_f and memory are given back"
    );
    assert_eq!(outcome, Some(AuditOutcome::Match));
    assert!(
        proxy.hits() >= 2,
        "fixture: the acquire and the repair's read were held"
    );
    assert_eq!(capacity(&client, &fx.cache), NOTHING_HELD);
}

/// A live reader lease nobody holds (forged into the fence key) makes every
/// acquire answer `RETRY_AFTER` `epoch`: the audit's fenced read sleeps it
/// out holding no pair, no `P_f` and no grant — the audit mode refuses to
/// wait for capacity, never for the fence — and once the lease is gone it
/// compares, `match`, and releases its own lease. A lease that outlasts
/// `reader_wait_max(L)` (`failed`) is out of reach here.
///
/// Fails if: the hook kept its capacity through the `RETRY_AFTER` sleep
/// (`release_capacity` dropped from `on_retry_after`), or treated a
/// `RETRY_AFTER` as a capacity refusal (the audit ends `Busy` or
/// `FencedReadFailed` while the lease lives).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_fenced_read_meeting_a_live_lease_waits_holding_nothing_then_compares() {
    let fx = Fx::new("audit-fenced-lease").await;
    let entry = fx.entry().await;
    let (client, _) = fx.client_one_item_under_the_bound().await;
    forge_live_lease(&fx.tv, fx.oid).await;

    let handle = fx
        .audit(true)
        .maybe_spawn(&client, fx.tv.key_schema(), &fx.cache, &entry)
        .expect("an audit at rate 1.0 starts");
    wait_until(
        "fixture: the audit's acquire must meet the forged lease",
        || fx.fence_acquires_of("retry_after_epoch") >= 1,
    )
    .await;
    wait_for_capacity(
        &client,
        &fx.cache,
        NOTHING_HELD,
        "sleeping out its RETRY_AFTER, the audit must hold no pair, P_f or grant",
    )
    .await;
    let waiting = !handle.is_finished();
    remove_forged_lease(&fx.tv, fx.oid).await;
    let outcome = finished(handle).await;

    assert!(waiting, "the audit waits on the live lease");
    assert_eq!(outcome, Some(AuditOutcome::Match));
    assert_eq!(fx.fenced_reads("ok"), 1);
    assert_eq!(fx.audit_fenced(), (1, 0, 0, 0));
    assert_eq!(capacity(&client, &fx.cache), NOTHING_HELD);
    wait_until_no_lease(&fx.tv, fx.oid, "the audit's own lease").await;
}

/// The audit's plain read takes no cold-start permit (at most one audit runs
/// per replica): with every cold-start permit held elsewhere — the
/// one-at-a-time permit and both plain permits — an audit within the shared
/// ceilings still compares.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_plain_read_takes_no_cold_start_permit() {
    let fx = Fx::new("audit-plain-no-permit").await;
    let entry = fx.entry().await;
    let pair = fx.cache.acquire_fenced_cold_start_read_permits().await;
    let plain = fx.cache.acquire_plain_cold_start_read_permit().await;

    let handle = fx
        .audit(false)
        .maybe_spawn(fx.tv.client(), fx.tv.key_schema(), &fx.cache, &entry)
        .expect("an audit at rate 1.0 starts");
    let outcome = tokio::time::timeout(AUDIT_BUDGET, handle)
        .await
        .expect(
            "the audit's plain read takes no cold-start permit, so the audit ends while every \
             one is held",
        )
        .expect("the audit task must not panic");

    assert_eq!(outcome, Some(AuditOutcome::Match));
    drop((pair, plain));
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. Isolation
// ═════════════════════════════════════════════════════════════════════════════

/// Every series the audit must not tick (SR-2.4): the cache's lookups,
/// reloads, divergence and repair histograms, the cold start's series, and
/// every snapshot-read, fence and catch-up read-permit series of the cold
/// start's own consumer.
///
/// Each series is keyed by its name and its labels sorted, and the set is
/// sorted: the rendering's label order differs from series to series, and a
/// histogram's series order can shift when another consumer's first
/// observation opens a cell in it, which neither is a tick.
fn isolated_series(metrics: &Metrics) -> BTreeMap<String, String> {
    const SERIES: [&str; 7] = [
        "live_edit_structural_cache_lookups_total",
        "live_edit_structural_cache_reloads_total",
        "live_edit_structural_cache_divergence_total",
        "live_edit_structural_cache_replayed_entries",
        "live_edit_structural_cache_repair_bytes",
        "live_edit_structural_cold_start_reads_total",
        "live_edit_structural_cold_start_read_duration_seconds",
    ];
    render(metrics)
        .lines()
        .filter(|l| {
            SERIES.iter().any(|s| l.starts_with(s))
                || ((l.starts_with("live_edit_snapshot_")
                    || l.starts_with("live_edit_catchup_snapshot_read_"))
                    && l.contains("consumer=\"structural_cache\""))
        })
        .map(|l| {
            let (series, value) = l.rsplit_once(' ').expect("a sample line ends in its value");
            let key = match series.split_once('{') {
                Some((name, labels)) => {
                    let mut labels: Vec<&str> = labels.trim_end_matches('}').split(',').collect();
                    labels.sort_unstable();
                    format!("{name}{{{}}}", labels.join(","))
                }
                None => series.to_owned(),
            };
            (key, value.to_owned())
        })
        .collect()
}

/// Three audits — a `match` whose repair replays three entries, a `match`
/// through the fenced read, and a `diverged` that drops the entry — tick the
/// audit counter and the audit's own snapshot-read cells, and nothing in
/// [`isolated_series`].
///
/// Fails if: the audit's fenced read ticked the cold start's series (e.g. its
/// hook built with `ColdStartFenceHook::new`, observed under
/// `consumer="structural_cache"`), or the audit ticked a cache series.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_ticks_only_its_own_counter_and_snapshot_read_consumer() {
    let fx = Fx::new("audit-isolation").await;
    let entry = fx.entry().await;
    fx.commit_fast("behind", 3).await;
    let (fenced_client, _) = fx.client_one_item_under_the_bound().await;
    let before = isolated_series(&fx.metrics);
    assert!(
        before
            .keys()
            .any(|k| k.starts_with("live_edit_structural_cache_replayed_entries")),
        "fixture: the warm-up's repair recorded the histogram, so a change to it shows"
    );

    let by_repair = fx.run(&fx.audit(false), &entry).await;
    let fenced = fx.run_on(&fx.audit(true), &fenced_client, &entry).await;
    rewrite_out_of_band(&fx.tv, fx.oid, out_of_band_changes().remove(0).0).await;
    let diverged = fx.run(&fx.audit(false), &entry).await;

    assert_eq!(
        (&by_repair, &fenced),
        (&AuditOutcome::Match, &AuditOutcome::Match),
        "fixture"
    );
    assert!(
        matches!(diverged, AuditOutcome::Diverged { .. }),
        "fixture: got {diverged:?}"
    );
    assert_eq!(
        isolated_series(&fx.metrics),
        before,
        "the audit ticks no cache, cold-start or structural_cache snapshot-read, fence or \
         read-permit series"
    );
    assert_eq!(fx.audits(), (2, 1, 0));
    assert_eq!(
        (
            fx.audit_reads("ok"),
            fx.audit_reads("too_large"),
            fx.fenced_reads("ok")
        ),
        (2, 1, 1),
        "the audit's reads are counted under its own consumer: two plain reads that answered, \
         one TOO_LARGE, and the fenced read after it"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 6. One at a time, and shutdown
// ═════════════════════════════════════════════════════════════════════════════

/// While an audit runs — parked in its before-read hook — a trigger on the
/// store, or on a clone of it, starts none; once it ends the flag is clear
/// and the next trigger starts one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_runs_one_at_a_time_per_store_and_its_clones() {
    let fx = Fx::new("audit-one").await;
    let entry = fx.entry().await;
    let (hook, mut entered_rx, open) = parking_hook();
    let ls = fx
        .ls
        .clone()
        .with_structural_cache_audit(fx.audit(false).with_before_read_hook_for_test(hook));
    let clone = ls.clone();

    let first = ls
        .maybe_spawn_structural_cache_audit(&entry)
        .expect("the first audit starts");
    entered(&mut entered_rx, 1).await;
    let running = ls.structural_cache_audit().is_running();
    let on_the_store = ls.maybe_spawn_structural_cache_audit(&entry);
    let on_a_clone = clone.maybe_spawn_structural_cache_audit(&entry);
    open.send_replace(true);
    let first = finished(first).await;
    let cleared = !clone.structural_cache_audit().is_running();
    let next = clone
        .maybe_spawn_structural_cache_audit(&entry)
        .expect("with the first audit ended, the next starts");
    let next = finished(next).await;

    assert!(running, "the flag is set while the audit runs");
    assert!(on_the_store.is_none(), "busy: no second audit on the store");
    assert!(on_a_clone.is_none(), "busy: no second audit on a clone");
    assert_eq!(first, Some(AuditOutcome::Match));
    assert!(cleared, "the flag clears when the audit ends");
    assert_eq!(next, Some(AuditOutcome::Match));
    assert_eq!(fx.audits(), (2, 0, 0), "the refused triggers tick nothing");
}

/// An audit that panics releases the flag on the unwind and ticks nothing;
/// the next trigger starts one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_flag_is_released_when_the_audit_panics() {
    let fx = Fx::new("audit-panic").await;
    let entry = fx.entry().await;
    let panicked = Arc::new(AtomicBool::new(false));
    let hook: TestHook = {
        let panicked = Arc::clone(&panicked);
        Arc::new(move || {
            let first = !panicked.swap(true, Ordering::SeqCst);
            async move {
                assert!(!first, "the first audit panics in its before-read hook");
            }
            .boxed()
        })
    };
    let audit = fx.audit(false).with_before_read_hook_for_test(hook);

    let first = audit
        .maybe_spawn(fx.tv.client(), fx.tv.key_schema(), &fx.cache, &entry)
        .expect("the first audit starts");
    let joined = tokio::time::timeout(AUDIT_BUDGET, first)
        .await
        .expect("the audit must end within the budget");
    let released = !audit.is_running();
    let next = fx.run(&audit, &entry).await;

    assert!(
        joined.as_ref().is_err_and(|e| e.is_panic()),
        "fixture: the first audit panicked; got {joined:?}"
    );
    assert!(released, "the panic's unwind released the flag");
    assert_eq!(next, AuditOutcome::Match);
    assert_eq!(fx.audits(), (1, 0, 0), "the panicked audit ticks nothing");
}

/// The shutdown token ends a running audit — here parked before its read —
/// with no outcome and no tick, releases the flag, and no audit starts after
/// it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_is_cancelled_by_the_shutdown_token_and_none_starts_after_it() {
    let fx = Fx::new("audit-shutdown").await;
    let entry = fx.entry().await;
    let (hook, mut entered_rx, _never_opened) = parking_hook();
    let shutdown = CancellationToken::new();
    let audit = StructuralCacheAudit::new(
        1.0,
        false,
        Arc::new(UnboundedAuditCharges),
        shutdown.clone(),
    )
    .with_before_read_hook_for_test(hook);

    let handle = audit
        .maybe_spawn(fx.tv.client(), fx.tv.key_schema(), &fx.cache, &entry)
        .expect("the audit starts");
    entered(&mut entered_rx, 1).await;
    shutdown.cancel();
    let outcome = finished(handle).await;
    let released = !audit.is_running();
    let after = audit.maybe_spawn(fx.tv.client(), fx.tv.key_schema(), &fx.cache, &entry);

    assert_eq!(outcome, None, "a cancelled audit answers no outcome");
    assert!(released, "the cancellation released the flag");
    assert!(after.is_none(), "no audit starts once shutdown has begun");
    assert_eq!(fx.audits(), (0, 0, 0), "a cancelled audit ticks nothing");
}

/// Shutdown during the fenced read: the read's fan-out is held back (the
/// read's fault seam delays its first command) while the audit holds its
/// lease, its pair, one `P_f` permit and its grant
/// (`(SNAPSHOT_MEMORY_FACTOR + 1) × b`, as [`expected_grant_bytes`]); the
/// token fires, and the audit ends with no outcome and no tick, every one of
/// them given back — the lease by its guard's background release — and the
/// one-audit flag cleared.
///
/// Needs `test-fault-injection` (the read's fault seam).
///
/// Fails if: the audit's future were not dropped on the token (the
/// `tokio::select!` in `maybe_spawn` awaiting the audit to its end), or the
/// hook's capacity outlived it (e.g. `std::mem::forget` of the hook's pair).
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_cancelled_mid_fenced_read_gives_back_its_pair_p_f_grant_and_lease() {
    use crate::storage::structural_cache_cold_start_test::assert_cold_start_permits_free;
    use live_edit::storage::diff_read::{
        ObjectSnapshotReadFaultInjector, ObjectSnapshotReadFaultTarget, ObjectSnapshotReader,
    };
    /// Longer than the test: the read is cancelled inside it.
    const FAN_OUT_DELAY: Duration = Duration::from_secs(60);
    let fx = Fx::new("audit-shutdown-fenced").await;
    let entry = fx.entry().await;
    let (client, plain_items) = fx.client_one_item_under_the_bound().await;
    let estimate = fx.estimate(plain_items).await;
    let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
    injector.push_delay_at(
        ObjectSnapshotReadFaultTarget::nth_command(ObjectSnapshotReader::FencedFanout, 0),
        FAN_OUT_DELAY,
    );
    client.install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));
    let reading = Capacity {
        in_flight: ONE_FENCED,
        p_f_held: 1,
        reserved_bytes: expected_grant_bytes(&client, estimate.bytes),
        overcommit_bytes: 0,
    };
    let shutdown = CancellationToken::new();
    let audit =
        StructuralCacheAudit::new(1.0, true, Arc::new(UnboundedAuditCharges), shutdown.clone());

    let handle = audit
        .maybe_spawn(&client, fx.tv.key_schema(), &fx.cache, &entry)
        .expect("the audit starts");
    let deadline = Instant::now() + AUDIT_BUDGET;
    while lease_fields(&fx.tv, fx.oid).await.is_empty() || injector.delays_remaining() > 0 {
        assert!(
            Instant::now() < deadline,
            "fixture: the audit must hold its lease, its fan-out held back"
        );
        tokio::time::sleep(POLL_PAUSE).await;
    }
    let mid_read = capacity(&client, &fx.cache);
    shutdown.cancel();
    let outcome = finished(handle).await;
    let after = capacity(&client, &fx.cache);

    assert_eq!(
        mid_read, reading,
        "fixture: mid-read, the audit holds its pair, P_f and its grant"
    );
    assert_eq!(outcome, None, "a cancelled audit answers no outcome");
    assert_eq!(
        after, NOTHING_HELD,
        "the cancelled audit gave back its pair, P_f and grant"
    );
    assert_cold_start_permits_free(&fx.cache, "after the audit was cancelled mid fenced read")
        .await;
    wait_until_no_lease(&fx.tv, fx.oid, "the cancelled audit's lease").await;
    assert!(!audit.is_running(), "and the one-audit flag");
    assert_eq!(fx.audits(), (0, 0, 0), "a cancelled audit ticks nothing");
    assert_eq!(
        fx.audit_fenced(),
        (0, 0, 0, 0),
        "nor any fenced result: its read never ended"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 7. Defaults
// ═════════════════════════════════════════════════════════════════════════════

/// A store built without an installed audit never starts one: tests and
/// tools that build a `LayerStore` read nothing they did not ask for.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn layer_store_without_an_installed_audit_never_starts_one() {
    let fx = Fx::new("audit-default").await;
    let entry = fx.entry().await;

    let started = fx.ls.maybe_spawn_structural_cache_audit(&entry);

    assert_eq!(fx.ls.structural_cache_audit().rate(), 0.0);
    assert!(started.is_none(), "rate 0 never samples");
    assert_eq!(fx.audits(), (0, 0, 0));
}

/// The audit's constructor refuses a rate its sampler cannot take, as
/// `Config::from_env` does at startup.
#[test]
fn structural_cache_audit_new_rejects_a_rate_outside_zero_to_one() {
    for rate in [1.5, -0.1, f64::NAN, f64::INFINITY] {
        let built = std::panic::catch_unwind(|| {
            StructuralCacheAudit::new(
                rate,
                false,
                Arc::new(UnboundedAuditCharges),
                CancellationToken::new(),
            )
        });
        assert!(built.is_err(), "rate {rate} must be refused");
    }
    for rate in [0.0, 1.0] {
        assert_eq!(
            StructuralCacheAudit::new(
                rate,
                false,
                Arc::new(UnboundedAuditCharges),
                CancellationToken::new()
            )
            .rate(),
            rate
        );
    }
}
