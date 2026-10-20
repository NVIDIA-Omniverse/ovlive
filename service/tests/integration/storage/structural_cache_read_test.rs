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

//! The structural cache's read side and stream repair —
//! `storage::structural_cache_read::read_for_structural_edit` over
//! `storage::structural_cache_repair` (b.dkt Epic 4 Task 5; structural-edits
//! SDD SR-4.1, SR-4.2, SR-4.2a, SR-4.3 step 0, SR-3.2, SR-3.4, SR-2.4, SR-5.4).
//!
//! Every object is seeded through the real writers — `create_layer`, then
//! `update_layer` commits on the Lua fast path, the `lua` backend's
//! structural route and the WATCH path — and every commit sends its delta's
//! OMPF as relay bytes, as a production commit does. The seeding store plans
//! its structural commits on a structural cache of its own, which reports into
//! the fixture's registry through the shared client, so a test that seeds
//! structural commits counts the read under test as the change across that
//! read ([`Fx::lookups_since`], [`Fx::reloads_since`]).
//! The repair folds `commit_chain` over the stream payloads and replays the
//! stream's `ompf`; a commit without relay bytes would replay as an empty
//! commit whose fold still matches, so an entry missing its content would
//! pass every check and the equality assertions here would prove nothing.
//! The few anomalies no writer produces (a tombstone, an unparseable or
//! unreplayable entry at a folded header) are hand-written stream entries.
//!
//! The equality oracle is the fan-out reader's state, normalized, and the
//! store's dangling `obj:chs` ids — a reader independent of the atomic script
//! the cold start uses and of the repair.
//!
//! | Area | What it proves |
//! |---|---|
//! | Acceptance | a hit repaired across fast, structural-route (create, move, delete, the C++ client library's create-plus-reorder), WATCH-path (a create), external (relay containers carrying their blobs), empty and reset commits, and a cold start over the same history, each lands on the header's `(seq, commit_chain)` and equals the fresh state |
//! | Chunks | 1,024 pending entries repair in five reads (the first capped at `STRUCTURAL_REPAIR_INITIAL_ENTRIES`, the later ones at `STRUCTURAL_REPAIR_MAX_ENTRIES`, the last at the budget's remainder; five for any first cap from 64 to 255); 1,025 reload `too_far` without replaying; an anomaly past the first chunk of an over-budget repair reports `too_far`; commits landing between reads past the remaining budget reload `too_far` with only the first read replayed; a record between the first read's 256 KiB bound and the 4 MiB hard cap repairs in two reads (the second at the entry's size) with no reload; a chunk its byte bound cut short with more pending doubles the next read's bound, up to the cap, so five records over 256 KiB repair in four reads (one `TOO_LARGE`, then reads at the first entry's size, twice it and four times it) and five of 150 KiB in three (256 KiB, 512 KiB, 1 MiB), each record replayed once; under injected bounds whose cap is not a power-of-two multiple of the first bound, the doubled bound stops at the cap and later reads keep it |
//! | Charge points | with the call's commit charge, each repair read is held first at its entry and byte bounds (RP0: the first read at `STRUCTURAL_REPAIR_INITIAL_ENTRIES` and `STRUCTURAL_REPAIR_INITIAL_BYTES`, the later ones at `STRUCTURAL_REPAIR_MAX_ENTRIES` and the grown read's entry size, kept for the read after it, or twice the bound of a chunk it cut short), each chunk's replay at its entries, its bytes, its largest record's delta part and its delta parts summed (RP1), and each of its records before it applies at the chunk's entries and bytes and its own delta part (RP2), a hit's first point being RP0 at the entry's stored weight; an entry with no OMPF counts no delta bytes there and one whose OMPF does not parse counts it whole; RP0 refused on a hit puts the entry back at its cut, nothing read or replayed, and answers `NeedsCharge` for its target on `structural`; refused after a cold start (CR0 granted), on `structural_full`, the entry stored; RP1 refused on the third of four one-entry rounds puts the entry back at the second round's cut, and the resumed repair replays the last two — the third record priced twice (refused, then replayed), each record replayed once, and an RP2 priced only for a record that applied — the rounds after the first weighing the grown entry above its stored weight, each RP2 at its round's weight |
//! | Reload reasons | one test per reason the repair answers — `regressed` (header `seq` lowered; a non-folding commit; the header `seq` going down between two repair reads), `identity`, `gap` (a trimmed stream; a header `seq` past the stream's last entry, `reached_end` not being contiguity), `tombstone`, `parse` (an unparseable OMPF; a payload naming another `seq` or object), `replay_err`, `too_far`, `too_large` (over an injected hard cap; over the production 4 MiB cap, after the first read), `dangling_adopted` (a parent holding an earlier build's stored list created by another store through `update_layer` — which must not trip the debug consistency check — then a child; an orphan's parent, which holds no stored list, adopted with no reload) — each ticking exactly its label, the reloaded entry equal to the fresh state, its put-back accepted and the next call a hit |
//! | Dangling upkeep | replay adds no dangling parent for a section created under a missing parent (no list is stored there), and removes a dangling parent whose stored list a commit deletes — the parent created and deleted again in one delta ([`create_then_delete_dangling_parent_delta`]) |
//! | Header mapping | absent → `NotFound(EMPTY_HEADER_SENTINEL)`, non-numeric `seq` → `FailedPrecondition(PARTIAL_HEADER_SENTINEL)`, `superseded` → `FailedPrecondition(SUPERSEDED_SENTINEL)`, on a hit and on a miss, each dropping the entry and none a reload; a real `DeleteLayer` through another store answers `NotFound`; a header made partial, superseded or absent between the probe and the cold start is re-classified, never `NotFound` for the first two |
//! | Token before plan | this call's landed token answers `AlreadyLanded` on a hit (and another call's token does not), on a lost-reply retry of a `CreateSection` whose commit is pending, on a commit landing between two repair reads (the entry put back at the first read's cut), on a miss (probe), on a cold start that read the record, on a superseded object (hit, probe, cold start), and on an unmarked object; a wrong-typed record answers `update_layer`'s own refusal byte for byte — on a hit, on a miss, and when the record key is retyped between the probe and the cold start (the loaded entry kept, the next call a hit; with the header superseded in the same moment, still the refusal); each puts back any entry held. With a client operation, on a hit, a probe and a cold start, superseded or not: an `op.<id>` record answers this call's own commit (`AlreadyLanded`), an earlier call's same request (`Recorded(EarlierCall)`) or different request (`Recorded(Conflict)`) ahead of the start record, a moved start record answers `Overtaken` ahead of `superseded`, and a held one goes on to plan (or the superseded refusal) |
//! | Foreign-typed index | the probe reports it as data, so a landed token is still answered from its record, a superseded header still maps to `SUPERSEDED` and an unmarked object (foreign-typed `obj:chs`) is still `NoChsComplete`, each with no full read; on a marked object the cold start runs unprobed and the atomic read's refusal fails it `Internal`, ticking `live_edit_snapshot_read_type_error_total{key_family="fields_index"}` once |
//! | Cold-start permits | with every one held elsewhere, a hit's repair and a miss's probe of an unmarked object answer, while a miss's cold start waits for a plain permit |
//! | Uncacheable | an unmarked object: one probe and no full read, then routed with no command while the negative entry lives; once it expires, probed again (a short TTL, in its own test); a cold start over the weight cap answers `Uncacheable(OverWeight)`, with no size within the plain ceilings; a held entry a repair grew past the per-entry cap, put back by the read side after the second repair read answers `AlreadyLanded` or the wrong-typed-record refusal, is refused `OverCap`, counted on `reloads_total{reason="uncacheable"}` and leaves an `OverWeight` negative entry, reported once as an `over_weight` verdict naming the request's layer, the weight and the cap; the probe's verdict reported once naming the layer, and a call the cached verdict routes reporting nothing |
//! | Above the ceilings (C1-7) | on a client whose plain ceilings the object is over, an uncacheable answer carries the layer's size: a fenced read failed for its call (no time left) carries the probe's estimate `{bound, bound × K}`, records nothing (a lookup is a miss) and the next call caches it fenced; one only predicted over the byte ceiling carries none; the probe's `OverWeight` carries the estimate and its negative entry's lookup none; an admission's `OverWeight` after a fenced read carries the measured size; the probe's `NoChsComplete` on an unmarked object over the item ceiling carries the probe's estimate (none at the ceiling, and none from its negative entry's lookup). `NoChsComplete` within the plain ceilings and `ReloadLimit` carry none (their own tests) |
//! | G4 | stream entries with `client_ts > i64::MAX` replay |
//! | Repair to a target | reaches the target cut, ticks no cache or cold-start metric and leaves the cache entry untouched; a wrong target `commit_chain`, a target above the header and a target below the entry are unreachable (`regressed`); with a charge (the sampled audit's), a refused point answers `ChargeRefused` for its target at a valid cut (at the first read, nothing replayed; at a chunk's second record, the copy normalized at the first's cut), and on a copy without plan indexes every point holds the stored weight plus the growth RP2 priced for the records before it — never their scratch, a cascade's included — and the delta parts replayed with its own, each record's RP2 at the index-less per-record estimate's growth plus scratch |
//! | Own creation | `plan_adopts_dangling_parent` is true exactly for a plan that creates a dangling parent, and not for one creating an orphan's parent that holds no stored list |
//! | Reload cap | a repair failing after every cold start (a non-folding commit each time) reloads up to `STRUCTURAL_READ_MAX_RELOADS_AFTER_COLD_START`, then answers `Uncacheable(ReloadLimit)`, each failure a `regressed` reload and none an uncacheable cold-start verdict, the call's verdict reported once as `reload_limit`; the cap records no negative entry, and the next call caches the object |
//! | Successor entry | the successor `successor_entry` builds from a real `update_layer` structural commit's plan is the stored cut at the commit — dangling parents unchanged (an orphan's creation among them), or shrunk by the plan's delete of a dangling parent's stored list — and the next call's hit; a commit adopting a dangling parent has none (`DanglingAdopted`), and the call after the discard cold-starts the stored cut |
//!
//! **Not covered here**: the read side's re-probe after
//! `ColdStart::Invalidated` (a retirement during the cold start's read; the
//! cold start's own test covers that answer). The empty-`SetChildren` arm of
//! the dangling-parent upkeep is reached only by a unit test in
//! `structural_cache_repair`: a real plan writes a list only under a section
//! it leaves with a row, and one that leaves a dangling parent live answers
//! `dangling_adopted` before the upkeep runs; real commits reach the removal
//! through the `DeleteChildrenList` of a dangling parent created and deleted
//! again in one delta ([`create_then_delete_dangling_parent_delta`]).
//! The repair's `too_far` after a round, once the budget is spent short of
//! the header, is unreachable by design: each round's pre-replay budget
//! check, made against that round's header, answers `too_far` first even
//! when commits land between reads (the between-reads `too_far` test), so
//! that branch is a backstop only.
//!
//! The guard's settling rules are inline in `src/storage/structural_cache.rs`;
//! the cold start's routes in `structural_cache_cold_start_test`; the repair
//! script in `structural_cache_sync_test`; the zero-seeded series in
//! `observability_metrics_test.rs`.
//!
//! Slow tests: the two chunk-boundary tests commit 1,024 and 1,025 entries
//! through `update_layer` (about a second each locally); they need a real
//! stream that long, so they belong here. The negative-entry expiry test
//! waits out a 200 ms TTL.

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use fred::clients::Pool;
use fred::interfaces::{HashesInterface, KeysInterface, StreamsInterface};
use fred::types::streams::XCap;
use futures::FutureExt;
use oodf_ffi::{CommandDiffChangeSection, CommandReorderChildren, OwnedDelta, StructCommand};

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::delta::{apply_delta, MutationAllowance, StorageMutation};
use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::replay::decode_stream_record;
use live_edit::error::ServiceError;
use live_edit::grpc::upload_admission::{
    object_state_charge_for_test, object_state_rewrite_allowance_for_test,
};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::commit_chain::{commit_chain_fold, update_stream_payload};
use live_edit::storage::commit_charge::{ObjectStateNeed, RepairTransient, SharedCommitCharge};
use live_edit::storage::diff_read::{
    read_object_snapshot_slim_with_ceilings, snapshot_item_bound, SlimSnapshotReadConsumer,
    SnapshotCeilings, SnapshotReadOf,
};
use live_edit::storage::layer::{
    operation_request_digest, LayerStore, Operation, RecordedBy, RecordedCommit, StartRecord,
    HEADER_FIELD_COMMIT_CHAIN,
};
use live_edit::storage::notifications::{
    DELETED_STREAM_PAYLOAD, NOTIFICATION_PAYLOAD_FIELD, RELAY_OMPF_FIELD,
};
use live_edit::storage::snapshot_fence::writer::RemainingDeadlineFn;
use live_edit::storage::snapshot_fence::LayerSize;
use live_edit::storage::structural_cache::{
    estimate_replay_record_growth, state_weight_lower_bound, CacheEntry, CacheLookup,
    StructuralCache, UncacheableReason,
};
use live_edit::storage::structural_cache_cold_start::STRUCTURAL_COLD_START_ROUTE_BYTES_PER_ITEM;
use live_edit::storage::structural_cache_read::{
    read_for_structural_edit, AfterColdStartHook, AfterProbeHook, BetweenRepairReadsHook,
    ReadyEntry, StructuralPath, StructuralRead, StructuralReadRequest,
    STRUCTURAL_READ_MAX_RELOADS_AFTER_COLD_START,
};
use live_edit::storage::structural_cache_repair::{
    plan_adopts_dangling_parent, repair_to_target, successor_entry, ReloadReason, RepairBounds,
    TargetRepair, STRUCTURAL_REPAIR_MAX_TOTAL,
};
use live_edit::storage::structural_cache_sync::{
    read_structural_cache_sync, StructuralSyncRequest, STRUCTURAL_REPAIR_INITIAL_BYTES,
    STRUCTURAL_REPAIR_INITIAL_ENTRIES, STRUCTURAL_REPAIR_MAX_BYTES, STRUCTURAL_REPAIR_MAX_ENTRIES,
};
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::{
    commit_without_fold, rematerialize_same_seq, write_commit_token_record, TestValkey,
};

use crate::common::owned_delta::{
    delta_part_len, delta_to_ompf_bytes, delta_with_diff_change_section,
    delta_with_new_field_names, delta_with_set_field_external, delta_with_set_field_inline,
    delta_with_time_sample_external, delta_with_time_sample_inline, empty_delta, reset_delta,
};
use crate::grpc::log_capture::{capture, capture_debug};
use crate::storage::snapshot_read_atomic_test::{
    create, create_cmd, create_session, delete_cmd, every_shape_state, ext_hash, layer_store,
    move_cmd, structural, unmark, CUBE, DANGLING_PARENT, ROOT, SIBLING, WORLD,
};
use crate::storage::snapshot_read_support::{
    client_with_limits, monitor_fence, test_valkey_addr, MonitorTap, MonitoredCommand,
};
use crate::storage::structural_cache_cold_start_test::{
    assert_one_verdict_line, assert_one_verdict_line_at, header, normalized_fresh_state,
    stored_dangling_parents, uncacheable_verdicts, CEILING_FIELDS, LAYER_FIELDS,
    UNCACHEABLE_VERDICT_MSG, WEIGHT_FIELDS,
};

/// A budget no fixture entry comes near.
const BUDGET_BYTES: u64 = 64 * 1024 * 1024;
/// A budget whose per-entry cap (a quarter of it) every fixture object is
/// over.
const TINY_BUDGET_BYTES: u64 = 1_024;
/// The `client_id` every seeded commit is made under: never the structural
/// call's, so no commit-token record of the call exists unless a test writes
/// one.
const WRITER: &str = "structural-read-writer";
/// The structural call's `client_id`.
const CLIENT_ID: &str = "structural-read-client";
/// The structural call's commit token (32 lowercase hex digits, as
/// `update_layer` mints them).
const TOKEN: &str = "0123456789abcdef0123456789abcdef";
/// Another call's commit token.
const OTHER_TOKEN: &str = "fedcba9876543210fedcba9876543210";
const CLIENT_TS: u64 = 7_000;
/// A `client_ts` above `i64::MAX` (SR-5.4, G4). `u64::MAX - 1` so an
/// off-by-one saturation cannot pass for a round trip.
const CLIENT_TS_ABOVE_I64_MAX: u64 = u64::MAX - 1;

/// The header's `seq` column (`HEADER_FIELD_SEQ` is crate-private).
const SEQ_COLUMN: &str = "seq";
/// The header column a `CreateLayer` overwrite sets; no public constant.
const SUPERSEDED_COLUMN: &str = "superseded";

/// The commit paths' header-mapping messages (SDD §7.5), byte for byte.
const EMPTY_HEADER_MESSAGE: &str = "object no longer exists (header deleted)";
const PARTIAL_HEADER_MESSAGE: &str = "object header partial (seq missing or malformed)";
const SUPERSEDED_MESSAGE: &str = "object superseded by a concurrent CreateLayer overwrite";

/// A section id the seeded tree does not use.
const NEW_SECTION: u64 = 900;
/// A parent id naming no section, under which [`orphan_delta`] creates
/// [`ORPHAN`].
const ORPHAN_PARENT: u64 = 500;
/// A section created under the missing [`ORPHAN_PARENT`].
const ORPHAN: u64 = 501;
/// A child created under [`ORPHAN_PARENT`] once it is live.
const ADOPTED_CHILD: u64 = 502;
/// The inline value of a commit whose stream entry is larger than
/// [`SMALL_BYTE_BOUND`]; every other commit's entry is far smaller.
const LARGE_VALUE_BYTES: usize = 16 * 1024;
/// An injected repair byte bound between the ordinary entries and the large
/// one.
const SMALL_BYTE_BOUND: u64 = 8 * 1024;
/// The rows of new field names [`growth_delta`] sets: enough rows and
/// interned names to grow a compact entry past a cap a few hundred bytes
/// above its weight.
const GROWTH_FIELDS: usize = 64;

/// A field-only delta setting [`GROWTH_FIELDS`] one-byte inline fields of
/// distinct new names on `WORLD` ([`delta_with_new_field_names`]): it grows a
/// compact entry by a row and an interned name each, where a large inline
/// value would grow it by nothing.
fn growth_delta() -> OwnedDelta {
    delta_with_new_field_names(WORLD, "primvars:growthAttribute", GROWTH_FIELDS)
}
/// The negative-entry TTL of the `NoChsComplete` expiry test's cache: short,
/// so that test waits a fraction of a second. The routing test, which needs
/// the call after the verdict inside the TTL, uses [`LONG_TTL`] instead, so
/// no TTL is raced.
const SHORT_NEGATIVE_TTL: Duration = Duration::from_millis(200);
/// A TTL no test comes near.
const LONG_TTL: Duration = Duration::from_secs(3_600);
/// Bound on waiting for the negative entry to expire; a correct cache gets
/// there in [`SHORT_NEGATIVE_TTL`] plus a poll. Well inside nextest's 120 s
/// hang-killer.
const WAIT_BUDGET: Duration = Duration::from_secs(20);
/// Poll interval while waiting for the negative entry to expire.
const EXPIRY_POLL: Duration = Duration::from_millis(20);
/// Bound on the capped-reload call: its healthy path is a few round trips,
/// and without the cap it would reload until this budget. Well inside
/// nextest's 120 s hang-killer.
const RELOAD_CAP_BUDGET: Duration = Duration::from_secs(30);

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// One object, a cache and a metrics registry hooked onto the store client.
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
    /// [`Self::with_cache`] with a cache of [`BUDGET_BYTES`].
    async fn new(layer: &str) -> Self {
        Self::with_cache(layer, StructuralCache::new(BUDGET_BYTES)).await
    }

    /// A marked `every_shape_state` object — which keeps a list under
    /// `DANGLING_PARENT`, so its entries carry a dangling parent — after one
    /// fast-path commit, so its header `seq` and `commit_chain` are a
    /// commit's.
    async fn with_cache(layer: &str, cache: StructuralCache) -> Self {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only(layer));
        tv.client()
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let ls = layer_store(&tv);
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

    fn wtok_key(&self) -> String {
        self.tv.key_schema().obj_write_token(&self.oid.to_string())
    }

    /// Commit `delta` as [`WRITER`] through this fixture's store.
    async fn commit(&self, delta: OwnedDelta) -> (u64, u64) {
        self.commit_through(&self.ls, WRITER, Some(delta), CLIENT_TS)
            .await
    }

    /// Commit `delta` through `ls` as `client_id`, sending the delta's OMPF as
    /// relay bytes (none for an empty `None` commit). Returns `(seq,
    /// server_ts)`.
    async fn commit_through(
        &self,
        ls: &LayerStore,
        client_id: &str,
        delta: Option<OwnedDelta>,
        client_ts: u64,
    ) -> (u64, u64) {
        commit_to(
            ls,
            &self.session_id,
            &self.layer,
            self.oid,
            client_id,
            delta,
            &[],
            client_ts,
        )
        .await
    }

    /// Commit `delta`, which names one external value at `ext_value_index`
    /// 1, as [`WRITER`], with `blob` as that value's part of the relay OMPF —
    /// the container a client sends.
    async fn commit_external(&self, delta: OwnedDelta, blob: &[u8]) -> (u64, u64) {
        commit_to(
            &self.ls,
            &self.session_id,
            &self.layer,
            self.oid,
            WRITER,
            Some(delta),
            &[blob],
            CLIENT_TS,
        )
        .await
    }

    /// `n` small fast-path field writes over eight keys.
    async fn commit_fast(&self, n: u64) {
        for i in 0..n {
            self.commit(delta_with_set_field_inline(
                WORLD,
                &format!("k{}", i % 8),
                i.to_le_bytes().to_vec(),
                i + 10,
            ))
            .await;
        }
    }

    async fn header_seq(&self) -> u64 {
        header(&self.tv, self.oid).await[SEQ_COLUMN]
            .parse()
            .expect("the header seq is decimal")
    }

    async fn set_header(&self, column: &str, value: &str) {
        let _: () = self
            .tv
            .client()
            .pool
            .hset(self.header_key(), [(column, value)])
            .await
            .expect("HSET obj:hdr must succeed");
    }

    /// The call under test, for `token`, under `bounds`.
    async fn read_bounded(
        &self,
        token: &str,
        bounds: RepairBounds,
    ) -> Result<StructuralRead, ServiceError> {
        self.read_request(
            &StructuralReadRequest::new(self.oid, CLIENT_ID, token).with_repair_bounds(bounds),
        )
        .await
    }

    /// The call under test, for [`TOKEN`], under the production bounds.
    async fn read(&self) -> Result<StructuralRead, ServiceError> {
        self.read_bounded(TOKEN, RepairBounds::PRODUCTION).await
    }

    async fn read_request(
        &self,
        request: &StructuralReadRequest<'_>,
    ) -> Result<StructuralRead, ServiceError> {
        read_for_structural_edit(self.tv.client(), self.tv.key_schema(), &self.cache, request).await
    }

    /// Cold-start the object into the cache through the entry point and put
    /// it back: the entry every hit test starts from. Returns a copy of it.
    async fn warm(&self) -> CacheEntry {
        let ready = expect_ready(self.read().await, "warm-up");
        assert_eq!(
            ready.path,
            StructuralPath::StructuralFull,
            "fixture: the warm-up is a cold start"
        );
        self.assert_current(ready.entry.entry(), "warm-up").await;
        let held = ready.entry.entry().clone();
        assert!(
            ready.entry.put_back_held().is_stored(),
            "fixture: the warm-up entry is stored"
        );
        held
    }

    /// Assert `entry` is the object's stored cut now: the header's `(seq,
    /// commit_chain)`, the store's dangling ids and the normalized fresh
    /// state.
    async fn assert_current(&self, entry: &CacheEntry, context: &str) {
        let hdr = header(&self.tv, self.oid).await;
        let seq: u64 = hdr[SEQ_COLUMN].parse().expect("the header seq is decimal");
        let commit_chain = hdr
            .get(HEADER_FIELD_COMMIT_CHAIN)
            .cloned()
            .unwrap_or_default();
        assert_eq!(entry.object_id, self.oid, "{context}: object_id");
        assert_eq!(
            (entry.seq, &entry.commit_chain),
            (seq, &commit_chain),
            "{context}: the entry must stand at the header's (seq, commit_chain)"
        );
        assert_eq!(
            entry.dangling_parents,
            stored_dangling_parents(&self.tv, self.oid).await,
            "{context}: dangling_parents must be the store's dangling obj:chs ids"
        );
        assert_eq!(
            entry.state,
            normalized_fresh_state(&self.tv, self.oid).await,
            "{context}: the entry must equal the normalized fresh state's compact projection"
        );
    }

    /// Append a stream entry at the header's next `seq` by hand and advance
    /// the header to it, folding `commit_chain` over `payload` exactly as a
    /// real commit does — so the repair's fold check passes and the check
    /// under test is the one that answers. `payload` is given the new `seq`.
    async fn append_folded_entry(
        &self,
        payload: impl FnOnce(u64) -> String,
        ompf: Option<Vec<u8>>,
    ) -> u64 {
        let hdr = header(&self.tv, self.oid).await;
        let seq: u64 = hdr[SEQ_COLUMN].parse().expect("the header seq is decimal");
        let new_seq = seq + 1;
        let payload = payload(new_seq);
        let chain = commit_chain_fold(&hdr[HEADER_FIELD_COMMIT_CHAIN], &payload);
        let mut fields: Vec<(&str, fred::types::Value)> =
            vec![(NOTIFICATION_PAYLOAD_FIELD, payload.as_str().into())];
        if let Some(ompf) = ompf {
            fields.push((RELAY_OMPF_FIELD, fred::types::Value::Bytes(ompf.into())));
        }
        let _: () = self
            .tv
            .client()
            .pool
            .xadd(
                self.tv.key_schema().layer_stream(self.oid),
                false,
                None::<()>,
                format!("{new_seq}-0").as_str(),
                fields,
            )
            .await
            .expect("XADD must succeed");
        let _: () = self
            .tv
            .client()
            .pool
            .hset(
                self.header_key(),
                [
                    (SEQ_COLUMN, new_seq.to_string()),
                    (HEADER_FIELD_COMMIT_CHAIN, chain),
                ],
            )
            .await
            .expect("HSET obj:hdr must succeed");
        new_seq
    }

    /// An `update` payload of this object at `seq`, as a real commit writes
    /// it.
    fn update_payload(&self, seq: u64) -> String {
        update_stream_payload(self.oid, seq, WRITER, CLIENT_TS, 1_700_000_000_000_000)
    }

    /// The nonzero cells of `name`, by `label`.
    fn ticks(&self, name: &str, label: &str) -> BTreeMap<String, u64> {
        nonzero_cells(&self.metrics, name, label)
    }

    fn lookups(&self) -> BTreeMap<String, u64> {
        self.ticks("live_edit_structural_cache_lookups_total", "outcome")
    }

    fn reloads(&self) -> BTreeMap<String, u64> {
        self.ticks("live_edit_structural_cache_reloads_total", "reason")
    }

    /// [`Self::lookups`] less `before`, an earlier reading: the lookups since.
    fn lookups_since(&self, before: &BTreeMap<String, u64>) -> BTreeMap<String, u64> {
        cells_since(self.lookups(), before)
    }

    /// [`Self::reloads`] less `before`, an earlier reading: the reloads since.
    fn reloads_since(&self, before: &BTreeMap<String, u64>) -> BTreeMap<String, u64> {
        cells_since(self.reloads(), before)
    }

    /// Every `live_edit_snapshot_read_total{consumer="structural_cache"}`
    /// cell, summed: the cold start's full reads.
    fn full_reads(&self) -> u64 {
        render(&self.metrics)
            .lines()
            .filter(|l| {
                l.starts_with("live_edit_snapshot_read_total{")
                    && l.contains("consumer=\"structural_cache\"")
            })
            .filter_map(|l| l.rsplit_once(' ')?.1.parse::<f64>().ok())
            .sum::<f64>() as u64
    }

    /// `live_edit_structural_cold_start_reads_total{route}`.
    fn cold_starts(&self, route: &str) -> u64 {
        sample_value(
            &render(&self.metrics),
            "live_edit_structural_cold_start_reads_total",
            &[("route", route)],
        )
        .unwrap_or_default() as u64
    }

    /// `(count, sum)` of the repair histogram `name`; zeros before its first
    /// observation.
    fn repair_histogram(&self, name: &str) -> (u64, u64) {
        let text = render(&self.metrics);
        let read = |suffix: &str| {
            sample_value(&text, &format!("{name}_{suffix}"), &[]).unwrap_or_default() as u64
        };
        (read("count"), read("sum"))
    }

    /// Whether the cache holds an entry for the object (a `Hit`, put back at
    /// once), with the entry.
    fn cached(&self) -> Option<CacheEntry> {
        match self.cache.lookup(self.oid) {
            CacheLookup::Hit(guard) => Some(guard.entry().clone()),
            CacheLookup::Miss => None,
            other => panic!("expected a hit or a miss; got {other:?}"),
        }
    }
}

/// Commit `delta` to `oid` through `ls` as `client_id`, sending the delta's
/// OMPF — with `external_values` as its external-value parts — as relay
/// bytes (none for an empty `None` commit). Returns `(seq, server_ts)`.
/// Shared with `structural_cache_audit_test`.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn commit_to(
    ls: &LayerStore,
    session_id: &str,
    layer: &str,
    oid: u64,
    client_id: &str,
    delta: Option<OwnedDelta>,
    external_values: &[&[u8]],
    client_ts: u64,
) -> (u64, u64) {
    let relay = delta
        .as_ref()
        .map(|d| Bytes::from(delta_to_ompf_bytes(d, external_values)));
    ls.update_layer(session_id, layer, oid, delta, relay, client_id, client_ts)
        .await
        .unwrap_or_else(|e| panic!("fixture commit must land: {e:?}"))
}

/// A structural delta creating [`ORPHAN`] under the missing
/// [`ORPHAN_PARENT`]. `CreateSection` ignores a missing parent (SDD §9.1.4c),
/// so the orphan is made the way a client makes one: a non-reset
/// `DiffChangeSection`, which creates the section and no list (SDD §7.7: no
/// list exists under a parent with no row).
fn orphan_delta() -> OwnedDelta {
    structural(vec![StructCommand::DiffChangeSection(
        CommandDiffChangeSection {
            parent_id: ORPHAN_PARENT,
            section_id: ORPHAN,
            section_name: "Orphan".to_owned(),
            section_type: 1,
            section_order: 0,
        },
    )])
}

/// A structural delta that creates `DANGLING_PARENT` — adopting the fixture's
/// orphan, the row naming it — and deletes it again, which cascades the
/// orphan away with it. The fixture stores an earlier build's list under
/// `DANGLING_PARENT` (`every_shape_state`); the plan holds the list the
/// creation built, so the cascade's delete of that list (a
/// `DeleteChildrenList`) deletes the stored one and the parent is no longer
/// dangling. Since no list is stored under a parent with no row, a commit
/// removes a dangling parent only this way: by creating it — which drops a
/// structural call's own entry (`DanglingAdopted`) unless the same delta
/// deletes it again.
fn create_then_delete_dangling_parent_delta() -> OwnedDelta {
    structural(vec![
        create_cmd(ROOT, DANGLING_PARENT, "Transient"),
        delete_cmd(ROOT, DANGLING_PARENT),
    ])
}

/// The nonzero cells of the counter `name` in `metrics`, keyed by `label`.
fn nonzero_cells(metrics: &Metrics, name: &str, label: &str) -> BTreeMap<String, u64> {
    let prefix = format!("{name}{{");
    let needle = format!("{label}=\"");
    render(metrics)
        .lines()
        .filter(|l| l.starts_with(&prefix))
        .filter_map(|l| {
            let (series, value) = l.rsplit_once(' ')?;
            let start = series.find(&needle)? + needle.len();
            let len = series[start..].find('"')?;
            let value = value.parse::<f64>().ok()? as u64;
            (value != 0).then(|| (series[start..start + len].to_owned(), value))
        })
        .collect()
}

fn counts<const N: usize>(cells: [(&str, u64); N]) -> BTreeMap<String, u64> {
    cells.into_iter().map(|(k, v)| (k.to_owned(), v)).collect()
}

/// The nonzero differences `after - before`, cell by cell. A counter never
/// goes down, so a cell below its earlier reading is a fixture bug.
fn cells_since(
    after: BTreeMap<String, u64>,
    before: &BTreeMap<String, u64>,
) -> BTreeMap<String, u64> {
    after
        .into_iter()
        .filter_map(|(cell, value)| {
            let earlier = before.get(&cell).copied().unwrap_or(0);
            let since = value
                .checked_sub(earlier)
                .unwrap_or_else(|| panic!("counter cell {cell} went down: {earlier} → {value}"));
            (since != 0).then_some((cell, since))
        })
        .collect()
}

fn expect_ready(answer: Result<StructuralRead, ServiceError>, context: &str) -> ReadyEntry {
    match answer {
        Ok(StructuralRead::Ready(ready)) => ready,
        other => panic!("{context}: the entry point must answer Ready; got {other:?}"),
    }
}

fn expect_already_landed(
    answer: Result<StructuralRead, ServiceError>,
    context: &str,
) -> (u64, u64) {
    match answer {
        Ok(StructuralRead::AlreadyLanded { seq, server_ts }) => (seq, server_ts),
        other => panic!("{context}: the entry point must answer AlreadyLanded; got {other:?}"),
    }
}

/// Assert `answer` is the header mapping's error for `change`.
fn assert_header_mapped(
    answer: Result<StructuralRead, ServiceError>,
    change: HeaderChange,
    context: &str,
) {
    match (change, &answer) {
        (HeaderChange::Absent, Err(ServiceError::NotFound(message)))
            if message == EMPTY_HEADER_MESSAGE => {}
        (HeaderChange::Partial, Err(ServiceError::FailedPrecondition(message)))
            if message == PARTIAL_HEADER_MESSAGE => {}
        (HeaderChange::Superseded, Err(ServiceError::FailedPrecondition(message)))
            if message == SUPERSEDED_MESSAGE => {}
        _ => panic!(
            "{context}: {change:?} must map to {}; got {answer:?}",
            change.expected()
        ),
    }
}

/// A header change the header mapping classifies.
#[derive(Debug, Clone, Copy)]
enum HeaderChange {
    /// The header deleted.
    Absent,
    /// A non-numeric `seq`.
    Partial,
    /// `superseded` set, as a `CreateLayer` overwrite does.
    Superseded,
}

impl HeaderChange {
    const ALL: [Self; 3] = [Self::Absent, Self::Partial, Self::Superseded];

    fn expected(self) -> &'static str {
        match self {
            Self::Absent => "NotFound(EMPTY_HEADER_SENTINEL)",
            Self::Partial => "FailedPrecondition(PARTIAL_HEADER_SENTINEL)",
            Self::Superseded => "FailedPrecondition(SUPERSEDED_SENTINEL)",
        }
    }

    async fn apply(self, pool: &Pool, header_key: &str) {
        match self {
            Self::Absent => {
                let _: i64 = pool.del(header_key).await.expect("DEL obj:hdr");
            }
            Self::Partial => {
                let _: () = pool
                    .hset(header_key, [(SEQ_COLUMN, "not-a-seq")])
                    .await
                    .expect("HSET obj:hdr seq");
            }
            Self::Superseded => {
                let _: () = pool
                    .hset(header_key, [(SUPERSEDED_COLUMN, "1")])
                    .await
                    .expect("HSET obj:hdr superseded");
            }
        }
    }
}

/// A hook, run between the probe and the cold start, that applies `change`
/// to the header and, with `record`, writes this call's commit-token record
/// (`<token>:<seq>:<server_ts>`, the format both commit backends write).
fn after_probe(
    fx: &Fx,
    change: Option<HeaderChange>,
    record: Option<(u64, u64)>,
) -> AfterProbeHook {
    let pool = fx.tv.client().pool.clone();
    let (header_key, wtok_key) = (fx.header_key(), fx.wtok_key());
    Arc::new(move || {
        let (pool, header_key, wtok_key) = (pool.clone(), header_key.clone(), wtok_key.clone());
        async move {
            if let Some(change) = change {
                change.apply(&pool, &header_key).await;
            }
            if let Some((seq, server_ts)) = record {
                let _: () = pool
                    .hset(
                        &wtok_key,
                        [(CLIENT_ID, format!("{TOKEN}:{seq}:{server_ts}"))],
                    )
                    .await
                    .expect("HSET obj:wtok");
            }
        }
        .boxed()
    })
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. Acceptance: the repaired entry is the stored cut
// ═════════════════════════════════════════════════════════════════════════════

/// The C++ client library's create shape: a `CreateSection` and a
/// `ReorderChildren` of its parent in one delta.
fn create_and_reorder(section_id: u64) -> OwnedDelta {
    structural(vec![
        create_cmd(ROOT, section_id, "Reordered"),
        StructCommand::ReorderChildren(CommandReorderChildren {
            section_id: ROOT,
            children_list_id: 0,
            children_list: vec![section_id, SIBLING, WORLD],
        }),
    ])
}

/// Every commit shape the repair replays: fast inline and external fields
/// and time samples, structural creates, a move, a delete, the C++ client
/// library's create-plus-reorder, empty commits with and without content bytes, and a
/// reset followed by new content (the `DiffChangeSection` helper's delta is
/// itself a reset). The structural shapes commit through the `lua` backend's
/// structural route, and one more create through the WATCH path (a store on
/// `lua-watch-structural`), so the repair replays both writers' entries. The
/// external commits' relay containers carry their blob parts, as a client's
/// do; the hashes are fixture tags, which nothing on the commit or replay path
/// resolves against the blobs.
async fn commit_every_shape(fx: &Fx) {
    let watch_path = LayerStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::LuaWatchStructural);
    fx.commit(delta_with_set_field_inline(
        WORLD,
        "color",
        vec![1, 2, 3],
        20,
    ))
    .await;
    fx.commit_external(
        delta_with_set_field_external(CUBE, "mesh", ext_hash(0xEE).to_vec(), 9_000, 1, 21),
        &[0xEE; 9_000],
    )
    .await;
    fx.commit(delta_with_time_sample_inline(WORLD, 2.5, vec![9]))
        .await;
    fx.commit_external(
        delta_with_time_sample_external(CUBE, 3.5, ext_hash(0x77).to_vec(), 1_000, 1),
        &[0x77; 1_000],
    )
    .await;
    fx.commit(structural(vec![create_cmd(WORLD, NEW_SECTION, "New")]))
        .await;
    fx.commit(structural(vec![move_cmd(
        NEW_SECTION,
        WORLD,
        SIBLING,
        "Moved",
    )]))
    .await;
    fx.commit(create_and_reorder(NEW_SECTION + 1)).await;
    fx.commit_through(
        &watch_path,
        WRITER,
        Some(structural(vec![create_cmd(
            SIBLING,
            NEW_SECTION + 3,
            "ByWatch",
        )])),
        CLIENT_TS,
    )
    .await;
    fx.commit(structural(vec![delete_cmd(WORLD, CUBE)])).await;
    fx.commit(empty_delta()).await;
    fx.commit_through(&fx.ls, WRITER, None, CLIENT_TS).await;
    fx.commit(reset_delta("usda")).await;
    fx.commit(delta_with_diff_change_section(
        ROOT,
        NEW_SECTION + 2,
        "AfterReset",
        1,
        0,
    ))
    .await;
    fx.commit(delta_with_set_field_inline(
        NEW_SECTION + 2,
        "after",
        vec![4],
        30,
    ))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_hit_repairs_every_commit_shape_to_the_stored_cut() {
    let fx = Fx::new("scr-accept-hit").await;
    fx.warm().await;
    let before = fx.header_seq().await;

    commit_every_shape(&fx).await;
    let pending = fx.header_seq().await - before;
    let (lookups_before, reloads_before) = (fx.lookups(), fx.reloads());
    let ready = expect_ready(fx.read().await, "a hit behind every commit shape");

    assert_eq!(
        (ready.path, ready.repair.replayed_entries),
        (StructuralPath::Structural, pending),
        "a hit repairs by replaying each pending entry"
    );
    fx.assert_current(ready.entry.entry(), "repaired hit").await;
    assert!(ready.entry.put_back_held().is_stored());
    assert_eq!(fx.lookups_since(&lookups_before), counts([("repaired", 1)]));
    assert_eq!(
        fx.reloads_since(&reloads_before),
        counts([]),
        "no anomaly, no reload"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_cold_start_after_every_commit_shape_is_the_stored_cut() {
    let fx = Fx::new("scr-accept-cold").await;
    commit_every_shape(&fx).await;
    let lookups_before = fx.lookups();

    let ready = expect_ready(fx.read().await, "a cold start");

    assert_eq!(ready.path, StructuralPath::StructuralFull);
    fx.assert_current(ready.entry.entry(), "cold start").await;
    assert_eq!(fx.lookups_since(&lookups_before), counts([("cold", 1)]));
}

/// The guard's own `Drop` puts a repaired entry back, as `put_back_held`
/// does: the next call is a hit with nothing to replay.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_ready_entry_dropped_unsettled_is_put_back_repaired() {
    let fx = Fx::new("scr-drop").await;
    fx.warm().await;
    fx.commit_fast(3).await;

    let ready = expect_ready(fx.read().await, "repaired");
    assert_eq!(ready.repair.replayed_entries, 3, "fixture: a repair ran");
    drop(ready);

    let again = expect_ready(fx.read().await, "after the drop");
    assert_eq!(
        (again.path, again.repair.replayed_entries),
        (StructuralPath::Structural, 0),
        "the dropped guard put the repaired entry back"
    );
    fx.assert_current(again.entry.entry(), "after the drop")
        .await;
}

/// Only the cold start's whole-layer reads take the cold-start permits: with
/// every one held elsewhere (the one-at-a-time permit and both plain
/// permits), a hit's repair answers, and so does a miss's probe of an
/// unmarked object, which answers from the probe alone. The control — a miss
/// of the marked object, whose cold start stays parked until a plain permit
/// is released — shows the held permits do block the cold start.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_hit_and_probe_take_no_cold_start_permit() {
    /// How long the control watches the parked cold start; a cold start
    /// that took no permit would finish well inside it.
    const BLOCKED_WINDOW: Duration = Duration::from_millis(500);
    let fx = Fx::new("scr-permits").await;
    fx.warm().await;
    fx.commit_fast(3).await;
    let raised = fx.cache.acquire_raised_cold_start_read_permits().await;
    let plain = fx.cache.acquire_plain_cold_start_read_permit().await;

    let hit = tokio::time::timeout(WAIT_BUDGET, fx.read())
        .await
        .expect("a hit's repair takes no cold-start permit, so it answers while all are held");
    let hit = expect_ready(hit, "hit");
    assert_eq!(
        (hit.path, hit.repair.replayed_entries),
        (StructuralPath::Structural, 3),
        "a hit repaired by replay"
    );
    assert!(hit.entry.put_back_held().is_stored());

    fx.cache.invalidate(fx.oid);
    let mut cold = Box::pin(fx.read());
    let while_held = tokio::time::timeout(BLOCKED_WINDOW, &mut cold).await;
    assert!(
        while_held.is_err(),
        "control: a miss's cold start must wait for a plain permit; got {while_held:?}"
    );
    drop(plain);
    let loaded = tokio::time::timeout(WAIT_BUDGET, cold)
        .await
        .expect("control: the cold start loads once a plain permit is free");
    let loaded = expect_ready(loaded, "control");
    assert_eq!(loaded.path, StructuralPath::StructuralFull, "control");
    assert!(loaded.entry.put_back_held().is_stored());
    let plain = fx.cache.acquire_plain_cold_start_read_permit().await;

    unmark(&fx.tv, fx.oid).await;
    fx.cache.invalidate(fx.oid);
    let probed = tokio::time::timeout(WAIT_BUDGET, fx.read())
        .await
        .expect("the probe takes no cold-start permit, so it answers while all are held");
    assert_no_chs_complete(probed, "a miss of an unmarked object");
    drop((raised, plain));
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. Chunks and the repair budget
// ═════════════════════════════════════════════════════════════════════════════

/// The repair's whole budget of pending entries (1,024) repairs with no
/// reload, in reads of at most the read's entry bound: the first read asks
/// only `STRUCTURAL_REPAIR_INITIAL_ENTRIES` (`E`, derived by the first-read rule so its reply is
/// priced within 1 MiB), every later one the full
/// `STRUCTURAL_REPAIR_MAX_ENTRIES`, the last the budget's remainder:
/// 1 + ⌈(1,024 − E) ÷ 256⌉ reads, five for any `E` from 64 to 255
/// (⌈769 ÷ 256⌉ = ⌈960 ÷ 256⌉ = 4). The test asserts that count and the
/// name's five, not `E` itself: the fitted value's one literal pin is
/// `upload_admission`'s `first_read_bound_is_the_rules_step`.
/// Each record here is a small fast-path write, so no chunk stops at its byte
/// bound and every read's size is its entry bound.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_hit_1024_entries_behind_repairs_in_five_chunks() {
    let fx = Fx::new("scr-chunks").await;
    fx.warm().await;
    let (warm_count, _) = fx.repair_histogram("live_edit_structural_cache_replayed_entries");
    assert_eq!(STRUCTURAL_REPAIR_MAX_TOTAL, 1_024);
    assert_eq!(STRUCTURAL_REPAIR_MAX_ENTRIES, 256);
    let reads = 1
        + (STRUCTURAL_REPAIR_MAX_TOTAL - STRUCTURAL_REPAIR_INITIAL_ENTRIES)
            .div_ceil(STRUCTURAL_REPAIR_MAX_ENTRIES);
    assert_eq!(
        reads, 5,
        "fixture: the name's five reads, which hold for any first-read bound from 64 to 255; \
         STRUCTURAL_REPAIR_INITIAL_ENTRIES is {STRUCTURAL_REPAIR_INITIAL_ENTRIES}"
    );
    let last = STRUCTURAL_REPAIR_MAX_TOTAL
        - STRUCTURAL_REPAIR_INITIAL_ENTRIES
        - (reads - 2) * STRUCTURAL_REPAIR_MAX_ENTRIES;

    fx.commit_fast(STRUCTURAL_REPAIR_MAX_TOTAL).await;
    let ready = expect_ready(fx.read().await, "1,024 behind");

    assert_eq!(
        (
            ready.path,
            u64::from(ready.repair.reads),
            ready.repair.replayed_entries
        ),
        (StructuralPath::Structural, reads, 1_024),
        "1,024 pending entries repair in {reads} reads — {STRUCTURAL_REPAIR_INITIAL_ENTRIES} \
         first, then {} of {STRUCTURAL_REPAIR_MAX_ENTRIES} and the budget's last {last} — no \
         reload",
        reads - 2
    );
    fx.assert_current(ready.entry.entry(), "1,024 replayed")
        .await;
    assert_eq!(fx.lookups(), counts([("cold", 1), ("repaired", 1)]));
    assert_eq!(fx.reloads(), counts([]));
    assert_eq!(
        fx.repair_histogram("live_edit_structural_cache_replayed_entries"),
        (warm_count + 1, 1_024),
        "one observation of 1,024 replayed entries"
    );
    assert_eq!(
        fx.repair_histogram("live_edit_structural_cache_repair_bytes")
            .1,
        ready.repair.bytes,
        "the repair's byte tally is observed (the warm-up's repair read no entry)"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_hit_1025_entries_behind_reloads_too_far_without_replaying() {
    let fx = Fx::new("scr-too-far").await;
    fx.warm().await;

    fx.commit_fast(STRUCTURAL_REPAIR_MAX_TOTAL + 1).await;
    assert_reloads(&fx, "too_far", RepairBounds::PRODUCTION).await;

    assert_eq!(
        fx.repair_histogram("live_edit_structural_cache_replayed_entries")
            .1,
        0,
        "a header beyond the budget answers too_far before any replay"
    );
}

/// The budget is decided from the header alone: a repair whose header is
/// beyond its budget reads one chunk and stops, so an anomaly past that chunk
/// — here an unparseable entry third of five pending, within the budget of
/// four but in the second chunk of two — is reported as `too_far`, not
/// `parse`. Small injected bounds keep the stream short.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_over_budget_with_an_anomaly_past_the_first_chunk_reloads_too_far()
{
    let fx = Fx::new("scr-too-far-anomaly").await;
    fx.warm().await;
    let bounds = RepairBounds {
        max_entries: 2,
        max_total: 4,
        ..RepairBounds::PRODUCTION
    };

    fx.commit_fast(2).await;
    fx.append_folded_entry(|seq| fx.update_payload(seq), Some(b"not an OMPF".to_vec()))
        .await;
    fx.commit_fast(2).await;

    assert_reloads(&fx, "too_far", bounds).await;
}

/// A between-reads hook that runs `act` the first time it is called and
/// does nothing after.
fn once(
    act: impl Fn() -> futures::future::BoxFuture<'static, ()> + Send + Sync + 'static,
) -> BetweenRepairReadsHook {
    let fired = Arc::new(AtomicBool::new(false));
    Arc::new(move || {
        if fired.swap(true, Ordering::SeqCst) {
            async {}.boxed()
        } else {
            act()
        }
    })
}

/// The budget is the remainder after earlier rounds: with three pending,
/// two entries per read and a budget of four, the first read replays two;
/// two commits then land between the reads, so the header is three entries
/// past the entry with two of the budget left, and the second read answers
/// `too_far` before replaying. Two entries replayed in all — a budget taken
/// whole on every round would replay two more first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_commits_between_repair_reads_past_the_remaining_budget_reload_too_far(
) {
    let fx = Fx::new("scr-too-far-between").await;
    fx.warm().await;
    let bounds = RepairBounds {
        max_entries: 2,
        max_total: 4,
        ..RepairBounds::PRODUCTION
    };
    fx.commit_fast(3).await;
    let (client, keys) = (fx.tv.client().clone(), fx.tv.key_schema().clone());
    let (session_id, layer, oid) = (fx.session_id.clone(), fx.layer.clone(), fx.oid);
    let hook = once(move || {
        let (client, keys) = (client.clone(), keys.clone());
        let (session_id, layer) = (session_id.clone(), layer.clone());
        async move {
            let ls = LayerStore::new(client, keys, 4096);
            for i in 0..2_u64 {
                commit_to(
                    &ls,
                    &session_id,
                    &layer,
                    oid,
                    WRITER,
                    Some(delta_with_set_field_inline(
                        WORLD,
                        "between",
                        vec![i as u8],
                        90 + i,
                    )),
                    &[],
                    CLIENT_TS,
                )
                .await;
            }
        }
        .boxed()
    });
    let request = StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN)
        .with_repair_bounds(bounds)
        .with_between_repair_reads_hook_for_test(hook);

    assert_reloads_on(&fx, "too_far", &request, "commits between reads").await;

    assert_eq!(
        fx.repair_histogram("live_edit_structural_cache_replayed_entries")
            .1,
        2,
        "only the first read's two entries were replayed"
    );
}

/// SR-4.1: each round re-checks that the header `seq` has not gone down
/// since the previous round. With two pending and one entry per read, the
/// first read replays one; between the reads the header is rolled back to
/// exactly the entry's new cut — `seq` and `commit_chain` both, so the
/// entry matches the header and only the comparison with the earlier
/// round's header can tell — and the second read reloads `regressed`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_header_seq_going_down_between_repair_reads_reloads_regressed() {
    let fx = Fx::new("scr-regressed-between").await;
    fx.warm().await;
    fx.commit_fast(1).await;
    let first_cut_seq = fx.header_seq().await;
    let first_cut_chain = header(&fx.tv, fx.oid).await[HEADER_FIELD_COMMIT_CHAIN].clone();
    fx.commit_fast(1).await;
    let pool = fx.tv.client().pool.clone();
    let header_key = fx.header_key();
    let hook = once(move || {
        let (pool, header_key, chain) = (pool.clone(), header_key.clone(), first_cut_chain.clone());
        async move {
            let _: () = pool
                .hset(
                    &header_key,
                    [
                        (SEQ_COLUMN, first_cut_seq.to_string()),
                        (HEADER_FIELD_COMMIT_CHAIN, chain),
                    ],
                )
                .await
                .expect("HSET obj:hdr");
        }
        .boxed()
    });
    let request = StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN)
        .with_repair_bounds(RepairBounds {
            max_entries: 1,
            ..RepairBounds::PRODUCTION
        })
        .with_between_repair_reads_hook_for_test(hook);

    assert_reloads_on(&fx, "regressed", &request, "header down between reads").await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 2a. The repair's read size and charge points (RP0, RP1, RP2)
// ═════════════════════════════════════════════════════════════════════════════

/// Which repair charge point a priced need is: RP0 (`Read`), before a read,
/// holds a reply at the read's entry and byte bounds and parses nothing;
/// RP1 (`Replay`), before replaying a chunk, the chunk's reply, its largest
/// record's delta part and its delta parts summed; RP2 (`Record`), before
/// each record of that chunk applies, the chunk's reply and that record's
/// delta part. [`repair_points`] tells them apart by their order, as the
/// repair prices them: a need with a repair transient that parses nothing
/// and follows no RP1 still owed its RP2s is RP0; one that parses is RP1,
/// whose chunk owes one RP2 per entry (`reply_entry_count`); every need
/// after it, while RP2s are owed, is an RP2 — whatever its delta part, so an
/// RP2 of a record with no OMPF (which parses nothing) is never read as an
/// RP0. A need without a repair transient (a cold start's CR0, the plan's
/// CR1) is none and ends any RP2s still owed (a reload ended the chunk). It
/// classifies one call's needs: a refused RP2 ends the call with RP2s still
/// owed, so a resumed call's needs are classified on their own.
/// So it relies on every chunk the tests below replay parsing a non-zero
/// largest delta part: a chunk made only of records with no readable delta
/// part (no OMPF counts nothing) would have its RP1 read as an RP0; the
/// unparseable-OMPF test's chunk stays an RP1 only because the bad OMPF
/// counts its whole length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RepairPoint {
    Read,
    Replay,
    Record,
}

/// The repair charge point each of `needs` is, in order, if any
/// ([`RepairPoint`]). Shared with `commit_charge_test`.
pub(crate) fn repair_points(needs: &[ObjectStateNeed]) -> Vec<Option<RepairPoint>> {
    let mut owed: u64 = 0;
    needs
        .iter()
        .map(|need| match need {
            ObjectStateNeed::Cached { repair, .. } if *repair != RepairTransient::default() => {
                if owed > 0 {
                    owed -= 1;
                    Some(RepairPoint::Record)
                } else if repair.parse_delta_bytes > 0 {
                    owed = repair.reply_entry_count;
                    Some(RepairPoint::Replay)
                } else {
                    Some(RepairPoint::Read)
                }
            }
            _ => {
                owed = 0;
                None
            }
        })
        .collect()
}

/// What a [`RepairCharge`] saw.
#[derive(Default, Debug)]
struct RepairChargeLog {
    held: u64,
    /// Every need priced, in order.
    needs: Vec<ObjectStateNeed>,
    /// Every non-waiting hold: its target and whether it was granted.
    tries: Vec<(u64, bool)>,
}

/// A structural call's commit charge of these tests' own, shared with the
/// read as the call's `CommitChargeCell` is: it prices every need as
/// production does (`object_state_charge_for_test`), grows only where asked
/// to (`try_hold_at_least`), and grants every hold but one: the hold of the
/// `nth` need (from 0) priced at the repair charge point `refuse`, which it
/// refuses once, changing nothing.
struct RepairCharge {
    refuse: Option<(RepairPoint, usize)>,
    log: Mutex<RepairChargeLog>,
}

impl RepairCharge {
    fn granting() -> Self {
        Self {
            refuse: None,
            log: Mutex::new(RepairChargeLog::default()),
        }
    }

    fn refusing(point: RepairPoint, nth: usize) -> Self {
        Self {
            refuse: Some((point, nth)),
            ..Self::granting()
        }
    }

    fn log(&self) -> std::sync::MutexGuard<'_, RepairChargeLog> {
        self.log
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The transient of every repair charge point priced, in order, with its
    /// point.
    fn repair_needs(&self) -> Vec<(RepairPoint, RepairTransient)> {
        let log = self.log();
        repair_points(&log.needs)
            .into_iter()
            .zip(&log.needs)
            .filter_map(|(point, need)| match (point, need) {
                (Some(point), ObjectStateNeed::Cached { repair, .. }) => Some((point, *repair)),
                _ => None,
            })
            .collect()
    }

    /// The transient of every RP2 priced, in order.
    fn record_needs(&self) -> Vec<RepairTransient> {
        self.repair_needs()
            .into_iter()
            .filter(|(point, _)| *point == RepairPoint::Record)
            .map(|(_, repair)| repair)
            .collect()
    }

    /// The held entry's weight every repair charge point priced, in order,
    /// with its point.
    fn repair_weights(&self) -> Vec<(RepairPoint, u64)> {
        let log = self.log();
        repair_points(&log.needs)
            .into_iter()
            .zip(&log.needs)
            .filter_map(|(point, need)| match (point, need) {
                (Some(point), ObjectStateNeed::Cached { held, .. }) => Some((point, *held)),
                _ => None,
            })
            .collect()
    }

    /// The last hold refused, if any: its target.
    fn refused(&self) -> Option<u64> {
        self.log()
            .tries
            .iter()
            .rev()
            .find(|(_, granted)| !granted)
            .map(|(target, _)| *target)
    }
}

impl SharedCommitCharge for RepairCharge {
    fn held(&self) -> u64 {
        self.log().held
    }

    fn target(&self, need: &ObjectStateNeed) -> u64 {
        self.log().needs.push(*need);
        object_state_charge_for_test(need)
    }

    fn rewrite_allowance(&self, need: &ObjectStateNeed) -> MutationAllowance {
        object_state_rewrite_allowance_for_test(self.held(), need)
    }

    fn try_hold(&self, target: u64) -> bool {
        let mut log = self.log();
        // The need this hold is for: the last one priced, and its ordinal
        // among the needs priced at its point.
        let points = repair_points(&log.needs);
        let refused = match (self.refuse, points.last().copied().flatten()) {
            (Some((point, nth)), Some(this)) if this == point => {
                let ordinal = points.iter().filter(|p| **p == Some(point)).count() - 1;
                ordinal == nth && log.tries.iter().all(|(_, granted)| *granted)
            }
            _ => false,
        };
        if !refused {
            log.held = target;
        }
        log.tries.push((target, !refused));
        !refused
    }

    fn try_hold_at_least(&self, target: u64) -> bool {
        target <= self.held() || self.try_hold(target)
    }
}

/// The stream entries of object `oid` after `after` through `through`, as a
/// client `XRANGE` returns them: each one's size as a repair read's byte
/// bound counts it (the id plus every field name and value), and its OMPF's
/// delta part in bytes — `None` with no OMPF or one whose container does not
/// parse. Shared with `structural_cache_audit_test`.
pub(crate) async fn stream_entries(
    tv: &TestValkey,
    oid: u64,
    after: u64,
    through: u64,
) -> Vec<(u64, Option<u64>)> {
    let raw: Vec<(String, std::collections::HashMap<String, Bytes>)> = tv
        .client()
        .pool
        .xrange_values(
            tv.key_schema().layer_stream(oid),
            format!("{}-0", after + 1),
            format!("{through}-0"),
            None,
        )
        .await
        .expect("XRANGE must succeed");
    raw.into_iter()
        .map(|(id, fields)| {
            let size = id.len() as u64
                + fields
                    .iter()
                    .map(|(name, value)| (name.len() + value.len()) as u64)
                    .sum::<u64>();
            let delta = fields.get(RELAY_OMPF_FIELD).and_then(|ompf| {
                OmpfContainer::parse(ompf)
                    .ok()
                    .map(|container| container.delta_bytes.len() as u64)
            });
            (size, delta)
        })
        .collect()
}

/// A record between the repair's first read bound (256 KiB) and its hard cap
/// (4 MiB): its first read answers `TOO_LARGE` with the entry's size, and
/// the repair reads it again at that size — two reads, no reload — and keeps
/// that size for the reads after it: with one entry per read, a small record
/// behind the large one is read at the grown bound. Each chunk stops at its
/// entry count, not its byte bound, so the grown bound is kept rather than
/// doubled (a chunk the byte bound cut short doubles it: the backlog tests
/// below). With the call's charge, each read is held first at its bounds
/// (RP0: one entry each, the first at 256 KiB of bytes, the later ones at the
/// large entry's size), each replay at its chunk's actual entry and bytes and
/// its record's delta part parsed and replayed (RP1), and that record again
/// just before it applies (RP2), at the same reply and delta part.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_hit_behind_a_record_over_the_first_read_bound_reads_again_at_its_size_with_no_reload(
) {
    let fx = Fx::new("scr-grown-read").await;
    fx.warm().await;
    let weight = fx.cached().expect("warmed").weight_bytes();
    let before = fx.header_seq().await;
    let large = delta_with_set_field_inline(WORLD, "large", vec![0xA5; 1024 * 1024], 40);
    let small = delta_with_set_field_inline(WORLD, "small", vec![0x5A; 16], 41);
    let (large_delta, small_delta) = (delta_part_len(&large), delta_part_len(&small));
    fx.commit(large).await;
    fx.commit(small).await;
    let [(entry_bytes, _), (small_bytes, _)] =
        stream_entries(&fx.tv, fx.oid, before, before + 2).await[..]
    else {
        panic!("fixture: two pending entries");
    };
    assert!(
        STRUCTURAL_REPAIR_INITIAL_BYTES < entry_bytes && entry_bytes <= STRUCTURAL_REPAIR_MAX_BYTES,
        "fixture: the large entry ({entry_bytes} B) lies between the first read's bound and the \
         hard cap"
    );
    let reloads_before = fx.reloads();
    let charge = RepairCharge::granting();

    let ready = expect_ready(
        fx.read_request(
            &StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN)
                .with_repair_bounds(RepairBounds {
                    max_entries: 1,
                    ..RepairBounds::PRODUCTION
                })
                .with_commit_charge(&charge),
        )
        .await,
        "a hit behind one large record and a small one",
    );

    assert_eq!(
        (
            ready.path,
            ready.repair.reads,
            ready.repair.replayed_entries,
            ready.repair.bytes
        ),
        (StructuralPath::Structural, 3, 2, entry_bytes + small_bytes),
        "three reads — the first at 256 KiB answered TOO_LARGE, the second at the large entry's \
         size, the third at that size again — and both entries replayed"
    );
    fx.assert_current(ready.entry.entry(), "replayed").await;
    assert_eq!(fx.reloads_since(&reloads_before), counts([]), "no reload");
    // Every read asks one entry: the injected `max_entries` caps the first
    // read's `initial_entries` too.
    let read = |reply_bytes| RepairTransient {
        reply_entry_count: 1,
        reply_bytes,
        ..RepairTransient::default()
    };
    let replay = |reply_bytes, delta| RepairTransient {
        reply_entry_count: 1,
        reply_bytes,
        parse_delta_bytes: delta,
        replay_delta_bytes: delta,
    };
    assert_eq!(
        charge.repair_needs(),
        vec![
            (RepairPoint::Read, read(STRUCTURAL_REPAIR_INITIAL_BYTES)),
            (RepairPoint::Read, read(entry_bytes)),
            (RepairPoint::Replay, replay(entry_bytes, large_delta)),
            (RepairPoint::Record, replay(entry_bytes, large_delta)),
            (RepairPoint::Read, read(entry_bytes)),
            (RepairPoint::Replay, replay(small_bytes, small_delta)),
            (RepairPoint::Record, replay(small_bytes, small_delta)),
        ],
        "RP0 at each read's bounds — the grown byte bound kept for the read after it — then RP1 \
         at the chunk's entry, bytes and record's delta part, and RP2 at the same for its one \
         record"
    );
    assert_eq!(
        charge.log().needs.first(),
        Some(&ObjectStateNeed::cached_repair(
            weight,
            read(STRUCTURAL_REPAIR_INITIAL_BYTES)
        )),
        "a hit's first charge point is RP0: the held entry at its stored weight and the first \
         read's reply"
    );
}

/// Commit one fast-path record per `value_lens` entry behind the warmed
/// entry, each setting one inline field of that many bytes (e.g. a
/// screenshot's size) under a field name of one length, so the records
/// differ in size by their values and the digits of their `seq` alone.
/// Returns each pending stream entry as [`stream_entries`] measures it: its
/// size and its delta part.
async fn commit_backlog(fx: &Fx, value_lens: &[usize]) -> Vec<(u64, u64)> {
    let before = fx.header_seq().await;
    for (i, len) in value_lens.iter().enumerate() {
        fx.commit(delta_with_set_field_inline(
            WORLD,
            &format!("shot{i:02}"),
            vec![0xC3; *len],
            40 + i as u64,
        ))
        .await;
    }
    let pending = stream_entries(&fx.tv, fx.oid, before, before + value_lens.len() as u64).await;
    assert_eq!(
        pending.len(),
        value_lens.len(),
        "fixture: one pending entry per record"
    );
    pending
        .into_iter()
        .map(|(size, delta)| (size, delta.expect("fixture: every record carries its OMPF")))
        .collect()
}

/// A hit behind `pending` (from [`commit_backlog`]) under `bounds`, with the
/// call's charge, must read and replay in exactly `rounds`: each read's byte
/// bound, priced at its RP0, and the number of records its chunk replays
/// (`None` for a read answered `TOO_LARGE`, which replays nothing), the
/// chunks taking the records in order. Each RP0 also prices the read's entry
/// bound, which only the first read caps (`bounds.initial_entries`; every
/// later read asks `bounds.max_entries`), each at most the entries left in
/// the repair's budget. Each chunk is priced at its RP1 at its entries, its
/// records' bytes, its largest delta part and its delta parts summed, then
/// each of its records at its RP2, before it applies, at the chunk's entries
/// and bytes and the record's own delta part. The repair lands on the stored
/// cut, every record replayed once, with no reload. `instead_of` names the
/// reads the rule under test saves, for the failure message.
async fn assert_backlog_repaired_in(
    fx: &Fx,
    pending: &[(u64, u64)],
    bounds: RepairBounds,
    rounds: &[(u64, Option<usize>)],
    instead_of: &str,
) {
    let reloads_before = fx.reloads();
    let charge = RepairCharge::granting();

    let ready = expect_ready(
        fx.read_request(
            &StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN)
                .with_repair_bounds(bounds)
                .with_commit_charge(&charge),
        )
        .await,
        "a hit behind a backlog of records",
    );

    let total: u64 = pending.iter().map(|(size, _)| size).sum();
    assert_eq!(
        (
            ready.path,
            ready.repair.reads,
            ready.repair.replayed_entries,
            ready.repair.bytes
        ),
        (
            StructuralPath::Structural,
            rounds.len() as u32,
            pending.len() as u64,
            total
        ),
        "a hit: {} reads, not {instead_of}, each record replayed once",
        rounds.len()
    );
    fx.assert_current(ready.entry.entry(), "replayed").await;
    assert_eq!(fx.reloads_since(&reloads_before), counts([]), "no reload");

    let mut records = pending.iter();
    let mut expected = Vec::new();
    let mut replayed_so_far: u64 = 0;
    for (i, (bound, replayed)) in rounds.iter().enumerate() {
        let entry_bound = if i == 0 {
            bounds.initial_entries.min(bounds.max_entries)
        } else {
            bounds.max_entries
        };
        expected.push((
            RepairPoint::Read,
            RepairTransient {
                reply_entry_count: entry_bound.min(bounds.max_total - replayed_so_far),
                reply_bytes: *bound,
                ..RepairTransient::default()
            },
        ));
        if let Some(n) = replayed {
            let chunk: Vec<_> = records.by_ref().take(*n).collect();
            assert_eq!(chunk.len(), *n, "fixture: `rounds` replays every record");
            let (entries, reply_bytes) = (*n as u64, chunk.iter().map(|(size, _)| size).sum());
            expected.push((
                RepairPoint::Replay,
                RepairTransient {
                    reply_entry_count: entries,
                    reply_bytes,
                    parse_delta_bytes: chunk.iter().map(|(_, delta)| *delta).max().unwrap_or(0),
                    replay_delta_bytes: chunk.iter().map(|(_, delta)| delta).sum(),
                },
            ));
            for (_, delta) in &chunk {
                expected.push((
                    RepairPoint::Record,
                    RepairTransient {
                        reply_entry_count: entries,
                        reply_bytes,
                        parse_delta_bytes: *delta,
                        replay_delta_bytes: *delta,
                    },
                ));
            }
            replayed_so_far += entries;
        }
    }
    assert_eq!(
        records.next(),
        None,
        "fixture: `rounds` replays every record"
    );
    assert_eq!(
        charge.repair_needs(),
        expected,
        "RP0 at each read's entry and byte bounds, the byte bound grown and kept for the reads \
         after it, then RP1 at its chunk's records and an RP2 per record"
    );
}

/// A backlog of records each over the first read's bound (256 KiB; a 300 KiB
/// screenshot each, the first 310 KiB): the first read answers `TOO_LARGE`
/// and the second reads the first record at its size; each later chunk stops
/// at its bound with more pending, so the next read asks twice the bound —
/// the third two records, the fourth the last two. Four reads, where one
/// record per read after the `TOO_LARGE` would take six. The first record
/// being the largest keeps the batches independent of a few bytes' drift in
/// the entries' sizes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_hit_behind_records_over_the_first_read_bound_doubles_the_grown_read(
) {
    let fx = Fx::new("scr-doubled-grown-read").await;
    fx.warm().await;
    let pending = commit_backlog(
        &fx,
        &[310 * 1024, 300 * 1024, 300 * 1024, 300 * 1024, 300 * 1024],
    )
    .await;
    let first = pending[0].0;
    let rest = || pending[1..].iter().map(|(size, _)| *size);
    assert!(
        STRUCTURAL_REPAIR_INITIAL_BYTES < first
            && 4 * first <= STRUCTURAL_REPAIR_MAX_BYTES
            && rest().all(|size| size <= first && 3 * size > 2 * first),
        "fixture: the first entry ({first} B) is over the first read's bound, four of it within \
         the hard cap, and every other entry at most it and over two thirds of it: {pending:?}"
    );

    assert_backlog_repaired_in(
        &fx,
        &pending,
        RepairBounds::PRODUCTION,
        &[
            (STRUCTURAL_REPAIR_INITIAL_BYTES, None),
            (first, Some(1)),
            (2 * first, Some(2)),
            (4 * first, Some(2)),
        ],
        "the 6 of one record per read after the TOO_LARGE",
    )
    .await;
}

/// A backlog of screenshot-sized records each under the first read's bound
/// (256 KiB) whose total is over it (five of 150 KiB): the first read takes
/// one record and stops at its bound with more pending, so the next asks
/// 512 KiB and takes three, and the last asks 1 MiB for the fifth. Three
/// reads, where reads kept at 256 KiB would take one record each: five.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_hit_behind_records_under_the_first_read_bound_doubles_the_read() {
    let fx = Fx::new("scr-doubled-read").await;
    fx.warm().await;
    let pending = commit_backlog(&fx, &[150 * 1024; 5]).await;
    assert!(
        pending
            .iter()
            .all(|(size, _)| 128 * 1024 < *size && *size <= 170 * 1024),
        "fixture: every entry between 128 KiB and 170 KiB, so two are over 256 KiB, three \
         within 512 KiB and four over it: {pending:?}"
    );

    assert_backlog_repaired_in(
        &fx,
        &pending,
        RepairBounds::PRODUCTION,
        &[
            (STRUCTURAL_REPAIR_INITIAL_BYTES, Some(1)),
            (2 * STRUCTURAL_REPAIR_INITIAL_BYTES, Some(3)),
            (4 * STRUCTURAL_REPAIR_INITIAL_BYTES, Some(1)),
        ],
        "the 5 of one record per read",
    )
    .await;
}

/// The doubled bound stops at the hard cap, `max_bytes`, and later reads
/// keep the cap. The bounds are injected so that the cap is not a
/// power-of-two multiple of the first read's bound: eleven small records of
/// about one measured size `s` (they differ only by their `seq`'s digits,
/// which the fixture assertion's margins absorb), a first bound of about
/// 2.5 `s` and a cap of about 4.5 `s`. Every chunk but the last stops at its
/// byte bound with more pending: the first read takes two records and asks
/// twice its bound next, which the cap cuts to 4.5 `s`; each read at the cap
/// takes four, and the fourth read takes the last record. Uncapped, the
/// second read would ask 5 `s` and take five, and the third 10 `s` and take
/// the last four: three reads at other bounds. The entry bound is far above
/// any chunk, so no chunk stops on its entry count.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_hit_behind_a_backlog_caps_the_doubled_read_at_the_hard_cap() {
    let fx = Fx::new("scr-doubled-read-capped").await;
    fx.warm().await;
    let pending = commit_backlog(&fx, &[1024; 11]).await;
    let s_min = pending
        .iter()
        .map(|(size, _)| *size)
        .min()
        .expect("records");
    let s_max = pending
        .iter()
        .map(|(size, _)| *size)
        .max()
        .expect("records");
    let initial = (5 * s_max).div_ceil(2);
    let cap = 9 * s_max / 2;
    assert!(
        2 * s_max <= initial
            && initial < 3 * s_min
            && 2 * initial > cap
            && 4 * s_max <= cap
            && cap < 5 * s_min
            && 5 * s_max <= 2 * initial,
        "fixture: two entries fit the first bound ({initial} B) and three do not, four fit the \
         cap ({cap} B) and five do not, twice the first bound is over the cap and holds five: \
         {pending:?}"
    );
    let bounds = RepairBounds {
        initial_bytes: initial,
        max_bytes: cap,
        ..RepairBounds::PRODUCTION
    };
    assert!(
        bounds.max_entries > pending.len() as u64,
        "fixture: no chunk stops on its entry count"
    );

    assert_backlog_repaired_in(
        &fx,
        &pending,
        bounds,
        &[
            (initial, Some(2)),
            (cap, Some(4)),
            (cap, Some(4)),
            (cap, Some(1)),
        ],
        "the 3 of an uncapped doubling (2, 5 and 4 records)",
    )
    .await;
}

/// RP0 refused on a hit: the call's charge cannot hold its first repair
/// read, so the read is never issued — the entry goes back at its cut (the
/// checked-out entry, three commits behind), nothing replayed — and the read
/// answers `NeedsCharge` for that point's whole target on the hit's path,
/// `structural`. The next read, its charge held, is a hit whose repair
/// replays all three.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_hit_whose_first_read_is_refused_its_charge_puts_the_entry_back_and_answers_needs_charge_on_structural(
) {
    let fx = Fx::new("scr-rp0-refused").await;
    fx.warm().await;
    let cut = fx.cached().expect("warmed");
    fx.commit_fast(3).await;
    let reloads_before = fx.reloads();
    let (_, replayed_before) = fx.repair_histogram("live_edit_structural_cache_replayed_entries");
    let refusing = RepairCharge::refusing(RepairPoint::Read, 0);

    let answer = fx
        .read_request(
            &StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN).with_commit_charge(&refusing),
        )
        .await;

    // The first read's RP0: its reply at the first read's entry bound
    // (`STRUCTURAL_REPAIR_INITIAL_ENTRIES`, under the production per-read
    // bound) and byte bound.
    let rp0 = object_state_charge_for_test(&ObjectStateNeed::cached_repair(
        cut.weight_bytes(),
        RepairTransient {
            reply_entry_count: STRUCTURAL_REPAIR_INITIAL_ENTRIES.min(STRUCTURAL_REPAIR_MAX_ENTRIES),
            reply_bytes: STRUCTURAL_REPAIR_INITIAL_BYTES,
            ..RepairTransient::default()
        },
    ));
    match answer {
        Ok(StructuralRead::NeedsCharge { target, path }) => assert_eq!(
            (target, path),
            (rp0, StructuralPath::Structural),
            "the whole target of the refused RP0, on the hit's path"
        ),
        other => panic!("a refused RP0 answers NeedsCharge; got {other:?}"),
    }
    assert_eq!(refusing.refused(), Some(rp0), "the refused hold was RP0's");
    assert_eq!(
        refusing.repair_needs().len(),
        1,
        "nothing past the refused point was priced: no read"
    );
    let held = fx.cached().expect("the entry is put back");
    assert_eq!(
        (held.seq, &held.commit_chain, &held.state),
        (cut.seq, &cut.commit_chain, &cut.state),
        "put back at its cut, with nothing replayed"
    );
    assert_eq!(
        fx.repair_histogram("live_edit_structural_cache_replayed_entries")
            .1,
        replayed_before,
        "nothing was replayed"
    );
    assert_eq!(
        fx.reloads_since(&reloads_before),
        counts([]),
        "not a reload"
    );

    let granting = RepairCharge::granting();
    let ready = expect_ready(
        fx.read_request(
            &StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN).with_commit_charge(&granting),
        )
        .await,
        "after the wait",
    );
    assert_eq!(
        (ready.path, ready.repair.replayed_entries),
        (StructuralPath::Structural, 3),
        "a hit that replays from the cut"
    );
    fx.assert_current(ready.entry.entry(), "after the wait")
        .await;
}

/// The repair after a cold start charges as a hit's does: with the cold
/// start's decoded entry held (CR0, granted) and its repair's first read
/// refused (RP0), the cold start's entry is stored and the read answers
/// `NeedsCharge` on the cold start's path, `structural_full`. The next read
/// is a hit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_cold_start_whose_repair_read_is_refused_its_charge_answers_needs_charge_on_structural_full(
) {
    let fx = Fx::new("scr-rp0-after-cold").await;
    assert!(fx.cached().is_none(), "fixture: the read cold-starts");
    let refusing = RepairCharge::refusing(RepairPoint::Read, 0);

    let answer = fx
        .read_request(
            &StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN).with_commit_charge(&refusing),
        )
        .await;

    match answer {
        Ok(StructuralRead::NeedsCharge { target, path }) => {
            assert_eq!(
                path,
                StructuralPath::StructuralFull,
                "the repair after a cold start waits on the cold start's path"
            );
            assert_eq!(
                refusing.refused(),
                Some(target),
                "for the refused RP0's target"
            );
        }
        other => panic!("a refused RP0 after a cold start answers NeedsCharge; got {other:?}"),
    }
    {
        let log = refusing.log();
        assert!(
            matches!(log.needs.first(), Some(ObjectStateNeed::Cached { repair, .. })
                if *repair == RepairTransient::default()),
            "fixture: the cold start's CR0, with no repair transient, came first; {log:?}"
        );
        assert_eq!(
            log.tries.first().map(|(_, granted)| *granted),
            Some(true),
            "fixture: CR0 was granted; {log:?}"
        );
    }
    let stored = fx.cached().expect("the cold start's entry is stored");
    let ready = expect_ready(fx.read().await, "after the wait");
    assert_eq!(
        ready.path,
        StructuralPath::Structural,
        "the next read is a hit"
    );
    assert_eq!(ready.entry.entry().seq, stored.seq);
    fx.assert_current(ready.entry.entry(), "after the wait")
        .await;
}

/// RP1 refused after earlier rounds: with one entry per read, the first two
/// rounds replay their records (their RP1 needs held) and the third round's
/// replay is refused its charge — a record large enough that its RP1 grows
/// past what RP0 held. The entry goes back at the second round's cut (the
/// stored state then), the read answers `NeedsCharge` on `structural`, and
/// the next read's repair continues from that cut: it replays the last two
/// records only. Across the two calls the RP1 needs price the first two
/// records' replays once, the third's at its refusal and again when it
/// replays, and the fourth's once: no record is replayed twice or skipped.
/// RP2 prices only a record about to apply: the first two, then the last
/// two, never the third in the refused call. Each round's replay grows the
/// entry, so the charge points after the first round weigh it again, above
/// the stored weight the first round priced; a round's RP2 weighs it as its
/// RP1 did (one record per round).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_hit_whose_third_replay_is_refused_its_charge_puts_the_entry_back_at_the_second_rounds_cut(
) {
    let fx = Fx::new("scr-rp1-refused").await;
    fx.warm().await;
    let records = [
        delta_with_set_field_inline(WORLD, "small0", vec![1; 16], 50),
        delta_with_set_field_inline(WORLD, "small1", vec![2; 16], 51),
        // Its RP1 holds the reply, `k_parse` and the growth per delta byte,
        // about 32 bytes at the shipped factors: more than twice the reply
        // term RP0 holds for a 256 KiB read, so the point is not a no-op.
        delta_with_set_field_inline(WORLD, "large", vec![3; 128 * 1024], 52),
        delta_with_set_field_inline(WORLD, "small2", vec![4; 16], 53),
    ];
    let deltas: Vec<u64> = records.iter().map(delta_part_len).collect();
    let mut records = records.into_iter();
    for record in records.by_ref().take(2) {
        fx.commit(record).await;
    }
    let cut_seq = fx.header_seq().await;
    let cut_chain = header(&fx.tv, fx.oid).await[HEADER_FIELD_COMMIT_CHAIN].clone();
    let cut_state = normalized_fresh_state(&fx.tv, fx.oid).await;
    for record in records {
        fx.commit(record).await;
    }
    let one_per_read = RepairBounds {
        max_entries: 1,
        ..RepairBounds::PRODUCTION
    };
    let refusing = RepairCharge::refusing(RepairPoint::Replay, 2);

    let answer = fx
        .read_request(
            &StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN)
                .with_repair_bounds(one_per_read)
                .with_commit_charge(&refusing),
        )
        .await;

    match answer {
        Ok(StructuralRead::NeedsCharge { target, path }) => {
            assert_eq!(path, StructuralPath::Structural, "on the hit's path");
            assert_eq!(
                refusing.refused(),
                Some(target),
                "for the refused RP1's target"
            );
        }
        other => panic!("a refused RP1 answers NeedsCharge; got {other:?}"),
    }
    let held = fx.cached().expect("the entry is put back");
    assert_eq!(
        (held.seq, &held.commit_chain, &held.state),
        (cut_seq, &cut_chain, &cut_state),
        "put back at the second round's cut: the third record was not replayed"
    );

    let granting = RepairCharge::granting();
    let ready = expect_ready(
        fx.read_request(
            &StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN)
                .with_repair_bounds(one_per_read)
                .with_commit_charge(&granting),
        )
        .await,
        "after the wait",
    );
    assert_eq!(
        (ready.path, ready.repair.replayed_entries),
        (StructuralPath::Structural, 2),
        "the resumed repair continues from the cut: the last two records"
    );
    fx.assert_current(ready.entry.entry(), "after the wait")
        .await;
    let replayed = |charge: &RepairCharge| -> Vec<u64> {
        charge
            .repair_needs()
            .into_iter()
            .filter(|(point, _)| *point == RepairPoint::Replay)
            .map(|(_, repair)| repair.replay_delta_bytes)
            .collect()
    };
    assert_eq!(
        (replayed(&refusing), replayed(&granting)),
        (deltas[..3].to_vec(), deltas[2..].to_vec()),
        "the refused call priced the first three records' replays (the third refused), the \
         resumed one the last two: the third record priced twice, each record replayed once"
    );
    for (point, repair) in granting.repair_needs() {
        if point == RepairPoint::Replay {
            assert_eq!(
                repair.parse_delta_bytes, repair.replay_delta_bytes,
                "one record per round: its parse is its whole replay"
            );
        }
    }
    // RP2 runs only for a record about to apply: the refused call's first
    // two records, never the third (its RP1 was refused), and the resumed
    // call's last two.
    let records = |charge: &RepairCharge| -> Vec<u64> {
        charge
            .record_needs()
            .into_iter()
            .map(|repair| repair.replay_delta_bytes)
            .collect()
    };
    assert_eq!(
        (records(&refusing), records(&granting)),
        (deltas[..2].to_vec(), deltas[2..].to_vec()),
        "an RP2 per record that applied: none for the record whose RP1 was refused"
    );
    // Each record sets a field of a new name, so each round's replay grows
    // the entry: the charge points after the first round weigh the entry
    // again, above the stored weight the first round's points priced. A
    // round's RP2 weighs it as its RP1 did: one record per round, so nothing
    // of the round has applied yet.
    let weights = refusing.repair_weights();
    assert_eq!(
        weights.iter().map(|(point, _)| *point).collect::<Vec<_>>(),
        [
            [RepairPoint::Read, RepairPoint::Replay, RepairPoint::Record].repeat(2),
            vec![RepairPoint::Read, RepairPoint::Replay],
        ]
        .concat(),
        "fixture: two rounds of RP0, RP1 and RP2, then the third's RP0 and its refused RP1"
    );
    let rounds: Vec<&[(RepairPoint, u64)]> = vec![&weights[0..3], &weights[3..6], &weights[6..]];
    let stored = weights[0].1;
    assert!(
        rounds[0].iter().all(|(_, weight)| *weight == stored),
        "the first round's RP1 and RP2 weigh the entry as RP0 did: nothing replayed yet: \
         {weights:?}"
    );
    for (round, points) in rounds.iter().enumerate().skip(1) {
        assert!(
            points.iter().all(|(_, weight)| *weight > stored),
            "round {}'s charge points weigh the entry the earlier replays grew, above the stored \
             {stored} B: {weights:?}",
            round + 1
        );
    }
    for points in &rounds[..2] {
        assert_eq!(
            points[2].1, points[1].1,
            "a one-record round's RP2 weighs the entry as its RP1 did: {weights:?}"
        );
    }
}

/// What RP1 counts for records whose delta part it cannot read: an entry
/// with no OMPF (a commit without relay bytes) counts no delta bytes, and
/// one whose OMPF container does not parse counts its whole OMPF, for both
/// its parse and the entry's growth. Both are in one chunk, so its RP1 is
/// priced — at the chunk's bytes and the bad OMPF's length — before its
/// replay meets the bad OMPF and the repair reloads `parse`. RP2 runs between
/// a record's decode and its apply: the record with no OMPF decodes (an
/// empty commit) and is priced at no delta bytes; the bad one fails its
/// decode, so it is never priced.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_replay_charge_counts_an_unparseable_ompf_whole_and_an_absent_one_as_nothing(
) {
    const BAD_OMPF: &[u8] = b"not an OMPF container";
    let fx = Fx::new("scr-rp1-unparseable").await;
    fx.warm().await;
    let before = fx.header_seq().await;
    fx.commit_through(&fx.ls, WRITER, None, CLIENT_TS).await;
    fx.append_folded_entry(|seq| fx.update_payload(seq), Some(BAD_OMPF.to_vec()))
        .await;
    let pending = stream_entries(&fx.tv, fx.oid, before, before + 2).await;
    assert!(
        pending.len() == 2 && pending.iter().all(|(_, delta)| delta.is_none()),
        "fixture: two pending entries, neither with a readable delta part: {pending:?}"
    );
    let chunk: u64 = pending.iter().map(|(size, _)| size).sum();
    let reloads_before = fx.reloads();
    let charge = RepairCharge::granting();

    let ready = expect_ready(
        fx.read_request(
            &StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN).with_commit_charge(&charge),
        )
        .await,
        "after the parse reload",
    );

    assert_eq!(
        fx.reloads_since(&reloads_before),
        counts([("parse", 1)]),
        "fixture: the bad OMPF reloads `parse` when it replays"
    );
    assert_eq!(
        ready.path,
        StructuralPath::StructuralFull,
        "fixture: reloaded"
    );
    let replays: Vec<RepairTransient> = charge
        .repair_needs()
        .into_iter()
        .filter(|(point, _)| *point == RepairPoint::Replay)
        .map(|(_, repair)| repair)
        .collect();
    let bad = BAD_OMPF.len() as u64;
    assert_eq!(
        replays,
        vec![RepairTransient {
            reply_entry_count: 2,
            reply_bytes: chunk,
            parse_delta_bytes: bad,
            replay_delta_bytes: bad,
        }],
        "one RP1, for the chunk of both: the absent OMPF counts nothing, the unparseable one its \
         whole length"
    );
    assert_eq!(
        charge.record_needs(),
        vec![RepairTransient {
            reply_entry_count: 2,
            reply_bytes: chunk,
            parse_delta_bytes: 0,
            replay_delta_bytes: 0,
        }],
        "one RP2, for the record with no OMPF, which parses nothing; the unparseable record fails \
         its decode, before its RP2"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. Reload reasons
// ═════════════════════════════════════════════════════════════════════════════

/// The call after an anomaly on a warm entry: a hit whose repair reloads with
/// exactly `reason`, and a reloaded entry that is the stored cut; its
/// put-back is accepted and the call after it is a hit with nothing to
/// replay. The lookups and reloads are counted across these two calls only,
/// so the seeding store's own structural commits are not among them.
async fn assert_reloads(fx: &Fx, reason: &str, bounds: RepairBounds) {
    assert_reloads_in(fx, reason, bounds, reason).await;
}

/// [`assert_reloads`], naming the case `context` in every failure message.
async fn assert_reloads_in(fx: &Fx, reason: &str, bounds: RepairBounds, context: &str) {
    let request = StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN).with_repair_bounds(bounds);
    assert_reloads_on(fx, reason, &request, context).await;
}

/// [`assert_reloads_in`] with both calls made with `request`, which may
/// carry a test hook; a hook that changes the store must act at most once,
/// so the call after the reload finds the store as the reload left it.
async fn assert_reloads_on(
    fx: &Fx,
    reason: &str,
    request: &StructuralReadRequest<'_>,
    context: &str,
) {
    let (lookups_before, reloads_before) = (fx.lookups(), fx.reloads());
    let ready = expect_ready(fx.read_request(request).await, context);
    assert_eq!(
        ready.path,
        StructuralPath::StructuralFull,
        "{context}: a reload reads the object again"
    );
    fx.assert_current(ready.entry.entry(), &format!("{context} reload"))
        .await;
    assert_eq!(
        fx.reloads_since(&reloads_before),
        counts([(reason, 1)]),
        "{context}: exactly one reload, under its own reason"
    );
    assert!(
        ready.entry.put_back_held().is_stored(),
        "{context}: the reloaded entry's put-back succeeds"
    );

    let next = expect_ready(
        fx.read_request(request).await,
        &format!("{context}: the call after the reload"),
    );
    assert_eq!(
        (next.path, next.repair.replayed_entries),
        (StructuralPath::Structural, 0),
        "{context}: the call after the reload hits the reloaded entry"
    );
    assert_eq!(
        fx.lookups_since(&lookups_before),
        counts([("reload", 1), ("hit", 1)]),
        "{context}: the reload, then a hit"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_header_seq_lowered_below_the_entry_reloads_regressed() {
    let fx = Fx::new("scr-regressed-seq").await;
    fx.warm().await;

    let seq = fx.header_seq().await;
    fx.set_header(SEQ_COLUMN, &(seq - 1).to_string()).await;

    assert_reloads(&fx, "regressed", RepairBounds::PRODUCTION).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_non_folding_commit_reloads_regressed() {
    let fx = Fx::new("scr-regressed-fold").await;
    fx.warm().await;

    commit_without_fold(&fx.tv, fx.oid).await;

    assert_reloads(&fx, "regressed", RepairBounds::PRODUCTION).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_rematerialized_at_the_same_seq_reloads_identity() {
    let fx = Fx::new("scr-identity").await;
    fx.warm().await;

    rematerialize_same_seq(&fx.tv, fx.oid).await;

    assert_reloads(&fx, "identity", RepairBounds::PRODUCTION).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_stream_trimmed_past_the_entry_reloads_gap() {
    let fx = Fx::new("scr-gap").await;
    fx.warm().await;

    fx.commit_fast(3).await;
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

    assert_reloads(&fx, "gap", RepairBounds::PRODUCTION).await;
}

/// `reached_end` is not contiguity: a header `seq` raised past the stream's
/// last entry with no entry written — the shape a restore leaves — makes the
/// range run out short of its end, with no entry pending and with one real
/// commit pending before the missing one alike: `gap`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_header_seq_past_the_streams_last_entry_reloads_gap() {
    for real_commits in [0, 1] {
        let fx = Fx::new("scr-gap-shortfall").await;
        fx.warm().await;

        fx.commit_fast(real_commits).await;
        let seq = fx.header_seq().await;
        fx.set_header(SEQ_COLUMN, &(seq + 1).to_string()).await;

        assert_reloads_in(
            &fx,
            "gap",
            RepairBounds::PRODUCTION,
            &format!("gap after {real_commits} real commit(s)"),
        )
        .await;
    }
}

/// The pre-replay checks read every pending payload: an `update` entry
/// naming another `seq` or another object is `parse`, though its OMPF (an
/// empty delta) would replay cleanly.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_payload_naming_another_seq_or_object_reloads_parse() {
    // (case, added to the payload's object id, added to its seq)
    for (what, oid_offset, seq_offset) in [("another seq", 0, 1), ("another object", 1, 0)] {
        let fx = Fx::new("scr-parse-payload").await;
        fx.warm().await;

        fx.append_folded_entry(
            |seq| {
                update_stream_payload(fx.oid + oid_offset, seq + seq_offset, WRITER, CLIENT_TS, 1)
            },
            Some(delta_to_ompf_bytes(&empty_delta(), &[])),
        )
        .await;

        assert_reloads_in(&fx, "parse", RepairBounds::PRODUCTION, what).await;
    }
}

/// A cold start whose entry is over the per-entry cap answers
/// `ColdStart::Uncacheable`, which the read side returns as
/// `StructuralRead::Uncacheable` with the `OverWeight` reason — and, the
/// object being within the default plain ceilings, with no size
/// (`above_ceiling: None`), so the WATCH fallback is not classified.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_cold_start_over_the_weight_cap_answers_uncacheable() {
    let fx = Fx::with_cache("scr-overweight", StructuralCache::new(TINY_BUDGET_BYTES)).await;

    match fx.read().await {
        Ok(StructuralRead::Uncacheable {
            reason: UncacheableReason::OverWeight(weight),
            above_ceiling: None,
        }) => assert!(
            weight > fx.cache.max_entry_bytes(),
            "the verdict names a weight over the cap"
        ),
        other => panic!(
            "an over-cap object within the plain ceilings must answer Uncacheable(OverWeight) \
             with no size; got {other:?}"
        ),
    }
    assert_eq!(fx.cold_starts("uncacheable"), 1);
    assert_eq!(fx.lookups(), counts([("cold", 1)]));
}

// ═════════════════════════════════════════════════════════════════════════════
// Above the plain ceilings: the size an uncacheable answer carries (C1-7)
// ═════════════════════════════════════════════════════════════════════════════

/// A byte ceiling no fixture object comes near.
const NO_BYTE_LIMIT: u64 = 1 << 40;

/// A client on the fixture's store whose plain atomic-read limits are
/// `max_items` / `max_bytes`, reporting into the fixture's registry.
async fn lowered_client(fx: &Fx, max_items: u64, max_bytes: u64) -> ValkeyClient {
    let client = client_with_limits(max_items, max_bytes).await;
    client
        .install_observability_hooks(Arc::clone(&fx.metrics))
        .await;
    client
}

/// The object's cardinalities as the cold start's probe sees them, read off
/// an early-out `TOO_LARGE` answer at an item ceiling of 1 on a client
/// reporting nowhere: `(item bound, weight lower bound)`, computed as the
/// cold start computes them (`snapshot_item_bound`,
/// `state_weight_lower_bound`).
async fn probed_bounds(fx: &Fx) -> (u64, u64) {
    let client = client_with_limits(1, NO_BYTE_LIMIT).await;
    match read_object_snapshot_slim_with_ceilings(
        &client,
        fx.tv.key_schema(),
        &fx.oid.to_string(),
        SlimSnapshotReadConsumer::StructuralCacheAudit,
        SnapshotCeilings::of_client(&client),
    )
    .await
    .expect("the bound read must answer")
    {
        SnapshotReadOf::TooLarge(too_large) => (
            snapshot_item_bound(&too_large.cardinalities),
            state_weight_lower_bound(&too_large.cardinalities),
        ),
        SnapshotReadOf::Snapshot(_) => {
            panic!("fixture: an item ceiling of 1 must refuse the object")
        }
    }
}

/// The size the probe gives a cold start whose item bound is `bound`: the
/// bound, and the bound times the route predictor's bytes per item.
fn probe_estimate(bound: u64) -> LayerSize {
    LayerSize {
        items: bound,
        bytes: bound.saturating_mul(STRUCTURAL_COLD_START_ROUTE_BYTES_PER_ITEM),
    }
}

/// A remaining deadline already spent: below any `hold + S`, so a fenced cold
/// start's waiting cap is zero and it refuses before any permit or acquire.
fn no_time_left() -> RemainingDeadlineFn {
    Arc::new(|| Some(Duration::ZERO))
}

/// `live_edit_structural_cache_uncacheable_total{reason}`.
fn uncacheable_ticks(fx: &Fx, reason: &str) -> u64 {
    sample_value(
        &render(&fx.metrics),
        "live_edit_structural_cache_uncacheable_total",
        &[("reason", reason)],
    )
    .unwrap_or_default() as u64
}

/// A fenced cold start that fails for its call (the waiting cap: no time
/// left) on an object whose probe bound is over the plain item ceiling answers
/// `Uncacheable(FencedReadFailed)` carrying the size the read was told of —
/// the probe's estimate, `{bound, bound × K}` — and records nothing: a lookup
/// is then a miss, and the same request with no deadline caches the object on
/// the fenced route.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_fenced_read_failed_above_the_item_ceiling_carries_the_estimate() {
    let fx = Fx::new("scr-fenced-failed-size").await;
    let (bound, _) = probed_bounds(&fx).await;
    let client = lowered_client(&fx, bound - 1, NO_BYTE_LIMIT).await;

    let answer = read_for_structural_edit(
        &client,
        fx.tv.key_schema(),
        &fx.cache,
        &StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN)
            .with_remaining_deadline(no_time_left()),
    )
    .await;

    match answer {
        Ok(StructuralRead::Uncacheable {
            reason: UncacheableReason::FencedReadFailed,
            above_ceiling: Some(size),
        }) => assert_eq!(
            size,
            probe_estimate(bound),
            "the size is the probe's estimate the failed read was told of"
        ),
        other => panic!("expected Uncacheable(FencedReadFailed) with a size; got {other:?}"),
    }
    assert_eq!(uncacheable_ticks(&fx, "fenced_read_failed"), 1);
    assert!(
        matches!(fx.cache.lookup(fx.oid), CacheLookup::Miss),
        "a failed fenced read records no negative entry"
    );

    let ready = expect_ready(
        read_for_structural_edit(
            &client,
            fx.tv.key_schema(),
            &fx.cache,
            &StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN),
        )
        .await,
        "the next call",
    );
    assert_eq!(ready.path, StructuralPath::StructuralFull);
    fx.assert_current(ready.entry.entry(), "the next call")
        .await;
    assert_eq!(fx.cold_starts("fenced"), 1, "the next call read fenced");
}

/// A fenced cold start that fails for its call on an object only PREDICTED
/// over the plain byte ceiling (its bound within the item ceiling, the bound
/// times K over the byte ceiling) answers `Uncacheable(FencedReadFailed)` with
/// no size: a prediction is not knowledge, and the failed read measured
/// nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_fenced_read_failed_only_predicted_over_the_byte_ceiling_carries_no_size(
) {
    let fx = Fx::new("scr-fenced-failed-predicted").await;
    let (bound, _) = probed_bounds(&fx).await;
    let predicted_bytes = probe_estimate(bound).bytes;
    let client = lowered_client(&fx, bound, predicted_bytes - 1).await;

    let answer = read_for_structural_edit(
        &client,
        fx.tv.key_schema(),
        &fx.cache,
        &StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN)
            .with_remaining_deadline(no_time_left()),
    )
    .await;

    match answer {
        Ok(StructuralRead::Uncacheable {
            reason: UncacheableReason::FencedReadFailed,
            above_ceiling: None,
        }) => {}
        other => panic!("expected Uncacheable(FencedReadFailed) with no size; got {other:?}"),
    }
    assert_eq!(
        fx.full_reads(),
        0,
        "fixture: the predictor sent the cold start straight to the fenced route, which the cap \
         refused before reading"
    );
}

/// `OverWeight` from the probe's weight check on an object whose bound is
/// over the plain item ceiling carries the probe's estimate; the next call,
/// routed by the negative entry, carries none (a negative entry holds no
/// size).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_over_weight_above_the_item_ceiling_carries_the_estimate_and_its_negative_entry_none(
) {
    let fx = Fx::with_cache(
        "scr-overweight-above",
        StructuralCache::new(TINY_BUDGET_BYTES),
    )
    .await;
    let (bound, lower_bound) = probed_bounds(&fx).await;
    assert!(
        lower_bound > fx.cache.max_entry_bytes(),
        "fixture: the probe's weight lower bound is over the per-entry cap"
    );
    let client = lowered_client(&fx, bound - 1, NO_BYTE_LIMIT).await;
    let request = StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN);

    match read_for_structural_edit(&client, fx.tv.key_schema(), &fx.cache, &request).await {
        Ok(StructuralRead::Uncacheable {
            reason: UncacheableReason::OverWeight(weight),
            above_ceiling: Some(size),
        }) => {
            assert_eq!(
                weight, lower_bound,
                "the probe's verdict, at the lower bound"
            );
            assert_eq!(size, probe_estimate(bound), "the probe's estimate");
        }
        other => panic!("expected Uncacheable(OverWeight) with a size; got {other:?}"),
    }
    assert_eq!(fx.full_reads(), 0, "fixture: refused before any read");

    match read_for_structural_edit(&client, fx.tv.key_schema(), &fx.cache, &request).await {
        Ok(StructuralRead::Uncacheable {
            reason: UncacheableReason::OverWeight(_),
            above_ceiling: None,
        }) => {}
        other => panic!(
            "the negative entry's lookup must answer Uncacheable(OverWeight) with no size; got \
             {other:?}"
        ),
    }
}

/// `OverWeight` from the admission (the weight lower bound within the cap,
/// the decoded entry over it) after a fenced read that measured the layer
/// above the plain item ceiling carries the measured size.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_over_weight_at_admission_after_a_fenced_read_carries_the_measured_size(
) {
    const MAX_ITEMS: u64 = 1;
    let fx = Fx::new("scr-overweight-admit").await;
    let (_, lower_bound) = probed_bounds(&fx).await;
    // A cap of `lower_bound + 1`: the weight check passes, and the decoded
    // entry, which also weighs every string, is over it.
    let cache = StructuralCache::new(4 * (lower_bound + 1));
    assert!(
        cache.max_entry_bytes() > lower_bound,
        "fixture: the weight check must pass ({} vs a lower bound of {lower_bound})",
        cache.max_entry_bytes()
    );
    let client = lowered_client(&fx, MAX_ITEMS, NO_BYTE_LIMIT).await;

    match read_for_structural_edit(
        &client,
        fx.tv.key_schema(),
        &cache,
        &StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN),
    )
    .await
    {
        Ok(StructuralRead::Uncacheable {
            reason: UncacheableReason::OverWeight(weight),
            above_ceiling: Some(measured),
        }) => {
            assert!(
                weight > cache.max_entry_bytes(),
                "the admission's verdict, over the cap"
            );
            assert!(
                measured.items > MAX_ITEMS && measured.bytes > 0,
                "the fenced read's measured size, above the plain item ceiling: {measured:?}"
            );
        }
        other => panic!(
            "expected Uncacheable(OverWeight) at admission with the measured size; got {other:?}"
        ),
    }
    assert_eq!(fx.cold_starts("uncacheable"), 1);
    assert_eq!(
        sample_value(
            &render(&fx.metrics),
            "live_edit_snapshot_read_total",
            &[
                ("consumer", "structural_cache"),
                ("path", "fenced"),
                ("outcome", "ok")
            ],
        ),
        Some(1.0),
        "fixture: the object was read once, fenced"
    );
}

/// The probe's `NoChsComplete` on an unmarked object whose item bound (the
/// unmarked bound: no `obj:chs` term) is over the plain item ceiling carries
/// the probe's estimate `{bound, bound × K}`, as a probed cold start's
/// verdicts do, with no full read; at an item ceiling equal to the bound it
/// carries none (only a bound over the item ceiling is known to be above the
/// ceilings), and so does the next call, routed by the negative entry (a
/// negative entry holds no size).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_no_chs_complete_above_the_item_ceiling_carries_the_estimate() {
    for (case, over) in [
        ("over the item ceiling", true),
        ("at the item ceiling", false),
    ] {
        let fx = Fx::new("scr-no-chs-above").await;
        unmark(&fx.tv, fx.oid).await;
        // Read after unmarking: the unmarked object's bound, which the probe's
        // cardinalities give with `children_index` unread.
        let (bound, _) = probed_bounds(&fx).await;
        let max_items = if over { bound - 1 } else { bound };
        let client = lowered_client(&fx, max_items, NO_BYTE_LIMIT).await;
        let request = StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN);

        let first =
            read_for_structural_edit(&client, fx.tv.key_schema(), &fx.cache, &request).await;
        let want = over.then(|| probe_estimate(bound));
        match first {
            Ok(StructuralRead::Uncacheable {
                reason: UncacheableReason::NoChsComplete,
                above_ceiling,
            }) => assert_eq!(
                above_ceiling, want,
                "{case}: the probe's estimate exactly when its bound is over the item ceiling"
            ),
            other => panic!("{case}: expected Uncacheable(NoChsComplete); got {other:?}"),
        }
        assert_eq!(
            fx.full_reads(),
            0,
            "{case}: answered by the probe, no full read"
        );
        assert_eq!(
            fx.cold_starts("uncacheable"),
            1,
            "{case}: the probe's verdict"
        );

        let routed =
            read_for_structural_edit(&client, fx.tv.key_schema(), &fx.cache, &request).await;
        assert_no_chs_complete(routed, &format!("{case}: routed by the negative entry"));
    }
}

/// A `deleted` entry in range under a header that still exists (and whose
/// `commit_chain` folds over it, so the fold check passes) reloads
/// `tombstone`. A real `DeleteLayer` removes the header too, which the header
/// mapping answers first: see the next test.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_tombstone_entry_in_range_reloads_tombstone() {
    let fx = Fx::new("scr-tombstone").await;
    fx.warm().await;

    fx.append_folded_entry(|_| DELETED_STREAM_PAYLOAD.to_owned(), None)
        .await;

    assert_reloads(&fx, "tombstone", RepairBounds::PRODUCTION).await;
}

/// An object deleted through another replica — a store whose cache is not
/// this one, so nothing here was invalidated — answers the header mapping's
/// `NotFound`, drops the entry, and is not a reload.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_hit_on_an_object_deleted_elsewhere_answers_not_found() {
    let fx = Fx::new("scr-deleted").await;
    fx.warm().await;

    layer_store(&fx.tv)
        .delete_layer(
            &fx.session_id,
            &fx.layer,
            crate::storage::snapshot_read_atomic_test::mock_ext_store(),
        )
        .await
        .expect("delete_layer must succeed");

    assert_header_mapped(fx.read().await, HeaderChange::Absent, "deleted elsewhere");
    assert!(fx.cached().is_none(), "the entry is dropped");
    assert_eq!(
        fx.reloads(),
        counts([]),
        "a header-mapped drop is not a reload"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_unparseable_ompf_reloads_parse() {
    let fx = Fx::new("scr-parse").await;
    fx.warm().await;

    fx.append_folded_entry(
        |seq| fx.update_payload(seq),
        Some(b"not an OMPF container".to_vec()),
    )
    .await;

    assert_reloads(&fx, "parse", RepairBounds::PRODUCTION).await;
}

/// A well-formed entry `apply_delta` refuses: a `CreateSection` of a section
/// that exists.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_unreplayable_entry_reloads_replay_err() {
    let fx = Fx::new("scr-replay-err").await;
    fx.warm().await;

    let duplicate = structural(vec![create_cmd(ROOT, WORLD, "Duplicate")]);
    fx.append_folded_entry(
        |seq| fx.update_payload(seq),
        Some(delta_to_ompf_bytes(&duplicate, &[])),
    )
    .await;

    assert_reloads(&fx, "replay_err", RepairBounds::PRODUCTION).await;
}

/// One pending entry alone larger than the repair reads' hard cap (an
/// injected `max_bytes` below a large inline value's entry, which caps the
/// first read's bound too, so no read can grow to the entry).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_pending_entry_over_the_byte_bound_reloads_too_large() {
    let fx = Fx::new("scr-too-large").await;
    fx.warm().await;
    let bounds = RepairBounds {
        max_bytes: SMALL_BYTE_BOUND,
        ..RepairBounds::PRODUCTION
    };
    const { assert!(SMALL_BYTE_BOUND < STRUCTURAL_REPAIR_MAX_BYTES) };

    fx.commit(delta_with_set_field_inline(
        WORLD,
        "large",
        vec![0xA5; LARGE_VALUE_BYTES],
        40,
    ))
    .await;

    assert_reloads(&fx, "too_large", bounds).await;
}

/// Under the production bounds: a pending entry above the 4 MiB hard cap
/// (a record carrying an external value just past it) reloads `too_large`
/// after the first, 256 KiB read answers `TOO_LARGE`, where an entry within
/// the cap grows the next read instead
/// ([`read_for_structural_edit_hit_behind_a_record_over_the_first_read_bound_reads_again_at_its_size_with_no_reload`]).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_pending_entry_over_the_hard_cap_reloads_too_large_under_the_production_bounds(
) {
    let fx = Fx::new("scr-too-large-cap").await;
    fx.warm().await;
    let before = fx.header_seq().await;
    let blob = vec![0xEE; STRUCTURAL_REPAIR_MAX_BYTES as usize + 1024];
    fx.commit_external(
        delta_with_set_field_external(
            CUBE,
            "mesh",
            ext_hash(0xEE).to_vec(),
            blob.len() as u64,
            1,
            42,
        ),
        &blob,
    )
    .await;
    let [(entry_bytes, _)] = stream_entries(&fx.tv, fx.oid, before, before + 1).await[..] else {
        panic!("fixture: one pending entry");
    };
    assert!(
        entry_bytes > STRUCTURAL_REPAIR_MAX_BYTES,
        "fixture: the entry ({entry_bytes} B) is over the hard cap"
    );

    assert_reloads(&fx, "too_large", RepairBounds::PRODUCTION).await;
}

/// SR-3.2 acceptance: an orphan stores no list, so its parent's creation by
/// another store adopts it with no reload; a parent holding an earlier
/// build's list reloads. After the warm-up a section ([`ORPHAN`]) is created
/// under the missing [`ORPHAN_PARENT`]; another store then creates
/// `DANGLING_PARENT`, under which the fixture keeps an earlier build's list
/// (`every_shape_state`), and a child under it, and then [`ORPHAN_PARENT`]
/// and a child under that.
///
/// Each creation is planned on a state holding an orphan under the parent
/// the delta makes live (the read model excludes lists under dangling
/// parents). In a debug build `apply_delta` checks the tree's cross-map
/// consistency after every delta; that check must accept the adoption, so a
/// panic in either commit fails this test.
///
/// The first call's repair replays the orphan's creation, which records no
/// dangling parent, then `DANGLING_PARENT`'s creation, which makes a listed
/// dangling parent live: a `dangling_adopted` reload, whose entry holds the
/// stored list — the fixture's orphan, then the child. The second call's
/// repair replays [`ORPHAN_PARENT`]'s creation and its child with no reload,
/// and the entry's list under it is [`ORPHAN`], then [`ADOPTED_CHILD`].
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_replayed_creation_of_a_dangling_parent_reloads_dangling_adopted()
{
    let fx = Fx::new("scr-dangling").await;
    fx.warm().await;
    let other = layer_store(&fx.tv);
    let commit_other = |delta: OwnedDelta| {
        let (fx, other) = (&fx, &other);
        async move {
            fx.commit_through(other, WRITER, Some(delta), CLIENT_TS)
                .await
        }
    };

    fx.commit(orphan_delta()).await;
    assert!(
        !stored_dangling_parents(&fx.tv, fx.oid)
            .await
            .contains(&ORPHAN_PARENT),
        "fixture: the orphan stores no list under its missing parent"
    );
    commit_other(structural(vec![create_cmd(
        ROOT,
        DANGLING_PARENT,
        "Adopter",
    )]))
    .await;
    commit_other(structural(vec![create_cmd(
        DANGLING_PARENT,
        NEW_SECTION,
        "Child",
    )]))
    .await;

    assert_reloads(&fx, "dangling_adopted", RepairBounds::PRODUCTION).await;
    let reloaded = fx.cached().expect("the reloaded entry is cached");
    let listed = reloaded
        .state
        .children
        .get(&DANGLING_PARENT)
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        listed.last(),
        Some(&NEW_SECTION),
        "the reloaded entry holds the adopted list, the child last; got {listed:?}"
    );
    assert_eq!(
        listed.len(),
        2,
        "the adopted orphan, then the child: {listed:?}"
    );

    commit_other(structural(vec![create_cmd(ROOT, ORPHAN_PARENT, "Adopter")])).await;
    commit_other(structural(vec![create_cmd(
        ORPHAN_PARENT,
        ADOPTED_CHILD,
        "Child",
    )]))
    .await;
    let ready = expect_ready(fx.read().await, "the orphan's parent created");
    assert_eq!(
        (ready.path, ready.repair.replayed_entries),
        (StructuralPath::Structural, 2),
        "a parent with no stored list is adopted on replay with no reload"
    );
    assert_eq!(
        ready.entry.entry().state.children.get(&ORPHAN_PARENT),
        Some(&vec![ORPHAN, ADOPTED_CHILD]),
        "the creation adopted the orphan, then the child was appended"
    );
    fx.assert_current(ready.entry.entry(), "after the adoption")
        .await;
}

/// SR-3.2 upkeep during replay, with no adoption: a section created under a
/// missing parent adds no dangling parent (no list is stored under it), and
/// a commit that deletes a dangling parent's stored list removes it — a
/// delta that creates `DANGLING_PARENT` and deletes it again
/// ([`create_then_delete_dangling_parent_delta`]). The repaired entry's set
/// is the store's dangling ids.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_repair_adds_and_removes_dangling_parents() {
    let fx = Fx::new("scr-dangling-upkeep").await;
    let warm = fx.warm().await;
    assert_eq!(
        warm.dangling_parents,
        HashSet::from([DANGLING_PARENT]),
        "fixture: the seeded orphan's parent is the only dangling parent"
    );

    fx.commit(orphan_delta()).await;
    fx.commit(create_then_delete_dangling_parent_delta()).await;
    assert!(
        stored_dangling_parents(&fx.tv, fx.oid).await.is_empty(),
        "fixture: the commit deleted the dangling parent's stored list, and the orphan stored \
         none"
    );
    let ready = expect_ready(fx.read().await, "dangling upkeep");

    assert_eq!(
        (ready.path, ready.repair.replayed_entries),
        (StructuralPath::Structural, 2),
        "both records replay; neither adopts a dangling parent it leaves live"
    );
    assert_eq!(
        ready.entry.entry().dangling_parents,
        HashSet::new(),
        "the orphan's parent not added, the deleted list's parent removed"
    );
    fx.assert_current(ready.entry.entry(), "dangling upkeep")
        .await;
}

/// SR-3.2's check for a call's own plan (Epic 6 drops the entry on
/// `Committed` instead of swapping the successor in): true exactly when the
/// plan creates a section the entry holds as a dangling parent — the
/// fixture's `DANGLING_PARENT`, under which it keeps an earlier build's list.
/// The plans run through `apply_delta` on the entry's state, as a structural
/// call's would; the adopting one passes the debug consistency check.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plan_adopts_dangling_parent_is_true_only_for_a_plan_creating_a_dangling_parent() {
    let fx = Fx::new("scr-own-creation").await;
    fx.commit(orphan_delta()).await;
    let warm = fx.warm().await;
    assert_eq!(
        warm.dangling_parents,
        HashSet::from([DANGLING_PARENT]),
        "fixture: the stored list's parent is the only dangling parent; the orphan stored none"
    );
    let plan = |cmd| {
        let mut state = warm.state.clone();
        apply_delta(&mut state, Some(structural(vec![cmd]))).expect("the plan applies");
        state
    };

    assert!(plan_adopts_dangling_parent(
        &warm,
        &plan(create_cmd(ROOT, DANGLING_PARENT, "Adopter"))
    ));
    assert!(
        !plan_adopts_dangling_parent(&warm, &plan(create_cmd(ROOT, ORPHAN_PARENT, "Adopter"))),
        "creating an orphan's parent with no stored list is no dangling adoption"
    );
    assert!(!plan_adopts_dangling_parent(
        &warm,
        &plan(create_cmd(ROOT, NEW_SECTION, "Unrelated"))
    ));
    assert!(
        !plan_adopts_dangling_parent(&warm, &warm.state),
        "the entry's own state never adopts"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. The header mapping
// ═════════════════════════════════════════════════════════════════════════════

/// On a hit (the repair's header check) and on a miss (the probe), each
/// change maps to the commit paths' error, the entry is dropped, and none is
/// a reload.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_maps_an_absent_partial_or_superseded_header() {
    for change in HeaderChange::ALL {
        for hit in [true, false] {
            let context = format!("{change:?} on a {}", if hit { "hit" } else { "miss" });
            let fx = Fx::new("scr-header-map").await;
            if hit {
                fx.warm().await;
            }

            change.apply(&fx.tv.client().pool, &fx.header_key()).await;
            let answer = fx.read().await;

            assert_header_mapped(answer, change, &context);
            assert!(fx.cached().is_none(), "{context}: no entry is left");
            assert_eq!(fx.reloads(), counts([]), "{context}: not a reload");
            assert!(
                !fx.lookups().contains_key("reload"),
                "{context}: not a reload lookup"
            );
        }
    }
}

/// A header changed between the probe and the cold start: the atomic read
/// answers `NotFound` for an absent header and a partial one alike, and
/// `superseded` on its own; the read side re-probes or maps it, so a partial
/// or superseded header never comes back `NotFound`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_header_changed_after_the_probe_is_reclassified() {
    for change in HeaderChange::ALL {
        let fx = Fx::new("scr-reclassify").await;
        let request = StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN)
            .with_after_probe_hook_for_test(after_probe(&fx, Some(change), None));

        let answer = fx.read_request(&request).await;

        assert_header_mapped(answer, change, &format!("{change:?} after the probe"));
        assert!(fx.cached().is_none(), "{change:?}: nothing is stored");
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. Token before plan (SR-4.2a)
// ═════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_hit_with_this_calls_landed_token_answers_already_landed() {
    let fx = Fx::new("scr-token-hit").await;
    let warm = fx.warm().await;
    let seq = fx.header_seq().await;
    write_commit_token_record(&fx.tv, fx.oid, CLIENT_ID, TOKEN, seq, 4_242).await;

    let landed = expect_already_landed(fx.read().await, "this call's token on a hit");
    let other = expect_ready(
        fx.read_bounded(OTHER_TOKEN, RepairBounds::PRODUCTION).await,
        "another call's token",
    );

    assert_eq!(landed, (seq, 4_242), "the record's seq and server_ts");
    assert_eq!(
        (other.path, other.entry.entry().seq),
        (StructuralPath::Structural, warm.seq),
        "AlreadyLanded put the entry back, and a record of another token plans"
    );
}

/// A lost-reply retry: the call's first attempt committed a `CreateSection`
/// (its reply lost) while the cache entry is behind it. The retry, with the
/// first attempt's real token, answers `AlreadyLanded` with that commit's
/// `seq` and `server_ts` — no entry to plan the repeated `CreateSection` on.
/// The entry went back short of the header; the next call replays the commit
/// once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_lost_reply_retry_of_a_create_section_answers_already_landed() {
    let fx = Fx::new("scr-token-retry").await;
    let warm = fx.warm().await;

    let (seq, server_ts) = fx
        .commit_through(
            &fx.ls,
            CLIENT_ID,
            Some(structural(vec![create_cmd(ROOT, NEW_SECTION, "Retried")])),
            CLIENT_TS,
        )
        .await;
    let record: String = fx
        .tv
        .client()
        .pool
        .hget(fx.wtok_key(), CLIENT_ID)
        .await
        .expect("HGET obj:wtok");
    let token = record
        .split(':')
        .next()
        .expect("a record begins with its token")
        .to_owned();

    let landed = expect_already_landed(
        fx.read_bounded(&token, RepairBounds::PRODUCTION).await,
        "the retry",
    );

    assert_eq!(landed, (seq, server_ts), "the first attempt's commit");
    assert_eq!(
        fx.cached().map(|e| e.seq),
        Some(warm.seq),
        "the entry went back as it was, short of the header"
    );
    let next = expect_ready(
        fx.read_bounded(OTHER_TOKEN, RepairBounds::PRODUCTION).await,
        "the next call",
    );
    assert_eq!(next.repair.replayed_entries, 1);
    fx.assert_current(next.entry.entry(), "after the retry")
        .await;
    assert_eq!(fx.header_seq().await, seq, "the retry committed nothing");
}

/// On a miss the probe answers the token before any full read; nothing is
/// stored.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_miss_with_this_calls_landed_token_answers_already_landed_without_a_full_read(
) {
    let fx = Fx::new("scr-token-miss").await;
    write_commit_token_record(&fx.tv, fx.oid, CLIENT_ID, TOKEN, 9, 9_999).await;

    let landed = expect_already_landed(fx.read().await, "this call's token on a miss");

    assert_eq!(landed, (9, 9_999));
    assert_eq!(fx.full_reads(), 0, "no cold start read");
    assert!(fx.cached().is_none(), "nothing is stored");
}

/// The record lands between the probe and the cold start: the cold start's
/// own token read answers it — or, should that check miss it, the repair
/// read after the cold start does, which reads the record again; the two are
/// not told apart here. Either way the loaded entry is stored.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_cold_start_reading_this_calls_token_answers_already_landed() {
    let fx = Fx::new("scr-token-cold").await;
    let request = StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN)
        .with_after_probe_hook_for_test(after_probe(&fx, None, Some((11, 1_111))));

    let landed = expect_already_landed(fx.read_request(&request).await, "cold start");

    assert_eq!(landed, (11, 1_111));
    let stored = fx.cached().expect("the loaded entry is put back");
    fx.assert_current(&stored, "stored by the cold start").await;
}

/// A hook, run between two repair reads, that lands this call's commit — a
/// `CreateSection` committed through `update_layer` as [`CLIENT_ID`] — once,
/// and records its `(seq, server_ts)` in `landed`.
///
/// `update_layer` mints its own random token; the hook then rewrites the
/// call's record to [`TOKEN`] with that commit's `seq` and `server_ts`, which
/// makes the landed commit this call's. (The known-token seam,
/// `LayerStore::with_commit_tokens_for_test`, exists only under
/// `test-fault-injection`, which this module does not require.)
fn land_this_calls_commit(
    fx: &Fx,
    landed: Arc<Mutex<Option<(u64, u64)>>>,
) -> BetweenRepairReadsHook {
    let (client, keys) = (fx.tv.client().clone(), fx.tv.key_schema().clone());
    let (session_id, layer, oid, wtok_key) = (
        fx.session_id.clone(),
        fx.layer.clone(),
        fx.oid,
        fx.wtok_key(),
    );
    let fired = Arc::new(AtomicBool::new(false));
    Arc::new(move || {
        let (client, keys, landed, fired) = (
            client.clone(),
            keys.clone(),
            Arc::clone(&landed),
            Arc::clone(&fired),
        );
        let (session_id, layer, wtok_key) = (session_id.clone(), layer.clone(), wtok_key.clone());
        async move {
            if fired.swap(true, Ordering::SeqCst) {
                return;
            }
            let ls = LayerStore::new(client.clone(), keys, 4096);
            let (seq, server_ts) = commit_to(
                &ls,
                &session_id,
                &layer,
                oid,
                CLIENT_ID,
                Some(structural(vec![create_cmd(ROOT, NEW_SECTION, "Landed")])),
                &[],
                CLIENT_TS,
            )
            .await;
            let _: () = client
                .pool
                .hset(
                    &wtok_key,
                    [(CLIENT_ID, format!("{TOKEN}:{seq}:{server_ts}"))],
                )
                .await
                .expect("HSET obj:wtok");
            *landed.lock().expect("landed lock") = Some((seq, server_ts));
        }
        .boxed()
    })
}

/// SR-4.2a on a later repair read: the call's own commit lands after the
/// repair's first read. With one entry per read, the first read replays one
/// of two pending entries; the commit lands between the reads; the second
/// read's token check answers `AlreadyLanded` with that commit, and the
/// entry goes back at the first read's cut — replayed, short of the header.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_commit_landing_between_repair_reads_answers_already_landed() {
    let fx = Fx::new("scr-token-between").await;
    fx.warm().await;
    fx.commit_fast(1).await;
    let first_cut_seq = fx.header_seq().await;
    let first_cut_chain = header(&fx.tv, fx.oid).await[HEADER_FIELD_COMMIT_CHAIN].clone();
    let first_cut_state = normalized_fresh_state(&fx.tv, fx.oid).await;
    fx.commit_fast(1).await;
    let landed = Arc::new(Mutex::new(None));
    let request = StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN)
        .with_repair_bounds(RepairBounds {
            max_entries: 1,
            ..RepairBounds::PRODUCTION
        })
        .with_between_repair_reads_hook_for_test(land_this_calls_commit(&fx, Arc::clone(&landed)));

    let answer = expect_already_landed(fx.read_request(&request).await, "between reads");

    let landed = landed
        .lock()
        .expect("landed lock")
        .expect("the hook ran between two repair reads");
    assert_eq!(answer, landed, "the commit that landed between the reads");
    assert_eq!(
        fx.repair_histogram("live_edit_structural_cache_replayed_entries")
            .1,
        1,
        "the first read's entry was replayed before the token answered"
    );
    let cached = fx.cached().expect("the entry is put back");
    assert_eq!(
        (cached.seq, &cached.commit_chain),
        (first_cut_seq, &first_cut_chain),
        "put back at the first read's cut"
    );
    assert_eq!(
        cached.state, first_cut_state,
        "the first read's cut's state"
    );
}

/// [`read_for_structural_edit_commit_landing_between_repair_reads_answers_already_landed`]'s
/// shape on a cache whose per-entry cap lies between the warm entry's weight
/// and the weight it grows to: the warm entry is cached under the cap, the
/// first pending entry is [`growth_delta`] — [`GROWTH_FIELDS`] rows of new
/// field names, the growth a compact entry is weighed by (an inline value's
/// bytes are not in it) — and the repair (one entry per read) replays it —
/// growing the held entry past the cap — before `between_reads` runs. The
/// second read then answers without planning, and the read side puts the
/// held entry back (`put_back_held`), which the cap refuses
/// (`PutBack::OverCap`).
///
/// Returns the fixture and the answer, after asserting what every such
/// answer must leave: the one growth entry ([`growth_delta`]) replayed,
/// `reloads_total{reason="uncacheable"}` ticked exactly once, and an
/// `OverWeight` negative entry above the cap.
async fn answer_after_the_repair_grew_past_the_cap(
    layer: &str,
    between_reads: impl FnOnce(&Fx) -> BetweenRepairReadsHook,
) -> (Fx, Result<StructuralRead, ServiceError>) {
    const SLACK_BYTES: u64 = 512;
    let probe = Fx::new(&format!("{layer}-probe")).await;
    let warm_weight = probe.warm().await.state_weight_bytes;
    let fx = Fx::with_cache(layer, StructuralCache::new(4 * (warm_weight + SLACK_BYTES))).await;
    let warm = fx.warm().await;
    assert_eq!(
        warm.state_weight_bytes,
        warm_weight,
        "fixture: the warm entry weighs what the probe's did, under the cap {}",
        fx.cache.max_entry_bytes()
    );
    fx.commit(growth_delta()).await;
    fx.commit_fast(1).await;
    let reloads_before = fx.reloads();
    let verdicts_before = uncacheable_verdicts(&fx.metrics);
    let request = StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN)
        .with_layer(&fx.session_id, &fx.layer)
        .with_repair_bounds(RepairBounds {
            max_entries: 1,
            ..RepairBounds::PRODUCTION
        })
        .with_between_repair_reads_hook_for_test(between_reads(&fx));

    let (answer, captured) = capture(fx.read_request(&request)).await;

    assert_eq!(
        fx.repair_histogram("live_edit_structural_cache_replayed_entries")
            .1,
        1,
        "fixture: the first read replayed the growing entry before the second read answered"
    );
    assert_eq!(
        fx.reloads_since(&reloads_before),
        counts([("uncacheable", 1)]),
        "the held entry the per-entry cap refused at the read side's put-back must be counted, \
         once; got the answer {answer:?}"
    );
    let weight = match fx.cache.lookup(fx.oid) {
        CacheLookup::Uncacheable {
            reason: UncacheableReason::OverWeight(weight),
        } => {
            assert!(
                weight > fx.cache.max_entry_bytes(),
                "the negative entry records the grown entry's weight, {weight}"
            );
            weight
        }
        other => panic!("the object must be uncacheable (OverWeight); got {other:?}"),
    };
    assert_eq!(
        cells_since(uncacheable_verdicts(&fx.metrics), &verdicts_before),
        counts([("over_weight", 1)]),
        "the refused put-back left a negative entry: one over_weight verdict"
    );
    assert_one_verdict_line(
        &captured,
        fx.oid,
        "over_weight",
        &[
            ("session_id", fx.session_id.as_str().into()),
            ("layer_name", fx.layer.as_str().into()),
            ("weight_bytes", weight.into()),
            ("max_entry_bytes", fx.cache.max_entry_bytes().into()),
        ],
        &CEILING_FIELDS,
    );
    (fx, answer)
}

/// The read side's `Answered` put-back (C3 amendment): the call's own commit
/// lands between two repair reads after the first grew the held entry past
/// the cap; the second read answers `AlreadyLanded`, and the over-cap
/// put-back is counted `uncacheable`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_already_landed_after_a_repair_grew_the_entry_past_the_cap_is_counted_uncacheable(
) {
    let landed = Arc::new(Mutex::new(None));
    let (_fx, answer) = answer_after_the_repair_grew_past_the_cap("scr-cap-landed", |fx| {
        land_this_calls_commit(fx, Arc::clone(&landed))
    })
    .await;

    let landed = landed
        .lock()
        .expect("landed lock")
        .expect("the hook ran between two repair reads");
    assert_eq!(
        expect_already_landed(answer, "over the cap"),
        landed,
        "the commit that landed between the reads"
    );
}

/// The read side's wrong-typed-record put-back (C3 amendment): the
/// commit-token record key is retyped to a string between two repair reads
/// after the first grew the held entry past the cap; the second read answers
/// the wrong-type refusal, and the over-cap put-back is counted
/// `uncacheable`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_wrong_typed_record_after_a_repair_grew_the_entry_past_the_cap_is_counted_uncacheable(
) {
    let (_fx, answer) = answer_after_the_repair_grew_past_the_cap("scr-cap-wrongtype", |fx| {
        let (pool, wtok_key) = (fx.tv.client().pool.clone(), fx.wtok_key());
        let fired = Arc::new(AtomicBool::new(false));
        Arc::new(move || {
            let (pool, wtok_key, fired) = (pool.clone(), wtok_key.clone(), Arc::clone(&fired));
            async move {
                if fired.swap(true, Ordering::SeqCst) {
                    return;
                }
                let _: i64 = pool.del(&wtok_key).await.expect("DEL obj:wtok");
                let _: () = pool
                    .set(&wtok_key, "planted-not-a-hash", None, None, false)
                    .await
                    .expect("SET obj:wtok to a string");
            }
            .boxed()
        })
    })
    .await;

    assert!(
        matches!(
            answer,
            Ok(StructuralRead::CommitTokenRecordWrongType { .. })
        ),
        "the second read answers the wrong-type refusal; got {answer:?}"
    );
}

/// The token check precedes the `chs_complete` check: an unmarked object
/// with this call's record answers `AlreadyLanded`, not `Uncacheable`, and
/// records no negative entry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_unmarked_object_with_this_calls_token_answers_already_landed() {
    let fx = Fx::new("scr-token-unmarked").await;
    unmark(&fx.tv, fx.oid).await;
    write_commit_token_record(&fx.tv, fx.oid, CLIENT_ID, TOKEN, 6, 600).await;

    let landed = expect_already_landed(fx.read().await, "unmarked");

    assert_eq!(landed, (6, 600));
    assert!(fx.cached().is_none(), "no negative entry is recorded");
    assert_eq!(fx.cold_starts("uncacheable"), 0);
}

/// The token check precedes the `superseded` mapping on every route: a hit's
/// repair, a miss's probe, and a cold start that read `superseded` after the
/// probe passed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_superseded_object_with_this_calls_token_answers_already_landed() {
    for route in ["hit", "probe", "cold start"] {
        let fx = Fx::new("scr-token-superseded").await;
        let request = match route {
            "hit" | "probe" => {
                if route == "hit" {
                    fx.warm().await;
                }
                HeaderChange::Superseded
                    .apply(&fx.tv.client().pool, &fx.header_key())
                    .await;
                write_commit_token_record(&fx.tv, fx.oid, CLIENT_ID, TOKEN, 8, 800).await;
                StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN)
            }
            _ => {
                StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN).with_after_probe_hook_for_test(
                    after_probe(&fx, Some(HeaderChange::Superseded), Some((8, 800))),
                )
            }
        };

        let landed = expect_already_landed(fx.read_request(&request).await, route);

        assert_eq!(landed, (8, 800), "{route}");
    }
}

/// The client `operation_id` the call in the operation cases carries.
const CALL_OPERATION_ID: &str = "structural-read-op";
/// Another `operation_id`: the writer's other write.
const OTHER_OPERATION_ID: &str = "structural-read-other-op";

/// An after-probe hook that applies `change` (if any) and then writes `record`
/// verbatim as [`CLIENT_ID`]'s commit-token record: the cold-start route's
/// record, in either record shape.
fn after_probe_writing(fx: &Fx, change: Option<HeaderChange>, record: String) -> AfterProbeHook {
    let pool = fx.tv.client().pool.clone();
    let (header_key, wtok_key) = (fx.header_key(), fx.wtok_key());
    Arc::new(move || {
        let (pool, header_key, wtok_key, record) = (
            pool.clone(),
            header_key.clone(),
            wtok_key.clone(),
            record.clone(),
        );
        async move {
            if let Some(change) = change {
                change.apply(&pool, &header_key).await;
            }
            let _: () = pool
                .hset(&wtok_key, [(CLIENT_ID, record)])
                .await
                .expect("HSET obj:wtok");
        }
        .boxed()
    })
}

/// What an operation case's read must answer.
#[derive(Debug, Clone, Copy)]
enum OperationAnswer {
    /// This call's own commit: `AlreadyLanded`.
    Own,
    /// An earlier call's commit: `Recorded` with this verdict.
    Recorded(RecordedBy),
    /// The start record moved: `Overtaken`.
    Overtaken,
    /// No record answers the call: the next guard (it plans, or on a
    /// superseded object the superseded refusal).
    NextGuard,
}

/// Plan b.v6d revision 3 on the structural read side (SR-4.2a): a call
/// carrying a client operation ([`StructuralReadRequest::with_operation`])
/// is answered before it plans, on every route — a hit's repair, a miss's
/// probe, a cold start that read the record — by the verdict, and in the
/// order, of the commit script's step 1b: a record under `op.<id>` is this
/// call's own commit (`AlreadyLanded`), an earlier call's same request
/// (`Recorded(EarlierCall)`) or different request (`Recorded(Conflict)`) —
/// matched before the start record, which these calls (started with the
/// field absent) no longer hold; otherwise a record that moved off the
/// call's start record is `Overtaken`; only a call whose start record still
/// holds goes on to the next guard. With the object superseded every answer
/// but the last is the same — the start record is checked before
/// `superseded` — and the last is the superseded refusal.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_with_an_operation_answers_by_the_record_before_superseded_on_every_route(
) {
    const SEQ: u64 = 8;
    const SERVER_TS: u64 = 800;
    let digest = operation_request_digest("structural-read-op-layer", b"request");
    let other_digest = operation_request_digest("structural-read-op-layer", b"another request");
    let record =
        |op: &str, token: &str, digest: &str| format!("op.{op}:{SEQ}:{SERVER_TS}:{token}:{digest}");
    let others = record(OTHER_OPERATION_ID, OTHER_TOKEN, &digest);
    // (case, the record, the call's start record, the answer)
    let cases = [
        (
            "own",
            record(CALL_OPERATION_ID, TOKEN, &digest),
            StartRecord::Absent,
            OperationAnswer::Own,
        ),
        (
            "earlier call, same request",
            record(CALL_OPERATION_ID, OTHER_TOKEN, &digest),
            StartRecord::Absent,
            OperationAnswer::Recorded(RecordedBy::EarlierCall),
        ),
        (
            "earlier call, another request",
            record(CALL_OPERATION_ID, OTHER_TOKEN, &other_digest),
            StartRecord::Absent,
            OperationAnswer::Recorded(RecordedBy::Conflict),
        ),
        (
            "start record moved",
            others.clone(),
            StartRecord::Absent,
            OperationAnswer::Overtaken,
        ),
        (
            "start record held",
            others.clone(),
            StartRecord::Record(others.clone()),
            OperationAnswer::NextGuard,
        ),
    ];

    let mut failures = Vec::new();
    for route in ["hit", "probe", "cold start"] {
        for superseded in [false, true] {
            for (case, written, start, answer) in &cases {
                let context = format!(
                    "{route}{}: {case}",
                    if superseded { ", superseded" } else { "" }
                );
                let fx = Fx::new("scr-token-operation").await;
                let op = Operation::new(CALL_OPERATION_ID, digest.clone())
                    .with_start_record(start.clone());
                let change = superseded.then_some(HeaderChange::Superseded);
                let mut request = StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN);
                if route == "cold start" {
                    // A call that started holding a record found it at the
                    // probe too (anything else is a moved record, which the
                    // probe answers `Overtaken` itself); the hook rewrites
                    // the same value, so the cold start reads it again.
                    if matches!(start, StartRecord::Record(_)) {
                        let _: () = fx
                            .tv
                            .client()
                            .pool
                            .hset(fx.wtok_key(), [(CLIENT_ID, written.as_str())])
                            .await
                            .expect("HSET obj:wtok");
                    }
                    request = request.with_after_probe_hook_for_test(after_probe_writing(
                        &fx,
                        change,
                        written.clone(),
                    ));
                } else {
                    if route == "hit" {
                        fx.warm().await;
                    }
                    if let Some(change) = change {
                        change.apply(&fx.tv.client().pool, &fx.header_key()).await;
                    }
                    let _: () = fx
                        .tv
                        .client()
                        .pool
                        .hset(fx.wtok_key(), [(CLIENT_ID, written.as_str())])
                        .await
                        .expect("HSET obj:wtok");
                }
                let request = request.with_operation(&op);

                let read = fx.read_request(&request).await;

                let recorded = |by| RecordedCommit {
                    by,
                    seq: SEQ,
                    server_ts: SERVER_TS,
                };
                let ok = match (answer, &read) {
                    (
                        OperationAnswer::Own,
                        Ok(StructuralRead::AlreadyLanded { seq, server_ts }),
                    ) => (*seq, *server_ts) == (SEQ, SERVER_TS),
                    (OperationAnswer::Recorded(by), Ok(StructuralRead::Recorded(got))) => {
                        *got == recorded(*by)
                    }
                    (OperationAnswer::Overtaken, Ok(StructuralRead::Overtaken)) => true,
                    (OperationAnswer::NextGuard, Ok(StructuralRead::Ready(_))) => !superseded,
                    (
                        OperationAnswer::NextGuard,
                        Err(ServiceError::FailedPrecondition(message)),
                    ) => superseded && message == SUPERSEDED_MESSAGE,
                    _ => false,
                };
                if !ok {
                    failures.push(format!("{context}: expected {answer:?}; got {read:?}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// What a chained case's read must answer.
#[derive(Debug, Clone, Copy)]
enum ChainedAnswer {
    /// This call's own commit: `AlreadyLanded`.
    Own,
    /// An earlier call's commit of the operation: `Recorded` with this verdict.
    Recorded(RecordedBy),
    /// The record does not name the predecessor: `PredecessorPending` with the
    /// record as written.
    Pending,
    /// The record names the predecessor (or, for a chain's first write, is
    /// absent): the next guard — it plans, or on a superseded object the
    /// superseded refusal.
    NextGuard,
}

/// b.pav group 15 on the structural read side (C1): a call carrying a
/// CHAINED client operation (its writer guard the predecessor
/// [`OTHER_OPERATION_ID`], or none for a chain's first write) is answered
/// before it plans, at all three read sites — a hit's repair, a miss's probe,
/// a cold start that read the record — in the commit script's order: the
/// own-key match (`AlreadyLanded` / `Recorded`), then the superseded check,
/// then the predecessor (`PredecessorPending`, carrying the record as read,
/// `None` when absent), and only a record naming the predecessor goes on to
/// plan. With the object superseded, every answer but the own-key ones is the
/// superseded refusal — superseded wins on the hit and the probe, and a cold
/// start that reads it superseded answers it first — so a predecessor check
/// placed before the superseded check fails the superseded rows.
///
/// The cold-start route reaches its loaded-entry site this way: the probe
/// reads a record that lets the call through (its predecessor's, or none for
/// a first write), and the after-probe hook then writes the case's record (or
/// deletes it), which the cold start reads.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_with_a_chained_operation_answers_its_predecessor_after_superseded_on_every_route(
) {
    const SEQ: u64 = 8;
    const SERVER_TS: u64 = 800;
    const THIRD_OPERATION_ID: &str = "structural-read-third-op";
    let digest = operation_request_digest("structural-read-op-layer", b"request");
    let record = |op: &str, token: &str| format!("op.{op}:{SEQ}:{SERVER_TS}:{token}:{digest}");
    let predecessors = record(OTHER_OPERATION_ID, OTHER_TOKEN);
    // (case, the predecessor, the record the read finds, the answer)
    let cases: [(&str, &str, Option<String>, ChainedAnswer); 8] = [
        (
            "own",
            OTHER_OPERATION_ID,
            Some(record(CALL_OPERATION_ID, TOKEN)),
            ChainedAnswer::Own,
        ),
        (
            "earlier call, same request",
            OTHER_OPERATION_ID,
            Some(record(CALL_OPERATION_ID, OTHER_TOKEN)),
            ChainedAnswer::Recorded(RecordedBy::EarlierCall),
        ),
        (
            "another operation",
            OTHER_OPERATION_ID,
            Some(record(THIRD_OPERATION_ID, OTHER_TOKEN)),
            ChainedAnswer::Pending,
        ),
        (
            "a write without an operation id",
            OTHER_OPERATION_ID,
            Some(format!("{OTHER_TOKEN}:{SEQ}:{SERVER_TS}")),
            ChainedAnswer::Pending,
        ),
        (
            "no record",
            OTHER_OPERATION_ID,
            None,
            ChainedAnswer::Pending,
        ),
        (
            "the predecessor",
            OTHER_OPERATION_ID,
            Some(predecessors.clone()),
            ChainedAnswer::NextGuard,
        ),
        (
            "a first write over a record",
            "",
            Some(predecessors.clone()),
            ChainedAnswer::Pending,
        ),
        (
            "a first write over none",
            "",
            None,
            ChainedAnswer::NextGuard,
        ),
    ];

    let mut failures = Vec::new();
    for route in ["hit", "probe", "cold start"] {
        for superseded in [false, true] {
            for (case, previous, written, answer) in &cases {
                let context = format!(
                    "{route}{}: {case}",
                    if superseded { ", superseded" } else { "" }
                );
                let fx = Fx::new("scr-token-chained").await;
                let pool = fx.tv.client().pool.clone();
                let op =
                    Operation::new(CALL_OPERATION_ID, digest.clone()).with_predecessor(previous);
                let change = superseded.then_some(HeaderChange::Superseded);
                let write = |record: Option<String>| {
                    let (pool, wtok_key) = (pool.clone(), fx.wtok_key());
                    async move {
                        match record {
                            Some(record) => {
                                let _: () = pool
                                    .hset(&wtok_key, [(CLIENT_ID, record)])
                                    .await
                                    .expect("HSET obj:wtok");
                            }
                            None => {
                                let _: i64 = pool
                                    .hdel(&wtok_key, CLIENT_ID)
                                    .await
                                    .expect("HDEL obj:wtok");
                            }
                        }
                    }
                };
                let mut request = StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN);
                if route == "cold start" {
                    // The probe reads a record that lets the call through;
                    // the hook then writes the case's, which the cold start
                    // reads.
                    write((!previous.is_empty()).then(|| predecessors.clone())).await;
                    let (header_key, wtok_key, written) =
                        (fx.header_key(), fx.wtok_key(), written.clone());
                    request = request.with_after_probe_hook_for_test(Arc::new(move || {
                        let (pool, header_key, wtok_key, written) = (
                            pool.clone(),
                            header_key.clone(),
                            wtok_key.clone(),
                            written.clone(),
                        );
                        async move {
                            if let Some(change) = change {
                                change.apply(&pool, &header_key).await;
                            }
                            match written {
                                Some(record) => {
                                    let _: () = pool
                                        .hset(&wtok_key, [(CLIENT_ID, record)])
                                        .await
                                        .expect("HSET obj:wtok");
                                }
                                None => {
                                    let _: i64 = pool
                                        .hdel(&wtok_key, CLIENT_ID)
                                        .await
                                        .expect("HDEL obj:wtok");
                                }
                            }
                        }
                        .boxed()
                    }));
                } else {
                    if route == "hit" {
                        fx.warm().await;
                    }
                    if let Some(change) = change {
                        change.apply(&pool, &fx.header_key()).await;
                    }
                    write(written.clone()).await;
                }
                let request = request.with_operation(&op);

                let read = fx.read_request(&request).await;

                let ok = match (answer, &read) {
                    (ChainedAnswer::Own, Ok(StructuralRead::AlreadyLanded { seq, server_ts })) => {
                        (*seq, *server_ts) == (SEQ, SERVER_TS)
                    }
                    (ChainedAnswer::Recorded(by), Ok(StructuralRead::Recorded(got))) => {
                        *got == RecordedCommit {
                            by: *by,
                            seq: SEQ,
                            server_ts: SERVER_TS,
                        }
                    }
                    (ChainedAnswer::Pending, Ok(StructuralRead::PredecessorPending { record })) => {
                        !superseded && record == written
                    }
                    (ChainedAnswer::NextGuard, Ok(StructuralRead::Ready(_))) => !superseded,
                    (
                        ChainedAnswer::Pending | ChainedAnswer::NextGuard,
                        Err(ServiceError::FailedPrecondition(message)),
                    ) => superseded && message == SUPERSEDED_MESSAGE,
                    _ => false,
                };
                if !ok {
                    let expected = match answer {
                        ChainedAnswer::Pending | ChainedAnswer::NextGuard if superseded => {
                            "the superseded refusal".to_owned()
                        }
                        other => format!("{other:?}"),
                    };
                    failures.push(format!("{context}: expected {expected}; got {read:?}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A commit-token record key of a foreign type answers the refusal
/// `update_layer` itself returns, byte for byte, and puts any entry back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_wrong_typed_token_record_answers_update_layers_refusal() {
    for hit in [true, false] {
        let fx = Fx::new("scr-token-wrongtype").await;
        if hit {
            fx.warm().await;
        }
        let _: () = fx
            .tv
            .client()
            .pool
            .set(fx.wtok_key(), "not-a-hash", None, None, false)
            .await
            .expect("SET obj:wtok");
        let commit_refusal = fx
            .ls
            .update_layer_relayed(
                &fx.session_id,
                &fx.layer,
                fx.oid,
                Some(delta_with_set_field_inline(WORLD, "x", vec![1], 50)),
                CLIENT_ID,
                CLIENT_TS,
            )
            .await
            .expect_err("fixture: update_layer refuses a wrong-typed record");

        let refusal = match fx.read().await {
            Ok(StructuralRead::CommitTokenRecordWrongType { refusal }) => refusal,
            other => panic!("hit={hit}: expected the wrong-type refusal; got {other:?}"),
        };

        assert_eq!(
            format!("{refusal:?}"),
            format!("{commit_refusal:?}"),
            "hit={hit}: byte-identical to the commit backends' refusal"
        );
        assert_eq!(
            fx.cached().is_some(),
            hit,
            "hit={hit}: an entry held is put back; a miss stores none"
        );
    }
}

/// The refusal both commit backends and the probe return for a commit-token
/// record key holding a string, byte for byte.
fn string_token_record_refusal(fx: &Fx) -> String {
    format!(
        "update_layer keyspace type error: WRONGTYPE_KEY: {} holds a string, expected hash \
         (commit-token record)",
        fx.wtok_key()
    )
}

/// Assert `answer` is the wrong-type refusal for a string-typed record key.
fn assert_string_token_record_refusal(
    fx: &Fx,
    answer: Result<StructuralRead, ServiceError>,
    context: &str,
) {
    match answer {
        Ok(StructuralRead::CommitTokenRecordWrongType {
            refusal: ServiceError::Internal(message),
        }) => assert_eq!(
            message,
            string_token_record_refusal(fx),
            "{context}: byte-identical to the probe's and the commit backends' refusal"
        ),
        other => panic!("{context}: expected the wrong-type refusal; got {other:?}"),
    }
}

/// A hook, run between the probe and the cold start, that retypes the
/// commit-token record key to a string with a plain `SET` — after the probe
/// read it as an absent hash — and, with `superseded`, also marks the header
/// superseded.
fn retype_token_record_after_probe(fx: &Fx, superseded: bool) -> AfterProbeHook {
    let pool = fx.tv.client().pool.clone();
    let (header_key, wtok_key) = (fx.header_key(), fx.wtok_key());
    Arc::new(move || {
        let (pool, header_key, wtok_key) = (pool.clone(), header_key.clone(), wtok_key.clone());
        async move {
            if superseded {
                HeaderChange::Superseded.apply(&pool, &header_key).await;
            }
            let _: () = pool
                .set(&wtok_key, "not-a-hash", None, None, false)
                .await
                .expect("SET obj:wtok");
        }
        .boxed()
    })
}

/// The commit-token record key changes type between the probe and the cold
/// start's token `HGET`: the cold start answers the probe's wrong-type
/// refusal, byte for byte, and keeps the entry it loaded — a valid cut — so
/// once the key is restored the next call is a hit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_token_record_retyped_after_the_probe_answers_the_refusal_and_keeps_the_entry(
) {
    let fx = Fx::new("scr-token-retyped").await;
    let request = StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN)
        .with_after_probe_hook_for_test(retype_token_record_after_probe(&fx, false));

    let answer = fx.read_request(&request).await;

    assert_string_token_record_refusal(&fx, answer, "retyped after the probe");
    let stored = fx
        .cached()
        .expect("the cold start's loaded entry is put back");
    fx.assert_current(&stored, "the entry put back").await;

    let _: i64 = fx
        .tv
        .client()
        .pool
        .del(fx.wtok_key())
        .await
        .expect("DEL obj:wtok");
    let next = expect_ready(fx.read().await, "after the key is restored");
    assert_eq!(
        (next.path, next.repair.replayed_entries),
        (StructuralPath::Structural, 0),
        "the call after the refusal hits the entry the cold start loaded"
    );
    assert_eq!(fx.lookups(), counts([("cold", 1), ("hit", 1)]));
}

/// The record key retyped and the header superseded in the same moment after
/// the probe: the token check comes before the `superseded` mapping on the
/// cold start's route too, so the answer is the wrong-type refusal, not
/// `FailedPrecondition(SUPERSEDED)`. A superseded read loads nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_token_record_retyped_and_header_superseded_after_the_probe_answers_the_refusal(
) {
    let fx = Fx::new("scr-token-retyped-superseded").await;
    let request = StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN)
        .with_after_probe_hook_for_test(retype_token_record_after_probe(&fx, true));

    let answer = fx.read_request(&request).await;

    assert_string_token_record_refusal(&fx, answer, "retyped and superseded after the probe");
    assert!(fx.cached().is_none(), "a superseded read stores nothing");
}

// ═════════════════════════════════════════════════════════════════════════════
// 6. NoChsComplete, G4
// ═════════════════════════════════════════════════════════════════════════════

/// Every command a client (not a script) issued against the object's keys
/// while `f` ran.
async fn object_commands<F: std::future::Future>(
    fx: &Fx,
    f: F,
) -> (F::Output, Vec<MonitoredCommand>) {
    let tag = format!("{}{{{}}}:", fx.tv.prefix(), fx.oid);
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let out = f.await;
    let fence = monitor_fence(fx.tv.prefix());
    let _: i64 = fx.tv.client().pool.exists(&fence).await.unwrap();
    let commands = tap
        .drain_until(&fence)
        .await
        .into_iter()
        .filter(|c| !c.issued_by_script() && c.args.iter().any(|a| a.contains(&tag)))
        .collect();
    (out, commands)
}

/// Assert `answer` is `Uncacheable(NoChsComplete)`.
fn assert_no_chs_complete(answer: Result<StructuralRead, ServiceError>, context: &str) {
    assert!(
        matches!(
            answer,
            Ok(StructuralRead::Uncacheable {
                reason: UncacheableReason::NoChsComplete,
                above_ceiling: None,
            })
        ),
        "{context}: expected Uncacheable(NoChsComplete); got {answer:?}"
    );
}

/// An object without `chs_complete`: the probe records an `NoChsComplete`
/// negative entry with no full read (one `EVALSHA`, counted under the
/// `uncacheable` cold-start route); the next call routes from the negative
/// entry with no command at all. The negative TTL is one no test comes near,
/// so the second call is inside it however slow the runner; the expiry is the
/// next test's.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_unmarked_object_is_probed_once_then_routed_uncacheable_with_no_command(
) {
    let fx = Fx::with_cache(
        "scr-no-chs",
        StructuralCache::with_ttls(BUDGET_BYTES, LONG_TTL, LONG_TTL),
    )
    .await;
    unmark(&fx.tv, fx.oid).await;
    // Load the repair script, so the probe is one EVALSHA, not a NOSCRIPT
    // round trip.
    read_structural_cache_sync(
        fx.tv.client(),
        fx.tv.key_schema(),
        &StructuralSyncRequest::probe(fx.oid, CLIENT_ID),
    )
    .await
    .expect("warm the script");

    let (first, probed) = object_commands(&fx, fx.read()).await;
    let (second, routed) = object_commands(&fx, fx.read()).await;

    assert_no_chs_complete(first, "first call");
    assert_no_chs_complete(second, "within the TTL");
    assert_eq!(
        probed
            .iter()
            .map(MonitoredCommand::name)
            .collect::<Vec<_>>(),
        vec!["EVALSHA"],
        "the first call is one probe"
    );
    assert!(
        routed.is_empty(),
        "within the TTL: no command; got {routed:?}"
    );
    assert_eq!(fx.full_reads(), 0, "no full read");
    assert_eq!(fx.cold_starts("uncacheable"), 1);
    assert_eq!(
        fx.lookups(),
        counts([("cold", 1)]),
        "an uncacheable lookup ticks nothing"
    );
}

/// The probe's `NoChsComplete` verdict is reported once, where it is reached:
/// one `no_chs_complete` tick and one INFO line naming the session and layer
/// the request carries ([`StructuralReadRequest::with_layer`]) and the
/// object. The next call, routed by the cached verdict (its negative entry),
/// ticks and logs nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_probe_verdict_reports_once_naming_the_layer_and_a_routed_call_reports_nothing(
) {
    let fx = Fx::with_cache(
        "scr-no-chs-report",
        StructuralCache::with_ttls(BUDGET_BYTES, LONG_TTL, LONG_TTL),
    )
    .await;
    unmark(&fx.tv, fx.oid).await;
    let request =
        StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN).with_layer(&fx.session_id, &fx.layer);

    let (first, first_log) = capture(fx.read_request(&request)).await;
    let after_first = uncacheable_verdicts(&fx.metrics);
    let (second, second_log) = capture(fx.read_request(&request)).await;

    assert_no_chs_complete(first, "first call");
    assert_no_chs_complete(second, "routed by the cached verdict");
    assert_eq!(
        after_first,
        counts([("no_chs_complete", 1)]),
        "the probe's verdict: one tick"
    );
    let mut absent = CEILING_FIELDS.to_vec();
    absent.extend(WEIGHT_FIELDS);
    assert_one_verdict_line(
        &first_log,
        fx.oid,
        "no_chs_complete",
        &[
            ("session_id", fx.session_id.as_str().into()),
            ("layer_name", fx.layer.as_str().into()),
        ],
        &absent,
    );
    assert!(
        second_log.with_message(UNCACHEABLE_VERDICT_MSG).is_empty(),
        "a call the cached verdict routes logs nothing:\n{}",
        second_log.text
    );
    assert_eq!(
        uncacheable_verdicts(&fx.metrics),
        after_first,
        "a call the cached verdict routes ticks nothing"
    );
}

/// An object without `chs_complete` whose `NoChsComplete` negative entry has
/// expired is probed again: a second `uncacheable` cold-start verdict and a
/// second `cold` lookup. The TTL is short, so the wait is a fraction of a
/// second; nothing here depends on a call landing inside it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_unmarked_object_is_probed_again_once_its_negative_entry_expires()
{
    let fx = Fx::with_cache(
        "scr-no-chs-expiry",
        StructuralCache::with_ttls(BUDGET_BYTES, LONG_TTL, SHORT_NEGATIVE_TTL),
    )
    .await;
    unmark(&fx.tv, fx.oid).await;
    assert_no_chs_complete(fx.read().await, "first call");
    assert_eq!(
        fx.cold_starts("uncacheable"),
        1,
        "fixture: the first call recorded the verdict"
    );

    let deadline = Instant::now() + WAIT_BUDGET;
    while matches!(fx.cache.lookup(fx.oid), CacheLookup::Uncacheable { .. }) {
        assert!(Instant::now() < deadline, "the negative entry must expire");
        tokio::time::sleep(EXPIRY_POLL).await;
    }
    assert_no_chs_complete(fx.read().await, "after the TTL");

    assert_eq!(
        fx.cold_starts("uncacheable"),
        2,
        "probed again after the TTL"
    );
    assert_eq!(fx.lookups(), counts([("cold", 2)]));
}

/// G4 (SR-5.4): entries whose payload carries a `client_ts` above `i64::MAX`
/// — on the fast path and the WATCH path — replay.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_entries_with_client_ts_above_i64_max_replay() {
    let fx = Fx::new("scr-g4").await;
    fx.warm().await;

    for delta in [
        delta_with_set_field_inline(WORLD, "g4", vec![4], 60),
        structural(vec![create_cmd(ROOT, NEW_SECTION, "G4")]),
    ] {
        fx.commit_through(&fx.ls, WRITER, Some(delta), CLIENT_TS_ABOVE_I64_MAX)
            .await;
    }
    let ready = expect_ready(fx.read().await, "G4");

    assert_eq!(
        (ready.path, ready.repair.replayed_entries),
        (StructuralPath::Structural, 2),
        "both entries replay; no parse reload"
    );
    fx.assert_current(ready.entry.entry(), "G4").await;
    assert_eq!(fx.reloads(), counts([]));
}

// ═════════════════════════════════════════════════════════════════════════════
// 6b. A foreign-typed index under the probe
// ═════════════════════════════════════════════════════════════════════════════

/// Replace `key` with a string, as an out-of-band writer would.
async fn plant_foreign_type(fx: &Fx, key: &str) {
    let _: () = fx
        .tv
        .client()
        .pool
        .set(key, "planted-by-an-out-of-band-writer", None, None, false)
        .await
        .unwrap_or_else(|e| panic!("planting a string at {key} must succeed: {e}"));
}

/// `live_edit_snapshot_read_type_error_total{key_family}`; 0 for an absent
/// cell.
fn type_errors(fx: &Fx, family: &str) -> u64 {
    sample_value(
        &render(&fx.metrics),
        "live_edit_snapshot_read_type_error_total",
        &[("key_family", family)],
    )
    .unwrap_or_default() as u64
}

/// The probe reports a foreign-typed index key as data, so the probe's other
/// answers still answer the call, with no full read: this call's landed
/// token from its record, a superseded header as `SUPERSEDED`, and an unmarked
/// object (here with a foreign-typed `obj:chs`) as `NoChsComplete`. None is
/// `Internal`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_with_a_foreign_typed_index_still_answers_from_the_probe() {
    let fx = Fx::new("scr-bad-index-token").await;
    let flds = fx.tv.key_schema().obj_fields_index(&fx.oid.to_string());
    plant_foreign_type(&fx, &flds).await;
    write_commit_token_record(&fx.tv, fx.oid, CLIENT_ID, TOKEN, 9, 9_999).await;
    assert_eq!(
        expect_already_landed(fx.read().await, "landed token, foreign-typed obj:flds"),
        (9, 9_999),
        "the landed token is answered from its record"
    );
    assert_eq!(fx.full_reads(), 0, "landed token: no full read");

    let fx = Fx::new("scr-bad-index-superseded").await;
    let secs = fx.tv.key_schema().obj_sections_index(&fx.oid.to_string());
    plant_foreign_type(&fx, &secs).await;
    HeaderChange::Superseded
        .apply(&fx.tv.client().pool, &fx.header_key())
        .await;
    assert_header_mapped(
        fx.read().await,
        HeaderChange::Superseded,
        "superseded, foreign-typed obj:secs",
    );
    assert_eq!(fx.full_reads(), 0, "superseded: no full read");

    let fx = Fx::new("scr-bad-index-unmarked").await;
    unmark(&fx.tv, fx.oid).await;
    let chs = fx.tv.key_schema().obj_children_index(&fx.oid.to_string());
    plant_foreign_type(&fx, &chs).await;
    assert_no_chs_complete(fx.read().await, "unmarked, foreign-typed obj:chs");
    assert_eq!(fx.full_reads(), 0, "unmarked: no full read");
}

/// On a marked object that goes on to a cold start, a foreign-typed index
/// leaves the probe no cardinalities to route on: the cold start runs
/// unprobed, and the atomic read's own per-family refusal fails the call
/// `Internal` and ticks `live_edit_snapshot_read_type_error_total` under that
/// index's family, as it did before the probe read the indexes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_cold_start_over_a_foreign_typed_index_ticks_its_type_error() {
    let fx = Fx::new("scr-bad-index-cold").await;
    let flds = fx.tv.key_schema().obj_fields_index(&fx.oid.to_string());
    plant_foreign_type(&fx, &flds).await;
    assert_eq!(type_errors(&fx, "fields_index"), 0, "fixture: no tick yet");

    let answer = fx.read().await;

    match &answer {
        Err(ServiceError::Internal(msg)) => assert!(
            msg.contains("WRONGTYPE"),
            "the atomic read's per-key refusal; got {msg:?}"
        ),
        other => panic!("a foreign-typed obj:flds fails the cold start Internal; got {other:?}"),
    }
    assert_eq!(
        type_errors(&fx, "fields_index"),
        1,
        "the structural path ticks the fields_index type error once"
    );
    assert!(fx.cached().is_none(), "nothing is stored");
}

// ═════════════════════════════════════════════════════════════════════════════
// 7. Repair to a target
// ═════════════════════════════════════════════════════════════════════════════

/// The structural cache and cold-start series, rendered.
fn cache_series(metrics: &Metrics) -> Vec<String> {
    render(metrics)
        .lines()
        .filter(|l| {
            l.starts_with("live_edit_structural_cache_")
                || l.starts_with("live_edit_structural_cold_start_")
        })
        .map(str::to_owned)
        .collect()
}

/// The audit's repair (Task 6) reaches a past cut of a caller-owned copy,
/// ticks no cache or cold-start metric, and changes no cache entry; a target
/// whose `commit_chain` the fold cannot reach is unreachable.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn repair_to_target_reaches_the_cut_ticks_nothing_and_leaves_the_cache_alone() {
    let fx = Fx::new("scr-target").await;
    let warm = fx.warm().await;
    fx.commit_fast(2).await;
    let target_seq = fx.header_seq().await;
    let target_chain = header(&fx.tv, fx.oid).await[HEADER_FIELD_COMMIT_CHAIN].clone();
    let target_state = normalized_fresh_state(&fx.tv, fx.oid).await;
    fx.commit_fast(1).await;
    let before = cache_series(&fx.metrics);

    let mut copy = warm.clone();
    let (reached, stats) = repair_to_target(
        fx.tv.client(),
        fx.tv.key_schema(),
        &mut copy,
        target_seq,
        &target_chain,
        RepairBounds::PRODUCTION,
        None,
        None,
    )
    .await
    .expect("the repair reads");
    let mut wrong = warm.clone();
    let (unreachable, _) = repair_to_target(
        fx.tv.client(),
        fx.tv.key_schema(),
        &mut wrong,
        target_seq,
        &"0".repeat(40),
        RepairBounds::PRODUCTION,
        None,
        None,
    )
    .await
    .expect("the repair reads");

    assert!(matches!(reached, TargetRepair::Reached), "got {reached:?}");
    assert_eq!(
        (copy.seq, &copy.commit_chain, stats.replayed_entries),
        (target_seq, &target_chain, 2)
    );
    assert_eq!(
        copy.state, target_state,
        "the copy is the target cut's state"
    );
    assert!(
        matches!(
            unreachable,
            TargetRepair::Unreachable(ReloadReason::Regressed)
        ),
        "a target commit_chain the fold cannot reach; got {unreachable:?}"
    );
    assert_eq!(
        cache_series(&fx.metrics),
        before,
        "no cache or cold-start series ticks"
    );
    let cached = fx.cached().expect("the cache entry is still there");
    assert_eq!(
        (cached.seq, &cached.commit_chain, &cached.state),
        (warm.seq, &warm.commit_chain, &warm.state),
        "the cache entry is untouched"
    );
}

/// A target the entry cannot reach by replaying forward — above the header,
/// or below the entry's own `seq` — is `Unreachable(Regressed)` before
/// anything is replayed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn repair_to_target_above_the_header_or_below_the_entry_is_regressed() {
    let fx = Fx::new("scr-target-out-of-range").await;
    let warm = fx.warm().await;
    assert_eq!(
        warm.seq,
        fx.header_seq().await,
        "fixture: the entry is current"
    );

    for (what, target_seq) in [
        ("above the header", warm.seq + 1),
        ("below the entry", warm.seq - 1),
    ] {
        let mut copy = warm.clone();
        let (answer, stats) = repair_to_target(
            fx.tv.client(),
            fx.tv.key_schema(),
            &mut copy,
            target_seq,
            &warm.commit_chain,
            RepairBounds::PRODUCTION,
            None,
            None,
        )
        .await
        .expect("the repair reads");

        assert!(
            matches!(answer, TargetRepair::Unreachable(ReloadReason::Regressed)),
            "{what}: expected Unreachable(Regressed); got {answer:?}"
        );
        assert_eq!(
            (stats.replayed_entries, copy.seq),
            (0, warm.seq),
            "{what}: nothing is replayed"
        );
    }
}

/// A section id the charged repair-to-target tests create under the root.
const TARGET_NEW_SECTION: u64 = 910;
/// A section id they create under `WORLD`.
const TARGET_NEW_CHILD: u64 = 911;

/// The OMPF of the object's stream entry at `seq`.
async fn stream_ompf(fx: &Fx, seq: u64) -> Vec<u8> {
    let raw: Vec<(String, std::collections::HashMap<String, Bytes>)> = fx
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
    let [(_, fields)] = &raw[..] else {
        panic!("fixture: one stream entry at {seq}");
    };
    fields
        .get(RELAY_OMPF_FIELD)
        .expect("fixture: the entry relays its OMPF")
        .to_vec()
}

/// With a charge — the sampled audit's — a refused charge point ends the
/// repair to a target with `TargetRepair::ChargeRefused` (before the charge,
/// such a refusal was an `Internal` error), for that point's target, at a
/// valid, normalized cut short of the target: refused at its first read
/// (RP0), nothing replayed; refused at the second record of a chunk (its
/// RP2), the first record replayed and the copy, an entry without plan
/// indexes, normalized at that record's cut — the stored state there.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn repair_to_target_under_a_refusing_charge_answers_charge_refused_at_a_valid_cut() {
    for (what, point, nth) in [
        ("its first read", RepairPoint::Read, 0),
        ("its second record", RepairPoint::Record, 1),
    ] {
        let fx = Fx::new("scr-target-refused").await;
        let warm = fx.warm().await;
        let before = fx.header_seq().await;
        fx.commit(structural(vec![create_cmd(
            ROOT,
            TARGET_NEW_SECTION,
            "First",
        )]))
        .await;
        let first_cut = (
            fx.header_seq().await,
            header(&fx.tv, fx.oid).await[HEADER_FIELD_COMMIT_CHAIN].clone(),
            normalized_fresh_state(&fx.tv, fx.oid).await,
        );
        fx.commit(structural(vec![create_cmd(
            WORLD,
            TARGET_NEW_CHILD,
            "The second record",
        )]))
        .await;
        let target_seq = fx.header_seq().await;
        let target_chain = header(&fx.tv, fx.oid).await[HEADER_FIELD_COMMIT_CHAIN].clone();
        // One read takes both records, at their size: its RP0 holds no more
        // than the chunk's RP1, so the second record's RP2 — its RP1's
        // reply and delta parts with its own parse (the larger part) and
        // what it grows the copy by — asks past what is held, and the
        // charge's refusal is reached.
        let pending = stream_entries(&fx.tv, fx.oid, before, target_seq).await;
        let bounds = RepairBounds {
            initial_entries: 2,
            initial_bytes: pending.iter().map(|(size, _)| size).sum(),
            ..RepairBounds::PRODUCTION
        };
        let refusing = RepairCharge::refusing(point, nth);
        let mut copy = warm.copy_without_plan();

        let (answer, stats) = repair_to_target(
            fx.tv.client(),
            fx.tv.key_schema(),
            &mut copy,
            target_seq,
            &target_chain,
            bounds,
            Some(&refusing),
            None,
        )
        .await
        .expect("the repair reads");

        match answer {
            TargetRepair::ChargeRefused { target } => assert_eq!(
                refusing.refused(),
                Some(target),
                "{what}: for the refused point's target"
            ),
            other => panic!("{what}: a refused charge point answers ChargeRefused; got {other:?}"),
        }
        if point == RepairPoint::Read {
            assert_eq!(
                (stats.replayed_entries, copy.seq, &copy.state),
                (0, warm.seq, &warm.state),
                "{what}: nothing is replayed"
            );
        } else {
            assert_eq!(
                (
                    stats.replayed_entries,
                    copy.seq,
                    &copy.commit_chain,
                    &copy.state
                ),
                (1, first_cut.0, &first_cut.1, &first_cut.2),
                "{what}: the first record replayed, the copy normalized at its cut"
            );
        }
    }
}

/// On an entry without plan indexes — the sampled audit's copy — the repair
/// never re-weighs the entry (an O(state) walk): each charge point holds the
/// copy's stored state weight `W0` plus `G`, the growth RP2 priced for every
/// record replayed before it — its `growth` alone, never its `scratch`,
/// which the record's apply frees before it returns — and its per-byte
/// growth term carries `D`, the delta parts replayed before it (RP0 and RP1:
/// of earlier rounds; RP2: of the records before it), plus its own. Four
/// records over two reads of two entries: a creation, a cascade (the world
/// and its cube deleted), a field write and a creation. Each RP2 prices its
/// record's growth plus its scratch, and its rewrite, on the copy as it
/// stands: the per-record estimate with no plan indexes
/// (`estimate_replay_record_growth`, which includes the index-less rewrite
/// — the per-delta name and adoption indexes and the full depth walk),
/// computed here independently of the repair, from each record's decoded
/// delta on a state replayed alongside, through the list-delete sink as the
/// entry's dangling parent has the repair replay it, and normalized as the
/// repair normalizes after each record and each round; that state ends equal
/// to the copy. The cascade's scratch is the largest of the four, and the
/// points after it hold `G` without it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn repair_to_target_on_an_entry_without_plan_indexes_charges_its_stored_weight_plus_the_priced_growth_and_the_replayed_delta_parts(
) {
    use live_edit::domain::delta::MutationSink;
    use live_edit::domain::replay::{
        apply_decoded_record_into, normalize_children, normalize_state,
    };

    let fx = Fx::new("scr-target-copy-charge").await;
    let warm = fx.warm().await;
    let before = fx.header_seq().await;
    fx.commit(structural(vec![create_cmd(
        ROOT,
        TARGET_NEW_SECTION,
        "First",
    )]))
    .await;
    fx.commit(structural(vec![delete_cmd(ROOT, WORLD)])).await;
    fx.commit(delta_with_set_field_inline(
        TARGET_NEW_SECTION,
        "between",
        vec![7; 32],
        60,
    ))
    .await;
    fx.commit(structural(vec![create_cmd(
        TARGET_NEW_SECTION,
        TARGET_NEW_CHILD,
        "Fourth",
    )]))
    .await;
    let target_seq = fx.header_seq().await;
    let target_chain = header(&fx.tv, fx.oid).await[HEADER_FIELD_COMMIT_CHAIN].clone();
    let pending: Vec<(u64, u64)> = stream_entries(&fx.tv, fx.oid, before, target_seq)
        .await
        .into_iter()
        .map(|(size, delta)| (size, delta.expect("a relayed record's OMPF parses")))
        .collect();
    assert_eq!(pending.len(), 4, "fixture: four pending records");
    let mut copy = warm.copy_without_plan();
    let w0 = copy.state_weight_bytes;
    assert!(
        !copy.dangling_parents.is_empty(),
        "fixture: the entry holds a dangling parent, so its records replay through the \
         list-delete sink"
    );
    // Each record's estimate on the copy as it stands before it, from a
    // state replayed alongside: normalized after each record (its lists, as
    // `advance_to_successor` normalizes an entry without plan indexes) and
    // after each round of two (`normalize_state`).
    let mut alongside = copy.state.clone();
    let mut estimates = Vec::new();
    for seq in before + 1..=target_seq {
        let record = decode_stream_record(seq - 1, Some(&stream_ompf(&fx, seq).await), seq)
            .expect("fixture: the record decodes");
        estimates.push(estimate_replay_record_growth(
            &alongside,
            None,
            record.delta(),
            true,
        ));
        apply_decoded_record_into(&mut alongside, record, &mut MutationSink::list_deletes())
            .expect("fixture: the record applies");
        normalize_children(&mut alongside);
        if (seq - before) % 2 == 0 {
            normalize_state(&mut alongside);
        }
    }
    let charge = RepairCharge::granting();
    let bounds = RepairBounds {
        max_entries: 2,
        ..RepairBounds::PRODUCTION
    };

    let (answer, stats) = repair_to_target(
        fx.tv.client(),
        fx.tv.key_schema(),
        &mut copy,
        target_seq,
        &target_chain,
        bounds,
        Some(&charge),
        None,
    )
    .await
    .expect("the repair reads");

    assert!(matches!(answer, TargetRepair::Reached), "got {answer:?}");
    assert_eq!(stats.replayed_entries, 4);
    assert_eq!(
        alongside, copy.state,
        "fixture self-check: the state replayed alongside is the repaired copy's"
    );
    let needs: Vec<(RepairPoint, ObjectStateNeed)> = {
        let log = charge.log();
        repair_points(&log.needs)
            .into_iter()
            .zip(log.needs.iter().copied())
            .filter_map(|(point, need)| point.map(|point| (point, need)))
            .collect()
    };
    let points: Vec<RepairPoint> = needs.iter().map(|(point, _)| *point).collect();
    use RepairPoint::{Read, Record, Replay};
    assert_eq!(
        points,
        [Read, Replay, Record, Record, Read, Replay, Record, Record],
        "fixture: two rounds of two records"
    );
    let cached = |i: usize| match needs[i].1 {
        ObjectStateNeed::Cached {
            held,
            growth,
            rewrite,
            repair,
        } => (held, growth, rewrite, repair),
        other => panic!("a repair point is a cached need; got {other:?}"),
    };
    // Each record's RP2, in record order.
    let records = [2, 3, 6, 7];
    for (k, &i) in records.iter().enumerate() {
        assert_eq!(
            (cached(i).1, cached(i).2),
            (
                estimates[k].growth + estimates[k].scratch,
                estimates[k].rewrite
            ),
            "record {}'s RP2 prices its growth plus its scratch, and its rewrite, as estimated on \
             the copy with no plan indexes: {:?}",
            k + 1,
            estimates[k]
        );
    }
    // G after each record: its growth alone.
    let g = |k: usize| -> u64 { estimates[..k].iter().map(|e| e.growth).sum() };
    let d: Vec<u64> = pending.iter().map(|(_, delta)| *delta).collect();
    let expected_held = [
        w0,
        w0,
        w0,
        w0 + g(1),
        w0 + g(2),
        w0 + g(2),
        w0 + g(2),
        w0 + g(3),
    ];
    let expected_replayed = [
        0,
        d[0] + d[1],
        d[0],
        d[0] + d[1],
        d[0] + d[1],
        d[0] + d[1] + d[2] + d[3],
        d[0] + d[1] + d[2],
        d[0] + d[1] + d[2] + d[3],
    ];
    for i in 0..needs.len() {
        let (held, _, _, repair) = cached(i);
        assert_eq!(
            (held, repair.replay_delta_bytes),
            (expected_held[i], expected_replayed[i]),
            "point #{i} ({:?}): held W0 + G — the growth priced for the records before it, \
             without their scratch — and the delta parts replayed with its own; {needs:?}",
            points[i]
        );
    }
    let scratch: Vec<u64> = estimates.iter().map(|e| e.scratch).collect();
    assert!(
        estimates[0].growth > 0
            && estimates[3].growth > 0
            && estimates[0].rewrite.indexed_sections > 0
            && estimates[0].rewrite.adoption_entries > 0
            && scratch.iter().all(|&s| s > 0)
            && scratch[1] > scratch[0].max(scratch[2]).max(scratch[3]),
        "fixture: each creation prices growth, and the index-less rewrite its name and adoption \
         indexes; every record prices scratch (the list-delete sink's), the cascade the most: \
         {estimates:?}"
    );
}

/// The copy-bound guard on the audit's repair path (b.5h3): a copy without
/// plan indexes, repaired under a charge by `repair_to_target` — which checks
/// the copy's bucket bound against its tables' true buckets after every
/// charged record (`RepairHooks::copy_bound_check`) — counts the tombstone a
/// record's children normalization leaves in its tombstone bound.
///
/// The object gains 48 parents under the root, each listing one child, so
/// its children table holds 50 lists: the root's, World's and the parents'.
/// On the copy that table is laid out again, same contents, in a map
/// reserved for 56 (64 buckets) whose keys sit on home slots 0 to 49 under
/// its own hasher, inserted in slot order: one contiguous run of full
/// buckets longer than 16, so an erase anywhere in it leaves a tombstone at
/// hash group width 8 and 16 alike (an erase leaves EMPTY only inside a run
/// shorter than a group). The parents' ids are picked by home slot under
/// that hasher before they are committed; the hasher is redrawn until the
/// root's and World's keys take distinct slots below 50. One record then
/// re-parents the first parent's only child to the root
/// (`DiffChangeSection`): nothing else removes from the table, and the
/// normalization after the record drops the emptied list, leaving one
/// tombstone at the bound's 64 buckets. The bound must count it (`t ≥ T`);
/// with the normalization left out of the record's removals the guard
/// panics, naming the table and both counts.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn repair_to_target_of_a_copy_counts_the_tombstone_a_normalized_list_leaves_in_its_bound() {
    use std::collections::HashMap;
    use std::hash::BuildHasher;

    use live_edit::domain::types::true_capacity_for_test;

    /// The lists on the run: home slots 0 to 49 of 64 buckets.
    const RUN_KEYS: usize = 50;
    /// The parents' children's ids: parent `k`'s is `CHILD_FIRST + k`.
    const CHILD_FIRST: u64 = 2_000_000;

    let fx = Fx::new("scr-copy-bound-normalize").await;
    let mut run: HashMap<u64, Vec<u64>> = loop {
        let map: HashMap<u64, Vec<u64>> = HashMap::with_capacity(56);
        let home = |id: u64| (map.hasher().hash_one(id) & 63) as usize;
        let (root, world) = (home(ROOT), home(WORLD));
        if root < RUN_KEYS && world < RUN_KEYS && root != world {
            break map;
        }
    };
    let hasher = run.hasher().clone();
    let home = |id: u64| (hasher.hash_one(id) & 63) as usize;
    let mut slots: Vec<Option<u64>> = vec![None; RUN_KEYS];
    slots[home(ROOT)] = Some(ROOT);
    slots[home(WORLD)] = Some(WORLD);
    let mut parents = Vec::new();
    let mut candidate = 1_000_000u64;
    while slots.iter().any(Option::is_none) {
        let slot = home(candidate);
        if slot < RUN_KEYS && slots[slot].is_none() {
            slots[slot] = Some(candidate);
            parents.push(candidate);
        }
        candidate += 1;
    }
    let mut creates = Vec::new();
    for (k, &parent) in parents.iter().enumerate() {
        creates.push(create_cmd(ROOT, parent, &format!("p{parent}")));
        creates.push(create_cmd(parent, CHILD_FIRST + k as u64, "x"));
    }
    fx.commit(structural(creates)).await;
    let warm = fx.warm().await;

    // The copy, its children table laid out on the run.
    let mut copy = warm.copy_without_plan();
    let mut lists = std::mem::take(&mut copy.state.children);
    for &key in slots.iter().flatten() {
        let list = lists
            .remove(&key)
            .unwrap_or_else(|| panic!("fixture: a list under run key {key}"));
        run.insert(key, list);
    }
    assert!(
        lists.is_empty(),
        "fixture: every list is on the run: {lists:?}"
    );
    copy.state.children = run;
    assert_eq!(
        (
            copy.state.children.len(),
            copy.state.children.capacity(),
            true_capacity_for_test(&copy.state.children)
        ),
        (RUN_KEYS, 56, 56),
        "fixture: 50 lists in 64 buckets, no tombstone"
    );

    fx.commit(structural(vec![StructCommand::DiffChangeSection(
        CommandDiffChangeSection {
            parent_id: ROOT,
            section_id: CHILD_FIRST,
            section_name: "moved".to_owned(),
            section_type: 1,
            section_order: 0,
        },
    )]))
    .await;
    let target_seq = fx.header_seq().await;
    let target_chain = header(&fx.tv, fx.oid).await[HEADER_FIELD_COMMIT_CHAIN].clone();
    let charge = RepairCharge::granting();

    let (answer, stats) = repair_to_target(
        fx.tv.client(),
        fx.tv.key_schema(),
        &mut copy,
        target_seq,
        &target_chain,
        RepairBounds::PRODUCTION,
        Some(&charge),
        None,
    )
    .await
    .expect("the repair reads");

    assert!(matches!(answer, TargetRepair::Reached), "got {answer:?}");
    assert_eq!(stats.replayed_entries, 1, "fixture: one record replayed");
    let children = &copy.state.children;
    assert!(
        !children.contains_key(&parents[0]),
        "fixture: the normalization dropped the emptied list"
    );
    assert_eq!(
        (
            children.len(),
            true_capacity_for_test(children) - children.capacity()
        ),
        (RUN_KEYS - 1, 1),
        "fixture premise on this platform: the dropped list left a tombstone in the run"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 8. The per-call reload cap after a cold start
// ═════════════════════════════════════════════════════════════════════════════

/// A hook, run after every cold start that loaded an entry, that lands a
/// commit which does not fold `commit_chain` — an older replica's during a
/// rolling deploy — on every run, counting its runs. Each such commit makes
/// the repair after that cold start reload `regressed`.
fn non_folding_commit_after_every_cold_start(
    fx: &Arc<Fx>,
    runs: Arc<AtomicUsize>,
) -> AfterColdStartHook {
    let fx = Arc::clone(fx);
    Arc::new(move || {
        let (fx, runs) = (Arc::clone(&fx), Arc::clone(&runs));
        async move {
            runs.fetch_add(1, Ordering::SeqCst);
            commit_without_fold(&fx.tv, fx.oid).await;
        }
        .boxed()
    })
}

/// One call whose every cold start meets a non-folding commit before its
/// repair. Returns the answer and how many times the hook ran.
///
/// Bounded by [`RELOAD_CAP_BUDGET`]: without the cap the call would reload
/// until its deadline, so a missing cap fails here with a diagnosis rather
/// than at the hang-killer.
async fn read_with_a_non_folding_commit_after_every_cold_start(
    fx: &Arc<Fx>,
) -> (Result<StructuralRead, ServiceError>, usize) {
    let runs = Arc::new(AtomicUsize::new(0));
    let request =
        StructuralReadRequest::new(fx.oid, CLIENT_ID, TOKEN).with_after_cold_start_hook_for_test(
            non_folding_commit_after_every_cold_start(fx, Arc::clone(&runs)),
        );
    let answer = tokio::time::timeout(RELOAD_CAP_BUDGET, fx.read_request(&request))
        .await
        .expect(
            "the call must end at the reload cap; still reloading at the budget means the cap \
             is gone",
        );
    (answer, runs.load(Ordering::SeqCst))
}

/// The cap: a cold start's repair that keeps failing reloads
/// [`STRUCTURAL_READ_MAX_RELOADS_AFTER_COLD_START`] times, then the call
/// answers `Uncacheable(ReloadLimit)`. Every failure is counted as the
/// reload it is (`regressed`), and none is an uncacheable cold-start verdict.
/// The call's own verdict is reported once: one `reload_limit` tick and one
/// line naming the object, with no bounds — at DEBUG, not INFO: the verdict
/// is never recorded, so it can recur on every edit of the layer and nothing
/// but the counter may carry its rate (SDD §17.2's log-volume rules).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_repair_failing_after_every_cold_start_answers_reload_limit_at_the_cap(
) {
    let fx = Arc::new(Fx::new("scr-reload-cap").await);
    let cold_starts = STRUCTURAL_READ_MAX_RELOADS_AFTER_COLD_START as usize + 1;

    let ((answer, runs), captured) =
        capture_debug(read_with_a_non_folding_commit_after_every_cold_start(&fx)).await;

    assert_eq!(
        uncacheable_verdicts(&fx.metrics),
        counts([("reload_limit", 1)]),
        "the capped call's verdict: one reload_limit tick, and no other verdict"
    );
    let mut absent = CEILING_FIELDS.to_vec();
    absent.extend(WEIGHT_FIELDS);
    absent.extend(LAYER_FIELDS);
    assert_one_verdict_line_at(&captured, "DEBUG", fx.oid, "reload_limit", &[], &absent);

    assert!(
        matches!(
            answer,
            Ok(StructuralRead::Uncacheable {
                reason: UncacheableReason::ReloadLimit,
                above_ceiling: None,
            })
        ),
        "expected Uncacheable(ReloadLimit); got {answer:?}"
    );
    assert_eq!(
        runs, cold_starts,
        "one cold start, then one per allowed reload ({cold_starts} at a cap of \
         {STRUCTURAL_READ_MAX_RELOADS_AFTER_COLD_START})"
    );
    assert_eq!(
        fx.reloads(),
        counts([("regressed", cold_starts as u64)]),
        "every failed repair after a cold start is a regressed reload"
    );
    assert_eq!(
        fx.cold_starts("uncacheable"),
        0,
        "the cap is not an uncacheable cold-start verdict"
    );
    assert_eq!(fx.lookups(), counts([("cold", 1)]), "one lookup, a miss");
}

/// The cap gives up for one call only: it records no verdict and no negative
/// entry — the object looks up as a `Miss`, not `Uncacheable` — and the next
/// call, with no commit racing it, cold-starts the object and caches it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_for_structural_edit_reload_limit_records_nothing_and_the_next_call_caches_the_object()
{
    let fx = Arc::new(Fx::new("scr-reload-cap-next").await);
    let (answer, _) = read_with_a_non_folding_commit_after_every_cold_start(&fx).await;
    assert!(
        matches!(
            answer,
            Ok(StructuralRead::Uncacheable {
                reason: UncacheableReason::ReloadLimit,
                above_ceiling: None,
            })
        ),
        "fixture: the capped call answers ReloadLimit; got {answer:?}"
    );

    match fx.cache.lookup(fx.oid) {
        CacheLookup::Miss => {}
        other => {
            panic!("ReloadLimit records nothing: the object must look up as a Miss; got {other:?}")
        }
    }
    let next = expect_ready(fx.read().await, "the call after the cap");

    assert_eq!(
        next.path,
        StructuralPath::StructuralFull,
        "the next call cold-starts the object"
    );
    fx.assert_current(next.entry.entry(), "the next call's entry")
        .await;
    assert!(
        next.entry.put_back_held().is_stored(),
        "its entry is stored"
    );
    assert!(fx.cached().is_some(), "the object is cached: a hit");
    assert_eq!(
        fx.cold_starts("uncacheable"),
        0,
        "no uncacheable verdict on either call"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 9. The successor entry of a real structural commit
// ═════════════════════════════════════════════════════════════════════════════

/// What a commit does to the checked-out entry's dangling parents.
#[derive(Debug, Clone, Copy)]
enum DanglingChange {
    Unchanged,
    Removes(u64),
}

/// A structural call's own path across one real commit: check the object out
/// (a cold start), plan `delta` on a clone of the entry with `apply_delta`,
/// commit the same delta through `update_layer`, and build the successor from
/// the plan's mutations at the commit's `(seq, commit_chain)`. Returns the
/// guard, the entry as checked out, the plan's mutations and the successor.
async fn plan_commit_and_build_successor(
    fx: &Fx,
    delta: OwnedDelta,
    context: &str,
) -> (
    ReadyEntry,
    CacheEntry,
    Vec<StorageMutation>,
    Result<CacheEntry, ReloadReason>,
) {
    let ready = expect_ready(fx.read().await, context);
    let predecessor = ready.entry.entry().clone();
    let mut planned = ready
        .entry
        .plan_state(live_edit::storage::structural_cache::PlannedInserts::default());
    let mutations = apply_delta(&mut planned, Some(delta.clone()))
        .unwrap_or_else(|e| panic!("{context}: the plan applies: {e:?}"));

    let (seq, _) = fx.commit(delta).await;
    assert_eq!(
        seq,
        predecessor.seq + 1,
        "{context}: fixture: the commit lands right after the planned cut"
    );
    let commit_chain = header(&fx.tv, fx.oid).await[HEADER_FIELD_COMMIT_CHAIN].clone();
    let successor = successor_entry(&predecessor, planned, &mutations, seq, commit_chain);
    (ready, predecessor, mutations, successor)
}

/// SR-3.4 on `Committed`: the successor built from a real structural
/// commit's planned state and mutations is the stored cut at the commit's
/// `(seq, commit_chain)` — the normalized fresh state and the store's
/// dangling ids — whether the commit leaves the dangling parents alone (a
/// creation under a live parent; an orphan's creation, which stores no list)
/// or removes one ([`create_then_delete_dangling_parent_delta`], whose
/// cascade's `DeleteChildrenList` deletes the stored list). No commit adds
/// one: no list is stored under a parent with no row. Swapped in, it is the
/// next call's hit, with nothing to replay.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn successor_entry_of_a_real_structural_commit_is_the_stored_cut() {
    let cases = [
        (
            "a creation under a live parent",
            structural(vec![create_cmd(WORLD, NEW_SECTION, "New")]),
            DanglingChange::Unchanged,
        ),
        (
            "an orphan's creation",
            orphan_delta(),
            DanglingChange::Unchanged,
        ),
        (
            "a create-then-delete of the dangling parent",
            create_then_delete_dangling_parent_delta(),
            DanglingChange::Removes(DANGLING_PARENT),
        ),
    ];

    for (context, delta, change) in cases {
        let fx = Fx::new("scr-successor").await;

        let (ready, predecessor, mutations, successor) =
            plan_commit_and_build_successor(&fx, delta, context).await;
        let successor =
            successor.unwrap_or_else(|e| panic!("{context}: no adoption, so a successor; {e:?}"));

        let mut expected = predecessor.dangling_parents.clone();
        match change {
            DanglingChange::Unchanged => {}
            DanglingChange::Removes(parent) => {
                assert!(
                    mutations.iter().any(|m| matches!(
                        m,
                        StorageMutation::DeleteChildrenList { section_id } if *section_id == parent
                    )),
                    "{context}: fixture: the plan deletes the dangling parent's list with a \
                     DeleteChildrenList; got {mutations:?}"
                );
                assert!(
                    expected.remove(&parent),
                    "{context}: fixture: {parent} is dangling before the commit"
                );
            }
        }
        assert_eq!(
            successor.dangling_parents, expected,
            "{context}: the commit's change to the dangling parents"
        );
        fx.assert_current(&successor, context).await;

        assert!(
            ready.entry.put_back(successor).is_stored(),
            "{context}: the successor's put-back succeeds"
        );
        let next = expect_ready(fx.read().await, &format!("{context}: the next call"));
        assert_eq!(
            (next.path, next.repair.replayed_entries),
            (StructuralPath::Structural, 0),
            "{context}: the next call hits the successor with nothing to replay"
        );
    }
}

/// SR-3.2 on `Committed`: a real commit whose plan creates a section with a
/// dangling parent's id adopts it — the stored list it makes visible is not
/// in the plan — so there is no successor (`DanglingAdopted`). The caller
/// discards the guard, and the next call cold-starts an entry that is the
/// stored cut, the adopted list included.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn successor_entry_of_a_real_commit_adopting_a_dangling_parent_is_refused() {
    const CONTEXT: &str = "adoption";
    let fx = Fx::new("scr-successor-adopted").await;

    let (ready, predecessor, _, successor) = plan_commit_and_build_successor(
        &fx,
        structural(vec![create_cmd(ROOT, DANGLING_PARENT, "Adopter")]),
        CONTEXT,
    )
    .await;

    assert!(
        predecessor.dangling_parents.contains(&DANGLING_PARENT),
        "fixture: the created id is a dangling parent of the entry"
    );
    assert!(
        matches!(successor, Err(ReloadReason::DanglingAdopted)),
        "a plan adopting a dangling parent has no successor; got {successor:?}"
    );
    ready.entry.discard();
    let next = expect_ready(fx.read().await, "the call after the adoption");
    assert_eq!(
        next.path,
        StructuralPath::StructuralFull,
        "the discarded object is read again"
    );
    fx.assert_current(next.entry.entry(), "after the adoption")
        .await;
    assert!(
        !next
            .entry
            .entry()
            .dangling_parents
            .contains(&DANGLING_PARENT),
        "the adopted parent is live, not dangling"
    );
}
