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

//! The cached structural attempt end to end: structural and reset
//! `UpdateLayer` calls on the `lua` backend, through `LayerStore::update_layer`
//! (plan `b.dkt` Epic 6 Task 6; structural-edits Spec SDD SR-7.3, SR-7.4,
//! SR-7.6, SR-3.2, SR-3.4–SR-3.6, SR-8.1, SR-10.4).
//!
//! Every structural call here reads through the structural cache's read side
//! and — unless that answers uncacheable or the call falls back — plans in
//! place on the held entry under a pre-image journal, commits through
//! `commit_structural` and settles the entry by the outcome: a commit keeps
//! the planned entry, every other exit rolls the plan back before the entry
//! goes back (`LayerStore::update_layer_once_structural`). Objects
//! are `snapshot_read_atomic_test`'s `every_shape_state` (a list under the
//! dangling parent `DANGLING_PARENT`) created through `create_layer`; every
//! commit sends its delta's OMPF as relay bytes, as a production commit does,
//! so a later repair can replay it. One registry is hooked onto both the
//! store (`with_metrics`: commit metrics, divergence, fallbacks, put-back and
//! own-adoption reloads) and its client (the read side's lookups, reloads and
//! cold-start reads, and the audit), so every count the route moves is read
//! from one place.
//!
//! | Test | What it pins |
//! |---|---|
//! | [`structural_calls_on_one_object_on_one_replica_serialize_with_no_seq_fence_conflict`] | same-replica serialization: concurrent structural calls on one object all commit, none meets a `seq` fence |
//! | [`structural_commit_racing_a_fast_writer_is_stale_repaired_and_commits_once`] | `STALE` under a racing fast-path writer: every call commits exactly once, `seq_fence` ticks on the structural paths only, the retry repairs (never a reload), one more call leaves the entry the stored cut, and each call reads its start record once whatever its `STALE` re-plans |
//! | [`structural_stale_on_a_cold_started_attempt_ticks_on_structural_full_and_the_retry_repairs_one_entry`] | `STALE` staged deterministically (the pre-dispatch hook lands another replica's commit): nothing written, the original entry put back exactly as cold-started, `seq_fence` on `structural_full`, and the retry repairs exactly the one landed entry and commits on `structural` |
//! | [`structural_reset_stale_ticks_on_the_reset_path`] | `seq_fence` on a reset ticks on `reset` |
//! | [`structural_replan_after_cache_diverged_carries_the_call_start_record_and_is_overtaken`] | a re-plan carries the call's one start record: its client's next commit, landed between the attempts, makes the re-plan `Overtaken`, applying nothing |
//! | [`structural_lost_reply_is_answered_by_the_retrys_token_check_and_applies_nothing_twice`] | a lost reply: the retry's read side answers the landed commit before planning, a repeated `CreateSection` included; one commit, one already-landed count |
//! | [`structural_already_landed_answers_build_the_shared_own_commit_result`] | the read side's already-landed answer becomes the result every route's own-commit constructor builds (`APPLIED` with an operation, `UNSPECIFIED` without), as the fast path's |
//! | [`structural_commit_already_landed_answer_is_the_own_commit_result_and_puts_the_entry_back`] | the commit script's own `ALREADY_LANDED` (the record hidden from the retry's read and restored before its commit) becomes the own-commit result (`APPLIED`), counted once; the repaired entry is put back |
//! | [`structural_commit_recorded_by_an_earlier_call_answers_already_committed_and_puts_the_entry_back`] | the commit script's `Recorded` (an earlier call of the operation lands between plan and commit): `ALREADY_COMMITTED` with that call's commit, nothing applied, the original entry put back |
//! | [`structural_commit_overtaken_answers_unavailable_on_the_lua_backend_and_puts_the_entry_back`] | the commit script's `OVERTAKEN` (the client's next commit lands between plan and commit): `UNAVAILABLE`, `operation_overtaken_total{backend="lua"}`, nothing applied, the original entry put back |
//! | [`structural_commit_refused_not_found_or_superseded_drops_the_entry`] | the commit script's `NOT_FOUND` / `SUPERSEDED` (another replica's delete or overwrite between plan and commit): the error, and the entry dropped |
//! | [`structural_resend_with_the_same_operation_id_is_answered_already_committed_and_writes_nothing`] | the decider's op-id structural regression: a re-send answers `OK` with the original commit, `ALREADY_COMMITTED`, applying and writing nothing; a different request under the id is `ALREADY_EXISTS` |
//! | [`structural_cache_diverged_writes_nothing_drops_the_entry_and_the_retry_reloads_and_commits`] | `CACHE_DIVERGED` via `rewrite_out_of_band`: nothing written, `divergence_total` and `cache_diverged` (on the hit's `structural` path) tick, the entry is dropped and the retry reloads and commits on the rewritten state |
//! | [`structural_reset_cache_diverged_ticks_on_the_reset_path`] | `cache_diverged` on a reset ticks on `reset` |
//! | `structural_divergence_on_*_reloads_once_then_commits_through_watch_and_never_loops` | divergence never loops: per known shape (a non-canonical list element, a section row outside `obj:secs`, a wrong-typed children list outside `obj:chs`), one reload, then a WATCH-path commit and one `repeat_divergence` fallback; the next call commits on the cache with no further divergence |
//! | [`structural_identity_fence_after_a_dr_restore_between_two_edits_forces_a_reload`] | the identity fence end to end: two DR restores of one checkpoint generation between two edits; the second edit reloads `identity` and commits, never a write from the stale copy |
//! | [`rollback_knob_flipped_between_commits_keeps_the_invariants_and_never_reloads`] | the rollback knob flipped between commits (SR-7.5): stores on `lua`, `lua-watch-structural` and `txn` commit one object in turn; after every commit the header `commit_chain` is the fold over the stream payloads, `obj:chs` ⟺ lists and `chs_complete` holds, and every `lua` structural call repairs across the others' commits with no reload |
//! | [`structural_call_on_an_unmarked_object_commits_through_watch_and_ticks_uncacheable_once`] / [`structural_call_on_an_object_over_the_entry_cap_commits_through_watch_and_ticks_uncacheable_once`] | uncacheable routing: `path="watch"`, the `uncacheable` cold-start route once, and the next call routed by the negative entry |
//! | [`structural_commit_whose_successor_outgrows_the_entry_cap_is_counted_uncacheable`] | the C3 amendment: a successor the per-entry cap refuses ticks `reloads_total{reason="uncacheable"}` and leaves the object uncacheable, reported once as an `over_weight` verdict naming the call's session and layer; the next call, routed by it, reports nothing |
//! | [`structural_call_dropped_after_its_commit_landed_rolls_back_and_the_next_call_repairs_ug2`] | a call dropped after its commit landed, its plan in place still pending: the guard's `Drop` rolls the plan back and puts the pre-plan cut back (one cut behind the store, `rollbacks_for_test` +1); the next call repairs across the landed commit and leaves the entry at the stored cut |
//! | `structural_rollback_panic_*_discards_the_entry_and_the_next_call_cold_loads_ug2` (two) | T8, a panic in the rollback: with the failpoint's panic mode armed, a rollback that panics part way — at the put-back (a delta `Err`) and in a dropped guard's `Drop` (a call dropped at its commit's landing) — leaves the call as the failpoint's panic; the entry is discarded and the object invalidated (lookup miss, no checkout, no weight hint), and the next call cold-loads the stored cut (the before-admit hook once, the entry the header's cut and the normalized fresh state), then a call commits on the hit |
//! | [`structural_call_dropped_mid_commit_over_the_entry_cap_is_reported_once_by_the_next_lookup`] | a call dropped mid-commit holding an entry its repair grew past the cap: the guard's `Drop` records an unreported `OverWeight` verdict, the next call's lookup reports it once (one tick, one INFO line with session, layer, weight and cap), and the call after reports nothing |
//! | `structural_held_entry_repaired_past_the_entry_cap_is_counted_uncacheable_after_*` | the C3 amendment's other half, one test per settle arm that puts the held entry back (a delta `Err`, `STALE`, `FENCED`, the commit script's `Recorded`, `OVERTAKEN` and `ALREADY_LANDED`, a transport error): a held entry its repair grew past the cap ticks `reloads_total{reason="uncacheable"}` once and leaves an `OverWeight` negative entry |
//! | `structural_put_back_*` | the put-back by outcome (SR-3.4): the original entry on a delta `Err`, a transport error, an in-doubt reply and a stream-ID collision; nothing on `NotFound` / `SUPERSEDED` from another store's delete or overwrite; a successor refused by the epoch check after a local `DeleteLayer` or overwrite during the checkout |
//! | `structural_audit_*` | the audit trigger at rate 1.0: audits of a cold-start commit's and a hit commit's normalized successors count `match` and drop nothing; an own adoption starts none |
//! | [`structural_own_adoption_drops_the_entry_and_the_next_call_cold_starts`] | own adoption (SR-3.2): the entry is dropped (`dangling_adopted`), and the next call cold-starts the stored cut |
//! | `structural_fenced_*` | `FENCED` (the test hook): an uncounted wait of at least the reply's `retry_after_ms` on the attempt's path, nothing written, the entry put back unchanged, then a commit; a fence held past `LIVE_EDIT_RPC_DEADLINE_MS` (by one long fence, or by several short ones together) or the remaining deadline fails `UNAVAILABLE` |
//!
//! Elsewhere: the per-path commit metrics (`structural_full`, `structural`,
//! `reset`, and `watch` on `lua-watch-structural`) in
//! `update_layer_commit_metrics_test.rs`; the structural route's commit and
//! its read-side already-landed answer counted and logged exactly once — to
//! completion and dropped at the landing — in
//! `update_layer_commit_count_test.rs`; the admission permits in
//! `update_layer_admission_test.rs`; the structural route's stream-ID
//! collision refusal in `update_layer_lua_test.rs`.
//!
//! **The commit-side outcomes are staged, not raced.** The structural
//! pre-dispatch hook (`LayerStore::with_structural_commit_pre_dispatch_hook_for_test`)
//! runs after an attempt has planned and sized its commit and before the
//! `EVALSHA`, so a test changes the store in exactly that window ([`Staged`]):
//! another replica's commit (`STALE`), an earlier call of the operation
//! (`Recorded`), the client's next commit (`OVERTAKEN`), a delete or an
//! overwrite (`NOT_FOUND` / `SUPERSEDED`), or this call's own record put back
//! (`ALREADY_LANDED`). The race test stays as the undirected twin of the
//! staged `STALE`.
//!
//! # Wait budget
//!
//! Every call is under [`CALL_BUDGET`] (15 s) and no test but the race makes
//! more than six (90 s). The race holds one [`RACE_BUDGET`] (60 s) deadline
//! across its whole loop, then joins its writer and makes one more call (two
//! `CALL_BUDGET`s: 90 s in all). The audit waits and the epoch test's wait
//! for its parked call are each one [`WAIT_BUDGET`] (10 s), at most two per
//! test beside at most four calls (80 s). A staged action runs inside the
//! call it disturbs, so its own commit is inside that call's budget. All
//! under nextest's 120 s hang-killer; the healthy path takes milliseconds,
//! except the fence waits (one [`FENCE_RETRY_AFTER_MS`] sleep; at most two
//! [`FENCE_RETRY_AFTER_SHORT_MS`] sleeps in the cumulative-cap case; one in
//! the held-entry test's `FENCED` case) and the race (well under a second
//! locally). Each held-entry test makes its probe call, then at most four
//! calls (its caching call, the growth commit, the call under test and one
//! staged commit): five `CALL_BUDGET`s, 75 s.
//!
//! Gated behind `test-fault-injection` (see `mod.rs`): the fault queues, the
//! hooks and the scripted `FENCED` replies exist only under that feature.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use fred::clients::Client as FredClient;
use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
use fred::interfaces::{HashesInterface, KeysInterface, ListInterface, StreamsInterface};
use futures::future::BoxFuture;
use futures::FutureExt;
use oodf_ffi::OwnedDelta;
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::apply::apply_checkpoint;
use live_edit::config::UpdateLayerBackend;
use live_edit::domain::delta::failpoint;
use live_edit::domain::types::{Field, FieldValue, ObjectState};
use live_edit::error::{refusal_reason, ServiceError, CAPACITY_RETRY_DELAY};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::commit_chain::commit_chain_fold;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::{
    operation_request_digest, BetweenAttemptsHook, CommitLandedHook, LayerStore, Operation,
    OperationOutcome, UpdateCommit, CHS_COMPLETE_MARKED, HEADER_FIELD_CHS_COMPLETE,
    OPERATION_OVERTAKEN_MESSAGE,
};
use live_edit::storage::structural_cache::{
    CacheEntry, CacheLookup, StructuralCache, UncacheableReason,
};
use live_edit::storage::structural_cache_audit::StructuralCacheAudit;
use live_edit::storage::test_support::structural_commit::{
    dump_object_keyspace, CommitTokens, ObjectKeyspaceDump,
};
use live_edit::storage::test_support::{rewrite_out_of_band, OutOfBandChange, TestValkey};
use live_edit::storage::watched_txn::WatchWindowHook;

use crate::common::owned_delta::{
    delta_to_ompf_bytes, delta_with_new_field_names, delta_with_set_field_inline, empty_delta,
    reset_delta,
};
use crate::common::refusal_detail::assert_refusal_error;
use crate::grpc::log_capture::capture;
use crate::storage::checkpoint_apply_test::build_checkpoint_ompf;
use crate::storage::snapshot_read_atomic_test::{
    create, create_cmd, create_session, delete_cmd, every_shape_state, indexed_parents,
    listed_parents, move_cmd, structural, unmark, CUBE, DANGLING_PARENT, ROOT, SIBLING, WORLD,
};
use crate::storage::structural_cache_cold_start_test::{
    assert_one_verdict_line, header, normalized_fresh_state, stored_dangling_parents,
    uncacheable_verdicts, CEILING_FIELDS, UNCACHEABLE_VERDICT_MSG,
};

// ═════════════════════════════════════════════════════════════════════════════
// Budgets and constants
// ═════════════════════════════════════════════════════════════════════════════

/// Deadline on one `update_layer` call. The healthy path is a few round trips
/// and at most a few millisecond-scale backoffs.
const CALL_BUDGET: Duration = Duration::from_secs(15);
/// Deadline on the whole `STALE` race loop.
const RACE_BUDGET: Duration = Duration::from_secs(60);
/// Deadline on a background condition: an audit ending, a hook being reached.
const WAIT_BUDGET: Duration = Duration::from_secs(10);
/// Poll step while waiting for a background condition.
const POLL: Duration = Duration::from_millis(5);
/// Retry budget of every store: small, so a regression that loops on
/// `Aborted` fails inside the call budget.
const MAX_ATTEMPTS: u32 = 8;
/// A structural cache budget no fixture object comes near.
const BUDGET_BYTES: u64 = 64 * 1024 * 1024;
/// A cache budget whose per-entry cap (a quarter of it) every fixture object
/// is over.
const TINY_BUDGET_BYTES: u64 = 1_024;

/// The writer of every call under test.
const WRITER: &str = "dispatch-writer";
/// Another writer, with a different `client_id`.
const OTHER_WRITER: &str = "dispatch-other-writer";
/// The `client_id` prefix of the `STALE` race's structural calls, one
/// `client_id` per call.
const RACE_CLIENT: &str = "dispatch-race-call-";
const CLIENT_TS: u64 = 9_000;
/// Section ids the seeded tree does not use.
const NEW_SECTION: u64 = 70;
const OTHER_NEW_SECTION: u64 = 71;
/// The scripted `FENCED` reply's `retry_after_ms`: well under the fence-wait
/// cap, and long enough that a call which skipped the wait would be seen to
/// return sooner than this (its healthy path is a few milliseconds).
const FENCE_RETRY_AFTER_MS: u64 = 150;
/// A `retry_after_ms` whose sleep (`min(retry_after, poll)`, ~1 s, on the
/// call's `FenceWaitClock`) crosses the narrowed caps of the cap tests
/// before it is taken.
const FENCE_RETRY_AFTER_LONG_MS: u64 = 1_000;
/// The cumulative-cap case's `retry_after_ms`, and its cap: each sleep is far
/// under the cap, but the third would end past it — two sleeps end near
/// 200 ms, inside the cap with a margin of about 50 ms for the re-reads, a
/// third near 300 ms.
const FENCE_RETRY_AFTER_SHORT_MS: u64 = 100;
const FENCE_CUMULATIVE_CAP: Duration = Duration::from_millis(250);

/// `live_edit_update_layer_*` `path` and `kind` label values. Restated, not
/// imported: they are the operator-facing contract.
const STRUCTURAL: &str = "structural";
const STRUCTURAL_FULL: &str = "structural_full";
const RESET: &str = "reset";
const WATCH: &str = "watch";
const SEQ_FENCE: &str = "seq_fence";
const CACHE_DIVERGED: &str = "cache_diverged";
const SNAPSHOT_FENCE_WAIT: &str = "snapshot_fence_wait";
/// Every `path` and every `kind` a conflict can carry.
const PATHS: [&str; 5] = ["fast", WATCH, STRUCTURAL, STRUCTURAL_FULL, RESET];
const CONFLICT_KINDS: [&str; 5] = [
    "watch_conflict",
    "watch_read_moved",
    SEQ_FENCE,
    CACHE_DIVERGED,
    SNAPSHOT_FENCE_WAIT,
];

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// One object, a `lua` store over it with its own structural cache, and one
/// registry hooked onto the store and its client.
struct Fx {
    tv: TestValkey,
    metrics: Arc<Metrics>,
    cache: StructuralCache,
    ls: Arc<LayerStore>,
    session_id: String,
    layer: &'static str,
    oid: u64,
}

impl Fx {
    /// `every_shape_state` on a default store.
    async fn new(layer: &'static str) -> Self {
        Self::with(layer, &every_shape_state(), BUDGET_BYTES, |ls| ls).await
    }

    /// `state` on a store with a cache of `budget_bytes`, built by
    /// `configure` from a `lua` store holding that cache, this registry and
    /// [`MAX_ATTEMPTS`].
    async fn with(
        layer: &'static str,
        state: &ObjectState,
        budget_bytes: u64,
        configure: impl FnOnce(LayerStore) -> LayerStore,
    ) -> Self {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only(layer));
        tv.client()
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let cache = StructuralCache::new(budget_bytes);
        let ls = Arc::new(configure(
            LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
                .with_update_layer_backend(UpdateLayerBackend::Lua)
                .with_update_layer_retry_max_attempts(MAX_ATTEMPTS)
                .with_structural_cache(cache.clone())
                .with_metrics(Arc::clone(&metrics)),
        ));
        let session_id = create_session(&tv, layer).await;
        let oid = create(&ls, &session_id, layer, state).await;
        Self {
            tv,
            metrics,
            cache,
            ls,
            session_id,
            layer,
            oid,
        }
    }

    /// Another replica's store: the `lua` backend, its own cache, no
    /// store-side metrics (its client still reports into this registry).
    fn replica(&self) -> LayerStore {
        LayerStore::new(self.tv.client().clone(), self.tv.key_schema().clone(), 4096)
            .with_update_layer_backend(UpdateLayerBackend::Lua)
            .with_update_layer_retry_max_attempts(MAX_ATTEMPTS)
    }

    /// One `update_layer` call by `client_id` through `ls`, relaying the
    /// delta's OMPF, under [`CALL_BUDGET`].
    async fn update_on(
        &self,
        ls: &LayerStore,
        delta: OwnedDelta,
        client_id: &str,
    ) -> Result<(u64, u64), ServiceError> {
        let relay = Some(Bytes::from(delta_to_ompf_bytes(&delta, &[])));
        tokio::time::timeout(
            CALL_BUDGET,
            ls.update_layer(
                &self.session_id,
                self.layer,
                self.oid,
                Some(delta),
                relay,
                client_id,
                CLIENT_TS,
            ),
        )
        .await
        .expect("update_layer did not settle within the call budget")
    }

    /// [`Self::update_on`] through the fixture's store, as [`WRITER`].
    async fn update(&self, delta: OwnedDelta) -> Result<(u64, u64), ServiceError> {
        self.update_on(&self.ls, delta, WRITER).await
    }

    /// One `update_layer_with_operation` call through `ls`, relaying the
    /// delta's OMPF, under [`CALL_BUDGET`].
    async fn update_with_op(
        &self,
        ls: &LayerStore,
        delta: OwnedDelta,
        client_id: &str,
        operation: Option<Operation>,
    ) -> Result<UpdateCommit, ServiceError> {
        update_with_op_on(
            ls,
            &self.session_id,
            self.layer,
            self.oid,
            delta,
            client_id,
            operation,
        )
        .await
    }

    /// An operation on this fixture's layer with id `operation_id` over
    /// `payload`.
    fn op(&self, operation_id: &str, payload: &[u8]) -> Operation {
        Operation::new(operation_id, operation_request_digest(self.layer, payload))
    }

    async fn header_seq(&self) -> u64 {
        header(&self.tv, self.oid).await["seq"]
            .parse()
            .expect("the header seq is decimal")
    }

    /// The layer stream as `(id, payload)`, in order. Fields are read as
    /// bytes: a relayed `ompf` is binary.
    async fn stream(&self) -> Vec<(String, String)> {
        let entries: Vec<(String, std::collections::HashMap<String, Bytes>)> = self
            .tv
            .client()
            .pool
            .xrange_values(self.tv.key_schema().layer_stream(self.oid), "-", "+", None)
            .await
            .expect("XRANGE over the layer stream must succeed");
        entries
            .into_iter()
            .map(|(id, fields)| {
                let payload = fields
                    .get("payload")
                    .map(|p| String::from_utf8_lossy(p).into_owned())
                    .unwrap_or_default();
                (id, payload)
            })
            .collect()
    }

    /// The layer stream's entry ids, in order.
    async fn stream_ids(&self) -> Vec<String> {
        self.stream().await.into_iter().map(|(id, _)| id).collect()
    }

    /// The payloads of the layer stream's commits by `client_id`
    /// (`update|{oid}|{seq}|{client_id}|{client_ts}|{server_ts}`), in order.
    /// CreateLayer's entry names no client and is never among them.
    async fn commits_by(&self, client_id: &str) -> Vec<String> {
        self.stream()
            .await
            .into_iter()
            .map(|(_, payload)| payload)
            .filter(|p| p.starts_with("update|") && p.split('|').nth(3) == Some(client_id))
            .collect()
    }

    /// The number of the layer stream's commits by any client whose id
    /// starts with `prefix`.
    async fn commits_by_prefix(&self, prefix: &str) -> usize {
        self.stream()
            .await
            .into_iter()
            .filter(|(_, p)| {
                p.starts_with("update|")
                    && p.split('|').nth(3).is_some_and(|c| c.starts_with(prefix))
            })
            .count()
    }

    /// The object's whole keyspace — every key under its tag, commit-token
    /// records verbatim — and its header in full.
    async fn keyspace(&self) -> (ObjectKeyspaceDump, BTreeMap<String, String>) {
        let dump = dump_object_keyspace(
            self.tv.client(),
            self.tv.key_schema(),
            self.oid,
            CommitTokens::Verbatim,
        )
        .await;
        assert!(
            dump.get("obj:hdr").is_some() && dump.get("obj:stream").is_some(),
            "fixture: the dump must hold the header and the stream, or a before / after \
             equality over it is vacuous"
        );
        (dump, header(&self.tv, self.oid).await.into_iter().collect())
    }

    /// The object's cache state: its entry (a hit, put back at once by the
    /// guard's drop), a miss, or its uncacheable verdict.
    fn cached(&self) -> Cached {
        cached_in(&self.cache, self.oid)
    }

    /// The cached entry, which must exist.
    fn cached_entry(&self, context: &str) -> CacheEntry {
        match self.cached() {
            Cached::Entry(entry) => *entry,
            other => panic!("{context}: the object must have a cached entry; got {other:?}"),
        }
    }

    /// Assert `entry` is the object's stored cut now: the header's `(seq,
    /// commit_chain)`, the store's dangling ids and the normalized fresh
    /// state.
    async fn assert_current(&self, entry: &CacheEntry, context: &str) {
        let hdr = header(&self.tv, self.oid).await;
        assert_eq!(
            (entry.seq, entry.commit_chain.as_str()),
            (
                hdr["seq"].parse::<u64>().expect("decimal seq"),
                hdr.get("commit_chain").map_or("", String::as_str)
            ),
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

    /// A counter cell, `0` when absent.
    fn cell(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        sample_value(&render(&self.metrics), name, labels).unwrap_or(0.0) as u64
    }

    /// `live_edit_update_layer_conflicts_total`: every nonzero `(path, kind)`
    /// cell.
    fn conflicts(&self) -> BTreeMap<(String, String), u64> {
        let text = render(&self.metrics);
        let mut cells = BTreeMap::new();
        for path in PATHS {
            for kind in CONFLICT_KINDS {
                let value = sample_value(
                    &text,
                    "live_edit_update_layer_conflicts_total",
                    &[("path", path), ("kind", kind)],
                )
                .unwrap_or_else(|| panic!("conflicts_total{{{path},{kind}}} is zero-seeded"))
                    as u64;
                if value != 0 {
                    cells.insert((path.to_owned(), kind.to_owned()), value);
                }
            }
        }
        cells
    }

    /// `(count, sum)` of the attempts histogram on `path`; zeros before its
    /// first sample.
    fn attempts(&self, path: &str) -> (u64, u64) {
        let text = render(&self.metrics);
        let read = |suffix: &str| {
            sample_value(
                &text,
                &format!("live_edit_update_layer_attempts_histogram_{suffix}"),
                &[("path", path)],
            )
            .unwrap_or_default() as u64
        };
        (read("count"), read("sum"))
    }

    /// The nonzero cells of `name` by `label`.
    fn ticks(&self, name: &str, label: &str) -> BTreeMap<String, u64> {
        let prefix = format!("{name}{{");
        let needle = format!("{label}=\"");
        render(&self.metrics)
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

    fn lookups(&self) -> BTreeMap<String, u64> {
        self.ticks("live_edit_structural_cache_lookups_total", "outcome")
    }

    fn reloads(&self) -> BTreeMap<String, u64> {
        self.ticks("live_edit_structural_cache_reloads_total", "reason")
    }

    fn divergences(&self) -> u64 {
        self.cell("live_edit_structural_cache_divergence_total", &[])
    }

    fn fallbacks(&self) -> u64 {
        self.cell(
            "live_edit_update_layer_structural_fallbacks_total",
            &[("reason", "repeat_divergence")],
        )
    }

    fn uncacheable_cold_starts(&self) -> u64 {
        self.cell(
            "live_edit_structural_cold_start_reads_total",
            &[("route", "uncacheable")],
        )
    }

    fn already_landed(&self, backend: &str) -> u64 {
        self.cell(
            "live_edit_update_layer_commit_already_landed_total",
            &[("backend", backend)],
        )
    }

    /// `live_edit_structural_cache_audit_total` as `(match, diverged,
    /// skipped)`.
    fn audits(&self) -> (u64, u64, u64) {
        let cell = |outcome: &str| {
            self.cell(
                "live_edit_structural_cache_audit_total",
                &[("outcome", outcome)],
            )
        };
        (cell("match"), cell("diverged"), cell("skipped"))
    }
}

/// An object's cache state, read by [`Fx::cached`].
#[derive(Debug)]
enum Cached {
    Entry(Box<CacheEntry>),
    Miss,
    Uncacheable(UncacheableReason),
}

/// One `update_layer_with_operation` call on `oid`, relaying the delta's
/// OMPF, under [`CALL_BUDGET`].
async fn update_with_op_on(
    ls: &LayerStore,
    session_id: &str,
    layer: &str,
    oid: u64,
    delta: OwnedDelta,
    client_id: &str,
    operation: Option<Operation>,
) -> Result<UpdateCommit, ServiceError> {
    let relay = Some(Bytes::from(delta_to_ompf_bytes(&delta, &[])));
    tokio::time::timeout(
        CALL_BUDGET,
        ls.update_layer_with_operation(
            session_id,
            layer,
            oid,
            std::sync::Arc::new(Some(delta)),
            relay,
            client_id,
            CLIENT_TS,
            operation,
        ),
    )
    .await
    .expect("update_layer_with_operation did not settle within the call budget")
}

/// The nonzero differences `after - before`, cell by cell.
fn since<K: Ord + Clone + std::fmt::Debug>(
    after: BTreeMap<K, u64>,
    before: &BTreeMap<K, u64>,
) -> BTreeMap<K, u64> {
    after
        .into_iter()
        .filter_map(|(cell, value)| {
            let earlier = before.get(&cell).copied().unwrap_or(0);
            let delta = value
                .checked_sub(earlier)
                .unwrap_or_else(|| panic!("counter cell {cell:?} went down"));
            (delta != 0).then_some((cell, delta))
        })
        .collect()
}

fn counts<const N: usize>(cells: [(&str, u64); N]) -> BTreeMap<String, u64> {
    cells.into_iter().map(|(k, v)| (k.to_owned(), v)).collect()
}

fn conflict_cells<const N: usize>(
    cells: [(&str, &str, u64); N],
) -> BTreeMap<(String, String), u64> {
    cells
        .into_iter()
        .map(|(path, kind, n)| ((path.to_owned(), kind.to_owned()), n))
        .collect()
}

/// A `CreateSection` of `section_id` under `parent_id`.
fn create_section(parent_id: u64, section_id: u64, name: &str) -> OwnedDelta {
    structural(vec![create_cmd(parent_id, section_id, name)])
}

/// A rename of `WORLD` in place: a `MoveSection` from the root to the root,
/// which rewrites `WORLD`'s section row and so carries its pre-image digest.
fn rename_world(name: &str) -> OwnedDelta {
    structural(vec![move_cmd(WORLD, ROOT, ROOT, name)])
}

/// A delta the domain refuses while planning: a `CreateSection` of `WORLD`,
/// which already exists (`DuplicateSection`, an `InvalidArgument`).
fn recreate_world() -> OwnedDelta {
    create_section(ROOT, WORLD, "Again")
}

/// An out-of-band rename of `WORLD`'s stored row.
fn world_renamed_out_of_band(name: &str) -> OutOfBandChange {
    OutOfBandChange::SectionName {
        section_id: WORLD,
        name: name.to_owned(),
    }
}

/// The stored name of `section_id`, if its row exists.
async fn stored_name(tv: &TestValkey, oid: u64, section_id: u64) -> Option<String> {
    tv.client()
        .pool
        .hget(
            tv.key_schema()
                .obj_section(&oid.to_string(), &section_id.to_string()),
            "name",
        )
        .await
        .expect("HGET obj:sec name must succeed")
}

/// A queue of `n` fred errors of `kind`.
fn fred_errors(
    n: usize,
    kind: FredErrorKind,
    details: &'static str,
) -> Arc<Mutex<VecDeque<FredError>>> {
    Arc::new(Mutex::new(
        (0..n)
            .map(|_| FredError::new(kind.clone(), details))
            .collect(),
    ))
}

/// Assert a seam queue drained, so the disturbance it stages happened.
fn assert_drained<T: std::fmt::Debug>(queue: &Mutex<VecDeque<T>>, seam: &str) {
    let left = queue.lock().unwrap();
    assert!(
        left.is_empty(),
        "every queued {seam} entry must have been consumed; {left:?} left"
    );
}

/// One staged disturbance: an async action a test hook runs once.
type Action = Box<dyn FnOnce() -> BoxFuture<'static, ()> + Send>;

/// Disturbances a test hook runs, one per firing, in the order they were
/// staged; a firing with none staged passes straight through. Installed as
/// the structural pre-dispatch hook ([`Self::pre_dispatch_hook`]: after an
/// attempt planned and sized its commit, before its `EVALSHA`) or as the
/// between-attempts hook ([`Self::between_attempts_hook`]: before a retry
/// reads anything).
#[derive(Clone, Default)]
struct Staged {
    actions: Arc<Mutex<VecDeque<Action>>>,
    fired: Arc<AtomicUsize>,
}

impl Staged {
    /// Stage `action` for the next firing that has none.
    fn push(&self, action: impl FnOnce() -> BoxFuture<'static, ()> + Send + 'static) {
        self.actions.lock().unwrap().push_back(Box::new(action));
    }

    /// One firing: count it and run the next staged action, if any.
    fn fire(&self) -> BoxFuture<'static, ()> {
        self.fired.fetch_add(1, Ordering::SeqCst);
        let action = self.actions.lock().unwrap().pop_front();
        async move {
            if let Some(action) = action {
                action().await;
            }
        }
        .boxed()
    }

    fn pre_dispatch_hook(&self) -> WatchWindowHook {
        let staged = self.clone();
        Arc::new(move |_eval_client: &FredClient| staged.fire())
    }

    fn between_attempts_hook(&self) -> BetweenAttemptsHook {
        let staged = self.clone();
        Arc::new(move |_attempt: u32| staged.fire())
    }

    /// How many times the hook fired.
    fn fired(&self) -> usize {
        self.fired.load(Ordering::SeqCst)
    }

    /// Assert every staged action ran, so the disturbance happened.
    fn assert_ran(&self, what: &str) {
        assert_eq!(
            self.actions.lock().unwrap().len(),
            0,
            "fixture: every staged {what} action must have run"
        );
    }
}

impl Fx {
    /// An action committing `delta` as `client_id` (with `operation`, if any)
    /// through another replica's store; its commit is published to `landed`.
    fn other_replica_commits(
        &self,
        delta: OwnedDelta,
        client_id: &'static str,
        operation: Option<Operation>,
        landed: Arc<Mutex<Option<UpdateCommit>>>,
    ) -> impl FnOnce() -> BoxFuture<'static, ()> + Send + 'static {
        let (ls, session_id, layer, oid) = (
            self.replica(),
            self.session_id.clone(),
            self.layer,
            self.oid,
        );
        move || {
            async move {
                let commit =
                    update_with_op_on(&ls, &session_id, layer, oid, delta, client_id, operation)
                        .await
                        .expect("the other replica's staged commit must land");
                *landed.lock().unwrap() = Some(commit);
            }
            .boxed()
        }
    }

    /// `live_edit_structural_cache_replayed_entries`'s sum: the stream entries
    /// every repair so far replayed.
    fn replayed_entries(&self) -> u64 {
        self.cell("live_edit_structural_cache_replayed_entries_sum", &[])
    }
}

/// A field write on `WORLD`: a fast-path delta.
fn world_field(value: u8) -> OwnedDelta {
    delta_with_set_field_inline(WORLD, "staged", vec![value], 0)
}

// ═════════════════════════════════════════════════════════════════════════════
// Serialization and STALE
// ═════════════════════════════════════════════════════════════════════════════

/// Concurrent structural calls on one object, all on one replica: the
/// object's `structural` permit runs them one at a time, each on the entry
/// its predecessor's commit put back, so none meets a moved `seq` (SR-7.2).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn structural_calls_on_one_object_on_one_replica_serialize_with_no_seq_fence_conflict() {
    const CALLS: u64 = 6;
    let fx = Arc::new(Fx::new("dispatch-serial").await);
    let seq_before = fx.header_seq().await;

    let calls: Vec<_> = (0..CALLS)
        .map(|i| {
            let fx = Arc::clone(&fx);
            tokio::spawn(async move {
                fx.update(create_section(ROOT, 100 + i, &format!("Serial{i}")))
                    .await
            })
        })
        .collect();
    for (i, call) in futures::future::join_all(calls)
        .await
        .into_iter()
        .enumerate()
    {
        call.unwrap_or_else(|e| panic!("call {i}'s task must not panic: {e}"))
            .unwrap_or_else(|e| panic!("call {i} must commit; got {e:?}"));
    }

    assert_eq!(
        fx.header_seq().await,
        seq_before + CALLS,
        "every call committed exactly once"
    );
    assert_eq!(
        fx.conflicts(),
        BTreeMap::new(),
        "calls serialized on one replica's structural permit meet no seq fence, no \
         divergence, no fence wait"
    );
    let (count, sum) = (
        fx.attempts(STRUCTURAL).0 + fx.attempts(STRUCTURAL_FULL).0,
        fx.attempts(STRUCTURAL).1 + fx.attempts(STRUCTURAL_FULL).1,
    );
    assert_eq!(
        (count, sum),
        (CALLS, CALLS),
        "every call committed on its first attempt, on the structural route"
    );
    fx.assert_current(&fx.cached_entry("after the calls"), "after the calls")
        .await;
}

/// `STALE` under a racing fast-path writer (SR-6.4, SR-10.4's
/// clone-before-plan `STALE` half; b.v6d revision 3 item 6).
///
/// Another replica's fast-path writer commits field writes to the object
/// in a loop while this replica makes structural calls, each carrying a
/// client operation, until at least one structural commit has missed its
/// `seq` fence. The object carries `RACE_FIELDS` fields so a plan — which
/// clones the entry's state — takes long enough for the writer to land
/// inside it, and the writer paces itself (`WRITER_PACE`, load shaping, not
/// synchronization) so that a retry also finds windows with no commit in
/// them; the structural store gets `RACE_MAX_ATTEMPTS` attempts per call. The
/// race is undirected — how many calls meet a `STALE`, and how many entries
/// each retry replays, is not fixed; the deterministic twin is
/// [`structural_stale_on_a_cold_started_attempt_ticks_on_structural_full_and_the_retry_repairs_one_entry`].
/// The loop is bounded by [`RACE_BUDGET`] and `RACE_MAX_CALLS`.
///
/// - Every structural call commits exactly once: one stream entry each, and
///   each created section present. A `STALE` attempt wrote nothing; a plan
///   that had mutated the held entry would put back an entry already holding
///   its section, and the retry's re-plan would refuse the `CreateSection`.
/// - `seq_fence` ticks, only on the structural paths, and nothing else.
/// - The retry repairs: after the first call's cold start no read is cold
///   again and nothing reloads, so a `STALE` put the original entry back.
/// - One more call, once the writer has stopped, leaves the cached entry as
///   the stored cut.
/// - Each call reads its start record once, however many `STALE` re-plans
///   it made: the start-record read seam's queue, one entry per call, is
///   consumed one per call. Each entry is a `WRONGTYPE` read, which the call
///   takes as no start record; every call is made under a `client_id` of its
///   own ([`RACE_CLIENT`] and its number), which truly has none.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn structural_commit_racing_a_fast_writer_is_stale_repaired_and_commits_once() {
    const RACE_FIELDS: usize = 500;
    const RACE_MAX_CALLS: u64 = 200;
    const RACE_MAX_ATTEMPTS: u32 = 64;
    const WRITER_PACE: Duration = Duration::from_millis(2);

    let mut state = every_shape_state();
    for i in 0..RACE_FIELDS {
        let key_name = format!("race-seed-{i:05}");
        state.fields.insert(
            (WORLD, key_name.clone()),
            Field {
                section_id: WORLD,
                key_name,
                value: FieldValue::Inline(vec![0xA5; 16]),
                set_order: 5,
                authoring_set_order: 5,
                version: 1,
            },
        );
    }
    let start_reads = fred_errors(
        RACE_MAX_CALLS as usize,
        FredErrorKind::InvalidArgument,
        "WRONGTYPE injected: the call's start record reads as absent",
    );
    let fx = Fx::with("dispatch-stale", &state, BUDGET_BYTES, |ls| {
        ls.with_update_layer_retry_max_attempts(RACE_MAX_ATTEMPTS)
            .with_start_record_read_fred_error_injector_for_test(Arc::clone(&start_reads))
    })
    .await;
    let seq_before = fx.header_seq().await;

    let stop = Arc::new(AtomicBool::new(false));
    let fast_commits = Arc::new(AtomicUsize::new(0));
    let writer = tokio::spawn({
        let (stop, fast_commits) = (Arc::clone(&stop), Arc::clone(&fast_commits));
        let ls = fx.replica();
        let (session_id, layer, oid) = (fx.session_id.clone(), fx.layer, fx.oid);
        async move {
            let mut i = 0u64;
            while !stop.load(Ordering::SeqCst) {
                let delta =
                    delta_with_set_field_inline(WORLD, "racing", i.to_le_bytes().to_vec(), 0);
                let relay = Some(Bytes::from(delta_to_ompf_bytes(&delta, &[])));
                ls.update_layer(&session_id, layer, oid, Some(delta), relay, OTHER_WRITER, i)
                    .await
                    .expect("the racing fast-path writer's commit must land");
                fast_commits.fetch_add(1, Ordering::SeqCst);
                i += 1;
                tokio::time::sleep(WRITER_PACE).await;
            }
        }
    });

    let stale = |fx: &Fx| {
        fx.conflicts()
            .iter()
            .filter(|((_, kind), _)| kind == SEQ_FENCE)
            .map(|(_, n)| n)
            .sum::<u64>()
    };
    let mut calls = 0u64;
    tokio::time::timeout(RACE_BUDGET, async {
        while stale(&fx) == 0 && calls < RACE_MAX_CALLS {
            let commit = fx
                .update_with_op(
                    &fx.ls,
                    create_section(ROOT, 1_000 + calls, &format!("Raced{calls}")),
                    &format!("{RACE_CLIENT}{calls}"),
                    Some(fx.op(&format!("race-op-{calls}"), &calls.to_le_bytes())),
                )
                .await
                .unwrap_or_else(|e| {
                    panic!(
                        "structural call {calls} must commit despite the racing writer; got \
                         {e:?}. An INVALID_ARGUMENT here is a re-plan refusing its own \
                         CreateSection: the plan mutated the entry the STALE put back"
                    )
                });
            assert_eq!(commit.outcome, OperationOutcome::Applied, "call {calls}");
            calls += 1;
        }
    })
    .await
    .expect("the race loop must finish within its budget");
    stop.store(true, Ordering::SeqCst);
    tokio::time::timeout(CALL_BUDGET, writer)
        .await
        .expect("the racing writer must stop within the call budget")
        .expect("the racing writer's task must not panic");

    assert!(
        stale(&fx) > 0,
        "fixture: no structural commit met a moved seq in {calls} calls racing {} fast commits \
         — the race this test is about never happened",
        fast_commits.load(Ordering::SeqCst)
    );
    let structural_entries = fx.commits_by_prefix(RACE_CLIENT).await as u64;
    assert_eq!(
        structural_entries, calls,
        "every structural call published exactly one entry: a STALE attempt writes nothing"
    );
    assert_eq!(
        fx.header_seq().await,
        seq_before + calls + fast_commits.load(Ordering::SeqCst) as u64,
        "the header advanced once per commit of either writer"
    );
    for i in 0..calls {
        assert_eq!(
            stored_name(&fx.tv, fx.oid, 1_000 + i).await.as_deref(),
            Some(format!("Raced{i}").as_str()),
            "call {i}'s section must be stored"
        );
    }

    let conflicts = fx.conflicts();
    assert!(
        conflicts.iter().all(|((path, kind), _)| kind == SEQ_FENCE
            && (path == STRUCTURAL || path == STRUCTURAL_FULL)),
        "only seq_fence conflicts, on the structural paths, may tick; got {conflicts:?}"
    );
    let lookups = fx.lookups();
    assert_eq!(
        lookups.get("cold").copied(),
        Some(1),
        "only the first call cold-starts: a STALE puts the original entry back, so every \
         retry is a hit; got {lookups:?}"
    );
    assert!(
        lookups.get("repaired").copied().unwrap_or(0) > 0,
        "the retries repaired the entry over the racing commits; got {lookups:?}"
    );
    assert_eq!(fx.reloads(), counts([]), "nothing reloaded");
    assert_eq!(
        RACE_MAX_CALLS as usize - start_reads.lock().unwrap().len(),
        calls as usize,
        "each call reads its start record exactly once, its STALE re-plans included"
    );

    // The writer's last commits landed after the last structural call: one
    // more call, unraced, repairs over them, and its successor is the stored
    // cut.
    fx.update_on(&fx.ls, create_section(ROOT, 999, "AfterTheRace"), WRITER)
        .await
        .expect("the call after the race must commit");
    fx.assert_current(&fx.cached_entry("after the race"), "after the race")
        .await;
}

/// A re-plan carries the call's one start record (b.v6d revision 3 item 6).
///
/// `WRITER` has a commit-token record (an earlier call's operation) when call
/// X starts, which reads it as X's start record. X's first attempt plans on a
/// cached entry whose `WORLD` row was rewritten out of band, so its commit is
/// `CACHE_DIVERGED` and the attempt is retried. Between the attempts
/// `WRITER`'s next call Y commits. X's re-plan, still carrying the start
/// record it read before Y, must find the record moved and be refused
/// `Overtaken` — applying nothing. A re-plan that re-read the start record,
/// or carried none, would apply X after Y.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn structural_replan_after_cache_diverged_carries_the_call_start_record_and_is_overtaken() {
    let landed_y = Arc::new(AtomicUsize::new(0));
    let hook_fired = Arc::new(AtomicUsize::new(0));
    let y: Arc<std::sync::OnceLock<YCall>> = Arc::new(std::sync::OnceLock::new());
    let hook: BetweenAttemptsHook = {
        let (y, landed_y, hook_fired) = (
            Arc::clone(&y),
            Arc::clone(&landed_y),
            Arc::clone(&hook_fired),
        );
        Arc::new(move |_attempt: u32| {
            let (y, landed_y) = (Arc::clone(&y), Arc::clone(&landed_y));
            let first = hook_fired.fetch_add(1, Ordering::SeqCst) == 0;
            async move {
                if first {
                    let y = y.get().expect("fixture: Y is staged").clone();
                    update_with_op_on(
                        &y.ls,
                        &y.session_id,
                        y.layer,
                        y.oid,
                        delta_with_set_field_inline(WORLD, "y", vec![0x59], 0),
                        WRITER,
                        Some(y.op),
                    )
                    .await
                    .expect("Y, the writer's next call, must commit");
                    landed_y.fetch_add(1, Ordering::SeqCst);
                }
            }
            .boxed()
        })
    };
    let fx = Fx::with(
        "dispatch-overtaken",
        &every_shape_state(),
        BUDGET_BYTES,
        |ls| ls.with_between_attempts_hook_for_test(hook),
    )
    .await;
    assert!(
        y.set(YCall {
            ls: fx.replica(),
            session_id: fx.session_id.clone(),
            layer: fx.layer,
            oid: fx.oid,
            op: fx.op("y-op", b"y"),
        })
        .is_ok(),
        "fixture: Y is staged once"
    );
    // The writer's earlier call: its record is X's start record, and its
    // structural commit caches the object.
    fx.update_with_op(
        &fx.ls,
        create_section(ROOT, NEW_SECTION, "Earlier"),
        WRITER,
        Some(fx.op("earlier-op", b"earlier")),
    )
    .await
    .expect("the writer's earlier call must commit");
    rewrite_out_of_band(&fx.tv, fx.oid, world_renamed_out_of_band("OutOfBand")).await;
    let ids_before_x = fx.stream_ids().await;

    let x = fx
        .update_with_op(
            &fx.ls,
            rename_world("ByX"),
            WRITER,
            Some(fx.op("x-op", b"x")),
        )
        .await;

    match &x {
        Err(ServiceError::Unavailable(msg)) => assert_eq!(
            msg, OPERATION_OVERTAKEN_MESSAGE,
            "X's re-plan must be refused as overtaken"
        ),
        other => panic!(
            "X's re-plan must be refused UNAVAILABLE (overtaken) by its client's next commit; \
             got {other:?}. An OK here is X applied after Y: the re-plan did not carry the \
             start record X read before Y"
        ),
    }
    assert_eq!(
        (
            hook_fired.load(Ordering::SeqCst),
            landed_y.load(Ordering::SeqCst)
        ),
        (1, 1),
        "fixture: exactly one retry of X, preceded by Y's commit"
    );
    assert_eq!(
        fx.divergences(),
        1,
        "fixture: X's first attempt was CACHE_DIVERGED"
    );
    assert_eq!(
        stored_name(&fx.tv, fx.oid, WORLD).await.as_deref(),
        Some("OutOfBand"),
        "X applied nothing"
    );
    let ids = fx.stream_ids().await;
    assert_eq!(
        ids.len(),
        ids_before_x.len() + 1,
        "only Y's commit was published"
    );
    assert_eq!(
        fx.cell(
            "live_edit_update_layer_operation_overtaken_total",
            &[("backend", "lua")]
        ),
        1,
        "the structural route's overtaken verdict is counted on the lua backend"
    );
}

/// The writer's next call Y, staged for the between-attempts hook: another
/// replica's store, and the object and operation it commits.
#[derive(Clone)]
struct YCall {
    ls: LayerStore,
    session_id: String,
    layer: &'static str,
    oid: u64,
    op: Operation,
}

/// `STALE` staged deterministically (SR-6.4, SR-10.4's clone-before-plan
/// `STALE` half): on a cold object, the call's first attempt cold-starts the
/// entry and plans, and the pre-dispatch hook lands another replica's
/// fast-path commit before the attempt's `EVALSHA`.
///
/// - The `STALE` attempt wrote nothing: the stream gained the other
///   replica's entry and the call's one.
/// - It put the original entry back exactly as cold-started: seen by the
///   between-attempts hook, before the retry reads, the cached entry stands
///   at the call's starting `seq` and holds the stored state as it was then
///   — not the plan's `CreateSection`.
/// - `seq_fence` ticks once, on the cold-started attempt's path,
///   `structural_full` (decider ruling, Q5).
/// - The retry repairs the put-back entry over exactly the one landed entry
///   (`replayed_entries` sums to 1; the cold start's own repair replays
///   none), reloads nothing, and commits on `structural`, its call's second
///   attempt; the entry it leaves is the stored cut.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_stale_on_a_cold_started_attempt_ticks_on_structural_full_and_the_retry_repairs_one_entry(
) {
    let before_commit = Staged::default();
    let between_attempts = Staged::default();
    let fx = Fx::with(
        "dispatch-staged-stale",
        &every_shape_state(),
        BUDGET_BYTES,
        |ls| {
            ls.with_structural_commit_pre_dispatch_hook_for_test(before_commit.pre_dispatch_hook())
                .with_between_attempts_hook_for_test(between_attempts.between_attempts_hook())
        },
    )
    .await;
    let seq_before = fx.header_seq().await;
    let state_before = normalized_fresh_state(&fx.tv, fx.oid).await;
    let ids_before = fx.stream_ids().await;
    let landed = Arc::new(Mutex::new(None));
    before_commit.push(fx.other_replica_commits(
        world_field(0x51),
        OTHER_WRITER,
        None,
        Arc::clone(&landed),
    ));
    let put_back: Arc<Mutex<Option<Cached>>> = Arc::new(Mutex::new(None));
    between_attempts.push({
        let (cache, oid, put_back) = (fx.cache.clone(), fx.oid, Arc::clone(&put_back));
        move || {
            async move {
                *put_back.lock().unwrap() = Some(cached_in(&cache, oid));
            }
            .boxed()
        }
    });

    let (seq, _) = fx
        .update(create_section(ROOT, NEW_SECTION, "AfterStale"))
        .await
        .expect("the retry must repair and commit");

    before_commit.assert_ran("other-replica commit");
    between_attempts.assert_ran("put-back snapshot");
    assert_eq!(
        before_commit.fired(),
        2,
        "fixture: two structural commits were dispatched, the STALE one and the retry's"
    );
    let other = landed
        .lock()
        .unwrap()
        .expect("fixture: the other replica's commit landed");
    assert_eq!(
        (other.seq, seq),
        (seq_before + 1, seq_before + 2),
        "the other replica's commit, then the call's — once"
    );
    assert_eq!(
        fx.stream_ids().await.len(),
        ids_before.len() + 2,
        "the STALE attempt wrote nothing: one entry each from the other replica and the retry"
    );
    match put_back.lock().unwrap().take() {
        Some(Cached::Entry(entry)) => {
            assert_eq!(
                entry.seq, seq_before,
                "the STALE attempt put back the cold-started entry, at the call's starting seq"
            );
            assert_eq!(
                entry.state, state_before,
                "the put-back entry is exactly as cold-started: the plan ran on a clone"
            );
        }
        other => panic!("between the attempts the original entry must be cached; got {other:?}"),
    }
    assert_eq!(
        fx.conflicts(),
        conflict_cells([(STRUCTURAL_FULL, SEQ_FENCE, 1)]),
        "the seq fence is counted on the cold-started attempt's path"
    );
    assert_eq!(
        fx.lookups(),
        counts([("cold", 1), ("repaired", 1)]),
        "the retry repaired the put-back entry rather than reading the object again"
    );
    assert_eq!(fx.reloads(), counts([]), "nothing reloaded");
    assert_eq!(
        fx.replayed_entries(),
        1,
        "the retry's repair replayed exactly the other replica's one entry"
    );
    assert_eq!(
        fx.attempts(STRUCTURAL),
        (1, 2),
        "the call committed on its retry, planned on the repaired entry"
    );
    fx.assert_current(&fx.cached_entry("after the retry"), "after the retry")
        .await;
}

/// `seq_fence` on a reset ticks on `reset`, the attempt's path (decider
/// ruling, Q5): the pre-dispatch hook lands another replica's commit before
/// the reset's first `EVALSHA`, and the reset commits on its retry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_reset_stale_ticks_on_the_reset_path() {
    let before_commit = Staged::default();
    let fx = Fx::with(
        "dispatch-reset-stale",
        &every_shape_state(),
        BUDGET_BYTES,
        |ls| {
            ls.with_structural_commit_pre_dispatch_hook_for_test(before_commit.pre_dispatch_hook())
        },
    )
    .await;
    before_commit.push(fx.other_replica_commits(
        world_field(0x52),
        OTHER_WRITER,
        None,
        Arc::new(Mutex::new(None)),
    ));

    fx.update(reset_delta("usda"))
        .await
        .expect("the reset's retry must commit");

    before_commit.assert_ran("other-replica commit");
    assert_eq!(fx.conflicts(), conflict_cells([(RESET, SEQ_FENCE, 1)]));
    assert_eq!(
        fx.attempts(RESET),
        (1, 2),
        "the reset committed on its second attempt"
    );
    assert_eq!(
        stored_name(&fx.tv, fx.oid, WORLD).await,
        None,
        "the reset cleared the object"
    );
}

/// `oid`'s state in `cache`: its entry (a hit, put back at once by the
/// guard's drop), a miss, or its uncacheable verdict.
fn cached_in(cache: &StructuralCache, oid: u64) -> Cached {
    match cache.lookup(oid) {
        CacheLookup::Hit(guard) => Cached::Entry(Box::new(guard.entry().clone())),
        CacheLookup::Miss => Cached::Miss,
        CacheLookup::Uncacheable { reason } => Cached::Uncacheable(reason),
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Lost reply, the shared constructor, and the op-id regression
// ═════════════════════════════════════════════════════════════════════════════

/// A structural commit lands and its reply is lost (the post-`EVALSHA`
/// fault queue). The retry's read side finds the call's token before it
/// plans and answers the landed commit — a repeated `CreateSection`, which
/// re-planned on the repaired entry would be refused, included — so nothing
/// is applied twice (SR-10.4, SR-4.2a, as amended: the repair answers the
/// token before replaying).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_lost_reply_is_answered_by_the_retrys_token_check_and_applies_nothing_twice() {
    let lost = fred_errors(1, FredErrorKind::Timeout, "dispatch: reply lost");
    let fx = Fx::with(
        "dispatch-lost-reply",
        &every_shape_state(),
        BUDGET_BYTES,
        |ls| ls.with_lua_post_eval_fred_error_injector_for_test(Arc::clone(&lost)),
    )
    .await;
    let seq_before = fx.header_seq().await;

    let (seq, server_ts) = fx
        .update(create_section(ROOT, NEW_SECTION, "Once"))
        .await
        .expect(
            "the commit landed, so the call must succeed. INVALID_ARGUMENT here is the retry \
             re-planning the CreateSection its lost attempt already made",
        );
    assert_drained(&lost, "post-EVALSHA lost reply");

    assert_eq!(
        (seq, fx.header_seq().await),
        (seq_before + 1, seq_before + 1),
        "one commit, reported at its own seq"
    );
    let payloads = fx.commits_by(WRITER).await;
    assert_eq!(payloads.len(), 1, "exactly one commit entry: {payloads:?}");
    assert_eq!(
        payloads[0].split('|').nth(5),
        Some(server_ts.to_string().as_str()),
        "the reported server timestamp is the landed commit's, from its record"
    );
    assert_eq!(
        (fx.already_landed("lua"), fx.already_landed("txn")),
        (1, 0),
        "the retry's read side reported the landed commit, on the lua backend"
    );
    assert_eq!(
        fx.lookups(),
        counts([("cold", 1), ("hit", 1)]),
        "the lost attempt cold-started and put its entry back; the retry's hit answered \
         the token before replaying the landed entry"
    );
    assert_eq!(
        stored_name(&fx.tv, fx.oid, NEW_SECTION).await.as_deref(),
        Some("Once")
    );
}

/// Both of the structural route's already-landed answers are the verdict
/// the fast and WATCH paths reach (`Recorded(ThisCall)`), so
/// `update_layer_with_summary` builds the result with the one own-commit
/// constructor: the landed `(seq, server_ts)`, `APPLIED` with a client
/// operation and `UNSPECIFIED` without — exactly what the fast path answers
/// for the same lost reply. The read side's answer is driven here; the
/// commit script's own answer is driven by
/// [`structural_commit_already_landed_answer_is_the_own_commit_result_and_puts_the_entry_back`].
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_already_landed_answers_build_the_shared_own_commit_result() {
    for (case, with_op) in [("with-op", true), ("without-op", false)] {
        // The structural route and the fast path, each losing one reply.
        let structural_lost = fred_errors(1, FredErrorKind::Timeout, "dispatch: reply lost");
        let structural_fx = Fx::with(
            "dispatch-own-commit-structural",
            &every_shape_state(),
            BUDGET_BYTES,
            |ls| ls.with_lua_post_eval_fred_error_injector_for_test(Arc::clone(&structural_lost)),
        )
        .await;
        let fast_lost = fred_errors(1, FredErrorKind::Timeout, "dispatch: reply lost");
        let fast_fx = Fx::with(
            "dispatch-own-commit-fast",
            &every_shape_state(),
            BUDGET_BYTES,
            |ls| ls.with_lua_post_eval_fred_error_injector_for_test(Arc::clone(&fast_lost)),
        )
        .await;
        let op = |fx: &Fx| with_op.then(|| fx.op("own-commit-op", b"own-commit"));

        let structural = structural_fx
            .update_with_op(
                &structural_fx.ls,
                create_section(ROOT, NEW_SECTION, "Own"),
                WRITER,
                op(&structural_fx),
            )
            .await
            .unwrap_or_else(|e| panic!("{case}: the structural call must succeed; got {e:?}"));
        let fast = fast_fx
            .update_with_op(
                &fast_fx.ls,
                delta_with_set_field_inline(WORLD, "own", vec![1], 0),
                WRITER,
                op(&fast_fx),
            )
            .await
            .unwrap_or_else(|e| panic!("{case}: the fast call must succeed; got {e:?}"));
        assert_drained(&structural_lost, "structural lost reply");
        assert_drained(&fast_lost, "fast lost reply");

        let landed = |payloads: Vec<String>| -> (u64, u64) {
            let [payload] = payloads.as_slice() else {
                panic!("{case}: exactly one commit entry; got {payloads:?}");
            };
            let field = |n: usize| payload.split('|').nth(n).unwrap().parse().unwrap();
            (field(2), field(5))
        };
        let expected_outcome = if with_op {
            OperationOutcome::Applied
        } else {
            OperationOutcome::Unspecified
        };
        let (seq, server_ts) = landed(structural_fx.commits_by(WRITER).await);
        assert_eq!(
            structural,
            UpdateCommit {
                seq,
                server_ts,
                outcome: expected_outcome,
            },
            "{case}: the structural read side's already-landed answer must be the own-commit \
             result: the landed commit, and APPLIED only with an operation. \
             ALREADY_COMMITTED here is the answer built as an earlier call's"
        );
        let (seq, server_ts) = landed(fast_fx.commits_by(WRITER).await);
        assert_eq!(
            fast,
            UpdateCommit {
                seq,
                server_ts,
                outcome: expected_outcome,
            },
            "{case}: control — the fast path's own already-landed answer"
        );
        assert_eq!(
            (
                structural_fx.already_landed("lua"),
                fast_fx.already_landed("lua")
            ),
            (1, 1),
            "{case}: each route counted its one already-landed answer"
        );
    }
}

/// The decider-required op-id structural regression (b.v6d, moved from Task
/// 12), verbatim: "an end-to-end re-send of an `UpdateLayer` carrying a
/// `CreateSection` with the same `operation_id`, through `update_layer` on the
/// structural route. It answers `OK` with the original `sequence_num` and
/// server timestamp and `operation_outcome = ALREADY_COMMITTED`, applies
/// nothing, and writes nothing. The same id with a different request answers
/// `ALREADY_EXISTS`."
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_resend_with_the_same_operation_id_is_answered_already_committed_and_writes_nothing(
) {
    const OPERATION_ID: &str = "5d0c9f3e-2b4a-4c1d-8e7f-6a5b4c3d2e1f";
    let fx = Fx::new("dispatch-op-id").await;
    let original_op = fx.op(OPERATION_ID, b"the original request");

    let original = fx
        .update_with_op(
            &fx.ls,
            create_section(ROOT, NEW_SECTION, "Created"),
            WRITER,
            Some(original_op.clone()),
        )
        .await
        .expect("the original call must commit");
    assert_eq!(
        original.outcome,
        OperationOutcome::Applied,
        "fixture: the original call commits and records its operation"
    );
    assert_eq!(
        fx.attempts(STRUCTURAL_FULL),
        (1, 1),
        "fixture: on the structural route"
    );
    let before = fx.keyspace().await;
    let lookups_before = fx.lookups();

    let resent = fx
        .update_with_op(
            &fx.ls,
            create_section(ROOT, NEW_SECTION, "Created"),
            WRITER,
            Some(original_op),
        )
        .await
        .expect(
            "the re-send must be answered OK. INVALID_ARGUMENT here is the re-send re-applying \
             the CreateSection the original landed",
        );
    assert_eq!(
        resent,
        UpdateCommit {
            seq: original.seq,
            server_ts: original.server_ts,
            outcome: OperationOutcome::AlreadyCommitted,
        },
        "the re-send reports the original commit, ALREADY_COMMITTED"
    );
    assert_eq!(
        fx.keyspace().await,
        before,
        "the re-send applied nothing and wrote nothing"
    );

    let conflicting = fx
        .update_with_op(
            &fx.ls,
            create_section(ROOT, OTHER_NEW_SECTION, "Different"),
            WRITER,
            Some(fx.op(OPERATION_ID, b"a different request")),
        )
        .await;
    assert!(
        matches!(conflicting, Err(ServiceError::AlreadyExists(_))),
        "the same id with a different request is ALREADY_EXISTS; got {conflicting:?}"
    );
    assert_eq!(
        fx.keyspace().await,
        before,
        "the conflicting call applied nothing and wrote nothing"
    );
    assert_eq!(
        since(fx.lookups(), &lookups_before).values().sum::<u64>(),
        2,
        "both answers came from the structural route's read side, one lookup each"
    );
    assert_eq!(
        (
            fx.cell(
                "live_edit_operation_id_match_total",
                &[("rpc", "update_layer"), ("outcome", "already_committed")]
            ),
            fx.cell(
                "live_edit_operation_id_match_total",
                &[("rpc", "update_layer"), ("outcome", "conflict")]
            )
        ),
        (1, 1)
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The commit script's own answers, staged between the plan and the commit
// ═════════════════════════════════════════════════════════════════════════════

/// The entry put back after a commit that changed nothing on this replica's
/// side is the original: the next call's read is a repair (`repaired`) over
/// whatever landed meanwhile, not a cold start. Asserted by making that call
/// (as [`OTHER_WRITER`], with no operation) and reading the one lookup it
/// ticks.
async fn assert_next_call_repairs_the_put_back_entry(fx: &Fx, context: &str) {
    let lookups_before = fx.lookups();
    fx.update_on(&fx.ls, create_section(ROOT, 90, "Next"), OTHER_WRITER)
        .await
        .unwrap_or_else(|e| panic!("{context}: the next call must commit; got {e:?}"));
    assert_eq!(
        since(fx.lookups(), &lookups_before),
        counts([("repaired", 1)]),
        "{context}: the original entry was put back, so the next call repairs it; a cold \
         start means the entry was dropped"
    );
    fx.assert_current(&fx.cached_entry(context), context).await;
}

/// The commit script's own `ALREADY_LANDED` (SR-4.2a, b.v6d), which the
/// structural route maps to the verdict every route's already-landed answer
/// reaches, `Recorded(ThisCall)`: `update_layer_with_summary` builds the
/// result with the one own-commit constructor.
///
/// Staging. Call X carries an operation (so the own-commit result is
/// `APPLIED`, where an earlier call's would be `ALREADY_COMMITTED`). Its
/// first attempt commits and the reply is lost (the post-`EVALSHA` fault
/// queue). Between the attempts the test hides `WRITER`'s commit-token record
/// (`HGET`, then `HDEL`), so the retry's read side finds no record: it
/// repairs over the landed entry and plans again (a rename of `WORLD` to the
/// name it already has, which re-plans cleanly). The pre-dispatch hook of the
/// retry's commit puts the record back, so the commit script finds its own
/// token at step 1b and answers `ALREADY_LANDED`, writing nothing.
///
/// - X answers the landed commit's `(seq, server_ts)` with `APPLIED`.
/// - One commit entry for X; `commit_already_landed_total{backend="lua"}`
///   ticks once (the observer, at the script's answer).
/// - The repaired entry is put back: the next call's read is a hit with
///   nothing to replay.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_commit_already_landed_answer_is_the_own_commit_result_and_puts_the_entry_back()
{
    let lost = fred_errors(0, FredErrorKind::Timeout, "unused");
    let before_commit = Staged::default();
    let between_attempts = Staged::default();
    let fx = Fx::with(
        "dispatch-commit-already-landed",
        &every_shape_state(),
        BUDGET_BYTES,
        |ls| {
            ls.with_lua_post_eval_fred_error_injector_for_test(Arc::clone(&lost))
                .with_structural_commit_pre_dispatch_hook_for_test(
                    before_commit.pre_dispatch_hook(),
                )
                .with_between_attempts_hook_for_test(between_attempts.between_attempts_hook())
        },
    )
    .await;
    // Warm the cache as another writer, so `WRITER` has no record (and X no
    // start record) when X starts.
    fx.update_on(
        &fx.ls,
        create_section(ROOT, NEW_SECTION, "Warm"),
        OTHER_WRITER,
    )
    .await
    .expect("the warm-up must commit");
    let seq_before = fx.header_seq().await;
    lost.lock().unwrap().push_back(FredError::new(
        FredErrorKind::Timeout,
        "dispatch: reply lost",
    ));
    // Attempt 1's commit runs undisturbed.
    before_commit.push(|| async {}.boxed());
    let wtok = fx.tv.key_schema().obj_write_token(&fx.oid.to_string());
    let hidden: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    between_attempts.push({
        let (client, wtok, hidden) = (fx.tv.client().clone(), wtok.clone(), Arc::clone(&hidden));
        move || {
            async move {
                let record: Option<String> = client
                    .pool
                    .hget(&wtok, WRITER)
                    .await
                    .expect("HGET the writer's record");
                let _: i64 = client
                    .pool
                    .hdel(&wtok, WRITER)
                    .await
                    .expect("HDEL the writer's record");
                *hidden.lock().unwrap() = record;
            }
            .boxed()
        }
    });
    before_commit.push({
        let (client, wtok, hidden) = (fx.tv.client().clone(), wtok.clone(), Arc::clone(&hidden));
        move || {
            async move {
                let record = hidden
                    .lock()
                    .unwrap()
                    .take()
                    .expect("fixture: attempt 1's commit recorded the writer's token");
                let _: i64 = client
                    .pool
                    .hset(&wtok, [(WRITER, record)])
                    .await
                    .expect("HSET the writer's record back");
            }
            .boxed()
        }
    });

    let x = fx
        .update_with_op(
            &fx.ls,
            rename_world("Renamed"),
            WRITER,
            Some(fx.op("x-op", b"x")),
        )
        .await
        .expect("X's commit landed, so X must succeed");

    assert_drained(&lost, "post-EVALSHA lost reply");
    between_attempts.assert_ran("record hide");
    before_commit.assert_ran("record restore");
    let payloads = fx.commits_by(WRITER).await;
    let [payload] = payloads.as_slice() else {
        panic!("exactly one commit entry for X: nothing applied twice; got {payloads:?}");
    };
    let server_ts: u64 = payload.split('|').nth(5).unwrap().parse().unwrap();
    assert_eq!(
        x,
        UpdateCommit {
            seq: seq_before + 1,
            server_ts,
            outcome: OperationOutcome::Applied,
        },
        "the commit script's own ALREADY_LANDED must become the own-commit result: the landed \
         commit, APPLIED. ALREADY_COMMITTED here is the answer built as an earlier call's"
    );
    assert_eq!(
        fx.header_seq().await,
        seq_before + 1,
        "the retry's commit wrote nothing"
    );
    assert_eq!(
        (fx.already_landed("lua"), fx.already_landed("txn")),
        (1, 0),
        "the commit script's already-landed answer is counted once, on the lua backend"
    );
    assert_eq!(
        stored_name(&fx.tv, fx.oid, WORLD).await.as_deref(),
        Some("Renamed")
    );

    let lookups_before = fx.lookups();
    fx.update_on(&fx.ls, create_section(ROOT, 90, "Next"), OTHER_WRITER)
        .await
        .expect("the next call must commit");
    assert_eq!(
        since(fx.lookups(), &lookups_before),
        counts([("hit", 1)]),
        "the retry's repaired entry, current at the landed commit, was put back: the next \
         read is a hit with nothing to replay"
    );
}

/// The commit script's `Recorded` (b.v6d): between X's plan and its commit,
/// an earlier call carrying X's operation id and request digest commits on
/// another replica, so X's commit finds that call's record under its match
/// key and answers it. X is `ALREADY_COMMITTED` with that commit's `(seq,
/// server_ts)`, applies and writes nothing of its own, and puts its original
/// entry back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_commit_recorded_by_an_earlier_call_answers_already_committed_and_puts_the_entry_back(
) {
    let before_commit = Staged::default();
    let fx = Fx::with(
        "dispatch-commit-recorded",
        &every_shape_state(),
        BUDGET_BYTES,
        |ls| {
            ls.with_structural_commit_pre_dispatch_hook_for_test(before_commit.pre_dispatch_hook())
        },
    )
    .await;
    fx.update_on(
        &fx.ls,
        create_section(ROOT, NEW_SECTION, "Warm"),
        OTHER_WRITER,
    )
    .await
    .expect("the warm-up must commit");
    let op = fx.op("recorded-op", b"the request");
    let earlier = Arc::new(Mutex::new(None));
    before_commit.push(fx.other_replica_commits(
        world_field(0x53),
        WRITER,
        Some(op.clone()),
        Arc::clone(&earlier),
    ));

    let x = fx
        .update_with_op(
            &fx.ls,
            create_section(ROOT, OTHER_NEW_SECTION, "ByX"),
            WRITER,
            Some(op),
        )
        .await
        .expect("X must be answered with the earlier call's commit");

    before_commit.assert_ran("earlier call");
    let earlier = earlier
        .lock()
        .unwrap()
        .expect("fixture: the earlier call committed");
    assert_eq!(
        earlier.outcome,
        OperationOutcome::Applied,
        "fixture: the earlier call applied"
    );
    assert_eq!(
        x,
        UpdateCommit {
            seq: earlier.seq,
            server_ts: earlier.server_ts,
            outcome: OperationOutcome::AlreadyCommitted,
        },
        "the commit script's Recorded answer is the earlier call's commit, ALREADY_COMMITTED"
    );
    assert_eq!(
        stored_name(&fx.tv, fx.oid, OTHER_NEW_SECTION).await,
        None,
        "X applied nothing"
    );
    assert_eq!(
        fx.commits_by(WRITER).await.len(),
        1,
        "only the earlier call's commit was published"
    );
    assert_eq!(
        fx.cell(
            "live_edit_operation_id_match_total",
            &[("rpc", "update_layer"), ("outcome", "already_committed")]
        ),
        1
    );
    assert_next_call_repairs_the_put_back_entry(&fx, "after a commit-side Recorded").await;
}

/// The commit script's `OVERTAKEN` (b.v6d revision 3): X carries an
/// operation whose start record is `WRITER`'s earlier commit; between X's
/// plan and its commit, `WRITER`'s next call commits on another replica, so
/// X's commit finds the record moved off its start record. X is refused
/// `UNAVAILABLE` (overtaken), counted under the `lua` backend label, applies
/// and writes nothing, and puts its original entry back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_commit_overtaken_answers_unavailable_on_the_lua_backend_and_puts_the_entry_back(
) {
    let before_commit = Staged::default();
    let fx = Fx::with(
        "dispatch-commit-overtaken",
        &every_shape_state(),
        BUDGET_BYTES,
        |ls| {
            ls.with_structural_commit_pre_dispatch_hook_for_test(before_commit.pre_dispatch_hook())
        },
    )
    .await;
    // `WRITER`'s earlier commit: X's start record. It also caches the object.
    before_commit.push(|| async {}.boxed());
    fx.update_with_op(
        &fx.ls,
        create_section(ROOT, NEW_SECTION, "Earlier"),
        WRITER,
        Some(fx.op("earlier-op", b"earlier")),
    )
    .await
    .expect("the writer's earlier call must commit");
    let next = Arc::new(Mutex::new(None));
    before_commit.push(fx.other_replica_commits(
        world_field(0x54),
        WRITER,
        Some(fx.op("next-op", b"next")),
        Arc::clone(&next),
    ));

    let x = fx
        .update_with_op(
            &fx.ls,
            create_section(ROOT, OTHER_NEW_SECTION, "ByX"),
            WRITER,
            Some(fx.op("x-op", b"x")),
        )
        .await;

    before_commit.assert_ran("the writer's next call");
    assert!(
        next.lock().unwrap().is_some(),
        "fixture: the writer's next call committed"
    );
    match &x {
        Err(ServiceError::Unavailable(msg)) => assert_eq!(
            msg, OPERATION_OVERTAKEN_MESSAGE,
            "X's commit must be refused as overtaken"
        ),
        other => panic!(
            "X's commit must be refused UNAVAILABLE (overtaken) by its client's next commit; \
             got {other:?}"
        ),
    }
    assert_eq!(
        (
            fx.cell(
                "live_edit_update_layer_operation_overtaken_total",
                &[("backend", "lua")]
            ),
            fx.cell(
                "live_edit_update_layer_operation_overtaken_total",
                &[("backend", "txn")]
            )
        ),
        (1, 0),
        "the structural route's commit-side overtaken verdict is counted on the lua backend"
    );
    assert_eq!(
        stored_name(&fx.tv, fx.oid, OTHER_NEW_SECTION).await,
        None,
        "X applied nothing"
    );
    assert_eq!(
        fx.commits_by(WRITER).await.len(),
        2,
        "only the writer's earlier and next calls were published"
    );
    assert_next_call_repairs_the_put_back_entry(&fx, "after a commit-side OVERTAKEN").await;
}

/// The commit script's `NOT_FOUND` and `SUPERSEDED`: between the plan and the
/// commit, another replica deletes the layer, or overwrites it with
/// `CreateLayer`. The commit refuses with the header mapping's error and
/// writes nothing, and the entry is dropped — not put back — since the
/// object it described is gone or replaced.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_commit_refused_not_found_or_superseded_drops_the_entry() {
    for overwrite in [false, true] {
        let before_commit = Staged::default();
        let fx = Fx::with(
            if overwrite {
                "dispatch-commit-superseded"
            } else {
                "dispatch-commit-not-found"
            },
            &every_shape_state(),
            BUDGET_BYTES,
            |ls| {
                ls.with_structural_commit_pre_dispatch_hook_for_test(
                    before_commit.pre_dispatch_hook(),
                )
            },
        )
        .await;
        before_commit.push(|| async {}.boxed());
        fx.update(create_section(ROOT, NEW_SECTION, "Warm"))
            .await
            .expect("the warm-up must commit");
        assert!(matches!(fx.cached(), Cached::Entry(_)), "fixture: cached");
        before_commit.push({
            let (other, session_id, layer) = (fx.replica(), fx.session_id.clone(), fx.layer);
            move || {
                async move {
                    if overwrite {
                        create(&other, &session_id, layer, &every_shape_state()).await;
                    } else {
                        other
                            .delete_layer(
                                &session_id,
                                layer,
                                Arc::new(MockExternalValueStore::new()),
                            )
                            .await
                            .expect("the other replica's delete must succeed");
                    }
                }
                .boxed()
            }
        });

        let refused = fx
            .update(create_section(ROOT, OTHER_NEW_SECTION, "Gone"))
            .await;

        before_commit.assert_ran("retirement");
        assert!(
            if overwrite {
                matches!(refused, Err(ServiceError::FailedPrecondition(_)))
            } else {
                matches!(refused, Err(ServiceError::NotFound(_)))
            },
            "overwrite={overwrite}: the commit's header mapping error; got {refused:?}"
        );
        assert_eq!(
            before_commit.fired(),
            2,
            "overwrite={overwrite}: fixture: the refusal came from the commit, after its plan \
             (the read side found the object in place)"
        );
        assert_eq!(
            stored_name(&fx.tv, fx.oid, OTHER_NEW_SECTION).await,
            None,
            "overwrite={overwrite}: the refused commit wrote nothing"
        );
        assert!(
            matches!(fx.cached(), Cached::Miss),
            "overwrite={overwrite}: the entry of an object the commit found gone or superseded \
             must be dropped; got {:?}",
            fx.cached()
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// CACHE_DIVERGED
// ═════════════════════════════════════════════════════════════════════════════

/// `CACHE_DIVERGED` end to end: the cached entry still holds `WORLD`'s old
/// name after an out-of-band rename, so the commit of a plan rewriting
/// `WORLD`'s row is refused. Nothing is written, the divergence counter and
/// `cache_diverged` tick on the hit's path (`structural`), the entry is
/// dropped (`reloads_total{reason="diverged"}`), and the retry reloads
/// (`structural_full`) and commits on the rewritten state.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_cache_diverged_writes_nothing_drops_the_entry_and_the_retry_reloads_and_commits(
) {
    let fx = Fx::new("dispatch-diverged").await;
    fx.update(create_section(ROOT, NEW_SECTION, "Warm"))
        .await
        .expect("the warm-up must commit");
    rewrite_out_of_band(&fx.tv, fx.oid, world_renamed_out_of_band("OutOfBand")).await;
    let ids_before = fx.stream_ids().await;
    let lookups_before = fx.lookups();

    fx.update(structural(vec![
        move_cmd(WORLD, ROOT, ROOT, "OutOfBand"),
        create_cmd(WORLD, OTHER_NEW_SECTION, "UnderWorld"),
    ]))
    .await
    .expect("the retry must reload and commit");

    assert_eq!(
        fx.stream_ids().await.len(),
        ids_before.len() + 1,
        "the diverged attempt wrote nothing: one entry, the retry's"
    );
    assert_eq!(fx.divergences(), 1);
    assert_eq!(
        fx.conflicts(),
        conflict_cells([(STRUCTURAL, CACHE_DIVERGED, 1)]),
        "the divergence is counted on the attempt's path, a hit's"
    );
    assert_eq!(fx.reloads(), counts([("diverged", 1)]));
    assert_eq!(
        since(fx.lookups(), &lookups_before),
        counts([("hit", 1), ("cold", 1)]),
        "the diverged attempt's entry was dropped, so the retry cold-started"
    );
    assert_eq!(
        fx.attempts(STRUCTURAL_FULL),
        (2, 3),
        "the retry's commit is on structural_full, its call's second attempt (beside the \
         warm-up's one)"
    );
    assert_eq!(fx.fallbacks(), 0, "one divergence is not a fallback");
    assert_eq!(
        stored_name(&fx.tv, fx.oid, OTHER_NEW_SECTION)
            .await
            .as_deref(),
        Some("UnderWorld")
    );
    fx.assert_current(&fx.cached_entry("after the retry"), "after the retry")
        .await;
}

/// `cache_diverged` on a reset ticks on `reset`, the attempt's path
/// (decider ruling, Q5).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_reset_cache_diverged_ticks_on_the_reset_path() {
    let fx = Fx::new("dispatch-reset-diverged").await;
    fx.update(create_section(ROOT, NEW_SECTION, "Warm"))
        .await
        .expect("the warm-up must commit");
    rewrite_out_of_band(&fx.tv, fx.oid, world_renamed_out_of_band("OutOfBand")).await;

    fx.update(reset_delta("usda"))
        .await
        .expect("the reset's retry must reload and commit");

    assert_eq!(fx.divergences(), 1, "fixture: the reset's commit diverged");
    assert_eq!(fx.conflicts(), conflict_cells([(RESET, CACHE_DIVERGED, 1)]));
    assert_eq!(
        fx.attempts(RESET),
        (1, 2),
        "the reset committed on its second attempt"
    );
    assert_eq!(
        stored_name(&fx.tv, fx.oid, WORLD).await,
        None,
        "the reset cleared the object"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Divergence never loops (decider ruling (a) and Q4)
// ═════════════════════════════════════════════════════════════════════════════

/// Plant a shape the cache's read and the script's digests disagree on,
/// then commit `delta` (whose plan touches the planted key) and a second
/// structural `next` delta:
///
/// - the call commits, once, through the WATCH path (`path="watch"`, the
///   call's second attempt): its first attempt's cold start diverged, the
///   reload rebuilt the same belief and diverged again, and the second
///   divergence fell back — one reload, no third read;
/// - `divergence_total`, `cache_diverged` (on the cold-started attempts'
///   `structural_full`) and `reloads_total{reason="diverged"}` read 2, and
///   `live_edit_update_layer_structural_fallbacks_total{reason="repeat_divergence"}`
///   reads 1;
/// - the WATCH path rewrote the key canonically, so `next` commits on the
///   structural route with no further divergence.
async fn divergence_reloads_once_then_commits_through_watch(
    layer: &'static str,
    plant: impl for<'a> FnOnce(&'a Fx) -> futures::future::BoxFuture<'a, ()>,
    delta: OwnedDelta,
    next: OwnedDelta,
) {
    let fx = Fx::new(layer).await;
    plant(&fx).await;
    let seq_before = fx.header_seq().await;

    fx.update(delta).await.unwrap_or_else(|e| {
        panic!("{layer}: the call must commit through the WATCH path; got {e:?}")
    });

    assert_eq!(fx.header_seq().await, seq_before + 1, "{layer}: one commit");
    assert_eq!(
        fx.attempts(WATCH),
        (1, 2),
        "{layer}: the call committed on the WATCH path, on its second attempt"
    );
    assert_eq!(
        (fx.divergences(), fx.reloads()),
        (2, counts([("diverged", 2)])),
        "{layer}: the first divergence and the reload's"
    );
    assert_eq!(
        fx.conflicts(),
        conflict_cells([(STRUCTURAL_FULL, CACHE_DIVERGED, 2)]),
        "{layer}: both divergences on the cold-started attempts' path"
    );
    assert_eq!(fx.fallbacks(), 1, "{layer}: one repeat-divergence fallback");
    assert_eq!(
        fx.lookups(),
        counts([("cold", 2)]),
        "{layer}: the cold start and one reload, then no third read"
    );

    fx.update(next)
        .await
        .unwrap_or_else(|e| panic!("{layer}: the next call must commit; got {e:?}"));
    assert_eq!(
        (fx.divergences(), fx.fallbacks()),
        (2, 1),
        "{layer}: the WATCH path's rewrite left nothing to diverge on"
    );
    assert_eq!(
        fx.attempts(STRUCTURAL_FULL),
        (1, 1),
        "{layer}: the next call committed on the structural route, first attempt"
    );
    fx.assert_current(&fx.cached_entry("after the next call"), layer)
        .await;
}

/// Non-canonical stored text: `WORLD`'s children list holds `CUBE` as `"02"`.
/// The read decodes it as `CUBE`; the script digests the stored text.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_divergence_on_a_non_canonical_list_element_reloads_once_then_commits_through_watch_and_never_loops(
) {
    divergence_reloads_once_then_commits_through_watch(
        "dispatch-diverge-noncanonical",
        |fx| {
            async move {
                let key = fx
                    .tv
                    .key_schema()
                    .obj_children(&fx.oid.to_string(), &WORLD.to_string());
                let _: i64 = fx.tv.client().pool.del(&key).await.expect("DEL ch:WORLD");
                let _: i64 = fx
                    .tv
                    .client()
                    .pool
                    .rpush(&key, format!("0{CUBE}"))
                    .await
                    .expect("RPUSH a non-canonical element");
            }
            .boxed()
        },
        create_section(WORLD, NEW_SECTION, "UnderWorld"),
        create_section(WORLD, OTHER_NEW_SECTION, "UnderWorldToo"),
    )
    .await;
}

/// A section row outside `obj:secs`: the read, which enumerates the index,
/// does not see it; the script digests the row that is there.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_divergence_on_a_section_row_outside_the_index_reloads_once_then_commits_through_watch_and_never_loops(
) {
    divergence_reloads_once_then_commits_through_watch(
        "dispatch-diverge-unindexed-row",
        |fx| {
            async move {
                let _: () = fx
                    .tv
                    .client()
                    .pool
                    .hset(
                        fx.tv
                            .key_schema()
                            .obj_section(&fx.oid.to_string(), &NEW_SECTION.to_string()),
                        [
                            ("parent_id", ROOT.to_string()),
                            ("name", "Ghost".to_owned()),
                            ("section_type", "1".to_owned()),
                        ],
                    )
                    .await
                    .expect("HSET an unindexed section row");
            }
            .boxed()
        },
        create_section(ROOT, NEW_SECTION, "Real"),
        create_section(ROOT, OTHER_NEW_SECTION, "RealToo"),
    )
    .await;
}

/// A wrong-typed children list of a live section not named in `obj:chs`, on
/// a `chs_complete` object: `SIBLING` has no children, and its list key holds
/// a string. The read trusts the index; the script finds the key present.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_divergence_on_a_wrong_typed_unindexed_children_list_reloads_once_then_commits_through_watch_and_never_loops(
) {
    divergence_reloads_once_then_commits_through_watch(
        "dispatch-diverge-wrongtyped-list",
        |fx| {
            async move {
                let _: () = fx
                    .tv
                    .client()
                    .pool
                    .set(
                        fx.tv
                            .key_schema()
                            .obj_children(&fx.oid.to_string(), &SIBLING.to_string()),
                        "planted-not-a-list",
                        None,
                        None,
                        false,
                    )
                    .await
                    .expect("SET over SIBLING's children list");
            }
            .boxed()
        },
        create_section(SIBLING, NEW_SECTION, "UnderSibling"),
        create_section(SIBLING, OTHER_NEW_SECTION, "UnderSiblingToo"),
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The identity fence
// ═════════════════════════════════════════════════════════════════════════════

/// The identity fence end to end (SR-4.5): a DR restore of the layer — the
/// same checkpoint generation applied twice, so the header lands at the same
/// `seq` both times with a new `commit_chain` each time — between two edits.
///
/// Edit 1, after restore 1, caches the object (its delta is refused by the
/// domain, so the entry is put back at the restored cut). Restore 2 lands,
/// from another replica, at that same `seq`. Edit 2's read finds the entry at
/// the header's `seq` under another `commit_chain`, reloads `identity` and
/// commits: no write planned on the stale copy (which its fence would refuse
/// `STALE`, every attempt), and no divergence.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_identity_fence_after_a_dr_restore_between_two_edits_forces_a_reload() {
    let fx = Fx::new("dispatch-identity").await;
    let restorer = fx.replica();
    let generation = build_checkpoint_ompf(&fx.tv, fx.oid).await;
    let restore = |restorer: &LayerStore| {
        let (ompf, snap_seq, meta) = generation.clone();
        let (tv, oid) = (&fx.tv, fx.oid);
        let restorer = restorer.clone();
        async move {
            crate::common::wipe_object_keys(tv, oid).await;
            apply_checkpoint(
                &restorer,
                oid,
                ompf,
                snap_seq,
                &meta,
                None,
                &crate::common::test_config(),
            )
            .await
            .expect("the restore must apply on a wiped object keyspace");
        }
    };

    restore(&restorer).await;
    let refused = fx.update(recreate_world()).await;
    assert!(
        matches!(refused, Err(ServiceError::InvalidArgument(_))),
        "fixture: edit 1 is refused by the domain (a duplicate section id); got {refused:?}"
    );
    let cached = fx.cached_entry("after edit 1");
    let restored_seq = fx.header_seq().await;
    assert_eq!(
        cached.seq, restored_seq,
        "fixture: edit 1 cached the restored cut"
    );

    restore(&restorer).await;
    assert_eq!(
        fx.header_seq().await,
        restored_seq,
        "fixture: the second restore of the generation lands at the same seq"
    );
    assert_ne!(
        header(&fx.tv, fx.oid).await["commit_chain"],
        cached.commit_chain,
        "fixture: under a new commit_chain"
    );
    let reloads_before = fx.reloads();

    fx.update(create_section(ROOT, NEW_SECTION, "AfterRestore"))
        .await
        .expect("edit 2 must reload and commit");

    assert_eq!(
        since(fx.reloads(), &reloads_before),
        counts([("identity", 1)]),
        "the entry from before the restore must be reloaded on its identity"
    );
    assert_eq!(fx.conflicts(), BTreeMap::new(), "no STALE, no divergence");
    assert_eq!(fx.header_seq().await, restored_seq + 1);
    assert_eq!(
        stored_name(&fx.tv, fx.oid, NEW_SECTION).await.as_deref(),
        Some("AfterRestore")
    );
    fx.assert_current(&fx.cached_entry("after edit 2"), "after edit 2")
        .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The rollback knob flipped between commits (SR-7.5)
// ═════════════════════════════════════════════════════════════════════════════

/// One commit of [`rollback_knob_flipped_between_commits_keeps_the_invariants_and_never_reloads`]:
/// the backend whose store makes it, what it is, its delta, the path it must
/// land on and — for a call on the `lua` store's structural route — the
/// cache lookup it must make and the stream entries its repair must replay.
struct KnobCommit {
    backend: UpdateLayerBackend,
    what: &'static str,
    delta: OwnedDelta,
    path: &'static str,
    lookup: Option<(&'static str, u64)>,
}

fn knob_commit(
    backend: UpdateLayerBackend,
    what: &'static str,
    delta: OwnedDelta,
    path: &'static str,
    lookup: Option<(&'static str, u64)>,
) -> KnobCommit {
    KnobCommit {
        backend,
        what,
        delta,
        path,
        lookup,
    }
}

/// Fourteen commits on one object, each by a store on a different knob value
/// from the one before and every ordered switch between the three among
/// them. The `lua` store's structural and reset calls read its cache: the
/// first cold-starts, each later one repairs exactly the commits the other
/// two stores (and its own fast calls) made since its last one.
fn knob_flip_history() -> Vec<KnobCommit> {
    use UpdateLayerBackend::{Lua, LuaWatchStructural, Txn};
    const AFTER_RESET: u64 = 72;
    const LATE: u64 = 73;
    const LATE_CHILD: u64 = 74;
    vec![
        knob_commit(
            Lua,
            "a create under WORLD",
            create_section(WORLD, NEW_SECTION, "Lamp"),
            STRUCTURAL_FULL,
            Some(("cold", 0)),
        ),
        knob_commit(
            Txn,
            "a create under SIBLING",
            create_section(SIBLING, OTHER_NEW_SECTION, "Desk"),
            WATCH,
            None,
        ),
        knob_commit(LuaWatchStructural, "a field", world_field(1), "fast", None),
        knob_commit(
            Lua,
            "a rename of WORLD",
            rename_world("World2"),
            STRUCTURAL,
            Some(("repaired", 2)),
        ),
        knob_commit(
            LuaWatchStructural,
            "a delete under SIBLING",
            structural(vec![delete_cmd(SIBLING, OTHER_NEW_SECTION)]),
            WATCH,
            None,
        ),
        knob_commit(
            Txn,
            "a field on CUBE",
            delta_with_set_field_inline(CUBE, "knob", vec![7], 0),
            WATCH,
            None,
        ),
        knob_commit(Lua, "a field", world_field(2), "fast", None),
        knob_commit(Txn, "a reset", reset_delta("usdc"), WATCH, None),
        knob_commit(
            Lua,
            "a create under the root after the reset",
            create_section(ROOT, AFTER_RESET, "AfterReset"),
            STRUCTURAL,
            Some(("repaired", 4)),
        ),
        knob_commit(
            LuaWatchStructural,
            "a reset",
            reset_delta("usda"),
            WATCH,
            None,
        ),
        knob_commit(
            Lua,
            "a reset",
            reset_delta("usdc"),
            RESET,
            Some(("repaired", 1)),
        ),
        knob_commit(
            LuaWatchStructural,
            "a create under the root",
            create_section(ROOT, LATE, "Late"),
            WATCH,
            None,
        ),
        knob_commit(Txn, "an empty commit", empty_delta(), WATCH, None),
        knob_commit(
            Lua,
            "a create under the late section",
            create_section(LATE, LATE_CHILD, "LateChild"),
            STRUCTURAL,
            Some(("repaired", 2)),
        ),
    ]
}

/// The rollback knob flipped between commits (SR-7.5; SR-10.4 "the rollback
/// knob flipped between commits on one object, with no reload"): one object
/// committed by three stores over one client, one per value of
/// `LIVE_EDIT_UPDATE_LAYER_BACKEND`, in turn — what a flip of the knob (or a
/// rollout that mixes replicas on different values) does to one object.
///
/// After every commit: it landed once on the path its store and delta take;
/// the header `commit_chain` equals [`commit_chain_fold`] over every stream
/// payload since `CreateLayer`, from the materializer's value, and its
/// `relay_gens` the relay byte budget's fold over every commit's relay length
/// (b.yvh: every value writes the same column); `obj:chs` names
/// exactly the parents owning an `obj:ch:*` key; and `chs_complete` is still
/// set. The `lua` store's structural and reset calls never reload: each makes
/// exactly the cache lookup its history predicts — the first a cold start,
/// every later one a repair replaying exactly the commits made since its last
/// structural call, on whichever path — and its entry stands at the stored
/// cut after its commit. Every value folds `commit_chain` in the same way,
/// so a repair across another value's commits never meets a mismatch
/// (`regressed`). No commit meets a conflict.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rollback_knob_flipped_between_commits_keeps_the_invariants_and_never_reloads() {
    let fx = Fx::new("dispatch-knob-flip").await;
    let other_store = |backend| {
        LayerStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone(), 4096)
            .with_update_layer_backend(backend)
            .with_update_layer_retry_max_attempts(MAX_ATTEMPTS)
            .with_metrics(Arc::clone(&fx.metrics))
    };
    let lua_watch_structural = other_store(UpdateLayerBackend::LuaWatchStructural);
    let txn = other_store(UpdateLayerBackend::Txn);

    let start_header = header(&fx.tv, fx.oid).await;
    let start_chain = start_header
        .get("commit_chain")
        .cloned()
        .expect("fixture: CreateLayer writes commit_chain");
    let start_seq: u64 = start_header["seq"].parse().expect("decimal seq");
    let history = knob_flip_history();
    let switches: BTreeSet<_> = history
        .windows(2)
        .map(|pair| {
            (
                format!("{:?}", pair[0].backend),
                format!("{:?}", pair[1].backend),
            )
        })
        .collect();
    assert!(
        history
            .windows(2)
            .all(|pair| pair[0].backend != pair[1].backend)
            && switches.len() == 6,
        "fixture precondition: every commit switches knob value, every ordered switch between \
         the three occurs; got {switches:?}"
    );

    // b.yvh: every value folds the relay byte accounting the same way, so the
    // header's `relay_gens` is the Rust fold of every commit's relay length
    // whichever value committed it. All three stores run the default budget.
    let relay_retention = live_edit::storage::relay_retention::RelayRetention::from_budget(
        live_edit::config::LAYER_STREAM_MAX_BYTES_DEFAULT,
    );
    let mut relay_gens: Option<String> = start_header
        .get(live_edit::storage::relay_retention::HEADER_FIELD_RELAY_GENS)
        .cloned();
    let mut replayed_before = fx.replayed_entries();
    for (i, commit) in history.into_iter().enumerate() {
        let what = format!("commit {} ({:?}, {})", i + 1, commit.backend, commit.what);
        // `update_on` relays exactly this container.
        let relay_len = delta_to_ompf_bytes(&commit.delta, &[]).len() as u64;
        let paths_before: BTreeMap<String, u64> = PATHS
            .iter()
            .map(|p| ((*p).to_owned(), fx.attempts(p).0))
            .collect();
        let lookups_before = fx.lookups();
        let reloads_before = fx.reloads();
        let store: &LayerStore = match commit.backend {
            UpdateLayerBackend::Lua => &fx.ls,
            UpdateLayerBackend::LuaWatchStructural => &lua_watch_structural,
            UpdateLayerBackend::Txn => &txn,
        };
        let (seq, _) = fx
            .update_on(store, commit.delta, WRITER)
            .await
            .unwrap_or_else(|e| panic!("{what}: must commit; got {e:?}"));
        relay_gens = relay_retention
            .fold(relay_gens.as_deref(), seq, relay_len)
            .map(|folded| folded.value);

        let paths_after: BTreeMap<String, u64> = PATHS
            .iter()
            .map(|p| ((*p).to_owned(), fx.attempts(p).0))
            .collect();
        assert_eq!(
            since(paths_after, &paths_before),
            counts([(commit.path, 1)]),
            "{what}: the commit must land once, on its path"
        );
        let stream = fx.stream().await;
        let folded = stream
            .iter()
            .filter(|(id, _)| {
                id.strip_suffix("-0")
                    .and_then(|seq| seq.parse::<u64>().ok())
                    .is_some_and(|seq| seq > start_seq)
            })
            .fold(start_chain.clone(), |prev, (_, payload)| {
                commit_chain_fold(&prev, payload)
            });
        let hdr = header(&fx.tv, fx.oid).await;
        assert_eq!(
            hdr.get("commit_chain"),
            Some(&folded),
            "{what}: the header commit_chain must equal the fold over every stream payload since \
             CreateLayer"
        );
        assert_eq!(
            hdr.get(live_edit::storage::relay_retention::HEADER_FIELD_RELAY_GENS),
            relay_gens.as_ref(),
            "{what}: the header relay_gens must equal the relay byte fold over every commit's \
             relay length since CreateLayer, whichever knob value committed it"
        );
        assert_eq!(
            indexed_parents(&fx.tv, fx.oid).await,
            listed_parents(&fx.tv, fx.oid).await,
            "{what}: obj:chs must name exactly the parents owning a children list"
        );
        assert_eq!(
            hdr.get(HEADER_FIELD_CHS_COMPLETE).map(String::as_str),
            Some(CHS_COMPLETE_MARKED),
            "{what}: chs_complete must still be set"
        );
        assert_eq!(
            since(fx.reloads(), &reloads_before),
            counts([]),
            "{what}: nothing reloads"
        );
        let replayed = fx.replayed_entries();
        match commit.lookup {
            Some((outcome, replays)) => {
                assert_eq!(
                    since(fx.lookups(), &lookups_before),
                    counts([(outcome, 1)]),
                    "{what}: the lua store's structural call must make one {outcome} lookup"
                );
                assert_eq!(
                    replayed - replayed_before,
                    replays,
                    "{what}: the repair must replay exactly the commits since the lua store's \
                     last structural call"
                );
                fx.assert_current(&fx.cached_entry(&what), &what).await;
            }
            None => assert_eq!(
                since(fx.lookups(), &lookups_before),
                counts([]),
                "{what}: a call off the lua store's structural route reads no cache"
            ),
        }
        replayed_before = replayed;
    }
    assert_eq!(fx.conflicts(), BTreeMap::new(), "no commit met a conflict");
}

// ═════════════════════════════════════════════════════════════════════════════
// Uncacheable routing and the per-entry cap
// ═════════════════════════════════════════════════════════════════════════════

/// Two structural calls on an object the cache cannot hold: each commits
/// through the WATCH path (`path="watch"`), the first records the negative
/// entry through the `uncacheable` cold-start route, and the second is routed
/// by it with no further cold-start read (SR-7.4).
async fn uncacheable_object_commits_through_watch(
    fx: &Fx,
    expected: fn(&UncacheableReason) -> bool,
) {
    fx.update(create_section(ROOT, NEW_SECTION, "First"))
        .await
        .expect("the first call must commit through the WATCH path");
    assert_eq!(fx.attempts(WATCH), (1, 1), "the first call's path is watch");
    assert_eq!(
        fx.uncacheable_cold_starts(),
        1,
        "the first call's cold start found the object uncacheable"
    );
    match fx.cached() {
        Cached::Uncacheable(reason) if expected(&reason) => {}
        other => panic!("the object must hold its negative entry; got {other:?}"),
    }

    fx.update(create_section(ROOT, OTHER_NEW_SECTION, "Second"))
        .await
        .expect("the second call must commit through the WATCH path");
    assert_eq!(
        fx.attempts(WATCH),
        (2, 2),
        "the second call's path is watch"
    );
    assert_eq!(
        fx.uncacheable_cold_starts(),
        1,
        "the negative entry routes the second call with no cold-start read"
    );
    assert_eq!(
        (fx.attempts(STRUCTURAL), fx.attempts(STRUCTURAL_FULL)),
        ((0, 0), (0, 0)),
        "no commit on the structural route"
    );
    for (id, name) in [(NEW_SECTION, "First"), (OTHER_NEW_SECTION, "Second")] {
        assert_eq!(stored_name(&fx.tv, fx.oid, id).await.as_deref(), Some(name));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_call_on_an_unmarked_object_commits_through_watch_and_ticks_uncacheable_once() {
    let fx = Fx::new("dispatch-unmarked").await;
    unmark(&fx.tv, fx.oid).await;
    uncacheable_object_commits_through_watch(&fx, |r| *r == UncacheableReason::NoChsComplete).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_call_on_an_object_over_the_entry_cap_commits_through_watch_and_ticks_uncacheable_once(
) {
    let fx = Fx::with(
        "dispatch-over-cap",
        &every_shape_state(),
        TINY_BUDGET_BYTES,
        |ls| ls,
    )
    .await;
    uncacheable_object_commits_through_watch(&fx, |r| {
        matches!(r, UncacheableReason::OverWeight(_))
    })
    .await;
}

/// The C3 amendment (decider-approved 2026-10-04): a commit whose successor
/// outgrows the per-entry cap is refused at the put-back (`PutBack::OverCap`),
/// and the structural route counts it on
/// `live_edit_structural_cache_reloads_total{reason="uncacheable"}` — not
/// dropped silently. The object then holds its `OverWeight` negative entry,
/// so its next structural call takes the WATCH path.
///
/// The cap is set just above the cold-started entry's weight (measured on a
/// probe store), so the cold start is admitted and a commit adding one
/// section with a [`GROWTH_NAME_BYTES`]-byte name is not.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_commit_whose_successor_outgrows_the_entry_cap_is_counted_uncacheable() {
    const GROWTH_NAME_BYTES: usize = 4_000;
    const SLACK_BYTES: u64 = 512;

    // The cold-started entry's weight, read off a probe store's cache after
    // a refused (domain-invalid) call, which caches the object unchanged.
    let probe = Fx::new("dispatch-cap-probe").await;
    let refused = probe.update(recreate_world()).await;
    assert!(
        matches!(refused, Err(ServiceError::InvalidArgument(_))),
        "fixture: the probe call is refused by the domain; got {refused:?}"
    );
    let cold_weight = probe.cached_entry("probe").state_weight_bytes;

    let fx = Fx::with(
        "dispatch-cap-successor",
        &every_shape_state(),
        4 * (cold_weight + SLACK_BYTES),
        |ls| ls,
    )
    .await;
    let (grown, grown_log) = capture(fx.update(create_section(
        ROOT,
        NEW_SECTION,
        &"g".repeat(GROWTH_NAME_BYTES),
    )))
    .await;
    grown.expect("the commit itself lands: an over-cap successor is not an error");

    assert_eq!(
        fx.attempts(STRUCTURAL_FULL),
        (1, 1),
        "fixture: a cold-start commit on the structural route (cold weight {cold_weight}, cap \
         {}); watch path: {:?}, uncacheable cold starts: {}",
        fx.cache.max_entry_bytes(),
        fx.attempts(WATCH),
        fx.uncacheable_cold_starts()
    );
    assert_eq!(
        fx.reloads(),
        counts([("uncacheable", 1)]),
        "the successor refused by the per-entry cap must be counted, once"
    );
    let weight = match fx.cached() {
        Cached::Uncacheable(UncacheableReason::OverWeight(weight)) => {
            assert!(
                weight > fx.cache.max_entry_bytes(),
                "the negative entry records the successor's weight, {weight}"
            );
            weight
        }
        other => panic!("the object must be uncacheable (OverWeight); got {other:?}"),
    };
    // The refused put-back is the structural route's own over_weight verdict
    // (`LayerStore::count_put_back`): one tick, and one line naming the call's
    // session and layer with the weight against the cap.
    assert_eq!(
        uncacheable_verdicts(&fx.metrics),
        BTreeMap::from([("over_weight".to_owned(), 1)]),
        "the refused successor is one over_weight verdict"
    );
    assert_one_verdict_line(
        &grown_log,
        fx.oid,
        "over_weight",
        &[
            ("session_id", fx.session_id.as_str().into()),
            ("layer_name", fx.layer.into()),
            ("weight_bytes", weight.into()),
            ("max_entry_bytes", fx.cache.max_entry_bytes().into()),
        ],
        &CEILING_FIELDS,
    );

    let (next, next_log) =
        capture(fx.update(create_section(ROOT, OTHER_NEW_SECTION, "Next"))).await;
    next.expect("the next call must commit through the WATCH path");
    assert_eq!(
        fx.attempts(WATCH),
        (1, 1),
        "the next call takes the WATCH path"
    );
    assert!(
        next_log.with_message(UNCACHEABLE_VERDICT_MSG).is_empty(),
        "the next call is routed by the cached verdict and must not report it again:\n{}",
        next_log.text
    );
    assert_eq!(
        uncacheable_verdicts(&fx.metrics),
        BTreeMap::from([("over_weight".to_owned(), 1)]),
        "the cached verdict ticks nothing on the next call"
    );
}

/// The outcomes of a structural attempt after which the route puts the held
/// entry back (`put_back_held`), each at its own settle arm of
/// `LayerStore::update_layer_once_structural`, and how
/// [`held_entry_repaired_past_the_cap_case`] stages it.
#[derive(Debug, Clone, Copy)]
enum HeldOutcome {
    /// A delta `Err`: the plan is refused by the domain (a duplicate
    /// section id).
    DeltaErr,
    /// `STALE`: the pre-dispatch hook lands another replica's commit.
    Stale,
    /// `FENCED`: a scripted `FENCED` reply.
    Fenced,
    /// The commit script's `Recorded`: the pre-dispatch hook lands an earlier
    /// call of the call's operation.
    Recorded,
    /// The commit script's `OVERTAKEN`: the pre-dispatch hook lands the
    /// client's next commit, off the call's start record.
    Overtaken,
    /// A transport error before the `EVALSHA` (the pre-`EVALSHA` fault
    /// queue).
    Transport,
    /// The commit script's own `ALREADY_LANDED`, staged as in
    /// [`structural_commit_already_landed_answer_is_the_own_commit_result_and_puts_the_entry_back`].
    AlreadyLanded,
}

/// One `structural_held_entry_repaired_past_the_entry_cap_is_counted_uncacheable_*`
/// case ([`held_entry_repaired_past_the_cap`]) on a fresh object, with a cache whose per-entry cap is `cap_budget / 4`.
///
/// A domain-refused call caches the cold-started entry, under the cap. Then
/// another replica commits [`growth_delta`]'s new fields, and the call under
/// test repairs the held entry over it — past the cap —
/// plans, and meets `outcome`, after which the route puts the held entry
/// back; the cap refuses it (`PutBack::OverCap`). For
/// [`HeldOutcome::AlreadyLanded`] the growth commit lands between the call's
/// two attempts instead, since its first attempt is the one whose commit
/// lands with the reply lost and is put back under the cap.
///
/// Asserted: the call's own answer for `outcome`; the call repaired the held
/// entry; `reloads_total{reason="uncacheable"}` reads exactly 1; the object
/// holds an `OverWeight` negative entry above the cap.
async fn held_entry_repaired_past_the_cap_case(outcome: HeldOutcome, cap_budget: u64) {
    let case = format!("{outcome:?}");
    let before_commit = Staged::default();
    let between_attempts = Staged::default();
    let lost = fred_errors(0, FredErrorKind::Timeout, "unused");
    let faults = fred_errors(0, FredErrorKind::IO, "unused");
    let fences = Arc::new(Mutex::new(VecDeque::new()));
    let fx = Fx::with(
        "dispatch-held-cap",
        &every_shape_state(),
        cap_budget,
        |ls| {
            ls.with_structural_commit_pre_dispatch_hook_for_test(before_commit.pre_dispatch_hook())
                .with_between_attempts_hook_for_test(between_attempts.between_attempts_hook())
                .with_lua_post_eval_fred_error_injector_for_test(Arc::clone(&lost))
                .with_lua_fred_error_injector_for_test(Arc::clone(&faults))
                .with_structural_fenced_replies_for_test(Arc::clone(&fences))
        },
    )
    .await;
    let refused = fx.update(recreate_world()).await;
    assert!(
        matches!(refused, Err(ServiceError::InvalidArgument(_))),
        "{case}: fixture: the caching call is refused by the domain; got {refused:?}"
    );
    assert!(
        matches!(fx.cached(), Cached::Entry(_)),
        "{case}: fixture: the cold-started entry is under the cap ({}), so the refused call \
         cached it",
        fx.cache.max_entry_bytes()
    );
    let growth = growth_delta;
    let grow = |client_id: &'static str, operation: Option<Operation>| {
        fx.other_replica_commits(growth(), client_id, operation, Arc::new(Mutex::new(None)))
    };
    let lookups_before = fx.lookups();

    let delta = match outcome {
        HeldOutcome::DeltaErr => recreate_world(),
        HeldOutcome::AlreadyLanded => rename_world("Renamed"),
        _ => create_section(ROOT, NEW_SECTION, "Held"),
    };
    let operation = match outcome {
        HeldOutcome::DeltaErr
        | HeldOutcome::Stale
        | HeldOutcome::Fenced
        | HeldOutcome::Transport => {
            grow(OTHER_WRITER, None)().await;
            None
        }
        HeldOutcome::Recorded => {
            grow(OTHER_WRITER, None)().await;
            let op = fx.op("held-recorded-op", b"held");
            before_commit.push(fx.other_replica_commits(
                world_field(0x61),
                WRITER,
                Some(op.clone()),
                Arc::new(Mutex::new(None)),
            ));
            Some(op)
        }
        HeldOutcome::Overtaken => {
            // The growth commit is `WRITER`'s: the call's start record.
            grow(WRITER, Some(fx.op("held-earlier-op", b"earlier")))().await;
            before_commit.push(fx.other_replica_commits(
                world_field(0x62),
                WRITER,
                Some(fx.op("held-next-op", b"next")),
                Arc::new(Mutex::new(None)),
            ));
            Some(fx.op("held-overtaken-op", b"x"))
        }
        HeldOutcome::AlreadyLanded => {
            lost.lock().unwrap().push_back(FredError::new(
                FredErrorKind::Timeout,
                "dispatch: reply lost",
            ));
            before_commit.push(|| async {}.boxed());
            let wtok = fx.tv.key_schema().obj_write_token(&fx.oid.to_string());
            let hidden: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
            between_attempts.push({
                let (client, wtok, hidden) =
                    (fx.tv.client().clone(), wtok.clone(), Arc::clone(&hidden));
                let grow = grow(OTHER_WRITER, None);
                move || {
                    async move {
                        let record: Option<String> = client
                            .pool
                            .hget(&wtok, WRITER)
                            .await
                            .expect("HGET the writer's record");
                        let _: i64 = client
                            .pool
                            .hdel(&wtok, WRITER)
                            .await
                            .expect("HDEL the writer's record");
                        *hidden.lock().unwrap() = record;
                        grow().await;
                    }
                    .boxed()
                }
            });
            before_commit.push({
                let (client, hidden) = (fx.tv.client().clone(), Arc::clone(&hidden));
                move || {
                    async move {
                        let record = hidden
                            .lock()
                            .unwrap()
                            .take()
                            .expect("fixture: attempt 1's commit recorded the writer's token");
                        let _: i64 = client
                            .pool
                            .hset(&wtok, [(WRITER, record)])
                            .await
                            .expect("HSET the writer's record back");
                    }
                    .boxed()
                }
            });
            Some(fx.op("held-own-op", b"own"))
        }
    };
    match outcome {
        HeldOutcome::Fenced => fences.lock().unwrap().push_back(FENCE_RETRY_AFTER_SHORT_MS),
        HeldOutcome::Transport => faults.lock().unwrap().push_back(FredError::new(
            FredErrorKind::IO,
            "dispatch: connection reset",
        )),
        HeldOutcome::Stale => before_commit.push(fx.other_replica_commits(
            world_field(0x63),
            OTHER_WRITER,
            None,
            Arc::new(Mutex::new(None)),
        )),
        _ => {}
    }

    let answer = fx.update_with_op(&fx.ls, delta, WRITER, operation).await;

    before_commit.assert_ran(&case);
    between_attempts.assert_ran(&case);
    assert_drained(&lost, "post-EVALSHA lost reply");
    assert_drained(&faults, "pre-EVALSHA transport error");
    assert_drained(&fences, "scripted FENCED reply");
    let answered = match outcome {
        HeldOutcome::DeltaErr => matches!(answer, Err(ServiceError::InvalidArgument(_))),
        HeldOutcome::Overtaken => matches!(
            &answer,
            Err(ServiceError::Unavailable(msg)) if msg == OPERATION_OVERTAKEN_MESSAGE
        ),
        HeldOutcome::Recorded => matches!(
            answer,
            Ok(UpdateCommit {
                outcome: OperationOutcome::AlreadyCommitted,
                ..
            })
        ),
        HeldOutcome::AlreadyLanded => matches!(
            answer,
            Ok(UpdateCommit {
                outcome: OperationOutcome::Applied,
                ..
            })
        ),
        // The retry finds the object uncacheable and commits through the
        // WATCH path.
        HeldOutcome::Stale | HeldOutcome::Fenced | HeldOutcome::Transport => answer.is_ok(),
    };
    assert!(answered, "{case}: unexpected answer {answer:?}");
    assert!(
        since(fx.lookups(), &lookups_before)
            .get("repaired")
            .is_some_and(|&n| n >= 1),
        "{case}: fixture: the call repaired the held entry over the growth commit"
    );
    assert_eq!(
        fx.reloads(),
        counts([("uncacheable", 1)]),
        "{case}: the held entry the per-entry cap refused at its put-back must be counted, once"
    );
    match fx.cached() {
        Cached::Uncacheable(UncacheableReason::OverWeight(weight)) => assert!(
            weight > fx.cache.max_entry_bytes(),
            "{case}: the negative entry records the repaired entry's weight, {weight}"
        ),
        other => panic!("{case}: the object must be uncacheable (OverWeight); got {other:?}"),
    }
}

/// The fields of new names [`growth_delta`] sets.
const HELD_GROWTH_FIELDS: usize = 64;

/// The field-only delta another replica commits to grow a held entry past
/// the cap: [`HELD_GROWTH_FIELDS`] one-byte inline fields of distinct new
/// names on `WORLD` ([`delta_with_new_field_names`]). A compact entry grows by
/// a row and an interned name each — several kilobytes, far past the cap's
/// slack — where a large inline value would grow it by nothing (the entry
/// holds no value).
fn growth_delta() -> OwnedDelta {
    delta_with_new_field_names(WORLD, "primvars:grownAttribute", HELD_GROWTH_FIELDS)
}

/// A section id the seeded tree, and the tests above, do not use.
const THIRD_NEW_SECTION: u64 = 75;

/// The verdict no caller hears, reported anyway (decider ruling, round 2 of
/// the uncacheable-reason observability change): a structural call dropped
/// mid-commit — an RPC deadline or a client cancel at the commit `await` —
/// while it holds an entry its repair grew past the per-entry cap. The
/// guard's `Drop` puts the entry back, the cap refuses it and records an
/// `OverWeight` negative entry, and nothing reports it there: no
/// `uncacheable_total` tick, no verdict line, no reload. The next structural
/// call is routed by that negative entry, and its lookup reports the verdict
/// once — one `over_weight` tick and one INFO line naming the call's session
/// and layer with the weight against the cap; a further call reports nothing.
///
/// Staged: the cap sits just above the cold-started entry's weight (measured
/// on a probe store); another replica commits [`growth_delta`]'s new fields;
/// the call under test repairs over it, plans, and its pre-dispatch
/// hook parks before the `EVALSHA`, where the test drops the call's future.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_call_dropped_mid_commit_over_the_entry_cap_is_reported_once_by_the_next_lookup()
{
    const SLACK_BYTES: u64 = 512;
    let probe = Fx::new("dispatch-drop-cap-probe").await;
    let refused = probe.update(recreate_world()).await;
    assert!(
        matches!(refused, Err(ServiceError::InvalidArgument(_))),
        "fixture: the probe call is refused by the domain; got {refused:?}"
    );
    let cold_weight = probe.cached_entry("probe").state_weight_bytes;

    let before_commit = Staged::default();
    let fx = Fx::with(
        "dispatch-drop-cap",
        &every_shape_state(),
        4 * (cold_weight + SLACK_BYTES),
        |ls| {
            ls.with_structural_commit_pre_dispatch_hook_for_test(before_commit.pre_dispatch_hook())
        },
    )
    .await;
    let refused = fx.update(recreate_world()).await;
    assert!(
        matches!(refused, Err(ServiceError::InvalidArgument(_))),
        "fixture: the caching call is refused by the domain; got {refused:?}"
    );
    assert!(
        matches!(fx.cached(), Cached::Entry(_)),
        "fixture: the cold-started entry is under the cap ({}), so it is cached",
        fx.cache.max_entry_bytes()
    );
    fx.other_replica_commits(
        growth_delta(),
        OTHER_WRITER,
        None,
        Arc::new(Mutex::new(None)),
    )()
    .await;
    let lookups_before = fx.lookups();
    let reloads_before = fx.reloads();
    let reached = Arc::new(tokio::sync::Notify::new());
    before_commit.push({
        let reached = Arc::clone(&reached);
        move || {
            async move {
                reached.notify_one();
                std::future::pending::<()>().await;
            }
            .boxed()
        }
    });

    let ((), dropped_log) = capture(async {
        let call = fx.update(create_section(ROOT, NEW_SECTION, "Dropped"));
        tokio::pin!(call);
        tokio::select! {
            answer = &mut call => panic!("the call must park at its commit; got {answer:?}"),
            () = reached.notified() => {}
        }
        // The call's future is dropped here, mid-commit, holding the
        // repaired entry.
    })
    .await;

    before_commit.assert_ran("park-at-commit");
    assert!(
        since(fx.lookups(), &lookups_before)
            .get("repaired")
            .is_some_and(|&n| n >= 1),
        "fixture: the dropped call repaired its entry over the growth commit"
    );
    let weight = match fx.cached() {
        Cached::Uncacheable(UncacheableReason::OverWeight(weight)) => {
            assert!(
                weight > fx.cache.max_entry_bytes(),
                "the dropped guard's put-back records the grown weight, {weight}"
            );
            weight
        }
        other => panic!(
            "the dropped guard's put-back must leave an OverWeight negative entry; got {other:?}"
        ),
    };
    assert!(
        dropped_log.with_message(UNCACHEABLE_VERDICT_MSG).is_empty()
            && uncacheable_verdicts(&fx.metrics).is_empty()
            && fx.reloads() == reloads_before,
        "fixture: the implicit put-back reports nothing itself (no verdict line, tick or \
         reload):\n{}",
        dropped_log.text
    );

    let (next, next_log) =
        capture(fx.update(create_section(ROOT, OTHER_NEW_SECTION, "Next"))).await;
    next.expect("the next call commits through the WATCH path");
    assert_eq!(
        uncacheable_verdicts(&fx.metrics),
        BTreeMap::from([("over_weight".to_owned(), 1)]),
        "the next call's lookup reports the unheard verdict: one tick"
    );
    assert_one_verdict_line(
        &next_log,
        fx.oid,
        "over_weight",
        &[
            ("session_id", fx.session_id.as_str().into()),
            ("layer_name", fx.layer.into()),
            ("weight_bytes", weight.into()),
            ("max_entry_bytes", fx.cache.max_entry_bytes().into()),
        ],
        &CEILING_FIELDS,
    );

    let (third, third_log) =
        capture(fx.update(create_section(ROOT, THIRD_NEW_SECTION, "Third"))).await;
    third.expect("the third call commits through the WATCH path");
    assert!(
        third_log.with_message(UNCACHEABLE_VERDICT_MSG).is_empty(),
        "a verdict already reported is not reported again:\n{}",
        third_log.text
    );
    assert_eq!(
        uncacheable_verdicts(&fx.metrics),
        BTreeMap::from([("over_weight".to_owned(), 1)]),
        "nor ticked again"
    );
    assert_eq!(
        fx.attempts(WATCH),
        (2, 2),
        "both later calls took the WATCH path"
    );
}

/// The C3 amendment's other half: a held entry that its repair grew past
/// the per-entry cap, put back after any outcome that leaves the object in
/// place without a successor (`put_back_held`), is refused by the cap
/// (`PutBack::OverCap`). Every such settle arm counts the refusal on
/// `live_edit_structural_cache_reloads_total{reason="uncacheable"}`, once,
/// and the object holds its `OverWeight` negative entry. One test per arm
/// below — a delta `Err`, `STALE`, `FENCED`, the commit script's `Recorded`,
/// `OVERTAKEN` and `ALREADY_LANDED`, and a transport error — each running
/// [`held_entry_repaired_past_the_cap_case`] (separate tests, not one loop,
/// so each stays within the module's wait budget).
///
/// The cap is set just above the cold-started entry's weight, measured on a
/// probe store, as in
/// [`structural_commit_whose_successor_outgrows_the_entry_cap_is_counted_uncacheable`].
async fn held_entry_repaired_past_the_cap(outcome: HeldOutcome) {
    const SLACK_BYTES: u64 = 512;

    let probe = Fx::new("dispatch-held-cap-probe").await;
    let refused = probe.update(recreate_world()).await;
    assert!(
        matches!(refused, Err(ServiceError::InvalidArgument(_))),
        "fixture: the probe call is refused by the domain; got {refused:?}"
    );
    let cold_weight = probe.cached_entry("probe").state_weight_bytes;
    held_entry_repaired_past_the_cap_case(outcome, 4 * (cold_weight + SLACK_BYTES)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_held_entry_repaired_past_the_entry_cap_is_counted_uncacheable_after_a_delta_err(
) {
    held_entry_repaired_past_the_cap(HeldOutcome::DeltaErr).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_held_entry_repaired_past_the_entry_cap_is_counted_uncacheable_after_stale() {
    held_entry_repaired_past_the_cap(HeldOutcome::Stale).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_held_entry_repaired_past_the_entry_cap_is_counted_uncacheable_after_fenced() {
    held_entry_repaired_past_the_cap(HeldOutcome::Fenced).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_held_entry_repaired_past_the_entry_cap_is_counted_uncacheable_after_recorded() {
    held_entry_repaired_past_the_cap(HeldOutcome::Recorded).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_held_entry_repaired_past_the_entry_cap_is_counted_uncacheable_after_overtaken()
{
    held_entry_repaired_past_the_cap(HeldOutcome::Overtaken).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_held_entry_repaired_past_the_entry_cap_is_counted_uncacheable_after_a_transport_error(
) {
    held_entry_repaired_past_the_cap(HeldOutcome::Transport).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_held_entry_repaired_past_the_entry_cap_is_counted_uncacheable_after_already_landed(
) {
    held_entry_repaired_past_the_cap(HeldOutcome::AlreadyLanded).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The put-back by outcome (SR-3.4)
// ═════════════════════════════════════════════════════════════════════════════

/// Commit once (caching the successor), run `call`, which must end in
/// `expected`'s error, and assert the original entry — the stored cut, which
/// the call left unchanged — is back in the cache: the same `(seq,
/// commit_chain)` as before the call.
async fn assert_original_put_back(
    fx: &Fx,
    call: impl std::future::Future<Output = Result<(u64, u64), ServiceError>>,
    expected: fn(&ServiceError) -> bool,
    context: &str,
) {
    let before = fx.cached_entry(&format!("{context}: before the call"));
    let refused = call.await;
    assert!(
        refused.as_ref().is_err_and(expected),
        "{context}: unexpected result {refused:?}"
    );
    let after = fx.cached_entry(&format!("{context}: after the call"));
    assert_eq!(
        (after.seq, &after.commit_chain, &after.state),
        (before.seq, &before.commit_chain, &before.state),
        "{context}: the original entry must be put back unchanged"
    );
}

/// A delta `Err` (a duplicate section id, refused by the domain while
/// planning): the original entry goes back, untouched by the plan.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_put_back_after_a_delta_err_is_the_original_entry() {
    let fx = Fx::new("dispatch-putback-delta-err").await;
    fx.update(create_section(ROOT, NEW_SECTION, "Warm"))
        .await
        .expect("the warm-up must commit");
    assert_original_put_back(
        &fx,
        fx.update(structural(vec![
            create_cmd(ROOT, OTHER_NEW_SECTION, "Planned"),
            create_cmd(ROOT, WORLD, "Again"),
        ])),
        |e| matches!(e, ServiceError::InvalidArgument(_)),
        "a delta Err",
    )
    .await;
    fx.assert_current(&fx.cached_entry("after"), "a delta Err")
        .await;
}

/// A transport error before the commit's `EVALSHA` (the pre-`EVALSHA` fault
/// queue), on a store with one attempt: the original entry goes back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_put_back_after_a_transport_error_is_the_original_entry() {
    let failures = fred_errors(0, FredErrorKind::IO, "unused");
    let fx = Fx::with(
        "dispatch-putback-transport",
        &every_shape_state(),
        BUDGET_BYTES,
        |ls| {
            ls.with_update_layer_retry_max_attempts(1)
                .with_lua_fred_error_injector_for_test(Arc::clone(&failures))
        },
    )
    .await;
    fx.update(create_section(ROOT, NEW_SECTION, "Warm"))
        .await
        .expect("the warm-up must commit");
    // Queued only now, so the warm-up's commit runs undisturbed.
    failures.lock().unwrap().push_back(FredError::new(
        FredErrorKind::IO,
        "dispatch: connection reset",
    ));
    let before = fx.keyspace().await;

    assert_original_put_back(
        &fx,
        fx.update(create_section(ROOT, OTHER_NEW_SECTION, "Failed")),
        |e| matches!(e, ServiceError::Aborted(_) | ServiceError::Unavailable(_)),
        "a transport error",
    )
    .await;
    assert_drained(&failures, "pre-EVALSHA transport error");
    assert_eq!(fx.keyspace().await, before, "nothing was written");
}

/// An in-doubt reply (the commit landed, its reply lost) on a store with one
/// attempt: the call ends retriable, and the original (repaired) entry goes
/// back — not a successor — so the next call's repair replays the landed
/// commit and plans on it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_put_back_after_an_in_doubt_reply_is_the_original_entry() {
    let lost = fred_errors(0, FredErrorKind::Timeout, "unused");
    let fx = Fx::with(
        "dispatch-putback-in-doubt",
        &every_shape_state(),
        BUDGET_BYTES,
        |ls| {
            ls.with_update_layer_retry_max_attempts(1)
                .with_lua_post_eval_fred_error_injector_for_test(Arc::clone(&lost))
        },
    )
    .await;
    fx.update(create_section(ROOT, NEW_SECTION, "Warm"))
        .await
        .expect("the warm-up must commit");
    let original = fx.cached_entry("before the in-doubt call");
    lost.lock().unwrap().push_back(FredError::new(
        FredErrorKind::Timeout,
        "dispatch: reply lost",
    ));

    let in_doubt = fx
        .update(create_section(ROOT, OTHER_NEW_SECTION, "Landed"))
        .await;
    assert!(
        matches!(
            in_doubt,
            Err(ServiceError::Aborted(_) | ServiceError::Unavailable(_))
        ),
        "a one-attempt call whose reply was lost ends retriable; got {in_doubt:?}"
    );
    assert_drained(&lost, "post-EVALSHA lost reply");
    assert_eq!(
        fx.header_seq().await,
        original.seq + 1,
        "fixture: the in-doubt commit landed"
    );
    let after = fx.cached_entry("after the in-doubt call");
    assert_eq!(
        (after.seq, &after.commit_chain),
        (original.seq, &original.commit_chain),
        "the original entry is put back, not a successor of a commit the call cannot confirm"
    );

    let lookups_before = fx.lookups();
    fx.update(create_section(ROOT, OTHER_NEW_SECTION + 1, "Next"))
        .await
        .expect("the next call must repair over the landed commit and commit");
    assert_eq!(
        since(fx.lookups(), &lookups_before),
        counts([("repaired", 1)]),
        "the next call's repair replayed the landed commit"
    );
    fx.assert_current(
        &fx.cached_entry("after the next call"),
        "after the next call",
    )
    .await;
}

/// A refusal of the commit script that changed nothing — a stream-ID
/// collision, planted on the stream top — goes back as the original entry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_put_back_after_a_stream_id_collision_is_the_original_entry() {
    let fx = Fx::new("dispatch-putback-collision").await;
    fx.update(create_section(ROOT, NEW_SECTION, "Warm"))
        .await
        .expect("the warm-up must commit");
    let next_id = format!("{}-0", fx.header_seq().await + 1);
    let _: String = fx
        .tv
        .client()
        .pool
        .xadd(
            fx.tv.key_schema().layer_stream(fx.oid),
            false,
            None,
            next_id.as_str(),
            vec![("payload", "out-of-band|planted-by-test")],
        )
        .await
        .expect("XADD a planted entry at the next seq");

    assert_original_put_back(
        &fx,
        fx.update(create_section(ROOT, OTHER_NEW_SECTION, "Collides")),
        |e| matches!(e, ServiceError::Internal(m) if m.contains("STREAM_ID_COLLISION")),
        "a stream-ID collision",
    )
    .await;
}

/// `NotFound` (another replica's `DeleteLayer`) and `SUPERSEDED` (another
/// replica's `CreateLayer` overwrite), found by the read side: the call fails
/// with the header mapping's error and the entry is dropped, not put back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_put_back_after_another_replicas_delete_or_overwrite_drops_the_entry() {
    for overwrite in [false, true] {
        let fx = Fx::new(if overwrite {
            "dispatch-drop-overwrite"
        } else {
            "dispatch-drop-delete"
        })
        .await;
        fx.update(create_section(ROOT, NEW_SECTION, "Warm"))
            .await
            .expect("the warm-up must commit");
        assert!(matches!(fx.cached(), Cached::Entry(_)), "fixture: cached");
        let other = fx.replica();
        if overwrite {
            create(&other, &fx.session_id, fx.layer, &every_shape_state()).await;
        } else {
            other
                .delete_layer(
                    &fx.session_id,
                    fx.layer,
                    Arc::new(MockExternalValueStore::new()),
                )
                .await
                .expect("the other replica's delete must succeed");
        }

        let refused = fx
            .update(create_section(ROOT, OTHER_NEW_SECTION, "Gone"))
            .await;
        assert!(
            if overwrite {
                matches!(refused, Err(ServiceError::FailedPrecondition(_)))
            } else {
                matches!(refused, Err(ServiceError::NotFound(_)))
            },
            "overwrite={overwrite}: the header mapping's error; got {refused:?}"
        );
        assert!(
            matches!(fx.cached(), Cached::Miss),
            "overwrite={overwrite}: the entry of a gone or superseded object must be dropped"
        );
    }
}

/// T9 (b.ug2): a call dropped after its commit landed but before it settled
/// — its plan, made in place on the held entry, still pending — has the
/// guard's `Drop` roll the plan back and put the pre-plan cut back, one cut
/// behind the store (a cut, not a half-planned entry); the next call repairs
/// across the landed commit, replaying its record, and commits on the cached
/// route, leaving the entry at the stored cut.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn structural_call_dropped_after_its_commit_landed_rolls_back_and_the_next_call_repairs_ug2()
{
    let parked = Arc::new(tokio::sync::Notify::new());
    let armed = Arc::new(AtomicBool::new(false));
    let hook: CommitLandedHook = {
        let (parked, armed) = (Arc::clone(&parked), Arc::clone(&armed));
        Arc::new(move || {
            let parked = Arc::clone(&parked);
            let park = armed.swap(false, Ordering::SeqCst);
            async move {
                if park {
                    parked.notify_one();
                    std::future::pending::<()>().await;
                }
            }
            .boxed()
        })
    };
    let fx = Fx::with(
        "dispatch-drop-landed",
        &every_shape_state(),
        BUDGET_BYTES,
        |ls| ls.with_commit_landed_hook_for_test(hook),
    )
    .await;
    fx.update(create_section(ROOT, NEW_SECTION, "Warm"))
        .await
        .expect("the warm-up must commit");
    let before = fx.cached_entry("before the dropped call");
    let (header_before, rollbacks_before) = (fx.header_seq().await, fx.cache.rollbacks_for_test());
    armed.store(true, Ordering::SeqCst);
    {
        let call = fx.update(create_section(ROOT, OTHER_NEW_SECTION, "Dropped"));
        tokio::pin!(call);
        tokio::select! {
            answer = &mut call => panic!("the call must park at its commit's landing; got {answer:?}"),
            () = parked.notified() => {}
        }
        // The call's future is dropped here: its commit landed, its plan is
        // pending on the held entry.
    }
    assert_eq!(
        fx.header_seq().await,
        header_before + 1,
        "fixture: the dropped call's commit landed"
    );
    assert_eq!(
        fx.cache.rollbacks_for_test() - rollbacks_before,
        1,
        "the dropped guard rolled its pending plan back"
    );
    let held = fx.cached_entry("after the drop");
    assert_eq!(
        (held.seq, &held.commit_chain, &held.state),
        (before.seq, &before.commit_chain, &before.state),
        "the pre-plan cut went back, one cut behind the landed commit"
    );

    let lookups_before = fx.lookups();
    fx.update(create_section(ROOT, THIRD_NEW_SECTION, "Next"))
        .await
        .expect("the next call commits");
    assert!(
        since(fx.lookups(), &lookups_before)
            .get("repaired")
            .is_some_and(|&n| n >= 1),
        "the next call repaired across the landed commit: {:?}",
        since(fx.lookups(), &lookups_before)
    );
    let after = fx.cached_entry("after the next call");
    fx.assert_current(&after, "after the next call").await;
}

/// The message of a caught panic's payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|m| (*m).to_owned()))
        .unwrap_or_default()
}

/// Assert `caught` is the failpoint's injected rollback panic, so the panic
/// fired inside `PlanJournal::rollback` on this thread.
fn assert_injected_rollback_panic<T: std::fmt::Debug>(
    caught: std::thread::Result<T>,
    context: &str,
) {
    match caught {
        Ok(answer) => panic!(
            "{context}: the armed rollback must panic out of the call; it answered {answer:?}"
        ),
        Err(payload) => {
            let message = panic_message(&*payload);
            assert!(
                message.starts_with(failpoint::INJECTED),
                "{context}: the panic is the failpoint's, in the rollback; got {message:?}"
            );
        }
    }
}

/// After a rollback that panicked: the entry is discarded and the object
/// invalidated — the lookup misses, nothing can be checked out, no weight
/// hint — and the next call cold-loads the stored cut. The next call is a
/// delta the domain refuses (it plans on the loaded entry and puts it back
/// unchanged), so the entry it leaves is the loaded one: the before-admit
/// hook fired once for the object, and the entry is the stored cut — the
/// header's `(seq, commit_chain)`, the store's dangling parents, the
/// normalized fresh state ([`Fx::assert_current`]). A call after it commits
/// on the cached route.
async fn assert_discarded_and_cold_loaded_next(fx: &Fx, context: &str) {
    assert!(
        matches!(fx.cached(), Cached::Miss),
        "{context}: the lookup misses: the entry was discarded"
    );
    assert!(
        fx.cache.checkout(fx.oid).is_none(),
        "{context}: nothing to check out"
    );
    assert_eq!(
        fx.cache.weight_hint(fx.oid),
        None,
        "{context}: no weight hint"
    );

    let loads = Arc::new(AtomicUsize::new(0));
    fx.cache.set_before_admit_hook_for_test(Some({
        let (loads, oid) = (Arc::clone(&loads), fx.oid);
        Arc::new(move |loaded: u64| {
            if loaded == oid {
                loads.fetch_add(1, Ordering::SeqCst);
            }
        })
    }));
    let refused = fx.update(recreate_world()).await;
    fx.cache.set_before_admit_hook_for_test(None);
    assert!(
        matches!(refused, Err(ServiceError::InvalidArgument(_))),
        "{context}: the next call's delta is refused by the domain; got {refused:?}"
    );
    assert_eq!(
        loads.load(Ordering::SeqCst),
        1,
        "{context}: the next call cold-loaded the object once"
    );
    fx.assert_current(
        &fx.cached_entry(&format!("{context}: after the cold load")),
        &format!("{context}: the cold-loaded entry"),
    )
    .await;

    let lookups_before = fx.lookups();
    fx.update(create_section(ROOT, NEW_SECTION + 10, "AfterReload"))
        .await
        .unwrap_or_else(|e| panic!("{context}: the call after commits; got {e:?}"));
    assert_eq!(
        since(fx.lookups(), &lookups_before),
        counts([("hit", 1)]),
        "{context}: the call after hits the cold-loaded entry"
    );
    fx.assert_current(
        &fx.cached_entry(&format!("{context}: after the commit")),
        &format!("{context}: after the commit"),
    )
    .await;
}

/// T8, a panic in the rollback on the put-back's path (decider): with the
/// failpoint's panic mode armed (`domain::delta::failpoint::arm_rollback_panic`),
/// a call whose plan in place the domain refuses part way (a create, then a
/// duplicate create) rolls the plan back at its put-back
/// (`CheckedOut::put_back_held`), and the rollback panics part way — after
/// restoring the plan indexes, before any state table. The panic leaves the
/// call, the guard discards the half-restored entry and invalidates the
/// object, and the next call cold-loads the stored cut
/// ([`assert_discarded_and_cold_loaded_next`]). "No panic escaped" alone is
/// not the check: the entry's absence and the reload are.
///
/// The failpoint is thread-local. A `#[tokio::test]` body is polled on the
/// test's own thread, and the call is awaited in it, so the rollback runs on
/// the thread that armed it; the panic's message proves the failpoint fired.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_rollback_panic_at_put_back_discards_the_entry_and_the_next_call_cold_loads_ug2()
{
    let fx = Fx::new("dispatch-rollback-panic-put-back").await;
    fx.update(create_section(ROOT, NEW_SECTION, "Warm"))
        .await
        .expect("the warm-up must commit");
    assert!(
        matches!(fx.cached(), Cached::Entry(_)),
        "fixture: the object is cached"
    );

    failpoint::arm_rollback_panic();
    let caught = std::panic::AssertUnwindSafe(fx.update(structural(vec![
        create_cmd(ROOT, OTHER_NEW_SECTION, "Planned"),
        create_cmd(ROOT, WORLD, "Again"),
    ])))
    .catch_unwind()
    .await;
    failpoint::disarm();
    assert_injected_rollback_panic(caught, "put-back");

    assert_discarded_and_cold_loaded_next(&fx, "put-back").await;
}

/// T8, a panic in the rollback on the dropped guard's path (decider): a call
/// parked at its commit's landing, its plan in place still pending, is
/// dropped with the failpoint's panic mode armed; the guard's `Drop` rolls
/// the plan back (inside `catch_unwind`), the rollback panics part way, and
/// the guard discards the entry, invalidates the object and resumes the
/// panic, which leaves the drop. The next call cold-loads the stored cut —
/// the dropped call's landed commit included
/// ([`assert_discarded_and_cold_loaded_next`]). The thread-local failpoint
/// is armed, and the call dropped, on the test's own thread.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn structural_rollback_panic_in_a_dropped_guard_discards_the_entry_and_the_next_call_cold_loads_ug2(
) {
    let parked = Arc::new(tokio::sync::Notify::new());
    let armed = Arc::new(AtomicBool::new(false));
    let hook: CommitLandedHook = {
        let (parked, armed) = (Arc::clone(&parked), Arc::clone(&armed));
        Arc::new(move || {
            let parked = Arc::clone(&parked);
            let park = armed.swap(false, Ordering::SeqCst);
            async move {
                if park {
                    parked.notify_one();
                    std::future::pending::<()>().await;
                }
            }
            .boxed()
        })
    };
    let fx = Fx::with(
        "dispatch-rollback-panic-drop",
        &every_shape_state(),
        BUDGET_BYTES,
        |ls| ls.with_commit_landed_hook_for_test(hook),
    )
    .await;
    fx.update(create_section(ROOT, NEW_SECTION, "Warm"))
        .await
        .expect("the warm-up must commit");
    let header_before = fx.header_seq().await;
    armed.store(true, Ordering::SeqCst);
    let mut call = Box::pin(fx.update(create_section(ROOT, OTHER_NEW_SECTION, "Dropped")));
    tokio::select! {
        answer = &mut call => panic!("the call must park at its commit's landing; got {answer:?}"),
        () = parked.notified() => {}
    }
    assert_eq!(
        fx.header_seq().await,
        header_before + 1,
        "fixture: the parked call's commit landed"
    );

    failpoint::arm_rollback_panic();
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || drop(call)));
    failpoint::disarm();
    assert_injected_rollback_panic(caught, "dropped guard");

    assert_discarded_and_cold_loaded_next(&fx, "dropped guard").await;
}

/// A local `DeleteLayer` or `CreateLayer` overwrite while a structural call
/// holds the object's entry checked out (parked at its commit's landing,
/// before it settles): the retirement invalidates the object, and the call's
/// put-back of its successor is refused by the epoch check — the call still
/// answers its commit, and no entry of the retired object is cached.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn structural_put_back_of_a_successor_is_refused_after_a_local_retirement_during_the_checkout(
) {
    for overwrite in [false, true] {
        let parked = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let armed = Arc::new(AtomicBool::new(false));
        let hook: CommitLandedHook = {
            let (parked, release, armed) = (
                Arc::clone(&parked),
                Arc::clone(&release),
                Arc::clone(&armed),
            );
            Arc::new(move || {
                let (parked, release) = (Arc::clone(&parked), Arc::clone(&release));
                let park = armed.swap(false, Ordering::SeqCst);
                async move {
                    if park {
                        parked.notify_one();
                        release.notified().await;
                    }
                }
                .boxed()
            })
        };
        let fx = Arc::new(
            Fx::with(
                if overwrite {
                    "dispatch-epoch-overwrite"
                } else {
                    "dispatch-epoch-delete"
                },
                &every_shape_state(),
                BUDGET_BYTES,
                |ls| ls.with_commit_landed_hook_for_test(hook),
            )
            .await,
        );
        fx.update(create_section(ROOT, NEW_SECTION, "Warm"))
            .await
            .expect("the warm-up must commit");
        armed.store(true, Ordering::SeqCst);

        let call = tokio::spawn({
            let fx = Arc::clone(&fx);
            async move {
                fx.update(create_section(ROOT, OTHER_NEW_SECTION, "Parked"))
                    .await
            }
        });
        tokio::time::timeout(WAIT_BUDGET, parked.notified())
            .await
            .expect("the call must park at its commit's landing");
        if overwrite {
            create(&fx.ls, &fx.session_id, fx.layer, &every_shape_state()).await;
        } else {
            fx.ls
                .delete_layer(
                    &fx.session_id,
                    fx.layer,
                    Arc::new(MockExternalValueStore::new()),
                )
                .await
                .expect("the local delete must succeed");
        }
        release.notify_one();
        tokio::time::timeout(CALL_BUDGET, call)
            .await
            .expect("the parked call must finish")
            .expect("the parked call's task must not panic")
            .unwrap_or_else(|e| {
                panic!("overwrite={overwrite}: the call's commit landed; got {e:?}")
            });

        assert!(
            matches!(fx.cached(), Cached::Miss),
            "overwrite={overwrite}: the successor's put-back must be refused by the epoch \
             check: the object was retired during the checkout"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Own adoption and the audit trigger
// ═════════════════════════════════════════════════════════════════════════════

/// This replica's own commit creating `DANGLING_PARENT` adopts the entry's
/// dangling parent (SR-3.2): the plan does not hold the list the creation
/// makes visible, so the entry is dropped (`reloads_total{reason=
/// "dangling_adopted"}`, at the commit), and the next call cold-starts the
/// stored cut — the adopted list included.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_own_adoption_drops_the_entry_and_the_next_call_cold_starts() {
    let fx = Fx::new("dispatch-own-adoption").await;
    fx.update(create_section(ROOT, NEW_SECTION, "Warm"))
        .await
        .expect("the warm-up must commit");
    assert!(
        fx.cached_entry("fixture")
            .dangling_parents
            .contains(&DANGLING_PARENT),
        "fixture: the entry holds the dangling parent"
    );

    fx.update(create_section(ROOT, DANGLING_PARENT, "Adopter"))
        .await
        .expect("the adopting commit must land");
    assert_eq!(
        fx.reloads(),
        counts([("dangling_adopted", 1)]),
        "the own adoption is counted at the commit"
    );
    assert!(
        matches!(fx.cached(), Cached::Miss),
        "the adopting commit's entry is dropped, not swapped in"
    );

    let lookups_before = fx.lookups();
    fx.update(create_section(
        DANGLING_PARENT,
        OTHER_NEW_SECTION,
        "Adopted",
    ))
    .await
    .expect("the next call must commit");
    assert_eq!(
        since(fx.lookups(), &lookups_before),
        counts([("cold", 1)]),
        "the next call cold-starts"
    );
    let entry = fx.cached_entry("after the next call");
    fx.assert_current(&entry, "after the next call").await;
    assert!(
        !entry.dangling_parents.contains(&DANGLING_PARENT),
        "the adopted parent is live, not dangling"
    );
}

/// Wait, under [`WAIT_BUDGET`], until the store's audit is not running and
/// `audit_total` reads `expected`; fail with the counts it reads instead.
async fn wait_for_audits(fx: &Fx, expected: (u64, u64, u64), context: &str) {
    let settled = || !fx.ls.structural_cache_audit().is_running() && fx.audits() == expected;
    let _ = tokio::time::timeout(WAIT_BUDGET, async {
        while !settled() {
            tokio::time::sleep(POLL).await;
        }
    })
    .await;
    assert!(
        !fx.ls.structural_cache_audit().is_running(),
        "{context}: the audit must have ended within {WAIT_BUDGET:?}"
    );
    assert_eq!(
        fx.audits(),
        expected,
        "{context}: audit_total as (match, diverged, skipped)"
    );
}

/// The audit trigger at `LIVE_EDIT_STRUCTURAL_CACHE_AUDIT_RATE` 1.0 through
/// `update_layer` (SR-3.6): the audit of a cold-start commit's successor and
/// of a hit commit's both count `match`, never `diverged`, and invalidate
/// nothing. The hit commit deletes `CUBE`, emptying `WORLD`'s list: its raw
/// planned state holds an empty list the normalized successor (and the store)
/// does not, so an audit of the un-normalized clone answers `diverged`. Each
/// commit waits for the previous audit to end: a commit made while one runs
/// starts none.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_audit_of_cold_start_and_hit_commits_counts_match_and_drops_nothing() {
    let fx = Fx::with("dispatch-audit", &every_shape_state(), BUDGET_BYTES, |ls| {
        ls.with_structural_cache_audit(StructuralCacheAudit::new(
            1.0,
            false,
            std::sync::Arc::new(live_edit::storage::commit_charge::UnboundedAuditCharges),
            CancellationToken::new(),
        ))
    })
    .await;

    fx.update(create_section(ROOT, NEW_SECTION, "Cold"))
        .await
        .expect("the cold-start commit must land");
    assert_eq!(
        fx.attempts(STRUCTURAL_FULL),
        (1, 1),
        "fixture: a cold-start commit"
    );
    wait_for_audits(&fx, (1, 0, 0), "after the cold-start commit").await;

    fx.update(structural(vec![delete_cmd(WORLD, CUBE)]))
        .await
        .expect("the hit commit must land");
    assert_eq!(fx.attempts(STRUCTURAL), (1, 1), "fixture: a hit commit");
    wait_for_audits(&fx, (2, 0, 0), "after the hit commit").await;

    fx.assert_current(&fx.cached_entry("after the audits"), "after the audits")
        .await;
    assert_eq!(fx.reloads(), counts([]), "no audit invalidated the entry");
}

/// An own adoption discards its successor, so it starts no audit
/// (`audit_total` unchanged, no `diverged`): auditing a successor that
/// misses the list the creation made visible would answer `diverged`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_audit_is_not_started_by_an_own_adoption() {
    let fx = Fx::with(
        "dispatch-audit-adoption",
        &every_shape_state(),
        BUDGET_BYTES,
        |ls| {
            ls.with_structural_cache_audit(StructuralCacheAudit::new(
                1.0,
                false,
                std::sync::Arc::new(live_edit::storage::commit_charge::UnboundedAuditCharges),
                CancellationToken::new(),
            ))
        },
    )
    .await;
    fx.update(create_section(ROOT, NEW_SECTION, "Warm"))
        .await
        .expect("the warm-up must commit");
    wait_for_audits(&fx, (1, 0, 0), "after the warm-up").await;

    fx.update(create_section(ROOT, DANGLING_PARENT, "Adopter"))
        .await
        .expect("the adopting commit must land");
    assert!(
        !fx.ls.structural_cache_audit().is_running(),
        "no audit may be running after an own adoption"
    );
    assert_eq!(
        fx.audits(),
        (1, 0, 0),
        "an own adoption starts no audit: its successor is discarded"
    );
    assert_eq!(fx.reloads(), counts([("dangling_adopted", 1)]), "fixture");
}

// ═════════════════════════════════════════════════════════════════════════════
// FENCED (Decider ruling 2026-10-03, Q6: the test hook)
// ═════════════════════════════════════════════════════════════════════════════

/// A `{"FENCED", "<retry_after_ms>"}` reply (the scripted-reply hook,
/// decoded by the one decoder both script modes use) waits at least
/// `retry_after_ms` without spending a retry attempt, ticks
/// `snapshot_fence_wait` on the attempt's path (its read was a cold start:
/// `structural_full`), writes nothing, puts the entry back unchanged — the
/// re-attempt's read is a hit with nothing to replay — and commits once the
/// injection ends. The wait is a lower bound on the call's duration: a call
/// that re-attempted without sleeping returns in a few milliseconds, well
/// under [`FENCE_RETRY_AFTER_MS`].
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_fenced_reply_waits_uncounted_puts_the_entry_back_and_commits_when_it_ends() {
    let fences = Arc::new(Mutex::new(VecDeque::from([FENCE_RETRY_AFTER_MS])));
    let fx = Fx::with(
        "dispatch-fenced",
        &every_shape_state(),
        BUDGET_BYTES,
        |ls| ls.with_structural_fenced_replies_for_test(Arc::clone(&fences)),
    )
    .await;
    let ids_before = fx.stream_ids().await;

    let started = tokio::time::Instant::now();
    fx.update(create_section(ROOT, NEW_SECTION, "AfterFence"))
        .await
        .expect("the call must commit once the fence is gone");
    let took = started.elapsed();

    assert_drained(&fences, "scripted FENCED reply");
    assert!(
        took >= Duration::from_millis(FENCE_RETRY_AFTER_MS),
        "the fenced attempt must wait out retry_after_ms ({FENCE_RETRY_AFTER_MS} ms) before \
         re-attempting; the call took {took:?}"
    );
    assert_eq!(
        fx.conflicts(),
        conflict_cells([(STRUCTURAL_FULL, SNAPSHOT_FENCE_WAIT, 1)]),
        "one fence wait, on the fenced attempt's path"
    );
    assert_eq!(
        fx.attempts(STRUCTURAL),
        (1, 1),
        "the wait spent no retry: one attempt, committed on the re-read's hit"
    );
    assert_eq!(
        fx.lookups(),
        counts([("cold", 1), ("hit", 1)]),
        "the fenced attempt put its entry back unchanged: the re-read is a hit with nothing \
         to replay"
    );
    assert_eq!(
        fx.stream_ids().await.len(),
        ids_before.len() + 1,
        "the fenced commit wrote nothing; one entry, the commit's"
    );
    fx.assert_current(&fx.cached_entry("after the commit"), "after the commit")
        .await;
}

/// A fence held past the cap fails the call `UNAVAILABLE` without the sleep
/// that would cross it — once for `LIVE_EDIT_RPC_DEADLINE_MS` and once for
/// the calling RPC's remaining deadline — with nothing written and the entry
/// put back unchanged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_fenced_reply_held_past_the_cap_fails_unavailable_and_writes_nothing() {
    let caps: [(&str, Duration, Option<Duration>); 2] = [
        ("rpc-deadline cap", Duration::from_millis(50), None),
        (
            "remaining deadline",
            Duration::from_secs(60),
            Some(Duration::from_millis(50)),
        ),
    ];
    for (case, cap, remaining) in caps {
        let fences = Arc::new(Mutex::new(VecDeque::new()));
        let fx = Fx::with(
            "dispatch-fence-cap",
            &every_shape_state(),
            BUDGET_BYTES,
            |ls| {
                ls.with_structural_fenced_replies_for_test(Arc::clone(&fences))
                    .with_fence_wait_deadline(cap, Arc::new(move || remaining))
            },
        )
        .await;
        // Cache the object first, so "put back unchanged" has an entry to
        // compare; the fences are queued only after the warm-up's commit.
        fx.update(create_section(ROOT, NEW_SECTION, "Warm"))
            .await
            .expect("the warm-up must commit");
        fences
            .lock()
            .unwrap()
            .extend([FENCE_RETRY_AFTER_LONG_MS, FENCE_RETRY_AFTER_LONG_MS]);
        let before = fx.keyspace().await;

        let original = fx.cached_entry(case);
        let refused = fx
            .update(create_section(ROOT, OTHER_NEW_SECTION, "Fenced"))
            .await;
        let err = refused
            .as_ref()
            .expect_err("a fence held past the cap fails the call");
        let message = assert_refusal_error(
            err,
            refusal_reason::SNAPSHOT_FENCE_TIMEOUT,
            &[],
            CAPACITY_RETRY_DELAY,
        );
        // With a remaining deadline in scope the text names what was left
        // of it beside the path.
        let paren = match remaining {
            Some(left) => format!(
                "{STRUCTURAL}; {} ms left of the call's deadline",
                left.as_millis()
            ),
            None => STRUCTURAL.to_owned(),
        };
        let tail = format!(
            " ms of its {} ms snapshot-fence wait cap on object {} ({paren}); nothing was \
             written, retry",
            cap.as_millis(),
            fx.oid
        );
        assert!(
            message.starts_with("update_layer waited ") && message.ends_with(&tail),
            "{case}: a fence held past the cap fails UNAVAILABLE with the fence-wait cap's \
             message, naming the cap, the object, the hit attempt's path and any remaining \
             deadline, and the SNAPSHOT_FENCE_TIMEOUT retry hint; got {message:?}"
        );
        assert_eq!(
            fences.lock().unwrap().len(),
            1,
            "{case}: one fenced attempt, then no further attempt"
        );
        assert_eq!(fx.keyspace().await, before, "{case}: nothing written");
        let after = fx.cached_entry(case);
        assert_eq!(
            (after.seq, &after.commit_chain, &after.state),
            (original.seq, &original.commit_chain, &original.state),
            "{case}: the entry is put back unchanged"
        );
        assert_eq!(
            fx.conflicts(),
            conflict_cells([(STRUCTURAL, SNAPSHOT_FENCE_WAIT, 1)]),
            "{case}: the fence is counted on the hit attempt's path"
        );
    }
}

/// The fence-wait cap is per call, not per wait: several short fences that
/// are each far under `LIVE_EDIT_RPC_DEADLINE_MS` fail the call
/// `UNAVAILABLE` once together they would cross it. With
/// [`FENCE_CUMULATIVE_CAP`] at 250 ms and four
/// [`FENCE_RETRY_AFTER_SHORT_MS`] (100 ms) fences queued, two sleeps are
/// taken (ending near 200 ms) and the third is refused (it would end near
/// 300 ms): three fences consumed, one left, nothing written, the entry put
/// back unchanged. On a loaded runner the re-reads between fences can eat the
/// second sleep's ~50 ms margin, so the refusal may come at the second fence
/// instead: one or two fences left, at least one sleep taken. Either way a
/// cap applied to each wait alone would take all four sleeps and commit,
/// which is what this test rules out.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_fenced_replies_that_together_cross_the_cap_fail_unavailable() {
    const QUEUED: usize = 4;
    let fences = Arc::new(Mutex::new(VecDeque::new()));
    let fx = Fx::with(
        "dispatch-fence-cumulative",
        &every_shape_state(),
        BUDGET_BYTES,
        |ls| {
            ls.with_structural_fenced_replies_for_test(Arc::clone(&fences))
                .with_fence_wait_deadline(FENCE_CUMULATIVE_CAP, Arc::new(|| None::<Duration>))
        },
    )
    .await;
    fx.update(create_section(ROOT, NEW_SECTION, "Warm"))
        .await
        .expect("the warm-up must commit");
    fences
        .lock()
        .unwrap()
        .extend([FENCE_RETRY_AFTER_SHORT_MS; QUEUED]);
    let before = fx.keyspace().await;
    let original = fx.cached_entry("before the fenced call");

    let started = tokio::time::Instant::now();
    let refused = fx
        .update(create_section(ROOT, OTHER_NEW_SECTION, "Fenced"))
        .await;
    let took = started.elapsed();

    let err = refused.as_ref().expect_err(
        "fences that together cross the per-call cap fail UNAVAILABLE. An OK here is a cap \
         applied to each wait alone",
    );
    let message = assert_refusal_error(
        err,
        refusal_reason::SNAPSHOT_FENCE_TIMEOUT,
        &[],
        CAPACITY_RETRY_DELAY,
    );
    let cap_ms = FENCE_CUMULATIVE_CAP.as_millis();
    let tail = format!(
        " ms of its {cap_ms} ms snapshot-fence wait cap on object {} ({STRUCTURAL}); nothing was \
         written, retry",
        fx.oid
    );
    assert!(
        message.starts_with("update_layer waited ") && message.ends_with(&tail),
        "the per-call cap's refusal names the cap, the object, the path and the nothing-written \
         outcome; got {message:?}"
    );
    let left = fences.lock().unwrap().len();
    assert!(
        (1..=2).contains(&left),
        "the cap refused the second or third fence, so one or two of the {QUEUED} are left \
         (no further attempt); {left} left"
    );
    let met = (QUEUED - left) as u64;
    assert!(
        took >= Duration::from_millis(FENCE_RETRY_AFTER_SHORT_MS),
        "the call waited out at least the first fence before the cap refused one; it took \
         {took:?}"
    );
    assert_eq!(fx.keyspace().await, before, "nothing written");
    let after = fx.cached_entry("after the fenced call");
    assert_eq!(
        (after.seq, &after.commit_chain, &after.state),
        (original.seq, &original.commit_chain, &original.state),
        "the entry is put back unchanged"
    );
    assert_eq!(
        fx.conflicts(),
        conflict_cells([(STRUCTURAL, SNAPSHOT_FENCE_WAIT, met)]),
        "every fence met is counted, on the hit attempts' path"
    );
}
