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

//! b.pgs — the commit release ring, the lost-reply lookup and the
//! first-attempt hand-back, through real commits on the fast, cached
//! structural and WATCH routes.
//!
//! A commit owing external-value releases writes an entry of the object's
//! commit release ring (`le:{oid}:obj:wrel`) in its own atomic unit:
//! `|t|<call token>` → `"<seq> <reason>:<hash_hex> …"`, the start slot
//! `|s|<k>` naming the token, covered slots `""`, and `|n|` the next slot. An
//! entry owing R releases takes ⌈R / 8⌉ slots of 256; writing it evicts the
//! entry of every other token whose start slot it takes. A call that ends
//! without proof of its outcome — retry exhaustion, a writer-guard refusal
//! after a counted retry, or a refusal-detail answer (`UNAVAILABLE` with a
//! detail) after a counted retry — looks its own entry up by token and
//! enqueues it
//! (`live_edit_update_layer_lost_reply_release_lookup_total{outcome}`). A
//! writer-guard refusal on a call's first attempt instead hands back the
//! call's pre-flight increments as `Resend(<token>)` members, because nothing
//! of the call ran.
//!
//! The ring's write rule edge by edge (`|n|` parsing, wrap, the slot clamp,
//! what eviction skips) is unit tested beside `plan_release_ring_write` in
//! `src/storage/layer.rs`; the chained lost reply whose successor commits
//! first is `b_yuo_wrel_tests`' test 30; the first-attempt hand-back of an
//! overtaken call with a stale start record is `b_v6d_tests`' T24; and the
//! recovery re-issue's first-attempt refusal, which hands back its own
//! pre-flight increment (the re-issue pre-stores for itself, b.5wb), is
//! `grpc::chained_writes_test` group 18.
//!
//! The ring's TTL (b.5wb): every releasing commit sets
//! `PEXPIRE le:{oid}:obj:wrel <TTL>` in its own atomic unit, so the key
//! expires that long after the object's last releasing commit, and an
//! expired ring reads as absent.
//!
//! # What lives here
//!
//! | Test | Item | What it pins |
//! |---|---|---|
//! | `*_overtaken_lost_reply_recovers_its_releases` | T-i1 | an unchained operation call whose reply is lost, overtaken at attempt 2 by another commit of its `client_id`: still answered overtaken; its own `Commit` member is enqueued by the lookup (`enqueued` 1) beside the other commit's, with no hand-back; GC drops each released hash by one |
//! | `*_successor_committing_before_the_own_call_read_leaves_the_entry_to_it` | T-i2 | a lost reply answered from the call's own commit, a successor of the same `client_id` committing between that verdict and the call's entry read (the read hook): both members pending, once each — the successor's write took the next slot |
//! | `*_evicted_entry_is_absent_to_the_lost_reply_lookup` | T-i3 | RING + 1 single-release commits between a lost reply and its retry: the call's own entry is evicted, the lookup finds it `absent` and enqueues nothing of it, exactly RING entries remain, and every successor's own member is pending once |
//! | `*_exhausted_call_recovers_its_landed_commits_releases` | T-i4 | attempt 1 commits and loses its reply, attempts 2 and 3 fail: the call ends `ABORTED` (the retry budget) and its lookup enqueues its commit's member |
//! | [`commit_release_ring_writes_are_identical_on_every_route`] | T-i5 | one scripted sequence, pinned tokens, on all three routes — a 2-slot entry, a token written twice by two `client_id`s, and RING single-release commits that wrap the ring past both — leaves byte-identical slots and `|n|`, the same field set, and the same entries as (sequence, token set), after the shared token and at the end; every route leaves the key a TTL |
//! | `*_legacy_per_client_field_is_never_read_or_written` | T-i6 | a legacy per-`client_id` field naming the next commit's sequence: an own-call verdict and a lost-reply lookup for commits owing nothing, then a releasing commit, leave it byte-unchanged, and no member derives from it; the releasing commit gives the key a TTL, and the field expires with the key |
//! | `*_releasing_commits_set_and_refresh_the_ring_ttl` | T-i16 | through a store with a 1 h ring TTL: a commit owing nothing creates no ring; a releasing commit leaves it a TTL in (30 min, 1 h]; a commit owing nothing leaves a TTL aged to 30 min alone; a second releasing commit raises it above 30 min again |
//! | `*_expired_ring_is_absent_to_the_lost_reply_lookup` | T-i17 | a lost reply whose ring expires before the call's retries run out: the lookup counts `absent` and enqueues nothing |
//! | `*_out_of_range_ring_ttls_are_clamped_and_commit` | T-i18 | a store given a ring TTL of `u64::MAX` or `0` clamps it to twelve digits or 1 ms, and its releasing commits land |
//! | [`commit_release_ring_ttl_is_the_deadline_plus_the_command_timeout_plus_a_minute_clamped`] | — | `keys::commit_release_ring_ttl`: 95,000 ms at the defaults, never 0, clamped to twelve digits |
//! | `*_twice_failed_call_refused_after_its_successor_recovers_once_and_hands_back_nothing` | T-i7, T-ii3 | attempt 1 commits and loses its reply, attempt 2 fails, the successor commits, attempt 3 is refused `PREDECESSOR`: one copy of each `Commit` member, no `Resend`, GC drops each hash by one |
//! | `*_ring_stays_bounded_under_writer_churn` | T-i8 | 3 × RING single-release commits by 3 × RING distinct `client_id`s leave exactly 2·RING + 1 fields, every one of the ring's own shapes |
//! | `*_empty_client_id_lost_reply_recovers_by_its_token` | T-i9 (a) | under the empty `client_id`, a lost reply's own-key retry enqueues its member, and the ring is exactly `|t|<token>`, `|s|0`, `|n|` |
//! | `*_empty_client_id_successor_leaves_the_first_calls_entry_intact` | T-i9 (b) | a second empty-`client_id` call committing with releases between the first's lost reply and its retry leaves the first's entry intact, read inside the hook |
//! | `*_wide_entry_evicts_the_two_oldest_and_its_lost_reply_enqueues_all_nine` | T-i10 | on a full ring a commit owing 9 releases takes two slots: it evicts exactly the two oldest entries, clears its second slot to `""`, advances `|n|` by two, lists all nine, and after its lost reply all nine are enqueued |
//! | `*_unparseable_entry_is_a_failed_lookup_that_enqueues_nothing` | T-i11 | an entry that does not parse, met by an exhausted call's lookup: still `ABORTED`, the lookup counts `failed` once, nothing is enqueued |
//! | `*_exhaustion_at_attempt_one_recovers_its_landed_commits_releases` | T-i12 | a one-attempt budget whose only reply is lost: `ABORTED` after one attempt, and the lookup still enqueues the commit's member |
//! | `{fast,watch}_path_fence_wait_refusal_after_a_lost_reply_recovers_its_releases` | T-i13 | a refusal-detail answer (the fence wait's `SNAPSHOT_FENCE_TIMEOUT`, a live reader's fence) after a counted retry: a plain `UNAVAILABLE`, outcome unknown, and the lookup enqueues the call's member beside the successor's; not on the structural route, whose retry re-plans the call's own `CreateSection` and is refused as a duplicate first |
//! | `*_endings_outside_the_lookup_set_never_look_up` | T-i14 | after a lost reply, a retry ending `NOT_FOUND`, `FAILED_PRECONDITION` or `INTERNAL` runs no lookup and enqueues nothing of the call's entry, which is there to be found |
//! | `*_own_call_read_ignores_an_entry_of_another_commit_under_its_token` | T-i15 | an own-call read whose token's entry names another commit's sequence (the same token, another `client_id`) enqueues nothing |
//! | `*_first_attempt_predecessor_refusal_hands_back_its_preflight_increment` | T-ii1 | a chained call refused `ABORTED` `PREDECESSOR_NOT_COMMITTED` on attempt 1 hands back one `Resend(<token>)` member; GC returns the hash to its baseline; no lookup runs |
//! | `*_first_attempt_overtaken_refusal_hands_back_its_preflight_increment` | T-ii2 | the same for an unchained call overtaken on attempt 1 |
//! | [`structural_route_fence_wait_keeps_attempt_one_so_a_predecessor_refusal_hands_back`], [`watch_path_recharge_keeps_attempt_one_so_a_predecessor_refusal_hands_back`] | T-ii5 | a `FENCED` reply (structural route) or a refused WA0 charge (WATCH path) re-runs attempt 1 under the same number: the call reports one attempt, is refused `PREDECESSOR_NOT_COMMITTED`, and hands back |
//!
//! T-ii3 (no hand-back after a counted retry) is the exact pending set of
//! T-i1, T-i7 and test 30: each would hold a `Resend` member if the
//! hand-back ran after attempt 1.
//!
//! # Fail-on-old proofs
//!
//! Each release-recovery proof perturbs the store under test with
//! `LayerStore::with_release_recovery_override_for_test` (applied to what
//! `release_recovery` decides, where the call applies it), never the source:
//!
//! - `SkipInDoubtLookup`: T-i1, T-i4, T-i7, T-i12 and T-i13 lose the call's
//!   own `Commit` member, T-i3 and T-i6 their lookup's `absent` tick, and
//!   T-i11 its `failed` tick, on every route they run on;
//! - `HandBackOnEveryAttempt`: T-i1, T-i3 and T-i7 gain a `Resend` member of
//!   the call's pre-flight increment, on every route;
//! - `SkipAttemptOneHandBack`: T-ii1 and T-ii2 on every route, and both
//!   T-ii5 cases, lose their `Resend` member.
//!
//! The TTL tests' red runs (b.5wb) perturb the source, not the store:
//! removing the ring's `PEXPIRE` from both the script's step 7a and the
//! WATCH path's `MULTI` — the pre-b.5wb ring — makes T-i5, T-i6, T-i16 and
//! T-i18 read `PTTL` `-1` and fails T-i17's premise (its lost commit's ring
//! carries a TTL), on every route; a store that does not clamp its ring TTL
//! fails T-i18 on every route (the script refuses the out-of-range TTL
//! `ENCODING`, the WATCH path's `EXEC` its `PEXPIRE`).
//!
//! T-i2, T-i5, T-i6, T-i8, T-i9 and T-i10 are inherent: a per-`client_id`
//! record — one field a successor of the same writer replaces, read by
//! `client_id` — fails each one's exact-field or exact-member assertion.
//!
//! T-i14 and T-i15 guard rules no seam can switch off — the closed set of
//! endings (unit-tested ending by ending beside `release_recovery` and
//! `lookup_on` in `src/storage/layer.rs`), and the own-call read's sequence
//! check — so they have no red run. Each checks its own premise instead: the call's entry is there for a
//! lookup to find (T-i14), and the token's entry names the other commit's
//! sequence (T-i15), so a lookup on those endings, or a read without the
//! check, would enqueue the very member each test refuses.
//!
//! # Fixtures
//!
//! Every call pre-stores each distinct external hash its delta references
//! before it is dispatched, as the `UpdateLayer` handler does
//! ([`Target::call`]): the hand-back returns exactly what was paid. A hash
//! whose GC count a test reads keeps a reference GC does not take, so a
//! count GC lowers stays readable — GC deletes a blob at zero, and an absent
//! count cannot tell one decrement from two. Where the test's own commits
//! release the hash, that reference is held outside the object
//! ([`hold_elsewhere`]); in the (ii) tests, which release nothing of their
//! own, it is the object's: an earlier commit of the object (P, or the
//! writer's first commit) still holds [`H`] when its count is read.
//!
//! # Budget
//!
//! Every call is bounded by [`CALL_BUDGET`]. The longest — T-i3's call,
//! whose between-attempts hook lands RING + 1 commits, and T-i8's 3 × RING
//! sequential commits — take seconds on the healthy path; the budget only
//! turns a hang into a failure inside nextest's 120 s hang-killer.
//!
//! # Feature gate
//!
//! The whole module drives `LayerStore`'s `test-fault-injection` seams and is
//! compiled out, not skipped, without `--features test-fault-injection`.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fred::clients::Client;
use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface};
use futures::FutureExt;

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::delta::MutationAllowance;
use live_edit::error::ServiceError;
use live_edit::grpc::upload_admission::{
    object_state_charge_for_test, object_state_rewrite_allowance_for_test,
};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::commit_charge::{CommitCharge, ObjectStateNeed};
use live_edit::storage::external_values::{
    process_pending_gc, ExtRelease, GcPolicy, ValkeyExternalValueStore,
};
use live_edit::storage::keys::{sha256_to_hex, KeySchema};
use live_edit::storage::layer::{
    operation_request_digest, BetweenAttemptsHook, LayerStore, Operation, ReleaseRead,
    ReleaseRecordReadHook, StartRecord, UpdateCommit, UpdateLayerCallSummary,
    OPERATION_OVERTAKEN_MESSAGE,
};
use live_edit::storage::test_support::relay::{relay_ompf, RelayedUpdates};
use live_edit::storage::test_support::TestValkey;
use live_edit::storage::watched_txn::{WatchWindowHook, WatchedExecOverride};
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand};

use super::b_qpf_tests::{create_session, empty_state, ext_hash, ext_store, pre_store, EXT_SIZE};

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

/// Deadline on one call. The healthy path takes milliseconds; T-i3's call
/// lands RING + 1 commits from its between-attempts hook inside it.
const CALL_BUDGET: Duration = Duration::from_secs(60);

/// The retry budget of every store here, as in `b_yuo_wrel_tests`.
const MAX_ATTEMPTS: u32 = 8;

/// The ring's slots, RING. Restated rather than read from
/// `storage::layer::COMMIT_RELEASE_RING_SLOTS` (crate-private): the counts
/// below are the contract's, and a resized ring should fail them.
const RING: usize = 256;

/// The releases one ring slot accounts for, M. Restated for the same reason.
const SLOT_RELEASES: usize = 8;

/// The commit release ring's TTL a store built with `LayerStore::new` sets,
/// in ms: the default RPC deadline (30 s) + the default Valkey command
/// timeout (5 s) + 60 s. Restated for the same reason as [`RING`].
const DEFAULT_RING_TTL_MS: i64 = 95_000;

/// The ring TTL of a store built to let the ring expire within a test
/// (`LayerStore::with_commit_release_ring_ttl`).
const SHORT_RING_TTL_MS: u64 = 300;

/// Bound on each wait for a ring whose TTL is at most [`SHORT_RING_TTL_MS`]
/// to expire: far past that TTL, so only a ring that never expires reaches it.
const EXPIRY_BUDGET: Duration = Duration::from_secs(20);

/// The writer whose commits each scenario is about.
const WRITER: &str = "b-pgs-writer";

/// The section every field lives in (created by a structural-route
/// fixture's first commit; the fast and WATCH routes write rows without it).
const SECTION_ID: u64 = 1;

/// The field each scenario's subject commit rewrites.
const FIELD: &str = "color";
/// A second field, for a delta that references a hash without rewriting
/// [`FIELD`].
const OTHER: &str = "other";
/// The field a churn of single-release commits rewrites back and forth.
const G_FIELD: &str = "churn";

/// The hash [`FIELD`] holds first, and the one a rewrite releases.
const A: u8 = 0xA1;
/// The hash a rewrite of [`FIELD`] writes.
const B: u8 = 0xB1;
/// The hash a successor's rewrite of [`FIELD`] writes.
const C: u8 = 0xC1;
/// The hash a first-attempt refusal's delta references.
const H: u8 = 0xD1;
/// The two hashes [`G_FIELD`] alternates between.
const G0: u8 = 0xE0;
const G1: u8 = 0xE1;

/// The hash of the `i`-th field a wide commit releases (`i` < 9).
fn wide_hash(i: usize) -> u8 {
    0x10 + i as u8
}

/// The `i`-th field a wide commit writes.
fn wide_field(i: usize) -> String {
    format!("wide{i}")
}

/// Pinned commit tokens (`LayerStore::with_commit_tokens_for_test`), each
/// naming its call's ring entry and its hand-back's `Resend` member.
const TOKEN_A: u128 = 0xA;
const TOKEN_B: u128 = 0xB;
const TOKEN_C: u128 = 0xC;
const TOKEN_W: u128 = 0xF;
const TOKEN_X: u128 = 0xE;

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// The route a scenario runs on, by its delta shape and backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    /// Field-only deltas on the `lua` backend: the fast-path script.
    Fast,
    /// Deltas carrying a `CreateSection` on the `lua` backend: the cached
    /// structural route and the script's structural mode.
    Structural,
    /// Field-only deltas on the `txn` backend: WATCH/MULTI/EXEC.
    Watch,
}

impl Route {
    fn backend(self) -> UpdateLayerBackend {
        match self {
            Route::Fast | Route::Structural => UpdateLayerBackend::Lua,
            Route::Watch => UpdateLayerBackend::Txn,
        }
    }

    /// The call-summary paths a commit on this route lands on.
    fn paths(self) -> &'static [&'static str] {
        match self {
            Route::Fast => &["fast"],
            Route::Structural => &["structural", "structural_full"],
            Route::Watch => &["watch"],
        }
    }
}

/// What a delta writes into one field.
#[derive(Clone, Copy, Debug)]
enum Val {
    /// The external value `[byte; 32]`.
    Ext(u8),
    /// The one-byte inline value.
    Inline(u8),
}

/// One object under test and everything a call on it needs; cheap to clone
/// into a hook.
#[derive(Clone)]
struct Target {
    client: ValkeyClient,
    keys: KeySchema,
    metrics: Arc<Metrics>,
    ext: Arc<ValkeyExternalValueStore>,
    session_id: String,
    layer: String,
    oid: u64,
    /// How many structural-route deltas were built: each creates its own
    /// section, `SECTION_ID + n`, so the first creates [`SECTION_ID`].
    sections: Arc<AtomicU64>,
}

/// The fixture: the `TestValkey` (whose drop cleans the keys up) and the
/// object.
struct Fx {
    _tv: TestValkey,
    t: Target,
}

/// A fresh object at `seq` 1 on a fresh session named `layer`.
async fn fx(layer: &str) -> Fx {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("b-pgs"));
    let ext = ext_store(&tv);
    let session_id = create_session(&tv, layer).await;
    let oid = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .create_layer(
            &session_id,
            layer,
            "b-pgs-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");
    let t = Target {
        client: tv.client().clone(),
        keys: tv.key_schema().clone(),
        metrics,
        ext,
        session_id,
        layer: layer.to_owned(),
        oid,
        sections: Arc::new(AtomicU64::new(0)),
    };
    Fx { _tv: tv, t }
}

/// The result of one call and its summary.
type Called = (
    Result<UpdateCommit, ServiceError>,
    Option<UpdateLayerCallSummary>,
);

impl Target {
    /// A store on `route`'s backend, reporting into the fixture's registry:
    /// the one place every store here is built.
    fn store(&self, route: Route) -> LayerStore {
        LayerStore::new(self.client.clone(), self.keys.clone(), 4096)
            .with_update_layer_backend(route.backend())
            .with_update_layer_retry_max_attempts(MAX_ATTEMPTS)
            .with_metrics(self.metrics.clone())
    }

    /// A delta writing `fields` into [`SECTION_ID`]; on the structural route
    /// it also creates a section of its own.
    fn delta(&self, route: Route, fields: &[(&str, Val)]) -> OwnedDelta {
        let n = self.sections.fetch_add(1, SeqCst);
        let struct_commands = match route {
            Route::Structural => vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id: SECTION_ID + n,
                section_name: format!("PgsSection{n}"),
                section_type: 1,
            })],
            Route::Fast | Route::Watch => Vec::new(),
        };
        let set_fields = fields
            .iter()
            .enumerate()
            .map(|(i, (key, val))| {
                let (value_or_ext_hash, ext_value_size, ext_value_index) = match val {
                    Val::Ext(h) => (ext_hash(*h).to_vec(), EXT_SIZE, i as u32 + 1),
                    Val::Inline(b) => (vec![*b], 0, 0),
                };
                SetField {
                    section_id: SECTION_ID,
                    key_name: (*key).to_owned(),
                    value_or_ext_hash,
                    ext_value_size,
                    ext_value_index,
                    set_order: i as u64,
                }
            })
            .collect();
        OwnedDelta {
            struct_commands,
            set_fields,
            ..Default::default()
        }
    }

    /// One call of `client_id` writing `fields` on `route` through `ls`,
    /// under [`CALL_BUDGET`], after the pre-flight store of each distinct
    /// external hash the delta references — once per call, as the handler
    /// pays it.
    async fn call(
        &self,
        ls: &LayerStore,
        route: Route,
        fields: &[(&str, Val)],
        client_id: &str,
        ts: u64,
        operation: Option<Operation>,
    ) -> Called {
        let hashes: BTreeSet<u8> = fields
            .iter()
            .filter_map(|(_, val)| match val {
                Val::Ext(h) => Some(*h),
                Val::Inline(_) => None,
            })
            .collect();
        let hashes: Vec<[u8; 32]> = hashes.into_iter().map(ext_hash).collect();
        pre_store(&self.ext, &hashes).await;
        let delta = self.delta(route, fields);
        tokio::time::timeout(
            CALL_BUDGET,
            ls.update_layer_with_summary_relayed(
                &self.session_id,
                &self.layer,
                self.oid,
                Some(delta),
                client_id,
                ts,
                operation,
            ),
        )
        .await
        .expect("update_layer did not settle within the call budget")
    }

    /// [`Self::call`] with no operation, which must land on `route`; returns
    /// its sequence.
    async fn commit(
        &self,
        ls: &LayerStore,
        route: Route,
        fields: &[(&str, Val)],
        client_id: &str,
        ts: u64,
    ) -> u64 {
        self.commit_with(ls, route, fields, client_id, ts, None)
            .await
    }

    /// [`Self::call`] carrying `operation`, which must land on `route`;
    /// returns its sequence.
    async fn commit_with(
        &self,
        ls: &LayerStore,
        route: Route,
        fields: &[(&str, Val)],
        client_id: &str,
        ts: u64,
        operation: Option<Operation>,
    ) -> u64 {
        let (result, summary) = self.call(ls, route, fields, client_id, ts, operation).await;
        let seq = result
            .unwrap_or_else(|e| panic!("{route:?}: the commit must land; got {e:?}"))
            .seq;
        let path = summary.map(|s| s.path);
        assert!(
            path.is_some_and(|p| route.paths().contains(&p)),
            "{route:?}: the commit at {seq} landed on {path:?}"
        );
        seq
    }

    /// The `Commit(seq)` member of `hash`'s release on this object.
    fn commit_member(&self, hash: u8, seq: u64) -> String {
        ExtRelease::commit(ext_hash(hash), self.oid, seq).member()
    }

    /// The `Resend(token)` member of `hash`'s hand-back on this object.
    fn resend_member(&self, hash: u8, token: u128) -> String {
        ExtRelease::resend(ext_hash(hash), self.oid, token).member()
    }

    /// Every member of `le:ext:{gc}:pending`, sorted.
    async fn pending(&self) -> Vec<String> {
        let mut members: Vec<String> = self
            .client
            .pool
            .smembers(self.keys.ext_pending())
            .await
            .expect("SMEMBERS ext pending must succeed");
        members.sort();
        members
    }

    /// The commit release ring, every field; empty when the key is absent.
    async fn ring(&self) -> BTreeMap<String, String> {
        self.client
            .pool
            .hgetall(self.keys.obj_write_releases(&self.oid.to_string()))
            .await
            .expect("HGETALL obj:wrel must succeed")
    }

    /// `ref_count` of `hash`, `None` when its blob is gone.
    async fn ref_count(&self, hash: u8) -> Option<i64> {
        let raw: Option<String> = self
            .client
            .pool
            .hget(
                self.keys.ext_meta(&sha256_to_hex(&ext_hash(hash))),
                "ref_count",
            )
            .await
            .expect("HGET ref_count must succeed");
        raw.map(|v| v.parse().expect("ref_count is an integer"))
    }

    /// The commit release ring's `PTTL` in ms: `-2` when the key is absent,
    /// `-1` when it has no TTL.
    async fn ring_pttl(&self) -> i64 {
        self.client
            .pool
            .pttl(self.keys.obj_write_releases(&self.oid.to_string()))
            .await
            .expect("PTTL obj:wrel must succeed")
    }

    /// Wait, within [`EXPIRY_BUDGET`], until the commit release ring key is
    /// gone. Polls `EXISTS`, which also expires the key on access.
    async fn await_ring_expiry(&self, what: &str) {
        let key = self.keys.obj_write_releases(&self.oid.to_string());
        let deadline = tokio::time::Instant::now() + EXPIRY_BUDGET;
        loop {
            let present: i64 = self
                .client
                .pool
                .exists(key.as_str())
                .await
                .expect("EXISTS obj:wrel must succeed");
            if present == 0 {
                return;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "{what}: the ring did not expire within {EXPIRY_BUDGET:?} (PTTL {})",
                self.ring_pttl().await
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// One GC pass over everything pending.
    async fn gc(&self) {
        process_pending_gc(self.ext.as_ref(), None, GcPolicy::IMMEDIATE).await;
    }

    /// `live_edit_update_layer_lost_reply_release_lookup_total` as
    /// `[enqueued, absent, failed]`. Panics on an absent cell: all three are
    /// zero-seeded in `Metrics::new`.
    fn lookups(&self) -> [u64; 3] {
        let text = render(&self.metrics);
        ["enqueued", "absent", "failed"].map(|outcome| {
            sample_value(
                &text,
                "live_edit_update_layer_lost_reply_release_lookup_total",
                &[("outcome", outcome)],
            )
            .unwrap_or_else(|| panic!("the lookup {outcome} cell is zero-seeded"))
                as u64
        })
    }

    /// An unchained operation `id` of this layer: its start record is read
    /// when the call starts.
    fn unchained(&self, id: &str) -> Operation {
        Operation::new(id, operation_request_digest(&self.layer, id.as_bytes()))
    }

    /// A chained operation `id` after `previous` (empty: a chain's first
    /// write).
    fn chained(&self, id: &str, previous: &str) -> Operation {
        self.unchained(id).with_predecessor(previous)
    }
}

/// One reference to each of `hashes` held outside the object under test, as
/// another object's would be, so a count GC lowers stays readable.
async fn hold_elsewhere(t: &Target, hashes: &[u8]) {
    let hashes: Vec<[u8; 32]> = hashes.iter().map(|h| ext_hash(*h)).collect();
    pre_store(&t.ext, &hashes).await;
}

/// `token` in its recorded form: 32 lowercase hex digits.
fn hex(token: u128) -> String {
    format!("{token:032x}")
}

/// A queue for `LayerStore::with_commit_tokens_for_test`, one token per call.
fn pinned(tokens: &[u128]) -> Arc<Mutex<VecDeque<u128>>> {
    Arc::new(Mutex::new(tokens.iter().copied().collect()))
}

/// `members`, sorted, as `Target::pending` reads the set.
fn sorted(mut members: Vec<String>) -> Vec<String> {
    members.sort();
    members
}

/// The ring's entries: token → value.
fn entries(ring: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    ring.iter()
        .filter_map(|(field, value)| {
            field
                .strip_prefix("|t|")
                .map(|token| (token.to_owned(), value.clone()))
        })
        .collect()
}

/// Assert every field of `ring` is one of the ring's own shapes — `|t|…`,
/// `|s|…` or `|n|` — so no field is named after a `client_id`.
fn assert_ring_shape(ring: &BTreeMap<String, String>, what: &str) {
    let foreign: Vec<&String> = ring
        .keys()
        .filter(|f| !(f.starts_with("|t|") || f.starts_with("|s|") || *f == "|n|"))
        .collect();
    assert!(
        foreign.is_empty(),
        "{what}: the ring holds only its own field shapes; foreign fields {foreign:?}"
    );
}

/// An entry value as (sequence, sorted release tokens): the routes plan a
/// commit's releases in different orders, and the entry's one reader treats
/// them as a set.
fn as_set(value: &str) -> (u64, Vec<String>) {
    let mut parts = value.split(' ');
    let seq = parts
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("an entry starts with its sequence; got {value:?}"));
    let mut tokens: Vec<String> = parts.map(str::to_owned).collect();
    tokens.sort();
    (seq, tokens)
}

/// The entry a rewrite at `seq` releasing `hash` writes.
fn rewrite_entry(seq: u64, hash: u8) -> String {
    format!("{seq} rewrite:{}", sha256_to_hex(&ext_hash(hash)))
}

/// `n` single-release commits through `ls`, the `i`-th by `client(i)`, each
/// rewriting [`G_FIELD`] from the hash it holds (`holds` before the first)
/// to the other of [`G0`] / [`G1`]. Returns the member each commit's own
/// received reply enqueued, and the hash the field holds after.
async fn churn(
    t: &Target,
    ls: &LayerStore,
    route: Route,
    n: usize,
    mut holds: u8,
    client: impl Fn(usize) -> String,
) -> (Vec<String>, u8) {
    let mut members = Vec::with_capacity(n);
    for i in 0..n {
        let next = if holds == G0 { G1 } else { G0 };
        let seq = t
            .commit(
                ls,
                route,
                &[(G_FIELD, Val::Ext(next))],
                &client(i),
                1_000 + i as u64,
            )
            .await;
        members.push(t.commit_member(holds, seq));
        holds = next;
    }
    (members, holds)
}

/// How the reply of a call's first attempt is lost on a route.
enum Staging {
    /// The post-`EVALSHA` injector: the script commits, its reply is lost.
    PostEval(Arc<Mutex<VecDeque<FredError>>>),
    /// The WATCH path's `EXEC` commits and its verdict is replaced by
    /// in-doubt, and the reconcile read fails too, so the attempt answers a
    /// retriable `Aborted` without knowing.
    Exec(
        live_edit::storage::watched_txn::WatchedExecOverrideQueue,
        Arc<Mutex<VecDeque<FredError>>>,
    ),
}

impl Staging {
    /// One lost reply on `route`.
    fn for_route(route: Route) -> Self {
        let timeout = || {
            Arc::new(Mutex::new(VecDeque::from([FredError::new(
                FredErrorKind::Timeout,
                "staged: reply lost",
            )])))
        };
        match route {
            Route::Fast | Route::Structural => Staging::PostEval(timeout()),
            Route::Watch => Staging::Exec(
                Arc::new(Mutex::new(VecDeque::from([
                    WatchedExecOverride::ReplaceOutcomeAfterExec,
                ]))),
                timeout(),
            ),
        }
    }

    fn install(&self, ls: LayerStore) -> LayerStore {
        match self {
            Staging::PostEval(q) => ls.with_lua_post_eval_fred_error_injector_for_test(q.clone()),
            Staging::Exec(o, r) => ls
                .with_watched_exec_override_for_test(o.clone())
                .with_reconcile_xrange_fred_error_injector_for_test(r.clone()),
        }
    }

    /// Every staged entry was taken: the reply really was lost.
    fn assert_drained(&self) {
        let drained = match self {
            Staging::PostEval(q) => q.lock().unwrap().is_empty(),
            Staging::Exec(o, r) => o.lock().unwrap().is_empty() && r.lock().unwrap().is_empty(),
        };
        assert!(
            drained,
            "the staged lost reply must have been taken, or the scenario describes an ordinary \
             commit"
        );
    }
}

/// An attempt fault queue (`LayerStore::with_fault_injector_for_test`),
/// empty: a hook fills it before the attempts it fails.
type Faults = Arc<Mutex<VecDeque<ServiceError>>>;

/// Push `n` retriable failures onto `faults`.
fn fail_next(faults: &Faults, n: usize) {
    let mut q = faults.lock().unwrap();
    for _ in 0..n {
        q.push_back(ServiceError::Aborted(
            "staged: the attempt failed transiently".to_owned(),
        ));
    }
}

/// Assert `result` is the plain `UNAVAILABLE`, outcome unknown, of a
/// writer-guard refusal at `attempt`.
fn assert_outcome_unknown(result: &Result<UpdateCommit, ServiceError>, attempt: u32, what: &str) {
    match result {
        Err(ServiceError::Unavailable(message))
            if message.starts_with("UpdateLayer outcome unknown:")
                && message.contains(&format!("attempt {attempt}")) => {}
        other => panic!(
            "{what}: a writer-guard refusal after a counted retry is a plain UNAVAILABLE, outcome \
             unknown, naming attempt {attempt}; got {other:?}"
        ),
    }
}

/// Assert `result` is the overtaken refusal.
fn assert_overtaken(result: &Result<UpdateCommit, ServiceError>, what: &str) {
    match result {
        Err(ServiceError::Unavailable(message)) if message == OPERATION_OVERTAKEN_MESSAGE => {}
        other => panic!("{what}: expected the overtaken UNAVAILABLE; got {other:?}"),
    }
}

/// Assert `result` is the first attempt's `ABORTED`
/// `PREDECESSOR_NOT_COMMITTED`.
fn assert_predecessor_not_committed(result: &Result<UpdateCommit, ServiceError>, what: &str) {
    match result {
        Err(ServiceError::PredecessorNotCommitted { .. }) => {}
        other => panic!("{what}: expected ABORTED PREDECESSOR_NOT_COMMITTED; got {other:?}"),
    }
}

/// Assert GC dropped each of `hashes` by exactly one from `before`.
async fn assert_each_dropped_by_one(t: &Target, hashes: &[u8], before: &[Option<i64>], what: &str) {
    let mut after = Vec::new();
    for hash in hashes {
        after.push(t.ref_count(*hash).await);
    }
    let expected: Vec<Option<i64>> = before.iter().map(|c| c.map(|c| c - 1)).collect();
    assert_eq!(
        after, expected,
        "{what}: each released hash drops by exactly the one release its commit owed (before \
         GC: {before:?})"
    );
}

/// The `ref_count`s of `hashes`, in order.
async fn ref_counts(t: &Target, hashes: &[u8]) -> Vec<Option<i64>> {
    let mut counts = Vec::new();
    for hash in hashes {
        counts.push(t.ref_count(*hash).await);
    }
    counts
}

/// Three `#[tokio::test]`s running `$scenario(route)`, one per route.
macro_rules! on_every_route {
    ($scenario:ident: $fast:ident, $structural:ident, $watch:ident) => {
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn $fast() {
            $scenario(Route::Fast).await;
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn $structural() {
            $scenario(Route::Structural).await;
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn $watch() {
            $scenario(Route::Watch).await;
        }
    };
}

// ═════════════════════════════════════════════════════════════════════════════
// (i) The lost-reply lookup and the ring
// ═════════════════════════════════════════════════════════════════════════════

/// T-i1. X, an unchained operation call, rewrites [`FIELD`] from [`A`] to
/// [`B`] at `seq` 3, owing the release of hash A, and loses its reply. Before
/// X's second attempt another commit of the same `client_id`, Y, rewrites
/// the field from B to [`C`] at `seq` 4 through another store. X's second
/// attempt finds its writer's record moved off its start record and is
/// overtaken — after a counted retry, so the call ends without proof of its
/// outcome and looks its own entry up:
///
/// - X is still answered the overtaken `UNAVAILABLE`, after two attempts;
/// - pending is exactly X's `Commit(3)` of hash A and Y's `Commit(4)` of
///   hash B: the lookup found X's entry (`enqueued` 1), and nothing of X's
///   pre-flight increment was handed back;
/// - X's entry still lists its release;
/// - GC drops hashes A and B by exactly one each.
async fn overtaken_lost_reply_recovers_its_releases(route: Route) {
    let fx = fx(&format!("pgs-i1-{route:?}")).await;
    let t = &fx.t;
    hold_elsewhere(t, &[A, B]).await;
    t.commit(&t.store(route), route, &[(FIELD, Val::Ext(A))], WRITER, 1)
        .await;

    let staging = Staging::for_route(route);
    let y_seq = Arc::new(Mutex::new(None));
    let hook: BetweenAttemptsHook = {
        let (t, y_store, y_seq) = (t.clone(), Arc::new(t.store(route)), y_seq.clone());
        Arc::new(move |attempt| {
            let (t, y_store, y_seq) = (t.clone(), y_store.clone(), y_seq.clone());
            Box::pin(async move {
                if attempt != 2 {
                    return;
                }
                let seq = t
                    .commit(&y_store, route, &[(FIELD, Val::Ext(C))], WRITER, 3)
                    .await;
                *y_seq.lock().unwrap() = Some(seq);
            })
        })
    };
    let x_store = staging
        .install(t.store(route))
        .with_commit_tokens_for_test(pinned(&[TOKEN_X]))
        .with_between_attempts_hook_for_test(hook);

    let (result, summary) = t
        .call(
            &x_store,
            route,
            &[(FIELD, Val::Ext(B))],
            WRITER,
            2,
            Some(t.unchained("pgs-i1-x")),
        )
        .await;

    staging.assert_drained();
    assert_eq!(
        *y_seq.lock().unwrap(),
        Some(4),
        "{route:?}: precondition: X landed at 3 and Y after it"
    );
    assert_overtaken(&result, &format!("{route:?}"));
    assert_eq!(summary.map(|s| s.attempts), Some(2), "{route:?}");
    assert_eq!(
        t.pending().await,
        sorted(vec![t.commit_member(A, 3), t.commit_member(B, 4)]),
        "{route:?}: X's own release, recovered by its lookup, beside Y's; no hand-back"
    );
    assert_eq!(
        t.lookups(),
        [1, 0, 0],
        "{route:?}: one lookup, which found X's entry"
    );
    assert_eq!(
        entries(&t.ring().await).get(&hex(TOKEN_X)),
        Some(&rewrite_entry(3, A)),
        "{route:?}: X's entry"
    );

    let before = ref_counts(t, &[A, B]).await;
    t.gc().await;
    assert_each_dropped_by_one(t, &[A, B], &before, &format!("{route:?}")).await;
}

on_every_route!(overtaken_lost_reply_recovers_its_releases:
    fast_path_overtaken_lost_reply_recovers_its_releases,
    structural_route_overtaken_lost_reply_recovers_its_releases,
    watch_path_overtaken_lost_reply_recovers_its_releases);

/// T-i2, the narrow window. A rewrites [`FIELD`] from [`A`] to [`B`] at
/// `seq` 3 and loses its reply; its second attempt is answered from its own
/// commit. Between that verdict and the call's read of its own entry (the
/// release-record read hook, called with [`ReleaseRead::OwnCall`]), S — a
/// commit of the SAME `client_id`, through a separate store, so neither
/// waits on the other's per-object admission — rewrites the field from B to
/// [`C`] at `seq` 4, owing the release of hash B. S's ring write takes the
/// next slot, so A's read still finds its own entry: both A's `Commit(3)`
/// and S's `Commit(4)` are pending, once each, and GC drops each hash by
/// one. A per-`client_id` record would have been replaced by S's write,
/// and A's read would have found S's sequence and enqueued nothing.
async fn successor_committing_before_the_own_call_read_leaves_the_entry_to_it(route: Route) {
    let fx = fx(&format!("pgs-i2-{route:?}")).await;
    let t = &fx.t;
    hold_elsewhere(t, &[A, B]).await;
    t.commit(&t.store(route), route, &[(FIELD, Val::Ext(A))], WRITER, 1)
        .await;

    let staging = Staging::for_route(route);
    let reads = Arc::new(Mutex::new(Vec::new()));
    let s_seq = Arc::new(Mutex::new(None));
    let hook: ReleaseRecordReadHook = {
        let (t, s_store, reads, s_seq) = (
            t.clone(),
            Arc::new(t.store(route)),
            reads.clone(),
            s_seq.clone(),
        );
        Arc::new(move |read| {
            let (t, s_store, reads, s_seq) =
                (t.clone(), s_store.clone(), reads.clone(), s_seq.clone());
            Box::pin(async move {
                reads.lock().unwrap().push(read);
                if read != ReleaseRead::OwnCall || s_seq.lock().unwrap().is_some() {
                    return;
                }
                let seq = t
                    .commit(&s_store, route, &[(FIELD, Val::Ext(C))], WRITER, 3)
                    .await;
                *s_seq.lock().unwrap() = Some(seq);
            })
        })
    };
    let a_store = staging
        .install(t.store(route))
        .with_commit_tokens_for_test(pinned(&[TOKEN_A]))
        .with_release_record_read_hook_for_test(hook);

    let (result, _) = t
        .call(&a_store, route, &[(FIELD, Val::Ext(B))], WRITER, 2, None)
        .await;

    staging.assert_drained();
    assert_eq!(
        result
            .unwrap_or_else(|e| panic!("{route:?}: A landed, so it is answered; got {e:?}"))
            .seq,
        3
    );
    assert_eq!(
        *reads.lock().unwrap(),
        vec![ReleaseRead::OwnCall],
        "{route:?}: A read its own entry once, on its own-call verdict"
    );
    assert_eq!(
        *s_seq.lock().unwrap(),
        Some(4),
        "{route:?}: S committed in the window"
    );
    assert_eq!(
        t.pending().await,
        sorted(vec![t.commit_member(A, 3), t.commit_member(B, 4)]),
        "{route:?}: A's release, read after S committed, and S's, once each"
    );
    assert_eq!(
        t.lookups(),
        [0, 0, 0],
        "{route:?}: an own-call read is not the lost-reply lookup"
    );

    let before = ref_counts(t, &[A, B]).await;
    t.gc().await;
    assert_each_dropped_by_one(t, &[A, B], &before, &format!("{route:?}")).await;
}

on_every_route!(successor_committing_before_the_own_call_read_leaves_the_entry_to_it:
    fast_path_successor_committing_before_the_own_call_read_leaves_the_entry_to_it,
    structural_route_successor_committing_before_the_own_call_read_leaves_the_entry_to_it,
    watch_path_successor_committing_before_the_own_call_read_leaves_the_entry_to_it);

/// T-i3. X, an unchained operation call, rewrites [`FIELD`] from [`A`] to
/// [`B`] at `seq` 3 — the object's first releasing commit, so its entry
/// starts at slot 0 — and loses its reply. Before X's second attempt,
/// RING + 1 single-release commits of the same `client_id` land: the RING-th
/// wraps the ring onto slot 0 and evicts X's entry. X's second attempt is
/// overtaken and its lookup finds nothing (`absent` 1): X's `Commit(3)` is
/// never enqueued — the leak an eviction costs, never a release — while
/// each churn commit's own member is pending once, and exactly RING entries
/// remain.
async fn evicted_entry_is_absent_to_the_lost_reply_lookup(route: Route) {
    let fx = fx(&format!("pgs-i3-{route:?}")).await;
    let t = &fx.t;
    t.commit(
        &t.store(route),
        route,
        &[(FIELD, Val::Ext(A)), (G_FIELD, Val::Ext(G0))],
        WRITER,
        1,
    )
    .await;

    let staging = Staging::for_route(route);
    let churned = Arc::new(Mutex::new(None));
    let hook: BetweenAttemptsHook = {
        let (t, c_store, churned) = (t.clone(), Arc::new(t.store(route)), churned.clone());
        Arc::new(move |attempt| {
            let (t, c_store, churned) = (t.clone(), c_store.clone(), churned.clone());
            Box::pin(async move {
                if attempt != 2 {
                    return;
                }
                let (members, _) =
                    churn(&t, &c_store, route, RING + 1, G0, |_| WRITER.to_owned()).await;
                *churned.lock().unwrap() = Some(members);
            })
        })
    };
    let x_store = staging
        .install(t.store(route))
        .with_commit_tokens_for_test(pinned(&[TOKEN_X]))
        .with_between_attempts_hook_for_test(hook);

    let (result, summary) = t
        .call(
            &x_store,
            route,
            &[(FIELD, Val::Ext(B))],
            WRITER,
            2,
            Some(t.unchained("pgs-i3-x")),
        )
        .await;

    staging.assert_drained();
    assert_overtaken(&result, &format!("{route:?}"));
    assert_eq!(summary.map(|s| s.attempts), Some(2), "{route:?}");
    let churned = churned
        .lock()
        .unwrap()
        .take()
        .expect("the churn ran between X's attempts");
    assert_eq!(
        t.pending().await,
        sorted(churned),
        "{route:?}: every churn commit's own release once, and X's {} not at all",
        t.commit_member(A, 3)
    );
    assert_eq!(
        t.lookups(),
        [0, 1, 0],
        "{route:?}: X's lookup found its entry evicted"
    );
    let ring = t.ring().await;
    assert_ring_shape(&ring, &format!("{route:?}"));
    let entries = entries(&ring);
    assert_eq!(
        entries.len(),
        RING,
        "{route:?}: exactly RING entries remain"
    );
    assert!(
        !entries.contains_key(&hex(TOKEN_X)),
        "{route:?}: X's entry was evicted"
    );
}

on_every_route!(evicted_entry_is_absent_to_the_lost_reply_lookup:
    fast_path_evicted_entry_is_absent_to_the_lost_reply_lookup,
    structural_route_evicted_entry_is_absent_to_the_lost_reply_lookup,
    watch_path_evicted_entry_is_absent_to_the_lost_reply_lookup);

/// The retry budget of T-i4's and T-i6's exhausting calls.
const EXHAUSTING_ATTEMPTS: u32 = 3;

/// T-i4. A plain call rewrites [`FIELD`] from [`A`] to [`B`] at `seq` 3 and
/// loses its reply; its between-attempts hook then fails every further
/// attempt transiently (the attempt fault queue), so its retries run out.
/// The call ends `ABORTED` with the retry budget's message — unchanged — and,
/// because any attempt may have committed, its lookup finds its entry and
/// enqueues its `Commit(3)` (`enqueued` 1); GC drops hash A by one.
async fn exhausted_call_recovers_its_landed_commits_releases(route: Route) {
    let fx = fx(&format!("pgs-i4-{route:?}")).await;
    let t = &fx.t;
    hold_elsewhere(t, &[A]).await;
    t.commit(&t.store(route), route, &[(FIELD, Val::Ext(A))], WRITER, 1)
        .await;

    let staging = Staging::for_route(route);
    let faults: Faults = Arc::default();
    let hook: BetweenAttemptsHook = {
        let faults = faults.clone();
        Arc::new(move |attempt| {
            if attempt == 2 {
                fail_next(&faults, (EXHAUSTING_ATTEMPTS - 1) as usize);
            }
            Box::pin(async {})
        })
    };
    let ls = staging
        .install(
            t.store(route)
                .with_update_layer_retry_max_attempts(EXHAUSTING_ATTEMPTS),
        )
        .with_fault_injector_for_test(faults.clone())
        .with_commit_tokens_for_test(pinned(&[TOKEN_A]))
        .with_between_attempts_hook_for_test(hook);

    let (result, summary) = t
        .call(&ls, route, &[(FIELD, Val::Ext(B))], WRITER, 2, None)
        .await;

    staging.assert_drained();
    assert!(
        faults.lock().unwrap().is_empty(),
        "{route:?}: attempts 2 and 3 each took their staged failure"
    );
    assert!(
        matches!(result, Err(ServiceError::Aborted(_))),
        "{route:?}: the call ends ABORTED, its retries spent; got {result:?}"
    );
    assert_eq!(
        summary.map(|s| s.attempts),
        Some(EXHAUSTING_ATTEMPTS),
        "{route:?}"
    );
    assert_eq!(
        t.pending().await,
        vec![t.commit_member(A, 3)],
        "{route:?}: the landed commit's release, recovered by the exhaustion's lookup"
    );
    assert_eq!(t.lookups(), [1, 0, 0], "{route:?}");

    let before = ref_counts(t, &[A]).await;
    t.gc().await;
    assert_each_dropped_by_one(t, &[A], &before, &format!("{route:?}")).await;
}

on_every_route!(exhausted_call_recovers_its_landed_commits_releases:
    fast_path_exhausted_call_recovers_its_landed_commits_releases,
    structural_route_exhausted_call_recovers_its_landed_commits_releases,
    watch_path_exhausted_call_recovers_its_landed_commits_releases);

/// The pinned tokens of T-i5's script, in call order: the seed (which owes
/// nothing, so writes no entry), the 9-release commit, the token two
/// `client_id`s each commit under, then RING single-release commits.
fn parity_tokens() -> Vec<u128> {
    let mut tokens = vec![0x5EED, 0x1D3, 0x5A, 0x5A];
    tokens.extend((0..RING as u128).map(|i| 0x1000 + i));
    tokens
}

/// The ring with each entry as (sequence, token set), comparable across
/// routes; every other field verbatim.
fn comparable(ring: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    ring.iter()
        .map(|(field, value)| {
            let value = if field.starts_with("|t|") {
                format!("{:?}", as_set(value))
            } else {
                value.clone()
            };
            (field.clone(), value)
        })
        .collect()
}

/// T-i5's script on `route`; returns the ring after the shared token and at
/// the end, each [`comparable`].
async fn parity_script(route: Route) -> (BTreeMap<String, String>, BTreeMap<String, String>) {
    let fx = fx(&format!("pgs-i5-{route:?}")).await;
    let t = &fx.t;
    let ls = t
        .store(route)
        .with_commit_tokens_for_test(pinned(&parity_tokens()));

    let mut seed: Vec<(String, Val)> = (0..9)
        .map(|i| (wide_field(i), Val::Ext(wide_hash(i))))
        .collect();
    seed.push((G_FIELD.to_owned(), Val::Ext(G0)));
    let seed: Vec<(&str, Val)> = seed.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    assert_eq!(t.commit(&ls, route, &seed, WRITER, 1).await, 2);

    let names: Vec<String> = (0..9).map(wide_field).collect();
    let wide: Vec<(&str, Val)> = names
        .iter()
        .enumerate()
        .map(|(i, k)| (k.as_str(), Val::Inline(i as u8)))
        .collect();
    assert_eq!(t.commit(&ls, route, &wide, WRITER, 2).await, 3);
    assert_eq!(
        t.commit(&ls, route, &[(G_FIELD, Val::Ext(G1))], "pgs-i5-x", 3)
            .await,
        4
    );
    assert_eq!(
        t.commit(&ls, route, &[(G_FIELD, Val::Ext(G0))], "pgs-i5-y", 4)
            .await,
        5
    );
    let after_shared = t.ring().await;
    assert_ring_shape(&after_shared, &format!("{route:?}"));

    churn(t, &ls, route, RING, G0, |_| WRITER.to_owned()).await;
    let at_end = t.ring().await;
    assert_ring_shape(&at_end, &format!("{route:?}"));
    // Every route's ring write sets the key's TTL (b.5wb), the store's
    // default here.
    let ttl = t.ring_pttl().await;
    assert!(
        0 < ttl && ttl <= DEFAULT_RING_TTL_MS,
        "{route:?}: the last ring write left the key a TTL; PTTL {ttl}"
    );
    (comparable(&after_shared), comparable(&at_end))
}

/// T-i5, parity. One scripted sequence of commits with pinned tokens runs on
/// the fast path, the structural route and the WATCH path: a seed owing
/// nothing; a commit owing 9 releases (a two-slot entry: its start slot and
/// one covered `""` slot); a commit of one `client_id` and then one of
/// another under the SAME token (its entry is rewritten in place, and its
/// first start slot is not evicted by its own write); then RING
/// single-release commits, which wrap the ring past all of them. After the
/// shared token and at the end, the three rings have the same field set,
/// byte-identical `|s|` and `|n|` values, and the same entries as
/// (sequence, token set) — the routes plan a commit's releases in different
/// orders.
///
/// Also pinned on the shared script, so the comparison is over the rule
/// rather than over three identical mistakes: after the shared token the
/// ring is exactly the 9-release entry at slots 0–1, the shared token's
/// entry (its second commit's) at slots 2 and 3, and `|n|` `4`; at the end
/// it holds exactly RING single-release entries, 2·RING + 1 fields, and
/// `|n|` `4` again. On every route the last ring write leaves the key a TTL
/// within the store's default (b.5wb).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_release_ring_writes_are_identical_on_every_route() {
    let mut runs = Vec::new();
    for route in [Route::Fast, Route::Structural, Route::Watch] {
        runs.push((route, parity_script(route).await));
    }

    let (wide, shared) = (hex(0x1D3), hex(0x5A));
    let mut wide_tokens: Vec<String> = (0..9)
        .map(|i| format!("rewrite:{}", sha256_to_hex(&ext_hash(wide_hash(i)))))
        .collect();
    wide_tokens.sort();
    let expected_after_shared: BTreeMap<String, String> = BTreeMap::from([
        (format!("|t|{wide}"), format!("{:?}", (3u64, wide_tokens))),
        (
            format!("|t|{shared}"),
            format!(
                "{:?}",
                (
                    5u64,
                    vec![format!("rewrite:{}", sha256_to_hex(&ext_hash(G1)))]
                )
            ),
        ),
        ("|s|0".to_owned(), wide.clone()),
        ("|s|1".to_owned(), String::new()),
        ("|s|2".to_owned(), shared.clone()),
        ("|s|3".to_owned(), shared.clone()),
        ("|n|".to_owned(), "4".to_owned()),
    ]);
    let (_, (fast_shared, fast_end)) = &runs[0];
    assert_eq!(
        fast_shared, &expected_after_shared,
        "Fast: the ring after the shared token"
    );
    assert_eq!(fast_end.len(), 2 * RING + 1, "Fast: a full ring at the end");
    assert_eq!(
        fast_end.keys().filter(|f| f.starts_with("|t|")).count(),
        RING,
        "Fast: RING single-release entries"
    );
    assert_eq!(fast_end.get("|n|").map(String::as_str), Some("4"));
    for gone in [&wide, &shared] {
        assert!(
            !fast_end.contains_key(&format!("|t|{gone}")),
            "Fast: the wrap evicted {gone}"
        );
    }

    for (route, (after_shared, at_end)) in &runs[1..] {
        assert_eq!(
            after_shared, fast_shared,
            "{route:?}: the ring after the shared token equals the fast path's"
        );
        assert_eq!(
            at_end, fast_end,
            "{route:?}: the ring at the end equals the fast path's"
        );
    }
}

/// T-i6. A field an older build would write — `obj:wrel[<client_id>]`, its
/// value naming `seq` 3 and hash [`A`] — is planted before:
///
/// - L1, a commit owing nothing at `seq` 3 whose reply is lost, answered
///   from its own commit (an own-call read);
/// - L2, a commit owing nothing at `seq` 4 whose reply is lost and whose
///   retries run out (a lost-reply lookup, `absent`);
/// - R, a releasing commit at `seq` 5 rewriting [`FIELD`] from A to [`B`].
///
/// The field is byte-unchanged after all three, and the only pending member
/// is R's own `Commit(5)`: an older build's reader, keyed by `client_id`
/// with a sequence check, would have enqueued `Commit(3)` of hash A for L1
/// from the planted field.
///
/// The ring's TTL reaches the legacy field too (b.5wb): R's write gives the
/// key a TTL, and after a later releasing commit through a store with a
/// short TTL the key — legacy field included — expires.
async fn legacy_per_client_field_is_never_read_or_written(route: Route) {
    let fx = fx(&format!("pgs-i6-{route:?}")).await;
    let t = &fx.t;
    t.commit(&t.store(route), route, &[(FIELD, Val::Ext(A))], WRITER, 1)
        .await;
    let wrel_key = t.keys.obj_write_releases(&t.oid.to_string());
    let legacy = rewrite_entry(3, A);
    let _: () = t
        .client
        .pool
        .hset(&wrel_key, [(WRITER, legacy.as_str())])
        .await
        .expect("plant the legacy field");

    let l1 = Staging::for_route(route);
    let (result, _) = t
        .call(
            &l1.install(t.store(route)),
            route,
            &[(OTHER, Val::Inline(1))],
            WRITER,
            2,
            None,
        )
        .await;
    l1.assert_drained();
    assert_eq!(result.expect("L1 landed").seq, 3);

    let l2 = Staging::for_route(route);
    let faults: Faults = Arc::default();
    let hook: BetweenAttemptsHook = {
        let faults = faults.clone();
        Arc::new(move |attempt| {
            if attempt == 2 {
                fail_next(&faults, (EXHAUSTING_ATTEMPTS - 1) as usize);
            }
            Box::pin(async {})
        })
    };
    let (result, _) = t
        .call(
            &l2.install(
                t.store(route)
                    .with_update_layer_retry_max_attempts(EXHAUSTING_ATTEMPTS),
            )
            .with_fault_injector_for_test(faults)
            .with_between_attempts_hook_for_test(hook),
            route,
            &[(OTHER, Val::Inline(2))],
            WRITER,
            3,
            None,
        )
        .await;
    l2.assert_drained();
    assert!(
        matches!(result, Err(ServiceError::Aborted(_))),
        "{route:?}: L2's retries run out; got {result:?}"
    );

    let r = t
        .commit(&t.store(route), route, &[(FIELD, Val::Ext(B))], WRITER, 4)
        .await;
    assert_eq!(r, 5);

    let after: Option<String> = t
        .client
        .pool
        .hget(&wrel_key, WRITER)
        .await
        .expect("HGET the legacy field");
    assert_eq!(
        after,
        Some(legacy),
        "{route:?}: the legacy field is byte-unchanged"
    );
    assert_eq!(
        t.pending().await,
        vec![t.commit_member(A, 5)],
        "{route:?}: R's own release alone; nothing derives from the legacy field"
    );
    assert_eq!(
        t.lookups(),
        [0, 1, 0],
        "{route:?}: L2's lookup found no entry of its own"
    );
    let ring = t.ring().await;
    assert_eq!(
        ring.len(),
        4,
        "{route:?}: the legacy field beside R's entry, its slot and |n|; got {ring:?}"
    );

    // The ring's TTL (b.5wb). The legacy field was planted into a key the
    // object's first commit never wrote (it owed nothing), and L1 and L2 owe
    // nothing either: neither sets a TTL. R's releasing commit sets one on
    // the whole key, the legacy field's bytes included (checked unchanged
    // above).
    let ttl = t.ring_pttl().await;
    assert!(
        0 < ttl && ttl <= DEFAULT_RING_TTL_MS,
        "{route:?}: R's ring write set the key's TTL; PTTL {ttl}"
    );
    // A later releasing commit through a store with a short TTL sets the
    // key's TTL again (a plain `PEXPIRE`, not `GT`), and the key — the legacy
    // field with it — is gone once that TTL passes.
    t.commit(
        &t.store(route)
            .with_commit_release_ring_ttl(SHORT_RING_TTL_MS),
        route,
        &[(FIELD, Val::Ext(C))],
        WRITER,
        5,
    )
    .await;
    t.await_ring_expiry(&format!("{route:?}")).await;
    let after: Option<String> = t
        .client
        .pool
        .hget(&wrel_key, WRITER)
        .await
        .expect("HGET the legacy field");
    assert_eq!(
        after, None,
        "{route:?}: the legacy field expired with the ring"
    );
}

on_every_route!(legacy_per_client_field_is_never_read_or_written:
    fast_path_legacy_per_client_field_is_never_read_or_written,
    structural_route_legacy_per_client_field_is_never_read_or_written,
    watch_path_legacy_per_client_field_is_never_read_or_written);

// ═════════════════════════════════════════════════════════════════════════════
// The ring's TTL (b.5wb)
// ═════════════════════════════════════════════════════════════════════════════

/// Set the ring's TTL to `ms` out of band, as if time had passed since its
/// last releasing commit.
async fn age_ring(t: &Target, ms: i64) {
    let set: bool = t
        .client
        .pool
        .pexpire(t.keys.obj_write_releases(&t.oid.to_string()), ms, None)
        .await
        .expect("PEXPIRE obj:wrel must succeed");
    assert!(set, "fixture: the ring exists to be aged");
}

/// The ring TTL of T-i16's store, in ms: one hour, so no runner stall lets
/// the key expire, or a TTL fall to [`AGED_RING_TTL_MS`], within the test.
const LONG_RING_TTL_MS: u64 = 3_600_000;

/// The TTL T-i16 ages its ring to before a commit whose effect on it is
/// checked: half of [`LONG_RING_TTL_MS`], so a TTL read above it is the
/// store's, set by a releasing commit, and an aged TTL no commit refreshed
/// reads at or below it.
const AGED_RING_TTL_MS: i64 = 1_800_000;

/// T-i16. The ring lives while the layer is being edited. Through a store
/// whose ring TTL is [`LONG_RING_TTL_MS`]:
///
/// - a commit owing no release (the object's first, setting [`FIELD`] to
///   [`A`]) creates no ring key;
/// - a releasing commit (A → [`B`]) writes the ring and leaves it the
///   store's TTL, in ([`AGED_RING_TTL_MS`], [`LONG_RING_TTL_MS`]];
/// - with the TTL aged to [`AGED_RING_TTL_MS`], a commit owing no release
///   (an inline [`OTHER`]) leaves it in (0, that]: it neither writes nor
///   refreshes the ring;
/// - a second releasing commit (B → [`C`]) raises it above that again.
///
/// Both bounds sit half an hour from any TTL the test reads, so no assertion
/// depends on how fast the runner is. The default TTL is checked by T-i5 and
/// T-i6.
///
/// Fail-on-old: before b.5wb the ring had no TTL, `PTTL` `-1`.
async fn releasing_commits_set_and_refresh_the_ring_ttl(route: Route) {
    let fx = fx(&format!("pgs-i16-{route:?}")).await;
    let t = &fx.t;
    let ls = t
        .store(route)
        .with_commit_release_ring_ttl(LONG_RING_TTL_MS);
    let long = LONG_RING_TTL_MS as i64;

    t.commit(&ls, route, &[(FIELD, Val::Ext(A))], WRITER, 1)
        .await;
    assert_eq!(
        t.ring_pttl().await,
        -2,
        "{route:?}: a commit owing no release creates no ring"
    );

    t.commit(&ls, route, &[(FIELD, Val::Ext(B))], WRITER, 2)
        .await;
    let ttl = t.ring_pttl().await;
    assert!(
        AGED_RING_TTL_MS < ttl && ttl <= long,
        "{route:?}: a releasing commit leaves the ring the store's TTL, in \
         ({AGED_RING_TTL_MS}, {long}]; PTTL {ttl}"
    );

    age_ring(t, AGED_RING_TTL_MS).await;
    t.commit(&ls, route, &[(OTHER, Val::Inline(1))], WRITER, 3)
        .await;
    let ttl = t.ring_pttl().await;
    assert!(
        0 < ttl && ttl <= AGED_RING_TTL_MS,
        "{route:?}: a commit owing no release leaves the aged TTL alone; PTTL {ttl}"
    );

    t.commit(&ls, route, &[(FIELD, Val::Ext(C))], WRITER, 4)
        .await;
    let ttl = t.ring_pttl().await;
    assert!(
        AGED_RING_TTL_MS < ttl && ttl <= long,
        "{route:?}: a second releasing commit refreshes the TTL to the store's, in \
         ({AGED_RING_TTL_MS}, {long}]; PTTL {ttl}"
    );
}

on_every_route!(releasing_commits_set_and_refresh_the_ring_ttl:
    fast_path_releasing_commits_set_and_refresh_the_ring_ttl,
    structural_route_releasing_commits_set_and_refresh_the_ring_ttl,
    watch_path_releasing_commits_set_and_refresh_the_ring_ttl);

/// T-i17. An expired ring is absent to the lost-reply lookup. Through a store
/// with the default ring TTL, a plain call rewrites [`FIELD`] from [`A`] to
/// [`B`] and loses its reply. Before its second attempt the between-attempts
/// hook checks the premise — the lost commit wrote its ring entry with a TTL
/// in (0, [`DEFAULT_RING_TTL_MS`]] — then ages that TTL to 1 ms
/// ([`age_ring`]) and waits until the store's own clock has expired the key.
/// Every further attempt fails transiently, so its retries run out. Its
/// lookup finds no entry — `absent` 1, `enqueued` 0 — and enqueues nothing:
/// the release the landed commit owed leaks to the reference-count
/// reconciler, never releases twice. The store-configured short TTL is
/// exercised by T-i6 and T-i18; aging it here keeps the premise free of any
/// timing assumption.
async fn expired_ring_is_absent_to_the_lost_reply_lookup(route: Route) {
    let fx = fx(&format!("pgs-i17-{route:?}")).await;
    let t = &fx.t;
    hold_elsewhere(t, &[A]).await;
    t.commit(&t.store(route), route, &[(FIELD, Val::Ext(A))], WRITER, 1)
        .await;

    let staging = Staging::for_route(route);
    let faults: Faults = Arc::default();
    let expired = Arc::new(AtomicBool::new(false));
    let hook: BetweenAttemptsHook = {
        let (t, faults, expired) = (t.clone(), faults.clone(), expired.clone());
        Arc::new(move |attempt| {
            let (t, faults, expired) = (t.clone(), faults.clone(), expired.clone());
            Box::pin(async move {
                if attempt != 2 {
                    return;
                }
                let ttl = t.ring_pttl().await;
                assert!(
                    0 < ttl && ttl <= DEFAULT_RING_TTL_MS,
                    "{route:?}: precondition: the lost commit wrote its ring entry with a TTL \
                     in (0, {DEFAULT_RING_TTL_MS}]; PTTL {ttl}"
                );
                age_ring(&t, 1).await;
                t.await_ring_expiry(&format!("{route:?}")).await;
                expired.store(true, SeqCst);
                fail_next(&faults, (EXHAUSTING_ATTEMPTS - 1) as usize);
            })
        })
    };
    let ls = staging
        .install(
            t.store(route)
                .with_update_layer_retry_max_attempts(EXHAUSTING_ATTEMPTS),
        )
        .with_fault_injector_for_test(faults.clone())
        .with_commit_tokens_for_test(pinned(&[TOKEN_A]))
        .with_between_attempts_hook_for_test(hook);

    let (result, _) = t
        .call(&ls, route, &[(FIELD, Val::Ext(B))], WRITER, 2, None)
        .await;

    staging.assert_drained();
    assert!(
        expired.load(SeqCst) && faults.lock().unwrap().is_empty(),
        "{route:?}: precondition: the ring expired before attempt 2, and attempts 2 and 3 \
         each took their staged failure"
    );
    assert!(
        matches!(result, Err(ServiceError::Aborted(_))),
        "{route:?}: the call ends ABORTED, its retries spent; got {result:?}"
    );
    assert_eq!(
        t.lookups(),
        [0, 1, 0],
        "{route:?}: the lookup found the expired ring absent"
    );
    assert_eq!(
        t.pending().await,
        Vec::<String>::new(),
        "{route:?}: nothing enqueued: the release leaks to the reconciler"
    );
}

on_every_route!(expired_ring_is_absent_to_the_lost_reply_lookup:
    fast_path_expired_ring_is_absent_to_the_lost_reply_lookup,
    structural_route_expired_ring_is_absent_to_the_lost_reply_lookup,
    watch_path_expired_ring_is_absent_to_the_lost_reply_lookup);

/// T-i18. A store given a ring TTL outside the script's accepted range
/// clamps it (`LayerStore::with_commit_release_ring_ttl`): `0` becomes 1 ms
/// and `u64::MAX` the twelve-digit maximum, so the update script's
/// `ENCODING` check on `ARGV[16]` never refuses a commit for it, and the WATCH
/// path's `PEXPIRE` never takes a value past it. A releasing commit through
/// each such store lands on every route; with the maximum it leaves the key
/// a TTL above the default.
async fn out_of_range_ring_ttls_are_clamped_and_commit(route: Route) {
    let fx = fx(&format!("pgs-i18-{route:?}")).await;
    let t = &fx.t;
    t.commit(&t.store(route), route, &[(FIELD, Val::Ext(A))], WRITER, 1)
        .await;

    t.commit(
        &t.store(route).with_commit_release_ring_ttl(u64::MAX),
        route,
        &[(FIELD, Val::Ext(B))],
        WRITER,
        2,
    )
    .await;
    let ttl = t.ring_pttl().await;
    assert!(
        DEFAULT_RING_TTL_MS < ttl && ttl <= 999_999_999_999,
        "{route:?}: u64::MAX is clamped to the twelve-digit maximum; PTTL {ttl}"
    );

    t.commit(
        &t.store(route).with_commit_release_ring_ttl(0),
        route,
        &[(FIELD, Val::Ext(C))],
        WRITER,
        3,
    )
    .await;
    t.await_ring_expiry(&format!("{route:?}: a TTL of 0 is clamped to 1 ms"))
        .await;
}

on_every_route!(out_of_range_ring_ttls_are_clamped_and_commit:
    fast_path_out_of_range_ring_ttls_are_clamped_and_commit,
    structural_route_out_of_range_ring_ttls_are_clamped_and_commit,
    watch_path_out_of_range_ring_ttls_are_clamped_and_commit);

/// `storage::keys::commit_release_ring_ttl`: the RPC deadline + the Valkey
/// command timeout + 60 s, so 95,000 ms at the defaults (30 s and 5 s), a
/// configured deadline lengthening it one for one; never `0` (a `PEXPIRE 0`
/// would delete the ring the commit just wrote), and clamped to the twelve
/// digits the update script accepts on `ARGV[16]` however large its inputs.
#[test]
fn commit_release_ring_ttl_is_the_deadline_plus_the_command_timeout_plus_a_minute_clamped() {
    use live_edit::config::{RPC_DEADLINE_MS_DEFAULT, VALKEY_COMMAND_TIMEOUT_MS_DEFAULT};
    use live_edit::storage::keys::commit_release_ring_ttl;

    assert_eq!(
        commit_release_ring_ttl(RPC_DEADLINE_MS_DEFAULT, VALKEY_COMMAND_TIMEOUT_MS_DEFAULT),
        DEFAULT_RING_TTL_MS as u64,
        "95 s at the defaults"
    );
    assert_eq!(commit_release_ring_ttl(120_000, 2_000), 182_000);
    assert_eq!(commit_release_ring_ttl(0, 0), 60_000, "never 0");
    for (deadline, timeout) in [
        (u64::MAX, u64::MAX),
        (u64::MAX, 0),
        (0, u64::MAX),
        (999_999_999_999, 1),
    ] {
        assert_eq!(
            commit_release_ring_ttl(deadline, timeout),
            999_999_999_999,
            "({deadline}, {timeout}) saturates and clamps to twelve digits"
        );
    }
    assert_eq!(
        commit_release_ring_ttl(999_999_939_999, 0),
        999_999_999_999,
        "the bound itself is reachable"
    );
}

/// T-i7 (and T-ii3). A is chained after P; it rewrites [`FIELD`] from [`A`]
/// to [`B`] at `seq` 3 and loses its reply. Its second attempt fails too
/// (the attempt fault queue, filled by its between-attempts hook), and
/// before its third its successor S — chained after A — rewrites the field
/// from B to [`C`] at `seq` 4. A's third attempt is refused by its
/// predecessor guard: a plain `UNAVAILABLE`, outcome unknown, naming attempt
/// 3. Its lookup enqueues its own `Commit(3)` once; nothing is handed back
/// (no `Resend` member: a counted retry came first); GC drops each released
/// hash by one.
async fn twice_failed_call_refused_after_its_successor_recovers_once_and_hands_back_nothing(
    route: Route,
) {
    let fx = fx(&format!("pgs-i7-{route:?}")).await;
    let t = &fx.t;
    hold_elsewhere(t, &[A, B]).await;
    t.commit_with(
        &t.store(route),
        route,
        &[(FIELD, Val::Ext(A))],
        WRITER,
        1,
        Some(t.chained("pgs-i7-p", "")),
    )
    .await;

    let staging = Staging::for_route(route);
    let faults: Faults = Arc::default();
    let s_seq = Arc::new(Mutex::new(None));
    let hook: BetweenAttemptsHook = {
        let (t, s_store, faults, s_seq) = (
            t.clone(),
            Arc::new(t.store(route)),
            faults.clone(),
            s_seq.clone(),
        );
        Arc::new(move |attempt| {
            let (t, s_store, faults, s_seq) =
                (t.clone(), s_store.clone(), faults.clone(), s_seq.clone());
            Box::pin(async move {
                match attempt {
                    2 => fail_next(&faults, 1),
                    3 => {
                        let seq = t
                            .commit_with(
                                &s_store,
                                route,
                                &[(FIELD, Val::Ext(C))],
                                WRITER,
                                3,
                                Some(t.chained("pgs-i7-s", "pgs-i7-a")),
                            )
                            .await;
                        *s_seq.lock().unwrap() = Some(seq);
                    }
                    _ => {}
                }
            })
        })
    };
    let a_store = staging
        .install(t.store(route))
        .with_fault_injector_for_test(faults.clone())
        .with_commit_tokens_for_test(pinned(&[TOKEN_A]))
        .with_between_attempts_hook_for_test(hook);

    let (result, summary) = t
        .call(
            &a_store,
            route,
            &[(FIELD, Val::Ext(B))],
            WRITER,
            2,
            Some(t.chained("pgs-i7-a", "pgs-i7-p")),
        )
        .await;

    staging.assert_drained();
    assert!(
        faults.lock().unwrap().is_empty(),
        "{route:?}: attempt 2 took its staged failure"
    );
    assert_eq!(*s_seq.lock().unwrap(), Some(4), "{route:?}: S committed");
    assert_outcome_unknown(&result, 3, &format!("{route:?}"));
    assert_eq!(summary.map(|s| s.attempts), Some(3), "{route:?}");
    assert_eq!(
        t.pending().await,
        sorted(vec![t.commit_member(A, 3), t.commit_member(B, 4)]),
        "{route:?}: one copy of each commit's member, and no Resend member"
    );
    assert_eq!(t.lookups(), [1, 0, 0], "{route:?}");

    let before = ref_counts(t, &[A, B]).await;
    t.gc().await;
    assert_each_dropped_by_one(t, &[A, B], &before, &format!("{route:?}")).await;
}

on_every_route!(twice_failed_call_refused_after_its_successor_recovers_once_and_hands_back_nothing:
    fast_path_twice_failed_call_refused_after_its_successor_recovers_once_and_hands_back_nothing,
    structural_route_twice_failed_call_refused_after_its_successor_recovers_once_and_hands_back_nothing,
    watch_path_twice_failed_call_refused_after_its_successor_recovers_once_and_hands_back_nothing);

/// T-i8. 3 × RING single-release commits, each by a `client_id` of its own,
/// leave the ring at exactly 2·RING + 1 fields — RING entries, RING slots
/// and `|n|` — every one of the ring's own shapes, none named after any of
/// the 3 × RING writers: the ring's size does not grow with the writers.
async fn ring_stays_bounded_under_writer_churn(route: Route) {
    let fx = fx(&format!("pgs-i8-{route:?}")).await;
    let t = &fx.t;
    let ls = t.store(route);
    t.commit(&ls, route, &[(G_FIELD, Val::Ext(G0))], WRITER, 1)
        .await;
    let writer = |i: usize| format!("pgs-i8-writer-{i}");

    churn(t, &ls, route, 3 * RING, G0, writer).await;

    let ring = t.ring().await;
    assert_ring_shape(&ring, &format!("{route:?}"));
    assert_eq!(
        ring.len(),
        2 * RING + 1,
        "{route:?}: at most 2·RING + 1 fields, and a full ring has exactly that"
    );
    assert_eq!(entries(&ring).len(), RING, "{route:?}");
    let named_after_a_writer: Vec<&String> = (0..3 * RING)
        .map(writer)
        .filter_map(|w| ring.get_key_value(&w).map(|(k, _)| k))
        .collect();
    assert!(
        named_after_a_writer.is_empty(),
        "{route:?}: no field is named after a writer; got {named_after_a_writer:?}"
    );
}

on_every_route!(ring_stays_bounded_under_writer_churn:
    fast_path_ring_stays_bounded_under_writer_churn,
    structural_route_ring_stays_bounded_under_writer_churn,
    watch_path_ring_stays_bounded_under_writer_churn);

/// T-i9 (a). Under the empty `client_id`, a rewrite of [`FIELD`] from [`A`]
/// to [`B`] at `seq` 3 loses its reply and its retry is answered from its
/// own commit: its `Commit(3)` is pending, and the ring is exactly its
/// entry, its start slot and `|n|` — no field named `""`.
async fn empty_client_id_lost_reply_recovers_by_its_token(route: Route) {
    let fx = fx(&format!("pgs-i9a-{route:?}")).await;
    let t = &fx.t;
    t.commit(&t.store(route), route, &[(FIELD, Val::Ext(A))], "", 1)
        .await;

    let staging = Staging::for_route(route);
    let ls = staging
        .install(t.store(route))
        .with_commit_tokens_for_test(pinned(&[TOKEN_A]));
    let (result, _) = t
        .call(&ls, route, &[(FIELD, Val::Ext(B))], "", 2, None)
        .await;

    staging.assert_drained();
    assert_eq!(
        result
            .unwrap_or_else(|e| panic!("{route:?}: its own commit answers; got {e:?}"))
            .seq,
        3
    );
    assert_eq!(t.pending().await, vec![t.commit_member(A, 3)], "{route:?}");
    assert_eq!(
        t.ring().await,
        BTreeMap::from([
            (format!("|t|{}", hex(TOKEN_A)), rewrite_entry(3, A)),
            ("|s|0".to_owned(), hex(TOKEN_A)),
            ("|n|".to_owned(), "1".to_owned()),
        ]),
        "{route:?}"
    );
}

on_every_route!(empty_client_id_lost_reply_recovers_by_its_token:
    fast_path_empty_client_id_lost_reply_recovers_by_its_token,
    structural_route_empty_client_id_lost_reply_recovers_by_its_token,
    watch_path_empty_client_id_lost_reply_recovers_by_its_token);

/// T-i9 (b). Under the empty `client_id`, T1 rewrites [`FIELD`] from [`A`]
/// to [`B`] at `seq` 3 and loses its reply. Before T1's retry, T2 — a second
/// empty-`client_id` call, through another store — rewrites the field from B
/// to [`C`] at `seq` 4, owing a release. Read inside the hook, right after
/// T2's commit, T1's entry is intact: T2's write took the next slot. (T1's
/// retry then misses its own key — the record is T2's — and, carrying no
/// operation id, applies its delta again: the fail-open re-apply, which
/// references hash B with no fresh increment, so neither counts nor its
/// answer are asserted after it.)
async fn empty_client_id_successor_leaves_the_first_calls_entry_intact(route: Route) {
    let fx = fx(&format!("pgs-i9b-{route:?}")).await;
    let t = &fx.t;
    t.commit(&t.store(route), route, &[(FIELD, Val::Ext(A))], "", 1)
        .await;

    let staging = Staging::for_route(route);
    let seen = Arc::new(Mutex::new(None));
    let hook: BetweenAttemptsHook = {
        let (t, t2_store, seen) = (
            t.clone(),
            Arc::new(
                t.store(route)
                    .with_commit_tokens_for_test(pinned(&[TOKEN_B])),
            ),
            seen.clone(),
        );
        Arc::new(move |attempt| {
            let (t, t2_store, seen) = (t.clone(), t2_store.clone(), seen.clone());
            Box::pin(async move {
                if attempt != 2 {
                    return;
                }
                let seq = t
                    .commit(&t2_store, route, &[(FIELD, Val::Ext(C))], "", 3)
                    .await;
                let entries = entries(&t.ring().await);
                *seen.lock().unwrap() = Some((
                    seq,
                    entries.get(&hex(TOKEN_A)).cloned(),
                    entries.get(&hex(TOKEN_B)).cloned(),
                ));
            })
        })
    };
    let ls = staging
        .install(t.store(route))
        .with_commit_tokens_for_test(pinned(&[TOKEN_A]))
        .with_between_attempts_hook_for_test(hook);

    // T1's answer is not this test's subject: its retry re-applies, which the
    // fast and WATCH routes commit and the structural route refuses as a
    // duplicate of the section its own landed commit created.
    let (_answer, _) = t
        .call(&ls, route, &[(FIELD, Val::Ext(B))], "", 2, None)
        .await;

    staging.assert_drained();
    assert_eq!(
        seen.lock().unwrap().take(),
        Some((4, Some(rewrite_entry(3, A)), Some(rewrite_entry(4, B)))),
        "{route:?}: after T2's commit, T1's entry is intact beside T2's"
    );
}

on_every_route!(empty_client_id_successor_leaves_the_first_calls_entry_intact:
    fast_path_empty_client_id_successor_leaves_the_first_calls_entry_intact,
    structural_route_empty_client_id_successor_leaves_the_first_calls_entry_intact,
    watch_path_empty_client_id_successor_leaves_the_first_calls_entry_intact);

/// T-i10, weighted slots. RING single-release commits fill the ring (`|n|`
/// back at 0). W then rewrites nine external fields to inline values,
/// owing 9 releases — ⌈9 / 8⌉ = 2 slots — and loses its reply:
///
/// - W's write evicts exactly the two oldest entries (slots 0 and 1): its
///   token starts at slot 0, slot 1 is cleared to `""`, `|n|` advances to
///   `2`, and every other slot and entry is as before — RING − 1 entries;
/// - W's entry lists all nine releases;
/// - W's retry is answered from its own commit, and all nine `Commit`
///   members are enqueued from the entry.
async fn wide_entry_evicts_the_two_oldest_and_its_lost_reply_enqueues_all_nine(route: Route) {
    assert_eq!(
        9usize.div_ceil(SLOT_RELEASES),
        2,
        "fixture precondition: nine releases take two slots"
    );
    let fx = fx(&format!("pgs-i10-{route:?}")).await;
    let t = &fx.t;
    let ls = t.store(route);
    let names: Vec<String> = (0..9).map(wide_field).collect();
    let mut seed: Vec<(&str, Val)> = names
        .iter()
        .enumerate()
        .map(|(i, k)| (k.as_str(), Val::Ext(wide_hash(i))))
        .collect();
    seed.push((G_FIELD, Val::Ext(G0)));
    assert_eq!(t.commit(&ls, route, &seed, WRITER, 1).await, 2);
    churn(t, &ls, route, RING, G0, |_| WRITER.to_owned()).await;
    let before = t.ring().await;
    assert_eq!(
        before.get("|n|").map(String::as_str),
        Some("0"),
        "{route:?}: precondition: the ring is full and wraps to slot 0"
    );
    let oldest: Vec<String> = ["|s|0", "|s|1"]
        .iter()
        .map(|slot| before[*slot].clone())
        .collect();

    let staging = Staging::for_route(route);
    let w_store = staging
        .install(t.store(route))
        .with_commit_tokens_for_test(pinned(&[TOKEN_W]));
    let inline: Vec<(&str, Val)> = names
        .iter()
        .enumerate()
        .map(|(i, k)| (k.as_str(), Val::Inline(i as u8)))
        .collect();
    let (result, _) = t.call(&w_store, route, &inline, WRITER, 2, None).await;
    staging.assert_drained();
    let w_seq = result
        .unwrap_or_else(|e| panic!("{route:?}: W's own commit answers; got {e:?}"))
        .seq;
    assert_eq!(w_seq, 2 + RING as u64 + 1);

    let after = t.ring().await;
    assert_ring_shape(&after, &format!("{route:?}"));
    assert_eq!(
        (
            after.get("|s|0").cloned(),
            after.get("|s|1").cloned(),
            after.get("|n|").cloned()
        ),
        (
            Some(hex(TOKEN_W)),
            Some(String::new()),
            Some("2".to_owned())
        ),
        "{route:?}: W starts at slot 0, covers slot 1, and |n| advances by two"
    );
    for token in &oldest {
        assert!(
            !after.contains_key(&format!("|t|{token}")),
            "{route:?}: the oldest entry {token} was evicted"
        );
    }
    for k in 2..RING {
        let slot = format!("|s|{k}");
        let token = &before[&slot];
        assert_eq!(after.get(&slot), Some(token), "{route:?}: {slot} untouched");
        let entry = format!("|t|{token}");
        assert_eq!(
            after.get(&entry),
            before.get(&entry),
            "{route:?}: the entry starting at {slot} untouched"
        );
    }
    assert_eq!(entries(&after).len(), RING - 1, "{route:?}");
    let mut nine: Vec<String> = (0..9)
        .map(|i| format!("rewrite:{}", sha256_to_hex(&ext_hash(wide_hash(i)))))
        .collect();
    nine.sort();
    assert_eq!(
        as_set(&after[&format!("|t|{}", hex(TOKEN_W))]),
        (w_seq, nine),
        "{route:?}: W's entry lists all nine releases"
    );

    let suffix = format!(":{}:{w_seq}", t.oid);
    let w_members: Vec<String> = t
        .pending()
        .await
        .into_iter()
        .filter(|m| m.ends_with(&suffix))
        .collect();
    assert_eq!(
        w_members,
        sorted(
            (0..9)
                .map(|i| t.commit_member(wide_hash(i), w_seq))
                .collect()
        ),
        "{route:?}: all nine of W's releases enqueued after its lost reply"
    );
}

on_every_route!(wide_entry_evicts_the_two_oldest_and_its_lost_reply_enqueues_all_nine:
    fast_path_wide_entry_evicts_the_two_oldest_and_its_lost_reply_enqueues_all_nine,
    structural_route_wide_entry_evicts_the_two_oldest_and_its_lost_reply_enqueues_all_nine,
    watch_path_wide_entry_evicts_the_two_oldest_and_its_lost_reply_enqueues_all_nine);

// ═════════════════════════════════════════════════════════════════════════════
// (i) The lookup's other endings and outcomes, and the endings outside its set
// ═════════════════════════════════════════════════════════════════════════════

/// Write `value` into the ring entry of `token` on `t`'s object, out of band.
async fn plant_entry(t: &Target, token: u128, value: &str) {
    let _: () = t
        .client
        .pool
        .hset(
            t.keys.obj_write_releases(&t.oid.to_string()),
            [(format!("|t|{}", hex(token)), value.to_owned())],
        )
        .await
        .expect("HSET a ring entry");
}

/// T-i11, the `failed` outcome. A plain call rewrites [`FIELD`] from [`A`]
/// to [`B`] at `seq` 3 and loses its reply. Its between-attempts hook then
/// overwrites the call's own entry with a value that does not parse (`x`)
/// and fails every further attempt, so its retries run out. The call is
/// still answered `ABORTED`; its lookup reads the entry, cannot parse it,
/// and counts `failed` once (`[0, 0, 1]`), enqueuing nothing — the leak
/// direction, left to the reconciler.
async fn unparseable_entry_is_a_failed_lookup_that_enqueues_nothing(route: Route) {
    let fx = fx(&format!("pgs-i11-{route:?}")).await;
    let t = &fx.t;
    t.commit(&t.store(route), route, &[(FIELD, Val::Ext(A))], WRITER, 1)
        .await;

    let staging = Staging::for_route(route);
    let faults: Faults = Arc::default();
    let hook: BetweenAttemptsHook = {
        let (t, faults) = (t.clone(), faults.clone());
        Arc::new(move |attempt| {
            let (t, faults) = (t.clone(), faults.clone());
            Box::pin(async move {
                if attempt != 2 {
                    return;
                }
                plant_entry(&t, TOKEN_A, "x").await;
                fail_next(&faults, (EXHAUSTING_ATTEMPTS - 1) as usize);
            })
        })
    };
    let ls = staging
        .install(
            t.store(route)
                .with_update_layer_retry_max_attempts(EXHAUSTING_ATTEMPTS),
        )
        .with_fault_injector_for_test(faults.clone())
        .with_commit_tokens_for_test(pinned(&[TOKEN_A]))
        .with_between_attempts_hook_for_test(hook);

    let (result, _) = t
        .call(&ls, route, &[(FIELD, Val::Ext(B))], WRITER, 2, None)
        .await;

    staging.assert_drained();
    assert!(
        faults.lock().unwrap().is_empty(),
        "{route:?}: attempts 2 and 3 each took their staged failure"
    );
    assert!(
        matches!(result, Err(ServiceError::Aborted(_))),
        "{route:?}: the answer is unchanged, ABORTED; got {result:?}"
    );
    assert_eq!(
        t.lookups(),
        [0, 0, 1],
        "{route:?}: the lookup could not parse the entry"
    );
    assert!(
        t.pending().await.is_empty(),
        "{route:?}: nothing enqueued, {} least of all",
        t.commit_member(A, 3)
    );
}

on_every_route!(unparseable_entry_is_a_failed_lookup_that_enqueues_nothing:
    fast_path_unparseable_entry_is_a_failed_lookup_that_enqueues_nothing,
    structural_route_unparseable_entry_is_a_failed_lookup_that_enqueues_nothing,
    watch_path_unparseable_entry_is_a_failed_lookup_that_enqueues_nothing);

/// T-i12, exhaustion at attempt 1. With a retry budget of one attempt, a
/// plain call rewriting [`FIELD`] from [`A`] to [`B`] at `seq` 3 loses its
/// only reply: the call ends `ABORTED` after one attempt, and because that
/// attempt may have committed — it did — its lookup enqueues the commit's
/// `Commit(3)` (`enqueued` 1). The lookup runs on exhaustion at any attempt
/// number, not only after a counted retry.
async fn exhaustion_at_attempt_one_recovers_its_landed_commits_releases(route: Route) {
    let fx = fx(&format!("pgs-i12-{route:?}")).await;
    let t = &fx.t;
    t.commit(&t.store(route), route, &[(FIELD, Val::Ext(A))], WRITER, 1)
        .await;

    let staging = Staging::for_route(route);
    let ls = staging
        .install(t.store(route).with_update_layer_retry_max_attempts(1))
        .with_commit_tokens_for_test(pinned(&[TOKEN_A]));
    let (result, summary) = t
        .call(&ls, route, &[(FIELD, Val::Ext(B))], WRITER, 2, None)
        .await;

    staging.assert_drained();
    assert!(
        matches!(result, Err(ServiceError::Aborted(_))),
        "{route:?}: one attempt, its reply lost: ABORTED; got {result:?}"
    );
    assert_eq!(summary.map(|s| s.attempts), Some(1), "{route:?}");
    assert_eq!(
        t.pending().await,
        vec![t.commit_member(A, 3)],
        "{route:?}: the landed commit's release, recovered by the lookup"
    );
    assert_eq!(t.lookups(), [1, 0, 0], "{route:?}");
}

on_every_route!(exhaustion_at_attempt_one_recovers_its_landed_commits_releases:
    fast_path_exhaustion_at_attempt_one_recovers_its_landed_commits_releases,
    structural_route_exhaustion_at_attempt_one_recovers_its_landed_commits_releases,
    watch_path_exhaustion_at_attempt_one_recovers_its_landed_commits_releases);

/// T-i13, a refusal-detail answer after a counted retry, on `route`. A plain
/// call rewrites [`FIELD`] from [`A`] to [`B`] at `seq` 3 and loses its
/// reply. Before its second attempt another commit of the same writer, S,
/// rewrites the field from B to [`C`] at `seq` 4 — so the attempt misses its
/// own key — and a snapshot reader takes the object's fence. The attempt
/// meets the live fence, and the call may not sleep (its remaining deadline,
/// 1 ms, is shorter than any fence sleep), so the fence wait fails it with
/// the `SNAPSHOT_FENCE_TIMEOUT` refusal detail, which after the counted
/// retry is answered a plain `UNAVAILABLE`, outcome unknown, naming attempt
/// 2. That ending runs the lookup: the call's `Commit(3)` is enqueued beside
/// S's `Commit(4)` (`enqueued` 1).
///
/// The fast and WATCH routes: on the structural route a retry that misses
/// its own key re-plans the call's own `CreateSection`, which the planner
/// refuses as a duplicate before any fence is met.
async fn fence_wait_refusal_after_a_lost_reply_recovers_its_releases(route: Route) {
    use live_edit::storage::snapshot_fence::{self as fence, AcquireVerdict, LayerSize};

    let fx = fx(&format!("pgs-i13-{route:?}")).await;
    let t = &fx.t;
    t.commit(&t.store(route), route, &[(FIELD, Val::Ext(A))], WRITER, 1)
        .await;

    let staging = Staging::for_route(route);
    let lease = Arc::new(Mutex::new(None));
    let hook: BetweenAttemptsHook = {
        let (t, s_store, lease) = (t.clone(), Arc::new(t.store(route)), lease.clone());
        Arc::new(move |attempt| {
            let (t, s_store, lease) = (t.clone(), s_store.clone(), lease.clone());
            Box::pin(async move {
                if attempt != 2 {
                    return;
                }
                t.commit(&s_store, route, &[(FIELD, Val::Ext(C))], WRITER, 3)
                    .await;
                let size = LayerSize {
                    items: 1_000,
                    bytes: 100_000,
                };
                match fence::acquire_fence(&t.client, &t.keys, t.oid, size)
                    .await
                    .expect("acquire_fence must answer")
                {
                    AcquireVerdict::Granted(granted) => *lease.lock().unwrap() = Some(granted),
                    other => panic!("the fence must be granted; got {other:?}"),
                }
            })
        })
    };
    let ls = staging
        .install(t.store(route).with_fence_wait_deadline(
            Duration::from_secs(30),
            Arc::new(|| Some(Duration::from_millis(1))),
        ))
        .with_commit_tokens_for_test(pinned(&[TOKEN_A]))
        .with_between_attempts_hook_for_test(hook);

    let (result, summary) = t
        .call(&ls, route, &[(FIELD, Val::Ext(B))], WRITER, 2, None)
        .await;

    staging.assert_drained();
    assert!(
        lease.lock().unwrap().is_some(),
        "{route:?}: precondition: a reader held the fence for attempt 2"
    );
    assert_outcome_unknown(&result, 2, &format!("{route:?}"));
    assert_eq!(summary.map(|s| s.attempts), Some(2), "{route:?}");
    assert_eq!(
        t.pending().await,
        sorted(vec![t.commit_member(A, 3), t.commit_member(B, 4)]),
        "{route:?}: the call's own release, recovered by the lookup, beside S's"
    );
    assert_eq!(t.lookups(), [1, 0, 0], "{route:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fast_path_fence_wait_refusal_after_a_lost_reply_recovers_its_releases() {
    fence_wait_refusal_after_a_lost_reply_recovers_its_releases(Route::Fast).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_path_fence_wait_refusal_after_a_lost_reply_recovers_its_releases() {
    fence_wait_refusal_after_a_lost_reply_recovers_its_releases(Route::Watch).await;
}

/// An ending outside the lookup's set, staged between a lost reply and the
/// retry.
#[derive(Clone, Copy, Debug)]
enum OutsideEnding {
    /// The header is deleted: the retry is answered `NOT_FOUND`.
    NotFound,
    /// Another commit of the writer moves its record, then the object is
    /// marked superseded: the retry misses its own key and is answered
    /// `FAILED_PRECONDITION`.
    FailedPrecondition,
    /// The commit-token record key is made to hold a string: the retry's
    /// own-key read is refused `INTERNAL` (`WRONGTYPE_KEY`) on every route,
    /// before anything is planned.
    Internal,
}

/// T-i14, the closed set. A plain call rewrites [`FIELD`] from [`A`] to
/// [`B`] at `seq` 3 and loses its reply; its entry is in the ring. Before
/// its retry, each [`OutsideEnding`] in turn makes the retry end `NOT_FOUND`,
/// `FAILED_PRECONDITION` or `INTERNAL`. None of them runs the lookup —
/// `NOT_FOUND` and `FAILED_PRECONDITION` include the retired-object exit,
/// whose retirement may be releasing the same references, and `INTERNAL` a
/// torn commit — so the counter stays `[0, 0, 0]` and the call's `Commit(3)`
/// is never enqueued, although its entry is still there to be found.
async fn endings_outside_the_lookup_set_never_look_up(route: Route) {
    for ending in [
        OutsideEnding::NotFound,
        OutsideEnding::FailedPrecondition,
        OutsideEnding::Internal,
    ] {
        let what = format!("{route:?}, {ending:?}");
        let fx = fx(&format!("pgs-i14-{route:?}-{ending:?}")).await;
        let t = &fx.t;
        t.commit(&t.store(route), route, &[(FIELD, Val::Ext(A))], WRITER, 1)
            .await;

        let staging = Staging::for_route(route);
        let hook: BetweenAttemptsHook = {
            let (t, s_store) = (t.clone(), Arc::new(t.store(route)));
            Arc::new(move |attempt| {
                let (t, s_store) = (t.clone(), s_store.clone());
                Box::pin(async move {
                    if attempt != 2 {
                        return;
                    }
                    let oid = t.oid.to_string();
                    match ending {
                        OutsideEnding::NotFound => {
                            let _: i64 = t
                                .client
                                .pool
                                .del(t.keys.obj_header(&oid))
                                .await
                                .expect("DEL obj:hdr");
                        }
                        OutsideEnding::FailedPrecondition => {
                            t.commit(&s_store, route, &[(FIELD, Val::Ext(C))], WRITER, 3)
                                .await;
                            let _: i64 = t
                                .client
                                .pool
                                .hset(t.keys.obj_header(&oid), [("superseded", "1")])
                                .await
                                .expect("HSET superseded=1");
                        }
                        OutsideEnding::Internal => {
                            let wtok = t.keys.obj_write_token(&oid);
                            let _: i64 = t.client.pool.del(&wtok).await.expect("DEL obj:wtok");
                            let _: () = t
                                .client
                                .pool
                                .set(&wtok, "not-a-hash", None, None, false)
                                .await
                                .expect("retype the commit-token record key");
                        }
                    }
                })
            })
        };
        let ls = staging
            .install(t.store(route))
            .with_commit_tokens_for_test(pinned(&[TOKEN_A]))
            .with_between_attempts_hook_for_test(hook);

        let (result, _) = t
            .call(&ls, route, &[(FIELD, Val::Ext(B))], WRITER, 2, None)
            .await;

        staging.assert_drained();
        let ended_so = match ending {
            OutsideEnding::NotFound => matches!(result, Err(ServiceError::NotFound(_))),
            OutsideEnding::FailedPrecondition => {
                matches!(result, Err(ServiceError::FailedPrecondition(_)))
            }
            OutsideEnding::Internal => matches!(result, Err(ServiceError::Internal(_))),
        };
        assert!(
            ended_so,
            "{what}: precondition: the retry ends so; got {result:?}"
        );
        assert_eq!(
            entries(&t.ring().await).get(&hex(TOKEN_A)),
            Some(&rewrite_entry(3, A)),
            "{what}: precondition: the call's entry is there for a lookup to find"
        );
        assert_eq!(t.lookups(), [0, 0, 0], "{what}: no lookup ran");
        assert!(
            !t.pending().await.contains(&t.commit_member(A, 3)),
            "{what}: the call's own release is never enqueued"
        );
    }
}

on_every_route!(endings_outside_the_lookup_set_never_look_up:
    fast_path_endings_outside_the_lookup_set_never_look_up,
    structural_route_endings_outside_the_lookup_set_never_look_up,
    watch_path_endings_outside_the_lookup_set_never_look_up);

/// T-i15, the own-call read's sequence check. Call X (writer `pgs-i15-x`,
/// token T) rewrites [`FIELD`] from [`A`] to [`B`] at `seq` 3, owing the
/// release of hash A: its received reply enqueues `Commit(3)`, and its entry
/// `|t|T` lists A at 3. Call Y (writer `pgs-i15-y`, the SAME token T) then
/// writes an inline field at `seq` 4, owing nothing — so it writes no entry —
/// and loses its reply; its retry is answered from its own commit at 4 and
/// reads `|t|T`, which names `seq` 3. The sequence check refuses that entry:
/// nothing more is enqueued. Without it, Y would enqueue `Commit(4)` of hash
/// A, a second release of a reference X's commit already released.
async fn own_call_read_ignores_an_entry_of_another_commit_under_its_token(route: Route) {
    const TOKEN_T: u128 = 0x7E;
    let fx = fx(&format!("pgs-i15-{route:?}")).await;
    let t = &fx.t;
    t.commit(&t.store(route), route, &[(FIELD, Val::Ext(A))], WRITER, 1)
        .await;
    assert_eq!(
        t.commit(
            &t.store(route)
                .with_commit_tokens_for_test(pinned(&[TOKEN_T])),
            route,
            &[(FIELD, Val::Ext(B))],
            "pgs-i15-x",
            2,
        )
        .await,
        3
    );
    assert_eq!(t.pending().await, vec![t.commit_member(A, 3)]);

    let staging = Staging::for_route(route);
    let ls = staging
        .install(t.store(route))
        .with_commit_tokens_for_test(pinned(&[TOKEN_T]));
    let (result, _) = t
        .call(&ls, route, &[(OTHER, Val::Inline(1))], "pgs-i15-y", 3, None)
        .await;

    staging.assert_drained();
    assert_eq!(
        result
            .unwrap_or_else(|e| panic!("{route:?}: Y's own commit answers; got {e:?}"))
            .seq,
        4
    );
    assert_eq!(
        entries(&t.ring().await).get(&hex(TOKEN_T)),
        Some(&rewrite_entry(3, A)),
        "{route:?}: precondition: the token's entry names X's commit at 3"
    );
    assert_eq!(
        t.pending().await,
        vec![t.commit_member(A, 3)],
        "{route:?}: Y enqueues nothing from an entry naming another commit; {} would be the \
         over-release",
        t.commit_member(A, 4)
    );
}

on_every_route!(own_call_read_ignores_an_entry_of_another_commit_under_its_token:
    fast_path_own_call_read_ignores_an_entry_of_another_commit_under_its_token,
    structural_route_own_call_read_ignores_an_entry_of_another_commit_under_its_token,
    watch_path_own_call_read_ignores_an_entry_of_another_commit_under_its_token);

// ═════════════════════════════════════════════════════════════════════════════
// (ii) The first-attempt hand-back
// ═════════════════════════════════════════════════════════════════════════════

/// Assert the call `token` refused on its first attempt handed back its
/// pre-flight increment of [`H`] — exactly one `Resend(token)` member, no
/// lookup — and that GC returns `H` to `baseline`.
async fn assert_handed_back(t: &Target, token: u128, baseline: Option<i64>, what: &str) {
    assert_eq!(
        t.pending().await,
        vec![t.resend_member(H, token)],
        "{what}: one Resend member of the call's pre-flight increment, under its token"
    );
    assert_eq!(t.lookups(), [0, 0, 0], "{what}: no lookup on attempt 1");
    t.gc().await;
    assert_eq!(
        t.ref_count(H).await,
        baseline,
        "{what}: GC returns the hash to its count before the call's pre-store"
    );
}

/// T-ii1. P, a chain's first write, references [`H`]. C, chained after an
/// operation that never committed, references `H` too (after its own
/// pre-store) and is refused by its predecessor guard on its first attempt:
/// `ABORTED` `PREDECESSOR_NOT_COMMITTED`, one attempt. Nothing of C ran, so
/// it hands back its pre-flight increment as one `Resend(<C's token>)`
/// member, and GC returns `H` to its count before C's pre-store.
async fn first_attempt_predecessor_refusal_hands_back_its_preflight_increment(route: Route) {
    let fx = fx(&format!("pgs-ii1-{route:?}")).await;
    let t = &fx.t;
    t.commit_with(
        &t.store(route),
        route,
        &[(FIELD, Val::Ext(H))],
        WRITER,
        1,
        Some(t.chained("pgs-ii1-p", "")),
    )
    .await;
    let baseline = t.ref_count(H).await;

    let (result, summary) = t
        .call(
            &t.store(route)
                .with_commit_tokens_for_test(pinned(&[TOKEN_C])),
            route,
            &[(OTHER, Val::Ext(H))],
            WRITER,
            2,
            Some(t.chained("pgs-ii1-c", "pgs-ii1-never-committed")),
        )
        .await;

    let what = format!("{route:?}");
    assert_predecessor_not_committed(&result, &what);
    assert_eq!(summary.map(|s| s.attempts), Some(1), "{what}");
    assert_handed_back(t, TOKEN_C, baseline, &what).await;
}

on_every_route!(first_attempt_predecessor_refusal_hands_back_its_preflight_increment:
    fast_path_first_attempt_predecessor_refusal_hands_back_its_preflight_increment,
    structural_route_first_attempt_predecessor_refusal_hands_back_its_preflight_increment,
    watch_path_first_attempt_predecessor_refusal_hands_back_its_preflight_increment);

/// T-ii2. A commit of the writer references [`H`]. X, an unchained operation
/// call whose start record says the writer had no record — stale: the
/// commit wrote one — references `H` too and is overtaken on its first
/// attempt, after one attempt. Nothing of X ran, so it hands back its
/// pre-flight increment as one `Resend(<X's token>)` member, and GC returns
/// `H` to its count before X's pre-store.
async fn first_attempt_overtaken_refusal_hands_back_its_preflight_increment(route: Route) {
    let fx = fx(&format!("pgs-ii2-{route:?}")).await;
    let t = &fx.t;
    t.commit(&t.store(route), route, &[(FIELD, Val::Ext(H))], WRITER, 1)
        .await;
    let baseline = t.ref_count(H).await;

    let (result, summary) = t
        .call(
            &t.store(route)
                .with_commit_tokens_for_test(pinned(&[TOKEN_X])),
            route,
            &[(OTHER, Val::Ext(H))],
            WRITER,
            2,
            Some(
                t.unchained("pgs-ii2-x")
                    .with_start_record(StartRecord::Absent),
            ),
        )
        .await;

    let what = format!("{route:?}");
    assert_overtaken(&result, &what);
    assert_eq!(summary.map(|s| s.attempts), Some(1), "{what}");
    assert_handed_back(t, TOKEN_X, baseline, &what).await;
}

on_every_route!(first_attempt_overtaken_refusal_hands_back_its_preflight_increment:
    fast_path_first_attempt_overtaken_refusal_hands_back_its_preflight_increment,
    structural_route_first_attempt_overtaken_refusal_hands_back_its_preflight_increment,
    watch_path_first_attempt_overtaken_refusal_hands_back_its_preflight_increment);

/// T-ii5, structural route. P, a chain's first write, references [`H`]. C,
/// chained after P, references `H` too. C's first structural commit is
/// answered `FENCED` once (the scripted fenced reply), so the call waits the
/// fence out and re-runs attempt 1 under the same number. Just before the
/// re-run's commit is dispatched (the structural pre-dispatch hook), a fork
/// — a plain commit of the same writer, through another store — moves the
/// writer's record off P, so the script answers `PREDECESSOR`. The fence
/// wait did not consume attempt 1: the call reports one attempt, is refused
/// `ABORTED` `PREDECESSOR_NOT_COMMITTED`, and hands back its pre-flight
/// increment.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_route_fence_wait_keeps_attempt_one_so_a_predecessor_refusal_hands_back() {
    let route = Route::Structural;
    let fx = fx("pgs-ii5-structural").await;
    let t = &fx.t;
    t.commit_with(
        &t.store(route),
        route,
        &[(FIELD, Val::Ext(H))],
        WRITER,
        1,
        Some(t.chained("pgs-ii5-p", "")),
    )
    .await;
    let baseline = t.ref_count(H).await;

    let fenced = Arc::new(Mutex::new(VecDeque::from([1u64])));
    let forked = Arc::new(AtomicBool::new(false));
    let hook: WatchWindowHook = {
        let (t, fork_store, forked) = (t.clone(), Arc::new(t.store(Route::Fast)), forked.clone());
        Arc::new(move |_: &Client| {
            let (t, fork_store, forked) = (t.clone(), fork_store.clone(), forked.clone());
            async move {
                if forked.swap(true, SeqCst) {
                    return;
                }
                t.commit(
                    &fork_store,
                    Route::Fast,
                    &[(G_FIELD, Val::Inline(9))],
                    WRITER,
                    50,
                )
                .await;
            }
            .boxed()
        })
    };
    let c_store = t
        .store(route)
        .with_structural_fenced_replies_for_test(fenced.clone())
        .with_structural_commit_pre_dispatch_hook_for_test(hook)
        .with_commit_tokens_for_test(pinned(&[TOKEN_C]));

    let (result, summary) = t
        .call(
            &c_store,
            route,
            &[(OTHER, Val::Ext(H))],
            WRITER,
            2,
            Some(t.chained("pgs-ii5-c", "pgs-ii5-p")),
        )
        .await;

    assert!(
        fenced.lock().unwrap().is_empty(),
        "precondition: the commit was answered FENCED once"
    );
    assert!(
        forked.load(SeqCst),
        "precondition: the fork committed before the re-run's dispatch"
    );
    assert_predecessor_not_committed(&result, "structural route");
    let summary = summary.expect("the call has a summary");
    assert_eq!(
        summary.attempts, 1,
        "the fence wait re-ran attempt 1 under the same number"
    );
    assert!(
        route.paths().contains(&summary.path),
        "the call ran on the structural route; got {}",
        summary.path
    );
    assert_handed_back(t, TOKEN_C, baseline, "structural route").await;
}

/// A commit charge that prices every need as production does, refuses the
/// first growth it is asked for once — WA0, the WATCH path's first charge
/// point, before its checkout and its read — and grants every waiting hold.
struct RefuseFirstGrowth {
    held: u64,
    refused: bool,
    holds: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl CommitCharge for RefuseFirstGrowth {
    fn held(&self) -> u64 {
        self.held
    }

    fn target(&self, need: &ObjectStateNeed) -> u64 {
        object_state_charge_for_test(need)
    }

    fn rewrite_allowance(&self, need: &ObjectStateNeed) -> MutationAllowance {
        object_state_rewrite_allowance_for_test(self.held, need)
    }

    fn try_hold(&mut self, target: u64) -> bool {
        if target > self.held && !self.refused {
            self.refused = true;
            return false;
        }
        self.held = target;
        true
    }

    async fn hold(&mut self, target: u64) -> Result<(), ServiceError> {
        self.holds.fetch_add(1, SeqCst);
        self.held = target;
        Ok(())
    }
}

/// T-ii5, WATCH path. P, a chain's first write, references [`H`]; then a
/// fork — a plain commit of the same writer — moves the writer's record off
/// P. C, chained after P, references `H` too and commits a structural delta
/// on the `txn` backend under a charge that refuses its first growth (WA0)
/// once: the attempt suspends, the call waits for the charge, and resumes
/// attempt 1 under the same number. Its predecessor check then refuses it.
/// The recharge did not consume attempt 1: one wait, one attempt, `ABORTED`
/// `PREDECESSOR_NOT_COMMITTED`, and the pre-flight increment handed back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_path_recharge_keeps_attempt_one_so_a_predecessor_refusal_hands_back() {
    let fx = fx("pgs-ii5-watch").await;
    let t = &fx.t;
    t.commit_with(
        &t.store(Route::Watch),
        Route::Watch,
        &[(FIELD, Val::Ext(H))],
        WRITER,
        1,
        Some(t.chained("pgs-ii5w-p", "")),
    )
    .await;
    t.commit(
        &t.store(Route::Watch),
        Route::Watch,
        &[(G_FIELD, Val::Inline(9))],
        WRITER,
        2,
    )
    .await;
    let baseline = t.ref_count(H).await;
    pre_store(&t.ext, &[ext_hash(H)]).await;

    // A structural delta, so the WATCH attempt starts at its WA0 charge point.
    let delta = t.delta(Route::Structural, &[(OTHER, Val::Ext(H))]);
    let holds = Arc::new(AtomicUsize::new(0));
    let mut charge = RefuseFirstGrowth {
        held: 0,
        refused: false,
        holds: holds.clone(),
    };
    let relay = relay_ompf(&delta);
    let ls = t
        .store(Route::Watch)
        .with_commit_tokens_for_test(pinned(&[TOKEN_C]));
    let (result, summary) = tokio::time::timeout(
        CALL_BUDGET,
        ls.update_layer_with_summary(
            &t.session_id,
            &t.layer,
            t.oid,
            Arc::new(Some(delta)),
            Some(relay),
            WRITER,
            3,
            Some(t.chained("pgs-ii5w-c", "pgs-ii5w-p")),
            &mut charge,
        ),
    )
    .await
    .expect("the call must settle");

    assert_eq!(
        holds.load(SeqCst),
        1,
        "precondition: the first growth was refused and the call waited once"
    );
    assert_predecessor_not_committed(&result, "WATCH path");
    assert_eq!(
        summary.map(|s| (s.attempts, s.path)),
        Some((1, "watch")),
        "the recharge resumed attempt 1 on the WATCH path"
    );
    assert_handed_back(t, TOKEN_C, baseline, "WATCH path").await;
}
