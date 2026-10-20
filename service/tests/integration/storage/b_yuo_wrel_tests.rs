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

//! b.yuo part E, b.pgs — the commit release ring `le:{oid}:obj:wrel`.
//!
//! A commit that owes external-value releases records them in its own atomic
//! unit, beside its commit-token record, as an ENTRY of the object's commit
//! release ring: the field `|t|<token>`, keyed by the committing call's commit
//! token, holding `"<seq> <reason>:<hash_hex> …"`. The ring's slots
//! (`|s|<k>`) name the token whose entry starts there, and `|n|` the next
//! slot. When the commit's reply is lost, the call itself enqueues the
//! entry's releases: a later attempt that finds its own commit landed (only
//! when the entry's sequence is that commit's), or — when the call ends
//! without proof of its outcome — its lost-reply lookup, by token. A
//! successor's write takes a later slot and never replaces the entry, so a
//! chained call whose successor commits before its retry recovers its
//! releases too. Before b.yuo those releases were lost with the reply and
//! stranded the released blobs until the hourly reference-count reconciler.
//!
//! The ring's write rule, its eviction and bounds, the lost-reply lookup's
//! other endings and the first-attempt hand-back are pinned in `b_pgs_tests`.
//!
//! # What lives here
//!
//! | Test | What it pins |
//! |---|---|
//! | [`a_rewrite_records_its_releases_in_one_grammar_on_every_route`] | the entry value, identical on the fast, structural and WATCH routes, in a ring of one entry: `|t|<token>`, `|s|0` naming it, `|n|` `1`, and no field named after the writer |
//! | [`a_wrong_typed_release_record_refuses_the_commit_on_both_backends`] | a non-hash `obj:wrel` refuses even a commit that owes nothing, before anything is written |
//! | `a_wrong_typed_release_record_is_refused_ahead_of_*`, [`with_a_healthy_release_record_the_ceiling_refusal_is_unchanged_and_answered_on_the_watch_path`] | the key's refusal sits ahead of the `seq` / `set_order` ceilings and the stream-id collision on the `lua` fast path (script step 1e), the WATCH path (step 2c, both delta shapes) and the cached structural route (its ceilings through the fallback to the WATCH path, its collision at step 1e), with the script's text and on the refusing backend's keyspace-type cell; with a healthy key the ceiling refusal is unchanged, and the cached structural route's is answered on the WATCH path |
//! | `lost_reply::*_lost_reply_enqueues_the_landed_commits_releases_once` | a lost reply on each route — the fast path's lost `EVALSHA` reply (once and twice), the structural route's, and the WATCH path's lost `EXEC` reply whose reconcile read fails — enqueues the landed commit's releases exactly once, byte-identical to an undisturbed commit's |
//! | [`lost_reply::an_earlier_calls_resend_never_reads_the_release_record`] | a re-send answered from an EARLIER call's commit enqueues none of that commit's releases |
//! | [`lost_reply::an_own_call_verdict_for_a_commit_owing_nothing_ignores_another_calls_entry`] | an own-call verdict for a commit that owed nothing finds no entry under its own token and ignores the older commit's entry, another call's |
//! | `lost_reply::*_chained_lost_reply_after_its_successor_commits_recovers_its_releases` | on each route, a chained call's lost reply whose successor commits before its retry: the call is still answered a plain `UNAVAILABLE`, outcome unknown, at attempt 2; its lost-reply lookup enqueues its own `Commit(3)` beside the successor's `Commit(4)` (`lookup{outcome="enqueued"}` 1) and hands nothing back; the ring holds both entries in slots 0 and 1; GC drops each released hash by exactly one; a re-send of the call is refused naming the successor and hands back its own pre-flight increment (`Resend(<token>)`), which GC returns |
//! | `lost_reply::*_wrong_typed_record_after_a_lost_reply_reports_landed_and_enqueues_nothing` | on each route, a release ring made wrong-typed between a lost reply and the retry: the retry still reports `ALREADY_LANDED` (every route reaches its own-call verdict before it probes the key's type), its entry read fails, and nothing is enqueued — pending, the ledger and `ref_count` unchanged (the leak direction) |
//!
//! Fail-on-old proof for the chained case: with the lost-reply lookup
//! skipped (`ReleaseRecoveryOverride::SkipInDoubtLookup` on the call's store)
//! the call's own `Commit(3)` member is missing from the pending set on every
//! route; with the first-attempt hand-back widened to every attempt
//! (`HandBackOnEveryAttempt`) a `Resend` member of the call's pre-flight
//! increment joins it, which the exact pending set refuses; and with the
//! first-attempt hand-back skipped (`SkipAttemptOneHandBack`) the re-send's
//! `Resend` member is missing.
//!
//! A wrong-typed release ring under a live snapshot fence (both backends wait
//! the fence out, then refuse `WRONGTYPE_KEY`, or answer the fence-wait cap's
//! `UNAVAILABLE`) is pinned beside the other fence parity cases in
//! `snapshot_fence_writer_test.rs` section 5.
//!
//! The WATCH path's other in-doubt answer — the reconcile finds the stream
//! entry absent because retention trimmed it — reaches the same retry and the
//! same own-call verdict as the failed reconcile read staged here; no seam
//! stages a trim between the `EXEC` and its reconcile, so that arm is covered
//! through the shared verdict.
//!
//! # Feature gate
//!
//! The `lost_reply` module drives `LayerStore`'s `test-fault-injection` seams
//! and is compiled out without `--features test-fault-injection`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{ClientLike, HashesInterface, KeysInterface, StreamsInterface};

use live_edit::config::UpdateLayerBackend;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::Metrics;
use live_edit::storage::external_values::{ExtRelease, ValkeyExternalValueStore};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand};

use super::b_qpf_tests::{
    create_session, empty_state, ext_hash, ext_hash_hex, ext_hash_vec, ext_store, pending_members,
    pre_store,
};
use crate::common::owned_delta::delta_with_set_field_inline;

/// Deadline on one `update_layer` call; the healthy path takes milliseconds.
const CALL_BUDGET: Duration = Duration::from_secs(25);

/// The writer whose commits each scenario is about.
const WRITER: &str = "b-yuo-wrel-writer";

/// The section every field here lives in (created by the structural route's
/// first commit; the fast and WATCH routes write rows without it).
const SECTION_ID: u64 = 1;

/// The external field each rewrite supersedes.
const FIELD: &str = "color";

/// The first hash the field holds, and the one a rewrite releases.
const A: u8 = 0xA7;
/// The hash a rewrite writes.
const B: u8 = 0xB7;

struct Fx {
    tv: TestValkey,
    metrics: Arc<Metrics>,
    ext: Arc<ValkeyExternalValueStore>,
    session_id: String,
    layer: String,
    oid: u64,
}

async fn fx(layer: &str) -> Fx {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("b-yuo-wrel"));
    let ext = ext_store(&tv);
    let session_id = create_session(&tv, layer).await;
    let oid = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .create_layer(
            &session_id,
            layer,
            "b-yuo-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");
    Fx {
        tv,
        metrics,
        ext,
        session_id,
        layer: layer.to_owned(),
        oid,
    }
}

fn store(fx: &Fx, backend: UpdateLayerBackend) -> LayerStore {
    LayerStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone(), 4096)
        .with_update_layer_backend(backend)
        .with_update_layer_retry_max_attempts(8)
        .with_metrics(fx.metrics.clone())
}

/// The route a scenario runs on, by its delta shape and backend.
#[derive(Clone, Copy, Debug)]
enum Route {
    /// A field-only delta on the `lua` backend: the fast-path script.
    Fast,
    /// A delta carrying a `CreateSection` on the `lua` backend: the script's
    /// structural mode.
    Structural,
    /// A field-only delta on the `txn` backend: WATCH/MULTI/EXEC.
    Watch,
}

impl Route {
    fn backend(self) -> UpdateLayerBackend {
        match self {
            Route::Fast | Route::Structural => UpdateLayerBackend::Lua,
            Route::Watch => UpdateLayerBackend::Txn,
        }
    }

    /// Commit `n` (0-based) setting [`FIELD`] to `hash_byte`'s external
    /// value. On the structural route each commit also creates a section, so
    /// it is structural.
    fn ext_delta(self, n: u64, hash_byte: u8) -> OwnedDelta {
        let field = SetField {
            section_id: SECTION_ID,
            key_name: FIELD.to_owned(),
            value_or_ext_hash: ext_hash_vec(hash_byte),
            ext_value_size: 1_024,
            ext_value_index: 1,
            set_order: n,
        };
        let struct_commands = match self {
            Route::Structural => vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id: SECTION_ID + n,
                section_name: format!("WrelSection{n}"),
                section_type: 1,
            })],
            Route::Fast | Route::Watch => Vec::new(),
        };
        OwnedDelta {
            struct_commands,
            set_fields: vec![field],
            ..Default::default()
        }
    }

    /// The `update_layer` call-summary paths a commit on this route lands on.
    fn paths(self) -> &'static [&'static str] {
        match self {
            Route::Fast => &["fast"],
            Route::Structural => &["structural", "structural_full"],
            Route::Watch => &["watch"],
        }
    }
}

/// One call by [`WRITER`], under [`CALL_BUDGET`], returning its sequence and
/// the path it committed on.
async fn commit(
    ls: &LayerStore,
    fx: &Fx,
    delta: OwnedDelta,
    ts: u64,
) -> Result<(u64, Option<&'static str>), ServiceError> {
    let (result, summary) = tokio::time::timeout(
        CALL_BUDGET,
        ls.update_layer_with_summary_relayed(
            &fx.session_id,
            &fx.layer,
            fx.oid,
            Some(delta),
            WRITER,
            ts,
            None,
        ),
    )
    .await
    .expect("update_layer did not settle within the call budget");
    result.map(|c| (c.seq, summary.map(|s| s.path)))
}

/// The commit release ring `obj:wrel`, every field; empty when the key is
/// absent. Asserts the ring holds no field named after [`WRITER`]: entries are
/// keyed by call token, never by `client_id`.
async fn ring(fx: &Fx) -> BTreeMap<String, String> {
    let ring: BTreeMap<String, String> = fx
        .tv
        .client()
        .pool
        .hgetall(fx.tv.key_schema().obj_write_releases(&fx.oid.to_string()))
        .await
        .expect("HGETALL obj:wrel must succeed");
    assert!(
        !ring.contains_key(WRITER),
        "the ring holds no field named after the writer's client_id; got {ring:?}"
    );
    ring
}

/// The value of the ring's one entry (`|t|<token>`), `None` when it holds
/// none. Panics when it holds more than one.
async fn release_record(fx: &Fx) -> Option<String> {
    let ring = ring(fx).await;
    let entries: Vec<&String> = ring
        .iter()
        .filter(|(field, _)| field.starts_with("|t|"))
        .map(|(_, value)| value)
        .collect();
    assert!(
        entries.len() <= 1,
        "expected at most one ring entry; got {ring:?}"
    );
    entries.first().map(|value| (*value).clone())
}

/// Commit 1 on `route`, undisturbed: [`FIELD`] takes hash [`A`]. A first
/// reference owes no release.
async fn seed(route: Route, fx: &Fx) {
    pre_store(&fx.ext, &[ext_hash(A)]).await;
    let (seq, path) = commit(&store(fx, route.backend()), fx, route.ext_delta(0, A), 1)
        .await
        .expect("the seed commit must land");
    assert_eq!(seq, 2);
    assert!(
        path.is_some_and(|p| route.paths().contains(&p)),
        "{route:?}: the seed commit landed on {path:?}"
    );
    assert!(pending_members(&fx.tv).await.is_empty());
    assert!(
        ring(fx).await.is_empty(),
        "a commit owing nothing writes nothing to the ring"
    );
}

/// What a rewrite of [`FIELD`] from [`A`] to [`B`] at `seq` records and
/// enqueues.
fn rewrite_record(seq: u64) -> String {
    format!("{seq} rewrite:{}", ext_hash_hex(A))
}

fn rewrite_member(fx: &Fx, seq: u64) -> String {
    ExtRelease::commit(ext_hash(A), fx.oid, seq).member()
}

// ═════════════════════════════════════════════════════════════════════════════
// The record, undisturbed
// ═════════════════════════════════════════════════════════════════════════════

/// A rewrite that owes a release records it in one grammar on every route —
/// the Lua script's step 7a in both modes and the WATCH path's `MULTI` write
/// the same value — and enqueues the same member. On a ring that held
/// nothing, the entry takes slot 0: the ring is exactly `|t|<token>`, `|s|0`
/// naming that token, and `|n|` `1`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rewrite_records_its_releases_in_one_grammar_on_every_route() {
    for (route, layer) in [
        (Route::Fast, "b-yuo-wrel-grammar-fast"),
        (Route::Structural, "b-yuo-wrel-grammar-structural"),
        (Route::Watch, "b-yuo-wrel-grammar-watch"),
    ] {
        let fx = fx(layer).await;
        seed(route, &fx).await;
        pre_store(&fx.ext, &[ext_hash(B)]).await;
        let (seq, path) = commit(&store(&fx, route.backend()), &fx, route.ext_delta(1, B), 2)
            .await
            .expect("the rewrite must land");
        assert!(
            path.is_some_and(|p| route.paths().contains(&p)),
            "{route:?}: landed on {path:?}"
        );
        assert_eq!(
            release_record(&fx).await,
            Some(rewrite_record(seq)),
            "{route:?}: the entry is `<seq> <reason>:<hash_hex>`"
        );
        let ring = ring(&fx).await;
        let token = ring
            .keys()
            .find_map(|field| field.strip_prefix("|t|"))
            .expect("one entry")
            .to_owned();
        assert_eq!(
            ring,
            BTreeMap::from([
                (format!("|t|{token}"), rewrite_record(seq)),
                ("|s|0".to_owned(), token.clone()),
                ("|n|".to_owned(), "1".to_owned()),
            ]),
            "{route:?}: one entry, starting at slot 0, the next slot 1"
        );
        assert_eq!(
            pending_members(&fx.tv).await,
            vec![rewrite_member(&fx, seq)],
            "{route:?}: the received reply enqueued the release itself"
        );
    }
}

/// A non-hash `obj:wrel` refuses the commit `WRONGTYPE_KEY` on both
/// backends before anything is written — even a commit that owes no release
/// and so would never write the ring: the type check runs on every commit,
/// so the ring write in the commit's atomic unit (the script's after its
/// stream `XADD`) can never be refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_wrong_typed_release_record_refuses_the_commit_on_both_backends() {
    const PLANTED: &str = "not-a-hash";
    let fx = fx("b-yuo-wrel-wrongtype").await;
    let wrel_key = fx.tv.key_schema().obj_write_releases(&fx.oid.to_string());
    let _: () = fx
        .tv
        .client()
        .pool
        .set(&wrel_key, PLANTED, None, None, false)
        .await
        .expect("plant a string on obj:wrel");
    let stream_len = || async {
        let n: u64 = fx
            .tv
            .client()
            .pool
            .xlen(fx.tv.key_schema().layer_stream(fx.oid))
            .await
            .unwrap();
        n
    };
    let before = stream_len().await;

    for backend in [UpdateLayerBackend::Lua, UpdateLayerBackend::Txn] {
        let err = commit(
            &store(&fx, backend),
            &fx,
            delta_with_set_field_inline(SECTION_ID, "plain", vec![1], 0),
            3,
        )
        .await
        .expect_err("a wrong-typed release record must refuse the commit");
        match err {
            ServiceError::Internal(msg) => {
                for fragment in ["WRONGTYPE_KEY", wrel_key.as_str()] {
                    assert!(
                        msg.contains(fragment),
                        "{backend:?}: the refusal must carry {fragment:?}; got {msg:?}"
                    );
                }
            }
            other => panic!("{backend:?}: expected Internal, got {other:?}"),
        }
    }
    assert_eq!(stream_len().await, before, "nothing was published");
    let still: String = fx.tv.client().pool.get(&wrel_key).await.unwrap();
    assert_eq!(still, PLANTED, "the key is left as it was found");
}

// ═════════════════════════════════════════════════════════════════════════════
// The record's refusal ahead of the ceilings and the stream-id collision
// ═════════════════════════════════════════════════════════════════════════════
//
// The canonical guard order (SDD §7.5) puts the release record's type check
// after the own-key match, the superseded guard and the fence, and ahead of
// the `seq` / `set_order` ceilings and the stream-id collision probe, on
// every route. The cases here cover four:
//
// - the `lua` backend's fast path (a field-only delta): the update script's
//   step 1e;
// - the `txn` backend's WATCH path, field-only and structural: its step 2c;
// - the `lua` backend's cached structural route: its plan hands a ceiling to
//   the WATCH path (step 2c answers) rather than refusing it itself, and a
//   collision, which is no plan refusal, reaches the script's step 1e.
//
// So an object or a delta at a ceiling, or a stream whose next id is taken,
// with a wrong-typed record is refused `WRONGTYPE_KEY` for the record — with
// the one text every route gives, counted on the keyspace-type cell of the
// backend that refused it — and nothing is written; with a healthy record the
// ceiling refusal is unchanged.
//
// In-place proof (fail-on-old): with the WATCH path's step-2c block deleted,
// the WATCH ceiling cases answer the ceiling, and the structural route's
// ceiling cases do too (they reach the WATCH path). The WATCH collision cases
// then fail either as a collision or on the keyspace probe's unqualified
// `WRONGTYPE_KEY` text — the old code's keyspace and collision probes race —
// which the exact-text match here refuses. With the structural route's
// ceiling fallback arm deleted, its ceiling cases answer the plan's own
// ceiling refusal and its healthy companion's sample is not labelled
// `path="watch"`. With the script's step 1e moved after its ceiling and
// collision checks, the fast path's cases and the structural route's
// collision case answer those instead.

/// The `seq` / `set_order` ceiling, `2^53 − 1`.
const CEILING: u64 = live_edit::domain::limits::MAX_DELTA_SET_ORDER;

/// A route by its backend and delta shape.
#[derive(Clone, Copy, Debug)]
enum GuardRoute {
    /// `lua`, a field-only delta: the update script's fast path.
    LuaFast,
    /// `txn`, a field-only delta: the WATCH path without a snapshot read.
    WatchFieldOnly,
    /// `txn`, a structural delta: the WATCH path with its snapshot read.
    WatchStructural,
    /// `lua`, a structural delta: the cached structural route.
    CachedStructural,
}

impl GuardRoute {
    fn backend(self) -> UpdateLayerBackend {
        match self {
            GuardRoute::WatchFieldOnly | GuardRoute::WatchStructural => UpdateLayerBackend::Txn,
            GuardRoute::LuaFast | GuardRoute::CachedStructural => UpdateLayerBackend::Lua,
        }
    }

    /// The `{backend, mode}` cell of `live_edit_keyspace_type_error_total`
    /// (`site="update"`) a refusal of the record ahead of `guard` counts on:
    /// the script's own cell where its step 1e answers, the `txn` cell (no
    /// mode) where the WATCH path's step 2c does — on the cached structural
    /// route, every ceiling, which its plan hands to the WATCH path.
    fn refusing_cell(self, guard: Guard) -> (&'static str, Option<&'static str>) {
        match (self, guard) {
            (GuardRoute::LuaFast, _) => ("lua", Some("fast")),
            (GuardRoute::CachedStructural, Guard::StreamCollision) => ("lua", Some("structural")),
            _ => ("txn", None),
        }
    }

    /// One inline field at `set_order`; on a structural route the delta also
    /// creates a section.
    fn delta(self, set_order: u64) -> OwnedDelta {
        let mut delta = delta_with_set_field_inline(SECTION_ID, "guarded", vec![7], set_order);
        if matches!(
            self,
            GuardRoute::WatchStructural | GuardRoute::CachedStructural
        ) {
            delta.struct_commands = vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id: SECTION_ID,
                section_name: "Guarded".to_owned(),
                section_type: 1,
            })];
        }
        delta
    }
}

/// What a case puts in front of the record's type check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Guard {
    /// A delta whose own `set_order` carries the stored order to the
    /// ceiling: `SET_ORDER_OVERFLOW`.
    SetOrderCeiling,
    /// The object's `seq` at the ceiling: `SEQ_OVERFLOW`.
    SeqCeiling,
    /// An entry on the object's stream at the id the commit would take:
    /// `STREAM_ID_COLLISION`.
    StreamCollision,
}

/// `live_edit_keyspace_type_error_total{site="update"}`, under `backend`
/// (`None`: every backend) and `mode` (`None`: every mode), summed over its
/// other labels.
fn update_keyspace_type_errors(
    metrics: &Metrics,
    backend: Option<&str>,
    mode: Option<&str>,
) -> u64 {
    let mut labels = vec![("site", "update")];
    if let Some(backend) = backend {
        labels.push(("backend", backend));
    }
    if let Some(mode) = mode {
        labels.push(("mode", mode));
    }
    live_edit::observability::metrics::test_util::sample_sum(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_keyspace_type_error_total",
        &labels,
    )
    .unwrap_or(0.0) as u64
}

/// `live_edit_update_layer_commit_duration_seconds` samples with
/// `outcome="error"` under `path` (`None`: every path).
fn error_samples(metrics: &Metrics, path: Option<&str>) -> u64 {
    let mut labels = vec![("outcome", "error")];
    if let Some(path) = path {
        labels.push(("path", path));
    }
    live_edit::observability::metrics::test_util::sample_sum(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_update_layer_commit_duration_seconds_count",
        &labels,
    )
    .unwrap_or(0.0) as u64
}

/// What the object holds that a commit would change: the header's `seq`,
/// the stream's length, and the release record's type.
async fn guarded_image(fx: &Fx) -> (Option<String>, u64, String) {
    let oid = fx.oid.to_string();
    let seq: Option<String> = fx
        .tv
        .client()
        .pool
        .hget(fx.tv.key_schema().obj_header(&oid), "seq")
        .await
        .expect("HGET seq");
    let len: u64 = fx
        .tv
        .client()
        .pool
        .xlen(fx.tv.key_schema().layer_stream(fx.oid))
        .await
        .expect("XLEN");
    let wrel_type: String = fx
        .tv
        .client()
        .pool
        .custom(
            fred::types::CustomCommand::new_static(
                "TYPE",
                fred::types::ClusterHash::FirstKey,
                false,
            ),
            vec![fx.tv.key_schema().obj_write_releases(&oid)],
        )
        .await
        .expect("TYPE obj:wrel");
    (seq, len, wrel_type)
}

/// Stage `guard` on a fresh layer, with a string planted on `obj:wrel` when
/// `wrong_typed`, commit `route`'s delta, and return the fixture, the
/// outcome, the path the call's summary names, and the image before.
async fn commit_past(
    route: GuardRoute,
    guard: Guard,
    wrong_typed: bool,
) -> (
    Fx,
    Result<u64, ServiceError>,
    Option<&'static str>,
    (Option<String>, u64, String),
) {
    let fx = fx(&format!(
        "b-yuo-wrel-guard-{route:?}-{guard:?}-{wrong_typed}"
    ))
    .await;
    let oid = fx.oid.to_string();
    match guard {
        Guard::SetOrderCeiling => {}
        Guard::SeqCeiling => {
            let _: () = fx
                .tv
                .client()
                .pool
                .hset(
                    fx.tv.key_schema().obj_header(&oid),
                    ("seq", CEILING.to_string()),
                )
                .await
                .expect("plant the sequence ceiling");
        }
        Guard::StreamCollision => {
            let _: String = fx
                .tv
                .client()
                .pool
                .xadd(
                    fx.tv.key_schema().layer_stream(fx.oid),
                    false,
                    None,
                    "2-0",
                    vec![("payload", "out-of-band|planted-by-test")],
                )
                .await
                .expect("plant the colliding stream entry");
        }
    }
    if wrong_typed {
        let _: () = fx
            .tv
            .client()
            .pool
            .set(
                fx.tv.key_schema().obj_write_releases(&oid),
                "not-a-hash",
                None,
                None,
                false,
            )
            .await
            .expect("plant a string on obj:wrel");
    }
    let set_order = if guard == Guard::SetOrderCeiling {
        CEILING
    } else {
        0
    };
    let before = guarded_image(&fx).await;
    let ls = store(&fx, route.backend());
    let (result, summary) = tokio::time::timeout(
        CALL_BUDGET,
        ls.update_layer_with_summary_relayed(
            &fx.session_id,
            &fx.layer,
            fx.oid,
            Some(route.delta(set_order)),
            WRITER,
            3,
            None,
        ),
    )
    .await
    .expect("update_layer did not settle within the call budget");
    (fx, result.map(|c| c.seq), summary.map(|s| s.path), before)
}

/// A wrong-typed release record is refused `WRONGTYPE_KEY` — the record's
/// own text, byte for byte the script's — ahead of `guard`, on `route`: one
/// keyspace-type error, on the cell of the step that refused it
/// ([`GuardRoute::refusing_cell`]), nothing written, the key left as found.
async fn assert_record_refused_ahead_of(route: GuardRoute, guard: Guard) {
    let (fx, outcome, _, before) = commit_past(route, guard, true).await;
    let what = format!("{route:?} / {guard:?} / wrong-typed obj:wrel");
    let wrel_key = fx.tv.key_schema().obj_write_releases(&fx.oid.to_string());
    match outcome {
        Err(ServiceError::Internal(msg)) => assert_eq!(
            msg,
            format!(
                "update_layer keyspace type error: WRONGTYPE_KEY: {wrel_key} holds a string, \
                 expected hash (commit release record)"
            ),
            "{what}: the release record is refused first"
        ),
        other => panic!("{what}: expected the record's WRONGTYPE_KEY Internal; got {other:?}"),
    }
    let (backend, mode) = route.refusing_cell(guard);
    assert_eq!(
        (
            update_keyspace_type_errors(&fx.metrics, Some(backend), mode),
            update_keyspace_type_errors(&fx.metrics, None, None),
        ),
        (1, 1),
        "{what}: refused once, counted on the {backend}/{mode:?} cell of the step that refused it"
    );
    assert_eq!(guarded_image(&fx).await, before, "{what}: nothing written");
}

/// Every route the section header lists.
const EVERY_ROUTE: [GuardRoute; 4] = [
    GuardRoute::LuaFast,
    GuardRoute::WatchFieldOnly,
    GuardRoute::WatchStructural,
    GuardRoute::CachedStructural,
];

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_wrong_typed_release_record_is_refused_ahead_of_the_set_order_ceiling_on_every_route() {
    for route in EVERY_ROUTE {
        assert_record_refused_ahead_of(route, Guard::SetOrderCeiling).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_wrong_typed_release_record_is_refused_ahead_of_the_sequence_ceiling_on_every_route() {
    for route in EVERY_ROUTE {
        assert_record_refused_ahead_of(route, Guard::SeqCeiling).await;
    }
}

/// The record is checked ahead of the stream-id collision too: by the
/// script's step 1e on the fast path and on the cached structural route (a
/// collision is no plan refusal, so that route sends the script), by step 2c
/// on the WATCH path, both delta shapes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_wrong_typed_release_record_is_refused_ahead_of_a_stream_id_collision_on_every_route() {
    for route in EVERY_ROUTE {
        assert_record_refused_ahead_of(route, Guard::StreamCollision).await;
    }
}

/// With a healthy release record the ceiling refusal is unchanged — the
/// apply's own `Internal` (`From<DeltaError>`), the text the WATCH path and
/// the structural route's plan each gave before, nothing written, no
/// keyspace-type error — and on the cached structural route the call is now
/// answered by the WATCH path, so its summary and its one commit-duration
/// sample are labelled `path="watch"`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_a_healthy_release_record_the_ceiling_refusal_is_unchanged_and_answered_on_the_watch_path(
) {
    use live_edit::domain::delta::DeltaError;
    let apply_refusal = |error: DeltaError| match ServiceError::from(error) {
        ServiceError::Internal(msg) => msg,
        other => panic!("a ceiling is an Internal refusal; got {other:?}"),
    };
    for (guard, expected) in [
        (
            Guard::SetOrderCeiling,
            apply_refusal(DeltaError::SetOrderOverflow {
                next: 0,
                delta: CEILING,
            }),
        ),
        (
            Guard::SeqCeiling,
            apply_refusal(DeltaError::SequenceOverflow),
        ),
    ] {
        for route in [
            GuardRoute::WatchFieldOnly,
            GuardRoute::WatchStructural,
            GuardRoute::CachedStructural,
        ] {
            let (fx, outcome, path, before) = commit_past(route, guard, false).await;
            let what = format!("{route:?} / {guard:?} / healthy obj:wrel");
            match outcome {
                Err(ServiceError::Internal(msg)) => {
                    assert_eq!(msg, expected, "{what}: the ceiling refusal, unchanged")
                }
                other => panic!("{what}: expected the ceiling's Internal; got {other:?}"),
            }
            assert_eq!(guarded_image(&fx).await, before, "{what}: nothing written");
            assert_eq!(
                update_keyspace_type_errors(&fx.metrics, None, None),
                0,
                "{what}"
            );
            assert_eq!(
                (
                    path,
                    error_samples(&fx.metrics, Some("watch")),
                    error_samples(&fx.metrics, None)
                ),
                (Some("watch"), 1, 1),
                "{what}: the call is answered on the WATCH path — its summary and its one \
                 commit-duration sample say `watch`"
            );
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Lost replies
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod lost_reply {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};

    use live_edit::observability::metrics::test_util::{render, sample_value};
    use live_edit::storage::external_values::{process_pending_gc, GcPolicy};
    use live_edit::storage::layer::{operation_request_digest, Operation, OperationOutcome};
    use live_edit::storage::watched_txn::{WatchedExecOverride, WatchedExecOverrideQueue};

    use super::*;

    type FredQueue = Arc<Mutex<VecDeque<FredError>>>;

    fn timeouts(n: usize) -> FredQueue {
        Arc::new(Mutex::new(
            (0..n)
                .map(|_| FredError::new(FredErrorKind::Timeout, "b.yuo: reply lost"))
                .collect(),
        ))
    }

    fn assert_drained<T: std::fmt::Debug>(queue: &Mutex<VecDeque<T>>, seam: &str) {
        let left = queue.lock().unwrap();
        assert!(
            left.is_empty(),
            "every queued {seam} entry must have been consumed; {left:?} left. An undrained \
             queue means the reply was never lost and the assertions describe an ordinary commit"
        );
    }

    fn already_landed(metrics: &Metrics, backend: &str) -> u64 {
        sample_value(
            &render(metrics),
            "live_edit_update_layer_commit_already_landed_total",
            &[("backend", backend)],
        )
        .expect("commit_already_landed_total cells are zero-seeded") as u64
    }

    /// How the reply of a commit on a route is lost.
    enum Staging {
        /// The post-`EVALSHA` injector: the script commits, `n` replies are
        /// lost.
        PostEval(FredQueue),
        /// The WATCH path's `EXEC` commits and its verdict is replaced by
        /// in-doubt, and the reconcile read fails too, so the retry's
        /// commit-token check answers.
        Exec(WatchedExecOverrideQueue, FredQueue),
    }

    impl Staging {
        fn for_route(route: Route, lost: usize) -> Self {
            match route {
                Route::Fast | Route::Structural => Staging::PostEval(timeouts(lost)),
                Route::Watch => Staging::Exec(
                    Arc::new(Mutex::new(
                        std::iter::repeat_n(WatchedExecOverride::ReplaceOutcomeAfterExec, lost)
                            .collect(),
                    )),
                    timeouts(lost),
                ),
            }
        }

        fn install(&self, ls: LayerStore) -> LayerStore {
            match self {
                Staging::PostEval(q) => {
                    ls.with_lua_post_eval_fred_error_injector_for_test(q.clone())
                }
                Staging::Exec(o, r) => ls
                    .with_watched_exec_override_for_test(o.clone())
                    .with_reconcile_xrange_fred_error_injector_for_test(r.clone()),
            }
        }

        fn assert_drained(&self) {
            match self {
                Staging::PostEval(q) => assert_drained(q, "post-EVALSHA lost reply"),
                Staging::Exec(o, r) => {
                    assert_drained(o, "watched-EXEC override");
                    assert_drained(r, "reconcile XRANGE");
                }
            }
        }
    }

    /// The rewrite at seq 3 loses `lost` replies on `route`: the call still
    /// succeeds (its retry finds its own commit landed) and enqueues exactly
    /// the member an undisturbed rewrite enqueues, once, accounted once.
    async fn lost_reply_enqueues_once(route: Route, lost: usize, layer: &str) {
        let fx = fx(layer).await;
        seed(route, &fx).await;
        let staging = Staging::for_route(route, lost);
        let ls = staging.install(store(&fx, route.backend()));

        pre_store(&fx.ext, &[ext_hash(B)]).await;
        let (seq, _) = commit(&ls, &fx, route.ext_delta(1, B), 2)
            .await
            .expect("the commit landed, so the call must succeed");
        staging.assert_drained();
        assert_eq!(seq, 3);

        let (lua, txn) = (
            already_landed(&fx.metrics, "lua"),
            already_landed(&fx.metrics, "txn"),
        );
        assert_eq!(
            lua + txn,
            1,
            "{route:?}: exactly one attempt decoded its own landed commit (lua={lua}, txn={txn})"
        );
        assert_eq!(
            pending_members(&fx.tv).await,
            vec![rewrite_member(&fx, seq)],
            "{route:?}: the landed commit's release is enqueued exactly as an undisturbed commit \
             enqueues it. Empty is the pre-b.yuo leak: the release rode the lost reply and \
             stranded the blob for the reconciler"
        );
        let member = rewrite_member(&fx, seq);
        let bytes: Option<i64> = fx
            .tv
            .client()
            .pool
            .hget(fx.tv.key_schema().ext_pending_bytes(), member.as_str())
            .await
            .unwrap();
        let total: Option<i64> = fx
            .tv
            .client()
            .pool
            .get(fx.tv.key_schema().ext_pending_total())
            .await
            .unwrap();
        assert_eq!(
            (bytes, total),
            (Some(22), Some(22)),
            "{route:?}: the release accounts the blob's size once (pre_store stores 22 bytes)"
        );
        assert_eq!(release_record(&fx).await, Some(rewrite_record(seq)));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fast_path_lost_reply_enqueues_the_landed_commits_releases_once() {
        lost_reply_enqueues_once(Route::Fast, 1, "b-yuo-wrel-lost-fast").await;
    }

    /// The retry's own `ALREADY_LANDED` reply is lost too; the third attempt's
    /// verdict is the one decoded, and the releases are still enqueued once.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fast_path_twice_lost_reply_enqueues_the_landed_commits_releases_once() {
        lost_reply_enqueues_once(Route::Fast, 2, "b-yuo-wrel-lost-fast-twice").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn structural_route_lost_reply_enqueues_the_landed_commits_releases_once() {
        lost_reply_enqueues_once(Route::Structural, 1, "b-yuo-wrel-lost-structural").await;
    }

    /// The WATCH path's lost `EXEC` reply whose reconcile read fails: the
    /// attempt answers `Aborted` without enqueuing, and the retry's step-2a
    /// own-call verdict enqueues the releases from the record.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn watch_path_lost_reply_enqueues_the_landed_commits_releases_once() {
        lost_reply_enqueues_once(Route::Watch, 1, "b-yuo-wrel-lost-watch").await;
    }

    /// A rewrite at seq 3 on `route` loses its reply, and before the retry
    /// runs — from the between-attempts hook — a string replaces the commit
    /// release ring. The retry still answers `ALREADY_LANDED` from its own
    /// commit-token record, ahead of the ring's type probe on every route:
    /// the fast path's script matches its own key at step 1b and probes the
    /// ring only at step 1e; the structural route's retry is answered by its
    /// pre-plan commit-token read (SR-4.2a) before any script runs; the WATCH
    /// path's step-2a verdict comes before its keyspace probe. So the landed
    /// commit is reported, never refused `WRONGTYPE_KEY`. The entry read
    /// (`HGET obj:wrel |t|<token>`) then fails
    /// and the call enqueues nothing — the leak direction, left to the
    /// reference-count reconciler: pending, the ledger and both hashes'
    /// `ref_count` are as the commit left them, and the planted string is
    /// untouched.
    async fn a_wrong_typed_record_after_the_commit_is_reported_landed(route: Route, layer: &str) {
        let fx = fx(layer).await;
        seed(route, &fx).await;
        let staging = Staging::for_route(route, 1);
        let wrel_key = fx.tv.key_schema().obj_write_releases(&fx.oid.to_string());
        let planted = Arc::new(Mutex::new(0usize));
        let hook: live_edit::storage::layer::BetweenAttemptsHook = {
            let client = fx.tv.client().clone();
            let wrel_key = wrel_key.clone();
            let planted = planted.clone();
            Arc::new(move |attempt| {
                let client = client.clone();
                let wrel_key = wrel_key.clone();
                let planted = planted.clone();
                Box::pin(async move {
                    if attempt != 2 {
                        return;
                    }
                    let _: i64 = client.pool.del(&wrel_key).await.expect("DEL obj:wrel");
                    let _: () = client
                        .pool
                        .set(&wrel_key, "not-a-hash", None, None, false)
                        .await
                        .expect("plant a string on obj:wrel");
                    *planted.lock().unwrap() += 1;
                })
            })
        };
        let ls = staging
            .install(store(&fx, route.backend()))
            .with_between_attempts_hook_for_test(hook);
        let stream_len = || async {
            let n: u64 = fx
                .tv
                .client()
                .pool
                .xlen(fx.tv.key_schema().layer_stream(fx.oid))
                .await
                .unwrap();
            n
        };
        let before = stream_len().await;

        pre_store(&fx.ext, &[ext_hash(B)]).await;
        let ref_counts_before = (ref_count(&fx, A).await, ref_count(&fx, B).await);
        let (seq, _) = commit(&ls, &fx, route.ext_delta(1, B), 2)
            .await
            .unwrap_or_else(|e| {
                panic!(
                    "{route:?}: the commit landed, so the retry must report it ALREADY_LANDED, \
                     not refuse the wrong-typed release record; got {e:?}"
                )
            });
        staging.assert_drained();
        assert_eq!(
            *planted.lock().unwrap(),
            1,
            "{route:?}: the record was replaced before the retry ran"
        );
        assert_eq!(seq, 3);
        assert_eq!(
            already_landed(&fx.metrics, "lua") + already_landed(&fx.metrics, "txn"),
            1,
            "{route:?}: the retry decoded its own landed commit"
        );
        assert_eq!(
            stream_len().await,
            before + 1,
            "{route:?}: one commit landed"
        );
        assert!(
            pending_members(&fx.tv).await.is_empty(),
            "{route:?}: the release read failed, so nothing was enqueued"
        );
        let total: Option<i64> = fx
            .tv
            .client()
            .pool
            .get(fx.tv.key_schema().ext_pending_total())
            .await
            .unwrap();
        let bytes: std::collections::HashMap<String, i64> = fx
            .tv
            .client()
            .pool
            .hgetall(fx.tv.key_schema().ext_pending_bytes())
            .await
            .unwrap();
        assert!(
            total.unwrap_or(0) == 0 && bytes.is_empty(),
            "{route:?}: the ledger accounts nothing; total {total:?}, bytes {bytes:?}"
        );
        assert_eq!(
            (ref_count(&fx, A).await, ref_count(&fx, B).await),
            ref_counts_before,
            "{route:?}: no reference count moved"
        );
        let still: String = fx.tv.client().pool.get(&wrel_key).await.unwrap();
        assert_eq!(still, "not-a-hash", "{route:?}: the key is left as found");
    }

    async fn ref_count(fx: &Fx, hash_byte: u8) -> Option<i64> {
        fx.tv
            .client()
            .pool
            .hget(
                fx.tv.key_schema().ext_meta(&ext_hash_hex(hash_byte)),
                "ref_count",
            )
            .await
            .expect("HGET ref_count")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fast_path_wrong_typed_record_after_a_lost_reply_reports_landed_and_enqueues_nothing() {
        a_wrong_typed_record_after_the_commit_is_reported_landed(
            Route::Fast,
            "b-yuo-wrel-landed-wrongtype-fast",
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn structural_route_wrong_typed_record_after_a_lost_reply_reports_landed_and_enqueues_nothing(
    ) {
        a_wrong_typed_record_after_the_commit_is_reported_landed(
            Route::Structural,
            "b-yuo-wrel-landed-wrongtype-structural",
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn watch_path_wrong_typed_record_after_a_lost_reply_reports_landed_and_enqueues_nothing()
    {
        a_wrong_typed_record_after_the_commit_is_reported_landed(
            Route::Watch,
            "b-yuo-wrel-landed-wrongtype-watch",
        )
        .await;
    }

    /// A re-send answered from an EARLIER call's commit (an operation id
    /// already recorded) never reads the release ring: that call's own
    /// attempts dealt with its releases. Here they were enqueued and
    /// collected, so re-enqueuing them would decrement the superseded hash a
    /// second time. (The re-send hands back its own pre-flight increment, a
    /// `Resend` member of hash B, which this test does not inspect.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_earlier_calls_resend_never_reads_the_release_record() {
        let fx = fx("b-yuo-wrel-earlier-call").await;
        seed(Route::Fast, &fx).await;
        let ls = store(&fx, UpdateLayerBackend::Lua);
        let op = || {
            Operation::new(
                "0b9a7c3e-yuo-wrel",
                operation_request_digest(&fx.layer, b"b.yuo wrel request"),
            )
        };
        let call = |ts| {
            let ls = &ls;
            let fx = &fx;
            async move {
                tokio::time::timeout(
                    CALL_BUDGET,
                    ls.update_layer_with_operation_relayed(
                        &fx.session_id,
                        &fx.layer,
                        fx.oid,
                        Some(Route::Fast.ext_delta(1, B)),
                        WRITER,
                        ts,
                        Some(op()),
                    ),
                )
                .await
                .expect("the call must settle")
            }
        };

        pre_store(&fx.ext, &[ext_hash(B)]).await;
        let original = call(2).await.expect("the original call commits");
        assert_eq!(original.seq, 3);
        assert_eq!(pending_members(&fx.tv).await, vec![rewrite_member(&fx, 3)]);
        process_pending_gc(fx.ext.as_ref(), None, GcPolicy::IMMEDIATE).await;
        assert!(
            pending_members(&fx.tv).await.is_empty(),
            "precondition: collected"
        );

        pre_store(&fx.ext, &[ext_hash(B)]).await;
        let resent = call(3).await.expect("the re-send is answered");
        assert_eq!(resent.outcome, OperationOutcome::AlreadyCommitted);
        let pending = pending_members(&fx.tv).await;
        assert!(
            !pending.iter().any(|m| m.starts_with(&ext_hash_hex(A))),
            "an earlier call's commit releases must never be enqueued by its re-send; got \
             {pending:?}"
        );
    }

    /// An own-call verdict for a commit that owed nothing finds no entry under
    /// its own token, while the ring still holds an OLDER commit's entry —
    /// another call's, of the same writer — and enqueues nothing: enqueuing
    /// that commit's releases again, at the new sequence, would over-release.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_own_call_verdict_for_a_commit_owing_nothing_ignores_another_calls_entry() {
        let fx = fx("b-yuo-wrel-other-seq").await;
        seed(Route::Fast, &fx).await;
        pre_store(&fx.ext, &[ext_hash(B)]).await;
        let (seq, _) = commit(
            &store(&fx, UpdateLayerBackend::Lua),
            &fx,
            Route::Fast.ext_delta(1, B),
            2,
        )
        .await
        .expect("the rewrite lands");
        assert_eq!(release_record(&fx).await, Some(rewrite_record(seq)));
        process_pending_gc(fx.ext.as_ref(), None, GcPolicy::IMMEDIATE).await;
        assert!(
            pending_members(&fx.tv).await.is_empty(),
            "precondition: collected"
        );

        // A commit that owes nothing, whose reply is lost.
        let lost = timeouts(1);
        let ls = store(&fx, UpdateLayerBackend::Lua)
            .with_lua_post_eval_fred_error_injector_for_test(lost.clone());
        let (next, _) = commit(
            &ls,
            &fx,
            delta_with_set_field_inline(SECTION_ID, "plain", vec![7], 2),
            3,
        )
        .await
        .expect("the commit landed");
        assert_drained(&lost, "post-EVALSHA lost reply");
        assert_eq!(next, seq + 1);
        assert_eq!(already_landed(&fx.metrics, "lua"), 1);
        assert_eq!(
            release_record(&fx).await,
            Some(rewrite_record(seq)),
            "a commit owing nothing writes no entry, so the ring holds the older commit's alone"
        );
        assert!(
            pending_members(&fx.tv).await.is_empty(),
            "another call's entry must be ignored"
        );
    }

    // ── b.pav: a chained call's lost reply, after its successor committed ───

    /// The hash the chained successor writes.
    const C: u8 = 0xC7;

    /// The chain's operation ids: P, then A (the call whose reply is lost),
    /// then its successor S.
    const CHAIN_P: &str = "wrel-chain-p";
    const CHAIN_A: &str = "wrel-chain-a";
    const CHAIN_S: &str = "wrel-chain-s";

    /// A chained operation `id` after `previous` (empty: the chain's first
    /// write).
    fn chained_op(fx: &Fx, id: &str, previous: &str) -> Operation {
        Operation::new(id, operation_request_digest(&fx.layer, id.as_bytes()))
            .with_predecessor(previous)
    }

    /// One chained call by [`WRITER`], under [`CALL_BUDGET`].
    async fn chained_commit(
        ls: &LayerStore,
        fx: &Fx,
        delta: OwnedDelta,
        ts: u64,
        operation: Operation,
    ) -> (
        Result<live_edit::storage::layer::UpdateCommit, ServiceError>,
        Option<live_edit::storage::layer::UpdateLayerCallSummary>,
    ) {
        tokio::time::timeout(
            CALL_BUDGET,
            ls.update_layer_with_summary_relayed(
                &fx.session_id,
                &fx.layer,
                fx.oid,
                Some(delta),
                WRITER,
                ts,
                Some(operation),
            ),
        )
        .await
        .expect("update_layer did not settle within the call budget")
    }

    /// The pinned commit tokens of the chained case's calls
    /// (`LayerStore::with_commit_tokens_for_test`): A, whose reply is lost;
    /// its successor S; and A's re-send R. Each names its call's ring entry,
    /// `|t|<token as 32 lowercase hex>`, and a hand-back's `Resend` member.
    const TOKEN_A: u128 = 0xA;
    const TOKEN_S: u128 = 0x5;
    const TOKEN_R: u128 = 0x7;

    /// A one-call queue for `LayerStore::with_commit_tokens_for_test`.
    fn pinned(token: u128) -> Arc<Mutex<VecDeque<u128>>> {
        Arc::new(Mutex::new(VecDeque::from([token])))
    }

    /// `token` in its recorded form: 32 lowercase hex digits.
    fn hex(token: u128) -> String {
        format!("{token:032x}")
    }

    /// `live_edit_update_layer_lost_reply_release_lookup_total` as
    /// `[enqueued, absent, failed]`. Panics on an absent cell: all three are
    /// zero-seeded in `Metrics::new`.
    fn lost_reply_lookups(metrics: &Metrics) -> [u64; 3] {
        let text = render(metrics);
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

    /// One reference to each of `hashes` held outside the object under test,
    /// as another object's would be, so a count GC lowers stays readable: GC
    /// deletes a blob whose count reaches zero, and an absent count cannot
    /// tell one decrement from two.
    async fn hold_elsewhere(fx: &Fx, hashes: &[u8]) {
        let hashes: Vec<[u8; 32]> = hashes.iter().map(|h| ext_hash(*h)).collect();
        pre_store(&fx.ext, &hashes).await;
    }

    /// b.pav test 30 on `route`, flipped by the commit release ring: a chained
    /// call's lost reply whose successor commits before the call's retry.
    ///
    /// A is chained after P; its rewrite of [`FIELD`] from [`A`] to [`B`] at
    /// `seq` 3 owes the release of hash A, records it in its ring entry
    /// `|t|<A's token>` beside its commit, and loses its reply. Before A's
    /// next attempt (the between-attempts hook), its successor S — chained
    /// after A, through another store, after its own pre-store of hash [`C`]
    /// — rewrites the field from B to C at `seq` 4, owing the release of hash
    /// B. A's next attempt reads S's record: it misses its own key, and its
    /// predecessor guard refuses it after a counted retry, so the call ends
    /// without proof of its outcome and runs its lost-reply lookup:
    ///
    /// - A is still answered a plain `UNAVAILABLE`, outcome unknown, naming
    ///   attempt 2 (the client settles it by finishing the chain);
    /// - pending is exactly A's `Commit(3)` of hash A and S's `Commit(4)` of
    ///   hash B: the lookup found A's entry by its token (its `enqueued` cell
    ///   ticks once) and, after the counted retry, nothing of A's pre-flight
    ///   increment is handed back;
    /// - the ring holds both entries, A's in slot 0 and S's in slot 1, `|n|`
    ///   `2`: S's write took the next slot rather than replacing A's entry;
    /// - collecting what is pending drops hashes A and B by exactly one each
    ///   (each also holds a reference elsewhere, so the count stays
    ///   readable): nothing leaks, nothing is released twice;
    /// - a re-send of A (the same operation and predecessor, after its own
    ///   pre-store of hash B; its commit answers on its first attempt, as a
    ///   `no_wait` re-send's would) is refused `ABORTED`
    ///   `PREDECESSOR_NOT_COMMITTED` naming S, which settles A as committed,
    ///   and hands back its pre-flight increment as a `Resend(<token>)`
    ///   member, which GC returns: hash B is back where the re-send found it.
    async fn chained_lost_reply_after_its_successor_commits_recovers_its_releases(
        route: Route,
        layer: &str,
    ) {
        let fx = fx(layer).await;
        hold_elsewhere(&fx, &[A, B]).await;
        pre_store(&fx.ext, &[ext_hash(A)]).await;
        let (p, _) = chained_commit(
            &store(&fx, route.backend()),
            &fx,
            route.ext_delta(0, A),
            1,
            chained_op(&fx, CHAIN_P, ""),
        )
        .await;
        assert_eq!(p.expect("P must commit").seq, 2);

        let staging = Staging::for_route(route, 1);
        let successor_seq = Arc::new(Mutex::new(None));
        let successor: live_edit::storage::layer::BetweenAttemptsHook = {
            let successor_store =
                Arc::new(store(&fx, route.backend()).with_commit_tokens_for_test(pinned(TOKEN_S)));
            let (session_id, layer, oid) = (fx.session_id.clone(), fx.layer.clone(), fx.oid);
            let ext = fx.ext.clone();
            let operation = chained_op(&fx, CHAIN_S, CHAIN_A);
            let successor_seq = successor_seq.clone();
            Arc::new(move |attempt| {
                let (successor_store, session_id, layer, ext, operation, successor_seq) = (
                    successor_store.clone(),
                    session_id.clone(),
                    layer.clone(),
                    ext.clone(),
                    operation.clone(),
                    successor_seq.clone(),
                );
                Box::pin(async move {
                    if attempt != 2 {
                        return;
                    }
                    pre_store(&ext, &[ext_hash(C)]).await;
                    let landed = successor_store
                        .update_layer_with_operation_relayed(
                            &session_id,
                            &layer,
                            oid,
                            Some(route.ext_delta(2, C)),
                            WRITER,
                            3,
                            Some(operation),
                        )
                        .await
                        .expect("the successor S must commit after A's lost commit");
                    *successor_seq.lock().unwrap() = Some(landed.seq);
                })
            })
        };
        let ls = staging
            .install(store(&fx, route.backend()))
            .with_commit_tokens_for_test(pinned(TOKEN_A))
            .with_between_attempts_hook_for_test(successor);
        pre_store(&fx.ext, &[ext_hash(B)]).await;

        let (result, summary) = chained_commit(
            &ls,
            &fx,
            route.ext_delta(1, B),
            2,
            chained_op(&fx, CHAIN_A, CHAIN_P),
        )
        .await;

        staging.assert_drained();
        assert_eq!(
            *successor_seq.lock().unwrap(),
            Some(4),
            "{route:?}: precondition: A's commit landed at 3 and S committed after it, before \
             A's second attempt"
        );
        match &result {
            Err(ServiceError::Unavailable(message))
                if message.starts_with("UpdateLayer outcome unknown:")
                    && message.contains("attempt 2") => {}
            other => panic!(
                "{route:?}: A's second attempt is refused by its predecessor guard after a \
                 counted retry: a plain UNAVAILABLE, outcome unknown; got {other:?}"
            ),
        }
        assert_eq!(
            summary.map(|s| s.attempts),
            Some(2),
            "{route:?}: two attempts"
        );
        let mut expected = vec![
            rewrite_member(&fx, 3),
            ExtRelease::commit(ext_hash(B), fx.oid, 4).member(),
        ];
        expected.sort();
        assert_eq!(
            pending_members(&fx.tv).await,
            expected,
            "{route:?}: A's own release of hash A at 3, recovered by its lost-reply lookup, \
             beside S's release of hash B at 4 — and no hand-back of A's pre-flight increment \
             after its counted retry"
        );
        assert_eq!(
            lost_reply_lookups(&fx.metrics),
            [1, 0, 0],
            "{route:?}: one lost-reply lookup, which found A's entry"
        );
        assert_eq!(
            ring(&fx).await,
            BTreeMap::from([
                (format!("|t|{}", hex(TOKEN_A)), rewrite_record(3)),
                (
                    format!("|t|{}", hex(TOKEN_S)),
                    format!("4 rewrite:{}", ext_hash_hex(B))
                ),
                ("|s|0".to_owned(), hex(TOKEN_A)),
                ("|s|1".to_owned(), hex(TOKEN_S)),
                ("|n|".to_owned(), "2".to_owned()),
            ]),
            "{route:?}: S's entry took the next slot; A's is intact"
        );

        let before_gc = (ref_count(&fx, A).await, ref_count(&fx, B).await);
        process_pending_gc(fx.ext.as_ref(), None, GcPolicy::IMMEDIATE).await;
        assert!(
            pending_members(&fx.tv).await.is_empty(),
            "{route:?}: both releases were collected"
        );
        assert_eq!(
            (ref_count(&fx, A).await, ref_count(&fx, B).await),
            (
                before_gc.0.map(|count| count - 1),
                before_gc.1.map(|count| count - 1)
            ),
            "{route:?}: each released hash drops by exactly the one release its commit owed \
             (before GC: {before_gc:?})"
        );

        let before_resend = ref_count(&fx, B).await;
        pre_store(&fx.ext, &[ext_hash(B)]).await;
        let (resent, _) = chained_commit(
            &store(&fx, route.backend()).with_commit_tokens_for_test(pinned(TOKEN_R)),
            &fx,
            route.ext_delta(1, B),
            5,
            chained_op(&fx, CHAIN_A, CHAIN_P),
        )
        .await;
        let status = tonic::Status::from(
            resent.expect_err("a re-send of A must be refused: its predecessor is not the record"),
        );
        assert_eq!(status.code(), tonic::Code::Aborted, "{route:?}: {status:?}");
        let wire = crate::common::refusal_detail::decode(&status)
            .unwrap_or_else(|| panic!("{route:?}: the refusal carries an ErrorInfo; {status:?}"));
        assert_eq!(
            (wire.reason.as_str(), wire.metadata),
            (
                live_edit::error::refusal_reason::PREDECESSOR_NOT_COMMITTED,
                vec![
                    ("last_operation_id".to_owned(), CHAIN_S.to_owned()),
                    ("last_record".to_owned(), "operation".to_owned()),
                ]
            ),
            "{route:?}: the re-send names S as the chain's latest write"
        );
        assert_eq!(
            pending_members(&fx.tv).await,
            vec![ExtRelease::resend(ext_hash(B), fx.oid, TOKEN_R).member()],
            "{route:?}: the re-send, refused on its first attempt, hands back its pre-flight \
             increment of hash B under its own token"
        );
        process_pending_gc(fx.ext.as_ref(), None, GcPolicy::IMMEDIATE).await;
        assert_eq!(
            ref_count(&fx, B).await,
            before_resend,
            "{route:?}: GC returns the re-send's increment: hash B is back at its count before \
             the re-send's pre-store"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fast_path_chained_lost_reply_after_its_successor_commits_recovers_its_releases() {
        chained_lost_reply_after_its_successor_commits_recovers_its_releases(
            Route::Fast,
            "b-yuo-wrel-chained-fast",
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn structural_route_chained_lost_reply_after_its_successor_commits_recovers_its_releases()
    {
        chained_lost_reply_after_its_successor_commits_recovers_its_releases(
            Route::Structural,
            "b-yuo-wrel-chained-structural",
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn watch_path_chained_lost_reply_after_its_successor_commits_recovers_its_releases() {
        chained_lost_reply_after_its_successor_commits_recovers_its_releases(
            Route::Watch,
            "b-yuo-wrel-chained-watch",
        )
        .await;
    }
}
