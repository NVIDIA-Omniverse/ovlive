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

//! The `UpdateLayer` commit metrics' `path` label, conflict counter and
//! commit-duration histogram (Spec SDD SR-2.1–SR-2.3), driven through real
//! `LayerStore::update_layer` calls against Valkey.
//!
//! - **Path.** A call's attempts histogram, retries and commit duration all
//!   land on the path it commits on: `fast` for a field delta on the `lua`
//!   backend; on the `lua` backend's structural route `structural_full` for a
//!   structural delta planned on a cold-started cache entry, `structural` for
//!   one planned on a cache hit, and `reset` for a reset; `watch` for a
//!   structural delta or a reset on `lua-watch-structural`, and any delta on
//!   the `txn` backend. Every path test takes one injected retry, so the retry
//!   counter has a non-zero cell whose path can be wrong.
//! - **Conflicts.** A nil `EXEC` (the WATCHed header written inside the WATCH
//!   window) is one `{path="watch",kind="watch_conflict"}`; a transient I/O
//!   abort on either path is retried the same way but counts no conflict. The
//!   `watch_read_moved` kind is driven by the inline tests in
//!   `src/storage/layer.rs`, whose `#[cfg(test)]` seams land a commit inside
//!   the structural snapshot read.
//! - **Duration.** Exactly one sample per call: `ok` on a commit, `error` on a
//!   validation refusal and on retry exhaustion, the per-object admission wait
//!   included, and one `error` sample for a call whose future is dropped while
//!   it queues for admission.
//!
//! Every metric read here comes from a `Metrics` registry built for the one
//! test, so counts are exact. Duration assertions are lower bounds read from
//! histogram buckets; there is no wall-clock upper bound anywhere.
//!
//! The admission-duration tests park their holding call between its
//! attempts on the `lua` backend (its first attempt failed by the fault
//! injector, its retry parked by the between-attempts hook), past admission.
//! The structural route's own conflict kinds (`seq_fence`,
//! `cache_diverged`, `snapshot_fence_wait`) and its per-attempt path labels
//! are driven in `structural_dispatch_test.rs`.
//!
//! Gated behind `test-fault-injection` (see `mod.rs`): the injectors and the
//! WATCH-window and between-attempts hooks these tests install exist only
//! under that feature.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fred::clients::Client;
use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
use fred::interfaces::HashesInterface;
use futures::FutureExt;

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{
    histogram_bucket, render, sample_sum, sample_value,
};
use live_edit::observability::metrics::{
    Metrics, UPDATE_LAYER_CONFLICT_KINDS, UPDATE_LAYER_PATHS, UPDATE_LAYER_RETRY_OUTCOMES,
};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::{BetweenAttemptsHook, LayerStore};
use live_edit::storage::session::SessionStore;
use live_edit::storage::structural_cache::{CacheLookup, StructuralCache};
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use live_edit::storage::watched_txn::WatchWindowHook;
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand};

use crate::storage::b_bhb_tests::once_only;

/// `path` of a commit on the Lua field fast path.
const FAST: &str = "fast";
/// `path` of a commit on the WATCH/MULTI/EXEC slow path.
const WATCH: &str = "watch";
/// `path` of a structural-route commit planned on a cache hit or repair.
const STRUCTURAL: &str = "structural";
/// `path` of a structural-route commit planned on a cold start or reload.
const STRUCTURAL_FULL: &str = "structural_full";
/// `path` of a reset on the structural route.
const RESET: &str = "reset";

/// Upper bound on any one call or wait below. Each healthy step returns in
/// milliseconds; this only turns a hang into a failure with a message.
const STEP_BUDGET: Duration = Duration::from_secs(20);

/// How long the admission tests hold the object's only permit while the call
/// under test queues for it. The call's recorded duration must be at least
/// this, which the histogram shows as an empty `le="0.25"` bucket.
const HELD_FOR: Duration = Duration::from_millis(300);
/// The largest `UPDATE_LAYER_COMMIT_DURATION_BUCKETS` boundary below
/// [`HELD_FOR`], as Prometheus renders it.
const LE_BELOW_HELD_FOR: &str = "0.25";
/// The structural cache budget of the path tests' stores: far above any
/// fixture object.
const STRUCTURAL_CACHE_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

// ═════════════════════════════════════════════════════════════════════════════
// Fixtures
// ═════════════════════════════════════════════════════════════════════════════

fn store(tv: &TestValkey, backend: UpdateLayerBackend, metrics: &Arc<Metrics>) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(backend)
        .with_metrics(Arc::clone(metrics))
}

async fn create_session(tv: &TestValkey) -> String {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("commit-metrics-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed")
}

async fn create_layer(ls: &LayerStore, session_id: &str, layer: &str) -> u64 {
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    ls.create_layer(
        session_id,
        layer,
        "test-user",
        &state,
        Arc::new(MockExternalValueStore::new()),
    )
    .await
    .expect("create_layer must succeed")
}

/// A field-only delta: no struct commands, not a reset. Fast-path eligible.
fn field_delta() -> OwnedDelta {
    OwnedDelta::default()
}

/// A structural delta: one `CreateSection` under the root, which needs the
/// section tree: the structural route on the `lua` backend, the WATCH path
/// on `lua-watch-structural` and `txn`.
fn structural_delta() -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: 1,
            section_name: "created".to_owned(),
            section_type: 1,
        })],
        ..Default::default()
    }
}

/// A reset (`is_diff` with `base_version` 0): the structural route's `reset`
/// path on the `lua` backend, the WATCH path on `lua-watch-structural` and
/// `txn`.
fn reset_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        source_format: "usd".to_owned(),
        ..Default::default()
    }
}

/// A field delta `validate_delta_bounds` refuses (a control character in the
/// field name) before any attempt runs. Fast-path classified.
fn invalid_field_delta() -> OwnedDelta {
    OwnedDelta {
        set_fields: vec![SetField {
            section_id: 1,
            key_name: "bad\u{1}name".to_owned(),
            value_or_ext_hash: vec![0x11],
            ..Default::default()
        }],
        ..Default::default()
    }
}

/// `count` retriable `Aborted`s for `with_fault_injector_for_test`: each fails
/// one attempt at its start, before it reads or writes anything. Shared with
/// `update_layer_commit_count_test`.
pub(crate) fn aborted_injector(count: usize) -> Arc<Mutex<VecDeque<ServiceError>>> {
    Arc::new(Mutex::new(
        (0..count)
            .map(|_| ServiceError::Aborted("injected retriable failure".to_owned()))
            .collect(),
    ))
}

fn transient_io_injector() -> Arc<Mutex<VecDeque<FredError>>> {
    Arc::new(Mutex::new(VecDeque::from([FredError::new(
        FredErrorKind::IO,
        "injected transient I/O failure",
    )])))
}

// ═════════════════════════════════════════════════════════════════════════════
// Metric readers
// ═════════════════════════════════════════════════════════════════════════════

// The four readers below are shared with `update_layer_commit_count_test`.

pub(crate) fn attempts_count(metrics: &Metrics, path: &str) -> Option<f64> {
    sample_value(
        &render(metrics),
        "live_edit_update_layer_attempts_histogram_count",
        &[("path", path)],
    )
}

pub(crate) fn attempts_sum(metrics: &Metrics, path: &str) -> Option<f64> {
    sample_value(
        &render(metrics),
        "live_edit_update_layer_attempts_histogram_sum",
        &[("path", path)],
    )
}

/// `live_edit_update_layer_attempts_histogram_count` summed over `path`: every
/// commit the service counted landed. `None` before the first.
pub(crate) fn attempts_count_all_paths(metrics: &Metrics) -> Option<f64> {
    sample_sum(
        &render(metrics),
        "live_edit_update_layer_attempts_histogram_count",
        &[],
    )
}

/// `live_edit_update_layer_relay_bytes_count` (b.yvh; unlabelled). `None`
/// before the first sample: a histogram has no series until then.
pub(crate) fn relay_bytes_count(metrics: &Metrics) -> Option<f64> {
    sample_value(
        &render(metrics),
        "live_edit_update_layer_relay_bytes_count",
        &[],
    )
}

/// `live_edit_update_layer_relay_bytes_sum`: the relay bytes of every sampled
/// commit.
pub(crate) fn relay_bytes_sum(metrics: &Metrics) -> Option<f64> {
    sample_value(
        &render(metrics),
        "live_edit_update_layer_relay_bytes_sum",
        &[],
    )
}

/// `live_edit_update_layer_retries_total{path,outcome}`. Every cell is
/// zero-seeded, so absence means the store holds no metrics handle.
pub(crate) fn retries(metrics: &Metrics, path: &str, outcome: &str) -> f64 {
    sample_value(
        &render(metrics),
        "live_edit_update_layer_retries_total",
        &[("path", path), ("outcome", outcome)],
    )
    .unwrap_or_else(|| panic!("retries_total{{path={path:?},outcome={outcome:?}}} is absent"))
}

pub(crate) fn duration_count(metrics: &Metrics, path: &str, outcome: &str) -> Option<f64> {
    sample_value(
        &render(metrics),
        "live_edit_update_layer_commit_duration_seconds_count",
        &[("path", path), ("outcome", outcome)],
    )
}

fn duration_bucket(metrics: &Metrics, path: &str, outcome: &str, le: &str) -> Option<f64> {
    histogram_bucket(
        &render(metrics),
        "live_edit_update_layer_commit_duration_seconds",
        le,
        &[("path", path), ("outcome", outcome)],
    )
}

/// Assert `live_edit_update_layer_conflicts_total` reads `expected` on
/// `{path="watch", kind=watch_kind}` and 0 on every other cell; `None` asserts
/// the whole matrix reads 0.
fn assert_conflicts(metrics: &Metrics, watch_kind: Option<(&str, f64)>, context: &str) {
    let text = render(metrics);
    for path in UPDATE_LAYER_PATHS {
        for kind in UPDATE_LAYER_CONFLICT_KINDS {
            let value = sample_value(
                &text,
                "live_edit_update_layer_conflicts_total",
                &[("path", path), ("kind", kind)],
            )
            .unwrap_or_else(|| {
                panic!("{context}: conflicts_total{{path={path:?},kind={kind:?}}} is absent")
            });
            let want = match watch_kind {
                Some((k, n)) if *path == WATCH && *kind == k => n,
                _ => 0.0,
            };
            assert_eq!(
                value, want,
                "{context}: conflicts_total{{path={path:?},kind={kind:?}}}"
            );
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Path: attempts, retries and duration land on the committing path
// ═════════════════════════════════════════════════════════════════════════════

/// Commit `delta` on a `backend` store whose first attempt is failed by an
/// injected `Aborted`, then assert every `UpdateLayer` commit metric of the
/// call is on `path` and on no other: one attempts sample of value 2, one
/// committed retry, one `ok` duration sample (and no `error` one — the drop
/// guard must not count the call twice). The injected abort is not a race,
/// so no conflict is counted.
async fn assert_commit_lands_on_path(
    backend: UpdateLayerBackend,
    delta: OwnedDelta,
    path: &str,
    layer: &str,
) {
    assert_commit_lands_on_path_after(None, backend, delta, path, layer).await;
}

/// [`assert_commit_lands_on_path`], after `warm_up` — when given — has been
/// committed through another store on the `lua` backend that shares the
/// measured store's structural cache but reports into no registry: the
/// measured call then finds the object's entry cached, and the registry
/// holds the measured call alone.
async fn assert_commit_lands_on_path_after(
    warm_up: Option<OwnedDelta>,
    backend: UpdateLayerBackend,
    delta: OwnedDelta,
    path: &str,
    layer: &str,
) {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("commit-path"));
    let cache = StructuralCache::new(STRUCTURAL_CACHE_BUDGET_BYTES);
    let ls = store(&tv, backend, &metrics)
        .with_structural_cache(cache.clone())
        .with_fault_injector_for_test(aborted_injector(1));
    let session_id = create_session(&tv).await;
    let oid = create_layer(&ls, &session_id, layer).await;
    if let Some(warm_up) = warm_up {
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(UpdateLayerBackend::Lua)
            .with_structural_cache(cache.clone())
            .update_layer_relayed(&session_id, layer, oid, Some(warm_up), "warm-up", 1)
            .await
            .expect("fixture: the warm-up commit must land");
        assert!(
            matches!(cache.lookup(oid), CacheLookup::Hit(_)),
            "fixture: the warm-up must leave the object's entry cached"
        );
    }

    tokio::time::timeout(
        STEP_BUDGET,
        ls.update_layer_relayed(&session_id, layer, oid, Some(delta), "writer", 1),
    )
    .await
    .expect("the update must settle within the budget")
    .expect("the update must commit on attempt 2");

    for other in UPDATE_LAYER_PATHS {
        if *other == path {
            continue;
        }
        assert_eq!(
            attempts_count(&metrics, other),
            None,
            "{layer}: a commit on path {path:?} must record no attempts sample on {other:?}"
        );
        for outcome in UPDATE_LAYER_RETRY_OUTCOMES {
            assert_eq!(
                retries(&metrics, other, outcome),
                0.0,
                "{layer}: a commit on path {path:?} must add no retries on path={other:?}, \
                 outcome={outcome:?}"
            );
        }
        for outcome in ["ok", "error"] {
            assert_eq!(
                duration_count(&metrics, other, outcome),
                None,
                "{layer}: a commit on path {path:?} must record no duration sample on \
                 path={other:?}, outcome={outcome:?}"
            );
        }
    }
    assert_eq!(
        (attempts_count(&metrics, path), attempts_sum(&metrics, path)),
        (Some(1.0), Some(2.0)),
        "{layer}: one commit on attempt 2 must be one attempts sample of value 2 on path {path:?}"
    );
    assert_eq!(
        (
            retries(&metrics, path, "committed"),
            retries(&metrics, path, "exhausted")
        ),
        (1.0, 0.0),
        "{layer}: one retry before the commit must be one committed retry on path {path:?}"
    );
    assert_eq!(
        (
            duration_count(&metrics, path, "ok"),
            duration_count(&metrics, path, "error")
        ),
        (Some(1.0), None),
        "{layer}: a committed call must record exactly one `ok` duration sample on path \
         {path:?} and no `error` one"
    );
    assert_conflicts(&metrics, None, layer);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_field_delta_on_lua_backend_records_every_commit_metric_on_fast_path() {
    assert_commit_lands_on_path(UpdateLayerBackend::Lua, field_delta(), FAST, "path-fast").await;
}

/// A cold cache: the committing attempt's read is a cold start. Attempt 1,
/// failed by the injector before it reads, records only the route's seed
/// path (`structural`); the committing attempt's `structural_full` must
/// replace it on every metric.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_structural_delta_on_lua_backend_cold_records_every_commit_metric_on_structural_full_path(
) {
    assert_commit_lands_on_path(
        UpdateLayerBackend::Lua,
        structural_delta(),
        STRUCTURAL_FULL,
        "path-structural-full",
    )
    .await;
}

/// The object's entry is cached by an earlier commit: the committing
/// attempt plans on a hit, `structural`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_structural_delta_on_lua_backend_hit_records_every_commit_metric_on_structural_path(
) {
    assert_commit_lands_on_path_after(
        Some(structural_delta()),
        UpdateLayerBackend::Lua,
        OwnedDelta {
            struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id: 2,
                section_name: "created-on-a-hit".to_owned(),
                section_type: 1,
            })],
            ..Default::default()
        },
        STRUCTURAL,
        "path-structural-hit",
    )
    .await;
}

/// A reset on the structural route is `reset`, whatever its read was.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_reset_on_lua_backend_records_every_commit_metric_on_reset_path() {
    assert_commit_lands_on_path(UpdateLayerBackend::Lua, reset_delta(), RESET, "path-reset").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_structural_delta_on_lua_watch_structural_backend_records_every_commit_metric_on_watch_path(
) {
    assert_commit_lands_on_path(
        UpdateLayerBackend::LuaWatchStructural,
        structural_delta(),
        WATCH,
        "path-watch-structural",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_reset_on_lua_watch_structural_backend_records_every_commit_metric_on_watch_path(
) {
    assert_commit_lands_on_path(
        UpdateLayerBackend::LuaWatchStructural,
        reset_delta(),
        WATCH,
        "path-watch-reset",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_field_delta_on_txn_backend_records_every_commit_metric_on_watch_path() {
    assert_commit_lands_on_path(
        UpdateLayerBackend::Txn,
        field_delta(),
        WATCH,
        "path-watch-txn",
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Conflicts
// ═════════════════════════════════════════════════════════════════════════════

/// A write to the WATCHed header inside attempt 1's WATCH window cancels its
/// `EXEC` (nil reply): one `{path="watch",kind="watch_conflict"}`, and nothing
/// on any other cell — in particular not `watch_read_moved`, which is the
/// snapshot proof's kind. The write re-sets `seq` to its own value, so attempt
/// 2 meets the same state and commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_nil_exec_counts_one_watch_conflict_and_commits_on_retry() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("nil-exec"));
    let session_id = create_session(&tv).await;
    let layer = "nil-exec";
    let oid = create_layer(
        &store(&tv, UpdateLayerBackend::Txn, &metrics),
        &session_id,
        layer,
    )
    .await;

    let writer = tv.client().pool.next().clone();
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let hook: WatchWindowHook = once_only(Arc::new(move |_watched: &Client| {
        let writer = writer.clone();
        let header_key = header_key.clone();
        async move {
            // `seq` — the header's sequence field. Any write to a WATCHed key
            // cancels the EXEC, even one that leaves its value unchanged.
            let seq: String = writer
                .hget(&header_key, "seq")
                .await
                .expect("HGET seq must succeed");
            let _: () = writer
                .hset(&header_key, [("seq", seq)])
                .await
                .expect("HSET seq must succeed");
        }
        .boxed()
    }));
    let ls = store(&tv, UpdateLayerBackend::Txn, &metrics).with_watch_window_hook_for_test(hook);

    tokio::time::timeout(
        STEP_BUDGET,
        ls.update_layer_relayed(&session_id, layer, oid, Some(field_delta()), "w", 1),
    )
    .await
    .expect("the update must settle within the budget")
    .expect("the update must commit on attempt 2");

    assert_eq!(
        retries(&metrics, WATCH, "committed"),
        1.0,
        "precondition: attempt 1's EXEC must have been cancelled and retried exactly once"
    );
    assert_conflicts(
        &metrics,
        Some(("watch_conflict", 1.0)),
        "a nil EXEC on the WATCH path",
    );
}

/// A transient I/O failure of the `WATCH` round trip is retried like a
/// conflict but is not one: the retry is counted, no conflict cell moves.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_transient_io_abort_on_watch_path_is_retried_and_counts_no_conflict() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("io-watch"));
    let ls = store(&tv, UpdateLayerBackend::Txn, &metrics)
        .with_fred_error_injector_for_test(transient_io_injector());
    let session_id = create_session(&tv).await;
    let layer = "io-watch";
    let oid = create_layer(&ls, &session_id, layer).await;

    tokio::time::timeout(
        STEP_BUDGET,
        ls.update_layer_relayed(&session_id, layer, oid, Some(field_delta()), "w", 1),
    )
    .await
    .expect("the update must settle within the budget")
    .expect("the update must commit on attempt 2");

    assert_eq!(
        retries(&metrics, WATCH, "committed"),
        1.0,
        "precondition: the transient abort must have been retried exactly once"
    );
    assert_conflicts(&metrics, None, "a transient I/O abort on the WATCH path");
}

/// The fast path's twin: a transient I/O failure of the `EVALSHA` is retried
/// and counts no conflict.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_transient_io_abort_on_fast_path_is_retried_and_counts_no_conflict() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("io-fast"));
    let ls = store(&tv, UpdateLayerBackend::Lua, &metrics)
        .with_lua_fred_error_injector_for_test(transient_io_injector());
    let session_id = create_session(&tv).await;
    let layer = "io-fast";
    let oid = create_layer(&ls, &session_id, layer).await;

    tokio::time::timeout(
        STEP_BUDGET,
        ls.update_layer_relayed(&session_id, layer, oid, Some(field_delta()), "w", 1),
    )
    .await
    .expect("the update must settle within the budget")
    .expect("the update must commit on attempt 2");

    assert_eq!(
        retries(&metrics, FAST, "committed"),
        1.0,
        "precondition: the transient abort must have been retried exactly once"
    );
    assert_conflicts(&metrics, None, "a transient I/O abort on the fast path");
}

// ═════════════════════════════════════════════════════════════════════════════
// Commit duration: one sample per call, whatever the ending
// ═════════════════════════════════════════════════════════════════════════════

/// A delta `validate_delta_bounds` refuses ends the call before any attempt:
/// one `error` sample on the path the call classified to (`fast`), no `ok`
/// sample, and no attempts sample.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_validation_refusal_records_one_error_duration_on_its_classified_path() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("duration-invalid"));
    let ls = store(&tv, UpdateLayerBackend::Lua, &metrics);
    let session_id = create_session(&tv).await;
    let layer = "duration-invalid";
    let oid = create_layer(&ls, &session_id, layer).await;

    let refused = tokio::time::timeout(
        STEP_BUDGET,
        ls.update_layer_relayed(&session_id, layer, oid, Some(invalid_field_delta()), "w", 1),
    )
    .await
    .expect("the update must settle within the budget");
    assert!(
        matches!(refused, Err(ServiceError::InvalidArgument(_))),
        "precondition: the delta must be refused by validation; got {refused:?}"
    );

    assert_eq!(
        (
            duration_count(&metrics, FAST, "error"),
            duration_count(&metrics, FAST, "ok")
        ),
        (Some(1.0), None),
        "a validation refusal must record exactly one `error` duration sample on the call's \
         classified path"
    );
    assert_eq!(
        attempts_count(&metrics, FAST),
        None,
        "a refused call commits nothing, so it records no attempts sample"
    );
}

/// A call that runs out of attempts records one `error` sample on its last
/// attempt's path, beside its exhausted retries.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_retry_exhaustion_records_one_error_duration_on_watch_path() {
    const MAX_ATTEMPTS: u32 = 2;

    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("duration-exhausted"));
    let ls = store(&tv, UpdateLayerBackend::Txn, &metrics)
        .with_update_layer_retry_max_attempts(MAX_ATTEMPTS)
        .with_fault_injector_for_test(aborted_injector(MAX_ATTEMPTS as usize));
    let session_id = create_session(&tv).await;
    let layer = "duration-exhausted";
    let oid = create_layer(&ls, &session_id, layer).await;

    let exhausted = tokio::time::timeout(
        STEP_BUDGET,
        ls.update_layer_relayed(&session_id, layer, oid, Some(field_delta()), "w", 1),
    )
    .await
    .expect("the update must settle within the budget");
    assert!(
        matches!(exhausted, Err(ServiceError::Aborted(_))),
        "precondition: every attempt fails, so the call must exhaust; got {exhausted:?}"
    );
    assert_eq!(
        retries(&metrics, WATCH, "exhausted"),
        f64::from(MAX_ATTEMPTS - 1),
        "precondition: the exhaustion must be counted on the WATCH path"
    );

    assert_eq!(
        (
            duration_count(&metrics, WATCH, "error"),
            duration_count(&metrics, WATCH, "ok")
        ),
        (Some(1.0), None),
        "an exhausted call must record exactly one `error` duration sample"
    );
}

/// Two calls on one object of a `lua` store admitting one writer at a time: a
/// structural call that parks between its attempts, so it holds the object's
/// only admission permit, and a fast-path call that queues for that permit.
/// The holder's first attempt is failed by the store's fault injector before
/// it reads anything, and the between-attempts hook parks its retry — past
/// admission, permits held (`update_layer_admission_test`'s structural
/// admission section parks its holder the same way). The fast call is polled
/// once — which starts its duration timer and leaves it waiting on the
/// semaphore — and only then is the wait measured, so every sample it records
/// spans at least [`HELD_FOR`]. Once released, the holder commits on its
/// second attempt, a cold start: `structural_full`.
struct AdmissionHold {
    _tv: TestValkey,
    metrics: Arc<Metrics>,
    ls: Arc<LayerStore>,
    session_id: String,
    layer: &'static str,
    oid: u64,
    holder: tokio::task::JoinHandle<Result<(u64, u64), ServiceError>>,
    release: Arc<tokio::sync::Notify>,
}

impl AdmissionHold {
    async fn start(layer: &'static str) -> Self {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only(layer));
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let hook: BetweenAttemptsHook = {
            let (entered, release) = (Arc::clone(&entered), Arc::clone(&release));
            let fired = Arc::new(AtomicBool::new(false));
            Arc::new(move |_attempt: u32| {
                let first = !fired.swap(true, Ordering::SeqCst);
                let (entered, release) = (Arc::clone(&entered), Arc::clone(&release));
                async move {
                    if first {
                        entered.notify_one();
                        release.notified().await;
                    }
                }
                .boxed()
            })
        };
        let session_id = create_session(&tv).await;
        let oid = create_layer(
            &store(&tv, UpdateLayerBackend::Lua, &metrics),
            &session_id,
            layer,
        )
        .await;
        let ls = Arc::new(
            store(&tv, UpdateLayerBackend::Lua, &metrics)
                .with_update_layer_per_object_concurrency(1)
                .with_fault_injector_for_test(aborted_injector(1))
                .with_between_attempts_hook_for_test(hook),
        );

        let holder = tokio::spawn({
            let ls = Arc::clone(&ls);
            let session_id = session_id.clone();
            async move {
                ls.update_layer_relayed(
                    &session_id,
                    layer,
                    oid,
                    Some(structural_delta()),
                    "holder",
                    1,
                )
                .await
            }
        });
        tokio::time::timeout(STEP_BUDGET, entered.notified())
            .await
            .expect("the holding call must park between its attempts, holding the permit");

        Self {
            _tv: tv,
            metrics,
            ls,
            session_id,
            layer,
            oid,
            holder,
            release,
        }
    }

    /// Let the holding call finish, wait for it, and assert it committed.
    async fn release_holder(self) -> Arc<Metrics> {
        // A no-op when the test already released it: the hook has returned,
        // so the stored notification is never consumed.
        self.release.notify_one();
        tokio::time::timeout(STEP_BUDGET, self.holder)
            .await
            .expect("the holding call must finish within the budget")
            .expect("the holding call's task must not panic")
            .expect("the holding call must commit");
        self.metrics
    }
}

/// The admission wait is inside the commit duration: the fast call queued for
/// [`HELD_FOR`] behind the holding call records one `ok` sample above the
/// `le="0.25"` boundary. A timer started after the permit is acquired would
/// record a few milliseconds instead. The queued call's sample is told apart
/// from the holding call's by path (`fast` against `structural_full`), which
/// the path tests above pin.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_admission_wait_is_included_in_commit_duration() {
    let hold = AdmissionHold::start("duration-admission").await;
    let ls = Arc::clone(&hold.ls);
    let mut queued = Box::pin(ls.update_layer_relayed(
        &hold.session_id,
        hold.layer,
        hold.oid,
        Some(field_delta()),
        "queued",
        2,
    ));
    assert!(
        futures::poll!(queued.as_mut()).is_pending(),
        "precondition: the queued call must wait on the held admission permit"
    );
    // The measured quantity, not synchronisation: the queued call's timer is
    // already running, so its sample spans at least this sleep.
    tokio::time::sleep(HELD_FOR).await;
    hold.release.notify_one();
    tokio::time::timeout(STEP_BUDGET, queued)
        .await
        .expect("the queued call must settle within the budget")
        .expect("the queued call must commit once admitted");
    let metrics = hold.release_holder().await;

    assert_eq!(
        duration_count(&metrics, FAST, "ok"),
        Some(1.0),
        "the queued call must record exactly one `ok` sample on its own path"
    );
    assert_eq!(
        duration_bucket(&metrics, FAST, "ok", LE_BELOW_HELD_FOR),
        Some(0.0),
        "the queued call waited {HELD_FOR:?} for admission, so its sample must be above \
         le=\"{LE_BELOW_HELD_FOR}\" — a sample inside it means the admission wait was not timed"
    );
}

/// A call whose future is dropped while it queues for admission — an elapsed
/// RPC deadline, a client gone — records exactly one `error` sample, on the
/// path it classified to, timed to the drop; it records no `ok` sample, and
/// the commit-duration series of the call that held the permit is untouched.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_dropped_while_queued_for_admission_records_one_error_duration() {
    let hold = AdmissionHold::start("duration-dropped").await;
    let ls = Arc::clone(&hold.ls);
    let mut queued = Box::pin(ls.update_layer_relayed(
        &hold.session_id,
        hold.layer,
        hold.oid,
        Some(field_delta()),
        "queued",
        2,
    ));
    assert!(
        futures::poll!(queued.as_mut()).is_pending(),
        "precondition: the queued call must wait on the held admission permit"
    );
    // The measured quantity, not synchronisation; see the admission test.
    tokio::time::sleep(HELD_FOR).await;
    drop(queued);

    assert_eq!(
        duration_count(&hold.metrics, FAST, "error"),
        Some(1.0),
        "the dropped call must record exactly one `error` sample, on its classified path"
    );
    assert_eq!(
        duration_bucket(&hold.metrics, FAST, "error", LE_BELOW_HELD_FOR),
        Some(0.0),
        "the dropped call queued for {HELD_FOR:?}, so its sample must be above \
         le=\"{LE_BELOW_HELD_FOR}\""
    );

    let metrics = hold.release_holder().await;
    assert_eq!(
        (
            duration_count(&metrics, FAST, "error"),
            duration_count(&metrics, FAST, "ok")
        ),
        (Some(1.0), None),
        "nothing may record a second sample for the dropped call"
    );
    assert_eq!(
        (
            duration_count(&metrics, STRUCTURAL_FULL, "ok"),
            duration_count(&metrics, STRUCTURAL_FULL, "error")
        ),
        (Some(1.0), None),
        "the holding call records its own single `ok` sample on its own path"
    );
}
