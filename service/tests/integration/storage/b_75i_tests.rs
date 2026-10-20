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

//! b.75i — one `UpdateLayer` call commits at most once, on both backends.
//!
//! # The defect
//!
//! An `update_layer` attempt can commit and still come back to the retry loop
//! as a retriable failure: the Lua fast path's `EVALSHA` ran but its reply was
//! lost (a fred `Timeout`, or fred replaying a script that had already run),
//! or the slow path's `EXEC` reply was lost and the reconciling `XRANGE`
//! failed too. The retry then applied the same delta a second time at
//! `new_seq + 1` — a duplicate commit for subscribers on a field-only delta,
//! and a `DuplicateSection` refusal (`INVALID_ARGUMENT`) for an edit that had
//! actually landed when the delta carried a `CreateSection`.
//!
//! The fix mints ONE random commit token per call, records
//! `<token>:<seq>:<server_ts>` in `le:{oid}:obj:wtok[client_id]` inside the
//! commit's own atomic unit, and has every attempt check that record before
//! it validates or writes anything. An attempt that finds its own token
//! reports the recorded commit and writes nothing.
//!
//! # What lives here
//!
//! | Test | Design item | What it pins |
//! |---|---|---|
//! | [`lost_reply::lua_commit_whose_reply_is_lost_is_reported_once_not_reapplied`] | T1, T9 | the fast path's lost reply: one commit, the response is that commit's `(seq, server_ts)`, the counter ticks once on `lua` |
//! | [`lost_reply::lua_commit_whose_reply_is_lost_twice_still_commits_once`] | T1, T9 | the retry's own `ALREADY_LANDED` reply can be lost too; the third attempt still reports the one commit |
//! | [`lost_reply::txn_landed_exec_whose_reconcile_read_fails_is_reported_by_the_retry`] | T2, T9 | the slow path's in-doubt `EXEC` whose reconcile cannot tell: the retry's step-2a check reports the commit instead of refusing it `DuplicateSection` |
//! | [`lost_reply::txn_retry_reports_its_own_commit_after_another_writer_lands_in_between`] | T3 | a different `client_id` commits between the lost-reply attempt and its retry, and the call still commits once |
//! | [`lost_reply::lua_retry_reports_its_own_commit_after_another_writer_lands_in_between`] | T3 | the same on the fast path, via the fixed-token seam (no hook exists between Lua attempts) |
//! | [`lost_reply::lua_token_match_after_superseded_reports_the_landed_commit`] / [`lost_reply::txn_token_match_after_superseded_reports_the_landed_commit`] | T8, T9 | a landed commit is reported as landed even after an overwrite set `superseded=1`; a fresh call is still refused |
//! | [`lost_reply::commit_token_record_with_this_calls_token_and_a_corrupt_tail_is_refused_internal`] | record parse | a record with this call's token and an unparseable tail is refused `Internal`, identically on both backends, with nothing written; the parser's pure cases are unit tests beside `collect_probe_keys` in `src/storage/layer.rs` |
//! | [`lua_distinct_calls_with_identical_content_both_commit`] / [`txn_distinct_calls_with_identical_content_both_commit`] | T4 | two calls with byte-identical content from one `client_id` are two commits: no content-based deduplication |
//! | [`wrong_typed_commit_token_record_is_refused_identically_on_both_backends`] | T7 | a non-hash `obj:wtok` is refused `WRONGTYPE_KEY` with the same text on both backends, nothing written |
//!
//! Elsewhere: T5 (the record is byte-identical across backends) in
//! `update_layer_lua_test.rs`'s parity dump and its direct-invocation section;
//! T6 (cleanup) in `b_9fn_tests.rs`, `orphan_sweep_test.rs` and
//! `src/checkpoint/residue.rs`; T9's zero-seeded cells in
//! `tests/integration/observability_metrics_test.rs`.
//!
//! # Feature gate
//!
//! The `lost_reply` module drives `LayerStore`'s `test-fault-injection` seams
//! (the post-`EVALSHA` lost-reply injector, the reconcile `XRANGE` injector,
//! the watched-`EXEC` override, the watch-window hook and the fixed-token
//! queue) and is compiled out — not skipped — without
//! `--features test-fault-injection`. The three ungated tests (the two T4
//! distinct-call tests and the T7 wrong-type test) need only Valkey.
//!
//! # Wait budget
//!
//! Every top-level call this file makes is wrapped in [`CALL_BUDGET`]; the
//! healthy path finishes in milliseconds. No test makes more than three such
//! calls (the competitor commit in the T3 txn hook runs inside the writer's
//! one call and so inside its budget), so the worst case is 3 × 25 s = 75 s,
//! under nextest's 120 s hang-killer.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface, StreamsInterface};

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_sum, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::OwnedDelta;

use crate::common::owned_delta::delta_with_set_field_inline;

// ═════════════════════════════════════════════════════════════════════════════
// Budgets
// ═════════════════════════════════════════════════════════════════════════════

/// Deadline on one `update_layer` call. The healthy path is a handful of round
/// trips plus at most a few millisecond-scale retry backoffs; 25 s is two
/// orders of magnitude of headroom for CI runner load. The longest test here
/// makes three calls, so its worst case (75 s) still sits under nextest's
/// 120 s hang-killer and every diagnostic panic below gets printed.
const CALL_BUDGET: Duration = Duration::from_secs(25);

/// Retry budget for every store here. The lost-reply scenarios need at most
/// three attempts; a small cap keeps a regression that loops on `Aborted`
/// from spending the call budget before it fails.
const MAX_ATTEMPTS: u32 = 8;

/// The writer whose call each scenario is about.
const WRITER: &str = "b75i-writer";

/// The writer's client timestamp. Fixed so two calls from the writer can be
/// byte-identical requests.
const WRITER_TS: u64 = 1_000;

// ═════════════════════════════════════════════════════════════════════════════
// Fixture + helpers
// ═════════════════════════════════════════════════════════════════════════════

struct Fixture {
    tv: TestValkey,
    metrics: Arc<Metrics>,
    session_id: String,
    layer: &'static str,
    object_id: u64,
}

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

/// A fresh object at `seq = 1` with one stream entry (`1-0`, CreateLayer's).
async fn fixture(layer: &'static str) -> Fixture {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(layer, &format!("/{layer}"), "b75i-user")
        .await
        .expect("session creation must succeed");
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 1,
        ..Default::default()
    };
    let object_id = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .create_layer(&session_id, layer, "b75i-user", &state, mock_ext_store())
        .await
        .expect("create_layer must succeed");
    Fixture {
        tv,
        metrics,
        session_id,
        layer,
        object_id,
    }
}

/// A store on `backend`, reporting into the fixture's registry.
fn store(fx: &Fixture, backend: UpdateLayerBackend) -> LayerStore {
    LayerStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone(), 4096)
        .with_update_layer_backend(backend)
        .with_update_layer_retry_max_attempts(MAX_ATTEMPTS)
        .with_metrics(fx.metrics.clone())
}

/// A field-only delta: fast-path eligible on the Lua backend, and the slow
/// path on the Txn backend.
fn field_delta(value: u8) -> OwnedDelta {
    delta_with_set_field_inline(1, "color", vec![value], 0)
}

/// One `update_layer` call by `client_id`, under [`CALL_BUDGET`].
async fn update(
    store: &LayerStore,
    fx: &Fixture,
    delta: OwnedDelta,
    client_id: &str,
) -> Result<(u64, u64), ServiceError> {
    tokio::time::timeout(
        CALL_BUDGET,
        store.update_layer_relayed(
            &fx.session_id,
            fx.layer,
            fx.object_id,
            Some(delta),
            client_id,
            WRITER_TS,
        ),
    )
    .await
    .expect("update_layer did not settle within the call budget")
}

async fn header(fx: &Fixture) -> HashMap<String, String> {
    fx.tv
        .client()
        .pool
        .hgetall(fx.tv.key_schema().obj_header(&fx.object_id.to_string()))
        .await
        .expect("HGETALL obj:hdr must succeed")
}

async fn header_seq(fx: &Fixture) -> u64 {
    header(fx)
        .await
        .get("seq")
        .expect("the header must carry seq")
        .parse()
        .expect("header seq must be numeric")
}

/// The per-object stream as `(id, payload)`, in order. Fields are read as
/// bytes: an entry's relay `ompf` field is binary.
async fn stream_payloads(fx: &Fixture) -> Vec<(String, String)> {
    let entries: Vec<(String, HashMap<String, Bytes>)> = fx
        .tv
        .client()
        .pool
        .xrange_values(
            fx.tv.key_schema().layer_stream(fx.object_id),
            "-",
            "+",
            None,
        )
        .await
        .expect("XRANGE over the layer stream must succeed");
    entries
        .into_iter()
        .map(|(id, fields)| {
            let payload = fields
                .get("payload")
                .map(|p| String::from_utf8(p.to_vec()).expect("a stream payload is UTF-8"))
                .unwrap_or_default();
            (id, payload)
        })
        .collect()
}

/// Field `n` of an `update|{oid}|{seq}|{client_id}|{client_ts}|{server_ts}`
/// payload.
fn payload_field(payload: &str, n: usize) -> &str {
    payload
        .split('|')
        .nth(n)
        .unwrap_or_else(|| panic!("update payload must carry six fields; got {payload:?}"))
}

/// Assert the stream is exactly `expected` — `(seq, client_id)` per entry in
/// order, CreateLayer's `1-0` excluded — and return the payload at each.
async fn assert_stream_commits(fx: &Fixture, expected: &[(u64, &str)], why: &str) -> Vec<String> {
    let entries = stream_payloads(fx).await;
    let ids: Vec<&str> = entries.iter().map(|(id, _)| id.as_str()).collect();
    let mut want_ids = vec!["1-0".to_owned()];
    want_ids.extend(expected.iter().map(|(seq, _)| format!("{seq}-0")));
    assert_eq!(
        ids,
        want_ids.iter().map(String::as_str).collect::<Vec<_>>(),
        "{why}. Full stream: {entries:?}"
    );
    entries
        .iter()
        .skip(1)
        .zip(expected)
        .map(|((id, payload), (_, client))| {
            assert_eq!(
                payload_field(payload, 3),
                *client,
                "entry {id} must name the writer that committed it; got {payload:?}"
            );
            payload.clone()
        })
        .collect()
}

/// `live_edit_update_layer_commit_already_landed_total{backend}`.
fn already_landed(metrics: &Metrics, backend: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_update_layer_commit_already_landed_total",
        &[("backend", backend)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_update_layer_commit_already_landed_total{{backend=\"{backend}\"}} is \
             absent from the exposition. Both cells are seeded at zero in `Metrics::new`, so \
             absence means the store never received this metrics handle and every count read \
             from it is vacuous"
        )
    }) as u64
}

/// `live_edit_keyspace_type_error_total{backend, site="update"}`, summed over
/// the `mode` series the `backend="lua"` cell is split into.
fn keyspace_type_errors(metrics: &Metrics, backend: &str) -> u64 {
    sample_sum(
        &render(metrics),
        "live_edit_keyspace_type_error_total",
        &[("backend", backend), ("site", "update")],
    )
    .expect("keyspace_type_error_total cells are seeded at zero") as u64
}

// ═════════════════════════════════════════════════════════════════════════════
// T4 — distinct calls are distinct commits (no seam needed)
// ═════════════════════════════════════════════════════════════════════════════

/// Two `update_layer` calls carrying byte-identical requests — same delta,
/// same `client_id`, same client timestamp — are two calls, and both commit.
///
/// This is the guard against "fixing" the double apply by deduplicating on
/// content: the token is random per call precisely so it can never match a
/// different call's commit. A Kit user who sets a field to the same value
/// twice made two edits, and subscribers must see two commits.
async fn distinct_calls_with_identical_content_both_commit(
    backend: UpdateLayerBackend,
    layer: &'static str,
) {
    let fx = fixture(layer).await;
    let store = store(&fx, backend);

    let first = update(&store, &fx, field_delta(0x11), WRITER)
        .await
        .expect("the first call must commit");
    let second = update(&store, &fx, field_delta(0x11), WRITER)
        .await
        .expect("the second, byte-identical call must commit too");

    assert_eq!(
        (first.0, second.0),
        (2, 3),
        "two calls are two commits: the second must land at the next sequence, not be \
         reported as the first one's commit"
    );
    assert_eq!(
        header_seq(&fx).await,
        3,
        "the header advances once per call"
    );
    assert_stream_commits(
        &fx,
        &[(2, WRITER), (3, WRITER)],
        "both calls must publish their own entry",
    )
    .await;
    for backend_label in ["lua", "txn"] {
        assert_eq!(
            already_landed(&fx.metrics, backend_label),
            0,
            "neither call lost a reply, so no attempt may have matched a recorded token \
             (backend={backend_label})"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lua_distinct_calls_with_identical_content_both_commit() {
    distinct_calls_with_identical_content_both_commit(UpdateLayerBackend::Lua, "b75i-t4-lua").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_distinct_calls_with_identical_content_both_commit() {
    distinct_calls_with_identical_content_both_commit(UpdateLayerBackend::Txn, "b75i-t4-txn").await;
}

// ═════════════════════════════════════════════════════════════════════════════
// T7 — a wrong-typed record key is refused, identically, with nothing written
// ═════════════════════════════════════════════════════════════════════════════

/// A non-hash `le:{oid}:obj:wtok` is refused `WRONGTYPE_KEY` on both backends,
/// with byte-identical text, before anything is written.
///
/// Both backends run against the SAME object and the same planted key, so the
/// two messages name the same key and can be compared for equality outright.
/// The record is the one key whose refusal is byte-identical across backends
/// (the per-record keys carry a Lua-only locating qualifier), and both arms
/// tick `keyspace_type_error_total` on their own backend cell.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wrong_typed_commit_token_record_is_refused_identically_on_both_backends() {
    const PLANTED: &str = "not-a-hash";

    let fx = fixture("b75i-t7-wrongtype").await;
    let wtok_key = fx
        .tv
        .key_schema()
        .obj_write_token(&fx.object_id.to_string());
    let _: () = fx
        .tv
        .client()
        .pool
        .set(&wtok_key, PLANTED, None, None, false)
        .await
        .expect("planting a string on obj:wtok must succeed");

    let mut messages = Vec::new();
    for backend in [UpdateLayerBackend::Lua, UpdateLayerBackend::Txn] {
        let err = update(&store(&fx, backend), &fx, field_delta(0x22), WRITER)
            .await
            .expect_err("a wrong-typed commit-token record must refuse the commit");
        match err {
            ServiceError::Internal(msg) => messages.push(msg),
            other => panic!(
                "{backend:?}: a wrong-typed key is keyspace corruption, not retriable and not a \
                 client error, so it must be `Internal`; got {other:?}"
            ),
        }
    }

    assert_eq!(
        messages[0], messages[1],
        "the Lua and txn refusals of the same wrong-typed record must be byte-identical"
    );
    let msg = &messages[0];
    for fragment in ["WRONGTYPE_KEY", wtok_key.as_str(), "(commit-token record)"] {
        assert!(
            msg.contains(fragment),
            "the refusal must carry {fragment:?} so an operator can find and fix the key; \
             got {msg:?}"
        );
    }

    assert_eq!(header_seq(&fx).await, 1, "nothing may have committed");
    assert_stream_commits(&fx, &[], "neither refusal may have published an entry").await;
    let still: String = fx
        .tv
        .client()
        .pool
        .get(&wtok_key)
        .await
        .expect("the planted key must still read as a string");
    assert_eq!(
        still, PLANTED,
        "the refusal must leave the key as it found it"
    );
    assert_eq!(
        (
            keyspace_type_errors(&fx.metrics, "lua"),
            keyspace_type_errors(&fx.metrics, "txn")
        ),
        (1, 1),
        "each backend's refusal ticks its own keyspace_type_error_total cell exactly once"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The lost-reply scenarios (test-fault-injection seams)
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod lost_reply {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use fred::clients::Client;
    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use futures::FutureExt;

    use live_edit::storage::watched_txn::{
        WatchWindowHook, WatchedExecOverride, WatchedExecOverrideQueue,
    };
    use oodf_ffi::{CommandCreateSection, StructCommand};

    use super::*;

    /// A fixed commit token for the seam-driven scenarios. Any 128-bit value
    /// works; what matters is that two calls handed it share it.
    const FIXED_TOKEN: u128 = 0x75a1_0000_0000_0000_0000_0000_0000_0001;

    /// A second writer, with a different `client_id`, that commits in between.
    const COMPETITOR: &str = "b75i-competitor";

    type FredQueue = Arc<Mutex<VecDeque<FredError>>>;

    /// A queue of fred `Timeout`s — "the request went out and no reply came
    /// back", which is the shape a lost reply takes on the wire.
    fn timeouts(n: usize) -> FredQueue {
        Arc::new(Mutex::new(
            (0..n)
                .map(|_| FredError::new(FredErrorKind::Timeout, "b75i: reply lost"))
                .collect(),
        ))
    }

    fn tokens(items: &[u128]) -> Arc<Mutex<VecDeque<u128>>> {
        Arc::new(Mutex::new(items.iter().copied().collect()))
    }

    fn overrides(items: &[WatchedExecOverride]) -> WatchedExecOverrideQueue {
        Arc::new(Mutex::new(items.iter().copied().collect()))
    }

    /// Assert a seam queue drained. Several assertions below are exactly what
    /// an UNDISTURBED commit produces — that is the point, the lost reply must
    /// be invisible — so they are vacuous if the seam silently stopped firing.
    fn assert_drained<T: std::fmt::Debug>(queue: &Mutex<VecDeque<T>>, seam: &str) {
        let left = queue.lock().unwrap();
        assert!(
            left.is_empty(),
            "every queued {seam} entry must have been consumed; {left:?} left. An undrained \
             queue means the disturbance never happened and the assertions above describe an \
             ordinary commit"
        );
    }

    /// A delta carrying one `CreateSection`: takes the WATCH/MULTI/EXEC slow
    /// path under the `lua-watch-structural` backend (the default `lua`
    /// backend plans it on the structural cache instead), and is the shape a
    /// blind re-apply turns into a `DuplicateSection` refusal of a landed edit.
    fn section_delta() -> OwnedDelta {
        OwnedDelta {
            struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id: 1,
                section_name: "B75iSection".to_owned(),
                section_type: 1,
            })],
            ..Default::default()
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // T1 — the fast path's lost reply
    // ─────────────────────────────────────────────────────────────────────────

    /// The `EVALSHA` runs and commits, and its reply is lost (a fred
    /// `Timeout`). The retry must report THAT commit — the sequence and server
    /// timestamp it actually landed with — not apply the delta again at
    /// `seq + 1`.
    ///
    /// Pre-fix the transient error went straight back through the retry loop
    /// with no reconcile step, so the header advanced twice and subscribers saw
    /// two commits of one edit.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lua_commit_whose_reply_is_lost_is_reported_once_not_reapplied() {
        let fx = fixture("b75i-t1-lua").await;
        let lost = timeouts(1);
        let store = store(&fx, UpdateLayerBackend::Lua)
            .with_lua_post_eval_fred_error_injector_for_test(lost.clone());

        let (seq, server_ts) = update(&store, &fx, field_delta(0x31), WRITER).await.expect(
            "the commit landed, so the call must succeed. An error here tells the client \
                 its edit failed while the store holds it",
        );
        assert_drained(&lost, "post-EVALSHA lost reply");

        assert_eq!(seq, 2, "the reported sequence is the one that committed");
        assert_eq!(
            header_seq(&fx).await,
            2,
            "the header must advance exactly once. 3 is the double apply: the retry re-ran \
             the delta against a header the lost attempt had already bumped"
        );
        let payloads = assert_stream_commits(
            &fx,
            &[(2, WRITER)],
            "exactly one commit entry may exist — a 3-0 is the duplicate commit subscribers \
             would see",
        )
        .await;
        assert_eq!(
            server_ts.to_string(),
            payload_field(&payloads[0], 5),
            "the reported server timestamp must be the landed commit's, read from its record, \
             not a fresh one from the retry"
        );
        assert_eq!(
            (
                already_landed(&fx.metrics, "lua"),
                already_landed(&fx.metrics, "txn")
            ),
            (1, 0),
            "exactly one attempt found its own commit, on the Lua backend"
        );
    }

    /// The retry's own reply can be lost as well. Attempt 1 commits and loses
    /// its reply; attempt 2 matches the token, answers `ALREADY_LANDED`, and
    /// loses THAT reply; attempt 3 matches again and is heard.
    ///
    /// The commit is still one commit, and the counter reads 1: it counts the
    /// matches the store actually decoded, and attempt 2's never reached it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lua_commit_whose_reply_is_lost_twice_still_commits_once() {
        let fx = fixture("b75i-t1-lua-twice").await;
        let lost = timeouts(2);
        let store = store(&fx, UpdateLayerBackend::Lua)
            .with_lua_post_eval_fred_error_injector_for_test(lost.clone());

        let (seq, server_ts) = update(&store, &fx, field_delta(0x32), WRITER)
            .await
            .expect("the commit landed, so the call must succeed");
        assert_drained(&lost, "post-EVALSHA lost reply");

        assert_eq!((seq, header_seq(&fx).await), (2, 2));
        let payloads = assert_stream_commits(
            &fx,
            &[(2, WRITER)],
            "a lost ALREADY_LANDED reply must itself be safe to retry: still one commit",
        )
        .await;
        assert_eq!(server_ts.to_string(), payload_field(&payloads[0], 5));
        assert_eq!(
            already_landed(&fx.metrics, "lua"),
            1,
            "the counter counts matches the store decoded; attempt 2's match was replaced by \
             the injected error before it was decoded"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // T2 — the slow path's in-doubt EXEC whose reconcile cannot tell
    // ─────────────────────────────────────────────────────────────────────────

    /// The `EXEC` runs and commits, its reply is lost (`ReplaceOutcomeAfterExec`
    /// hands the site `InDoubt`), and the reconciling `XRANGE` fails too, so
    /// the attempt answers the retriable `Aborted`. The next attempt's step-2a
    /// check must find the record the `MULTI` wrote and report the commit.
    ///
    /// Pre-fix the retry re-applied the delta. Its `CreateSection` then met the
    /// section the landed commit had created and the call failed
    /// `DuplicateSection` — the client was told its edit was refused when it
    /// had landed.
    ///
    /// On `lua-watch-structural`: the in-doubt `EXEC` and its reconcile are
    /// the WATCH path's. The structural route's lost reply is
    /// `structural_dispatch_test.rs`'s.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn txn_landed_exec_whose_reconcile_read_fails_is_reported_by_the_retry() {
        let fx = fixture("b75i-t2-txn").await;
        let exec = overrides(&[WatchedExecOverride::ReplaceOutcomeAfterExec]);
        let reconcile = timeouts(1);
        let store = store(&fx, UpdateLayerBackend::LuaWatchStructural)
            .with_watched_exec_override_for_test(exec.clone())
            .with_reconcile_xrange_fred_error_injector_for_test(reconcile.clone());

        let (seq, server_ts) = update(&store, &fx, section_delta(), WRITER).await.expect(
            "the commit landed, so the retry must report it. A DuplicateSection here is the \
                 retry re-applying the CreateSection the landed commit already made",
        );
        assert_drained(&exec, "watched-EXEC override");
        assert_drained(&reconcile, "reconcile XRANGE");

        assert_eq!(seq, 2, "the reported sequence is the one that committed");
        assert_eq!(header_seq(&fx).await, 2, "the header must advance once");
        let payloads = assert_stream_commits(&fx, &[(2, WRITER)], "exactly one commit entry").await;
        assert_eq!(
            server_ts.to_string(),
            payload_field(&payloads[0], 5),
            "the reported server timestamp must be the one the landed MULTI recorded"
        );
        assert_eq!(
            (
                already_landed(&fx.metrics, "txn"),
                already_landed(&fx.metrics, "lua")
            ),
            (1, 0),
            "exactly one attempt found its own commit, on the txn backend"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // T3 — another writer lands between the lost reply and the retry
    // ─────────────────────────────────────────────────────────────────────────

    /// Attempt 1 commits at seq 2 and loses its reply (the T2 staging). Inside
    /// attempt 2's WATCH window — after attempt 1 landed and before attempt 2
    /// reads anything — a DIFFERENT `client_id` commits at seq 3. Attempt 2
    /// must still report its own commit at seq 2.
    ///
    /// This is what separates a per-`client_id` record from a header-level
    /// "last token" field: the competitor's commit overwrites the header, and
    /// a header-only token would no longer match, so the retry would re-apply.
    ///
    /// The writer runs on `lua-watch-structural`: the staging is the WATCH
    /// path's in-doubt `EXEC` and its WATCH-window hook.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn txn_retry_reports_its_own_commit_after_another_writer_lands_in_between() {
        let fx = fixture("b75i-t3-txn").await;

        let competitor = Arc::new(store(&fx, UpdateLayerBackend::Lua));
        let (session_id, layer, object_id) = (fx.session_id.clone(), fx.layer, fx.object_id);
        let invocations = Arc::new(AtomicUsize::new(0));
        let hook_invocations = invocations.clone();
        let hook: WatchWindowHook = Arc::new(move |_client: &Client| {
            let competitor = competitor.clone();
            let session_id = session_id.clone();
            let n = hook_invocations.fetch_add(1, Ordering::SeqCst) + 1;
            async move {
                // Invocation 1 is the attempt whose reply is lost; land the
                // competitor at the head of the retry.
                if n == 2 {
                    let (seq, _) = competitor
                        .update_layer_relayed(
                            &session_id,
                            layer,
                            object_id,
                            Some(field_delta(0x33)),
                            COMPETITOR,
                            WRITER_TS,
                        )
                        .await
                        .expect("the competing commit must succeed — it is uncontended");
                    assert_eq!(seq, 3, "the competitor lands after the writer's seq 2");
                }
            }
            .boxed()
        });

        let exec = overrides(&[WatchedExecOverride::ReplaceOutcomeAfterExec]);
        let reconcile = timeouts(1);
        let disturbed = store(&fx, UpdateLayerBackend::LuaWatchStructural)
            .with_watched_exec_override_for_test(exec.clone())
            .with_reconcile_xrange_fred_error_injector_for_test(reconcile.clone())
            .with_watch_window_hook_for_test(hook);

        let (seq, server_ts) = update(&disturbed, &fx, section_delta(), WRITER)
            .await
            .expect("the writer's commit landed, so its call must succeed");
        assert_drained(&exec, "watched-EXEC override");
        assert_drained(&reconcile, "reconcile XRANGE");
        assert_eq!(
            invocations.load(Ordering::SeqCst),
            2,
            "precondition: exactly two attempts, the second preceded by the competitor's \
             commit. Any other count means the interleaving this test is about never happened"
        );

        assert_eq!(
            seq, 2,
            "the call must report its OWN commit at seq 2, not the competitor's 3 and not a \
             re-apply at 4"
        );
        assert_eq!(header_seq(&fx).await, 3, "two commits, one per writer");
        let payloads = assert_stream_commits(
            &fx,
            &[(2, WRITER), (3, COMPETITOR)],
            "one commit per writer; a 4-0 is the writer's delta applied twice",
        )
        .await;
        assert_eq!(server_ts.to_string(), payload_field(&payloads[0], 5));
        assert_eq!(already_landed(&fx.metrics, "txn"), 1);
    }

    /// The fast-path twin. There is no hook between Lua attempts, so the
    /// retry is staged with the fixed-token seam: two calls handed the SAME
    /// token are, to the store, one call's two attempts — the second carries
    /// exactly what the first call's own retry would carry. Between them a
    /// different `client_id` commits.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lua_retry_reports_its_own_commit_after_another_writer_lands_in_between() {
        let fx = fixture("b75i-t3-lua").await;
        let shared = tokens(&[FIXED_TOKEN, FIXED_TOKEN]);
        let writer =
            store(&fx, UpdateLayerBackend::Lua).with_commit_tokens_for_test(shared.clone());
        let competitor = store(&fx, UpdateLayerBackend::Lua);

        let landed = update(&writer, &fx, field_delta(0x34), WRITER)
            .await
            .expect("attempt 1 must commit");
        assert_eq!(landed.0, 2);
        let (competitor_seq, _) = update(&competitor, &fx, field_delta(0x35), COMPETITOR)
            .await
            .expect("the competing commit must succeed");
        assert_eq!(competitor_seq, 3);

        let retried = update(&writer, &fx, field_delta(0x34), WRITER)
            .await
            .expect("the retry of a landed commit must succeed");
        assert_drained(&shared, "fixed commit token");

        assert_eq!(
            retried, landed,
            "the retry must report the landed commit's (seq, server_ts) verbatim"
        );
        assert_eq!(header_seq(&fx).await, 3);
        assert_stream_commits(
            &fx,
            &[(2, WRITER), (3, COMPETITOR)],
            "one commit per writer; a 4-0 is the writer's delta applied twice",
        )
        .await;
        assert_eq!(already_landed(&fx.metrics, "lua"), 1);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // A record carrying this call's token with a corrupt tail
    // ─────────────────────────────────────────────────────────────────────────

    /// A commit-token record that carries THIS call's token but whose
    /// `<seq>:<server_ts>` tail does not parse is refused `Internal` on both
    /// backends, with identical text, and nothing is written.
    ///
    /// Neither backend writes that shape, so it means the key was edited out of
    /// band. Both easy answers are wrong: applying the delta risks the double
    /// commit the record exists to stop, and reporting a commit means guessing
    /// a sequence. The record does match the token, so it is not "another
    /// call's" either. Each backend gets its own call, handed the planted
    /// token through the fixed-token seam.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn commit_token_record_with_this_calls_token_and_a_corrupt_tail_is_refused_internal() {
        let fx = fixture("b75i-corrupt-record").await;
        let wtok_key = fx
            .tv
            .key_schema()
            .obj_write_token(&fx.object_id.to_string());
        let corrupt = format!("{FIXED_TOKEN:032x}:not-a-seq:x");
        let _: () = fx
            .tv
            .client()
            .pool
            .hset(&wtok_key, [(WRITER, corrupt.as_str())])
            .await
            .expect("planting the corrupt record must succeed");

        let mut messages = Vec::new();
        for backend in [UpdateLayerBackend::Lua, UpdateLayerBackend::Txn] {
            let store = store(&fx, backend).with_commit_tokens_for_test(tokens(&[FIXED_TOKEN]));
            match update(&store, &fx, field_delta(0x51), WRITER).await {
                Err(ServiceError::Internal(msg)) => messages.push(msg),
                other => panic!(
                    "{backend:?}: a record with this call's token and an unparseable tail must be \
                     refused Internal — Ok means a guessed or re-applied commit, a retriable \
                     error means a retry against the same bytes; got {other:?}"
                ),
            }
        }
        assert_eq!(
            messages[0], messages[1],
            "both backends parse the record with the same function and must refuse identically"
        );

        assert_eq!(header_seq(&fx).await, 1, "nothing may have committed");
        assert_stream_commits(&fx, &[], "neither refusal may have published an entry").await;
        let still: Option<String> = fx
            .tv
            .client()
            .pool
            .hget(&wtok_key, WRITER)
            .await
            .expect("HGET commit-token record must succeed");
        assert_eq!(
            still.as_deref(),
            Some(corrupt.as_str()),
            "the refusal must leave the record as it found it"
        );
        for backend_label in ["lua", "txn"] {
            assert_eq!(
                already_landed(&fx.metrics, backend_label),
                0,
                "a corrupt record is not a landed commit (backend={backend_label})"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // T8 — a token match after `superseded=1`
    // ─────────────────────────────────────────────────────────────────────────

    /// A commit lands, an overwrite then sets `superseded=1` on the object,
    /// and the call's retry arrives. The token check runs ahead of the
    /// superseded guard, so the retry reports the commit that landed — OK,
    /// with the recorded sequence — rather than `FailedPrecondition` for an
    /// edit that is in the store. A FRESH call afterwards is still refused,
    /// which is what proves the guard is live and the OK is the token's doing.
    async fn token_match_after_superseded_reports_the_landed_commit(
        backend: UpdateLayerBackend,
        backend_label: &str,
        layer: &'static str,
    ) {
        let fx = fixture(layer).await;
        // Two entries: the commit and its retry. The third call below finds the
        // queue empty and mints a random token, as production does.
        let shared = tokens(&[FIXED_TOKEN, FIXED_TOKEN]);
        let store = store(&fx, backend).with_commit_tokens_for_test(shared.clone());

        let landed = update(&store, &fx, field_delta(0x41), WRITER)
            .await
            .expect("the commit must land");
        let _: () = fx
            .tv
            .client()
            .pool
            .hset(
                fx.tv.key_schema().obj_header(&fx.object_id.to_string()),
                [("superseded", "1")],
            )
            .await
            .expect("HSET superseded must succeed");

        let retried = update(&store, &fx, field_delta(0x41), WRITER).await.expect(
            "the commit landed before the overwrite, so its retry must be told so. \
                 FailedPrecondition here reports a landed edit as refused",
        );
        assert_drained(&shared, "fixed commit token");
        assert_eq!(
            retried, landed,
            "the retry must report the recorded (seq, server_ts)"
        );

        let fresh = update(&store, &fx, field_delta(0x42), WRITER).await;
        assert!(
            matches!(fresh, Err(ServiceError::FailedPrecondition(_))),
            "control: a new call against the superseded object must still be refused; got \
             {fresh:?}"
        );

        let hdr = header(&fx).await;
        assert_eq!(
            (
                hdr.get("seq").map(String::as_str),
                hdr.get("superseded").map(String::as_str)
            ),
            (Some("2"), Some("1")),
            "the retry wrote nothing: the header is still the landed commit's, still superseded"
        );
        assert_stream_commits(&fx, &[(2, WRITER)], "only the landed commit's entry").await;
        assert_eq!(already_landed(&fx.metrics, backend_label), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lua_token_match_after_superseded_reports_the_landed_commit() {
        token_match_after_superseded_reports_the_landed_commit(
            UpdateLayerBackend::Lua,
            "lua",
            "b75i-t8-lua",
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn txn_token_match_after_superseded_reports_the_landed_commit() {
        token_match_after_superseded_reports_the_landed_commit(
            UpdateLayerBackend::Txn,
            "txn",
            "b75i-t8-txn",
        )
        .await;
    }
}
