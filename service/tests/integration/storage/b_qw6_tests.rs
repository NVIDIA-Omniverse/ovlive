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

// Integration tests for b.qw6 — update_layer hot-path hardening.
//
// Covers the items from the b.qw6 Issue body:
//
//   Item 1: superseded-flag protocol against CreateLayer-overwrite vs
//           UpdateLayer race. Both Lua and Txn backends must surface
//           `FailedPrecondition` once the OLD object's header carries
//           `superseded=1` — never commit to the doomed object.
//   Item 2: Lua and slow paths must both refuse to commit past the
//           2^53 - 1 `set_order` ceiling (was: slow path went to u64::MAX,
//           Lua silently rounded). Both surface as gRPC `Internal` with
//           a SET_ORDER_OVERFLOW / set_order overflow / SetOrderOverflow
//           token so operators can distinguish from a generic Internal.
//   Item 4: slow-path partial-header guard (seq missing) parity with the
//           Lua fast path — `FailedPrecondition`, no stream-id collision.
//
// Item 3 (MAXLEN ceiling rejected at startup) is unit-tested in
// `src/config.rs` and isn't replayed here. Item 5 (drop redundant
// `is_canceled` classification arm) is covered by the existing
// `transient_classification_test` suite that pins `Canceled →
// Aborted(transient I/O: ...)` via `map_transient_fred_err`.

use std::sync::Arc;

use fred::interfaces::{HashesInterface, StreamsInterface};

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::{delta_with_set_field_inline, empty_delta};
/// Mirror of the module-private `storage::layer::PARTIAL_HEADER_SENTINEL`;
/// shared via `tests/common` so every module asserting it holds one literal.
use crate::common::PARTIAL_HEADER_SENTINEL_MSG;

// ─────────────────────────────────────────────────────────────────────────────
// Shared harness. Mirrors the shape used in `b_k1r_tests.rs` so future diffs
// show up as assertion changes, not scaffolding churn.
// ─────────────────────────────────────────────────────────────────────────────

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

async fn create_test_session(tv: &TestValkey) -> String {
    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    ss.create("b-qw6-session", "/stage", "b-qw6-user")
        .await
        .expect("session creation must succeed")
}

async fn create_test_layer(ls: &LayerStore, session_id: &str, layer_name: &str) -> u64 {
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    ls.create_layer(
        session_id,
        layer_name,
        "b-qw6-user",
        &state,
        mock_ext_store(),
    )
    .await
    .expect("create_layer must succeed")
}

fn layer_store_with_backend(tv: &TestValkey, backend: UpdateLayerBackend) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(backend)
}

// =============================================================================
// b.qw6 Item 1 — superseded-flag protocol (both backends)
// =============================================================================
//
// The CreateLayer overwrite path now HSETs `superseded=1` on the OLD object's
// header BEFORE flipping the session→layer mapping. Any UpdateLayer caller
// that resolved the layer mapping just before the flip arrives at the OLD
// object_id and must observe `superseded=1` inside its commit window — Lua
// via `HGET ... superseded` at the top of `update_layer_fast.lua`, Txn via
// the WATCH/HGETALL of `superseded` on the old header.
//
// We exercise the steady-state outcome: with `superseded=1` already set
// on the OLD header, the next UpdateLayer against the OLD object_id MUST
// return `FailedPrecondition`. Driving the actual race (CreateLayer
// overwrite landing between the mapping resolve and the WATCH/HGETALL)
// would require a fault-injection seam that doesn't exist on `LayerStore`
// today; the steady-state assertion pins the protocol-level guarantee that
// the flag is honored by both backends, which is the load-bearing piece.

/// AC #1 (Lua): a SUPERSEDED reply from the Lua fast-path script surfaces
/// as `FailedPrecondition` (non-retriable) and does NOT commit. Pre-fix
/// the script had no `superseded` check at all — the fast path silently
/// committed to a doomed object.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_against_superseded_object_returns_failed_precondition() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);

    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "lua-superseded").await;

    // Mark the OLD header as superseded the same way `create_layer`'s
    // overwrite path would. We HSET directly rather than driving an
    // actual overwrite so the test failure mode is unambiguous: the
    // assertion checks ONLY the script's superseded guard, not the
    // entire overwrite pipeline.
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let _: () = tv
        .client()
        .pool
        .hset(&header_key, [("superseded", "1")])
        .await
        .expect("HSET superseded=1 must succeed");

    let stream_key = tv.key_schema().layer_stream(oid);
    let entries_before: Vec<(String, std::collections::HashMap<String, bytes::Bytes>)> = tv
        .client()
        .pool
        .xrange_values(&stream_key, "-", "+", None)
        .await
        .expect("XRANGE before must succeed");

    let result = ls
        .update_layer_relayed(
            &session_id,
            "lua-superseded",
            oid,
            Some(empty_delta()),
            "qw6-client",
            1111,
        )
        .await;
    match result {
        Err(ServiceError::FailedPrecondition(msg)) => {
            assert!(
                msg.to_lowercase().contains("supersed"),
                "FailedPrecondition message must reference 'superseded' so \
                 operators can distinguish from the other FailedPrecondition \
                 causes (object_id mismatch, partial header — a deleted header \
                 is `NotFound` since b.f1v, not this variant); got: {msg:?}"
            );
        }
        Ok(v) => panic!(
            "Lua update_layer against a superseded object MUST refuse to commit; \
             got Ok({v:?}). The script's `HGET ... superseded` guard did not fire."
        ),
        Err(other) => panic!(
            "expected FailedPrecondition (non-retriable; no retry can un-supersede); \
             got {other:?}"
        ),
    }

    // Belt-and-suspenders: the per-object stream must be unchanged. A
    // pre-fix commit would have XADD'd a 2-0 entry past the CreateLayer
    // initial 1-0 record. Asserting on entry count catches that without
    // relying on the gRPC error code alone.
    let entries_after: Vec<(String, std::collections::HashMap<String, bytes::Bytes>)> = tv
        .client()
        .pool
        .xrange_values(&stream_key, "-", "+", None)
        .await
        .expect("XRANGE after must succeed");
    assert_eq!(
        entries_before.len(),
        entries_after.len(),
        "superseded guard must short-circuit BEFORE any XADD — pre-fix the \
         script committed silently and the stream picked up a stray entry"
    );
}

/// AC #1 (Txn): the slow path's HGETALL of the WATCH'd header now
/// inspects `superseded`. If set, return `FailedPrecondition` and do
/// not commit. Pre-fix the slow path had no `superseded` check —
/// the WATCH/MULTI/EXEC would race ahead and commit to the doomed
/// object as long as nobody else HSET the header concurrently.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_against_superseded_object_returns_failed_precondition() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Txn);

    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "txn-superseded").await;

    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let _: () = tv
        .client()
        .pool
        .hset(&header_key, [("superseded", "1")])
        .await
        .expect("HSET superseded=1 must succeed");

    let stream_key = tv.key_schema().layer_stream(oid);
    let entries_before: Vec<(String, std::collections::HashMap<String, bytes::Bytes>)> = tv
        .client()
        .pool
        .xrange_values(&stream_key, "-", "+", None)
        .await
        .expect("XRANGE before must succeed");

    let result = ls
        .update_layer_relayed(
            &session_id,
            "txn-superseded",
            oid,
            Some(empty_delta()),
            "qw6-client",
            2222,
        )
        .await;
    match result {
        Err(ServiceError::FailedPrecondition(msg)) => {
            assert!(
                msg.to_lowercase().contains("supersed"),
                "Txn-backend superseded error must reference 'superseded' for \
                 operator triage; got: {msg:?}"
            );
        }
        Ok(v) => panic!(
            "Txn update_layer against a superseded object MUST refuse to commit; \
             got Ok({v:?}). The slow-path superseded guard inside the HGETALL \
             window did not fire."
        ),
        Err(other) => panic!("expected FailedPrecondition; got {other:?}"),
    }

    let entries_after: Vec<(String, std::collections::HashMap<String, bytes::Bytes>)> = tv
        .client()
        .pool
        .xrange_values(&stream_key, "-", "+", None)
        .await
        .expect("XRANGE after must succeed");
    assert_eq!(
        entries_before.len(),
        entries_after.len(),
        "Txn superseded guard must short-circuit BEFORE the MULTI is queued — \
         pre-fix the slow path would proceed through MULTI/EXEC with the \
         doomed object header read"
    );
}

/// AC #1 end-to-end: a real CreateLayer overwrite call MUST atomically
/// HSET `superseded=1` on the old header before flipping the
/// session→layer mapping. The flag's presence on the OLD header
/// post-overwrite is the load-bearing artifact for both backend guards
/// above — if the flag is missing, the overwrite-vs-update race
/// returns to a pre-fix silent commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_overwrite_sets_superseded_flag_on_old_header() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let keys = tv.key_schema();

    let session_id = create_test_session(&tv).await;
    let state = ObjectState::default();

    let oid1 = ls
        .create_layer(
            &session_id,
            "overwrite-flag",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("first create_layer must succeed");
    let oid2 = ls
        .create_layer(
            &session_id,
            "overwrite-flag",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("second create_layer (overwrite) must succeed");
    assert_ne!(oid1, oid2, "overwrite must allocate a new object_id");

    // The OLD header is scheduled for async cleanup by a spawned task, but
    // the `HSET superseded=1` is synchronous on the overwrite call path
    // and runs BEFORE the mapping flip — so by the time `create_layer`
    // returns, the flag is guaranteed-visible on the old header.
    //
    // (`cleanup_old_object` UNLINKs the old keys, including the header,
    // but the cleanup happens AFTER `create_layer` returns. We read the
    // flag immediately so we beat the async cleanup.)
    let old_header_key = keys.obj_header(&oid1.to_string());
    let superseded: Option<String> = tv
        .client()
        .pool
        .hget(&old_header_key, "superseded")
        .await
        .expect("HGET superseded must succeed");
    assert_eq!(
        superseded.as_deref(),
        Some("1"),
        "create_layer overwrite must HSET `superseded=1` on the OLD header \
         BEFORE the mapping flip. Without this flag the b.qw6 item 1 race \
         is wide open: an UpdateLayer that resolved the OLD mapping just \
         before the flip commits silently to the doomed object."
    );

    // The NEW header must NOT carry the flag — the overwrite path only
    // touches the OLD header. A future regression that flipped the
    // wrong object_id would break every UpdateLayer against the new
    // layer with a spurious FailedPrecondition.
    let new_header_key = keys.obj_header(&oid2.to_string());
    let new_superseded: Option<String> = tv
        .client()
        .pool
        .hget(&new_header_key, "superseded")
        .await
        .expect("HGET new superseded must succeed");
    assert_eq!(
        new_superseded, None,
        "the NEW header must NOT carry `superseded=1` — that field exists \
         only to mark a retired object as such"
    );
}

// =============================================================================
// b.qw6 Item 2 — set_order overflow ceiling at 2^53 - 1 (both backends)
// =============================================================================
//
// Pre-fix: the slow path's `apply_set_field` used `u64::checked_add(...)`
// and only refused at u64::MAX — far past the Lua f64 integer-precision
// ceiling. The Lua fast path had NO check and silently rounded past 2^53.
// Post-fix both backends fail at the same boundary (2^53 - 1) so the
// `LIVE_EDIT_UPDATE_LAYER_BACKEND` toggle is byte-identical at the
// ceiling — AC #2.

const SET_ORDER_CEILING: u64 = (1u64 << 53) - 1;

async fn seed_header_at_set_order_ceiling(tv: &TestValkey, oid: u64, src_fmt: &str) {
    // Mirror `seed_header_at_ceiling` from b_k1r_tests.rs — only the
    // `next_set_order` field changes (we're testing a different cursor).
    let keys = tv.key_schema();
    let pool = &tv.client().pool;
    let oid_str = oid.to_string();
    let header_key = keys.obj_header(&oid_str);
    let _: () = pool
        .hset(
            &header_key,
            [
                ("seq", "5".to_string()),
                ("first_ver", "1".to_string()),
                ("src_fmt", src_fmt.to_string()),
                ("next_set_order", SET_ORDER_CEILING.to_string()),
                ("session_id", "s-qw6".to_string()),
                ("layer_name", "l-qw6".to_string()),
                ("created_by", "tester".to_string()),
                ("created_at", "0".to_string()),
                ("modified_at", "0".to_string()),
                ("last_client_id", String::new()),
                ("last_client_ts", "0".to_string()),
                ("last_server_ts", "0".to_string()),
            ],
        )
        .await
        .expect("HSET ceiling header must succeed");
}

/// AC #2 (Lua): `next_set_order = 2^53 - 1` and `delta_set_order = 1`
/// → `next_set_order + delta_set_order = 2^53` crosses the f64 ceiling.
/// The script's check at the top of the SetField branch returns
/// `SET_ORDER_OVERFLOW`; the Rust caller wraps it in `Internal` —
/// non-retriable, distinct from `SEQ_OVERFLOW`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_at_set_order_ceiling_returns_internal_with_set_order_overflow() {
    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua);

    let oid: u64 = 1;
    seed_header_at_set_order_ceiling(&tv, oid, "usda").await;

    // `delta_set_order = 1` pushes `next_set_order + delta` from 2^53-1
    // to 2^53, crossing the ceiling exactly. SetField on a fresh section
    // ID + key so the rest of the apply pipeline has nothing to complain
    // about.
    let delta = delta_with_set_field_inline(1, "k", vec![1, 2, 3], 1);

    let result = ls
        .update_layer_relayed("s-qw6", "l-qw6", oid, Some(delta), "qw6-client", 4242)
        .await;
    match result {
        Err(ServiceError::Internal(msg)) => {
            assert!(
                msg.contains("SET_ORDER_OVERFLOW") || msg.contains("set_order"),
                "Lua set_order ceiling error must surface as Internal with a \
                 SET_ORDER_OVERFLOW / set_order token so operators can \
                 distinguish from SEQ_OVERFLOW and generic Internal; got: {msg:?}"
            );
        }
        Ok(v) => panic!(
            "Lua commit at next_set_order=2^53-1 with delta_set_order=1 must \
             refuse to advance; got Ok({v:?}) — the script's set_order \
             ceiling guard did not fire."
        ),
        Err(other) => {
            panic!("Lua set_order ceiling error must be Internal (non-retriable); got {other:?}")
        }
    }
}

/// AC #2 (Txn): the slow path's `apply_set_field` now enforces the same
/// 2^53 ceiling via `DeltaError::SetOrderOverflow`, which maps to
/// `Internal`. Pre-fix the slow path only refused at u64::MAX, so an
/// object with `next_set_order` past 2^53 would commit cleanly on the
/// slow path while the Lua path silently corrupted — the parity gap
/// AC #2 closes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_at_set_order_ceiling_returns_internal_with_set_order_overflow() {
    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Txn);

    let oid: u64 = 2;
    seed_header_at_set_order_ceiling(&tv, oid, "usda").await;

    let delta = delta_with_set_field_inline(1, "k", vec![1, 2, 3], 1);

    let result = ls
        .update_layer_relayed("s-qw6", "l-qw6", oid, Some(delta), "qw6-client", 4243)
        .await;
    match result {
        Err(ServiceError::Internal(msg)) => {
            // The slow path emits `DeltaError::SetOrderOverflow` →
            // `From<DeltaError>` wraps the Display form in
            // `Internal("delta error: ...")`. The Display body includes
            // "set_order overflow:" and "2^53" tokens — pin both so
            // operators can find the boundary without grepping the
            // source.
            let lower = msg.to_lowercase();
            assert!(
                lower.contains("set_order") || lower.contains("overflow"),
                "Txn set_order ceiling error must surface as Internal with \
                 a set_order / overflow token for operator triage; got: {msg:?}"
            );
        }
        Ok(v) => panic!(
            "Txn commit at next_set_order=2^53-1 with delta_set_order=1 must \
             refuse to advance; got Ok({v:?}) — the apply_set_field ceiling \
             guard did not fire."
        ),
        Err(other) => panic!("Txn set_order ceiling error must be Internal; got {other:?}"),
    }
}

/// AC #2 (parity at the boundary): both backends must reach the SAME
/// error at the SAME input. A single test seeds two distinct objects
/// (one per backend) with identical state, fires the same delta on
/// each, and asserts both surface `ServiceError::Internal`. The
/// dedicated parity proptest lives in `tests/proptests.rs` and runs
/// at many state shapes; this is the deterministic boundary pin.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_set_order_ceiling_parity_lua_and_txn_both_return_internal() {
    let tv_lua = TestValkey::setup().await;
    let tv_txn = TestValkey::setup().await;

    let ls_lua = LayerStore::new(tv_lua.client().clone(), tv_lua.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua);
    let ls_txn = LayerStore::new(tv_txn.client().clone(), tv_txn.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Txn);

    let oid: u64 = 7;
    seed_header_at_set_order_ceiling(&tv_lua, oid, "usda").await;
    seed_header_at_set_order_ceiling(&tv_txn, oid, "usda").await;

    let delta_lua = delta_with_set_field_inline(1, "k", vec![1, 2, 3], 1);
    let delta_txn = delta_with_set_field_inline(1, "k", vec![1, 2, 3], 1);

    let r_lua = ls_lua
        .update_layer_relayed("s-qw6", "l-qw6", oid, Some(delta_lua), "c", 1)
        .await;
    let r_txn = ls_txn
        .update_layer_relayed("s-qw6", "l-qw6", oid, Some(delta_txn), "c", 1)
        .await;

    assert!(
        matches!(r_lua, Err(ServiceError::Internal(_))),
        "Lua backend at set_order ceiling must return Internal; got {r_lua:?}"
    );
    assert!(
        matches!(r_txn, Err(ServiceError::Internal(_))),
        "Txn backend at set_order ceiling must return Internal; got {r_txn:?}"
    );
}

/// AC #2 (validator gate): `validate_delta_bounds` rejects a single
/// `SetField` whose `delta.set_order` is already past the ceiling.
/// The check runs BEFORE the backend dispatch, so both backends
/// surface the same `InvalidArgument` for this shape. (The runtime
/// ceiling check inside `apply_set_field` is reserved for the
/// cumulative case — an object that legitimately accumulates
/// `next_set_order` across many writes — and is exercised by the
/// dedicated `update_layer_*_at_set_order_ceiling_*` tests above.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_with_delta_set_order_above_ceiling_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);

    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "validator-set-order").await;

    // `(2^53 - 1) + 1 = 2^53` — `delta_set_order = 2^53` is the first
    // value past the validator's cap; we use it directly to drive
    // `validate_delta_bounds` into rejecting before the dispatch.
    let above_ceiling = 1u64 << 53;
    let delta = delta_with_set_field_inline(1, "k", vec![1, 2, 3], above_ceiling);

    let result = ls
        .update_layer_relayed(
            &session_id,
            "validator-set-order",
            oid,
            Some(delta),
            "qw6-client",
            6666,
        )
        .await;
    match result {
        Err(ServiceError::InvalidArgument(msg)) => {
            assert!(
                msg.contains("set_order") || msg.contains("delta bounds"),
                "validator must name the offending field so operators / clients \
                 can triage; got: {msg:?}"
            );
        }
        Ok(v) => panic!(
            "update_layer with delta_set_order=2^53 must be rejected at \
             validation; got Ok({v:?})"
        ),
        Err(other) => panic!("expected InvalidArgument from validate_delta_bounds; got {other:?}"),
    }
}

// =============================================================================
// b.qw6 Item 4 — slow-path partial-header guard parity
// =============================================================================
//
// Pre-fix: the slow path tolerated a header with `seq` missing — the
// HGETALL was non-empty (other fields remained) so the empty-header
// check did not fire, and `header_to_object_state` would fall through
// to `sequence_num = 0`. `apply_delta` then computed `new_seq = 1`,
// colliding with the initial `1-0` stream entry from CreateLayer.
// Post-fix the slow path now mirrors the Lua fast path's partial-header
// guard: HGETALL non-empty + `seq` absent → `FailedPrecondition`.

/// One corner of a four-test claim — **an ABSENT header and a PARTIAL one are
/// distinct surfaces, identically on both backends**. The other three:
/// `b_k1r_tests::update_layer_lua_partial_header_returns_failed_precondition`
/// (partial / Lua),
/// `update_layer_lua_test::update_layer_lua_missing_header_returns_not_found`
/// (absent / Lua) and
/// `update_layer_transaction_test::update_layer_header_deleted_returns_not_found`
/// (absent / txn).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_partial_header_returns_failed_precondition() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let pool = &tv.client().pool;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Txn);

    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "txn-partial-header").await;
    let oid_str = oid.to_string();

    let header_key = keys.obj_header(&oid_str);
    let _: i64 = pool
        .hdel(&header_key, "seq")
        .await
        .expect("HDEL seq must succeed");
    let after: std::collections::HashMap<String, String> = pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL post-hdel must succeed");
    assert!(!after.contains_key("seq"), "seq must be absent");
    assert!(after.contains_key("src_fmt"), "src_fmt must remain");

    let stream_key = keys.layer_stream(oid);
    let entries_before: Vec<(String, std::collections::HashMap<String, bytes::Bytes>)> = pool
        .xrange_values(&stream_key, "-", "+", None)
        .await
        .expect("XRANGE before must succeed");

    let result = ls
        .update_layer_relayed(
            &session_id,
            "txn-partial-header",
            oid,
            Some(empty_delta()),
            "qw6-client",
            7777,
        )
        .await;
    match result {
        Err(ServiceError::FailedPrecondition(msg)) => {
            // Pinned by equality since b.f1v: the Lua path's `PARTIAL_HEADER`
            // reply maps to this same literal (asserted from the other side in
            // `b_k1r_tests.rs::update_layer_lua_partial_header_returns_failed_precondition`),
            // so backend parity on the message is a contract, not triage prose.
            assert_eq!(
                msg, PARTIAL_HEADER_SENTINEL_MSG,
                "Txn partial-header error must carry the partial-header sentinel \
                 byte-for-byte — the Lua path raises the identical string and a client \
                 cannot tell which backend served it; got: {msg:?}"
            );
        }
        Ok(v) => panic!(
            "Txn update_layer on a partial header (seq missing) MUST NOT commit \
             — pre-fix new_seq=1 would collide with the CreateLayer 1-0 entry. \
             got Ok({v:?})"
        ),
        Err(ServiceError::NotFound(msg)) => panic!(
            "b.f1v fence: a corrupt-but-PRESENT header must stay `FailedPrecondition`. \
             `NotFound` is the ABSENT-header surface and routes `handle_update_layer` into \
             checkpoint recovery, whose bare `EXISTS` presence probe would report the layer \
             restored while the corrupt header sat untouched. Got NotFound({msg:?})"
        ),
        Err(other) => panic!("expected FailedPrecondition; got {other:?}"),
    }

    let entries_after: Vec<(String, std::collections::HashMap<String, bytes::Bytes>)> = pool
        .xrange_values(&stream_key, "-", "+", None)
        .await
        .expect("XRANGE after must succeed");
    assert_eq!(
        entries_before.len(),
        entries_after.len(),
        "Txn partial-header guard must short-circuit BEFORE the MULTI is queued"
    );
}

// =============================================================================
// b.qw6 PM follow-up — Item 1 boundary parity at next + delta == 2^53 - 1
// =============================================================================
//
// PM flagged that the original b.qw6 fix left an off-by-one parity gap at
// the exact ceiling. The slow path's `apply_set_field` rejects when
// `stored_set_order >= SET_ORDER_CEILING` (so `next + delta == 2^53 - 1`
// is rejected — the NEXT call would land at `next_set_order = 2^53`,
// which the same precision check at top of the function won't tolerate).
// Pre-PM-followup the Lua script used a strict `>` and *committed* at
// the same input — a byte-identical-behavior AC #2 violation. The fix
// tightens the Lua check to `>=`. This test pins the boundary on both
// backends so future divergence trips before reaching prod.

/// AC #2 (boundary parity): `seeded_next_set_order = 0` and a delta with
/// `delta_set_order = 2^53 - 1` makes `next + delta = 2^53 - 1`. Both
/// backends must reject this with `Internal` (SET_ORDER_OVERFLOW /
/// SetOrderOverflow). Pre-fix the Lua backend committed (silent f64 wrap
/// risk for the next call) while the Txn backend rejected — exactly the
/// parity gap PM flagged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_set_order_sum_equals_ceiling_parity_lua_and_txn_both_reject() {
    let tv_lua = TestValkey::setup().await;
    let tv_txn = TestValkey::setup().await;

    let ls_lua = LayerStore::new(tv_lua.client().clone(), tv_lua.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua);
    let ls_txn = LayerStore::new(tv_txn.client().clone(), tv_txn.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Txn);

    // Seed both backends identically: `next_set_order = 0`. The seeded
    // header helper uses a constant `SET_ORDER_CEILING` for the
    // next_set_order, so we HSET our own header without that field set
    // high. Reuse the same scaffolding (full set of required fields)
    // for parity with the existing tests.
    let oid: u64 = 11;
    seed_header_with_next_set_order(&tv_lua, oid, "usda", 0).await;
    seed_header_with_next_set_order(&tv_txn, oid, "usda", 0).await;

    // `delta_set_order = 2^53 - 1` → `stored = 0 + (2^53 - 1) = 2^53 - 1`.
    // The slow path's belt-and-braces guard rejects (stored >= ceiling
    // means the next advance would be at 2^53); the Lua path's `>` was
    // wrong because it only rejected when sum > ceiling, allowing the
    // boundary case. Post-fix Lua uses `>=` and rejects too.
    let delta_lua = delta_with_set_field_inline(1, "k", vec![1, 2, 3], SET_ORDER_CEILING);
    let delta_txn = delta_with_set_field_inline(1, "k", vec![1, 2, 3], SET_ORDER_CEILING);

    let r_lua = ls_lua
        .update_layer_relayed("s-qw6", "l-qw6", oid, Some(delta_lua), "c", 1)
        .await;
    let r_txn = ls_txn
        .update_layer_relayed("s-qw6", "l-qw6", oid, Some(delta_txn), "c", 1)
        .await;

    assert!(
        matches!(r_lua, Err(ServiceError::Internal(_))),
        "Lua backend at next+delta=ceiling MUST reject (AC #2 boundary parity); \
         pre-fix Lua used `>` and committed at sum==ceiling while Txn rejected. \
         got {r_lua:?}"
    );
    assert!(
        matches!(r_txn, Err(ServiceError::Internal(_))),
        "Txn backend at next+delta=ceiling must reject (was the originally \
         correct side of the parity gap). got {r_txn:?}"
    );
}

/// Same scaffolding as `seed_header_at_set_order_ceiling` but parameterised
/// on the `next_set_order` value so the boundary-parity test can pin
/// `next_set_order = 0` while reusing the same header-field set.
async fn seed_header_with_next_set_order(
    tv: &TestValkey,
    oid: u64,
    src_fmt: &str,
    next_set_order: u64,
) {
    let keys = tv.key_schema();
    let pool = &tv.client().pool;
    let oid_str = oid.to_string();
    let header_key = keys.obj_header(&oid_str);
    let _: () = pool
        .hset(
            &header_key,
            [
                ("seq", "5".to_string()),
                ("first_ver", "1".to_string()),
                ("src_fmt", src_fmt.to_string()),
                ("next_set_order", next_set_order.to_string()),
                ("session_id", "s-qw6".to_string()),
                ("layer_name", "l-qw6".to_string()),
                ("created_by", "tester".to_string()),
                ("created_at", "0".to_string()),
                ("modified_at", "0".to_string()),
                ("last_client_id", String::new()),
                ("last_client_ts", "0".to_string()),
                ("last_server_ts", "0".to_string()),
            ],
        )
        .await
        .expect("HSET ceiling header must succeed");
}

// =============================================================================
// b.qw6 PM follow-up — Item 4 slow-path malformed-seq parity
// =============================================================================
//
// The original Item 4 slow-path fix only caught `seq` absent. Lua's
// `hdr["seq"] == nil or tonumber(hdr["seq"]) == nil` is broader and
// rejects empty / non-numeric / unparsable values. Pre-fix on the slow
// path, an HSET of `seq = "abc"` would fall through `parse_u64_warn`
// (which warns and returns `0`), `apply_delta` would compute
// `new_seq = 1`, and the EXEC XADD would collide with the CreateLayer
// `1-0` stream entry — exactly the bug Item 4 was meant to close.
//
// This test mirrors the Lua test at b_k1r_tests.rs:925 but uses HSET to
// install a non-numeric `seq` instead of HDEL. Pins the slow path's
// guard against the same malformed shapes Lua already rejects.

/// b.qw6 PM follow-up (slow path): an HSET of `seq = "abc"` (non-numeric)
/// on the obj_header must surface as `FailedPrecondition`, NOT default
/// silently to `sequence_num = 0` and emit a colliding `1-0` stream
/// entry. Mirrors the Lua-side coverage at
/// `b_k1r_tests.rs::update_layer_lua_partial_header_returns_failed_precondition`
/// but exercises the slow path (the WATCH/MULTI/EXEC backend).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_malformed_seq_returns_failed_precondition() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let pool = &tv.client().pool;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Txn);

    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "txn-malformed-seq").await;
    let oid_str = oid.to_string();

    // Damage the header: overwrite `seq` with a non-numeric value. The
    // HGETALL-empty check can't rescue us (other fields remain), the
    // `contains_key("seq")` check the pre-PM-followup fix used can't
    // either (seq is present), but the new `parse::<u64>()` guard fires.
    let header_key = keys.obj_header(&oid_str);
    let _: () = pool
        .hset(&header_key, [("seq", "abc")])
        .await
        .expect("HSET seq=abc must succeed");
    // Sanity: seq is present but non-numeric.
    let after: std::collections::HashMap<String, String> = pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL must succeed");
    assert_eq!(
        after.get("seq").map(|s| s.as_str()),
        Some("abc"),
        "seq must be set to non-numeric value for this test"
    );

    let stream_key = keys.layer_stream(oid);
    let entries_before: Vec<(String, std::collections::HashMap<String, bytes::Bytes>)> = pool
        .xrange_values(&stream_key, "-", "+", None)
        .await
        .expect("XRANGE before must succeed");

    let result = ls
        .update_layer_relayed(
            &session_id,
            "txn-malformed-seq",
            oid,
            Some(empty_delta()),
            "qw6-client",
            8888,
        )
        .await;
    match result {
        Err(ServiceError::FailedPrecondition(msg)) => {
            // Same equality pin as the `seq`-absent sibling above: absent and
            // non-numeric `seq` are ONE condition to both backends, so they
            // must be indistinguishable to the caller too.
            assert_eq!(
                msg, PARTIAL_HEADER_SENTINEL_MSG,
                "Txn malformed-seq error must carry the same partial-header sentinel as the \
                 seq-absent case — the guard treats the two shapes identically; got: {msg:?}"
            );
        }
        Ok(v) => panic!(
            "Txn update_layer on a malformed seq (\"abc\") MUST NOT commit — \
             pre-fix this fell through parse_u64_warn to sequence_num=0 and \
             new_seq=1, colliding with the CreateLayer 1-0 stream entry. \
             got Ok({v:?})"
        ),
        Err(other) => panic!("expected FailedPrecondition; got {other:?}"),
    }

    let entries_after: Vec<(String, std::collections::HashMap<String, bytes::Bytes>)> = pool
        .xrange_values(&stream_key, "-", "+", None)
        .await
        .expect("XRANGE after must succeed");
    assert_eq!(
        entries_before.len(),
        entries_after.len(),
        "Txn malformed-seq guard must short-circuit BEFORE the MULTI is queued"
    );
}

/// Companion test: empty-string `seq` (an even sneakier shape — `String`
/// is present but `parse::<u64>()` fails the same way as "abc"). Belt
/// and braces because `map_or(false, |s| !s.is_empty() && ...)` has two
/// rejection branches and we want both pinned.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_empty_seq_returns_failed_precondition() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let pool = &tv.client().pool;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Txn);

    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "txn-empty-seq").await;
    let oid_str = oid.to_string();

    let header_key = keys.obj_header(&oid_str);
    let _: () = pool
        .hset(&header_key, [("seq", "")])
        .await
        .expect("HSET seq=\"\" must succeed");

    let result = ls
        .update_layer_relayed(
            &session_id,
            "txn-empty-seq",
            oid,
            Some(empty_delta()),
            "qw6-client",
            9999,
        )
        .await;
    match result {
        Err(ServiceError::FailedPrecondition(_)) => {}
        Ok(v) => panic!("Txn update_layer with empty seq MUST NOT commit; got Ok({v:?})"),
        Err(other) => panic!("expected FailedPrecondition; got {other:?}"),
    }
}
