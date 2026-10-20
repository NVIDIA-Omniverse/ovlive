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

//! Structural mode keeps every write-safety guard the fast path keeps (plan
//! `b.dkt` Epic 5 Task 3; Spec SDD SR-10.4, script side).
//!
//! Each test is the structural-mode counterpart of a named fast-path test, so
//! the two modes read side by side. Commits are planned by the test-support
//! planner and sent through `commit_structural` with `structural_commit_test`'s
//! fixture (an `every_shape_state` object created through `create_layer`), or
//! hand-built on such a plan where the planner itself refuses first (the
//! ceilings) or the shape is one the engine does not emit; the replay goes
//! through a raw `EVALSHA` (`StructuralInvocation`).
//!
//! | Guard | Structural test here | Fast-path counterpart |
//! |---|---|---|
//! | superseded | `…against_a_superseded_object…` | `b_qw6_tests::update_layer_lua_against_superseded_object_returns_failed_precondition` |
//! | token before superseded | `…token_match_after_superseded…` | `b_75i_tests::lua_token_match_after_superseded_reports_the_landed_commit` |
//! | stream top, fail closed | `…stream_entry_at_the_next_seq…`, `…above_the_next_seq…`, `…wrong_typed_stream…` | `update_layer_lua_test::update_layer_lua_stream_entry_at_exact_next_seq_…`, `…_stale_stream_top_…` |
//! | ceilings | `…at_the_sequence_ceiling…`, `…set_order_at_the_ceiling…`, `…next_set_order_past_the_ceiling…`, `…next_set_order_at_the_ceiling_commits` | `b_k1r_tests::update_layer_lua_at_sequence_ceiling_…`, `b_qw6_tests::update_layer_lua_at_set_order_ceiling_…`, `update_layer_lua_test::update_layer_lua_set_order_overflow_on_second_record_writes_nothing` |
//! | `obj:chs` ⟺ lists, `chs_complete` | `…keep_obj_chs_equal_to_the_lists…` (marked, unmarked) | `chs_complete_tests::set_children_*`, `…marked_object_keeps_its_marker…`, `…unmarked_object_stays_unmarked…` |
//! | `obj:ext` counting | `…ext_index_…` (four transitions) | `b_qpf_tests`, `b_yda_tests` |
//! | `obj:wtok` and a replay | `…replayed_after_landing…` | `update_layer_lua_test::update_layer_fast_script_replayed_invocation_answers_the_landed_commit_without_writing` |
//! | operation ids: own-key match → start record → superseded → `STALE` / `CACHE_DIVERGED` | `operation_step_1b_order_answers_the_same_in_both_script_modes` (both modes, one identity, byte-equal replies) | `update_layer_lua_test::update_layer_fast_script_own_key_match_precedes_the_start_record_check`, `…_stale_start_record_is_overtaken_without_writing` |
//! | chained operations (b.pav): own-key match → superseded → PREDECESSOR → `FENCED` (and its wrong-typed fence key) → the release record's type → `STALE` / `CACHE_DIVERGED` | `predecessor_step_order_answers_the_same_in_both_script_modes` (both modes, one identity, byte-equal replies) | `update_layer_lua_test::update_layer_fast_script_predecessor_guard_commits_only_directly_after_the_predecessor`, `…_own_key_match_precedes_the_predecessor_check` |
//! | `FENCED`: a live lease refuses, nothing written, commits after release; the step-1b order under a live fence (`FENCED` after `SUPERSEDED`, ahead of `STALE` / `CACHE_DIVERGED`; `WRONGTYPE_KEY` for either fence key) | `structural_mode_under_a_real_lease_answers_fenced_writes_nothing_and_commits_after_release`, `step_1b_order_under_a_live_fence_answers_the_same_in_both_script_modes` (both modes) | `update_layer_lua_test::update_layer_fast_script_under_a_real_lease_answers_fenced_then_commits_after_release`, `…_reports_an_already_landed_commit_under_a_live_fence`, `…_refuses_superseded_before_fenced` |
//! | operation record parity | `operation_record_is_byte_identical_across_the_fast_structural_and_watch_commits`, `…_without_an_operation_keeps_the_per_call_shape…` | `update_layer_lua_test::update_layer_operation_record_is_identical_on_both_backends` |
//! | partial / absent header | `…partial_header…`, `…absent_header…` | `b_k1r_tests::update_layer_lua_partial_header_returns_failed_precondition`, `update_layer_lua_test::update_layer_lua_missing_header_returns_not_found` |
//! | `WRONGTYPE_KEY` per family | `…wrong_typed_key_of_every_written_family…` | `update_layer_lua_test` section 14, `b_459_tests` structural-path families |
//! | wrong-typed header | `…wrong_typed_header_is_refused_exactly_as_the_fast_path…` | (the fast path's own refusal, driven in the same test) |
//! | children-list heal / digest | `…children_list_under_a_*_parent_commits_and_heals`, `…digest_checked_children_list…` | `b_459_tests::txn_wrongtype_children_list_the_marked_index_does_not_name_is_healed_by_the_commit` |
//! | one value shape | `…rewrites_leave_exactly_one_value_shape…` | `b_zvd_tests` (`assert_exactly_one_value_shape`) |
//! | `STALE` under a racing fast commit | `…planned_before_a_fast_path_commit…` | (structural only: the fence) |
//! | delete over an old tombstone | `…deleted_recreated_and_deleted_again…` | (structural only: `TOMB` overwrites) |
//!
//! Every refusal is checked to write nothing against the object's whole
//! keyspace (`dump_object_keyspace` plus the full header and the external-value
//! pending set, the fixture's `keyspace`), or — where the header itself is
//! absent or wrong-typed — the dump and the pending set alone.
//!
//! `FENCED` (guard 5, the script's step 1d) is here in both modes, at script
//! level: on a real lease, and in the step-1b order under a live fence (the
//! `FENCED` section); its fast-mode twins are in `update_layer_lua_test.rs`
//! section 13b and the writer side (the wait, the cap, every commit path) in
//! `snapshot_fence_writer_test.rs`.
//!
//! Not here: the operation id end to end through
//! `update_layer` on the structural route (that dispatch is not wired yet; its
//! tests come with it), and
//! keyspace equality with the WATCH path (Task 4: the proptest in
//! `tests/proptests.rs`, `structural_watch_equivalence`, and the cases of a
//! drifted `obj:ext` in `structural_watch_equivalence_test`).

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use fred::interfaces::{HashesInterface, KeysInterface, ListInterface, StreamsInterface};
use oodf_ffi::{OwnedDelta, SetField, TimeSample as OodfTimeSample};

use live_edit::domain::delta::StorageMutation;
use live_edit::domain::types::{Field, FieldValue};
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::layer::{
    operation_request_digest, LayerStore, Operation, StartRecord, CHS_COMPLETE_MARKED,
    HEADER_FIELD_CHS_COMPLETE,
};
use live_edit::storage::lua::{
    encode_delta, is_fast_path, update_script_shared_keys, UpdateScriptMode,
    UPDATE_LAYER_FAST_SCRIPT, UPDATE_SCRIPT_SHARED_ARGV, UPDATE_SCRIPT_SHARED_KEYS,
};
use live_edit::storage::snapshot_fence::{
    self as fence, AcquireVerdict, LayerSize, ReleaseVerdict,
};
use live_edit::storage::structural::{CommitOutcome, StructuralCommit};
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::structural_commit::{
    dump_object_keyspace, CommitTokens, ObjectKeyspaceDump, PlannedCommit,
};

use crate::common::owned_delta::{delta_with_set_field_inline, delta_with_time_sample_inline};
use crate::storage::b_qpf_tests::{
    ext_hash, ext_hash_hex, ext_hash_vec, ext_ref_released, ext_store, obj_ext_sorted,
    pending_members, pre_store, EXT_SIZE, REASON_REWRITE, REASON_SURPLUS,
};
use crate::storage::b_zvd_tests::assert_exactly_one_value_shape;
use crate::storage::snapshot_read_atomic_test::{
    create_cmd, delete_cmd, indexed_parents, layer_store, listed_parents, structural, unmark, CUBE,
    DANGLING_PARENT, ROOT, SIBLING, WORLD,
};
use crate::storage::structural_commit_test::shared_argv;
use crate::storage::structural_commit_test::{
    assert_sent_without_landing, fixture, hand_built, hand_built_exact, operation, recorded_token,
    Fixture, ObserverCall, StructuralInvocation, CLIENT_TS, CUBE_FIELD, CUBE_SAMPLE_TIME,
    NEW_SECTION, OP_CALL_TOKEN, OP_ID, OP_PAYLOAD, OTHER_CALL_TOKEN, OTHER_NEW_SECTION,
    OTHER_OP_ID, OTHER_PAYLOAD, WORLD_FIELD, WORLD_SAMPLE_TIME, WRITER,
};

// ═════════════════════════════════════════════════════════════════════════════
// Contract strings and shared helpers
// ═════════════════════════════════════════════════════════════════════════════

/// The script's one sequence / `set_order` ceiling, `2^53 - 1` (its
/// `F64_INT_CEILING`).
const F64_INT_CEILING: u64 = (1u64 << 53) - 1;

/// The error texts `LayerStore::map_update_script_error` produces, restated
/// because the constants in `src/storage/layer.rs` are private or `pub(crate)`.
/// Every one is the same string the fast path's callers see.
const SUPERSEDED_SENTINEL_MSG: &str = "object superseded by a concurrent CreateLayer overwrite";
const EMPTY_HEADER_SENTINEL_MSG: &str = "object no longer exists (header deleted)";
const PARTIAL_HEADER_SENTINEL_MSG: &str = "object header partial (seq missing or malformed)";
const COLLISION_UPDATE_PREFIX: &str = "update_layer stream-id collision: STREAM_ID_COLLISION: ";
const WRONGTYPE_KEY_PREFIX: &str = "update_layer keyspace type error: WRONGTYPE_KEY: ";
const SEQ_OVERFLOW_PREFIX: &str = "update_layer sequence ceiling reached: SEQ_OVERFLOW";
const SET_ORDER_OVERFLOW_PREFIX: &str =
    "update_layer set_order ceiling reached: SET_ORDER_OVERFLOW";

/// The `keyspace_type_error` and `stream_id_collision` cells both modes tick.
const KEYSPACE_TYPE_ERROR_TOTAL: &str = "live_edit_keyspace_type_error_total";
const STREAM_ID_COLLISION_TOTAL: &str = "live_edit_stream_id_collision_total";

/// The fixture store with its own metrics registry, so absolute counts are
/// sound.
fn metered(fx: &Fixture, label: &str) -> (LayerStore, Arc<Metrics>) {
    let metrics = Arc::new(crate::common::metrics_scrape_only(label));
    (layer_store(&fx.tv).with_metrics(metrics.clone()), metrics)
}

/// One `mode` series of `metric`'s `{backend="lua", site="update"}` cell.
/// Panics when it is absent: both counters are zero-seeded per mode in
/// `Metrics::new`, so absence means the store under test carries no metrics
/// handle and the count is vacuous.
fn lua_update_mode_count(text: &str, metric: &str, mode: &str) -> u64 {
    sample_value(
        text,
        metric,
        &[("backend", "lua"), ("site", "update"), ("mode", mode)],
    )
    .unwrap_or_else(|| {
        panic!("{metric}{{backend=\"lua\",site=\"update\",mode=\"{mode}\"}} must be present")
    }) as u64
}

/// The `mode="structural"` series of `metric`'s `{backend="lua",
/// site="update"}` cell — the series every refusal here ticks, since each is
/// raised by the script's structural mode — after asserting the
/// `mode="fast"` series still reads 0. A refusal counted under the fast mode
/// fails here instead of adding up to the right total.
fn structural_mode_count(metrics: &Metrics, metric: &str) -> u64 {
    let text = render(metrics);
    assert_eq!(
        lua_update_mode_count(&text, metric, "fast"),
        0,
        "a structural-mode refusal must not tick {metric}'s mode=\"fast\" series"
    );
    lua_update_mode_count(&text, metric, "structural")
}

/// Replace `key` with a plain string, whatever it held.
async fn plant_string(fx: &Fixture, key: &str) {
    let pool = &fx.tv.client().pool;
    let _: u64 = pool
        .del(key)
        .await
        .unwrap_or_else(|e| panic!("DEL {key} must succeed: {e}"));
    let _: () = pool
        .set(key, "planted-by-an-out-of-band-writer", None, None, false)
        .await
        .unwrap_or_else(|e| panic!("SET {key} must succeed: {e}"));
}

/// The object's keyspace without the fixture's header read: the dump and the
/// external-value pending set. For the cases where the header itself is absent
/// or not a hash, which the fixture's `keyspace` cannot read.
async fn raw_keyspace(fx: &Fixture) -> (ObjectKeyspaceDump, Vec<String>) {
    (
        dump_object_keyspace(fx.tv.client(), fx.keys(), fx.oid, CommitTokens::Verbatim).await,
        pending_members(&fx.tv).await,
    )
}

/// Commit on `ls`, require an error with the commit sent and not landed and the
/// object's keyspace unchanged, and return the error.
async fn assert_refused_with_error(
    fx: &Fixture,
    ls: &LayerStore,
    commit: &StructuralCommit,
) -> ServiceError {
    let before = fx.keyspace().await;

    let (result, calls) = fx.commit_with(ls, commit).await;

    let err = match result {
        Err(e) => e,
        Ok(outcome) => panic!("the commit must be refused with an error; got {outcome:?}"),
    };
    assert_sent_without_landing(&calls, &format!("{err:?}"));
    assert_eq!(
        fx.keyspace().await,
        before,
        "a commit refused with {err:?} must write nothing"
    );
    err
}

/// [`assert_refused_with_error`] for a header that is absent or not a hash:
/// the keyspace compare is [`raw_keyspace`].
async fn assert_refused_with_error_raw(
    fx: &Fixture,
    ls: &LayerStore,
    commit: &StructuralCommit,
) -> ServiceError {
    let before = raw_keyspace(fx).await;

    let (result, calls) = fx.commit_with(ls, commit).await;

    let err = match result {
        Err(e) => e,
        Ok(outcome) => panic!("the commit must be refused with an error; got {outcome:?}"),
    };
    assert_sent_without_landing(&calls, &format!("{err:?}"));
    assert_eq!(
        raw_keyspace(fx).await,
        before,
        "a commit refused with {err:?} must write nothing"
    );
    err
}

/// Commit on `ls` and require `Committed` at `expected_seq + 1`.
async fn assert_committed_with(fx: &Fixture, ls: &LayerStore, commit: &StructuralCommit) {
    match fx.commit_with(ls, commit).await.0 {
        Ok(CommitOutcome::Committed { seq, .. }) => assert_eq!(
            seq,
            commit.expected_seq + 1,
            "a commit lands at expected_seq + 1"
        ),
        other => panic!("the commit must land; got {other:?}"),
    }
}

/// The plan of a section created under `WORLD`: `SEC`, `CHL` and `HDR`
/// records, touching `sec:*`, `secs`, `ch:{WORLD}` and `chs`.
fn create_lamp() -> OwnedDelta {
    structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")])
}

// ═════════════════════════════════════════════════════════════════════════════
// Superseded, and the token check ahead of it
// ═════════════════════════════════════════════════════════════════════════════

/// Mirrors `b_qw6_tests::update_layer_lua_against_superseded_object_returns_failed_precondition`:
/// a commit against an object an overwrite marked `superseded` is the fast
/// path's `FailedPrecondition(SUPERSEDED_SENTINEL)`, and writes nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_against_a_superseded_object_is_failed_precondition_and_writes_nothing() {
    let fx = fixture("structural-superseded").await;
    let commit = fx.plan(create_lamp(), "tok-superseded").await.commit;
    fx.set_header_column("superseded", "1").await;

    let err = assert_refused_with_error(&fx, &fx.ls, &commit).await;

    match err {
        ServiceError::FailedPrecondition(msg) => assert_eq!(msg, SUPERSEDED_SENTINEL_MSG),
        other => panic!("expected the superseded FailedPrecondition; got {other:?}"),
    }
}

/// Mirrors `b_75i_tests::lua_token_match_after_superseded_reports_the_landed_commit`:
/// a structural commit lands, an overwrite then marks the object superseded,
/// and the call's resend answers `AlreadyLanded` with the landed commit — the
/// token check runs ahead of the superseded guard — writing nothing. A fresh
/// commit is still refused, which is what proves the guard is live.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_token_match_after_superseded_is_already_landed() {
    let fx = fixture("structural-superseded-token").await;
    let commit = fx.plan(create_lamp(), "tok-landed").await.commit;
    let server_ts = fx.commit_landed(&commit).await;
    let seq = commit.expected_seq + 1;
    fx.set_header_column("superseded", "1").await;
    let before = fx.keyspace().await;

    let (resent, calls) = fx.commit(&commit).await;

    assert!(
        matches!(
            &resent,
            Ok(CommitOutcome::AlreadyLanded { seq: s, server_ts: t, .. })
                if *s == seq && *t == server_ts
        ),
        "the resend must report the landed commit, not refuse it as superseded; got {resent:?}"
    );
    assert!(
        matches!(
            calls.as_slice(),
            [
                ObserverCall::CommitSize { .. },
                ObserverCall::AlreadyLanded { .. }
            ]
        ),
        "got {calls:?}"
    );
    assert_eq!(fx.keyspace().await, before, "the resend writes nothing");
    let fresh = fx
        .plan(
            structural(vec![create_cmd(WORLD, OTHER_NEW_SECTION, "Desk")]),
            "tok-fresh",
        )
        .await
        .commit;
    let (fresh_result, _) = fx.commit(&fresh).await;
    assert!(
        matches!(
            &fresh_result,
            Err(ServiceError::FailedPrecondition(msg)) if msg == SUPERSEDED_SENTINEL_MSG
        ),
        "control: a new commit against the superseded object must still be refused; got \
         {fresh_result:?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Stream top: fail closed
// ═════════════════════════════════════════════════════════════════════════════

/// What occupies the object's stream before the commit.
#[derive(Debug, Clone, Copy)]
enum StreamObstruction {
    /// An entry at `{expected_seq + offset}-0`, written out of band.
    EntryAbove(u64),
    /// A string where the stream belongs.
    WrongType,
}

/// Mirrors `update_layer_lua_test`'s `assert_stale_stream_top_refuses_commit_whole`:
/// the commit's `XADD {expected_seq+1}-0` is refused, which is the
/// `STREAM_ID_COLLISION`-mapped `Internal` (the stream key named in it), counted
/// once on `stream_id_collision{lua, update, mode="structural"}` (never under
/// `mode="fast"`) and not as a type error, and the whole keyspace is unchanged.
async fn assert_stream_obstruction_refuses_the_commit_whole(
    layer: &'static str,
    obstruction: StreamObstruction,
) {
    let fx = fixture(layer).await;
    let (ls, metrics) = metered(&fx, layer);
    let commit = fx.plan(create_lamp(), "tok-stream-top").await.commit;
    let stream_key = fx.keys().layer_stream(fx.oid);
    match obstruction {
        StreamObstruction::EntryAbove(offset) => {
            let _: String = fx
                .tv
                .client()
                .pool
                .xadd(
                    &stream_key,
                    false,
                    None,
                    format!("{}-0", commit.expected_seq + offset),
                    vec![("payload", "out-of-band|planted-by-test")],
                )
                .await
                .expect("planting the out-of-band stream entry must succeed");
        }
        StreamObstruction::WrongType => plant_string(&fx, &stream_key).await,
    }

    let err = assert_refused_with_error(&fx, &ls, &commit).await;

    match err {
        ServiceError::Internal(msg) => assert!(
            msg.starts_with(&format!(
                "{COLLISION_UPDATE_PREFIX}XADD {}-0",
                commit.expected_seq + 1
            )),
            "{obstruction:?}: expected the stream-id collision Internal; got {msg:?}"
        ),
        other => {
            panic!("{obstruction:?}: expected the stream-id collision Internal; got {other:?}")
        }
    }
    assert_eq!(
        structural_mode_count(&metrics, STREAM_ID_COLLISION_TOTAL),
        1,
        "{obstruction:?}: the collision is counted once"
    );
    assert_eq!(
        structural_mode_count(&metrics, KEYSPACE_TYPE_ERROR_TOTAL),
        0,
        "{obstruction:?}: a stream collision is not a keyspace type error"
    );
}

/// The exact-equality boundary: an entry already at `{expected_seq+1}-0`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_stream_entry_at_the_next_seq_refuses_the_commit_whole() {
    assert_stream_obstruction_refuses_the_commit_whole(
        "structural-stream-exact",
        StreamObstruction::EntryAbove(1),
    )
    .await;
}

/// An entry above the next sequence (a stream top the header is behind).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_stream_entry_above_the_next_seq_refuses_the_commit_whole() {
    assert_stream_obstruction_refuses_the_commit_whole(
        "structural-stream-above",
        StreamObstruction::EntryAbove(5),
    )
    .await;
}

/// A wrong-typed stream key is folded into the same tag, as in fast mode.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_wrong_typed_stream_refuses_the_commit_whole() {
    assert_stream_obstruction_refuses_the_commit_whole(
        "structural-stream-wrongtype",
        StreamObstruction::WrongType,
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Ceilings
// ═════════════════════════════════════════════════════════════════════════════
//
// The planner's `apply_delta` refuses at these boundaries itself, so each
// commit is hand-built on a plan of the empty delta made at the fixture's cut.

/// Mirrors `b_k1r_tests::update_layer_lua_at_sequence_ceiling_returns_internal_with_seq_overflow`:
/// a header at `seq = 2^53 - 1` (the plan fenced on it) is the fast path's
/// `SEQ_OVERFLOW` `Internal`, and nothing is written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_at_the_sequence_ceiling_is_seq_overflow_and_writes_nothing() {
    let fx = fixture("structural-seq-ceiling").await;
    let base = fx.plan(OwnedDelta::default(), "tok-seq-ceiling").await;
    let header_bump = StorageMutation::SetObjectHeader {
        object_id: fx.oid,
        sequence_num: F64_INT_CEILING + 1,
        first_version: base.pre_state.first_version,
        source_format: None,
    };
    let commit = StructuralCommit {
        expected_seq: F64_INT_CEILING,
        ..hand_built_exact(&base, &base.pre_state, vec![header_bump])
    };
    fx.set_header_column("seq", &F64_INT_CEILING.to_string())
        .await;

    let err = assert_refused_with_error(&fx, &fx.ls, &commit).await;

    match err {
        ServiceError::Internal(msg) => assert!(
            msg.starts_with(SEQ_OVERFLOW_PREFIX),
            "expected the SEQ_OVERFLOW Internal; got {msg:?}"
        ),
        other => panic!("expected the SEQ_OVERFLOW Internal; got {other:?}"),
    }
}

/// Mirrors the fast path's `>=` `set_order` boundary
/// (`b_qw6_tests::update_layer_lua_at_set_order_ceiling_returns_internal_with_set_order_overflow`,
/// `update_layer_lua_test::update_layer_lua_set_order_overflow_on_second_record_writes_nothing`):
/// an `SFA` record carrying `set_order = 2^53 - 1` is
/// `SET_ORDER_OVERFLOW`, and nothing is written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_set_order_at_the_ceiling_is_set_order_overflow_and_writes_nothing() {
    let fx = fixture("structural-sfa-ceiling").await;
    let base = fx.plan(OwnedDelta::default(), "tok-sfa-ceiling").await;
    let commit = hand_built(
        &base,
        &base.pre_state,
        vec![StorageMutation::SetField {
            field: Field {
                section_id: WORLD,
                key_name: WORLD_FIELD.to_owned(),
                value: FieldValue::Inline(vec![0x42]),
                set_order: F64_INT_CEILING,
                authoring_set_order: 0,
                version: base.commit.expected_seq + 1,
            },
        }],
    );

    let err = assert_refused_with_error(&fx, &fx.ls, &commit).await;

    match err {
        ServiceError::Internal(msg) => assert!(
            msg.starts_with(&format!(
                "{SET_ORDER_OVERFLOW_PREFIX}: set_order {F64_INT_CEILING}"
            )),
            "expected the SFA set_order SET_ORDER_OVERFLOW Internal; got {msg:?}"
        ),
        other => panic!("expected the SET_ORDER_OVERFLOW Internal; got {other:?}"),
    }
}

/// An `HDR` record carrying `next_set_order = 2^53` (a cursor no later commit
/// could advance) is `SET_ORDER_OVERFLOW`, and nothing is written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_next_set_order_past_the_ceiling_is_set_order_overflow_and_writes_nothing(
) {
    let fx = fixture("structural-hdr-ceiling").await;
    let base = fx.plan(OwnedDelta::default(), "tok-hdr-ceiling").await;
    let commit = StructuralCommit {
        next_set_order: F64_INT_CEILING + 1,
        ..base.commit.clone()
    };

    let err = assert_refused_with_error(&fx, &fx.ls, &commit).await;

    match err {
        ServiceError::Internal(msg) => assert!(
            msg.starts_with(&format!(
                "{SET_ORDER_OVERFLOW_PREFIX}: HDR next_set_order {}",
                F64_INT_CEILING + 1
            )),
            "expected the HDR next_set_order SET_ORDER_OVERFLOW Internal; got {msg:?}"
        ),
        other => panic!("expected the SET_ORDER_OVERFLOW Internal; got {other:?}"),
    }
}

/// The other side of that boundary: `next_set_order = 2^53 - 1` is a legal
/// cursor (the value a stored `set_order` of `2^53 - 2` leaves), so the commit
/// lands and the header holds it as an exact decimal.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_next_set_order_at_the_ceiling_commits() {
    let fx = fixture("structural-hdr-at-ceiling").await;
    let base = fx.plan(OwnedDelta::default(), "tok-hdr-at-ceiling").await;
    let commit = StructuralCommit {
        next_set_order: F64_INT_CEILING,
        ..base.commit.clone()
    };

    fx.commit_landed(&commit).await;

    assert_eq!(
        fx.header().await.get("next_set_order"),
        Some(&F64_INT_CEILING.to_string())
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// obj:chs ⟺ lists, and chs_complete
// ═════════════════════════════════════════════════════════════════════════════

/// The `chs_complete` header column, `None` when absent.
async fn marker(fx: &Fixture) -> Option<String> {
    fx.header().await.remove(HEADER_FIELD_CHS_COMPLETE)
}

/// Mirrors `chs_complete_tests`' children-arm and marker-lifecycle tests: after
/// each structural commit that creates, empties or deletes children lists,
/// `obj:chs` names exactly the parents owning an `obj:ch:*` key (found by a
/// keyspace `SCAN`, independent of the index), and the marker is unchanged.
async fn structural_commits_keep_obj_chs_equal_to_the_lists(layer: &'static str, marked: bool) {
    let fx = fixture(layer).await;
    if !marked {
        unmark(&fx.tv, fx.oid).await;
    }
    let marker_before = marker(&fx).await;
    assert_eq!(
        marker_before.as_deref(),
        marked.then_some(CHS_COMPLETE_MARKED),
        "fixture precondition: the object is {} marked",
        if marked { "" } else { "not" }
    );
    // (what, delta, lists that must exist after, lists that must be gone)
    let steps: [(&str, OwnedDelta, &[u64], &[u64]); 4] = [
        (
            "a new list under SIBLING",
            structural(vec![create_cmd(SIBLING, NEW_SECTION, "Lamp")]),
            &[SIBLING],
            &[],
        ),
        (
            "WORLD's list emptied",
            structural(vec![delete_cmd(WORLD, CUBE)]),
            &[],
            &[WORLD],
        ),
        (
            "a nested list under the new section",
            structural(vec![create_cmd(NEW_SECTION, OTHER_NEW_SECTION, "Desk")]),
            &[NEW_SECTION],
            &[],
        ),
        (
            "both lists deleted in a cascade",
            structural(vec![delete_cmd(ROOT, SIBLING)]),
            &[],
            &[SIBLING, NEW_SECTION],
        ),
    ];

    for (i, (what, delta, must_list, must_not_list)) in steps.into_iter().enumerate() {
        let commit = fx.plan(delta, &format!("tok-chs-{i}")).await.commit;

        fx.commit_landed(&commit).await;

        let listed = listed_parents(&fx.tv, fx.oid).await;
        for parent in must_list {
            assert!(
                listed.contains(parent),
                "fixture precondition after {what}: obj:ch:{parent} must exist; listed {listed:?}"
            );
        }
        for parent in must_not_list {
            assert!(
                !listed.contains(parent),
                "fixture precondition after {what}: obj:ch:{parent} must be gone; listed \
                 {listed:?}"
            );
        }
        assert_eq!(
            indexed_parents(&fx.tv, fx.oid).await,
            listed,
            "after {what}, obj:chs must name exactly the parents owning a children list"
        );
        assert_eq!(
            marker(&fx).await,
            marker_before,
            "after {what}, a structural commit must leave chs_complete as it found it"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_commits_keep_obj_chs_equal_to_the_lists_on_a_marked_object() {
    structural_commits_keep_obj_chs_equal_to_the_lists("structural-chs-marked", true).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_commits_keep_obj_chs_equal_to_the_lists_on_an_unmarked_object() {
    structural_commits_keep_obj_chs_equal_to_the_lists("structural-chs-unmarked", false).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// obj:ext counting
// ═════════════════════════════════════════════════════════════════════════════
//
// `every_shape_state` holds two external values, each the only row holding its
// hash: `CUBE`'s field `mesh` (`0xAB`) and `CUBE`'s sample at `1.5` (`0xCD`).
// `obj:ext` counts value rows per hash; a hash whose count reaches zero is
// released `rewrite`, and a hash the object already held that a commit writes
// again hands its pre-flight `+1` back as `surplus_acquire` (`b_qpf_tests`).
// Each commit pays that pre-flight `+1` first (`pre_store`), as `update_layer`'s
// handler does.

const HASH_AB: u8 = 0xAB;
const HASH_CD: u8 = 0xCD;
const HASH_E1: u8 = 0xE1;
const HASH_E2: u8 = 0xE2;
const HASH_E3: u8 = 0xE3;

/// What one commit released: the count per reason, and the hashes its new
/// pending members name.
#[derive(Debug, PartialEq, Eq)]
struct Released {
    rewrite: u64,
    surplus_acquire: u64,
    hashes: BTreeSet<String>,
}

fn released(rewrite: u64, surplus_acquire: u64, hashes: &[u8]) -> Released {
    Released {
        rewrite,
        surplus_acquire,
        hashes: hashes.iter().map(|tag| ext_hash_hex(*tag)).collect(),
    }
}

/// Plan `delta` structurally, pay the pre-flight `+1` for `referenced`, commit
/// it on `ls` (which must land), and return what it released.
async fn commit_and_collect_releases(
    fx: &Fixture,
    ls: &LayerStore,
    metrics: &Metrics,
    delta: OwnedDelta,
    token: &str,
    referenced: &[u8],
) -> (PlannedCommit, Released) {
    let planned = fx.plan(delta, token).await;
    let hashes: Vec<[u8; 32]> = referenced.iter().map(|tag| ext_hash(*tag)).collect();
    pre_store(&ext_store(&fx.tv), &hashes).await;
    let rewrite = ext_ref_released(metrics, REASON_REWRITE);
    let surplus = ext_ref_released(metrics, REASON_SURPLUS);
    let pending: BTreeSet<String> = pending_members(&fx.tv).await.into_iter().collect();

    assert_committed_with(fx, ls, &planned.commit).await;

    let new_pending: BTreeSet<String> = pending_members(&fx.tv)
        .await
        .into_iter()
        .filter(|member| !pending.contains(member))
        .map(|member| member.split(':').next().unwrap_or_default().to_owned())
        .collect();
    let got = Released {
        rewrite: ext_ref_released(metrics, REASON_REWRITE) - rewrite,
        surplus_acquire: ext_ref_released(metrics, REASON_SURPLUS) - surplus,
        hashes: new_pending,
    };
    (planned, got)
}

/// `obj:ext` as `(hash_hex, count)` pairs, sorted.
fn ext_index(rows: &[(u8, i64)]) -> Vec<(String, i64)> {
    let mut rows: Vec<(String, i64)> = rows
        .iter()
        .map(|(tag, count)| (ext_hash_hex(*tag), *count))
        .collect();
    rows.sort();
    rows
}

/// A delta setting `(section, key)` fields and `(section, time)` samples, each
/// external (`Some(tag)`) or inline (`None`), in order.
fn value_delta(
    fields: &[(u64, &str, Option<u8>)],
    samples: &[(u64, f64, Option<u8>)],
) -> OwnedDelta {
    let value = |tag: Option<u8>| match tag {
        Some(tag) => (ext_hash_vec(tag), EXT_SIZE, 1),
        None => (vec![0x5A], 0, 0),
    };
    OwnedDelta {
        set_fields: fields
            .iter()
            .enumerate()
            .map(|(i, (section_id, key, tag))| {
                let (value_or_ext_hash, ext_value_size, ext_value_index) = value(*tag);
                SetField {
                    section_id: *section_id,
                    key_name: (*key).to_owned(),
                    value_or_ext_hash,
                    ext_value_size,
                    ext_value_index,
                    set_order: i as u64,
                }
            })
            .collect(),
        time_samples: samples
            .iter()
            .map(|(section_id, time, tag)| {
                let (value_or_ext_hash, ext_value_size, ext_value_index) = value(*tag);
                OodfTimeSample {
                    section_id: *section_id,
                    time: *time,
                    value_or_ext_hash,
                    ext_value_size,
                    ext_value_index,
                }
            })
            .collect(),
        ..Default::default()
    }
}

/// The fixture's starting index: one row each for `0xAB` and `0xCD`.
async fn assert_fixture_ext_index(fx: &Fixture) {
    assert_eq!(
        obj_ext_sorted(&fx.tv, fx.oid).await,
        ext_index(&[(HASH_AB, 1), (HASH_CD, 1)]),
        "fixture precondition: create_layer indexes mesh's and the 1.5 sample's hashes"
    );
}

/// Inline → external (`WORLD`'s field takes `E1`) and external → inline
/// (`CUBE`'s sample gives up `CD`) in one commit: `E1` is counted, `CD` drops
/// to zero and is released `rewrite`, and nothing else is released.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_ext_index_counts_inline_external_rewrites_and_releases_the_dropped_hash()
{
    let fx = fixture("structural-ext-inline").await;
    let (ls, metrics) = metered(&fx, "structural-ext-inline");
    assert_fixture_ext_index(&fx).await;

    let (_, got) = commit_and_collect_releases(
        &fx,
        &ls,
        &metrics,
        value_delta(
            &[(WORLD, WORLD_FIELD, Some(HASH_E1))],
            &[(CUBE, CUBE_SAMPLE_TIME, None)],
        ),
        "tok-ext-inline",
        &[HASH_E1],
    )
    .await;

    assert_eq!(
        obj_ext_sorted(&fx.tv, fx.oid).await,
        ext_index(&[(HASH_AB, 1), (HASH_E1, 1)])
    );
    assert_eq!(got, released(1, 0, &[HASH_CD]));
}

/// External `A` → external `B` (`CUBE`'s field `AB` → `E2`): `E2` is counted,
/// `AB` drops to zero and is released `rewrite`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_ext_index_counts_an_external_to_external_rewrite_and_releases_the_old_hash(
) {
    let fx = fixture("structural-ext-a-to-b").await;
    let (ls, metrics) = metered(&fx, "structural-ext-a-to-b");
    assert_fixture_ext_index(&fx).await;

    let (_, got) = commit_and_collect_releases(
        &fx,
        &ls,
        &metrics,
        value_delta(&[(CUBE, CUBE_FIELD, Some(HASH_E2))], &[]),
        "tok-ext-a-to-b",
        &[HASH_E2],
    )
    .await;

    assert_eq!(
        obj_ext_sorted(&fx.tv, fx.oid).await,
        ext_index(&[(HASH_CD, 1), (HASH_E2, 1)])
    );
    assert_eq!(got, released(1, 0, &[HASH_AB]));
}

/// One row written twice in one commit (`CUBE`'s field `AB` → `E3` → `CD`):
/// the row folds once, from its original `AB` to its final `CD`. `AB` drops to
/// zero (`rewrite`); `CD`, which the object already held, is now held by two
/// rows and its pre-flight `+1` is handed back (`surplus_acquire`); the
/// intermediate `E3` is neither counted nor released.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_ext_index_folds_a_row_written_twice_from_its_original_to_its_final_hash()
{
    let fx = fixture("structural-ext-twice").await;
    let (ls, metrics) = metered(&fx, "structural-ext-twice");
    assert_fixture_ext_index(&fx).await;

    let (planned, got) = commit_and_collect_releases(
        &fx,
        &ls,
        &metrics,
        value_delta(
            &[
                (CUBE, CUBE_FIELD, Some(HASH_E3)),
                (CUBE, CUBE_FIELD, Some(HASH_CD)),
            ],
            &[],
        ),
        "tok-ext-twice",
        &[HASH_E3, HASH_CD],
    )
    .await;

    assert_eq!(
        planned
            .commit
            .mutations
            .iter()
            .filter(|m| matches!(
                m,
                StorageMutation::SetField { field } if field.key_name == CUBE_FIELD
            ))
            .count(),
        2,
        "fixture precondition: the plan writes mesh's row twice"
    );
    assert_eq!(
        obj_ext_sorted(&fx.tv, fx.oid).await,
        ext_index(&[(HASH_CD, 2)])
    );
    assert_eq!(got, released(1, 1, &[HASH_AB, HASH_CD]));
}

/// A cascade delete counts the rows it deletes with their sections. `WORLD`'s
/// field first takes `AB` (two rows hold it; the object already held it, so
/// `surplus_acquire`). Deleting `CUBE` deletes its field (`AB` 2 → 1, no
/// release) and its sample (`CD` 1 → 0, `rewrite`). Deleting `WORLD` then
/// deletes the last `AB` row (`rewrite`), leaving the index empty.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_ext_index_counts_rows_a_cascade_delete_removes_and_releases_at_zero() {
    let fx = fixture("structural-ext-cascade").await;
    let (ls, metrics) = metered(&fx, "structural-ext-cascade");
    assert_fixture_ext_index(&fx).await;

    let (_, shared) = commit_and_collect_releases(
        &fx,
        &ls,
        &metrics,
        value_delta(&[(WORLD, WORLD_FIELD, Some(HASH_AB))], &[]),
        "tok-ext-share",
        &[HASH_AB],
    )
    .await;
    assert_eq!(
        (obj_ext_sorted(&fx.tv, fx.oid).await, shared),
        (
            ext_index(&[(HASH_AB, 2), (HASH_CD, 1)]),
            released(0, 1, &[HASH_AB])
        ),
        "fixture precondition: two rows hold AB"
    );

    let (cube_plan, cube) = commit_and_collect_releases(
        &fx,
        &ls,
        &metrics,
        structural(vec![delete_cmd(WORLD, CUBE)]),
        "tok-ext-cube",
        &[],
    )
    .await;
    assert!(
        cube_plan.commit.mutations.iter().any(|m| matches!(
            m,
            StorageMutation::DeleteField { section_id, .. } if *section_id == CUBE
        )) && cube_plan.commit.mutations.iter().any(|m| matches!(
            m,
            StorageMutation::DeleteTimeSample { section_id, .. } if *section_id == CUBE
        )),
        "fixture precondition: deleting CUBE deletes its field and sample rows"
    );
    assert_eq!(
        (obj_ext_sorted(&fx.tv, fx.oid).await, cube),
        (ext_index(&[(HASH_AB, 1)]), released(1, 0, &[HASH_CD])),
        "after CUBE's cascade"
    );

    let (_, world) = commit_and_collect_releases(
        &fx,
        &ls,
        &metrics,
        structural(vec![delete_cmd(ROOT, WORLD)]),
        "tok-ext-world",
        &[],
    )
    .await;
    assert_eq!(
        (obj_ext_sorted(&fx.tv, fx.oid).await, world),
        (Vec::new(), released(1, 0, &[HASH_AB])),
        "after WORLD's cascade"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// obj:wtok, and a replayed EVALSHA
// ═════════════════════════════════════════════════════════════════════════════

/// Mirrors `update_layer_lua_test::update_layer_fast_script_replayed_invocation_answers_the_landed_commit_without_writing`:
/// one structural `EVALSHA` records `<token>:<seq>:<server_ts>` under its
/// client in `obj:wtok` (in its atomic unit: the record is there when the call
/// returns), and a byte-identical replay — what fred sends when it re-issues a
/// command in flight on a dropped connection — answers `ALREADY_LANDED` with
/// that record and writes nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_invocation_replayed_after_landing_answers_already_landed_without_writing() {
    let fx = fixture("structural-replay").await;
    let commit = fx.plan(create_lamp(), "tok-replay").await.commit;
    let invocation =
        StructuralInvocation::from_commit(&fx.tv, &commit, fx.ls.layer_stream_maxlen());
    let seq = commit.expected_seq + 1;
    let first = invocation
        .eval(&fx.tv)
        .await
        .expect("the first run must commit");
    assert_eq!(
        first,
        vec![seq.to_string(), "drift:0:0".to_owned()],
        "the first run commits, with no settle drift"
    );
    // ARGV[1] (index 0) is the invocation's server timestamp.
    let expected_record = format!("{}:{seq}:{}", commit.commit_token, invocation.argv[0]);
    assert_eq!(
        recorded_token(&fx, WRITER).await.1,
        expected_record,
        "the commit records `<token>:<seq>:<server_ts>` under its client"
    );
    let before = fx.keyspace().await;

    let replay = invocation
        .eval(&fx.tv)
        .await
        .expect("a replayed run must answer, not error");

    assert_eq!(
        replay,
        vec!["ALREADY_LANDED".to_owned(), expected_record],
        "the replay must answer the landed commit's record verbatim"
    );
    assert_eq!(fx.keyspace().await, before, "the replay writes nothing");
}

// ═════════════════════════════════════════════════════════════════════════════
// Client operation ids: the step-1b order, in both modes
// ═════════════════════════════════════════════════════════════════════════════

/// A raw script reply: the reply's bulk strings, or the error reply's text.
type Reply = Result<Vec<String>, String>;

/// A raw run of `update_layer_fast.lua`, either mode.
struct RawRun {
    keys: Vec<String>,
    argv: Vec<String>,
}

impl RawRun {
    async fn eval(&self, fx: &Fixture) -> Reply {
        let argv: Vec<fred::types::Value> = self
            .argv
            .iter()
            .map(|a| fred::types::Value::from(a.clone()))
            .collect();
        UPDATE_LAYER_FAST_SCRIPT
            .evalsha_with_reload::<Vec<String>, _, _>(
                fx.tv.client().pool.next(),
                self.keys.clone(),
                argv,
            )
            .await
            .map_err(|e| e.details().to_owned())
    }
}

/// The structural-mode run `commit_structural` sends for `commit`
/// (`StructuralInvocation`), its identity — token and operation, start
/// record included — spelled by `commit_identity_argv`.
fn structural_run(fx: &Fixture, commit: &StructuralCommit) -> RawRun {
    let invocation = StructuralInvocation::from_commit(&fx.tv, commit, fx.ls.layer_stream_maxlen());
    RawRun {
        keys: invocation.keys,
        argv: invocation.argv,
    }
}

/// A fast-mode run — one inline `SetField` on `WORLD`, as `WRITER` — carrying
/// the very commit identity `commit` carries, in the fast mode's layout: the
/// shared prefix (`structural_commit_test::shared_argv`, its identity slots
/// spelled by `commit_identity_argv`), the mode
/// slot, then the fast records.
fn fast_run(fx: &Fixture, commit: &StructuralCommit) -> RawRun {
    let schema = fx.keys();
    let o = fx.oid_str();
    let encoded = encode_delta(
        &Some(delta_with_set_field_inline(
            WORLD,
            WORLD_FIELD,
            vec![0x5A],
            0,
        )),
        schema,
        &o,
    );
    // The shared prefix through the one builder both modes use.
    let mut keys = update_script_shared_keys(schema, fx.oid, encoded.mutation_keys.len());
    assert_eq!(keys.len(), UPDATE_SCRIPT_SHARED_KEYS);
    keys.extend(encoded.mutation_keys);
    let mut argv = shared_argv(commit, fx.ls.layer_stream_maxlen(), encoded.num_mutations);
    assert_eq!(argv.len(), UPDATE_SCRIPT_SHARED_ARGV);
    argv.push(UpdateScriptMode::Fast.as_str().to_owned());
    argv.extend(encoded.mutation_argv);
    RawRun { keys, argv }
}

/// What one guard-order case must answer.
#[derive(Debug)]
enum Answer {
    /// Exactly this reply, in BOTH modes.
    Both(Reply),
    /// The step-1d snapshot-fence reply, `{FENCED, <ms>}` with `ms ≥ 1`, in
    /// BOTH modes (the hint varies with the store's clock, so only its shape
    /// is compared).
    Fenced,
    /// Structural mode only (a guard fast mode does not have, so a fast run
    /// would commit): an error reply beginning with this text.
    StructuralRefusal(String),
}

/// Run each case's commit identity in structural mode — and, for an
/// [`Answer::Both`], in fast mode against the same store — and record every
/// answer that differs from the case's, and every run that wrote, in
/// `failures`.
async fn run_guard_cases(
    fx: &Fixture,
    failures: &mut Vec<String>,
    cases: Vec<(&'static str, StructuralCommit, Answer)>,
) {
    for (case, commit, answer) in cases {
        let before = fx.keyspace().await;
        let structural = structural_run(fx, &commit).eval(fx).await;
        match &answer {
            Answer::Both(expected) => {
                let fast = fast_run(fx, &commit).eval(fx).await;
                if &structural != expected || &fast != expected {
                    failures.push(format!(
                        "{case}: expected {expected:?} in both modes; structural \
                         {structural:?}, fast {fast:?}"
                    ));
                }
            }
            Answer::Fenced => {
                let fast = fast_run(fx, &commit).eval(fx).await;
                let fenced = |reply: &Reply| {
                    matches!(reply, Ok(v) if v.len() == 2
                        && v[0] == "FENCED"
                        && v[1].parse::<u64>().is_ok_and(|ms| ms >= 1))
                };
                if !fenced(&structural) || !fenced(&fast) {
                    failures.push(format!(
                        "{case}: expected {{FENCED, <ms>}} in both modes; structural \
                         {structural:?}, fast {fast:?}"
                    ));
                }
            }
            Answer::StructuralRefusal(prefix) => {
                if !matches!(&structural, Err(text) if text.starts_with(prefix.as_str())) {
                    failures.push(format!(
                        "{case}: expected structural mode's {prefix:?}; got {structural:?}"
                    ));
                }
            }
        }
        if fx.keyspace().await != before {
            failures.push(format!("{case}: the run wrote to the keyspace"));
        }
    }
}

/// Plan b.v6d revision 3, item 4, pinned in both script modes: the canonical
/// step-1b order — own-key match → start record (`OVERTAKEN`) → superseded →
/// `FENCED` (with no fence here; under a live one in
/// [`step_1b_order_under_a_live_fence_answers_the_same_in_both_script_modes`])
/// → `STALE` / `CACHE_DIVERGED` — is one
/// piece of code both modes run, so the same commit identity against the
/// same store answers the same, byte for byte, in fast mode and in
/// structural mode, and structural mode's own guards come only after it.
///
/// One structural commit carrying `OP_ID` lands first; its record is `R`.
/// Each case then sends a run with one of five identities, in both modes
/// against the same object (no case writes, so the two runs meet the same
/// store):
///
/// | Identity | `ARGV[8]` / suffix / start record | Answer |
/// |---|---|---|
/// | own: the landing call again | `op.OP_ID`, its token, `=` (stale) | `ALREADY_LANDED R` |
/// | earlier: a re-send of the same request | `op.OP_ID`, another token, `=R` | `ALREADY_LANDED R` |
/// | conflict: the id reused | `op.OP_ID`, another token and digest, `=R` | `ALREADY_LANDED R` |
/// | behind: another operation started before `R` | `op.OTHER_OP_ID`, `=` | `OVERTAKEN` |
/// | current: another operation started after `R` | `op.OTHER_OP_ID`, `=R` | the next guard |
///
/// in three stores: `superseded` set (structural on a plan `R` made stale) —
/// the first four answer ahead of `SUPERSEDED`, and current is refused
/// `SUPERSEDED` in both modes; `superseded` cleared, on the stale plan — the
/// first four ahead of `STALE`, current `STALE <seq>`; and on a fresh plan
/// whose children-list digest an out-of-band edit broke — own and behind
/// ahead of `CACHE_DIVERGED`, current `CACHE_DIVERGED`. The control: once the
/// edit is undone, current's fresh plan commits, so the guards above were
/// live. Which call a matched record belongs to is the Rust parser's verdict,
/// not the script's (`commit_structural_with_an_operation_answers_each_rerun_by_whose_record_it_is`).
///
/// Fast-mode counterparts of single rows:
/// `update_layer_lua_test::update_layer_fast_script_own_key_match_precedes_the_start_record_check`,
/// `…_stale_start_record_is_overtaken_without_writing`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn operation_step_1b_order_answers_the_same_in_both_script_modes() {
    let fx = fixture("structural-op-guard-order").await;
    let layer = fx.layer;
    // Planned before the landing commit, so `R` makes every run of it STALE.
    let stale_plan = fx.plan(create_lamp(), OP_CALL_TOKEN).await.commit;
    let landing = StructuralCommit {
        operation: Some(operation(OP_ID, layer, OP_PAYLOAD, StartRecord::Absent)),
        ..stale_plan.clone()
    };
    fx.commit_landed(&landing).await;
    let (_, record) = recorded_token(&fx, WRITER).await;
    let header_seq = fx.header_seq().await;
    let after = StartRecord::Record(record.clone());

    // The five identities, each on a given plan.
    let own = |plan: &StructuralCommit| StructuralCommit {
        operation: landing.operation.clone(),
        ..plan.clone()
    };
    let other = |plan: &StructuralCommit, op: Operation| StructuralCommit {
        commit_token: OTHER_CALL_TOKEN.to_owned(),
        operation: Some(op),
        ..plan.clone()
    };
    let earlier =
        |plan: &StructuralCommit| other(plan, operation(OP_ID, layer, OP_PAYLOAD, after.clone()));
    let conflict = |plan: &StructuralCommit| {
        other(plan, operation(OP_ID, layer, OTHER_PAYLOAD, after.clone()))
    };
    let behind = |plan: &StructuralCommit| {
        other(
            plan,
            operation(OTHER_OP_ID, layer, OP_PAYLOAD, StartRecord::Absent),
        )
    };
    let current = |plan: &StructuralCommit| {
        other(
            plan,
            operation(OTHER_OP_ID, layer, OP_PAYLOAD, after.clone()),
        )
    };
    let landed = || Answer::Both(Ok(vec!["ALREADY_LANDED".to_owned(), record.clone()]));
    let overtaken = || Answer::Both(Ok(vec!["OVERTAKEN".to_owned()]));

    let mut failures = Vec::new();

    fx.set_header_column("superseded", "1").await;
    run_guard_cases(
        &fx,
        &mut failures,
        vec![
            ("superseded: own", own(&stale_plan), landed()),
            ("superseded: earlier", earlier(&stale_plan), landed()),
            ("superseded: conflict", conflict(&stale_plan), landed()),
            ("superseded: behind", behind(&stale_plan), overtaken()),
            (
                "superseded: current",
                current(&stale_plan),
                Answer::Both(Err(format!("SUPERSEDED: {SUPERSEDED_SENTINEL_MSG}"))),
            ),
        ],
    )
    .await;
    let cleared: u64 = fx
        .tv
        .client()
        .pool
        .hdel(fx.keys().obj_header(&fx.oid_str()), "superseded")
        .await
        .expect("HDEL superseded must succeed");
    assert_eq!(cleared, 1, "fixture: the superseded mark was set");

    run_guard_cases(
        &fx,
        &mut failures,
        vec![
            ("stale: own", own(&stale_plan), landed()),
            ("stale: earlier", earlier(&stale_plan), landed()),
            ("stale: conflict", conflict(&stale_plan), landed()),
            ("stale: behind", behind(&stale_plan), overtaken()),
            (
                "stale: current",
                current(&stale_plan),
                Answer::StructuralRefusal(format!("STALE {header_seq}")),
            ),
        ],
    )
    .await;

    // A fresh plan of another creation under `WORLD`, whose `CHL` record
    // digest-checks `WORLD`'s list; then the list edited out of band.
    let fresh_plan = fx
        .plan(
            structural(vec![create_cmd(WORLD, OTHER_NEW_SECTION, "Desk")]),
            OP_CALL_TOKEN,
        )
        .await
        .commit;
    let world_list = fx.children_key(WORLD);
    let _: u64 = fx
        .tv
        .client()
        .pool
        .rpush(&world_list, SIBLING.to_string())
        .await
        .expect("RPUSH must succeed");
    run_guard_cases(
        &fx,
        &mut failures,
        vec![
            ("diverged: own", own(&fresh_plan), landed()),
            ("diverged: behind", behind(&fresh_plan), overtaken()),
            (
                "diverged: current",
                current(&fresh_plan),
                Answer::StructuralRefusal("CACHE_DIVERGED:".to_owned()),
            ),
        ],
    )
    .await;
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    let popped: Option<String> = fx
        .tv
        .client()
        .pool
        .rpop(&world_list, None)
        .await
        .expect("RPOP must succeed");
    assert_eq!(
        popped,
        Some(SIBLING.to_string()),
        "fixture: the edit undone"
    );
    let control = structural_run(&fx, &current(&fresh_plan)).eval(&fx).await;
    assert_eq!(
        control,
        Ok(vec![(header_seq + 1).to_string(), "drift:0:0".to_owned()]),
        "control: with the store as planned and the start record current, the run commits — \
         every refusal above came from a live guard"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// FENCED (guard 5, the script's step 1d), both modes, under a live fence
// ═════════════════════════════════════════════════════════════════════════════

/// The size a test's reader brings to its grant: 4× the shipped ceiling.
const FENCE_SIZE: LayerSize = LayerSize {
    items: 150_216,
    bytes: 1_799_752,
};

/// A reader deadline 99999999999999 ms after the epoch (the year 5138): live
/// on any store clock.
const LIVE_DEADLINE_MS: &str = "99999999999999";

/// The store clock in whole ms, read as the fence scripts read it.
async fn store_now_ms(fx: &Fixture) -> i64 {
    fred::types::scripts::Script::from_lua(
        "local t = redis.call('TIME') \
         return tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)",
    )
    .evalsha_with_reload::<i64, _, _>(
        fx.tv.client().pool.next(),
        vec![fx.fence_key()],
        Vec::<String>::new(),
    )
    .await
    .expect("TIME must answer")
}

impl Fixture {
    fn fence_key(&self) -> String {
        self.keys().obj_snapshot_fence(&self.oid_str())
    }

    fn intent_key(&self) -> String {
        self.keys().obj_snapshot_intent(&self.oid_str())
    }

    /// Replace the fence key with one reader field whose deadline is
    /// `deadline_ms` (plus the epoch bookkeeping a grant leaves).
    async fn stage_reader(&self, deadline_ms: &str) {
        let _: i64 = self
            .tv
            .client()
            .pool
            .del(self.fence_key())
            .await
            .expect("DEL fence must succeed");
        let _: () = self
            .tv
            .client()
            .pool
            .hset(
                self.fence_key(),
                vec![("r:t2", deadline_ms), ("epoch_start", "0")],
            )
            .await
            .expect("staging the fence must succeed");
    }

    /// Put a string on `key`, in place of whatever it held.
    async fn plant_string(&self, key: &str) {
        let _: () = self
            .tv
            .client()
            .pool
            .set(key, "not-a-hash", None, None, false)
            .await
            .expect("planting a string must succeed");
    }
}

/// T1, structural mode, on a real lease (`snapshot_fence::acquire_fence`):
/// the run `commit_structural` sends answers `{FENCED, <ms>}` with
/// `1 ≤ ms ≤` what is left of the lease and writes nothing; through
/// `commit_structural` itself the same reply is `CommitOutcome::Fenced`
/// (the one decoder both modes share), with no landing reported to the
/// observer; once the reader releases — `VERIFIED`, itself proof that
/// nothing landed under it — the same run commits. The fast-mode twin is
/// `update_layer_lua_test::update_layer_fast_script_under_a_real_lease_answers_fenced_then_commits_after_release`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_mode_under_a_real_lease_answers_fenced_writes_nothing_and_commits_after_release(
) {
    let fx = fixture("structural-fenced-real-lease").await;
    let commit = fx.plan(create_lamp(), "tok-fenced-real-lease").await.commit;
    let seq = fx.header_seq().await;
    let lease = match fence::acquire_fence(fx.tv.client(), fx.keys(), fx.oid, FENCE_SIZE)
        .await
        .expect("acquire_fence must answer")
    {
        AcquireVerdict::Granted(lease) => lease,
        other => panic!("the fence must be granted; got {other:?}"),
    };
    let fence: HashMap<String, String> = fx
        .tv
        .client()
        .pool
        .hgetall(fx.fence_key())
        .await
        .expect("HGETALL fence must succeed");
    let deadline: i64 = fence
        .iter()
        .find(|(field, _)| field.starts_with("r:"))
        .expect("the lease's reader field")
        .1
        .parse()
        .expect("a deadline");
    let before = fx.keyspace().await;
    let run = structural_run(&fx, &commit);

    let t0 = store_now_ms(&fx).await;
    let reply = run.eval(&fx).await;
    let (result, calls) = fx.commit(&commit).await;

    match &reply {
        Ok(v) if v.len() == 2 && v[0] == "FENCED" => {
            let ms: i64 = v[1].parse().expect("the hint is a decimal");
            assert!(
                (1..=deadline - t0).contains(&ms),
                "the hint is what is left of the lease, at least 1: {ms} ∉ 1..={}",
                deadline - t0
            );
        }
        other => {
            panic!("structural mode under a live lease answers {{FENCED, <ms>}}; got {other:?}")
        }
    }
    match result {
        Ok(CommitOutcome::Fenced { retry_after_ms }) => assert!(
            retry_after_ms >= 1 && retry_after_ms as i64 <= deadline - t0,
            "commit_structural decodes the same reply: {retry_after_ms} ms"
        ),
        other => panic!("commit_structural answers CommitOutcome::Fenced; got {other:?}"),
    }
    assert!(
        calls
            .iter()
            .all(|c| matches!(c, ObserverCall::CommitSize { .. })),
        "a fenced commit landed nothing: the observer heard only the commit's size, no \
         landing: {calls:?}"
    );
    assert_eq!(fx.keyspace().await, before, "a FENCED run writes nothing");

    assert_eq!(
        lease
            .release(Some(FENCE_SIZE))
            .await
            .expect("release must answer"),
        ReleaseVerdict::Verified,
        "nothing landed under the lease"
    );
    assert_eq!(
        run.eval(&fx).await,
        Ok(vec![(seq + 1).to_string(), "drift:0:0".to_owned()]),
        "with the lease released the same run commits"
    );
}

/// T2: the canonical step-1b order under a live fence, in both script modes,
/// one identity against one store, byte for byte — the rows of
/// [`operation_step_1b_order_answers_the_same_in_both_script_modes`] with a
/// reader holding the object:
///
/// | Store | own | behind | current |
/// |---|---|---|---|
/// | `superseded` set | `ALREADY_LANDED R` | `OVERTAKEN` | `SUPERSEDED` |
/// | a stale plan (`expected_seq` behind) | `ALREADY_LANDED R` | `OVERTAKEN` | `FENCED`, not `STALE` |
/// | a broken children-list digest | `ALREADY_LANDED R` | `OVERTAKEN` | `FENCED`, not `CACHE_DIVERGED` |
/// | a wrong-typed intent key, fence live | | | `WRONGTYPE_KEY` (intent), not `FENCED` |
/// | a wrong-typed fence key | | | `WRONGTYPE_KEY` (fence) |
///
/// So a commit this call already landed is reported, and an object that can
/// never take the commit is refused, without waiting for the reader; a fresh
/// apply waits whatever else its plan would meet (structural mode's guard 6
/// and the pre-pass come after the fence); and a wrong-typed fence or intent
/// key is the same refusal in both modes. The control: with the lease
/// lapsed (an expired reader field left on the key) and the store as
/// planned, current's fresh plan commits — every refusal above came from a
/// live guard. The WATCH path's rows are in `snapshot_fence_writer_test.rs`
/// (section 6).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn step_1b_order_under_a_live_fence_answers_the_same_in_both_script_modes() {
    let fx = fixture("structural-op-guard-order-fenced").await;
    let layer = fx.layer;
    // Planned before the landing commit, so `R` makes every run of it STALE.
    let stale_plan = fx.plan(create_lamp(), OP_CALL_TOKEN).await.commit;
    let landing = StructuralCommit {
        operation: Some(operation(OP_ID, layer, OP_PAYLOAD, StartRecord::Absent)),
        ..stale_plan.clone()
    };
    fx.commit_landed(&landing).await;
    let (_, record) = recorded_token(&fx, WRITER).await;
    let header_seq = fx.header_seq().await;
    let after = StartRecord::Record(record.clone());

    let own = |plan: &StructuralCommit| StructuralCommit {
        operation: landing.operation.clone(),
        ..plan.clone()
    };
    let other = |plan: &StructuralCommit, op: Operation| StructuralCommit {
        commit_token: OTHER_CALL_TOKEN.to_owned(),
        operation: Some(op),
        ..plan.clone()
    };
    let behind = |plan: &StructuralCommit| {
        other(
            plan,
            operation(OTHER_OP_ID, layer, OP_PAYLOAD, StartRecord::Absent),
        )
    };
    let current = |plan: &StructuralCommit| {
        other(
            plan,
            operation(OTHER_OP_ID, layer, OP_PAYLOAD, after.clone()),
        )
    };
    let landed = || Answer::Both(Ok(vec!["ALREADY_LANDED".to_owned(), record.clone()]));
    let overtaken = || Answer::Both(Ok(vec!["OVERTAKEN".to_owned()]));
    let wrong_typed = |key: String| {
        Answer::Both(Err(format!(
            "WRONGTYPE_KEY: {key} holds a string, expected hash"
        )))
    };

    let mut failures = Vec::new();
    fx.stage_reader(LIVE_DEADLINE_MS).await;

    fx.set_header_column("superseded", "1").await;
    run_guard_cases(
        &fx,
        &mut failures,
        vec![
            ("fenced, superseded: own", own(&stale_plan), landed()),
            (
                "fenced, superseded: behind",
                behind(&stale_plan),
                overtaken(),
            ),
            (
                "fenced, superseded: current",
                current(&stale_plan),
                Answer::Both(Err(format!("SUPERSEDED: {SUPERSEDED_SENTINEL_MSG}"))),
            ),
        ],
    )
    .await;
    let cleared: u64 = fx
        .tv
        .client()
        .pool
        .hdel(fx.keys().obj_header(&fx.oid_str()), "superseded")
        .await
        .expect("HDEL superseded must succeed");
    assert_eq!(cleared, 1, "fixture: the superseded mark was set");

    run_guard_cases(
        &fx,
        &mut failures,
        vec![
            ("fenced, stale: own", own(&stale_plan), landed()),
            ("fenced, stale: behind", behind(&stale_plan), overtaken()),
            (
                "fenced, stale: current",
                current(&stale_plan),
                Answer::Fenced,
            ),
        ],
    )
    .await;

    // A fresh plan of another creation under `WORLD`, whose `CHL` record
    // digest-checks `WORLD`'s list; then the list edited out of band.
    let fresh_plan = fx
        .plan(
            structural(vec![create_cmd(WORLD, OTHER_NEW_SECTION, "Desk")]),
            OP_CALL_TOKEN,
        )
        .await
        .commit;
    let world_list = fx.children_key(WORLD);
    let _: u64 = fx
        .tv
        .client()
        .pool
        .rpush(&world_list, SIBLING.to_string())
        .await
        .expect("RPUSH must succeed");
    run_guard_cases(
        &fx,
        &mut failures,
        vec![
            ("fenced, diverged: own", own(&fresh_plan), landed()),
            ("fenced, diverged: behind", behind(&fresh_plan), overtaken()),
            (
                "fenced, diverged: current",
                current(&fresh_plan),
                Answer::Fenced,
            ),
        ],
    )
    .await;
    let popped: Option<String> = fx
        .tv
        .client()
        .pool
        .rpop(&world_list, None)
        .await
        .expect("RPOP must succeed");
    assert_eq!(
        popped,
        Some(SIBLING.to_string()),
        "fixture: the edit undone"
    );

    // The fence is still live: a wrong-typed intent key is refused first.
    fx.plant_string(&fx.intent_key()).await;
    run_guard_cases(
        &fx,
        &mut failures,
        vec![(
            "live fence, wrong-typed intent: current",
            current(&fresh_plan),
            wrong_typed(fx.intent_key()),
        )],
    )
    .await;
    fx.delete_key(&fx.intent_key()).await;
    fx.plant_string(&fx.fence_key()).await;
    run_guard_cases(
        &fx,
        &mut failures,
        vec![(
            "wrong-typed fence: current",
            current(&fresh_plan),
            wrong_typed(fx.fence_key()),
        )],
    )
    .await;
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    // A lapsed lease: the reader's field is still on the key, expired.
    fx.stage_reader("1").await;
    let control = structural_run(&fx, &current(&fresh_plan)).eval(&fx).await;
    assert_eq!(
        control,
        Ok(vec![(header_seq + 1).to_string(), "drift:0:0".to_owned()]),
        "control: with the lease lapsed and the store as planned, the run commits — every \
         refusal above came from a live guard"
    );
}

/// b.pav group 15, the PREDECESSOR step in both script modes: a chained
/// identity's order — own-key match → superseded → PREDECESSOR → `FENCED`
/// (with its wrong-typed fence key) → the commit release record's type (step
/// 1e) → `STALE` / `CACHE_DIVERGED` — is one piece of code both modes run, so
/// the same identity against the same store answers the same, byte for byte,
/// in fast mode and in structural mode.
///
/// One structural commit carrying `OP_ID` lands first; its record is `R`.
/// Each case then sends a run with one of four chained identities (the writer
/// guard's `^` tag), in both modes against the same object:
///
/// | Identity | `ARGV[8]` / writer guard | Answer |
/// |---|---|---|
/// | own: the landing operation again | `op.OP_ID`, `^op.<another>` | `ALREADY_LANDED R` |
/// | pending: after a predecessor `R` is not | `op.OTHER_OP_ID`, `^op.<another>` | `{PREDECESSOR, R}` |
/// | first: a chain's first write | `op.OTHER_OP_ID`, `^` | `{PREDECESSOR, R}` |
/// | ready: after `OP_ID` | `op.OTHER_OP_ID`, `^op.OP_ID` | the next guard |
///
/// in six stores, each changed from the last:
///
/// | Store | own | pending, first | ready |
/// |---|---|---|---|
/// | `superseded` set, the stale plan | `ALREADY_LANDED R` | `SUPERSEDED` | `SUPERSEDED` |
/// | a live fence, the stale plan | `ALREADY_LANDED R` | `PREDECESSOR`, not `FENCED` | `FENCED` |
/// | a wrong-typed fence key | `ALREADY_LANDED R` | `PREDECESSOR` | `WRONGTYPE_KEY` (fence) |
/// | a wrong-typed `obj:wrel` | `ALREADY_LANDED R` | `PREDECESSOR` | `WRONGTYPE_KEY` (release record) |
/// | the stale plan | `ALREADY_LANDED R` | `PREDECESSOR` | `STALE` (structural) |
/// | a fresh plan, a broken children-list digest | `ALREADY_LANDED R` | `PREDECESSOR` | `CACHE_DIVERGED` (structural) |
///
/// So a chained call's own landed commit is reported, an object that can
/// never take the commit is refused `SUPERSEDED`, and a call whose
/// predecessor is not the record is refused before it waits for a reader,
/// trips a wrong-typed key or meets a structural guard. The control: with the
/// store as planned, ready's fresh plan commits — every refusal above came
/// from a live guard. Fast-mode single rows:
/// `update_layer_lua_test::update_layer_fast_script_predecessor_guard_commits_only_directly_after_the_predecessor`,
/// `…_own_key_match_precedes_the_predecessor_check`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn predecessor_step_order_answers_the_same_in_both_script_modes() {
    let fx = fixture("structural-predecessor-order").await;
    let layer = fx.layer;
    // Planned before the landing commit, so `R` makes every run of it STALE.
    let stale_plan = fx.plan(create_lamp(), OP_CALL_TOKEN).await.commit;
    let landing = StructuralCommit {
        operation: Some(operation(OP_ID, layer, OP_PAYLOAD, StartRecord::Absent)),
        ..stale_plan.clone()
    };
    fx.commit_landed(&landing).await;
    let (_, record) = recorded_token(&fx, WRITER).await;
    let header_seq = fx.header_seq().await;

    let chained =
        |plan: &StructuralCommit, id: &str, token: &str, previous: &str| StructuralCommit {
            commit_token: token.to_owned(),
            operation: Some(
                Operation::new(id, operation_request_digest(layer, OP_PAYLOAD))
                    .with_predecessor(previous),
            ),
            ..plan.clone()
        };
    let own = |plan: &StructuralCommit| {
        chained(plan, OP_ID, OP_CALL_TOKEN, "structural-predecessor-another")
    };
    let pending = |plan: &StructuralCommit| {
        chained(
            plan,
            OTHER_OP_ID,
            OTHER_CALL_TOKEN,
            "structural-predecessor-another",
        )
    };
    let first = |plan: &StructuralCommit| chained(plan, OTHER_OP_ID, OTHER_CALL_TOKEN, "");
    let ready = |plan: &StructuralCommit| chained(plan, OTHER_OP_ID, OTHER_CALL_TOKEN, OP_ID);
    let landed = || Answer::Both(Ok(vec!["ALREADY_LANDED".to_owned(), record.clone()]));
    let refused = || Answer::Both(Ok(vec!["PREDECESSOR".to_owned(), record.clone()]));
    let mut failures = Vec::new();
    // The four identities' cases on `plan`, labelled `labels` in the order
    // own, pending, first, ready; ready answers `next`.
    let four = |labels: [&'static str; 4], plan: &StructuralCommit, next: Answer| {
        vec![
            (labels[0], own(plan), landed()),
            (labels[1], pending(plan), refused()),
            (labels[2], first(plan), refused()),
            (labels[3], ready(plan), next),
        ]
    };

    fx.set_header_column("superseded", "1").await;
    let superseded = || Answer::Both(Err(format!("SUPERSEDED: {SUPERSEDED_SENTINEL_MSG}")));
    run_guard_cases(
        &fx,
        &mut failures,
        vec![
            ("superseded: own", own(&stale_plan), landed()),
            ("superseded: pending", pending(&stale_plan), superseded()),
            ("superseded: first", first(&stale_plan), superseded()),
            ("superseded: ready", ready(&stale_plan), superseded()),
        ],
    )
    .await;
    let cleared: u64 = fx
        .tv
        .client()
        .pool
        .hdel(fx.keys().obj_header(&fx.oid_str()), "superseded")
        .await
        .expect("HDEL superseded must succeed");
    assert_eq!(cleared, 1, "fixture: the superseded mark was set");

    fx.stage_reader(LIVE_DEADLINE_MS).await;
    run_guard_cases(
        &fx,
        &mut failures,
        four(
            [
                "live fence: own",
                "live fence: pending",
                "live fence: first",
                "live fence: ready",
            ],
            &stale_plan,
            Answer::Fenced,
        ),
    )
    .await;

    fx.plant_string(&fx.fence_key()).await;
    let wrong_typed = |key: String, what: &str| {
        Answer::Both(Err(format!(
            "WRONGTYPE_KEY: {key} holds a string, expected hash{what}"
        )))
    };
    run_guard_cases(
        &fx,
        &mut failures,
        four(
            [
                "wrong-typed fence: own",
                "wrong-typed fence: pending",
                "wrong-typed fence: first",
                "wrong-typed fence: ready",
            ],
            &stale_plan,
            wrong_typed(fx.fence_key(), ""),
        ),
    )
    .await;
    fx.delete_key(&fx.fence_key()).await;

    let wrel_key = fx.keys().obj_write_releases(&fx.oid_str());
    fx.plant_string(&wrel_key).await;
    run_guard_cases(
        &fx,
        &mut failures,
        four(
            [
                "wrong-typed release record: own",
                "wrong-typed release record: pending",
                "wrong-typed release record: first",
                "wrong-typed release record: ready",
            ],
            &stale_plan,
            wrong_typed(wrel_key.clone(), " (commit release record)"),
        ),
    )
    .await;
    fx.delete_key(&wrel_key).await;

    run_guard_cases(
        &fx,
        &mut failures,
        four(
            [
                "stale: own",
                "stale: pending",
                "stale: first",
                "stale: ready",
            ],
            &stale_plan,
            Answer::StructuralRefusal(format!("STALE {header_seq}")),
        ),
    )
    .await;

    // A fresh plan of another creation under `WORLD`, whose `CHL` record
    // digest-checks `WORLD`'s list; then the list edited out of band.
    let fresh_plan = fx
        .plan(
            structural(vec![create_cmd(WORLD, OTHER_NEW_SECTION, "Desk")]),
            OP_CALL_TOKEN,
        )
        .await
        .commit;
    let world_list = fx.children_key(WORLD);
    let _: u64 = fx
        .tv
        .client()
        .pool
        .rpush(&world_list, SIBLING.to_string())
        .await
        .expect("RPUSH must succeed");
    run_guard_cases(
        &fx,
        &mut failures,
        four(
            [
                "diverged: own",
                "diverged: pending",
                "diverged: first",
                "diverged: ready",
            ],
            &fresh_plan,
            Answer::StructuralRefusal("CACHE_DIVERGED:".to_owned()),
        ),
    )
    .await;
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    let popped: Option<String> = fx
        .tv
        .client()
        .pool
        .rpop(&world_list, None)
        .await
        .expect("RPOP must succeed");
    assert_eq!(
        popped,
        Some(SIBLING.to_string()),
        "fixture: the edit undone"
    );
    let control = structural_run(&fx, &ready(&fresh_plan)).eval(&fx).await;
    assert_eq!(
        control,
        Ok(vec![(header_seq + 1).to_string(), "drift:0:0".to_owned()]),
        "control: with the store as planned and the record naming its predecessor, the chained \
         run commits — every refusal above came from a live guard"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Partial vs absent header
// ═════════════════════════════════════════════════════════════════════════════

/// Mirrors `update_layer_lua_test::update_layer_lua_missing_header_returns_not_found`:
/// an absent header is `NotFound(EMPTY_HEADER_SENTINEL)` — the variant the
/// handler's recovery arm keys on — and nothing is written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_absent_header_is_not_found_and_writes_nothing() {
    let fx = fixture("structural-absent-header").await;
    let commit = fx.plan(create_lamp(), "tok-absent-header").await.commit;
    fx.delete_key(&fx.keys().obj_header(&fx.oid_str())).await;

    let err = assert_refused_with_error_raw(&fx, &fx.ls, &commit).await;

    match err {
        ServiceError::NotFound(msg) => assert_eq!(msg, EMPTY_HEADER_SENTINEL_MSG),
        other => panic!("an absent header must be NotFound; got {other:?}"),
    }
}

/// Mirrors `b_k1r_tests::update_layer_lua_partial_header_returns_failed_precondition`:
/// a header whose `seq` does not parse is `FailedPrecondition(PARTIAL_HEADER_SENTINEL)`
/// — never `NotFound`, which would route into recovery — and nothing is written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_partial_header_is_failed_precondition_and_writes_nothing() {
    let fx = fixture("structural-partial-header").await;
    let commit = fx.plan(create_lamp(), "tok-partial-header").await.commit;
    fx.set_header_column("seq", "not-a-number").await;

    let err = assert_refused_with_error(&fx, &fx.ls, &commit).await;

    match err {
        ServiceError::FailedPrecondition(msg) => assert_eq!(msg, PARTIAL_HEADER_SENTINEL_MSG),
        other => panic!("a partial header must be FailedPrecondition; got {other:?}"),
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// WRONGTYPE_KEY per written key family
// ═════════════════════════════════════════════════════════════════════════════

/// One written key family: a plan that writes it, the key, and the tail of
/// the refusal after `<key> holds a string, `.
struct WrongTypeCase {
    family: &'static str,
    delta: fn() -> OwnedDelta,
    key: fn(&Fixture) -> String,
    expected: &'static str,
}

fn delete_cube() -> OwnedDelta {
    structural(vec![delete_cmd(WORLD, CUBE)])
}

fn world_field_value() -> OwnedDelta {
    delta_with_set_field_inline(WORLD, WORLD_FIELD, vec![0x42], 0)
}

fn world_sample_value() -> OwnedDelta {
    delta_with_time_sample_inline(WORLD, WORLD_SAMPLE_TIME, vec![0x43])
}

/// The record-key qualifier; the record number depends on the plan's order.
const RECORD_KEY: &str = "expected hash (mutation record ";

/// Every key family a structural commit writes, and the digest-checked row
/// keys its `DSEC` / `DF` / `DT` records read: the WATCH path's coverage
/// (`b_459_tests`) plus the script's own (`obj:ext`, `obj:wtok`).
const WRONG_TYPE_CASES: &[WrongTypeCase] = &[
    WrongTypeCase {
        family: "sec row (SEC)",
        delta: create_lamp,
        key: |fx| fx.section_key(NEW_SECTION),
        expected: RECORD_KEY,
    },
    WrongTypeCase {
        family: "sec row (DSEC, digest-checked)",
        delta: delete_cube,
        key: |fx| fx.section_key(CUBE),
        expected: RECORD_KEY,
    },
    WrongTypeCase {
        family: "secs",
        delta: create_lamp,
        key: |fx| fx.keys().obj_sections_index(&fx.oid_str()),
        expected: "expected set (sections index)",
    },
    WrongTypeCase {
        family: "del row (TOMB)",
        delta: delete_cube,
        key: |fx| {
            fx.keys()
                .obj_deleted_section(&fx.oid_str(), &CUBE.to_string())
        },
        expected: RECORD_KEY,
    },
    WrongTypeCase {
        family: "dels",
        delta: delete_cube,
        key: |fx| fx.keys().obj_deleted_sections_index(&fx.oid_str()),
        expected: "expected set (deleted-sections index)",
    },
    WrongTypeCase {
        family: "chs",
        delta: create_lamp,
        key: |fx| fx.keys().obj_children_index(&fx.oid_str()),
        expected: "expected set (children-list index)",
    },
    WrongTypeCase {
        family: "field row (SFA)",
        delta: world_field_value,
        key: |fx| fx.field_key(WORLD, WORLD_FIELD),
        expected: RECORD_KEY,
    },
    WrongTypeCase {
        family: "field row (DF, digest-checked)",
        delta: delete_cube,
        key: |fx| fx.field_key(CUBE, CUBE_FIELD),
        expected: RECORD_KEY,
    },
    WrongTypeCase {
        family: "flds",
        delta: world_field_value,
        key: |fx| fx.keys().obj_fields_index(&fx.oid_str()),
        expected: "expected zset (fields index)",
    },
    WrongTypeCase {
        family: "ts row (STA)",
        delta: world_sample_value,
        key: |fx| fx.time_sample_key(WORLD, WORLD_SAMPLE_TIME),
        expected: RECORD_KEY,
    },
    WrongTypeCase {
        family: "ts row (DT, digest-checked)",
        delta: delete_cube,
        key: |fx| fx.time_sample_key(CUBE, CUBE_SAMPLE_TIME),
        expected: RECORD_KEY,
    },
    WrongTypeCase {
        family: "tss",
        delta: world_sample_value,
        key: |fx| fx.keys().obj_time_samples_index(&fx.oid_str()),
        expected: "expected zset (time-samples index)",
    },
    WrongTypeCase {
        family: "obj:ext",
        delta: world_field_value,
        key: |fx| fx.keys().obj_ext_index(&fx.oid_str()),
        expected: "expected hash (external-value reference index)",
    },
    WrongTypeCase {
        family: "obj:wtok",
        delta: create_lamp,
        key: |fx| fx.keys().obj_write_token(&fx.oid_str()),
        expected: "expected hash (commit-token record)",
    },
];

/// Mirrors `update_layer_lua_test` section 14 and `b_459_tests`' structural
/// families: a string where a written key belongs is refused with the fast
/// path's `WRONGTYPE_KEY` `Internal` naming the key, nothing is written, and
/// `keyspace_type_error{backend="lua", site="update"}` ticks once, under
/// `mode="structural"` and not `mode="fast"`. Each family
/// gets its own object (planned before the key is wrong-typed, since the
/// planner's read would refuse first); every case runs and failures are
/// reported together.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_wrong_typed_key_of_every_written_family_is_refused_wrongtype_key() {
    let mut failures = Vec::new();
    for case in WRONG_TYPE_CASES {
        let fx = fixture("structural-wrongtype").await;
        let (ls, metrics) = metered(&fx, "structural-wrongtype");
        let commit = fx.plan((case.delta)(), "tok-wrongtype").await.commit;
        let key = (case.key)(&fx);
        plant_string(&fx, &key).await;
        let before = fx.keyspace().await;

        let (result, calls) = fx.commit_with(&ls, &commit).await;

        let expected = format!(
            "{WRONGTYPE_KEY_PREFIX}{key} holds a string, {}",
            case.expected
        );
        match &result {
            Err(ServiceError::Internal(msg)) if msg.starts_with(&expected) => {}
            other => failures.push(format!(
                "{}: expected an Internal starting {expected:?}; got {other:?}",
                case.family
            )),
        }
        if !matches!(calls.as_slice(), [ObserverCall::CommitSize { .. }]) {
            failures.push(format!(
                "{}: the commit must not land; calls {calls:?}",
                case.family
            ));
        }
        if fx.keyspace().await != before {
            failures.push(format!(
                "{}: the refusal wrote to the keyspace",
                case.family
            ));
        }
        let ticks = structural_mode_count(&metrics, KEYSPACE_TYPE_ERROR_TOTAL);
        if ticks != 1 {
            failures.push(format!(
                "{}: keyspace_type_error must tick once; ticked {ticks}",
                case.family
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A wrong-typed header is refused exactly as fast mode refuses it: both modes'
/// first command is the header `HGETALL`, whose server `WRONGTYPE` the shared
/// mapping classifies — the same `Internal`, word for word — with no
/// `WRONGTYPE_KEY` probe and no type-error tick. Nothing is written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_wrong_typed_header_is_refused_exactly_as_the_fast_path_refuses_it() {
    let fx = fixture("structural-wrongtype-header").await;
    let (ls, metrics) = metered(&fx, "structural-wrongtype-header");
    let commit = fx.plan(create_lamp(), "tok-wrongtype-header").await.commit;
    plant_string(&fx, &fx.keys().obj_header(&fx.oid_str())).await;
    let fast_delta = world_field_value();
    assert!(
        is_fast_path(&Some(fast_delta.clone())),
        "fixture precondition: the comparison delta takes the fast path"
    );
    let fast = ls
        .update_layer_relayed(
            &fx.session_id,
            fx.layer,
            fx.oid,
            Some(fast_delta),
            WRITER,
            CLIENT_TS,
        )
        .await;
    let Err(ServiceError::Internal(fast_msg)) = fast else {
        panic!("fixture precondition: the fast path refuses a wrong-typed header Internal; got {fast:?}");
    };
    assert!(
        fast_msg.contains("WRONGTYPE"),
        "fixture precondition: the fast refusal is the server's WRONGTYPE; got {fast_msg:?}"
    );

    let err = assert_refused_with_error_raw(&fx, &ls, &commit).await;

    match err {
        ServiceError::Internal(msg) => assert_eq!(msg, fast_msg),
        other => panic!("expected the fast path's Internal {fast_msg:?}; got {other:?}"),
    }
    assert_eq!(
        structural_mode_count(&metrics, KEYSPACE_TYPE_ERROR_TOTAL),
        0,
        "neither mode probes the header, so neither ticks a type error"
    );
}

/// Mirrors `b_459_tests::txn_wrongtype_children_list_the_marked_index_does_not_name_is_healed_by_the_commit`:
/// a children list under the fixture's dangling parent (a list the cache does
/// not hold, so its record carries no digest and its key is never probed),
/// wrong-typed, does not refuse the commit: the record's `DEL` + `RPUSH`
/// rebuilds it and no type error ticks. (`obj:chs` already names this parent —
/// `create_layer` indexed its list — so the index half of the repair is pinned
/// by the parent-new-in-the-plan test below.) Hand-built: the engine writes no
/// list under a parent that has no section row.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_wrong_typed_children_list_under_a_dangling_parent_commits_and_heals() {
    let fx = fixture("structural-heal-dangling").await;
    let (ls, metrics) = metered(&fx, "structural-heal-dangling");
    let base = fx.plan(OwnedDelta::default(), "tok-heal-dangling").await;
    assert!(
        !base.pre_state.sections.contains_key(&DANGLING_PARENT)
            && !base.pre_state.children.contains_key(&DANGLING_PARENT),
        "fixture precondition: the parent has no row, and the planned state holds no list under it"
    );
    let commit = hand_built(
        &base,
        &base.pre_state,
        vec![StorageMutation::SetChildren {
            section_id: DANGLING_PARENT,
            children: vec![NEW_SECTION],
        }],
    );
    let list_key = fx.children_key(DANGLING_PARENT);
    plant_string(&fx, &list_key).await;

    assert_committed_with(&fx, &ls, &commit).await;

    assert_eq!(fx.list(list_key).await, vec![NEW_SECTION.to_string()]);
    assert_eq!(
        structural_mode_count(&metrics, KEYSPACE_TYPE_ERROR_TOTAL),
        0
    );
}

/// The engine-emitted form of the same heal: a parent created in the same
/// delta has no row before the plan, so its list carries no digest either. A
/// string already at its list key is replaced by the list, and `obj:chs`,
/// which did not name the parent before the commit, names it after.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_wrong_typed_children_list_under_a_parent_new_in_the_plan_commits_and_heals(
) {
    let fx = fixture("structural-heal-new-parent").await;
    let (ls, metrics) = metered(&fx, "structural-heal-new-parent");
    let commit = fx
        .plan(
            structural(vec![
                create_cmd(WORLD, NEW_SECTION, "Lamp"),
                create_cmd(NEW_SECTION, OTHER_NEW_SECTION, "Bulb"),
            ]),
            "tok-heal-new-parent",
        )
        .await
        .commit;
    let list_key = fx.children_key(NEW_SECTION);
    plant_string(&fx, &list_key).await;
    let chs_key = fx.keys().obj_children_index(&fx.oid_str());
    assert!(
        !fx.members(chs_key.clone())
            .await
            .contains(&NEW_SECTION.to_string()),
        "fixture precondition: obj:chs does not name the new parent before the commit"
    );

    assert_committed_with(&fx, &ls, &commit).await;

    assert_eq!(fx.list(list_key).await, vec![OTHER_NEW_SECTION.to_string()]);
    assert!(
        fx.members(chs_key).await.contains(&NEW_SECTION.to_string()),
        "the commit's repair must index the rebuilt list in obj:chs"
    );
    assert_eq!(
        structural_mode_count(&metrics, KEYSPACE_TYPE_ERROR_TOTAL),
        0
    );
}

/// A digest-checked children list holding a string is not what the cache
/// believed: the commit answers `CacheDiverged` (not `WRONGTYPE_KEY` — list
/// keys are never type-refused), writes nothing, and ticks no type error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_wrong_typed_digest_checked_children_list_is_cache_diverged() {
    let fx = fixture("structural-list-diverged").await;
    let (ls, metrics) = metered(&fx, "structural-list-diverged");
    let commit = fx.plan(create_lamp(), "tok-list-diverged").await.commit;
    plant_string(&fx, &fx.children_key(WORLD)).await;
    let before = fx.keyspace().await;

    let (result, calls) = fx.commit_with(&ls, &commit).await;

    assert!(
        matches!(result, Ok(CommitOutcome::CacheDiverged)),
        "got {result:?}"
    );
    assert_sent_without_landing(&calls, "CacheDiverged");
    assert_eq!(fx.keyspace().await, before, "CacheDiverged writes nothing");
    assert_eq!(
        structural_mode_count(&metrics, KEYSPACE_TYPE_ERROR_TOTAL),
        0
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// One value shape per row
// ═════════════════════════════════════════════════════════════════════════════

async fn row(fx: &Fixture, key: &str) -> HashMap<String, String> {
    fx.tv
        .client()
        .pool
        .hgetall(key)
        .await
        .unwrap_or_else(|e| panic!("HGETALL {key} must succeed: {e}"))
}

/// Mirrors `b_zvd_tests`' rewrites: `WORLD`'s inline field and sample are
/// rewritten external through `SFA` / `STA`, then inline again, and after
/// each commit every row holds exactly one value shape — the one just written —
/// with its non-value columns intact.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_inline_external_rewrites_leave_exactly_one_value_shape_per_row() {
    let fx = fixture("structural-one-shape").await;
    let field_key = fx.field_key(WORLD, WORLD_FIELD);
    let sample_key = fx.time_sample_key(WORLD, WORLD_SAMPLE_TIME);
    pre_store(&ext_store(&fx.tv), &[ext_hash(HASH_E1)]).await;

    for (i, external) in [true, false].into_iter().enumerate() {
        let tag = external.then_some(HASH_E1);
        let commit = fx
            .plan(
                value_delta(
                    &[(WORLD, WORLD_FIELD, tag)],
                    &[(WORLD, WORLD_SAMPLE_TIME, tag)],
                ),
                &format!("tok-one-shape-{i}"),
            )
            .await
            .commit;

        fx.commit_landed(&commit).await;

        let shape = if external { "external" } else { "inline" };
        assert_exactly_one_value_shape(
            &row(&fx, &field_key).await,
            external,
            &["set_order", "authoring_set_order", "version"],
            &format!("field rewritten {shape} (SFA)"),
        );
        assert_exactly_one_value_shape(
            &row(&fx, &sample_key).await,
            external,
            &["version"],
            &format!("sample rewritten {shape} (STA)"),
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// STALE under a racing fast-path commit
// ═════════════════════════════════════════════════════════════════════════════

/// A structural plan made before a fast-mode commit lands is `Stale` with the
/// fast commit's `seq`, and the structural commit writes nothing — the fence
/// holds against the other mode, not only against another structural plan.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_planned_before_a_fast_path_commit_is_stale_with_its_seq() {
    let fx = fixture("structural-stale-fast").await;
    let commit = fx.plan(create_lamp(), "tok-stale-fast").await.commit;
    let fast_delta = world_field_value();
    assert!(
        is_fast_path(&Some(fast_delta.clone())),
        "fixture precondition: the racing commit takes the fast path"
    );
    let (fast_seq, _) = fx
        .ls
        .update_layer_relayed(
            &fx.session_id,
            fx.layer,
            fx.oid,
            Some(fast_delta),
            "racing-fast-writer",
            CLIENT_TS,
        )
        .await
        .expect("the fast-path commit must land");
    assert_eq!(
        fast_seq,
        commit.expected_seq + 1,
        "fixture precondition: the fast commit takes the sequence the plan was fenced below"
    );

    fx.assert_refused(&commit, CommitOutcome::Stale { seq: fast_seq })
        .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Delete, re-create, delete over an old tombstone
// ═════════════════════════════════════════════════════════════════════════════

async fn tombstone(fx: &Fixture, section_id: u64) -> HashMap<String, String> {
    row(
        fx,
        &fx.keys()
            .obj_deleted_section(&fx.oid_str(), &section_id.to_string()),
    )
    .await
}

/// The `first_ver` / `last_ver` columns a plan's tombstone of `section_id`
/// writes.
fn planned_tombstone(planned: &PlannedCommit, section_id: u64) -> HashMap<String, String> {
    planned
        .commit
        .mutations
        .iter()
        .find_map(|m| match m {
            StorageMutation::CreateTombstone {
                section_id: id,
                first_version,
                last_version,
            } if *id == section_id => Some(HashMap::from([
                ("first_ver".to_owned(), first_version.to_string()),
                ("last_ver".to_owned(), last_version.to_string()),
            ])),
            _ => None,
        })
        .unwrap_or_else(|| panic!("fixture precondition: the plan tombstones {section_id}"))
}

/// `CUBE` is deleted, created again and deleted again, each a structural
/// commit. The second delete lands over the first delete's tombstone (the
/// `TOMB` record carries no digest), and the tombstone then holds the second
/// delete's versions.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_section_deleted_recreated_and_deleted_again_rewrites_its_tombstone() {
    let fx = fixture("structural-tombstone-again").await;
    let first = fx.plan(delete_cube(), "tok-tomb-first").await;
    fx.commit_landed(&first.commit).await;
    let recreate = fx
        .plan(
            structural(vec![create_cmd(WORLD, CUBE, "Cube")]),
            "tok-tomb-recreate",
        )
        .await;
    fx.commit_landed(&recreate.commit).await;
    let first_tombstone = planned_tombstone(&first, CUBE);
    assert_eq!(
        tombstone(&fx, CUBE).await,
        first_tombstone,
        "fixture precondition: the first delete's tombstone is still there"
    );
    let second = fx.plan(delete_cube(), "tok-tomb-second").await;
    let second_tombstone = planned_tombstone(&second, CUBE);
    assert_ne!(
        second_tombstone, first_tombstone,
        "fixture precondition: the two deletes tombstone different versions"
    );

    fx.commit_landed(&second.commit).await;

    assert_eq!(tombstone(&fx, CUBE).await, second_tombstone);
    fx.assert_store_holds(&second.post_state).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The commit-token record, byte for byte across the three commit routes
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod record_parity {
    //! Needs `test-fault-injection`: the fixed commit-token seam
    //! (`LayerStore::with_commit_tokens_for_test`) the fast and WATCH routes
    //! mint their call token through is compiled only under it.

    use std::collections::VecDeque;
    use std::sync::Mutex;

    use live_edit::config::UpdateLayerBackend;
    use live_edit::storage::layer::{OperationOutcome, UpdateCommit};
    use live_edit::storage::structural::commit_structural;
    use live_edit::storage::test_support::structural_commit::DUMP_SERVER_TS;

    use super::*;

    /// The one call token every route records: the fast and WATCH routes
    /// mint it through the seam, the structural commit carries its rendering.
    const TOKEN: u128 = 0x0dc7_5011_0000_0000_0000_0000_0000_6d6d;
    /// The layer name the operation's request digest is computed over. Each
    /// route's fixture has its own layer; the digest is the client's, so it
    /// is the same for all three.
    const DIGEST_LAYER: &str = "structural-record-parity";

    /// One route's commit: what the call returned, and `WRITER`'s raw
    /// record and dumped (server-timestamp-normalized) record.
    #[derive(Debug)]
    struct Recorded {
        seq: u64,
        server_ts: u64,
        raw: String,
        dumped: String,
    }

    async fn writer_record(fx: &Fixture) -> Option<String> {
        fx.tv
            .client()
            .pool
            .hget(fx.keys().obj_write_token(&fx.oid_str()), WRITER)
            .await
            .expect("HGET obj:wtok must succeed")
    }

    /// `WRITER`'s record in the object's dump, tokens verbatim.
    async fn dumped_record(fx: &Fixture) -> String {
        let dump =
            dump_object_keyspace(fx.tv.client(), fx.keys(), fx.oid, CommitTokens::Verbatim).await;
        match dump.get("obj:wtok") {
            Some(live_edit::storage::test_support::structural_commit::DumpedValue::Hash(
                records,
            )) => records
                .get(WRITER)
                .cloned()
                .expect("the dump holds the writer's record"),
            other => panic!("obj:wtok must dump as a hash; got {other:?}"),
        }
    }

    /// The operation every route's call carries, `None` without one. Its
    /// start record says the writer's field was absent at call start (each
    /// fixture is fresh; asserted per route).
    fn route_operation(with_operation: bool) -> Option<Operation> {
        with_operation.then(|| operation(OP_ID, DIGEST_LAYER, OP_PAYLOAD, StartRecord::Absent))
    }

    /// Commit `delta` through `update_layer` on `backend` with the fixed token.
    async fn through_update_layer(
        layer: &'static str,
        backend: UpdateLayerBackend,
        delta: OwnedDelta,
        with_operation: bool,
    ) -> Recorded {
        // The fixture and the commit are boxed here, as are the fixture, the
        // plan and the commit in `through_structural_mode`. Inline, each
        // future is held in every helper future above it (the fixture's alone
        // is about 47 KB in a debug build), and in a debug build each of their
        // poll frames on the test thread's stack holds copies of it. The
        // WATCH route's store path, about 1 MB deep in a debug build, then
        // brought these tests within 256 KiB of std's default 2 MiB test-thread stack.
        let fx = Box::pin(fixture(layer)).await;
        assert_eq!(
            writer_record(&fx).await,
            None,
            "{layer}: fixture precondition: no record for the writer yet"
        );
        let ls = layer_store(&fx.tv)
            .with_update_layer_backend(backend)
            .with_commit_tokens_for_test(Arc::new(Mutex::new(VecDeque::from([TOKEN]))));
        let UpdateCommit {
            seq,
            server_ts,
            outcome,
        } = Box::pin(ls.update_layer_with_operation_relayed(
            &fx.session_id,
            fx.layer,
            fx.oid,
            Some(delta),
            WRITER,
            CLIENT_TS,
            route_operation(with_operation),
        ))
        .await
        .unwrap_or_else(|e| panic!("{layer}: the update must commit: {e:?}"));
        assert_eq!(
            outcome,
            if with_operation {
                OperationOutcome::Applied
            } else {
                OperationOutcome::Unspecified
            },
            "{layer}: the call's own commit"
        );
        Recorded {
            seq,
            server_ts,
            raw: writer_record(&fx).await.expect("the commit recorded"),
            dumped: dumped_record(&fx).await,
        }
    }

    /// Commit a planned `CreateSection` through `commit_structural`, the
    /// fixed token rendered as the commit's.
    async fn through_structural_mode(layer: &'static str, with_operation: bool) -> Recorded {
        // Boxed: see `through_update_layer`.
        let fx = Box::pin(fixture(layer)).await;
        assert_eq!(
            writer_record(&fx).await,
            None,
            "{layer}: fixture precondition: no record for the writer yet"
        );
        let mut commit = Box::pin(fx.plan(create_lamp(), &format!("{TOKEN:032x}")))
            .await
            .commit;
        commit.operation = route_operation(with_operation);
        let (seq, server_ts) =
            match Box::pin(commit_structural(&fx.ls, &commit, &mut NoObserver)).await {
                Ok(CommitOutcome::Committed { seq, server_ts }) => (seq, server_ts),
                other => panic!("{layer}: the structural commit must land; got {other:?}"),
            };
        Recorded {
            seq,
            server_ts,
            raw: writer_record(&fx).await.expect("the commit recorded"),
            dumped: dumped_record(&fx).await,
        }
    }

    struct NoObserver;

    impl live_edit::storage::structural::CommitObserver for NoObserver {
        fn commit_size(&mut self, _records: u64, _bytes: u64) {}
        fn landed(&mut self, _seq: u64, _server_ts: u64) {}
        fn already_landed(&mut self, _seq: u64, _server_ts: u64) {}
    }

    /// The record of one call's commit through each route — the fast mode
    /// (an inline `SetField`), structural mode (a planned `CreateSection`
    /// through `commit_structural`) and the WATCH path (the same
    /// `CreateSection` through `update_layer` on the txn backend) — each on a
    /// fresh fixture object, so each lands at the same `seq`. Every raw
    /// record is the documented string for the call's own `(seq,
    /// server_ts)`; the three dumped records (the server timestamp, the one
    /// wall-clock part, normalized) are byte-identical.
    async fn record_is_byte_identical_across_the_three_routes(
        with_operation: bool,
        expected: impl Fn(u64, &str) -> String,
    ) {
        let routes = [
            (
                "fast",
                through_update_layer(
                    if with_operation {
                        "structural-record-fast-op"
                    } else {
                        "structural-record-fast"
                    },
                    UpdateLayerBackend::Lua,
                    delta_with_set_field_inline(WORLD, WORLD_FIELD, vec![0x6d], 0),
                    with_operation,
                )
                .await,
            ),
            (
                "structural",
                through_structural_mode(
                    if with_operation {
                        "structural-record-structural-op"
                    } else {
                        "structural-record-structural"
                    },
                    with_operation,
                )
                .await,
            ),
            (
                "WATCH",
                through_update_layer(
                    if with_operation {
                        "structural-record-watch-op"
                    } else {
                        "structural-record-watch"
                    },
                    UpdateLayerBackend::Txn,
                    create_lamp(),
                    with_operation,
                )
                .await,
            ),
        ];
        for (route, recorded) in &routes {
            assert_eq!(
                recorded.raw,
                expected(recorded.seq, &recorded.server_ts.to_string()),
                "{route}: the record carries the call's own (seq, server_ts)"
            );
            assert_eq!(
                recorded.dumped,
                expected(recorded.seq, DUMP_SERVER_TS),
                "{route}: the dump normalizes the server timestamp alone"
            );
        }
        let (first_route, first) = &routes[0];
        for (route, recorded) in &routes[1..] {
            assert_eq!(
                (recorded.seq, &recorded.dumped),
                (first.seq, &first.dumped),
                "the {route} record must be byte-identical to the {first_route} record, the \
                 server timestamp aside"
            );
        }
    }

    /// Plan b.v6d revision 3, the op-mode record parity: with a client
    /// `operation_id`, every route records
    /// `op.<operation_id>:<seq>:<server_ts>:<token>:<request_digest>`, byte
    /// for byte the same string — so a re-send is answered from the record
    /// whichever route made it. (The fast / WATCH pair alone:
    /// `update_layer_lua_test::update_layer_operation_record_is_identical_on_both_backends`.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn operation_record_is_byte_identical_across_the_fast_structural_and_watch_commits() {
        let digest = operation(OP_ID, DIGEST_LAYER, OP_PAYLOAD, StartRecord::Absent)
            .digest()
            .to_owned();
        record_is_byte_identical_across_the_three_routes(true, |seq, server_ts| {
            format!("op.{OP_ID}:{seq}:{server_ts}:{TOKEN:032x}:{digest}")
        })
        .await;
    }

    /// Without an `operation_id` every route keeps the b.75i record,
    /// `<token>:<seq>:<server_ts>` — nothing after the timestamp — byte for
    /// byte the same.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn record_without_an_operation_keeps_the_per_call_shape_across_the_three_routes() {
        record_is_byte_identical_across_the_three_routes(false, |seq, server_ts| {
            format!("{TOKEN:032x}:{seq}:{server_ts}")
        })
        .await;
    }
}
