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

//! b.v6d — a client re-send carrying the same `operation_id` is exactly-once
//! across calls, for `UpdateLayer` and `CreateLayer`, at the storage layer.
//!
//! # The defect
//!
//! b.75i made ONE `update_layer` call commit at most once: a per-call random
//! token recorded in `le:{oid}:obj:wtok` lets an in-process retry find the
//! commit its own earlier attempt landed. A client re-send is a NEW call with
//! a new token, so a re-send of an `UpdateLayer` whose reply was lost applied
//! the delta again (a duplicate commit for subscribers; a `DuplicateSection`
//! refusal for a landed `CreateSection`), and a re-sent `CreateLayer` minted a
//! fresh `object_id` and replaced the layer it had already created.
//!
//! The fix lets the client name its write intent. With an `operation_id`, the
//! update record is keyed `op.<id>` and carries the call token and a request
//! digest; a create is recorded at `le:{sid}:createop:{id}` by its commit
//! script. A matched re-send is answered `ALREADY_COMMITTED` with the original
//! commit and applies nothing; the same id over different bytes is
//! `ALREADY_EXISTS`.
//!
//! Revision 3 closes the late-attempt window. The record holds a writer's
//! LATEST commit, so an attempt that ran after that writer's next commit — a
//! retry, a fred replay, a delayed `EVALSHA` — matched nothing and applied
//! again, after the later write. Each operation-id call now reads its
//! writer's record once at call start (its start record), and every attempt
//! whose start record no longer matches is refused as overtaken; and the
//! commit `EVALSHA` is written to the wire at most once, so fred never
//! replays it.
//!
//! Amendment R4-1 covers `CreateLayer`'s own commit point on an overwrite. That
//! commit `EVALSHA` is sent once on the `txn_pool`, which follows no redirect,
//! after the old object is already marked superseded; when it fails, the call
//! reads the mapping, its commit claim and — with an operation id — its
//! operation record and, if no run landed and the mapping still names the old
//! object, tries the commit once more
//! (`overwrite_second_try`, at the end of this file). Since b.ua9 a moved
//! mapping is re-planned against, and a new layer is tried twice too.
//!
//! # What lives here
//!
//! | Test | Design item | What it pins |
//! |---|---|---|
//! | [`lua_resend_of_a_landed_operation_is_answered_already_committed`] / [`txn_resend_of_a_landed_operation_is_answered_already_committed`] | T1 | a re-send is answered with the original `(seq, server_ts)`, nothing applied, `operation_id_match_total{update_layer,already_committed}` ticks |
//! | [`structural_resend_of_a_landed_create_section_is_answered_not_refused`] | T2 | the same on the default `lua` backend's structural route, where a re-apply would be refused `DuplicateSection` |
//! | [`lua_earlier_call_answer_hands_back_the_resends_preflight_increment`] / [`txn_earlier_call_answer_hands_back_the_resends_preflight_increment`] | T4 | the re-send's pre-flight `ref_count` +1 is released (`rsd<token>` member) and a GC tick returns `ref_count` to its value before the re-send |
//! | [`lua_resend_with_different_bytes_is_refused_already_exists_and_releases`] / [`txn_resend_with_different_bytes_is_refused_already_exists_and_releases`] | T5 | same id, different digest: `ALREADY_EXISTS`, nothing written, the pre-flight increment released |
//! | [`same_operation_id_under_another_client_id_or_object_applies_normally`] | T6 | the id's scope is (object, `client_id`) |
//! | [`lua_concurrent_duplicates_commit_once`] / [`txn_concurrent_duplicates_commit_once`] | T7 | two simultaneous calls with one id: one `APPLIED`, one `ALREADY_COMMITTED`, one stream entry |
//! | [`update_without_an_operation_id_answers_unspecified`] | answer table | no id → `UNSPECIFIED`, and two such calls commit twice |
//! | [`lua_start_record_guard_refuses_a_call_its_client_has_moved_past`] / [`txn_start_record_guard_refuses_a_call_its_client_has_moved_past`] | T24 | a call whose start record predates its client's next commit — a late re-send of A, a never-committed C — is refused `UNAVAILABLE` (overtaken), applies nothing, ticks `overtaken{backend}`, and, refused on its first attempt, hands back its pre-flight increment (one `rsd<token>` member per call; GC returns `ref_count`) |
//! | [`lua_wrong_typed_record_key_is_refused_by_the_attempt_as_without_an_id`] / [`txn_wrong_typed_record_key_is_refused_by_the_attempt_as_without_an_id`] | T28 (`WRONGTYPE`) | a wrong-typed record key makes the start-record read give no record; the attempt's own `WRONGTYPE_KEY` refusal answers, byte-identical to an id-less call's |
//! | [`overtaken_call_logs_one_info_line_without_a_record_or_token`] | T31 | one INFO line naming the call; no record, token or digest anywhere in the log output |
//! | [`create_resend_is_answered_already_committed_and_allocates_nothing`] | T13, T15 (`MGET` hit) | the re-send gets the recorded `object_id`, the `object_id` counter does not move, the record's `PTTL` is in (0, 24 h] |
//! | [`resolve_create_layer_operation_answers_a_recorded_operation_and_nothing_else`] | T13 | the handler's pre-allocation read: `None` for an unknown id, the recorded answer for a known one |
//! | [`create_resend_after_the_layer_was_deleted_is_answered_without_recreating_it`] | answer table | a re-send after `DeleteLayer` answers the recorded id and leaves the layer deleted |
//! | [`create_resend_with_different_bytes_or_layer_is_refused_already_exists`] | T5 | same id, different digest (bytes or layer): `ALREADY_EXISTS`, nothing allocated, nothing mapped |
//! | [`create_resend_hands_back_its_preflight_increment`] | Invariant 4 | the `MGET`-hit answer releases the caller's pre-stored hashes |
//! | [`create_conflict_hands_back_its_preflight_increment`] | T5, Invariant 4 | a conflicting create answered at the `MGET` releases the caller's pre-stored hashes too |
//! | [`create_resend_redrive_does_not_recreate_a_reaped_replaced_stream`] | re-drive | the re-drive's `deleted` wake-up is `NOMKSTREAM`: a replaced object's stream that is gone stays gone |
//! | [`create_resend_redrive_retires_the_replaced_objects_admission_and_cache_entries`] | re-drive | the re-send retires the replaced object's admission entry and structural-cache entry on the store that answers it |
//! | [`commit_record_retract_leaves_a_field_holding_a_different_record`] | T12 (compare half) | the retract script removes the field only while it holds the torn call's exact record |
//! | [`operation_request_digest_matches_a_known_vector`] | digest format | lowercase-hex SHA-256 over `layer_name`, `0x00`, then the OMPF bytes, against an independently computed vector |
//! | [`create_without_an_operation_id_maps_and_indexes_the_layer_on_both_paths`] | T16 | the commit script still writes the mapping and the layers index for an id-less create, fresh and overwrite |
//! | [`create_with_a_wrong_typed_operation_record_is_refused_before_allocation`] | fix-1 `WRONGTYPE` | the handler's record read refuses a foreign-typed record as a keyspace refusal, ticks `{txn,create}`, allocates nothing |
//! | [`create_commit_script_refuses_a_wrong_typed_session_key`] | Invariant 6 backstop | the commit script's own type check refuses a foreign-typed record or layers index without moving the mapping |
//! | [`lost_reply::lua_resend_after_a_lost_reply_is_answered_already_committed`] | T1 (lost reply) | the original call answers the retriable `Aborted` (gRPC `UNAVAILABLE`) for a commit that landed; the re-send is answered from it |
//! | [`lost_reply::slow_path_resend_after_a_lost_exec_reply_is_answered_already_committed`] | T2 (lost reply) | the same with a `CreateSection`: the lost `EXEC` reply on the slow path (`lua-watch-structural`), the re-send on the default backend's structural route |
//! | [`lost_reply::lua_own_call_match_with_an_operation_id_answers_applied_and_releases_nothing`] / [`lost_reply::txn_own_call_match_with_an_operation_id_answers_applied_and_releases_nothing`] | T3, T4 | an in-call retry that finds its OWN commit under an `op.` key answers `APPLIED`, ticks `commit_already_landed_total`, not the new counter, and releases nothing |
//! | [`lost_reply::torn_commit_with_an_operation_id_retracts_its_record_so_a_resend_applies`] | T12 | a torn slow-path `EXEC` leaves no record carrying the call's token, so the re-send applies |
//! | [`lost_reply::torn_commit_record_retract_survives_a_dropped_call`] | T12 | the call is dropped while the retract is parked before its `EVALSHA`; the retract still runs and removes the record |
//! | [`lost_reply::lua_late_attempt_of_a_live_call_is_overtaken`] / [`lost_reply::txn_late_attempt_of_a_live_call_is_overtaken`] | T27 | X's attempt 1 lands with its reply lost; parked between attempts, X is re-sent (answered) and Y commits; X's attempt 2 is overtaken and applies nothing |
//! | [`lost_reply::lua_failed_start_record_read_ends_the_call_before_any_attempt`] / [`lost_reply::txn_failed_start_record_read_ends_the_call_before_any_attempt`] | T28 | a failed start-record read answers `UNAVAILABLE` with no commit dispatched, and is read exactly once |
//! | [`lost_reply::dropped_callers_commit_with_an_operation_id_is_not_replayed_after_a_reconnect`] / [`lost_reply::dropped_callers_commit_without_an_operation_id_is_not_replayed_after_a_reconnect`] | T25 | a dropped caller's `EVALSHA`, queued behind a stalled connection, is not re-run after the reconnect — after its client's next write. The no-id variant is the one that discriminates the once-only dispatch |
//! | [`lost_reply::commit_evalsha_is_written_once_so_a_killed_connection_fails_the_attempt`] | T32 | the commit's one-attempt budget reaches fred: a killed connection fails attempt 1 and the call's own loop dispatches attempt 2, rather than fred replaying attempt 1 |
//! | [`lost_reply::create_duplicate_that_commits_first_is_answered_from_the_commit_script`] | T15 (script race) | a duplicate whose commit script meets the winner's record answers the winner's id, writes nothing, and releases nothing (its object is an orphan) |
//! | [`overwrite_second_try::overwrite_second_try_commits_after_each_transient_shape_of_a_failed_first_try`] | R4-1 | an overwrite whose first commit try fails before its script runs — `IO`, and the three `txn_pool` redirect / budget shapes — commits on its second try and the mapping names the new object; the last one then takes an update |
//! | [`overwrite_second_try::overwrite_second_try_whose_reply_is_lost_succeeds_through_the_tail`] | R4-1 | the second try's reply is lost: the mapping check finds the new object, INFO with `second_try = true`, the overwrite tail runs |
//! | [`overwrite_second_try::overwrite_second_try_that_meets_its_own_record_answers_applied_through_the_tail`] | R4-1 | the second try meets the call's own operation record (staged: a lost first-try reply and a check that read the old mapping and no claim or record): `APPLIED`, the own-record INFO, the overwrite tail runs |
//! | [`overwrite_second_try::overwrite_first_try_whose_reply_is_lost_succeeds_without_a_second_try`] | R4-1 | the first try's reply is lost: no second dispatch, no WARN, INFO with `second_try = false` |
//! | [`overwrite_second_try::overwrite_failing_on_both_tries_logs_one_error_and_leaves_the_old_object_mapped`] | R4-1, b.ua9 | both tries fail: one ERROR (run budget spent; `consequence`: not landed), the old object stays mapped and refuses updates `FAILED_PRECONDITION` (the finisher given no budget) |
//! | [`overwrite_second_try::overwrite_whose_layer_was_taken_by_another_create_re_plans_against_it_and_commits`] | b.ua9 (I2) | a third object holds the mapping after the first try: one re-plan INFO, the third object fenced naming this call's, the next run commits |
//! | [`overwrite_second_try::overwrite_whose_layer_was_taken_between_the_tries_re_plans_and_commits`] | b.ua9 (I2) | a third object takes the mapping between the tries: the second-try WARN, one re-plan INFO, the next run commits |
//! | [`overwrite_second_try::overwrite_whose_mapping_check_fails_logs_one_error_that_the_outcome_is_unknown`] | R4-1 | the check's `MGET` fails: one ERROR (`consequence`: unknown), no second try |
//! | [`overwrite_second_try::new_layer_create_whose_first_run_fails_commits_on_a_second_run`] | b.ua9 test 20 | a new layer's failed first run is tried once more and commits |
//! | [`overwrite_second_try::overwrite_run_that_meets_its_own_record_after_a_re_plan_answers_applied`] | b.ua9 test 8 | a record carrying the call's object id is the call's own, though the re-plan changed its previous id |
//!
//! Elsewhere: T8, T11's handler half, T13/T14/T20 through the handler, and
//! the creation-base spawns of the handler's record-read answer and of T15 in
//! `tests/integration/grpc/b_v6d_operation_id_test.rs`;
//! T9 (record parity) in `update_layer_lua_test.rs`; T10 (record verdicts)
//! beside `parse_commit_token_record` in `src/storage/layer.rs`; T11's
//! validator cases in `src/domain/validation.rs`; T17 in `src/storage/keys.rs`;
//! T18's zero-seeded cells in `tests/integration/observability_metrics_test.rs`
//! and its log field in `tests/integration/grpc/rpc_failure_log_test.rs`;
//! T21-T23 in `tests/integration/checkpoint/b_v6d_stamp_floor_test.rs`; T26
//! (the start record on the script itself) in `update_layer_lua_test.rs`; T29
//! and T30 (P6) through the handler in
//! `tests/integration/grpc/b_v6d_operation_id_test.rs`; T31's and T39's
//! zero-seeded cells in `tests/integration/observability_metrics_test.rs`; the
//! router tests T33-T38 in `b_v6d_router_test.rs` (T37 beside
//! `router_restart_delay` in `src/storage/router_supervisor.rs`). For
//! Amendment R4-1, the overwrite second try through the handlers (an
//! `UpdateLayer` of the new object, with and without an operation id) is in
//! `b_v6d_operation_id_test.rs`, and the real `-ASK` / `-MOVED` with the
//! routing sync before the second try in `b_v6d_router_test.rs`.
//!
//! # Feature gate
//!
//! The `lost_reply` module drives `LayerStore`'s `test-fault-injection` seams
//! (the post-`EVALSHA` lost-reply injector, the watched-`EXEC` override and
//! reconcile `XRANGE` injector, the pre-dispatch hooks of both commit paths,
//! the between-attempts hook, the start-record read injector, the torn-commit
//! retract hook, the create pre-commit hook) and is compiled out — not skipped —
//! without `--features test-fault-injection`; so is `overwrite_second_try` (the
//! create commit-dispatch fault injector and the create pre-commit hook).
//! Everything above them needs only Valkey, and the digest vector needs
//! nothing.
//!
//! # Wait budget
//!
//! Every `update_layer_with_operation` / `create_layer_with_operation` call
//! under test is wrapped in [`CALL_BUDGET`] (15 s); the healthy path finishes in
//! milliseconds, and no test awaits more than three such calls (45 s; a call a
//! hook makes runs inside its caller's budget) except the R4-1 shapes test,
//! which awaits four overwrites and one update (75 s; its client connects
//! once, through the fixture). The R4-1 proxy tests also connect one client
//! through the proxy (the fixture's connect timeout). The re-drive tests add one
//! `wait_for_cleanup` bounded by the same budget (three calls plus that wait,
//! 60 s). The dropped-call retract test awaits no call budget — it aborts its
//! call — and instead waits twice on `RETRACT_WAIT` (10 s): for the retract to
//! park, then for the record to go (20 s). The T25 replay tests are the
//! longest: `ConnStall::park`'s wait for the `BLPOP` to block
//! (`test_support::CONN_STALL_WAIT`, 10 s), two `RETRACT_WAIT`s for the
//! dropped call's `EVALSHA` to queue (20 s), at most two calls (30 s), the
//! kill's settle (`CONN_STALL_WAIT`, 10 s) and one `RETRACT_WAIT` for the
//! reconnect (10 s, plus its last 1 s `PING`): about 81 s. On top of that come
//! the steps bounded only by fred's 5 s command timeout — the `with_id`
//! variant's call-start read, the stall's `CLIENT ID` and `CLIENT KILL`, and
//! one `CLIENT LIST` probe that may overrun each of the two probe loops'
//! deadlines — for at most about 106 s. Every test stays under nextest's 120 s
//! hang-killer, so every diagnostic panic is printed.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface, StreamsInterface};

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_sum, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::external_values::{process_pending_gc, ValkeyExternalValueStore};
use live_edit::storage::keys::{sha256_to_hex, CREATE_LAYER_OPERATION_TTL_MS};
use live_edit::storage::layer::{
    operation_request_digest, CreateCommit, CreateOperation, LayerStore, Operation,
    OperationOutcome, StartRecord, UpdateCommit, OPERATION_CONFLICT_MESSAGE,
    OPERATION_OVERTAKEN_MESSAGE,
};
use live_edit::storage::lua::COMMIT_RECORD_RETRACT_SCRIPT;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::{relay_ompf, RelayedUpdates};
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandCreateSection, OwnedDelta, StructCommand};

use crate::common::owned_delta::{delta_with_set_field_external, delta_with_set_field_inline};
use crate::storage::b_qpf_tests::{
    assert_blob_alive, ext_hash, ext_hash_vec, ext_store, pending_members, pre_store, ref_count,
    EXT_SIZE, SECTION_ID,
};
use crate::storage::structural_cache_retirement_test::{assert_retired, cache_entry_for};

// ═════════════════════════════════════════════════════════════════════════════
// Budgets and contract constants
// ═════════════════════════════════════════════════════════════════════════════

/// Deadline on one storage call. The healthy path is a handful of round trips;
/// 15 s is orders of magnitude of headroom for CI runner load while keeping the
/// worst case tabulated in the module header under nextest's 120 s kill.
const CALL_BUDGET: Duration = Duration::from_secs(15);

/// Retry budget for every store here, as in `b_75i_tests`: small, so a
/// regression that loops on `Aborted` fails inside the call budget.
const MAX_ATTEMPTS: u32 = 8;

/// The writer whose re-send each scenario is about.
const WRITER: &str = "bv6d-writer";

/// A second writer, with a different `client_id`.
const OTHER_WRITER: &str = "bv6d-other-writer";

/// The client operation id the scenarios re-send. A UUIDv4 in canonical text,
/// the shape the proto comment recommends.
const OPERATION_ID: &str = "6f1d2c3b-4a59-4e8d-9b7a-0c1e2f3a4b5c";

/// The original call's client timestamp, and the re-send's. A re-send may
/// carry a fresh timestamp; the digest covers the bytes, not the header.
const ORIGINAL_TS: u64 = 1_000;
const RESEND_TS: u64 = 2_000;

/// The request bytes the scenarios' digests are computed over. At this layer
/// the digest is the caller's (the gRPC handler hashes the streamed OMPF), so
/// any stable bytes model "the same request"; [`OTHER_PAYLOAD`] models a
/// different one.
const PAYLOAD: &[u8] = b"b.v6d original request bytes";
const OTHER_PAYLOAD: &[u8] = b"b.v6d a different request";

/// `live_edit_operation_id_match_total` label values. Restated, not imported:
/// they are the operator-facing contract, and a test reading the emitter's own
/// constants could not fail when one was renamed.
const RPC_UPDATE: &str = "update_layer";
const RPC_CREATE: &str = "create_layer";
const OUTCOME_ALREADY_COMMITTED: &str = "already_committed";
const OUTCOME_CONFLICT: &str = "conflict";

/// `live_edit_ext_ref_released_total{reason}` for a handed-back pre-flight
/// increment.
const REASON_SURPLUS_ACQUIRE: &str = "surplus_acquire";

// ═════════════════════════════════════════════════════════════════════════════
// Fixture + helpers
// ═════════════════════════════════════════════════════════════════════════════

struct Fixture {
    tv: TestValkey,
    metrics: Arc<Metrics>,
    ext: Arc<ValkeyExternalValueStore>,
    session_id: String,
    layer: &'static str,
    object_id: u64,
}

/// A fresh object at `seq = 1` (stream `1-0`), created with no operation id.
async fn fixture(layer: &'static str) -> Fixture {
    fixture_on(TestValkey::setup().await, layer).await
}

async fn fixture_on(tv: TestValkey, layer: &'static str) -> Fixture {
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-v6d"));
    let ext = ext_store(&tv);
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(layer, &format!("/{layer}"), "bv6d-user")
        .await
        .expect("session creation must succeed");
    let object_id = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .create_layer(&session_id, layer, "bv6d-user", &base_state(), ext.clone())
        .await
        .expect("create_layer must succeed");
    Fixture {
        tv,
        metrics,
        ext,
        session_id,
        layer,
        object_id,
    }
}

/// The state a fresh layer is created with.
fn base_state() -> ObjectState {
    ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        next_set_order: 0,
        ..Default::default()
    }
}

/// A store on `backend`, reporting into the fixture's registry.
fn store(fx: &Fixture, backend: UpdateLayerBackend) -> LayerStore {
    LayerStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone(), 4096)
        .with_update_layer_backend(backend)
        .with_update_layer_retry_max_attempts(MAX_ATTEMPTS)
        .with_metrics(fx.metrics.clone())
}

/// A field-only delta: the fast path on the Lua backend, the slow path on Txn.
fn field_delta(value: u8) -> OwnedDelta {
    delta_with_set_field_inline(SECTION_ID, "color", vec![value], 0)
}

/// A delta carrying one `CreateSection`: the structural route on the `lua`
/// backend, the WATCH/MULTI/EXEC slow path on `lua-watch-structural` and
/// `txn`, and the shape a blind re-apply turns into a `DuplicateSection`
/// refusal of a landed edit.
fn section_delta() -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: 1,
            section_name: "BV6dSection".to_owned(),
            section_type: 1,
        })],
        ..Default::default()
    }
}

/// A field referencing external hash `[byte; 32]`.
fn external_delta(byte: u8, set_order: u64) -> OwnedDelta {
    delta_with_set_field_external(
        SECTION_ID,
        "color",
        ext_hash_vec(byte),
        EXT_SIZE,
        1,
        set_order,
    )
}

/// The operation an update re-send of `payload` carries on the fixture's layer.
fn update_op(fx: &Fixture, payload: &[u8]) -> Operation {
    Operation::new(OPERATION_ID, operation_request_digest(fx.layer, payload))
}

/// The operation a create of `payload` carries on `layer`.
fn create_op(layer: &str, payload: &[u8]) -> CreateOperation {
    CreateOperation::new(OPERATION_ID, operation_request_digest(layer, payload))
}

/// One `update_layer_with_operation` call on the fixture's object, under
/// [`CALL_BUDGET`].
async fn update(
    store: &LayerStore,
    fx: &Fixture,
    delta: OwnedDelta,
    client_id: &str,
    client_ts: u64,
    operation: Option<Operation>,
) -> Result<UpdateCommit, ServiceError> {
    tokio::time::timeout(
        CALL_BUDGET,
        store.update_layer_with_operation_relayed(
            &fx.session_id,
            fx.layer,
            fx.object_id,
            Some(delta),
            client_id,
            client_ts,
            operation,
        ),
    )
    .await
    .expect("update_layer_with_operation did not settle within the call budget")
}

/// One `create_layer_with_operation` call, under [`CALL_BUDGET`].
async fn create(
    store: &LayerStore,
    fx: &Fixture,
    layer: &str,
    operation: Option<CreateOperation>,
) -> Result<CreateCommit, ServiceError> {
    tokio::time::timeout(
        CALL_BUDGET,
        store.create_layer_with_operation(
            &fx.session_id,
            layer,
            "bv6d-user",
            &base_state(),
            fx.ext.clone(),
            operation,
        ),
    )
    .await
    .expect("create_layer_with_operation did not settle within the call budget")
}

async fn header_seq(fx: &Fixture, object_id: u64) -> u64 {
    let raw: Option<String> = fx
        .tv
        .client()
        .pool
        .hget(fx.tv.key_schema().obj_header(&object_id.to_string()), "seq")
        .await
        .expect("HGET obj:hdr seq must succeed");
    raw.expect("the header must carry seq")
        .parse()
        .expect("header seq must be numeric")
}

/// The per-object stream as `(id, payload)`, in order.
async fn stream_payloads(fx: &Fixture, object_id: u64) -> Vec<(String, String)> {
    // Values read as bytes: a commit's relay `ompf` field is binary.
    let entries: Vec<(String, HashMap<String, bytes::Bytes>)> = fx
        .tv
        .client()
        .pool
        .xrange_values(fx.tv.key_schema().layer_stream(object_id), "-", "+", None)
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

/// Assert the object's stream is exactly CreateLayer's `1-0` plus one entry per
/// `(seq, client_id)` in `expected`, in order; return those entries' payloads.
async fn assert_stream_commits(
    fx: &Fixture,
    object_id: u64,
    expected: &[(u64, &str)],
    why: &str,
) -> Vec<String> {
    let entries = stream_payloads(fx, object_id).await;
    let ids: Vec<&str> = entries.iter().map(|(id, _)| id.as_str()).collect();
    let mut want = vec!["1-0".to_owned()];
    want.extend(expected.iter().map(|(seq, _)| format!("{seq}-0")));
    assert_eq!(
        ids,
        want.iter().map(String::as_str).collect::<Vec<_>>(),
        "{why}. Full stream: {entries:?}"
    );
    entries
        .iter()
        .skip(1)
        .zip(expected)
        .map(|((id, payload), (_, client))| {
            assert_eq!(
                payload.split('|').nth(3),
                Some(*client),
                "entry {id} must name the writer that committed it; got {payload:?}"
            );
            payload.clone()
        })
        .collect()
}

/// The server timestamp published in an `update|…` stream payload.
fn payload_server_ts(payload: &str) -> u64 {
    payload
        .split('|')
        .nth(5)
        .and_then(|ts| ts.parse().ok())
        .unwrap_or_else(|| panic!("update payload must carry a numeric server_ts; got {payload:?}"))
}

/// `live_edit_operation_id_match_total{rpc, outcome}`. Panics on an absent
/// series: all four cells are zero-seeded in `Metrics::new`, so absence means
/// the store under test never received this metrics handle.
fn operation_matches(metrics: &Metrics, rpc: &str, outcome: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_operation_id_match_total",
        &[("rpc", rpc), ("outcome", outcome)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_operation_id_match_total{{rpc=\"{rpc}\",outcome=\"{outcome}\"}} is absent \
             from the exposition — the store under test carries no metrics handle, so every \
             count read off it is vacuous"
        )
    }) as u64
}

/// `live_edit_update_layer_operation_overtaken_total{backend}`, as `[lua, txn]`.
/// Panics on an absent series: both cells are zero-seeded in `Metrics::new`.
fn overtaken_cells(metrics: &Metrics) -> [u64; 2] {
    let text = render(metrics);
    ["lua", "txn"].map(|backend| {
        sample_value(
            &text,
            "live_edit_update_layer_operation_overtaken_total",
            &[("backend", backend)],
        )
        .unwrap_or_else(|| panic!("the overtaken {backend} cell is seeded in Metrics::new"))
            as u64
    })
}

/// `[lua, txn]` with `n` on the cell of the path `backend` commits a field
/// delta on: the Lua fast path under `lua` and `lua-watch-structural`.
fn overtaken_only_on(backend: UpdateLayerBackend, n: u64) -> [u64; 2] {
    match backend {
        UpdateLayerBackend::Lua | UpdateLayerBackend::LuaWatchStructural => [n, 0],
        UpdateLayerBackend::Txn => [0, n],
    }
}

/// The start record a call starting NOW would carry: one call-start read of
/// `WRITER`'s field, exactly as the store does it.
async fn start_record_now(store: &LayerStore, fx: &Fixture) -> StartRecord {
    store
        .resolve_start_record(store.spawn_start_record_read(fx.object_id, WRITER))
        .await
        .expect("the start-record read must succeed")
}

/// An operation on the fixture's layer with id `operation_id` over `payload`.
fn named_op(fx: &Fixture, operation_id: &str, payload: &[u8]) -> Operation {
    Operation::new(operation_id, operation_request_digest(fx.layer, payload))
}

/// Assert `result` is the overtaken refusal: `UNAVAILABLE` carrying
/// [`OPERATION_OVERTAKEN_MESSAGE`].
fn assert_overtaken(result: &Result<UpdateCommit, ServiceError>, what: &str) {
    match result {
        Err(ServiceError::Unavailable(msg)) => assert_eq!(
            msg, OPERATION_OVERTAKEN_MESSAGE,
            "{what}: the refusal must carry the overtaken message"
        ),
        other => panic!(
            "{what}: an attempt that would apply after its client's next commit must be refused \
             UNAVAILABLE (overtaken); got {other:?}"
        ),
    }
}

/// All four `operation_id_match_total` cells, in a fixed order.
fn operation_match_cells(metrics: &Metrics) -> [u64; 4] {
    [
        operation_matches(metrics, RPC_UPDATE, OUTCOME_ALREADY_COMMITTED),
        operation_matches(metrics, RPC_UPDATE, OUTCOME_CONFLICT),
        operation_matches(metrics, RPC_CREATE, OUTCOME_ALREADY_COMMITTED),
        operation_matches(metrics, RPC_CREATE, OUTCOME_CONFLICT),
    ]
}

/// `live_edit_update_layer_relay_bytes` (b.yvh) as `(_count, _sum)`, `(0, 0)`
/// before the first sample (a histogram has no series until then), beside the
/// attempts histogram's `_count` summed over `path`: the relay-bytes histogram
/// samples exactly the commits the attempts histogram counts.
fn relay_bytes_samples(metrics: &Metrics) -> (u64, u64, u64) {
    let text = render(metrics);
    let read = |name: &str| sample_sum(&text, name, &[]).unwrap_or(0.0) as u64;
    (
        read("live_edit_update_layer_relay_bytes_count"),
        read("live_edit_update_layer_relay_bytes_sum"),
        read("live_edit_update_layer_attempts_histogram_count"),
    )
}

/// `live_edit_update_layer_commit_already_landed_total{backend}`.
fn already_landed(metrics: &Metrics, backend: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_update_layer_commit_already_landed_total",
        &[("backend", backend)],
    )
    .expect("commit_already_landed_total cells are seeded at zero") as u64
}

/// `live_edit_ext_ref_released_total{reason}`.
fn ext_ref_released(metrics: &Metrics, reason: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_ext_ref_released_total",
        &[("reason", reason)],
    )
    .expect("ext_ref_released_total cells are seeded at zero") as u64
}

/// `live_edit_keyspace_type_error_total{backend, site}`, summed over the
/// `mode` series the `{backend="lua", site="update"}` cell is split into.
fn keyspace_type_errors(metrics: &Metrics, backend: &str, site: &str) -> u64 {
    sample_sum(
        &render(metrics),
        "live_edit_keyspace_type_error_total",
        &[("backend", backend), ("site", site)],
    )
    .expect("keyspace_type_error_total cells are seeded at zero") as u64
}

/// Assert `le:ext:{gc}:pending` holds exactly one member, and that it is a
/// re-send release (`ReleaseEvent::Resend`) of `hash` against `object_id`:
/// `{hash_hex}:{object_id}:rsd<32 lowercase hex>`. The call token is random,
/// so its shape is checked rather than its value.
async fn assert_single_resend_release(fx: &Fixture, hash: [u8; 32], object_id: u64, why: &str) {
    assert_resend_releases(fx, hash, object_id, 1, why).await;
}

/// Assert `le:ext:{gc}:pending` holds exactly `calls` members, each a
/// re-send release of `hash` against `object_id` under a different call
/// token: `{hash_hex}:{object_id}:rsd<32 lowercase hex>`, one per call that
/// handed back its pre-flight increment. The tokens are random, so their
/// shape and distinctness are checked rather than their values.
async fn assert_resend_releases(
    fx: &Fixture,
    hash: [u8; 32],
    object_id: u64,
    calls: usize,
    why: &str,
) {
    let members = pending_members(&fx.tv).await;
    assert_eq!(
        members.len(),
        calls,
        "{why}: exactly {calls} owed release(s) must be pending; got {members:?}"
    );
    let prefix = format!("{}:{object_id}:rsd", sha256_to_hex(&hash));
    for member in &members {
        let token = member.strip_prefix(&prefix).unwrap_or_else(|| {
            panic!(
                "{why}: the release must be `{prefix}<token>` (a re-send release); got {member:?}"
            )
        });
        assert!(
            token.len() == 32
                && token
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "{why}: the re-send release's event carries the call's 32-hex commit token; got \
             {member:?}"
        );
    }
}

/// `GET` of the session's mapping for `layer`, parsed.
async fn mapping(fx: &Fixture, layer: &str) -> Option<u64> {
    let raw: Option<String> = fx
        .tv
        .client()
        .pool
        .get(fx.tv.key_schema().layer_mapping(&fx.session_id, layer))
        .await
        .expect("GET layer mapping must succeed");
    raw.map(|v| v.parse().expect("the mapping holds a decimal object id"))
}

/// The object-id counter's current value.
async fn object_id_counter(fx: &Fixture) -> u64 {
    let raw: Option<String> = fx
        .tv
        .client()
        .pool
        .get(fx.tv.key_schema().counter_object_id())
        .await
        .expect("GET object-id counter must succeed");
    raw.expect("the fixture's own create allocated, so the counter exists")
        .parse()
        .expect("the counter is decimal")
}

/// Whether `object_id`'s header exists.
async fn header_exists(fx: &Fixture, object_id: u64) -> bool {
    fx.tv
        .client()
        .pool
        .exists(fx.tv.key_schema().obj_header(&object_id.to_string()))
        .await
        .expect("EXISTS obj:hdr must succeed")
}

// ═════════════════════════════════════════════════════════════════════════════
// UpdateLayer — T1 / T2: a re-send of a landed operation is answered, not
// applied
// ═════════════════════════════════════════════════════════════════════════════

/// Call A carries operation X and commits. Call B — a NEW call: its own random
/// token, a fresh client timestamp, the same id and the same request digest —
/// is answered `ALREADY_COMMITTED` with A's sequence and server timestamp, and
/// writes nothing: the header and the stream still show one commit.
///
/// Pre-fix B was simply a second commit at `seq + 1` (and, for a
/// `CreateSection`, a `DuplicateSection` refusal of a landed edit).
///
/// `original_path` is the commit path A must be recorded on
/// (`live_edit_update_layer_attempts_histogram{path}`), so each caller's
/// claim about which route it exercises is checked, not assumed.
async fn resend_of_a_landed_operation_is_answered_already_committed(
    backend: UpdateLayerBackend,
    delta: fn() -> OwnedDelta,
    original_path: &'static str,
    layer: &'static str,
) {
    let fx = fixture(layer).await;
    let store = store(&fx, backend);

    let first = update(
        &store,
        &fx,
        delta(),
        WRITER,
        ORIGINAL_TS,
        Some(update_op(&fx, PAYLOAD)),
    )
    .await
    .expect("the original call must commit");
    assert_eq!(
        (first.seq, first.outcome),
        (2, OperationOutcome::Applied),
        "the original call commits at seq 2 and records its operation"
    );
    assert_eq!(
        sample_value(
            &render(&fx.metrics),
            "live_edit_update_layer_attempts_histogram_count",
            &[("path", original_path)],
        ),
        Some(1.0),
        "fixture: the original call committed on the {original_path} path"
    );

    let resent = update(
        &store,
        &fx,
        delta(),
        WRITER,
        RESEND_TS,
        Some(update_op(&fx, PAYLOAD)),
    )
    .await
    .expect(
        "a re-send of a landed operation must be answered OK. An INVALID_ARGUMENT here is the \
         re-send re-applying a CreateSection the original already made",
    );
    assert_eq!(
        resent,
        UpdateCommit {
            seq: first.seq,
            server_ts: first.server_ts,
            outcome: OperationOutcome::AlreadyCommitted,
        },
        "the re-send must report the ORIGINAL commit's (seq, server_ts), flagged \
         ALREADY_COMMITTED. A seq of 3 is the delta applied a second time"
    );

    assert_eq!(
        header_seq(&fx, fx.object_id).await,
        2,
        "the header advanced once"
    );
    let payloads = assert_stream_commits(
        &fx,
        fx.object_id,
        &[(2, WRITER)],
        "exactly one commit entry — a 3-0 is the duplicate commit subscribers would see",
    )
    .await;
    assert_eq!(
        resent.server_ts,
        payload_server_ts(&payloads[0]),
        "the re-send's timestamp is the one the commit published to subscribers"
    );
    assert_eq!(
        operation_match_cells(&fx.metrics),
        [1, 0, 0, 0],
        "exactly one update re-send was answered from an earlier call's record"
    );
    assert_eq!(
        (
            already_landed(&fx.metrics, "lua"),
            already_landed(&fx.metrics, "txn")
        ),
        (0, 0),
        "an earlier CALL's commit is not an in-call already-landed match"
    );
    let original_relay = relay_ompf(&delta()).len() as u64;
    assert_eq!(
        relay_bytes_samples(&fx.metrics),
        (1, original_relay, 1),
        "b.yvh: the original call sampled its commit's relay bytes once; the re-send's \
         ALREADY_COMMITTED answer reports that commit and must not sample it again"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lua_resend_of_a_landed_operation_is_answered_already_committed() {
    resend_of_a_landed_operation_is_answered_already_committed(
        UpdateLayerBackend::Lua,
        || field_delta(0x11),
        "fast",
        "bv6d-t1-lua",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_resend_of_a_landed_operation_is_answered_already_committed() {
    resend_of_a_landed_operation_is_answered_already_committed(
        UpdateLayerBackend::Txn,
        || field_delta(0x12),
        "watch",
        "bv6d-t1-txn",
    )
    .await;
}

/// T2: a `CreateSection` — the structural route on the default `lua` backend
/// — is the shape whose blind re-apply is REFUSED rather than duplicated — so
/// pre-fix the client was told its landed edit failed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_resend_of_a_landed_create_section_is_answered_not_refused() {
    resend_of_a_landed_operation_is_answered_already_committed(
        UpdateLayerBackend::Lua,
        section_delta,
        "structural_full",
        "bv6d-t2-section",
    )
    .await;
}

/// Without an operation id the outcome is `UNSPECIFIED`, and a byte-identical
/// second call is a second commit — the b.75i contract is unchanged for
/// clients that send no id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_without_an_operation_id_answers_unspecified() {
    let fx = fixture("bv6d-no-id-update").await;
    let store = store(&fx, UpdateLayerBackend::Lua);

    let first = update(&store, &fx, field_delta(0x13), WRITER, ORIGINAL_TS, None)
        .await
        .expect("the first id-less call must commit");
    let second = update(&store, &fx, field_delta(0x13), WRITER, ORIGINAL_TS, None)
        .await
        .expect("the second id-less call must commit");

    assert_eq!(
        [(first.seq, first.outcome), (second.seq, second.outcome)],
        [
            (2, OperationOutcome::Unspecified),
            (3, OperationOutcome::Unspecified)
        ],
        "id-less calls are answered UNSPECIFIED and each commits"
    );
    assert_eq!(operation_match_cells(&fx.metrics), [0, 0, 0, 0]);
}

// ═════════════════════════════════════════════════════════════════════════════
// T4 — an earlier-call answer hands back the re-send's pre-flight increment
// ═════════════════════════════════════════════════════════════════════════════

/// The gRPC handler pre-stores (and so `+1`s) every distinct external hash a
/// delta references BEFORE the commit path decides anything. A re-send that is
/// answered from an earlier call's record commits nothing, so its `+1` is owed
/// back: one `surplus_acquire` release at `ReleaseEvent::Resend(token)`.
/// After one GC tick `ref_count` is back at its value before the re-send, and
/// the blob — still referenced by the object — survives.
///
/// Without the hand-back the re-send's `+1` would be stranded: `ref_count`
/// stays one above the number of referencing objects for as long as the
/// reconciler takes to notice.
async fn earlier_call_answer_hands_back_the_resends_preflight_increment(
    backend: UpdateLayerBackend,
    layer: &'static str,
) {
    let fx = fixture(layer).await;
    let store = store(&fx, backend);
    let h = ext_hash(0xC4);

    pre_store(&fx.ext, &[h]).await;
    update(
        &store,
        &fx,
        external_delta(0xC4, 0),
        WRITER,
        ORIGINAL_TS,
        Some(update_op(&fx, PAYLOAD)),
    )
    .await
    .expect("the original external commit must land");
    let before_resend = ref_count(&fx.tv, h).await;
    assert_eq!(
        before_resend,
        Some(1),
        "fixture precondition: one object references the hash"
    );
    assert!(
        pending_members(&fx.tv).await.is_empty(),
        "fixture precondition: a first reference owes no release"
    );

    pre_store(&fx.ext, &[h]).await;
    let released_before = ext_ref_released(&fx.metrics, REASON_SURPLUS_ACQUIRE);
    let resent = update(
        &store,
        &fx,
        external_delta(0xC4, 0),
        WRITER,
        RESEND_TS,
        Some(update_op(&fx, PAYLOAD)),
    )
    .await
    .expect("the re-send must be answered");
    assert_eq!(resent.outcome, OperationOutcome::AlreadyCommitted);

    assert_single_resend_release(
        &fx,
        h,
        fx.object_id,
        "the answered re-send owes back its pre-flight increment",
    )
    .await;
    assert_eq!(
        ext_ref_released(&fx.metrics, REASON_SURPLUS_ACQUIRE),
        released_before + 1,
        "the hand-back is counted under reason=\"surplus_acquire\""
    );

    let stats = process_pending_gc(
        fx.ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.processed, 1, "one owed decrement, one claim");
    assert_eq!(
        ref_count(&fx.tv, h).await,
        before_resend,
        "after one GC tick ref_count must be back at its value before the re-send"
    );
    assert_blob_alive(&fx.tv, h, "the hash the object still references").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lua_earlier_call_answer_hands_back_the_resends_preflight_increment() {
    earlier_call_answer_hands_back_the_resends_preflight_increment(
        UpdateLayerBackend::Lua,
        "bv6d-t4-lua",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_earlier_call_answer_hands_back_the_resends_preflight_increment() {
    earlier_call_answer_hands_back_the_resends_preflight_increment(
        UpdateLayerBackend::Txn,
        "bv6d-t4-txn",
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// T5 — the same id over different bytes is refused, with nothing written
// ═════════════════════════════════════════════════════════════════════════════

/// An id reused for a different request (a different digest) is a client bug,
/// and guessing either way is wrong: answering the earlier commit would report
/// an edit the client never made, applying would make the id mean two things.
/// So the call is refused `ALREADY_EXISTS`, nothing is written, the `conflict`
/// cell ticks — and its pre-flight increment is handed back like a re-send's,
/// because it committed nothing either.
async fn resend_with_different_bytes_is_refused_already_exists_and_releases(
    backend: UpdateLayerBackend,
    layer: &'static str,
) {
    let fx = fixture(layer).await;
    let store = store(&fx, backend);
    let h = ext_hash(0xC5);

    update(
        &store,
        &fx,
        field_delta(0x15),
        WRITER,
        ORIGINAL_TS,
        Some(update_op(&fx, PAYLOAD)),
    )
    .await
    .expect("the original call must commit");

    pre_store(&fx.ext, &[h]).await;
    let refused = update(
        &store,
        &fx,
        external_delta(0xC5, 1),
        WRITER,
        RESEND_TS,
        Some(update_op(&fx, OTHER_PAYLOAD)),
    )
    .await;
    match refused {
        Err(ServiceError::AlreadyExists(msg)) => assert_eq!(msg, OPERATION_CONFLICT_MESSAGE),
        other => panic!(
            "the same operation id over a different request must be refused ALREADY_EXISTS; got \
             {other:?}"
        ),
    }

    assert_eq!(header_seq(&fx, fx.object_id).await, 2, "nothing committed");
    assert_stream_commits(
        &fx,
        fx.object_id,
        &[(2, WRITER)],
        "no entry for the refusal",
    )
    .await;
    assert_eq!(
        operation_match_cells(&fx.metrics),
        [0, 1, 0, 0],
        "exactly one update conflict"
    );
    assert_single_resend_release(
        &fx,
        h,
        fx.object_id,
        "the refused call committed nothing, so its pre-flight increment is owed back",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lua_resend_with_different_bytes_is_refused_already_exists_and_releases() {
    resend_with_different_bytes_is_refused_already_exists_and_releases(
        UpdateLayerBackend::Lua,
        "bv6d-t5-lua",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_resend_with_different_bytes_is_refused_already_exists_and_releases() {
    resend_with_different_bytes_is_refused_already_exists_and_releases(
        UpdateLayerBackend::Txn,
        "bv6d-t5-txn",
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// T6 — the id's scope is (object, client_id)
// ═════════════════════════════════════════════════════════════════════════════

/// The record is one field per `client_id` on one object, so the same id under
/// a different `client_id`, or on a different object, matches nothing and
/// applies normally. Answering across either boundary would report another
/// writer's (or another layer's) commit as this call's.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn same_operation_id_under_another_client_id_or_object_applies_normally() {
    let fx = fixture("bv6d-t6-scope").await;
    let store = store(&fx, UpdateLayerBackend::Lua);
    let op = update_op(&fx, PAYLOAD);

    update(
        &store,
        &fx,
        field_delta(0x16),
        WRITER,
        ORIGINAL_TS,
        Some(op.clone()),
    )
    .await
    .expect("the original call must commit");

    let other_writer = update(
        &store,
        &fx,
        field_delta(0x16),
        OTHER_WRITER,
        ORIGINAL_TS,
        Some(op.clone()),
    )
    .await
    .expect("another client_id's call must commit");
    assert_eq!(
        (other_writer.seq, other_writer.outcome),
        (3, OperationOutcome::Applied),
        "the same id under a different client_id is a different write intent"
    );

    // A second object: the same layer name in a second session.
    let second_session = SessionStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone())
        .create("bv6d-t6-second", "/bv6d-t6-second", "bv6d-user")
        .await
        .expect("second session creation must succeed");
    let second_object = store
        .create_layer(
            &second_session,
            fx.layer,
            "bv6d-user",
            &base_state(),
            fx.ext.clone(),
        )
        .await
        .expect("second create_layer must succeed");
    let other_object = tokio::time::timeout(
        CALL_BUDGET,
        store.update_layer_with_operation_relayed(
            &second_session,
            fx.layer,
            second_object,
            Some(field_delta(0x16)),
            WRITER,
            ORIGINAL_TS,
            Some(op),
        ),
    )
    .await
    .expect("the second-object call did not settle within the call budget")
    .expect("the same id on another object must commit");
    assert_eq!(
        (other_object.seq, other_object.outcome),
        (2, OperationOutcome::Applied),
        "the same id on a different object is a different write intent"
    );

    assert_stream_commits(
        &fx,
        fx.object_id,
        &[(2, WRITER), (3, OTHER_WRITER)],
        "both writers committed to the first object",
    )
    .await;
    assert_eq!(operation_match_cells(&fx.metrics), [0, 0, 0, 0]);
}

// ═════════════════════════════════════════════════════════════════════════════
// T7 — concurrent duplicates commit once
// ═════════════════════════════════════════════════════════════════════════════

/// Two calls carrying the same id, released together by a barrier. Exactly one
/// commits; the other — whichever loses — is answered from the winner's record:
/// one `APPLIED`, one `ALREADY_COMMITTED`, both at the same sequence, one
/// stream entry.
///
/// On the Lua backend the script is atomic, so the loser's step 1b sees the
/// record. On the slow path the loser's `EXEC` is voided by the winner's
/// header write (`WATCH`) and its retry's step-2a check finds the record.
/// Neither is overtaken: the own-key match precedes the start-record check.
async fn concurrent_duplicates_commit_once(backend: UpdateLayerBackend, layer: &'static str) {
    let fx = fixture_on(TestValkey::setup_with_pool_size(4).await, layer).await;
    let store = Arc::new(store(&fx, backend));
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let fx = Arc::new(fx);

    let mut tasks = Vec::new();
    for client_ts in [ORIGINAL_TS, RESEND_TS] {
        let (store, barrier, fx) = (store.clone(), barrier.clone(), fx.clone());
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            update(
                &store,
                &fx,
                field_delta(0x17),
                WRITER,
                client_ts,
                Some(update_op(&fx, PAYLOAD)),
            )
            .await
        }));
    }
    let mut commits = Vec::new();
    for task in tasks {
        commits.push(
            task.await
                .expect("the update task must not panic")
                .expect("both duplicates must be answered OK"),
        );
    }

    let mut outcomes: Vec<OperationOutcome> = commits.iter().map(|c| c.outcome).collect();
    outcomes.sort_by_key(|o| *o == OperationOutcome::AlreadyCommitted);
    assert_eq!(
        outcomes,
        [
            OperationOutcome::Applied,
            OperationOutcome::AlreadyCommitted
        ],
        "one duplicate commits and the other is answered from its record; got {commits:?}"
    );
    assert_eq!(
        (commits[0].seq, commits[0].server_ts),
        (commits[1].seq, commits[1].server_ts),
        "both are answered with the one commit"
    );
    assert_eq!(header_seq(&fx, fx.object_id).await, 2);
    assert_stream_commits(
        &fx,
        fx.object_id,
        &[(2, WRITER)],
        "exactly one commit entry",
    )
    .await;
    assert_eq!(operation_match_cells(&fx.metrics), [1, 0, 0, 0]);
    assert_eq!(
        overtaken_cells(&fx.metrics),
        [0, 0],
        "the loser may have started before the winner committed, so its start record no longer \
         matches — but its own key matches first, so it is answered, never overtaken"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lua_concurrent_duplicates_commit_once() {
    concurrent_duplicates_commit_once(UpdateLayerBackend::Lua, "bv6d-t7-lua").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn txn_concurrent_duplicates_commit_once() {
    concurrent_duplicates_commit_once(UpdateLayerBackend::Txn, "bv6d-t7-txn").await;
}

// ═════════════════════════════════════════════════════════════════════════════
// T24 — the start-record guard: no attempt applies after its client's next
// commit
// ═════════════════════════════════════════════════════════════════════════════

/// The record is one field per `client_id`, holding that writer's LATEST
/// commit. Without the guard, an attempt of a call that executes after the
/// same writer's next commit sees a record it does not match and applies
/// again — after the later write, reordering it.
///
/// WRITER commits operation A; a call starting now would read A's record
/// (`S0`); WRITER then commits operation B. Two calls carrying `S0` arrive
/// late, each after its own pre-flight store of hash `h`: a re-send of A,
/// and a never-committed operation C. Each is refused `UNAVAILABLE` with
/// [`OPERATION_OVERTAKEN_MESSAGE`] and applies nothing: the header and stream
/// are unchanged, `overtaken{backend}` ticks once per call, and no
/// `operation_id_match` cell moves. Each is refused on its FIRST attempt, so
/// nothing of it ran and it hands back its pre-flight increment, as an
/// answered re-send does: one `Resend(<token>)` member per call, each under
/// its own call's token, and after GC `ref_count(h)` is back where it was
/// before both pre-stores. (After a counted retry an overtaken call hands
/// back nothing — an earlier attempt may have committed — and looks up its
/// own commit's releases instead: `b_pgs_tests`.)
///
/// Fail-on-old proof: with the hand-back skipped
/// (`ReleaseRecoveryOverride::SkipAttemptOneHandBack` on the store) nothing
/// is pending and `h` keeps both increments.
async fn start_record_guard_refuses_a_call_its_client_has_moved_past(
    backend: UpdateLayerBackend,
    layer: &'static str,
) {
    let fx = fixture(layer).await;
    let store = store(&fx, backend);

    let a = update(
        &store,
        &fx,
        field_delta(0x24),
        WRITER,
        ORIGINAL_TS,
        Some(named_op(&fx, "bv6d-t24-a", PAYLOAD)),
    )
    .await
    .expect("operation A must commit");
    assert_eq!((a.seq, a.outcome), (2, OperationOutcome::Applied));
    let s0 = start_record_now(&store, &fx).await;
    assert!(
        matches!(s0, StartRecord::Record(_)),
        "precondition: the start record a call reads now is A's record; got {s0:?}"
    );
    let b = update(
        &store,
        &fx,
        field_delta(0x25),
        WRITER,
        ORIGINAL_TS,
        Some(named_op(&fx, "bv6d-t24-b", OTHER_PAYLOAD)),
    )
    .await
    .expect("operation B must commit");
    assert_eq!((b.seq, b.outcome), (3, OperationOutcome::Applied));

    let h = ext_hash(0xD4);
    let ref_count_before = ref_count(&fx.tv, h).await;
    let stream_before = stream_payloads(&fx, fx.object_id).await;
    let relay_samples_before = relay_bytes_samples(&fx.metrics);
    assert_eq!(
        (relay_samples_before.0, relay_samples_before.2),
        (2, 2),
        "fixture: A and B each sampled their commit once"
    );

    for (operation_id, payload, what) in [
        ("bv6d-t24-a", PAYLOAD, "a late re-send of A, after B"),
        (
            "bv6d-t24-c",
            b"b.v6d never committed".as_slice(),
            "a never-committed C",
        ),
    ] {
        pre_store(&fx.ext, &[h]).await;
        let late = update(
            &store,
            &fx,
            external_delta(0xD4, 1),
            WRITER,
            RESEND_TS,
            Some(named_op(&fx, operation_id, payload).with_start_record(s0.clone())),
        )
        .await;
        assert_overtaken(&late, what);
    }

    assert_eq!(
        header_seq(&fx, fx.object_id).await,
        3,
        "nothing applied: a seq of 4 is the late call applied after B"
    );
    assert_eq!(
        stream_payloads(&fx, fx.object_id).await,
        stream_before,
        "no stream entry for an overtaken call"
    );
    assert_eq!(
        overtaken_cells(&fx.metrics),
        overtaken_only_on(backend, 2),
        "each overtaken call ticks its backend's cell once"
    );
    assert_eq!(
        operation_match_cells(&fx.metrics),
        [0, 0, 0, 0],
        "an overtaken call is not an earlier-call answer or a conflict"
    );
    assert_resend_releases(
        &fx,
        h,
        fx.object_id,
        2,
        "each call overtaken on its first attempt hands back its own pre-flight increment, \
         under its own token",
    )
    .await;
    process_pending_gc(
        fx.ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(
        ref_count(&fx.tv, h).await,
        ref_count_before,
        "after GC the hash is back where it was before both calls' pre-stores"
    );
    assert_eq!(
        relay_bytes_samples(&fx.metrics),
        relay_samples_before,
        "b.yvh: an overtaken call landed no commit of its own, so it samples no relay bytes"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lua_start_record_guard_refuses_a_call_its_client_has_moved_past() {
    start_record_guard_refuses_a_call_its_client_has_moved_past(
        UpdateLayerBackend::Lua,
        "bv6d-t24-lua",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_start_record_guard_refuses_a_call_its_client_has_moved_past() {
    start_record_guard_refuses_a_call_its_client_has_moved_past(
        UpdateLayerBackend::Txn,
        "bv6d-t24-txn",
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// T28 — a wrong-typed record key: the attempt's own refusal answers
// ═════════════════════════════════════════════════════════════════════════════

/// A foreign value on the commit-token record key makes the call-start `HGET`
/// fail `WRONGTYPE`. That read is treated as giving no start record, and the
/// call's attempt then refuses the keyspace itself: exactly the
/// `WRONGTYPE_KEY` refusal a call WITHOUT an operation id gets on the same
/// keyspace, byte for byte, counted on the attempt's own backend cell.
async fn wrong_typed_record_key_is_refused_by_the_attempt_as_without_an_id(
    backend: UpdateLayerBackend,
    label: &str,
    layer: &'static str,
) {
    let fx = fixture(layer).await;
    let store = store(&fx, backend);
    let wtok_key = fx
        .tv
        .key_schema()
        .obj_write_token(&fx.object_id.to_string());
    let _: () = fx
        .tv
        .client()
        .pool
        .set(&wtok_key, "not-a-hash", None, None, false)
        .await
        .expect("planting a string on the record key must succeed");

    let mut details = Vec::new();
    for operation in [None, Some(update_op(&fx, PAYLOAD))] {
        let with_id = operation.is_some();
        match update(
            &store,
            &fx,
            field_delta(0x28),
            WRITER,
            ORIGINAL_TS,
            operation,
        )
        .await
        {
            Err(ServiceError::Internal(detail)) => {
                assert!(
                    detail.contains("WRONGTYPE_KEY")
                        && detail.contains(&wtok_key)
                        && detail.contains("(commit-token record)"),
                    "with_id={with_id}: the refusal must name the record key; got {detail:?}"
                );
                details.push(detail);
            }
            other => panic!(
                "with_id={with_id}: a wrong-typed record key must be refused by the attempt's \
                 `WRONGTYPE_KEY` check (non-retriable INTERNAL); got {other:?}"
            ),
        }
    }
    assert_eq!(
        details[1], details[0],
        "the operation-id call's refusal must be the attempt's own, identical to the id-less one"
    );
    assert_eq!(
        keyspace_type_errors(&fx.metrics, label, "update"),
        2,
        "each refusal ticks the attempt's own backend cell"
    );
    assert_eq!(header_seq(&fx, fx.object_id).await, 1, "nothing applied");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lua_wrong_typed_record_key_is_refused_by_the_attempt_as_without_an_id() {
    wrong_typed_record_key_is_refused_by_the_attempt_as_without_an_id(
        UpdateLayerBackend::Lua,
        "lua",
        "bv6d-t28-wrongtype-lua",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_wrong_typed_record_key_is_refused_by_the_attempt_as_without_an_id() {
    wrong_typed_record_key_is_refused_by_the_attempt_as_without_an_id(
        UpdateLayerBackend::Txn,
        "txn",
        "bv6d-t28-wrongtype-txn",
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// T31 — the overtaken INFO line names the call, never a record or token
// ═════════════════════════════════════════════════════════════════════════════

/// The overtaken verdict logs one INFO line naming the call — session, layer,
/// object, `client_id`, `operation_id`, backend — and nothing from the commit
/// records: a record carries a commit token and a request digest, which stay
/// out of logs above DEBUG.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn overtaken_call_logs_one_info_line_without_a_record_or_token() {
    // The whole message, matched exactly: the line is this test's deliverable,
    // and `tracing::info!` takes it as a literal, so there is no constant in
    // `storage::layer` to share.
    const OVERTAKEN_MSG: &str =
        "UpdateLayer call overtaken: its client_id's commit record on this object changed \
         while the call was in flight (another UpdateLayer from that client_id committed, or \
         a checkpoint restore); this attempt applies nothing";
    let fx = fixture("bv6d-t31-log").await;
    let store = store(&fx, UpdateLayerBackend::Lua);
    update(
        &store,
        &fx,
        field_delta(0x31),
        WRITER,
        ORIGINAL_TS,
        Some(named_op(&fx, "bv6d-t31-a", PAYLOAD)),
    )
    .await
    .expect("A must commit");
    let s0 = start_record_now(&store, &fx).await;
    update(
        &store,
        &fx,
        field_delta(0x32),
        WRITER,
        ORIGINAL_TS,
        Some(named_op(&fx, "bv6d-t31-b", OTHER_PAYLOAD)),
    )
    .await
    .expect("B must commit");
    let StartRecord::Record(record_a) = s0.clone() else {
        panic!("precondition: S0 is A's record");
    };
    let record_b = start_record_now(&store, &fx).await;
    let StartRecord::Record(record_b) = record_b else {
        panic!("precondition: the field holds B's record");
    };

    let (late, captured) = crate::grpc::log_capture::capture(update(
        &store,
        &fx,
        field_delta(0x31),
        WRITER,
        RESEND_TS,
        Some(named_op(&fx, "bv6d-t31-a", PAYLOAD).with_start_record(s0)),
    ))
    .await;
    assert_overtaken(&late, "precondition: the late re-send of A is overtaken");

    let line = captured.only(OVERTAKEN_MSG);
    assert_eq!(
        line["level"], "INFO",
        "the overtaken line logs at INFO: {line}"
    );
    let fields = &line["fields"];
    assert_eq!(fields["session_id"], fx.session_id.as_str());
    assert_eq!(fields["layer_name"], fx.layer);
    assert_eq!(fields["object_id"], fx.object_id);
    assert_eq!(fields["client_id"], WRITER);
    assert_eq!(fields["operation_id"], "bv6d-t31-a");
    assert_eq!(fields["backend"], "lua");
    for record in [&record_a, &record_b] {
        captured.assert_absent_everywhere(record);
        // `op.<id>:<seq>:<server_ts>:<token>:<digest>`: the token and the
        // digest must not leak on their own either.
        let parts: Vec<&str> = record.split(':').collect();
        assert_eq!(
            parts.len(),
            5,
            "precondition: an operation record has five parts"
        );
        captured.assert_absent_everywhere(parts[3]);
        captured.assert_absent_everywhere(parts[4]);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// CreateLayer — T13 / T15 (MGET hit): a re-send allocates nothing
// ═════════════════════════════════════════════════════════════════════════════

/// A create re-send carrying the recorded id and the same bytes is answered
/// with the original `object_id`, `ALREADY_COMMITTED`, and allocates nothing:
/// the `object_id` counter does not move, no new header appears, the mapping
/// still names the original object. The record carries the 24 h TTL.
///
/// Pre-fix the re-send allocated a fresh id and replaced the layer — the
/// object the C++ client library had just opened became terminal under it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_resend_is_answered_already_committed_and_allocates_nothing() {
    const LAYER: &str = "bv6d-t13-create";
    let fx = fixture("bv6d-t13-fixture").await;
    let store = store(&fx, UpdateLayerBackend::Lua);

    let original = create(&store, &fx, LAYER, Some(create_op(LAYER, PAYLOAD)))
        .await
        .expect("the original create must commit");
    assert_eq!(original.outcome, OperationOutcome::Applied);
    let counter_after_original = object_id_counter(&fx).await;

    let resent = create(&store, &fx, LAYER, Some(create_op(LAYER, PAYLOAD)))
        .await
        .expect("the create re-send must be answered OK");
    assert_eq!(
        resent,
        CreateCommit {
            object_id: original.object_id,
            outcome: OperationOutcome::AlreadyCommitted,
        },
        "the re-send must answer the original object_id, flagged ALREADY_COMMITTED"
    );

    assert_eq!(
        object_id_counter(&fx).await,
        counter_after_original,
        "the re-send must allocate nothing — the record is read in the MGET before allocation"
    );
    assert!(
        !header_exists(&fx, original.object_id + 1).await,
        "no object may have been materialized for the re-send"
    );
    assert_eq!(
        mapping(&fx, LAYER).await,
        Some(original.object_id),
        "the layer still maps to the original object; a different id is the layer replaced by \
         its own re-send"
    );
    assert_eq!(
        operation_match_cells(&fx.metrics),
        [0, 0, 1, 0],
        "exactly one create re-send was answered from the record"
    );

    let pttl: i64 = fx
        .tv
        .client()
        .pool
        .pttl(
            fx.tv
                .key_schema()
                .create_layer_operation(&fx.session_id, OPERATION_ID),
        )
        .await
        .expect("PTTL of the create operation record must succeed");
    assert!(
        pttl > 0 && pttl <= CREATE_LAYER_OPERATION_TTL_MS as i64,
        "the record must carry its 24 h TTL — its only removal path. PTTL={pttl} ms (-1 is a \
         record that lives forever, -2 one that was never written)"
    );
    assert_eq!(
        CREATE_LAYER_OPERATION_TTL_MS, 86_400_000,
        "the record lives 24 hours, as the proto comment promises clients"
    );
}

/// `resolve_create_layer_operation` — the handler's pre-allocation read —
/// answers nothing for an id no call has committed, and the recorded answer for
/// one a call has.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resolve_create_layer_operation_answers_a_recorded_operation_and_nothing_else() {
    const LAYER: &str = "bv6d-t13-resolve";
    let fx = fixture("bv6d-t13-resolve-fixture").await;
    let store = store(&fx, UpdateLayerBackend::Lua);

    let unknown = store
        .resolve_create_layer_operation(&fx.session_id, LAYER, &create_op(LAYER, PAYLOAD))
        .await
        .expect("the record read must succeed");
    assert_eq!(unknown, None, "no call has committed this id yet");

    let original = create(&store, &fx, LAYER, Some(create_op(LAYER, PAYLOAD)))
        .await
        .expect("the original create must commit");
    let counter = object_id_counter(&fx).await;

    let known = store
        .resolve_create_layer_operation(&fx.session_id, LAYER, &create_op(LAYER, PAYLOAD))
        .await
        .expect("the record read must succeed");
    assert_eq!(
        known,
        Some(CreateCommit {
            object_id: original.object_id,
            outcome: OperationOutcome::AlreadyCommitted,
        })
    );
    assert_eq!(
        object_id_counter(&fx).await,
        counter,
        "a read allocates nothing"
    );
}

/// A re-send after the layer was deleted is answered from the record — the
/// recorded `object_id`, `ALREADY_COMMITTED` — and does NOT re-create the
/// layer. The proto promises this ("even if the layer has since been replaced
/// or deleted"): re-creating would resurrect a layer another client deleted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_resend_after_the_layer_was_deleted_is_answered_without_recreating_it() {
    const LAYER: &str = "bv6d-create-deleted";
    let fx = fixture("bv6d-create-deleted-fixture").await;
    let store = store(&fx, UpdateLayerBackend::Lua);

    let original = create(&store, &fx, LAYER, Some(create_op(LAYER, PAYLOAD)))
        .await
        .expect("the original create must commit");
    store
        .delete_layer(&fx.session_id, LAYER, fx.ext.clone())
        .await
        .expect("delete_layer must succeed");
    assert_eq!(
        mapping(&fx, LAYER).await,
        None,
        "precondition: the layer is gone"
    );

    let resent = create(&store, &fx, LAYER, Some(create_op(LAYER, PAYLOAD)))
        .await
        .expect("the re-send must be answered");
    assert_eq!(
        (resent.object_id, resent.outcome),
        (original.object_id, OperationOutcome::AlreadyCommitted)
    );
    assert_eq!(
        mapping(&fx, LAYER).await,
        None,
        "the re-send must not re-create a deleted layer"
    );
}

/// T5 (create): the same id over a different request — different bytes on the
/// same layer, or the same bytes on a different layer (the digest covers the
/// layer name) — is refused `ALREADY_EXISTS`, allocates nothing, and maps
/// nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_resend_with_different_bytes_or_layer_is_refused_already_exists() {
    const LAYER: &str = "bv6d-t5-create";
    const OTHER_LAYER: &str = "bv6d-t5-create-other";
    let fx = fixture("bv6d-t5-create-fixture").await;
    let store = store(&fx, UpdateLayerBackend::Lua);

    let original = create(&store, &fx, LAYER, Some(create_op(LAYER, PAYLOAD)))
        .await
        .expect("the original create must commit");
    let counter = object_id_counter(&fx).await;

    for (layer, payload, what) in [
        (LAYER, OTHER_PAYLOAD, "different bytes on the same layer"),
        (OTHER_LAYER, PAYLOAD, "the same bytes on a different layer"),
    ] {
        match create(&store, &fx, layer, Some(create_op(layer, payload))).await {
            Err(ServiceError::AlreadyExists(msg)) => {
                assert_eq!(msg, OPERATION_CONFLICT_MESSAGE, "{what}")
            }
            other => panic!("{what}: must be refused ALREADY_EXISTS; got {other:?}"),
        }
    }

    assert_eq!(object_id_counter(&fx).await, counter, "nothing allocated");
    assert_eq!(mapping(&fx, LAYER).await, Some(original.object_id));
    assert_eq!(mapping(&fx, OTHER_LAYER).await, None, "nothing mapped");
    assert_eq!(operation_match_cells(&fx.metrics), [0, 0, 0, 2]);
}

/// A create answered from an earlier call's record at the `MGET` committed
/// nothing and materialized nothing, so the increments its caller pre-stored
/// are handed back (`ReleaseEvent::Resend`), against the RECORDED object.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_resend_hands_back_its_preflight_increment() {
    const LAYER: &str = "bv6d-create-ext";
    let fx = fixture("bv6d-create-ext-fixture").await;
    let store = store(&fx, UpdateLayerBackend::Lua);
    let h = ext_hash(0xC6);

    pre_store(&fx.ext, &[h]).await;
    let original = create(
        &store,
        &fx,
        LAYER,
        Some(create_op(LAYER, PAYLOAD).with_prestored_ext_hashes(vec![h])),
    )
    .await
    .expect("the original create must commit");
    assert!(
        pending_members(&fx.tv).await.is_empty(),
        "a create that commits releases nothing"
    );
    let before_resend = ref_count(&fx.tv, h).await;

    pre_store(&fx.ext, &[h]).await;
    let resent = create(
        &store,
        &fx,
        LAYER,
        Some(create_op(LAYER, PAYLOAD).with_prestored_ext_hashes(vec![h])),
    )
    .await
    .expect("the re-send must be answered");
    assert_eq!(resent.outcome, OperationOutcome::AlreadyCommitted);

    assert_single_resend_release(
        &fx,
        h,
        original.object_id,
        "the answered create re-send owes back its pre-flight increment",
    )
    .await;
    process_pending_gc(
        fx.ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(
        ref_count(&fx.tv, h).await,
        before_resend,
        "after one GC tick ref_count is back at its value before the re-send"
    );
}

/// T5 (create) with external values: a create refused `ALREADY_EXISTS` at the
/// `MGET` committed nothing and materialized nothing either, so the increments
/// its caller pre-stored are owed back exactly as an answered re-send's are —
/// one `surplus_acquire` release (`ReleaseEvent::Resend`), against the
/// RECORDED object. Without it the refused call's `+1` would stay stranded
/// until the reference-count reconciler noticed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_conflict_hands_back_its_preflight_increment() {
    const LAYER: &str = "bv6d-create-conflict-ext";
    let fx = fixture("bv6d-create-conflict-ext-fixture").await;
    let store = store(&fx, UpdateLayerBackend::Lua);
    let h = ext_hash(0xC8);

    let original = create(&store, &fx, LAYER, Some(create_op(LAYER, PAYLOAD)))
        .await
        .expect("the original create must commit");

    pre_store(&fx.ext, &[h]).await;
    let released_before = ext_ref_released(&fx.metrics, REASON_SURPLUS_ACQUIRE);
    let refused = create(
        &store,
        &fx,
        LAYER,
        Some(create_op(LAYER, OTHER_PAYLOAD).with_prestored_ext_hashes(vec![h])),
    )
    .await;
    match refused {
        Err(ServiceError::AlreadyExists(msg)) => assert_eq!(msg, OPERATION_CONFLICT_MESSAGE),
        other => panic!(
            "precondition: the same id over a different request must be refused ALREADY_EXISTS; \
             got {other:?}"
        ),
    }

    assert_single_resend_release(
        &fx,
        h,
        original.object_id,
        "the refused create committed nothing, so its pre-flight increment is owed back",
    )
    .await;
    assert_eq!(
        ext_ref_released(&fx.metrics, REASON_SURPLUS_ACQUIRE),
        released_before + 1,
        "the hand-back is counted under reason=\"surplus_acquire\""
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The re-drive of a recorded overwrite's post-commit tail
// ═════════════════════════════════════════════════════════════════════════════

/// Stage a recorded overwrite of the fixture's layer: `writer` updates the
/// object about to be replaced (`fx.object_id`), which leaves that object's
/// admission-semaphore entry in `writer`'s map; then a SECOND store overwrites
/// the layer carrying [`OPERATION_ID`] and its own cleanup task settles. Each
/// store has its own map, so the overwrite's retirement of the replaced
/// object's entry never reaches `writer` — `writer` stands for the replica a
/// client's re-send lands on. Returns the overwrite's new object.
async fn recorded_overwrite(fx: &Fixture, writer: &LayerStore) -> u64 {
    update(writer, fx, field_delta(0x41), WRITER, ORIGINAL_TS, None)
        .await
        .expect("the update of the object about to be replaced must commit");
    let overwriter = store(fx, UpdateLayerBackend::Lua);
    let overwrite = create(
        &overwriter,
        fx,
        fx.layer,
        Some(create_op(fx.layer, PAYLOAD)),
    )
    .await
    .expect("the recorded overwrite must commit");
    assert_eq!(
        overwrite.outcome,
        OperationOutcome::Applied,
        "precondition: the overwrite is the operation's first commit"
    );
    overwriter.wait_for_cleanup(CALL_BUDGET).await;
    assert_eq!(
        writer.semaphore_count(),
        1,
        "precondition: `writer` still holds the replaced object's admission entry"
    );
    overwrite.object_id
}

/// A re-send answered from a recorded overwrite re-appends `deleted` to the
/// replaced object's stream with `NOMKSTREAM`. Once that stream is gone — its
/// grace TTL ran out, or a sweep reaped it — the re-drive must not create it
/// again: a stream the re-drive created would carry no TTL and belong to no
/// live object, so nothing would ever remove it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_resend_redrive_does_not_recreate_a_reaped_replaced_stream() {
    let fx = fixture("bv6d-redrive-nomkstream").await;
    let writer = store(&fx, UpdateLayerBackend::Lua);
    let committed = recorded_overwrite(&fx, &writer).await;
    let old_stream = fx.tv.key_schema().layer_stream(fx.object_id);
    let _: i64 = fx
        .tv
        .client()
        .pool
        .del(&old_stream)
        .await
        .expect("DEL of the replaced object's stream must succeed");

    let resent = create(&writer, &fx, fx.layer, Some(create_op(fx.layer, PAYLOAD)))
        .await
        .expect("the re-send must be answered");
    assert_eq!(
        resent,
        CreateCommit {
            object_id: committed,
            outcome: OperationOutcome::AlreadyCommitted,
        },
        "precondition: the re-send was answered from the record, so its re-drive ran"
    );

    let exists: u64 = fx
        .tv
        .client()
        .pool
        .exists(&old_stream)
        .await
        .expect("EXISTS must succeed");
    assert_eq!(
        exists, 0,
        "the re-drive's XADD must not create the replaced object's stream once it is gone \
         (NOMKSTREAM)"
    );
}

/// A re-send answered from a recorded overwrite retires the replaced object's
/// admission entry and its structural-cache entry on the store that answers
/// it. No writer can resolve the layer to the replaced object any more, so an
/// admission entry left behind would sit in the map — and on the
/// admission-entries gauge — for the life of the process, and a cache entry
/// would hold the replaced object's state against the cache budget.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_resend_redrive_retires_the_replaced_objects_admission_and_cache_entries() {
    let fx = fixture("bv6d-redrive-semaphore").await;
    let writer = store(&fx, UpdateLayerBackend::Lua);
    let committed = recorded_overwrite(&fx, &writer).await;
    // `writer`'s cache is its own, as its admission map is, so the
    // overwriter's retirement never reached it.
    cache_entry_for(&writer, fx.object_id);

    let resent = create(&writer, &fx, fx.layer, Some(create_op(fx.layer, PAYLOAD)))
        .await
        .expect("the re-send must be answered");
    assert_eq!(
        resent,
        CreateCommit {
            object_id: committed,
            outcome: OperationOutcome::AlreadyCommitted,
        },
        "precondition: the re-send was answered from the record, so its re-drive ran"
    );
    assert_eq!(
        writer.semaphore_count(),
        0,
        "the re-drive must retire the replaced object's admission entry"
    );
    assert_retired(
        &writer,
        fx.object_id,
        "the replaced object's structural-cache entry, after the re-drive",
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T12 — the retract script's compare half
// ═════════════════════════════════════════════════════════════════════════════

/// `commit_record_retract.lua` is a compare-and-delete. It removes the writer's
/// field only while the field holds exactly the torn call's record. When a
/// later commit by the same `client_id` has overwritten the field, the script
/// returns `0` and leaves the later record in place; deleting it would let
/// that later operation's re-send apply a second time. A matching record is
/// then removed (`1`), so the `0` above is the compare refusing, not a script
/// that never deletes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_record_retract_leaves_a_field_holding_a_different_record() {
    /// Any object id: the test touches only this object's record key.
    const OBJECT_ID: &str = "990001";
    let tv = TestValkey::setup().await;
    let wtok_key = tv.key_schema().obj_write_token(OBJECT_ID);
    let digest = operation_request_digest("bv6d-retract-compare", PAYLOAD);
    let torn = format!("op.{OPERATION_ID}:2:1000:{}:{digest}", "a".repeat(32));
    let later = format!("op.{OPERATION_ID}:3:2000:{}:{digest}", "b".repeat(32));
    let _: () = tv
        .client()
        .pool
        .hset(&wtok_key, [(WRITER, later.as_str())])
        .await
        .expect("HSET of the later record must succeed");

    let mismatched = retract_record(&tv, &wtok_key, torn).await;
    assert_eq!(
        (mismatched, writer_field(&tv, &wtok_key).await),
        (0, Some(later.clone())),
        "a field holding a different record must be left intact and answer 0"
    );

    let matched = retract_record(&tv, &wtok_key, later).await;
    assert_eq!(
        (matched, writer_field(&tv, &wtok_key).await),
        (1, None),
        "control: the field holding exactly the given record is removed and answers 1"
    );
}

/// One run of `commit_record_retract.lua` for [`WRITER`]'s field.
async fn retract_record(tv: &TestValkey, wtok_key: &str, record: String) -> i64 {
    COMMIT_RECORD_RETRACT_SCRIPT
        .evalsha_with_reload(
            tv.client().pool.next(),
            vec![wtok_key.to_owned()],
            vec![WRITER.to_owned(), record],
        )
        .await
        .expect("the retract script must run")
}

/// `HGET <wtok_key> <WRITER>`.
async fn writer_field(tv: &TestValkey, wtok_key: &str) -> Option<String> {
    tv.client()
        .pool
        .hget(wtok_key, WRITER)
        .await
        .expect("HGET of the record must succeed")
}

// ═════════════════════════════════════════════════════════════════════════════
// The request digest's format
// ═════════════════════════════════════════════════════════════════════════════

/// The request digest is persisted in both operation records for 24 h and
/// compared by whichever replica answers a re-send, so its format is a
/// contract between replicas: a change during a rolling deploy would answer
/// every in-flight re-send `ALREADY_EXISTS`. Pinned against a vector computed
/// outside this crate — Python's
/// `hashlib.sha256(b"bv6d-digest-layer" + b"\x00" + OMPF).hexdigest()` — so the
/// format is lowercase-hex SHA-256 over the layer name, one `0x00` byte, then
/// the OMPF bytes. (Without the separator the digest would be
/// `a36b13aa…40313`.)
#[test]
fn operation_request_digest_matches_a_known_vector() {
    const LAYER: &str = "bv6d-digest-layer";
    /// Arbitrary bytes, including a NUL and high bytes: the digest hashes the
    /// payload as streamed and never parses it.
    const OMPF: &[u8] = b"OMPF\x00\x01\xfe\xff";
    const EXPECTED: &str = "1274fb5547630361c61f7f038d75dac2db106ee31c1fe452b828f7312c1831d5";

    assert_eq!(
        operation_request_digest(LAYER, OMPF),
        EXPECTED,
        "the recorded digest format changed; records written by other replicas would no longer \
         match a re-send"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T16 — creates without an id still map and index the layer
// ═════════════════════════════════════════════════════════════════════════════

/// The commit script replaced the new-layer `MULTI` (`SET` mapping + `SADD`
/// layers index) AND the overwrite's bare mapping `SET`. For an id-less call it
/// must still write both on both paths, answer `UNSPECIFIED`, and write no
/// operation record.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_without_an_operation_id_maps_and_indexes_the_layer_on_both_paths() {
    const LAYER: &str = "bv6d-t16-create";
    let fx = fixture("bv6d-t16-fixture").await;
    let store = store(&fx, UpdateLayerBackend::Lua);
    let index_key = fx.tv.key_schema().layers_index(&fx.session_id);
    let member = live_edit::storage::keys::hex_encode(LAYER);

    let fresh = create(&store, &fx, LAYER, None)
        .await
        .expect("an id-less fresh create must commit");
    assert_eq!(fresh.outcome, OperationOutcome::Unspecified);
    assert_eq!(
        mapping(&fx, LAYER).await,
        Some(fresh.object_id),
        "fresh: mapped"
    );
    let indexed: bool = fx
        .tv
        .client()
        .pool
        .sismember(&index_key, &member)
        .await
        .expect("SISMEMBER must succeed");
    assert!(
        indexed,
        "fresh: the layer must be listed in the layers index"
    );

    let overwrite = create(&store, &fx, LAYER, None)
        .await
        .expect("an id-less overwrite must commit");
    assert_eq!(overwrite.outcome, OperationOutcome::Unspecified);
    assert_ne!(
        overwrite.object_id, fresh.object_id,
        "an overwrite allocates"
    );
    assert_eq!(
        mapping(&fx, LAYER).await,
        Some(overwrite.object_id),
        "overwrite: the mapping flips to the new object"
    );
    let indexed: bool = fx
        .tv
        .client()
        .pool
        .sismember(&index_key, &member)
        .await
        .expect("SISMEMBER must succeed");
    assert!(indexed, "overwrite: the layer stays listed");

    let record_keys: u64 = fx
        .tv
        .client()
        .pool
        .exists(
            fx.tv
                .key_schema()
                .create_layer_operation(&fx.session_id, OPERATION_ID),
        )
        .await
        .expect("EXISTS must succeed");
    assert_eq!(
        record_keys, 0,
        "an id-less create writes no operation record"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Wrong-typed session keys on the create path
// ═════════════════════════════════════════════════════════════════════════════

/// A foreign-typed value on the operation-record key is refused by the
/// handler's pre-allocation read as a non-retriable keyspace refusal: it ticks
/// `keyspace_type_error_total{txn,create}`, allocates no `object_id` and writes
/// nothing — so, unlike the commit script's backstop below, it orphans nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_with_a_wrong_typed_operation_record_is_refused_before_allocation() {
    const LAYER: &str = "bv6d-wrongtype-read";
    let fx = fixture("bv6d-wrongtype-read-fixture").await;
    let store = store(&fx, UpdateLayerBackend::Lua);
    let record_key = fx
        .tv
        .key_schema()
        .create_layer_operation(&fx.session_id, OPERATION_ID);
    let _: () = fx
        .tv
        .client()
        .pool
        .hset(&record_key, [("planted", "hash")])
        .await
        .expect("planting a hash on the record key must succeed");
    let counter = object_id_counter(&fx).await;

    let refused = store
        .resolve_create_layer_operation(&fx.session_id, LAYER, &create_op(LAYER, PAYLOAD))
        .await;
    match refused {
        Err(ServiceError::PermanentKeyspaceRefusal { detail, .. }) => {
            for fragment in [
                "WRONGTYPE_KEY",
                record_key.as_str(),
                "create-layer operation record",
            ] {
                assert!(
                    detail.contains(fragment),
                    "the refusal must name {fragment:?} so an operator can find the key; got \
                     {detail:?}"
                );
            }
        }
        other => panic!(
            "a foreign-typed record is keyspace corruption: no retry can GET it, so it must be a \
             PermanentKeyspaceRefusal; got {other:?}"
        ),
    }
    assert_eq!(keyspace_type_errors(&fx.metrics, "txn", "create"), 1);
    assert_eq!(object_id_counter(&fx).await, counter, "nothing allocated");
    assert_eq!(mapping(&fx, LAYER).await, None, "nothing mapped");
}

/// The commit script type-checks the session keys it writes before its first
/// write. A foreign-typed operation record (retyped after the read above) or
/// layers index is refused `WRONGTYPE_KEY` as a keyspace refusal, ticks
/// `{txn,create}`, and leaves the mapping where it was.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_commit_script_refuses_a_wrong_typed_session_key() {
    let fx = fixture("bv6d-wrongtype-script-fixture").await;
    let store = store(&fx, UpdateLayerBackend::Lua);

    // The record key: the `MGET` reads a non-string as nil, so only the
    // script's own check can see it.
    const RECORD_LAYER: &str = "bv6d-wrongtype-record";
    let record_key = fx
        .tv
        .key_schema()
        .create_layer_operation(&fx.session_id, OPERATION_ID);
    let _: () = fx
        .tv
        .client()
        .pool
        .hset(&record_key, [("planted", "hash")])
        .await
        .expect("planting a hash on the record key must succeed");
    match create(
        &store,
        &fx,
        RECORD_LAYER,
        Some(create_op(RECORD_LAYER, PAYLOAD)),
    )
    .await
    {
        Err(ServiceError::PermanentKeyspaceRefusal { detail, .. }) => assert!(
            detail.contains("WRONGTYPE_KEY") && detail.contains(&record_key),
            "the script refusal must name the record key; got {detail:?}"
        ),
        other => panic!("a foreign-typed record must refuse the commit; got {other:?}"),
    }
    assert_eq!(
        mapping(&fx, RECORD_LAYER).await,
        None,
        "the mapping did not move"
    );

    // The layers index, for an id-less create.
    const INDEX_LAYER: &str = "bv6d-wrongtype-index";
    let index_key = fx.tv.key_schema().layers_index(&fx.session_id);
    let _: i64 = fx
        .tv
        .client()
        .pool
        .del(&index_key)
        .await
        .expect("DEL layers index must succeed");
    let _: () = fx
        .tv
        .client()
        .pool
        .set(&index_key, "not-a-set", None, None, false)
        .await
        .expect("planting a string on the layers index must succeed");
    match create(&store, &fx, INDEX_LAYER, None).await {
        Err(ServiceError::PermanentKeyspaceRefusal { detail, .. }) => assert!(
            detail.contains("WRONGTYPE_KEY") && detail.contains(&index_key),
            "the script refusal must name the layers index; got {detail:?}"
        ),
        other => panic!("a foreign-typed layers index must refuse the commit; got {other:?}"),
    }
    assert_eq!(
        mapping(&fx, INDEX_LAYER).await,
        None,
        "the mapping did not move"
    );

    assert_eq!(
        keyspace_type_errors(&fx.metrics, "txn", "create"),
        2,
        "each script refusal ticks the create cell once"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The lost-reply, torn-commit and race scenarios (test-fault-injection seams)
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod lost_reply {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use fred::clients::Client;
    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use futures::FutureExt;

    use live_edit::storage::test_support::ConnStall;
    use live_edit::storage::watched_txn::{
        WatchWindowHook, WatchedExecOverride, WatchedExecOverrideQueue,
    };

    use super::*;

    /// Bound on each wait in the dropped-call tests: for the retract task to
    /// park at its gate and for the record to go once the gate is released
    /// (T12); for a commit to reach, and queue behind, a stalled connection,
    /// and for the pool to serve again after the kill (T25, T32).
    const RETRACT_WAIT: Duration = Duration::from_secs(10);

    type FredQueue = Arc<Mutex<VecDeque<FredError>>>;

    /// `n` fred `Timeout`s — "the request went out and no reply came back".
    fn timeouts(n: usize) -> FredQueue {
        Arc::new(Mutex::new(
            (0..n)
                .map(|_| FredError::new(FredErrorKind::Timeout, "bv6d: reply lost"))
                .collect(),
        ))
    }

    fn overrides(items: &[WatchedExecOverride]) -> WatchedExecOverrideQueue {
        Arc::new(Mutex::new(items.iter().copied().collect()))
    }

    /// Assert a seam queue drained, so assertions that describe an
    /// undisturbed commit are not satisfied by a seam that never fired.
    fn assert_drained<T: std::fmt::Debug>(queue: &Mutex<VecDeque<T>>, seam: &str) {
        let left = queue.lock().unwrap();
        assert!(
            left.is_empty(),
            "every queued {seam} entry must have been consumed; {left:?} left"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // T1 / T2 with the original call's reply lost
    // ─────────────────────────────────────────────────────────────────────────

    /// The case operation ids exist for. A single-attempt store commits A and
    /// loses its reply, so A answers the retriable `Aborted` (gRPC
    /// `UNAVAILABLE`) for a commit that LANDED. The client re-sends with the
    /// same id; B is answered from A's record, with the sequence and server
    /// timestamp A's commit actually published.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lua_resend_after_a_lost_reply_is_answered_already_committed() {
        let fx = fixture("bv6d-t1-lost-lua").await;
        let lost = timeouts(1);
        let losing = store(&fx, UpdateLayerBackend::Lua)
            .with_update_layer_retry_max_attempts(1)
            .with_lua_post_eval_fred_error_injector_for_test(lost.clone());

        let original = update(
            &losing,
            &fx,
            field_delta(0x21),
            WRITER,
            ORIGINAL_TS,
            Some(update_op(&fx, PAYLOAD)),
        )
        .await;
        assert_drained(&lost, "post-EVALSHA lost reply");
        assert!(
            matches!(original, Err(ServiceError::Aborted(_))),
            "precondition: with its only reply lost the original call exhausts its retries and \
             answers the retriable Aborted; got {original:?}"
        );
        let payloads =
            assert_stream_commits(&fx, fx.object_id, &[(2, WRITER)], "the original landed").await;

        let resent = update(
            &store(&fx, UpdateLayerBackend::Lua),
            &fx,
            field_delta(0x21),
            WRITER,
            RESEND_TS,
            Some(update_op(&fx, PAYLOAD)),
        )
        .await
        .expect("the re-send of a landed operation must be answered OK");
        assert_eq!(
            resent,
            UpdateCommit {
                seq: 2,
                server_ts: payload_server_ts(&payloads[0]),
                outcome: OperationOutcome::AlreadyCommitted,
            },
            "the re-send must report the commit the lost reply described"
        );
        assert_stream_commits(&fx, fx.object_id, &[(2, WRITER)], "still one commit").await;
        assert_eq!(header_seq(&fx, fx.object_id).await, 2);
    }

    /// The slow-path twin, with a `CreateSection`: the `EXEC` runs and commits,
    /// its reply is lost (`ReplaceOutcomeAfterExec`), and the reconciling
    /// `XRANGE` fails too, so the single attempt answers `Aborted`. The re-send
    /// is answered `ALREADY_COMMITTED` — pre-fix it re-applied the
    /// `CreateSection` and was refused `DuplicateSection`.
    ///
    /// The original call runs on `lua-watch-structural`, whose structural
    /// deltas take the WATCH path that the in-doubt `EXEC` override and the
    /// reconcile belong to; the re-send runs on the default `lua` backend's
    /// structural route.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn slow_path_resend_after_a_lost_exec_reply_is_answered_already_committed() {
        let fx = fixture("bv6d-t2-lost-txn").await;
        let exec = overrides(&[WatchedExecOverride::ReplaceOutcomeAfterExec]);
        let reconcile = timeouts(1);
        let losing = store(&fx, UpdateLayerBackend::LuaWatchStructural)
            .with_update_layer_retry_max_attempts(1)
            .with_watched_exec_override_for_test(exec.clone())
            .with_reconcile_xrange_fred_error_injector_for_test(reconcile.clone());

        let original = update(
            &losing,
            &fx,
            section_delta(),
            WRITER,
            ORIGINAL_TS,
            Some(update_op(&fx, PAYLOAD)),
        )
        .await;
        assert_drained(&exec, "watched-EXEC override");
        assert_drained(&reconcile, "reconcile XRANGE");
        assert!(
            matches!(original, Err(ServiceError::Aborted(_))),
            "precondition: the in-doubt EXEC with a failed reconcile answers Aborted; got \
             {original:?}"
        );
        let payloads =
            assert_stream_commits(&fx, fx.object_id, &[(2, WRITER)], "the original landed").await;

        let resent = update(
            &store(&fx, UpdateLayerBackend::Lua),
            &fx,
            section_delta(),
            WRITER,
            RESEND_TS,
            Some(update_op(&fx, PAYLOAD)),
        )
        .await
        .expect(
            "the re-send must be answered OK. INVALID_ARGUMENT here is the re-send re-applying the \
             CreateSection the original landed",
        );
        assert_eq!(
            resent,
            UpdateCommit {
                seq: 2,
                server_ts: payload_server_ts(&payloads[0]),
                outcome: OperationOutcome::AlreadyCommitted,
            }
        );
        assert_stream_commits(&fx, fx.object_id, &[(2, WRITER)], "still one commit").await;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // T3 — an own-call match with an operation id
    // ─────────────────────────────────────────────────────────────────────────

    /// Within ONE call carrying an id, an attempt whose reply was lost is
    /// retried, and the retry finds a record under `op.<id>` carrying THIS
    /// call's token. That is the b.75i case, not a re-send: the call answers
    /// `APPLIED` (it did commit), ticks `commit_already_landed_total` on its
    /// backend, does not tick `operation_id_match_total`, and releases nothing
    /// — the attempt that landed consumed the pre-flight increment.
    async fn own_call_match_with_an_operation_id_answers_applied_and_releases_nothing(
        fx: Fixture,
        disturbed: LayerStore,
        backend_label: &str,
    ) {
        let h = ext_hash(0xC3);
        pre_store(&fx.ext, &[h]).await;

        let commit = update(
            &disturbed,
            &fx,
            external_delta(0xC3, 0),
            WRITER,
            ORIGINAL_TS,
            Some(update_op(&fx, PAYLOAD)),
        )
        .await
        .expect("the commit landed, so the call must succeed");

        assert_eq!(
            (commit.seq, commit.outcome),
            (2, OperationOutcome::Applied),
            "an own-call match is this call's commit: APPLIED, not ALREADY_COMMITTED"
        );
        assert_eq!(
            already_landed(&fx.metrics, backend_label),
            1,
            "the in-call match is counted where b.75i counts it"
        );
        assert_eq!(
            operation_match_cells(&fx.metrics),
            [0, 0, 0, 0],
            "an own-call match is not a re-send"
        );
        assert!(
            pending_members(&fx.tv).await.is_empty(),
            "an own-call match must release NOTHING: the attempt that landed consumed the \
             pre-flight increment, and releasing it would leave the object's reference uncounted"
        );
        assert_eq!(ref_count(&fx.tv, h).await, Some(1));
        assert_stream_commits(&fx, fx.object_id, &[(2, WRITER)], "one commit").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lua_own_call_match_with_an_operation_id_answers_applied_and_releases_nothing() {
        let fx = fixture("bv6d-t3-lua").await;
        let lost = timeouts(1);
        let disturbed = store(&fx, UpdateLayerBackend::Lua)
            .with_lua_post_eval_fred_error_injector_for_test(lost.clone());
        own_call_match_with_an_operation_id_answers_applied_and_releases_nothing(
            fx, disturbed, "lua",
        )
        .await;
        assert_drained(&lost, "post-EVALSHA lost reply");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn txn_own_call_match_with_an_operation_id_answers_applied_and_releases_nothing() {
        let fx = fixture("bv6d-t3-txn").await;
        let exec = overrides(&[WatchedExecOverride::ReplaceOutcomeAfterExec]);
        let reconcile = timeouts(1);
        let disturbed = store(&fx, UpdateLayerBackend::Txn)
            .with_watched_exec_override_for_test(exec.clone())
            .with_reconcile_xrange_fred_error_injector_for_test(reconcile.clone());
        own_call_match_with_an_operation_id_answers_applied_and_releases_nothing(
            fx, disturbed, "txn",
        )
        .await;
        assert_drained(&exec, "watched-EXEC override");
        assert_drained(&reconcile, "reconcile XRANGE");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // T12 — a torn slow-path commit with an id retracts its record
    // ─────────────────────────────────────────────────────────────────────────

    /// A pre-dispatch hook that, on its FIRST invocation only, retypes the
    /// field key the commit is about to `HSET` — after the pre-`MULTI` type
    /// probe passed — so the `EXEC` runs every frame but that one and comes back
    /// with an error frame: a torn commit.
    fn tearing_hook(fx: &Fixture) -> (WatchWindowHook, Arc<AtomicUsize>) {
        let field_key = fx.tv.key_schema().obj_field(
            &fx.object_id.to_string(),
            &SECTION_ID.to_string(),
            "color",
        );
        let writer = fx.tv.client().maintenance_pool.next().clone();
        let invocations = Arc::new(AtomicUsize::new(0));
        let counter = invocations.clone();
        let hook: WatchWindowHook = Arc::new(move |_client: &Client| {
            let first = counter.fetch_add(1, Ordering::SeqCst) == 0;
            let (field_key, writer) = (field_key.clone(), writer.clone());
            async move {
                if !first {
                    return;
                }
                let _: () = writer
                    .set(&field_key, "retyped-after-the-probe", None, None, false)
                    .await
                    .expect("retyping the field key must succeed");
            }
            .boxed()
        });
        (hook, invocations)
    }

    /// `HGET le:{oid}:obj:wtok <WRITER>`, on the maintenance pool.
    async fn writer_record(fx: &Fixture) -> Option<String> {
        fx.tv
            .client()
            .maintenance_pool
            .next()
            .hget(
                fx.tv
                    .key_schema()
                    .obj_write_token(&fx.object_id.to_string()),
                WRITER,
            )
            .await
            .expect("HGET commit-token record must succeed")
    }

    /// Undo the tear's retype so the next commit can write the field again.
    async fn restore_field_key(fx: &Fixture) {
        let _: i64 = fx
            .tv
            .client()
            .pool
            .del(fx.tv.key_schema().obj_field(
                &fx.object_id.to_string(),
                &SECTION_ID.to_string(),
                "color",
            ))
            .await
            .expect("DEL of the retyped field key must succeed");
    }

    /// The torn `MULTI` writes the call's `op.<id>` record with its other
    /// frames. Left in place, the client's re-send (`INTERNAL` is retried)
    /// would read it as an earlier call's landed commit and answer
    /// `ALREADY_COMMITTED` for an edit that did not fully land. The slow path
    /// therefore retracts it before answering; the re-send then finds no
    /// record and applies.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn torn_commit_with_an_operation_id_retracts_its_record_so_a_resend_applies() {
        let fx = fixture("bv6d-t12-torn").await;
        let (hook, invocations) = tearing_hook(&fx);
        let tearing = store(&fx, UpdateLayerBackend::Txn).with_pre_dispatch_hook_for_test(hook);

        let torn = update(
            &tearing,
            &fx,
            field_delta(0x31),
            WRITER,
            ORIGINAL_TS,
            Some(update_op(&fx, PAYLOAD)),
        )
        .await;
        assert_eq!(
            invocations.load(Ordering::SeqCst),
            1,
            "precondition: one attempt, which the hook tore"
        );
        assert!(
            matches!(torn, Err(ServiceError::Internal(ref msg)) if msg.contains("EXEC")),
            "precondition: a torn EXEC answers the non-retriable Internal; got {torn:?}"
        );
        assert_eq!(
            header_seq(&fx, fx.object_id).await,
            2,
            "precondition: the tear's surviving frames landed — the header was bumped, which is \
             exactly what makes the record they wrote alongside it dangerous"
        );

        assert_eq!(
            writer_record(&fx).await,
            None,
            "no record carrying the torn call may survive the call: its field is retracted"
        );

        restore_field_key(&fx).await;
        let resent = update(
            &store(&fx, UpdateLayerBackend::Txn),
            &fx,
            field_delta(0x31),
            WRITER,
            RESEND_TS,
            Some(update_op(&fx, PAYLOAD)),
        )
        .await
        .expect("the re-send must commit");
        assert_eq!(
            (resent.seq, resent.outcome),
            (3, OperationOutcome::Applied),
            "the re-send must APPLY. ALREADY_COMMITTED at seq 2 is the torn commit reported as a \
             success"
        );
        assert_eq!(operation_match_cells(&fx.metrics), [0, 0, 0, 0]);
    }

    /// The retract is a task on the store's cleanup tracker, so dropping the
    /// call's future — a deadline drop while it awaits the retract — does not
    /// cancel it.
    ///
    /// The drop has to land BEFORE the retract puts its `EVALSHA` on the wire.
    /// A dropped fred future does not take back a command already written, so
    /// a drop after that point would see the record vanish whether or not the
    /// retract outlived the call. The store's retract hook holds the task at
    /// exactly that point: inside the tracked task, before the `EVALSHA`. The
    /// test waits for the task to park there, checks the torn record is still
    /// present, drops the call, and only then opens the gate. The record's
    /// later disappearance is the retract running after its caller was gone.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn torn_commit_record_retract_survives_a_dropped_call() {
        let fx = Arc::new(fixture("bv6d-t12-dropped").await);
        let (tear, _) = tearing_hook(&fx);
        let parked = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        // The hook's client is ignored: it is not necessarily the connection
        // the retract's `EVALSHA` goes out on.
        let gate: WatchWindowHook = {
            let (parked, release) = (parked.clone(), release.clone());
            Arc::new(move |_client: &Client| {
                let (parked, release) = (parked.clone(), release.clone());
                async move {
                    parked.notify_one();
                    release.notified().await;
                }
                .boxed()
            })
        };
        let tearing = Arc::new(
            store(&fx, UpdateLayerBackend::Txn)
                .with_pre_dispatch_hook_for_test(tear)
                .with_torn_commit_retract_hook_for_test(gate),
        );

        let call = {
            let (tearing, fx) = (tearing.clone(), fx.clone());
            tokio::spawn(async move {
                update(
                    &tearing,
                    &fx,
                    field_delta(0x32),
                    WRITER,
                    ORIGINAL_TS,
                    Some(update_op(&fx, PAYLOAD)),
                )
                .await
            })
        };

        tokio::time::timeout(RETRACT_WAIT, parked.notified())
            .await
            .expect("the retract never reached its gate: the torn arm did not start it");
        assert!(
            writer_record(&fx).await.is_some(),
            "precondition: while the retract is parked before its EVALSHA, the torn record is \
             still present"
        );

        call.abort();
        assert!(
            call.await.unwrap_err().is_cancelled(),
            "precondition: the call's future was dropped while it awaited the retract"
        );
        release.notify_one();

        let deadline = tokio::time::Instant::now() + RETRACT_WAIT;
        while writer_record(&fx).await.is_some() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the torn record must be retracted after the call was dropped: the retract task \
                 has to outlive the handler future"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // T27 — a live call's late attempt is overtaken (the retry loop, (i))
    // ─────────────────────────────────────────────────────────────────────────

    /// The first way an attempt can run after its client's next commit: the
    /// call's own retry loop. X's attempt 1 commits and loses its reply
    /// (`disturbed`'s seam), so X retries — and is parked between attempts.
    /// Meanwhile the client, having seen no reply, re-sends X (X' is answered
    /// `ALREADY_COMMITTED`) and moves on to Y, which commits. X's attempt 2 then
    /// runs: its own key no longer matches (the field holds Y's record) and its
    /// start record no longer matches either, so it is overtaken —
    /// `UNAVAILABLE`, nothing applied. Without the guard it applied X a second
    /// time, after Y.
    async fn late_attempt_of_a_live_call_is_overtaken(
        fx: Fixture,
        disturbed: LayerStore,
        backend: UpdateLayerBackend,
    ) {
        let parked = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let hook: live_edit::storage::layer::BetweenAttemptsHook = {
            let (parked, release) = (Arc::clone(&parked), Arc::clone(&release));
            Arc::new(move |_attempt: u32| {
                let (parked, release) = (Arc::clone(&parked), Arc::clone(&release));
                async move {
                    parked.notify_one();
                    release.notified().await;
                }
                .boxed()
            })
        };
        let fx = Arc::new(fx);
        let caller = Arc::new(disturbed.with_between_attempts_hook_for_test(hook));
        let x = {
            let (caller, fx) = (Arc::clone(&caller), Arc::clone(&fx));
            tokio::spawn(async move {
                update(
                    &caller,
                    &fx,
                    field_delta(0x27),
                    WRITER,
                    ORIGINAL_TS,
                    Some(named_op(&fx, "bv6d-t27-x", PAYLOAD)),
                )
                .await
            })
        };
        tokio::time::timeout(CALL_BUDGET, parked.notified())
            .await
            .expect("X never reached its second attempt: attempt 1 did not fail transiently");
        assert_stream_commits(&fx, fx.object_id, &[(2, WRITER)], "precondition: X landed").await;

        let plain = store(&fx, backend);
        let resent = update(
            &plain,
            &fx,
            field_delta(0x27),
            WRITER,
            RESEND_TS,
            Some(named_op(&fx, "bv6d-t27-x", PAYLOAD)),
        )
        .await
        .expect("the client's re-send of X must be answered");
        assert_eq!(
            (resent.seq, resent.outcome),
            (2, OperationOutcome::AlreadyCommitted)
        );
        let y = update(
            &plain,
            &fx,
            field_delta(0x28),
            WRITER,
            RESEND_TS,
            Some(named_op(&fx, "bv6d-t27-y", OTHER_PAYLOAD)),
        )
        .await
        .expect("the client's next write Y must commit");
        assert_eq!(y.seq, 3);

        release.notify_one();
        let late = x.await.expect("the X task must not panic");
        assert_overtaken(&late, "X's attempt after Y");
        assert_stream_commits(
            &fx,
            fx.object_id,
            &[(2, WRITER), (3, WRITER)],
            "X and Y only — a 4-0 is X applied again after Y",
        )
        .await;
        assert_eq!(header_seq(&fx, fx.object_id).await, 3);
        assert_eq!(overtaken_cells(&fx.metrics), overtaken_only_on(backend, 1));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lua_late_attempt_of_a_live_call_is_overtaken() {
        let fx = fixture("bv6d-t27-lua").await;
        let lost = timeouts(1);
        let disturbed = store(&fx, UpdateLayerBackend::Lua)
            .with_lua_post_eval_fred_error_injector_for_test(lost.clone());
        late_attempt_of_a_live_call_is_overtaken(fx, disturbed, UpdateLayerBackend::Lua).await;
        assert_drained(&lost, "post-EVALSHA lost reply");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn txn_late_attempt_of_a_live_call_is_overtaken() {
        let fx = fixture("bv6d-t27-txn").await;
        let exec = overrides(&[WatchedExecOverride::ReplaceOutcomeAfterExec]);
        let reconcile = timeouts(1);
        let disturbed = store(&fx, UpdateLayerBackend::Txn)
            .with_watched_exec_override_for_test(exec.clone())
            .with_reconcile_xrange_fred_error_injector_for_test(reconcile.clone());
        late_attempt_of_a_live_call_is_overtaken(fx, disturbed, UpdateLayerBackend::Txn).await;
        assert_drained(&exec, "watched-EXEC override");
        assert_drained(&reconcile, "reconcile XRANGE");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // T28 — a failed start-record read ends the call before any attempt
    // ─────────────────────────────────────────────────────────────────────────

    /// The start record is read once, before the retry loop. When that read
    /// fails the call ends `UNAVAILABLE` with nothing sent: no commit
    /// `EVALSHA` / slow-path pipeline is dispatched (the dispatch seams never
    /// fire) and nothing is applied, so a re-send is safe. The read is not
    /// retried within the call: of two queued read failures exactly one is
    /// consumed. A retried read could observe the client's NEXT commit as the
    /// call's start, which is the window the guard closes.
    async fn failed_start_record_read_ends_the_call_before_any_attempt(
        backend: UpdateLayerBackend,
        layer: &'static str,
    ) {
        let fx = fixture(layer).await;
        let read_failures: FredQueue = Arc::new(Mutex::new(
            (0..2)
                .map(|_| FredError::new(FredErrorKind::IO, "bv6d: start-record read failed"))
                .collect(),
        ));
        let dispatched = Arc::new(AtomicUsize::new(0));
        let count: WatchWindowHook = {
            let dispatched = Arc::clone(&dispatched);
            Arc::new(move |_client: &Client| {
                dispatched.fetch_add(1, Ordering::SeqCst);
                async {}.boxed()
            })
        };
        let caller = store(&fx, backend)
            .with_start_record_read_fred_error_injector_for_test(read_failures.clone())
            .with_lua_commit_pre_dispatch_hook_for_test(count.clone())
            .with_pre_dispatch_hook_for_test(count);

        let refused = update(
            &caller,
            &fx,
            field_delta(0x29),
            WRITER,
            ORIGINAL_TS,
            Some(update_op(&fx, PAYLOAD)),
        )
        .await;
        assert!(
            matches!(refused, Err(ServiceError::Unavailable(_))),
            "a failed start-record read must end the call UNAVAILABLE; got {refused:?}"
        );
        assert_eq!(
            read_failures.lock().unwrap().len(),
            1,
            "the start record is read exactly once per call: one injected failure consumed, one \
             left"
        );
        assert_eq!(
            dispatched.load(Ordering::SeqCst),
            0,
            "no commit may be dispatched without a start record"
        );
        assert_eq!(header_seq(&fx, fx.object_id).await, 1, "nothing applied");
        assert_stream_commits(&fx, fx.object_id, &[], "no commit entry").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lua_failed_start_record_read_ends_the_call_before_any_attempt() {
        failed_start_record_read_ends_the_call_before_any_attempt(
            UpdateLayerBackend::Lua,
            "bv6d-t28-read-lua",
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn txn_failed_start_record_read_ends_the_call_before_any_attempt() {
        failed_start_record_read_ends_the_call_before_any_attempt(
            UpdateLayerBackend::Txn,
            "bv6d-t28-read-txn",
        )
        .await;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // T25 / T32 — the commit EVALSHA is written to the wire at most once
    // ─────────────────────────────────────────────────────────────────────────

    /// Wait until `stall`'s connection has read more than `baseline` bytes —
    /// a command was written to it, and queues behind the stall.
    async fn await_queued_behind(stall: &ConnStall, baseline: u64, what: &str) {
        let deadline = tokio::time::Instant::now() + RETRACT_WAIT;
        loop {
            if stall.net_in().await.unwrap_or(0) > baseline {
                return;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "{what} never reached the stalled connection"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Wait for `client`'s data pool to serve again after its connection was
    /// killed. A command fred re-writes after the reconnect (a replay) is
    /// written before this `PING`, on the same connection, so by the time the
    /// `PING` is answered any replay has executed.
    async fn await_reconnected(client: &live_edit::storage::client::ValkeyClient) {
        use fred::interfaces::ClientLike;
        let deadline = tokio::time::Instant::now() + RETRACT_WAIT;
        loop {
            let ping = tokio::time::timeout(
                Duration::from_secs(1),
                client.pool.next().ping::<fred::types::Value>(None),
            )
            .await;
            if matches!(ping, Ok(Ok(_))) {
                return;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "the data pool never reconnected after the kill"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// T25: a dropped caller's commit is never replayed by fred.
    ///
    /// Store A has ONE data-pool connection, parked in a server-side `BLPOP`,
    /// so call X's commit `EVALSHA` is written but queued behind it. X's future
    /// is then dropped (a deadline drop: it never times out itself, so fred
    /// would not filter it). Through store B the client moves on — with an
    /// id, its re-send X' applies (X never ran) and its next write Y commits;
    /// without one, Y commits. Then the parked connection is killed and A
    /// reconnects. With the commit written once, the queued `EVALSHA` died with
    /// the connection; with fred's default three attempts it would be re-written
    /// after the reconnect and run AFTER Y.
    ///
    /// With an id, either half of the fix holds the line (the guard would
    /// refuse the replay as overtaken); WITHOUT one only the once-only dispatch
    /// does, so the no-id variant is the one that discriminates it.
    async fn dropped_callers_commit_is_not_replayed_after_a_reconnect(
        with_id: bool,
        layer: &'static str,
    ) {
        let fx = fixture_on(TestValkey::setup_with_pool_size(1).await, layer).await;
        let other_client = TestValkey::setup().await;
        let store_b = LayerStore::new(
            other_client.client().clone(),
            fx.tv.key_schema().clone(),
            4096,
        )
        .with_update_layer_retry_max_attempts(MAX_ATTEMPTS)
        .with_metrics(fx.metrics.clone());

        let dispatched = Arc::new(tokio::sync::Notify::new());
        let hook: WatchWindowHook = {
            let dispatched = Arc::clone(&dispatched);
            Arc::new(move |_client: &Client| {
                dispatched.notify_one();
                async {}.boxed()
            })
        };
        let store_a = Arc::new(
            store(&fx, UpdateLayerBackend::Lua).with_lua_commit_pre_dispatch_hook_for_test(hook),
        );
        let x_operation = if with_id {
            let start = start_record_now(&store_a, &fx).await;
            Some(named_op(&fx, "bv6d-t25-x", PAYLOAD).with_start_record(start))
        } else {
            None
        };

        let stall = ConnStall::park(
            fx.tv.client().pool.next(),
            fx.tv.client().maintenance_pool.next(),
            format!("{}bv6d-t25-stall", fx.tv.prefix()),
        )
        .await;
        let baseline = stall.net_in().await.unwrap_or(0);
        let fx = Arc::new(fx);
        let x = {
            let (store_a, fx) = (Arc::clone(&store_a), Arc::clone(&fx));
            tokio::spawn(async move {
                update(
                    &store_a,
                    &fx,
                    field_delta(0x25),
                    WRITER,
                    ORIGINAL_TS,
                    x_operation,
                )
                .await
            })
        };
        tokio::time::timeout(RETRACT_WAIT, dispatched.notified())
            .await
            .expect("X never reached its commit dispatch");
        await_queued_behind(&stall, baseline, "X's commit EVALSHA").await;
        x.abort();
        assert!(
            x.await.unwrap_err().is_cancelled(),
            "precondition: X's future was dropped with its EVALSHA queued"
        );

        let mut expected = Vec::new();
        if with_id {
            let resent = update(
                &store_b,
                &fx,
                field_delta(0x25),
                WRITER,
                RESEND_TS,
                Some(named_op(&fx, "bv6d-t25-x", PAYLOAD)),
            )
            .await
            .expect("the re-send X' must apply");
            assert_eq!(
                (resent.seq, resent.outcome),
                (2, OperationOutcome::Applied),
                "precondition: X never executed, so its re-send is the first commit"
            );
            expected.push((2, WRITER));
        }
        let y = update(
            &store_b,
            &fx,
            field_delta(0x26),
            WRITER,
            RESEND_TS,
            with_id.then(|| named_op(&fx, "bv6d-t25-y", OTHER_PAYLOAD)),
        )
        .await
        .expect("the client's next write Y must commit");
        expected.push((y.seq, WRITER));

        let killed = stall.kill().await;
        assert!(
            killed.is_err(),
            "precondition: the stall was ended by the kill, not released; got {killed:?}"
        );
        await_reconnected(fx.tv.client()).await;

        assert_stream_commits(
            &fx,
            fx.object_id,
            &expected,
            "the dropped X must not have run after the reconnect — an extra entry after Y is \
             fred replaying its EVALSHA, reordered after the client's next write",
        )
        .await;
        assert_eq!(header_seq(&fx, fx.object_id).await, y.seq);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropped_callers_commit_with_an_operation_id_is_not_replayed_after_a_reconnect() {
        dropped_callers_commit_is_not_replayed_after_a_reconnect(true, "bv6d-t25-id").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropped_callers_commit_without_an_operation_id_is_not_replayed_after_a_reconnect() {
        dropped_callers_commit_is_not_replayed_after_a_reconnect(false, "bv6d-t25-no-id").await;
    }

    /// T32: the commit `EVALSHA`'s attempt budget of one actually reaches fred.
    /// (`with_options(..).evalsha(..)` compiles and silently drops it, because
    /// fred 10.1.0's `WithOptions` has no Lua interface.)
    ///
    /// Attempt 1's `EVALSHA` is written to the very connection the pre-dispatch
    /// hook hands over, which the hook has parked in a server-side `BLPOP`;
    /// once the `EVALSHA` is queued behind it, the connection is killed. With
    /// one attempt fred fails attempt 1 (transiently) and `update_layer`'s own
    /// loop runs attempt 2, which commits: the hook fires twice. With fred's
    /// default budget the connection's reconnect re-writes attempt 1's
    /// `EVALSHA` — a replay — which commits, so attempt 1 "succeeds" and the
    /// hook fires once. Either way exactly one commit lands; the dispatch count
    /// is what tells a loop retry from a fred replay.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn commit_evalsha_is_written_once_so_a_killed_connection_fails_the_attempt() {
        let fx = fixture_on(TestValkey::setup_with_pool_size(1).await, "bv6d-t32").await;
        let observer = fx.tv.client().maintenance_pool.next().clone();
        let stall_key = format!("{}bv6d-t32-stall", fx.tv.prefix());
        let dispatches = Arc::new(AtomicUsize::new(0));
        type Killer = tokio::task::JoinHandle<Result<fred::types::Value, FredError>>;
        let killer: Arc<Mutex<Option<Killer>>> = Arc::new(Mutex::new(None));
        let hook: WatchWindowHook = {
            let (dispatches, killer) = (Arc::clone(&dispatches), Arc::clone(&killer));
            Arc::new(move |client: &Client| {
                let first = dispatches.fetch_add(1, Ordering::SeqCst) == 0;
                let (client, observer, stall_key, killer) = (
                    client.clone(),
                    observer.clone(),
                    stall_key.clone(),
                    Arc::clone(&killer),
                );
                async move {
                    if !first {
                        return;
                    }
                    let stall = ConnStall::park(&client, &observer, stall_key).await;
                    let baseline = stall.net_in().await.unwrap_or(0);
                    *killer.lock().unwrap() = Some(tokio::spawn(async move {
                        await_queued_behind(&stall, baseline, "attempt 1's commit EVALSHA").await;
                        stall.kill().await
                    }));
                }
                .boxed()
            })
        };
        let caller =
            store(&fx, UpdateLayerBackend::Lua).with_lua_commit_pre_dispatch_hook_for_test(hook);

        let committed = update(&caller, &fx, field_delta(0x32), WRITER, ORIGINAL_TS, None)
            .await
            .expect("the update must commit");
        let killed = killer
            .lock()
            .unwrap()
            .take()
            .expect("precondition: attempt 1's hook parked its connection");
        let killed = killed.await.expect("the killer task must not panic");
        assert!(
            killed.is_err(),
            "precondition: attempt 1's connection was killed, not released; got {killed:?}"
        );

        assert_eq!(
            dispatches.load(Ordering::SeqCst),
            2,
            "attempt 1's EVALSHA must FAIL with its connection and the call's own loop must \
             dispatch attempt 2. One dispatch means fred replayed attempt 1 after the reconnect: \
             the commit's one-attempt budget did not reach fred"
        );
        assert_eq!(committed.seq, 2);
        assert_stream_commits(&fx, fx.object_id, &[(2, WRITER)], "exactly one commit").await;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // T15 — a duplicate create that loses the race at the commit script
    // ─────────────────────────────────────────────────────────────────────────

    /// Call H reads the mapping and the record (no record yet), allocates and
    /// materializes its object. Inside the window before its commit script, a
    /// duplicate W with the same id runs to completion and commits. H's script
    /// then meets W's record: it writes nothing, and H is answered W's
    /// `object_id` with `ALREADY_COMMITTED`.
    ///
    /// Run as an OVERWRITE of an existing layer, which is where the race leaves
    /// something behind: H's own materialized object is an orphan the mapping
    /// never names. H's pre-stored hashes are NOT released here: its orphan's
    /// retirement releases them, and releasing here too would release them
    /// twice.
    ///
    /// The test also checks that W's object carries no `superseded` mark, as
    /// T15 asks. In this ordering that check cannot fail: H marks the object
    /// its `MGET` read as the mapping, which is `previous`, not W's. The hazard
    /// behind it — a read that sees W's object as the mapping without W's
    /// record, so the loser marks the live layer `superseded` and every write
    /// to it fails `FAILED_PRECONDITION` — is ruled out by reading the mapping
    /// and the record in one `MGET`, which no ordering this test can drive
    /// splits.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_duplicate_that_commits_first_is_answered_from_the_commit_script() {
        const LAYER: &str = "bv6d-t15-race";
        let fx = Arc::new(fixture("bv6d-t15-race-fixture").await);
        let plain = store(&fx, UpdateLayerBackend::Lua);
        let previous = create(&plain, &fx, LAYER, None)
            .await
            .expect("the layer to overwrite must exist")
            .object_id;

        let winner_store = Arc::new(store(&fx, UpdateLayerBackend::Lua));
        let winner: Arc<Mutex<Option<CreateCommit>>> = Arc::new(Mutex::new(None));
        let hook: WatchWindowHook = {
            let (winner_store, winner, fx) = (winner_store.clone(), winner.clone(), fx.clone());
            Arc::new(move |_client: &Client| {
                let (winner_store, winner, fx) = (winner_store.clone(), winner.clone(), fx.clone());
                async move {
                    let commit = create(&winner_store, &fx, LAYER, Some(create_op(LAYER, PAYLOAD)))
                        .await
                        .expect("the competing duplicate must commit — it is uncontended");
                    *winner.lock().unwrap() = Some(commit);
                }
                .boxed()
            })
        };
        let racing =
            store(&fx, UpdateLayerBackend::Lua).with_create_layer_pre_commit_hook_for_test(hook);

        let h = ext_hash(0xC7);
        pre_store(&fx.ext, &[h]).await;
        let answered = create(
            &racing,
            &fx,
            LAYER,
            Some(create_op(LAYER, PAYLOAD).with_prestored_ext_hashes(vec![h])),
        )
        .await
        .expect("the duplicate that lost the race must be answered OK");
        let winner = winner
            .lock()
            .unwrap()
            .expect("precondition: the hook ran and the competing duplicate committed");

        assert_eq!(
            winner.outcome,
            OperationOutcome::Applied,
            "precondition: the duplicate in the window committed first"
        );
        assert_eq!(
            answered,
            CreateCommit {
                object_id: winner.object_id,
                outcome: OperationOutcome::AlreadyCommitted,
            },
            "the loser is answered the WINNER's object_id; its own object is not the layer"
        );
        assert_eq!(
            mapping(&fx, LAYER).await,
            Some(winner.object_id),
            "the losing script wrote nothing: the mapping still names the winner"
        );
        let winner_superseded: Option<String> = fx
            .tv
            .client()
            .pool
            .hget(
                fx.tv.key_schema().obj_header(&winner.object_id.to_string()),
                "superseded",
            )
            .await
            .expect("HGET superseded must succeed");
        assert_eq!(
            winner_superseded, None,
            "the live winner carries no superseded mark (the loser's mark targets `previous`, \
             the object its MGET read)"
        );
        // The loser allocated first (before its hook ran), the winner second.
        let loser_object = previous + 1;
        assert_eq!(
            winner.object_id,
            previous + 2,
            "precondition: the loser allocated before the winner"
        );
        assert!(
            header_exists(&fx, loser_object).await,
            "precondition: the loser materialized its own object before losing at the script"
        );
        assert_ne!(
            mapping(&fx, LAYER).await,
            Some(loser_object),
            "the loser's object is an orphan"
        );
        assert!(
            pending_members(&fx.tv).await.is_empty(),
            "the script-race answer releases nothing — the orphan's retirement owns those \
             references"
        );
        assert_eq!(
            operation_match_cells(&fx.metrics),
            [0, 0, 1, 0],
            "the loser's answer is one create already_committed match"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Amendment R4-1 — an overwrite whose commit script failed tries it once more
// ═════════════════════════════════════════════════════════════════════════════

/// Amendment R4-1: `CreateLayer`'s commit point is one `create_layer_commit.lua`
/// `EVALSHA` on the `txn_pool`, which follows no redirect and is never
/// replayed. An overwrite has already marked the old object superseded when it
/// gets there, so a flip that does not land leaves the layer refusing every
/// `UpdateLayer` with `FAILED_PRECONDITION`. When that `EVALSHA` fails, the
/// call reads the mapping, its commit claim and — with an operation id — its
/// operation record in one `MGET`, and decides in the commit script's order:
/// record, claim, mapping (b.ua9):
///
/// | The check finds | First try failed | Second try failed too |
/// |---|---|---|
/// | the call's own record, the claim `committed`, or the mapping naming this call's object | INFO, the overwrite tail, success | INFO with `second_try`, success |
/// | a record naming another object | answered from that record, nothing more fenced | the same |
/// | no record or claim, the mapping still the expected value | WARN, one more try | one ERROR (`consequence`: not landed), the error as a plain `UNAVAILABLE`, a finisher |
/// | no record or claim, a third object | re-plan against it (INFO), another run | the same |
/// | (the read fails) | one ERROR (`consequence`: unknown), the error as a plain `UNAVAILABLE`, a finisher | — |
///
/// A new layer, whose expected value is an absent mapping, is tried twice the
/// same way. Every failure here is injected with the commit-dispatch seam
/// (`with_create_layer_commit_fault_injector_for_test`) or, where a test needs
/// the store itself to answer differently, a [`ScriptReplyProxy`] in front of
/// it. An injected fault proves the classification and the ordering of the
/// tries, not the rerouting of a stale `txn_pool` member: the real `-ASK` /
/// `-MOVED` on a clustered facade, and the routing sync before the second try,
/// are in `b_v6d_router_test.rs`. An overwrite that gives up spawns an
/// in-process finisher; the tests that give up install one with no budget, so
/// the state the call left holds for their assertions (the finisher itself is
/// `b_ua9_tests.rs`'s, its routing syncs `b_v6d_router_test.rs`'s). The tests
/// here give up on transient faults; the plain `UNAVAILABLE` a give-up after
/// a permanent-class fault answers, and its lack of a refusal detail, are
/// pinned at the wire in `tests/integration/grpc/upstream_unavailable_test.rs`.
///
/// Log lines are pinned where they are the deliverable: the ERROR lines'
/// `consequence`, and the INFO that tells a lost reply from a second-try
/// commit.
#[cfg(feature = "test-fault-injection")]
mod overwrite_second_try {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use fred::clients::Client;
    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use futures::FutureExt;

    use live_edit::storage::client::BUDGET_EXHAUSTED_SHAPES;
    use live_edit::storage::layer::CreateCommitFault;
    use live_edit::storage::lua::CREATE_LAYER_COMMIT_SCRIPT;
    use live_edit::storage::notifications::DELETED_STREAM_PAYLOAD;
    use live_edit::storage::watched_txn::WatchWindowHook;

    use crate::grpc::log_capture::{capture, Captured};
    use crate::storage::snapshot_read_support::{
        client_through, error_reply_script, test_valkey_addr, ProxyAction, ProxyRule,
        ScriptReplyProxy,
    };

    use super::*;

    type Faults = Arc<Mutex<VecDeque<CreateCommitFault>>>;

    const WARN_SECOND_TRY: &str = "CreateLayer overwrite commit failed and the mapping still \
                                   names the superseded old object; trying the commit once more";
    const WARN_NEW_LAYER_SECOND_TRY: &str = "CreateLayer commit failed and the layer is still \
                                             unmapped; trying the commit once more";
    const INFO_REPLY_LOST: &str =
        "CreateLayer commit's reply was lost, but its check found the commit landed";
    const INFO_OWN_RECORD: &str = "CreateLayer commit found its own operation record: an earlier \
                                   run committed and lost its reply";
    const INFO_REPLAN: &str = "CreateLayer re-plans: another CreateLayer replaced the layer since \
                               this call read its mapping; replacing that object instead";
    const ERROR_BUDGET_SPENT: &str =
        "CreateLayer overwrite commit failed and its run budget is spent";
    const ERROR_UNKNOWN: &str = "CreateLayer overwrite commit failed and the mapping check \
                                 failed too; whether the commit landed is unknown";
    /// The opening words of each ERROR line's `consequence`. Restated, not
    /// imported: they are the operator contract.
    const NOT_LANDED_CONSEQUENCE: &str = "the replaced object stays mapped and marked superseded";
    const UNKNOWN_CONSEQUENCE: &str = "unknown whether the commit landed";

    /// A store whose in-process commit finisher gives up before its first
    /// run, so a call that gives up leaves the store exactly as the call left
    /// it for the test's assertions (b.ua9: the finisher otherwise completes
    /// the overwrite in the background).
    fn without_finisher(store: LayerStore) -> LayerStore {
        store.with_create_commit_finisher_timing_for_test(
            Duration::from_millis(1),
            Duration::from_millis(1),
            Duration::ZERO,
        )
    }

    /// Release the overwrite cleanup body held on `gate` (the store's
    /// `with_overwrite_cleanup_gate_for_test`), within [`CALL_BUDGET`].
    async fn release(gate: &tokio::sync::Barrier) {
        tokio::time::timeout(CALL_BUDGET, gate.wait())
            .await
            .expect("the held overwrite cleanup must reach its gate");
    }

    /// A proxy rule answering the call's post-failure check — the `MGET` of
    /// the layer mapping, the call's commit claim and, for a call carrying an
    /// operation id, its operation record; the only `MGET` naming a commit
    /// claim — with `mapping` and every other key absent: a check that read
    /// the store before an earlier run landed (so it misses the run's claim
    /// and record too). The reply is sized to the keys the `MGET` names.
    fn stale_check(mapping: String) -> ProxyRule {
        Arc::new(move |args: &[Vec<u8>]| {
            (args.first()?.eq_ignore_ascii_case(b"MGET")
                && args
                    .iter()
                    .any(|a| a.windows(13).any(|w| w == b":createclaim:")))
            .then(|| {
                let absent = ", false".repeat(args.len().saturating_sub(2));
                ProxyAction::Eval(format!("return {{'{mapping}'{absent}}}"))
            })
        })
    }

    /// A transient failure of the commit `EVALSHA`: a connection that dropped
    /// before the write (`script_ran: false`), or a lost reply (`true`).
    fn io(script_ran: bool) -> CreateCommitFault {
        CreateCommitFault {
            error: FredError::new(FredErrorKind::IO, "bv6d: connection reset"),
            script_ran,
        }
    }

    fn faults(list: impl IntoIterator<Item = CreateCommitFault>) -> Faults {
        Arc::new(Mutex::new(list.into_iter().collect()))
    }

    fn remaining(faults: &Faults) -> usize {
        faults.lock().unwrap().len()
    }

    /// A store whose commit dispatches pop `faults`.
    fn overwriter(fx: &Fixture, faults: &Faults) -> LayerStore {
        store(fx, UpdateLayerBackend::Lua)
            .with_create_layer_commit_fault_injector_for_test(Arc::clone(faults))
    }

    /// The object `layer`'s mapping names, read directly.
    async fn mapping(fx: &Fixture, layer: &str) -> Option<String> {
        fx.tv
            .client()
            .pool
            .get(fx.tv.key_schema().layer_mapping(&fx.session_id, layer))
            .await
            .expect("GET of the layer mapping must succeed")
    }

    /// One update of `object_id` on the fixture's layer, under [`CALL_BUDGET`].
    async fn update_object(
        store: &LayerStore,
        fx: &Fixture,
        object_id: u64,
        value: u8,
    ) -> Result<UpdateCommit, ServiceError> {
        tokio::time::timeout(
            CALL_BUDGET,
            store.update_layer_with_operation_relayed(
                &fx.session_id,
                fx.layer,
                object_id,
                Some(field_delta(value)),
                WRITER,
                ORIGINAL_TS,
                None,
            ),
        )
        .await
        .expect("update_layer_with_operation did not settle within the call budget")
    }

    /// The events at `level` carrying `message`.
    fn lines<'c>(captured: &'c Captured, level: &str, message: &str) -> Vec<&'c serde_json::Value> {
        captured
            .with_message(message)
            .into_iter()
            .filter(|event| event["level"] == level)
            .collect()
    }

    fn error_lines(captured: &Captured) -> Vec<&serde_json::Value> {
        captured
            .events
            .iter()
            .filter(|event| event["level"] == "ERROR")
            .collect()
    }

    /// Assert the old object's stream got the overwrite tail's `deleted`
    /// wake-up: the call went on through the tail after its commit.
    async fn assert_tail_ran(fx: &Fixture) {
        let entries = stream_payloads(fx, fx.object_id).await;
        assert_eq!(
            entries.last().map(|(_, payload)| payload.as_str()),
            Some(DELETED_STREAM_PAYLOAD),
            "the overwrite tail must append `deleted` to the old object's stream; got {entries:?}"
        );
    }

    /// A first try that fails before its script runs, in each transient shape
    /// the `txn_pool` gives a commit `EVALSHA`: a dropped connection (`IO`),
    /// a write budget exhausted (`Unknown` "Too many failed write attempts.",
    /// the `MOVED` shape a real cluster gave), and the two `-ASK` / redirect
    /// shapes (`Routing` "Too many redirections." — the `txn_pool` follows no
    /// redirect — and "Max attempts reached."). Each time the second try
    /// commits: the call answers the new object and the mapping names it. An
    /// update of the last one then commits.
    ///
    /// The faults are injected: this proves each shape takes the second-try
    /// path, and that the second try runs after the mapping check, not that a
    /// stale `txn_pool` member is rerouted. The store is standalone, so the
    /// routing sync a clustered second try runs first never runs here — its
    /// counter stays at zero; `b_v6d_router_test.rs` runs it on a clustered
    /// facade.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn overwrite_second_try_commits_after_each_transient_shape_of_a_failed_first_try() {
        let fx = fixture("bv6d-r41-shapes").await;
        let mut shapes = vec![(FredErrorKind::IO, "bv6d: connection reset")];
        shapes.extend(BUDGET_EXHAUSTED_SHAPES[..3].iter().cloned());
        let syncs = Arc::new(AtomicUsize::new(0));
        let mut replaced = fx.object_id;
        for (kind, message) in shapes {
            let old = mapping(&fx, fx.layer)
                .await
                .expect("precondition: the layer is mapped");
            let injected = faults([CreateCommitFault {
                error: FredError::new(kind.clone(), message),
                script_ran: false,
            }]);
            let store = overwriter(&fx, &injected)
                .with_create_layer_commit_cluster_sync_counter_for_test(Arc::clone(&syncs));

            let created = create(&store, &fx, fx.layer, None)
                .await
                .unwrap_or_else(|e| {
                    panic!(
                        "{kind:?} {message:?}: the overwrite must commit on its second try; got \
                         {e:?}"
                    )
                });

            assert_eq!(
                remaining(&injected),
                0,
                "precondition: the first try's {kind:?} failure was injected"
            );
            assert_ne!(created.object_id.to_string(), old);
            assert_eq!(
                mapping(&fx, fx.layer).await,
                Some(created.object_id.to_string()),
                "{kind:?} {message:?}: the second try must flip the mapping to the new object"
            );
            replaced = created.object_id;
        }
        let updated = update_object(&store(&fx, UpdateLayerBackend::Lua), &fx, replaced, 0x41)
            .await
            .expect("an update of the last overwrite's object must commit");
        assert_eq!(updated.seq, 2);
        assert_eq!(
            syncs.load(Ordering::SeqCst),
            0,
            "the routing sync runs only against a clustered store"
        );
    }

    /// The second try's own reply is lost (it ran: `script_ran: true` on the
    /// second dispatch, after a first try that never ran). The mapping check
    /// after it names this call's object, so the call succeeds through the
    /// overwrite tail and logs the lost-reply INFO with `second_try = true` —
    /// after exactly one WARN for the second try. With an operation id, the
    /// answer is `APPLIED`: the second try wrote the call's record.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn overwrite_second_try_whose_reply_is_lost_succeeds_through_the_tail() {
        let fx = fixture("bv6d-r41-second-lost").await;
        let injected = faults([io(false), io(true)]);
        let store = overwriter(&fx, &injected);

        let (created, captured) = capture(create(
            &store,
            &fx,
            fx.layer,
            Some(create_op(fx.layer, PAYLOAD)),
        ))
        .await;

        let created = created.expect("the overwrite must succeed: its second try committed");
        assert_eq!(remaining(&injected), 0, "precondition: two dispatches");
        assert_eq!(created.outcome, OperationOutcome::Applied);
        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(created.object_id.to_string())
        );
        assert_eq!(lines(&captured, "WARN", WARN_SECOND_TRY).len(), 1);
        let lost = lines(&captured, "INFO", INFO_REPLY_LOST);
        assert_eq!(lost.len(), 1, "full capture:\n{}", captured.text);
        assert_eq!(lost[0]["fields"]["second_try"], true);
        assert!(error_lines(&captured).is_empty(), "{}", captured.text);
        assert_tail_ran(&fx).await;
    }

    /// The second try meets this call's own operation record. Staged: the
    /// first try runs (it writes the mapping, the claim and the record) and
    /// its reply is lost, and a proxy answers the check's `MGET` with the OLD
    /// object and no claim or record — as a check would that read the store before a
    /// late-running first try landed. The second try then runs for real and
    /// the script answers `ALREADY_COMMITTED` with the call's own record
    /// (the record is checked before the claim): the commit point is this
    /// call's, so the answer is `APPLIED`, the own-record INFO is logged once,
    /// and the overwrite tail runs. (A record that is not the call's own is a
    /// concurrent duplicate, answered `ALREADY_COMMITTED` with the tail
    /// skipped.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn overwrite_second_try_that_meets_its_own_record_answers_applied_through_the_tail() {
        let fx = fixture("bv6d-r41-own-record").await;
        let proxy =
            ScriptReplyProxy::start(test_valkey_addr(), stale_check(fx.object_id.to_string()))
                .await;
        let injected = faults([io(true)]);
        let store = LayerStore::new(
            client_through(&proxy).await,
            fx.tv.key_schema().clone(),
            4096,
        )
        .with_create_layer_commit_fault_injector_for_test(Arc::clone(&injected));

        let (created, captured) = capture(create(
            &store,
            &fx,
            fx.layer,
            Some(create_op(fx.layer, PAYLOAD)),
        ))
        .await;

        let created = created.expect("the overwrite must succeed: its first try committed");
        assert_eq!(
            (remaining(&injected), proxy.hits()),
            (0, 1),
            "precondition: the first try was injected and the mapping check reached the proxy"
        );
        assert_eq!(
            created.outcome,
            OperationOutcome::Applied,
            "the record the second try met is this call's own"
        );
        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(created.object_id.to_string())
        );
        assert_eq!(lines(&captured, "WARN", WARN_SECOND_TRY).len(), 1);
        assert_eq!(
            lines(&captured, "INFO", INFO_OWN_RECORD).len(),
            1,
            "full capture:\n{}",
            captured.text
        );
        assert!(error_lines(&captured).is_empty(), "{}", captured.text);
        assert_tail_ran(&fx).await;
    }

    /// The first try's reply is lost (it ran). The mapping check names this
    /// call's object: the call succeeds through the tail with no second
    /// dispatch (the second queued fault is still there) and no WARN, and the
    /// INFO carries `second_try = false`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn overwrite_first_try_whose_reply_is_lost_succeeds_without_a_second_try() {
        let fx = fixture("bv6d-r41-first-lost").await;
        let injected = faults([io(true), io(false)]);
        let store = overwriter(&fx, &injected);

        let (created, captured) = capture(create(&store, &fx, fx.layer, None)).await;

        let created = created.expect("the overwrite must succeed: its first try committed");
        assert_eq!(
            remaining(&injected),
            1,
            "the mapping named the new object, so no second try is dispatched"
        );
        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(created.object_id.to_string())
        );
        assert!(lines(&captured, "WARN", WARN_SECOND_TRY).is_empty());
        let lost = lines(&captured, "INFO", INFO_REPLY_LOST);
        assert_eq!(lost.len(), 1, "full capture:\n{}", captured.text);
        assert_eq!(lost[0]["fields"]["second_try"], false);
        assert_tail_ran(&fx).await;
    }

    /// Both tries fail before their script runs, and the mapping check after
    /// the second still names the old object: one ERROR whose `consequence`
    /// says so, the call's error as a plain `UNAVAILABLE`, and the state the
    /// call leaves for its
    /// finisher and the orphan sweep (here the finisher has no budget, so the
    /// state holds) — the old object stays mapped and superseded, so an update
    /// of it fails `FAILED_PRECONDITION`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn overwrite_failing_on_both_tries_logs_one_error_and_leaves_the_old_object_mapped() {
        let fx = fixture("bv6d-r41-both-fail").await;
        let injected = faults([io(false), io(false)]);
        let store = without_finisher(overwriter(&fx, &injected));

        let (refused, captured) = capture(create(&store, &fx, fx.layer, None)).await;

        assert!(
            matches!(refused, Err(ServiceError::Unavailable(_))),
            "both tries failed transiently; got {refused:?}"
        );
        assert_eq!(remaining(&injected), 0, "precondition: two dispatches");
        assert_eq!(lines(&captured, "WARN", WARN_SECOND_TRY).len(), 1);
        let errors = error_lines(&captured);
        assert_eq!(errors.len(), 1, "full capture:\n{}", captured.text);
        assert_eq!(errors[0]["fields"]["message"], ERROR_BUDGET_SPENT);
        assert!(
            errors[0]["fields"]["consequence"]
                .as_str()
                .is_some_and(|c| c.starts_with(NOT_LANDED_CONSEQUENCE)),
            "{}",
            errors[0]
        );
        assert_eq!(mapping(&fx, fx.layer).await, Some(fx.object_id.to_string()));
        let stale = update_object(&store, &fx, fx.object_id, 0x42).await;
        assert!(
            matches!(stale, Err(ServiceError::FailedPrecondition(_))),
            "the old object stays mapped but superseded; got {stale:?}"
        );
    }

    /// The mapping names a third object after the first try (another create
    /// took the layer: staged in the window before the commit point). No run
    /// of this call landed — its claim is absent — so the call re-plans
    /// against the third object: it fences it, runs again, and commits. One
    /// re-plan INFO, no second-try WARN, no ERROR.
    ///
    /// Before b.ua9 the call answered its error and left the layer to the
    /// third object, its own object an orphan.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn overwrite_whose_layer_was_taken_by_another_create_re_plans_against_it_and_commits() {
        const THIRD: &str = "987654321";
        let fx = fixture("bv6d-r41-third-first").await;
        let injected = faults([io(false)]);
        let mapping_key = fx.tv.key_schema().layer_mapping(&fx.session_id, fx.layer);
        let take_layer: WatchWindowHook = Arc::new(move |client: &Client| {
            let (client, key) = (client.clone(), mapping_key.clone());
            async move {
                let _: () = client
                    .set(key, THIRD, None, None, false)
                    .await
                    .expect("the competing create's SET must succeed");
            }
            .boxed()
        });
        // Holds the overwrite tail's spawned cleanup, which would delete the
        // third object's (fence-only) header, until the fence is read.
        let gate = Arc::new(tokio::sync::Barrier::new(2));
        let store = overwriter(&fx, &injected)
            .with_create_layer_pre_commit_hook_for_test(take_layer)
            .with_overwrite_cleanup_gate_for_test(Arc::clone(&gate));

        let (created, captured) = capture(create(&store, &fx, fx.layer, None)).await;

        let created = created.expect("the call re-plans against the third object and commits");
        assert_eq!(
            remaining(&injected),
            0,
            "precondition: the first run failed"
        );
        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(created.object_id.to_string())
        );
        let third_header = fx.tv.key_schema().obj_header(THIRD);
        let fenced_by: Option<String> = fx
            .tv
            .client()
            .pool
            .hget(&third_header, "superseded_by")
            .await
            .expect("HGET superseded_by must succeed");
        release(&gate).await;
        store.wait_for_cleanup(CALL_BUDGET).await;
        assert_eq!(
            fenced_by,
            Some(created.object_id.to_string()),
            "the re-plan fences the third object, naming this call's object"
        );
        assert_eq!(lines(&captured, "INFO", INFO_REPLAN).len(), 1);
        assert!(lines(&captured, "WARN", WARN_SECOND_TRY).is_empty());
        assert!(error_lines(&captured).is_empty(), "{}", captured.text);
    }

    /// The second try's `EVALSHA` fails after another create took the layer
    /// between the tries (the proxy answers it — the first commit `EVALSHA` it
    /// sees, and only that one — with a script that flips the mapping to a
    /// third object and then fails). The mapping check after it names that
    /// third object and no run of this call landed, so the call re-plans
    /// against it and its next run commits: OK, one second-try WARN, one
    /// re-plan INFO, no ERROR.
    ///
    /// Before b.ua9 the call answered its error and left the layer to the
    /// third object.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn overwrite_whose_layer_was_taken_between_the_tries_re_plans_and_commits() {
        const THIRD: &str = "987654322";
        let fx = fixture("bv6d-r41-third-second").await;
        let mapping_key = fx.tv.key_schema().layer_mapping(&fx.session_id, fx.layer);
        let take_layer_then_fail = format!(
            "redis.call('SET', '{mapping_key}', '{THIRD}') \
             return redis.error_reply('ERR bv6d: another create took the layer')"
        );
        let taken = Arc::new(AtomicUsize::new(0));
        let rule: ProxyRule = Arc::new({
            let sha = CREATE_LAYER_COMMIT_SCRIPT.sha1().to_owned();
            let taken = Arc::clone(&taken);
            move |args: &[Vec<u8>]| {
                (args.first()?.eq_ignore_ascii_case(b"EVALSHA")
                    && args.iter().any(|a| a.as_slice() == sha.as_bytes())
                    && taken.fetch_add(1, Ordering::SeqCst) == 0)
                    .then(|| ProxyAction::Eval(take_layer_then_fail.clone()))
            }
        });
        let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
        // The first try is injected and never sent, so the first commit
        // EVALSHA the proxy sees is the second try's.
        let injected = faults([io(false)]);
        let store = LayerStore::new(
            client_through(&proxy).await,
            fx.tv.key_schema().clone(),
            4096,
        )
        .with_create_layer_commit_fault_injector_for_test(Arc::clone(&injected));

        let (created, captured) = capture(create(&store, &fx, fx.layer, None)).await;

        let created = created.expect("the call re-plans against the third object and commits");
        assert_eq!(
            (remaining(&injected), proxy.hits()),
            (0, 1),
            "precondition: the first try was injected and the second met the competing create"
        );
        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(created.object_id.to_string())
        );
        assert_eq!(lines(&captured, "WARN", WARN_SECOND_TRY).len(), 1);
        assert_eq!(lines(&captured, "INFO", INFO_REPLAN).len(), 1);
        assert!(error_lines(&captured).is_empty(), "{}", captured.text);
    }

    /// The first try fails and the mapping check fails too (the proxy answers
    /// the check's `MGET` of the mapping, the commit claim and — this call
    /// carries an operation id — the operation record with an error):
    /// whether the commit landed is unknown, so the call is not tried again —
    /// one ERROR whose `consequence` says the outcome is unknown, and the
    /// commit's own error as a plain `UNAVAILABLE`. (The call's finisher,
    /// which would carry on in the
    /// background, is given no budget here.) With an operation id the call's
    /// opening read is an `MGET` too, of the mapping and the operation record;
    /// only the check's names a commit claim.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn overwrite_whose_mapping_check_fails_logs_one_error_that_the_outcome_is_unknown() {
        let fx = fixture("bv6d-r41-check-fails").await;
        let rule: ProxyRule = Arc::new(move |args: &[Vec<u8>]| {
            (args.first()?.eq_ignore_ascii_case(b"MGET")
                && args
                    .iter()
                    .any(|a| a.windows(13).any(|w| w == b":createclaim:")))
            .then(|| ProxyAction::Eval(error_reply_script("ERR bv6d: mapping read failed")))
        });
        let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
        let injected = faults([io(false), io(false)]);
        let store = without_finisher(
            LayerStore::new(
                client_through(&proxy).await,
                fx.tv.key_schema().clone(),
                4096,
            )
            .with_create_layer_commit_fault_injector_for_test(Arc::clone(&injected)),
        );

        let (refused, captured) = capture(create(
            &store,
            &fx,
            fx.layer,
            Some(create_op(fx.layer, PAYLOAD)),
        ))
        .await;

        assert!(
            matches!(refused, Err(ServiceError::Unavailable(_))),
            "the call answers the commit's own transient failure; got {refused:?}"
        );
        assert_eq!(
            (remaining(&injected), proxy.hits()),
            (1, 1),
            "precondition: the mapping check reached the proxy; and no second try"
        );
        let errors = error_lines(&captured);
        assert_eq!(errors.len(), 1, "full capture:\n{}", captured.text);
        assert_eq!(errors[0]["fields"]["message"], ERROR_UNKNOWN);
        assert!(
            errors[0]["fields"]["consequence"]
                .as_str()
                .is_some_and(|c| c.starts_with(UNKNOWN_CONSEQUENCE)),
            "{}",
            errors[0]
        );
        assert!(lines(&captured, "WARN", WARN_SECOND_TRY).is_empty());
        assert_eq!(mapping(&fx, fx.layer).await, Some(fx.object_id.to_string()));
    }

    /// b.ua9 test 20: a new layer's first commit run fails before its script
    /// runs. The check reads no mapping and no commit claim — an absent
    /// mapping is the value a new layer planned against, and the claim tells
    /// "not flipped" from "flipped, then deleted" — so the call runs the
    /// commit once more, and it lands: OK, one new-layer second-try WARN.
    ///
    /// Before the commit claim, a new layer got no second try (an absent
    /// mapping could not tell the two apart) and answered `UNAVAILABLE` after
    /// one dispatch.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn new_layer_create_whose_first_run_fails_commits_on_a_second_run() {
        const NEW_LAYER: &str = "bv6d-r41-new-layer";
        let fx = fixture("bv6d-r41-new").await;
        let injected = faults([io(false)]);
        let store = overwriter(&fx, &injected);

        let (created, captured) = capture(create(&store, &fx, NEW_LAYER, None)).await;

        let created = created.expect("the new layer commits on its second run");
        assert_eq!(
            remaining(&injected),
            0,
            "precondition: the first run failed"
        );
        assert_eq!(
            mapping(&fx, NEW_LAYER).await,
            Some(created.object_id.to_string())
        );
        assert_eq!(
            lines(&captured, "WARN", WARN_NEW_LAYER_SECOND_TRY).len(),
            1,
            "full capture:\n{}",
            captured.text
        );
        assert!(lines(&captured, "WARN", WARN_SECOND_TRY).is_empty());
        assert!(error_lines(&captured).is_empty(), "{}", captured.text);
    }

    /// b.ua9 test 8 (the call's half): the call recognizes its own operation
    /// record by its object id, not by the whole record. Staged: the first run
    /// lands with its reply lost — writing the record
    /// `<X>:<old object>:<digest>` — and a proxy answers the check with a
    /// third object and no claim or record, as a check would that read the store before
    /// the landing while another create had moved the layer. The call then
    /// re-plans against that third object: its next run carries the record
    /// `<X>:<third>:<digest>`, and the script answers `ALREADY_COMMITTED` with
    /// the first run's record. Its object id is the call's, so the call
    /// answers `APPLIED` with the own-record INFO.
    ///
    /// Matched on the whole record, the two would differ and the call would
    /// answer the record as another call's — `ALREADY_COMMITTED`, its own
    /// landed commit disowned.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn overwrite_run_that_meets_its_own_record_after_a_re_plan_answers_applied() {
        /// The object the stale check reports, which the re-plan fences.
        const THIRD: &str = "987654323";
        let fx = fixture("bv6d-ua9-own-record-replan").await;
        let proxy =
            ScriptReplyProxy::start(test_valkey_addr(), stale_check(THIRD.to_owned())).await;
        let injected = faults([io(true)]);
        // Holds the overwrite tail's spawned cleanup of the third object.
        let gate = Arc::new(tokio::sync::Barrier::new(2));
        let store = LayerStore::new(
            client_through(&proxy).await,
            fx.tv.key_schema().clone(),
            4096,
        )
        .with_create_layer_commit_fault_injector_for_test(Arc::clone(&injected))
        .with_overwrite_cleanup_gate_for_test(Arc::clone(&gate));

        let (created, captured) = capture(create(
            &store,
            &fx,
            fx.layer,
            Some(create_op(fx.layer, PAYLOAD)),
        ))
        .await;
        let third_fenced: Option<String> = fx
            .tv
            .client()
            .pool
            .hget(fx.tv.key_schema().obj_header(THIRD), "superseded")
            .await
            .expect("HGET superseded must succeed");
        let created = created.expect("the call's first run landed");
        assert_eq!(
            (remaining(&injected), proxy.hits()),
            (0, 1),
            "precondition: the first run lost its reply and the check read the stale mapping"
        );
        assert_eq!(
            third_fenced.as_deref(),
            Some("1"),
            "precondition: the call re-planned against the third object before its next run"
        );
        // Asserted before the held cleanup is released: a call that disowned
        // its record runs no overwrite tail, so nothing would reach the gate.
        assert_eq!(
            created.outcome,
            OperationOutcome::Applied,
            "a record whose object id is the call's own is the call's commit"
        );
        release(&gate).await;
        store.wait_for_cleanup(CALL_BUDGET).await;
        assert_eq!(
            lines(&captured, "INFO", INFO_OWN_RECORD).len(),
            1,
            "full capture:\n{}",
            captured.text
        );
        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(created.object_id.to_string())
        );
    }
}
