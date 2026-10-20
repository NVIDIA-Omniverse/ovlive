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

//! b.ua9 — a `CreateLayer` overwrite's post-commit failures converge without
//! a client retry, at the storage layer.
//!
//! # The defect
//!
//! A `CreateLayer` overwrite fenced the old object (`superseded=1`), flipped
//! the layer mapping, then appended `deleted` to the old object's stream. A
//! failure after the fence left the layer refusing every `UpdateLayer` until a
//! client re-sent the create; a failed wake-up answered an error after the
//! layer was already replaced, skipped the old object's cleanup and the
//! session checkpoint, and left the old stream's subscribers unaware of the
//! move. Concurrent creates could leave an intermediate object unfenced and
//! unwoken, and the orphan sweep, with no grace, could retire a slow create's
//! object before its flip.
//!
//! # The fix, as these tests see it
//!
//! - The flip is the commit point: once a run of the call's commit landed,
//!   the call answers OK and the overwrite tail — wake-up, retried by the
//!   cleanup, then retirement — never fails it.
//! - The commit script flips only from the value the call fenced, and claims
//!   the new object (`le:{sid}:createclaim:{oid}`) `committed` with the flip,
//!   so no object is mapped twice; a moved mapping is re-planned.
//! - A call that gives up with its fence possibly still mapped spawns an
//!   in-process finisher; the orphan sweep completes what the finisher cannot,
//!   retires replaced objects whose tail did not finish, and vetoes an unmapped
//!   object (claim `vetoed`) before retiring it, only past a grace.
//! - Every retirement route wakes the stream through one script
//!   (`retired_object_wake.lua`) before it deletes anything.
//! - The veto opens its one write with a no-op `PEXPIRE` on the absent claim,
//!   so it runs on a full store, and never touches an existing claim (A10).
//! - An operation-id create stores its id and digest on its object, so the
//!   sweep's completion of its fence writes the record the call's own landing
//!   would have (A11); the check after a failed run reads the record first,
//!   so a same-id duplicate never re-plans onto the original's object.
//!
//! # What lives here
//!
//! The numbers are the design's owed tests.
//!
//! | Test | # | What it pins |
//! |---|---|---|
//! | [`create_layer_commit_script_writes_nothing_on_a_moved_mapping_or_a_veto`] | 8 | `MAPPING_CHANGED` (a value or an absent mapping) and `VETOED` write nothing |
//! | [`create_layer_commit_script_claims_the_object_committed_with_its_flip`] | 8 | `COMMITTED` writes the mapping, the index member and the `committed` claim with its TTL |
//! | [`create_layer_veto_script_vetoes_only_an_unclaimed_unmapped_object`] | 8 | the veto writes `vetoed` with its TTL; `MAPPED` and `CLAIMED` write nothing |
//! | [`create_layer_commit_script_never_maps_an_object_a_run_already_mapped`] | 19 | a second run of a landed commit answers `COMMITTED_EARLIER <current>` and writes nothing |
//! | [`orphan_sweep_retires_a_committed_unmapped_object_and_keeps_its_committed_claim`] | 21, 10 (storage half) | a `DeleteLayer` that stopped after its mapping `DEL`: one pass wakes and retires the object, the veto keeps the `committed` claim, a later run answers `COMMITTED_EARLIER` |
//! | [`orphan_sweep_retires_an_orphan_whose_wake_up_the_store_refused_out_of_memory`] | 11, b.pa1 | the sweep's wake-up refused `-OOM` still deletes the orphan, and requests the GC's out-of-memory collection |
//! | [`orphan_sweep_whose_veto_the_store_refused_out_of_memory_leaves_the_object_and_requests_an_oom_collection`] | b.pa1 | the sweep's veto refused `-OOM`: the object kept, no claim, the collection requested |
//! | [`create_layer_overwrite_of_a_non_numeric_mapping_answers_ok_and_writes_no_ghost_header`] | 13 | a corrupt mapping value is overwritten, unfenced |
//! | [`retired_object_wake_script_recreates_an_absent_stream_with_one_deleted_entry_and_an_expiry`] | 26 | the wake script on an absent stream; its field and value are `NOTIFICATION_PAYLOAD_FIELD` / `DELETED_STREAM_PAYLOAD` |
//! | [`seams::create_layer_overwrite_whose_wake_up_fails_once_answers_ok_and_its_cleanup_retries_the_wake_up`] | 1 | OK, the session checkpoint names the new object, the cleanup's retry appends `deleted`, the old header goes |
//! | [`seams::create_layer_overwrite_whose_every_wake_up_fails_keeps_the_old_object_until_one_sweep_pass_retires_it`] | 3 | OK; header and stream kept with no expiry; one pass wakes, expires, deletes, releases once, registers dead |
//! | [`seams::create_layer_overwrite_whose_commit_runs_fail_answers_unavailable_and_its_finisher_completes_it`] | 4 | `UNAVAILABLE`, one ERROR with the not-landed consequence; the finisher's run lands and runs the tail |
//! | [`seams::orphan_sweep_completes_an_overwrite_stranded_after_its_finisher_ran_out_of_budget`] | 5 | the write-refusal window ends with one sweep pass |
//! | [`seams::concurrent_creates_on_one_layer_fence_wake_and_clean_every_replaced_object`] | 6 | the later create wins; the intermediate object is fenced, woken, refused and cleaned |
//! | [`seams::orphan_sweep_vetoes_and_retires_a_slow_creates_object_past_the_grace_and_the_create_answers_unavailable`] | 7 | grace 0: veto, retirement, `UNAVAILABLE`, no mapping to a deleted object |
//! | [`seams::orphan_sweep_leaves_a_slow_creates_object_alone_within_the_grace_and_the_create_commits`] | 7 | default grace: the object survives and the create commits |
//! | [`seams::orphan_sweep_neither_vetoes_nor_retires_the_pending_successor_of_a_mapped_fenced_object`] | 9 | the unmapped successor a mapped fence names is skipped |
//! | [`seams::create_layer_whose_mapping_moves_under_every_run_answers_unavailable_and_fences_no_current_mapping`] | 12 | the re-plan budget |
//! | [`seams::orphan_sweep_runs_the_cleanup_bodies_of_its_overwrite_fenced_orphans_one_at_a_time`] | 14 | max one cleanup body in flight across K = 2 orphans |
//! | [`seams::create_layer_whose_commit_the_sweep_completed_and_another_create_replaced_answers_ok_and_leaves_the_replacement_live`] | 15 | landed (by the sweep), then replaced: OK, the replacement stays live |
//! | [`seams::create_layer_whose_lost_reply_run_landed_before_another_create_replaced_it_answers_ok_and_leaves_the_replacement_live`] | 16 | the same through the check after a lost reply |
//! | [`seams::create_layer_whose_commit_the_sweep_completed_before_a_delete_layer_answers_ok_and_the_layer_stays_deleted`] | 17 | landed, then deleted: OK, the layer stays deleted |
//! | [`seams::create_layer_whose_lost_reply_run_landed_before_a_delete_layer_answers_ok_and_the_layer_stays_deleted`] | 18 | the same through the check |
//! | [`seams::create_layer_commit_finisher_runs_the_overwrite_tail_when_its_object_landed_and_was_replaced`] | 22 | the finisher meets `COMMITTED_EARLIER <B>` and still retires the fenced object |
//! | [`seams::orphan_sweep_never_retires_a_mapped_object_whose_fenced_successor_was_already_committed`] | 23 | arm (i) leaves a mapped object, with one WARN |
//! | [`seams::create_layer_overwrite_whose_wake_up_the_store_refused_out_of_memory_still_retires_the_old_object`] | 25, b.pa1 | the tail's out-of-memory fallback, one WARN, the GC's out-of-memory collection requested |
//! | [`seams::create_layer_overwrite_whose_refused_out_of_memory_commit_run_is_retried_and_lands_requests_an_oom_collection`] | b.pa1 | a commit run refused `-OOM` (injected), then a landing run: OK, and the collection requested — the dispatch's injected-fault branch only |
//! | [`seams::create_layer_overwrite_whose_two_commit_runs_time_out_counts_each_timeout_once`] | b.pa1 | two timed-out runs and a give-up count two txn-pool command timeouts, not three |
//! | [`seams::orphan_sweep_completion_refused_out_of_memory_leaves_the_overwrite_and_requests_an_oom_collection`] | b.pa1 | the sweep's completion refused `-OOM`: the old object still mapped, its successor kept, the collection requested |
//! | [`seams::create_layer_overwrite_inline_wake_up_leaves_one_deleted_entry_and_an_expiry_before_the_cleanup_runs`] | 26 | the tail's wake-up on a present stream |
//! | [`seams::orphan_sweep_runs_no_second_cleanup_body_during_a_healthy_overwrite_within_the_grace`] | 28 | A9: no second body within the successor's grace; one at grace 0 |
//! | [`create_layer_veto_script_answering_claimed_keeps_the_existing_claims_ttl`] | 29 | A10: `CLAIMED` leaves a `committed` and a `vetoed` claim's `PTTL` where it was |
//! | [`create_layer_header_carries_the_operation_id_and_digest_only_for_a_create_with_one`] | A11 | `create_op_id` / `create_op_digest` on an operation-id create's object, neither without an id |
//! | [`seams::orphan_sweep_completed_operation_id_overwrite_answers_a_same_id_resend_already_committed_and_keeps_later_edits`] | 30 (a) | after arm (i) completes E → X and X is edited, a same-id re-send is `ALREADY_COMMITTED` with X, the edit kept; the record is `<X>:<E>:<digest>` |
//! | [`seams::orphan_sweep_completed_operation_id_overwrite_refuses_a_different_request_with_the_same_id`] | 30 (b) | a different request with the same id is `ALREADY_EXISTS` |
//! | [`seams::orphan_sweep_completion_of_a_re_planned_operation_id_overwrite_records_the_re_planned_object`] | 30 (c) | re-planned to E2 and stranded: the completion records `<X>:<E2>:<digest>` |
//! | [`seams::create_layer_duplicate_whose_check_finds_the_original_recorded_answers_from_the_record_and_fences_nothing`] | 31 | an in-flight same-id duplicate's check reads the original's record: `ALREADY_COMMITTED`, the original's object unfenced and writable |
//! | [`seams::orphan_sweep_logs_one_error_and_leaves_a_fence_whose_operation_id_another_layer_recorded`] | 31 (path (b)) | the cross-layer misuse: one ERROR naming the layer and the operation id, the fenced object stays mapped |
//! | [`seams::create_layer_check_finding_its_own_record_while_the_replaced_object_is_still_mapped_retires_nothing`] | 31 (own record) | the check's own-record verdict carries the mapping it read: the tail guard keeps a still-mapped replaced object, one ERROR |
//! | [`create_layer_veto_script_refuses_a_wrong_typed_mapping_or_claim_writing_nothing`] | 8 | the veto's `WRONGTYPE_KEY:` refusals of a foreign-typed mapping or claim write nothing |
//! | [`create_layer_veto_script_answers_session_deleted_writing_nothing`] | 8 | `SESSION_DELETED` writes no claim |
//! | [`create_layer_commit_script_refuses_a_wrong_typed_claim_writing_nothing`] | 8 | the commit script's claim type check: `WRONGTYPE_KEY:`, no mapping, no index member |
//! | [`orphan_sweep_retires_an_unmapped_object_of_a_deleted_session_without_a_claim`] | arm (iii) | after `SESSION_DELETED` the object is retired, unclaimed |
//! | [`orphan_sweep_leaves_an_unmapped_object_whose_veto_meets_a_wrong_typed_claim`] | arm (iii) | after a `WRONGTYPE_KEY` refusal the object is left |
//! | [`seams::orphan_sweep_leaves_a_stranded_overwrite_alone_while_its_successor_is_within_the_grace`] | arm (i) | the default grace: E stays mapped, X unclaimed, both headers intact |
//! | [`seams::orphan_sweep_runs_the_tail_of_a_stranded_overwrite_whose_successor_committed_earlier_and_moved_on`] | arm (i) | `COMMITTED_EARLIER` naming another object: E's tail runs, awaited |
//! | [`seams::orphan_sweep_logs_one_error_and_leaves_a_stranded_overwrite_whose_successor_is_gone`] | arm (i) | a gone successor: one ERROR, E stays mapped and whole |
//! | [`seams::orphan_sweep_completes_a_stranded_overwrite_without_a_record_for_an_invalid_header_operation_id`] | arm (i), A11 | an invalid header operation id is dropped: completed, no record under it |
//! | [`seams::create_layer_overwrite_whose_layer_a_delete_layer_removed_before_its_check_re_plans_as_a_new_layer`] | re-plan | a `DeleteLayer` before the check: the call re-plans as a new layer and commits |
//!
//! Elsewhere: 2, the subscriber half of 6, and 10 through the handlers in
//! `tests/integration/grpc/b_ua9_subscriber_test.rs`; 20 and the
//! own-record-by-object-id half of 8 in `b_v6d_tests.rs`
//! (`overwrite_second_try`); 24 (a real `-OOM`, disruptive; its case (c) is
//! A10's full-store veto) in `store_write_refusal_test.rs`, beside an
//! overwrite's commit run refused `-OOM` by the store itself (a proxy, where
//! the commit test here injects the refusal); the source pin that the commit
//! dispatch counts a failed run and the give-up's mapping does not, in
//! `mapper_routing_test.rs`; 27 beside
//! `KeySchema::create_layer_claim` in `src/storage/keys.rs`; the decision
//! table `decide_create_commit` reads, row by row, in
//! `src/storage/layer/create_commit_decision_tests.rs`.
//!
//! # One sweep pass
//!
//! The sweep is a periodic task: it runs once on start, then sleeps
//! `ORPHAN_SWEEP_INTERVAL`. [`sweep_one_pass`] marks the end of its first
//! pass with a canary: a layer mapping of a session that does not exist,
//! which only the pass's layer phase removes, and which runs after the
//! object phase has retired every object it was going to (one at a time,
//! awaited). The canary's removal is therefore the event "the object phase is
//! done", and a pass that ends in an error leaves it in place and fails the
//! wait with its own message. The task is then cancelled before a second pass.
//!
//! # Feature gate
//!
//! The [`seams`] module drives `LayerStore`'s `test-fault-injection` seams (the
//! commit-dispatch and wake-up injectors, the pre-commit and post-failed-run
//! hooks, the finisher timing, the cleanup gate and probe) and is compiled
//! out, not skipped, without `--features test-fault-injection`. Everything
//! above it needs only Valkey.
//!
//! # Wait budget
//!
//! Every storage call and every wait on a background outcome is a named
//! deadline over one whole operation, never a sleep: [`FIXTURE_BUDGET`] (5 s)
//! over the fixture's session and layer creates, [`CALL_BUDGET`] (15 s) per
//! storage call (a create, a delete or an update),
//! [`HOOKED_CALL_BUDGET`] (45 s) for a call whose hook runs a sweep pass or
//! another create or delete inside it, [`SWEEP_PASS_BUDGET`] +
//! [`SWEEP_JOIN_BUDGET`] (25 s) per pass, [`CLEANUP_BUDGET`] (15 s) per
//! `wait_for_cleanup`, [`SETTLE_BUDGET`] (15 s) per poll of a background
//! outcome or release of a held cleanup, `OOM_REQUEST_BUDGET` (10 s,
//! `oom_request_support`) per wait for the GC's out-of-memory collection
//! request. A single store command a test sends itself — a script run, a
//! read, a planted key, an external-value pre-store — is not wrapped: it is
//! bounded by the client's command timeout, and a test
//! sends a handful. The healthy path pays milliseconds of each. Worst cases,
//! every named wait at its full budget, the fixture's 5 s included (a hook's
//! inner waits are inside its call's budget):
//!
//! | Test | Worst case |
//! |---|---|
//! | 8, 19, 26 (script), 29, veto and commit wrong-type / session-deleted replies | the fixture: 5 s |
//! | 21, arm (iii) after `SESSION_DELETED` / `WRONGTYPE_KEY` | the fixture, one pass: 30 s |
//! | 11, the veto refused `-OOM` | the fixture, one pass, one request wait: 40 s |
//! | 12, 13, A11 header | the fixture, one call: 20 s |
//! | 31 (own record) | the fixture, one call (its hook inside), one cleanup wait: 35 s |
//! | two timed-out commit runs | the fixture, one call, one cleanup wait: 35 s |
//! | 25, a commit run refused `-OOM` then landed | the fixture, one call, one request wait, one cleanup wait: 45 s |
//! | the sweep's completion refused `-OOM` | the fixture, one call, one cleanup wait, one pass, one request wait: 70 s |
//! | 9, 23, arm (i) graced / gone successor / invalid header id / `COMMITTED_EARLIER` | the fixture, one call, one cleanup wait, one pass: 60 s |
//! | 3, 30 (b), 31 (path (b)) | the fixture, two calls, one cleanup wait, one pass: 75 s |
//! | 5, 30 (a) | the fixture, three calls, one cleanup wait, one pass: 90 s |
//! | 30 (c) | the fixture, one hooked call, two cleanup waits, one pass: 105 s |
//! | 31 | the fixture, one hooked call, two cleanup waits, one update: 95 s |
//! | re-plan as a new layer | the fixture, one hooked call, two cleanup waits: 80 s |
//! | 1, 26 (tail) | the fixture, one call, one cleanup wait, one checkpoint drain or release: 50 s |
//! | 4 | the fixture, one call, two settles, one cleanup wait, one update: 80 s |
//! | 7 | the fixture, one hooked call: 50 s |
//! | 14 | the fixture, three calls, one cleanup wait, one pass: 90 s |
//! | 28 | the fixture, one call, one settle, two passes, one release, one cleanup wait: 115 s |
//! | 6 | the fixture, one hooked call, one update, one release, two cleanup waits: 110 s |
//! | 15, 16 | the fixture, one hooked call, two cleanup waits, one update: 95 s |
//! | 17, 18 | the fixture, one hooked call, two cleanup waits: 80 s |
//! | 22 | the fixture, one hooked call, one settle, two cleanup waits: 95 s |
//!
//! Every row is under nextest's 120 s kill.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface, StreamsInterface};
use fred::types::Expiration;
use tokio_util::sync::CancellationToken;

use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::external_values::{ExternalValueStore, ValkeyExternalValueStore};
use live_edit::storage::keys::{hex_encode, KeySchema};
use live_edit::storage::layer::{
    operation_request_digest, CreateCommit, CreateOperation, LayerStore,
};
use live_edit::storage::lua::{
    CREATE_LAYER_COMMIT_SCRIPT, CREATE_LAYER_VETO_SCRIPT, RETIRED_OBJECT_WAKE_SCRIPT,
};
use live_edit::storage::notifications::{DELETED_STREAM_PAYLOAD, NOTIFICATION_PAYLOAD_FIELD};
use live_edit::storage::orphan_sweep::spawn_orphan_sweep;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use crate::storage::b_qpf_tests::ext_store;
use crate::storage::oom_request_support::wait_for_oom_request;
use crate::storage::snapshot_read_support::{
    client_through, error_reply_script, test_valkey_addr, ProxyAction, ProxyRule, ScriptReplyProxy,
};

// ═════════════════════════════════════════════════════════════════════════════
// Budgets and contract constants
// ═════════════════════════════════════════════════════════════════════════════

/// Deadline on one storage call.
const CALL_BUDGET: Duration = Duration::from_secs(15);

/// One deadline over the fixture's setup: its session create and its layer
/// create, a handful of round trips together.
const FIXTURE_BUDGET: Duration = Duration::from_secs(5);

/// Deadline on a call whose test hook runs a sweep pass, another create or
/// delete, or both, inside it: the pass (25 s) and the inner call (15 s) are
/// each bounded by their own budget, and this covers them plus the call's own
/// round trips.
#[cfg(feature = "test-fault-injection")]
const HOOKED_CALL_BUDGET: Duration = Duration::from_secs(45);

/// Deadline on one orphan-sweep pass reaching its canary.
pub(crate) const SWEEP_PASS_BUDGET: Duration = Duration::from_secs(20);

/// Deadline on the sweep task exiting after its cancellation.
const SWEEP_JOIN_BUDGET: Duration = Duration::from_secs(5);

/// Deadline on `LayerStore::wait_for_cleanup`.
#[cfg(feature = "test-fault-injection")]
const CLEANUP_BUDGET: Duration = Duration::from_secs(15);

/// Deadline on a background outcome (a finisher's run, a spawned cleanup
/// entering its probe) becoming observable.
#[cfg(feature = "test-fault-injection")]
const SETTLE_BUDGET: Duration = Duration::from_secs(15);

/// Interval between probes of a polled condition.
const POLL: Duration = Duration::from_millis(20);

/// The replica label a test arms the store out-of-memory signal with: the
/// value a collection request carries.
const OOM_REQUESTER: &str = "bua9-oom-requester";

/// The layer mapping of a session that never exists: the canary
/// [`sweep_one_pass`] waits on.
const CANARY_SESSION: &str = "b-ua9-sweep-canary-session";
const CANARY_LAYER: &str = "b-ua9-sweep-canary";

/// The commit script's expected-mapping encoding (its `ARGV[3]` and the
/// element 1 of a `MAPPING_CHANGED` / `COMMITTED_EARLIER` reply). Restated,
/// not imported: it is the contract between the script and every run.
const MAPPING_ABSENT: &str = "absent";

fn mapping_value(value: &str) -> String {
    format!("value:{value}")
}

/// The commit claim's two values. Restated: the two scripts and the call's
/// check agree on them.
const CLAIM_COMMITTED: &str = "committed";
const CLAIM_VETOED: &str = "vetoed";

/// The `committed` claim's TTL a run passes; any value the test can bound.
const COMMITTED_CLAIM_TTL_MS: u64 = 600_000;

/// The `vetoed` claim's TTL (`keys::CREATE_LAYER_CLAIM_VETOED_TTL_MS`): 24 h.
const VETOED_CLAIM_TTL_MS: i64 = 86_400_000;

/// `STREAM_DELETE_GRACE_MS`: the expiry every retirement gives a stream.
const STREAM_GRACE_MS: i64 = 30_000;

/// The client operation id the operation-id scenarios carry: a UUIDv4 in
/// canonical text, the shape the proto comment recommends.
const OPERATION_ID: &str = "3c9e1f07-5b2d-4a86-8e41-7d0f2a6b9c13";

/// The request bytes an operation's digest is computed over. At this layer
/// the digest is the caller's (the gRPC handler hashes the streamed OMPF), so
/// any stable bytes model "the same request"; [`OTHER_PAYLOAD`] models a
/// different one.
const PAYLOAD: &[u8] = b"b.ua9 original request bytes";
#[cfg(feature = "test-fault-injection")]
const OTHER_PAYLOAD: &[u8] = b"b.ua9 a different request";

/// The object-header columns an operation-id `CreateLayer` writes on its new
/// object (A11), which the orphan sweep's completion of its fence reads.
/// Restated, not imported: the sweep of a newer build reads what an older
/// build wrote, so the names are a contract between builds.
const HEADER_CREATE_OP_ID: &str = "create_op_id";
const HEADER_CREATE_OP_DIGEST: &str = "create_op_digest";

// ═════════════════════════════════════════════════════════════════════════════
// One orphan-sweep pass
// ═════════════════════════════════════════════════════════════════════════════

/// Run exactly one orphan-sweep pass over `client`'s keyspace with
/// `layer_store` (the store the sweep completes overwrites and runs tails
/// through, and whose grace it reads), and return once its object phase is
/// done. See the module docs for the canary.
pub(crate) async fn sweep_one_pass(
    client: &ValkeyClient,
    keys: &KeySchema,
    ext: Arc<dyn ExternalValueStore>,
    layer_store: Arc<LayerStore>,
) {
    plant_sweep_canary(client, keys).await;
    run_planted_sweep_pass(client, keys, ext, layer_store).await;
}

/// The first half of [`sweep_one_pass`]: write the canary its pass removes.
/// Separate for a test that must write it before the store stops taking
/// writes.
pub(crate) async fn plant_sweep_canary(client: &ValkeyClient, keys: &KeySchema) {
    let _: () = client
        .pool
        .set(
            keys.layer_mapping(CANARY_SESSION, CANARY_LAYER),
            "1",
            None,
            None,
            false,
        )
        .await
        .expect("SET of the sweep canary must succeed");
}

/// The second half of [`sweep_one_pass`]: run one pass and wait for it to
/// remove the canary [`plant_sweep_canary`] wrote.
pub(crate) async fn run_planted_sweep_pass(
    client: &ValkeyClient,
    keys: &KeySchema,
    ext: Arc<dyn ExternalValueStore>,
    layer_store: Arc<LayerStore>,
) {
    let canary = keys.layer_mapping(CANARY_SESSION, CANARY_LAYER);
    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        client.clone(),
        keys.clone(),
        ext,
        layer_store,
        crate::common::test_leader("b-ua9-sweep"),
        cancel.clone(),
        None,
    );
    let deadline = tokio::time::Instant::now() + SWEEP_PASS_BUDGET;
    let mut finished = false;
    while tokio::time::Instant::now() < deadline {
        let present: bool = client
            .pool
            .exists(&canary)
            .await
            .expect("EXISTS of the sweep canary must succeed");
        if !present {
            finished = true;
            break;
        }
        tokio::time::sleep(POLL).await;
    }
    cancel.cancel();
    tokio::time::timeout(SWEEP_JOIN_BUDGET, handle)
        .await
        .expect("the orphan sweep must exit within its join budget of cancellation")
        .expect("the orphan sweep must not panic");
    assert!(
        finished,
        "the sweep pass did not reach its layer phase within {SWEEP_PASS_BUDGET:?}: the pass \
         ended in an error, or is stuck in its object phase"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Fixture + readers
// ═════════════════════════════════════════════════════════════════════════════

struct Fx {
    tv: TestValkey,
    ext: Arc<ValkeyExternalValueStore>,
    session_id: String,
    layer: &'static str,
    /// The layer's object at the start: `E` in the design's scenarios.
    object_id: u64,
}

/// A session named `layer` holding one layer `layer`, created with no
/// operation id.
async fn fixture(layer: &'static str) -> Fx {
    fixture_on(TestValkey::setup().await, layer).await
}

async fn fixture_on(tv: TestValkey, layer: &'static str) -> Fx {
    let ext = ext_store(&tv);
    let (session_id, object_id) = tokio::time::timeout(FIXTURE_BUDGET, async {
        let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .create(layer, &format!("/{layer}"), "bua9-user")
            .await
            .expect("session creation must succeed");
        let object_id = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .create_layer(&session_id, layer, "bua9-user", &base_state(), ext.clone())
            .await
            .expect("create_layer must succeed");
        (session_id, object_id)
    })
    .await
    .expect("the fixture's session and layer creates did not settle within their budget");
    Fx {
        tv,
        ext,
        session_id,
        layer,
        object_id,
    }
}

fn base_state() -> ObjectState {
    ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        ..Default::default()
    }
}

/// A store on the fixture's client with no seam.
fn plain_store(fx: &Fx) -> LayerStore {
    LayerStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone(), 4096)
}

/// One `create_layer_with_operation` of `layer` carrying `operation`, under
/// `budget`.
async fn create_op_within(
    store: &LayerStore,
    fx: &Fx,
    layer: &str,
    operation: Option<CreateOperation>,
    budget: Duration,
) -> Result<CreateCommit, ServiceError> {
    tokio::time::timeout(
        budget,
        store.create_layer_with_operation(
            &fx.session_id,
            layer,
            "bua9-user",
            &base_state(),
            fx.ext.clone(),
            operation,
        ),
    )
    .await
    .expect("create_layer_with_operation did not settle within its budget")
}

/// One `create_layer_with_operation` of `layer` (no operation id) under
/// `budget`.
async fn create_within(
    store: &LayerStore,
    fx: &Fx,
    layer: &str,
    budget: Duration,
) -> Result<CreateCommit, ServiceError> {
    create_op_within(store, fx, layer, None, budget).await
}

async fn create(store: &LayerStore, fx: &Fx, layer: &str) -> Result<CreateCommit, ServiceError> {
    create_within(store, fx, layer, CALL_BUDGET).await
}

/// The operation a create of `payload` on `layer` carries ([`OPERATION_ID`]).
fn create_op(layer: &str, payload: &[u8]) -> CreateOperation {
    CreateOperation::new(OPERATION_ID, operation_request_digest(layer, payload))
}

/// [`OPERATION_ID`]'s record in the fixture's session, as stored.
#[cfg(feature = "test-fault-injection")]
async fn operation_record(fx: &Fx) -> Option<String> {
    fx.tv
        .client()
        .pool
        .get(
            fx.tv
                .key_schema()
                .create_layer_operation(&fx.session_id, OPERATION_ID),
        )
        .await
        .expect("GET of the create operation record must succeed")
}

/// One column of `object_id`'s header.
async fn header_field(fx: &Fx, object_id: &str, field: &str) -> Option<String> {
    fx.tv
        .client()
        .pool
        .hget(fx.tv.key_schema().obj_header(object_id), field)
        .await
        .expect("HGET obj:hdr must succeed")
}

/// The value `layer`'s mapping holds.
async fn mapping(fx: &Fx, layer: &str) -> Option<String> {
    fx.tv
        .client()
        .pool
        .get(fx.tv.key_schema().layer_mapping(&fx.session_id, layer))
        .await
        .expect("GET of the layer mapping must succeed")
}

async fn header_exists(fx: &Fx, object_id: &str) -> bool {
    fx.tv
        .client()
        .pool
        .exists(fx.tv.key_schema().obj_header(object_id))
        .await
        .expect("EXISTS obj:hdr must succeed")
}

/// Every entry of `object_id`'s stream, its fields as text.
async fn stream_entries(fx: &Fx, object_id: u64) -> Vec<HashMap<String, String>> {
    // Read as bytes: a commit entry's relay `ompf` field is binary.
    let entries: Vec<(String, HashMap<String, bytes::Bytes>)> = fx
        .tv
        .client()
        .pool
        .xrange_values(fx.tv.key_schema().layer_stream(object_id), "-", "+", None)
        .await
        .expect("XRANGE of the object stream must succeed");
    entries
        .into_iter()
        .map(|(_, fields)| {
            fields
                .into_iter()
                .map(|(k, v)| (k, String::from_utf8_lossy(&v).into_owned()))
                .collect()
        })
        .collect()
}

/// How many `payload=deleted` entries `object_id`'s stream holds.
async fn deleted_entries(fx: &Fx, object_id: u64) -> usize {
    stream_entries(fx, object_id)
        .await
        .iter()
        .filter(|fields| {
            fields.get(NOTIFICATION_PAYLOAD_FIELD).map(String::as_str)
                == Some(DELETED_STREAM_PAYLOAD)
        })
        .count()
}

/// `PTTL` of `object_id`'s stream: -2 absent, -1 no expiry.
async fn stream_pttl(fx: &Fx, object_id: u64) -> i64 {
    fx.tv
        .client()
        .pool
        .pttl(fx.tv.key_schema().layer_stream(object_id))
        .await
        .expect("PTTL of the object stream must succeed")
}

/// The commit claim of `object_id`.
async fn claim(fx: &Fx, object_id: u64) -> Option<String> {
    fx.tv
        .client()
        .pool
        .get(
            fx.tv
                .key_schema()
                .create_layer_claim(&fx.session_id, object_id),
        )
        .await
        .expect("GET of the commit claim must succeed")
}

async fn claim_pttl(fx: &Fx, object_id: u64) -> i64 {
    fx.tv
        .client()
        .pool
        .pttl(
            fx.tv
                .key_schema()
                .create_layer_claim(&fx.session_id, object_id),
        )
        .await
        .expect("PTTL of the commit claim must succeed")
}

/// Whether `layer` is a member of the session's layers index.
async fn indexed(fx: &Fx, layer: &str) -> bool {
    fx.tv
        .client()
        .pool
        .sismember(
            fx.tv.key_schema().layers_index(&fx.session_id),
            hex_encode(layer),
        )
        .await
        .expect("SISMEMBER of the layers index must succeed")
}

/// One `create_layer_commit.lua` run for `object_id` on `layer`, planned
/// against `expected` (an encoded mapping), with no operation record.
async fn commit_run(fx: &Fx, layer: &str, object_id: u64, expected: &str) -> Vec<String> {
    commit_try(fx, layer, object_id, expected)
        .await
        .expect("the commit script must run")
}

/// [`commit_run`], its error reply returned rather than unwrapped.
async fn commit_try(
    fx: &Fx,
    layer: &str,
    object_id: u64,
    expected: &str,
) -> Result<Vec<String>, fred::error::Error> {
    let ks = fx.tv.key_schema();
    CREATE_LAYER_COMMIT_SCRIPT
        .evalsha_with_reload(
            fx.tv.client().pool.next(),
            vec![
                ks.layer_mapping(&fx.session_id, layer),
                ks.layers_index(&fx.session_id),
                ks.session_deleted_marker(&fx.session_id),
                ks.create_layer_claim(&fx.session_id, object_id),
            ],
            vec![
                object_id.to_string(),
                hex_encode(layer),
                expected.to_owned(),
                COMMITTED_CLAIM_TTL_MS.to_string(),
            ],
        )
        .await
}

/// One `create_layer_veto.lua` run for `object_id` on `layer`.
async fn veto_run(fx: &Fx, layer: &str, object_id: u64) -> Vec<String> {
    veto_try(fx, layer, object_id)
        .await
        .expect("the veto script must run")
}

/// [`veto_run`], its error reply returned rather than unwrapped.
async fn veto_try(fx: &Fx, layer: &str, object_id: u64) -> Result<Vec<String>, fred::error::Error> {
    let ks = fx.tv.key_schema();
    CREATE_LAYER_VETO_SCRIPT
        .evalsha_with_reload(
            fx.tv.client().pool.next(),
            vec![
                ks.layer_mapping(&fx.session_id, layer),
                ks.session_deleted_marker(&fx.session_id),
                ks.create_layer_claim(&fx.session_id, object_id),
            ],
            vec![object_id.to_string(), VETOED_CLAIM_TTL_MS.to_string()],
        )
        .await
}

/// Plant `key` as a hash: a foreign Valkey type on a key a script expects to
/// be a string.
async fn plant_hash(fx: &Fx, key: String) {
    let _: () = fx
        .tv
        .client()
        .pool
        .hset(key, [("foreign", "1")])
        .await
        .expect("HSET of the wrong-typed key must succeed");
}

/// Write the session's deletion marker, as `DeleteSession` does.
async fn mark_session_deleted(fx: &Fx) {
    let _: () = fx
        .tv
        .client()
        .pool
        .set(
            fx.tv.key_schema().session_deleted_marker(&fx.session_id),
            "1",
            None,
            None,
            false,
        )
        .await
        .expect("SET of the session deletion marker must succeed");
}

/// Plant an unmapped, unclaimed object `object_id` of `layer` in the
/// fixture's session: a header naming the session and the layer, with no
/// `created_at`, so it counts as older than any grace (arm (iii)'s case).
async fn plant_orphan_header(fx: &Fx, object_id: &str, layer: &str) {
    let _: () = fx
        .tv
        .client()
        .pool
        .hset(
            fx.tv.key_schema().obj_header(object_id),
            [
                ("seq", "1"),
                ("session_id", fx.session_id.as_str()),
                ("layer_name", layer),
            ],
        )
        .await
        .expect("HSET of the orphan header must succeed");
}

async fn set_mapping(fx: &Fx, layer: &str, value: &str) {
    let _: () = fx
        .tv
        .client()
        .pool
        .set(
            fx.tv.key_schema().layer_mapping(&fx.session_id, layer),
            value,
            None,
            None,
            false,
        )
        .await
        .expect("SET of the layer mapping must succeed");
}

async fn delete_mapping(fx: &Fx, layer: &str) {
    let removed: i64 = fx
        .tv
        .client()
        .pool
        .del(fx.tv.key_schema().layer_mapping(&fx.session_id, layer))
        .await
        .expect("DEL of the layer mapping must succeed");
    assert_eq!(removed, 1, "fixture precondition: the mapping existed");
}

/// One sweep pass over the fixture's keyspace with `store`.
async fn sweep(fx: &Fx, store: LayerStore) {
    sweep_one_pass(
        fx.tv.client(),
        fx.tv.key_schema(),
        fx.ext.clone(),
        Arc::new(store),
    )
    .await;
}

/// Arm `client`'s store out-of-memory signal for the fixture's keyspace,
/// labelled [`OOM_REQUESTER`], as startup arms a replica's: from then on a
/// store write refused `-OOM` that `client` receives requests the GC's
/// out-of-memory collection (b.pa1). Armed before anything can be refused,
/// since the first note takes the signal's once-a-second turn armed or not.
fn arm_oom_signal(fx: &Fx, client: &ValkeyClient) {
    client
        .store_oom_signal()
        .arm(fx.tv.key_schema(), OOM_REQUESTER);
}

/// The label of the GC's out-of-memory collection request
/// (`le:ext:{gc}:oom:request`) once it is in the fixture's keyspace, through
/// the shared poll ([`wait_for_oom_request`], which fails after
/// `OOM_REQUEST_BUDGET`). Called as soon as the refusal can have been
/// received, well inside the request's 10 s lifetime.
async fn oom_request_label(fx: &Fx) -> String {
    wait_for_oom_request(&fx.tv).await.0
}

// ═════════════════════════════════════════════════════════════════════════════
// 8 / 19 — the commit and veto scripts
// ═════════════════════════════════════════════════════════════════════════════

/// Test 8: a run that answers `MAPPING_CHANGED` or `VETOED` writes nothing —
/// no mapping, no layers-index member, no claim. A moved mapping is answered
/// with its current value, encoded as the expected one is.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_commit_script_writes_nothing_on_a_moved_mapping_or_a_veto() {
    const LAYER: &str = "bua9-script-nothing";
    const MOVED: u64 = 910_001;
    const VETOED: u64 = 910_002;
    let fx = fixture("bua9-script-nothing-fixture").await;

    // An absent mapping, planned against a value.
    assert_eq!(
        commit_run(&fx, LAYER, MOVED, &mapping_value("5001")).await,
        ["MAPPING_CHANGED", MAPPING_ABSENT],
        "an absent mapping is answered as `absent`"
    );
    assert_eq!(mapping(&fx, LAYER).await, None, "no mapping written");

    // A stored value other than the planned one.
    set_mapping(&fx, LAYER, "5001").await;
    assert_eq!(
        commit_run(&fx, LAYER, MOVED, &mapping_value("5002")).await,
        ["MAPPING_CHANGED".to_owned(), mapping_value("5001")],
    );
    assert_eq!(mapping(&fx, LAYER).await.as_deref(), Some("5001"));
    assert_eq!(
        claim(&fx, MOVED).await,
        None,
        "MAPPING_CHANGED claims nothing"
    );

    // A vetoed object, planned against the mapping it would replace.
    let _: () = fx
        .tv
        .client()
        .pool
        .set(
            fx.tv
                .key_schema()
                .create_layer_claim(&fx.session_id, VETOED),
            CLAIM_VETOED,
            None,
            None,
            false,
        )
        .await
        .expect("SET of the vetoed claim must succeed");
    assert_eq!(
        commit_run(&fx, LAYER, VETOED, &mapping_value("5001")).await,
        ["VETOED"],
    );
    assert_eq!(
        mapping(&fx, LAYER).await.as_deref(),
        Some("5001"),
        "VETOED flips nothing"
    );
    assert_eq!(claim(&fx, VETOED).await.as_deref(), Some(CLAIM_VETOED));
    assert!(
        !indexed(&fx, LAYER).await,
        "neither reply may list the layer in the index"
    );
}

/// Test 8: the run that flips the mapping writes the mapping, the index member
/// and the object's claim `committed`, with the TTL it was handed, in the one
/// step.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_commit_script_claims_the_object_committed_with_its_flip() {
    const LAYER: &str = "bua9-script-committed";
    const OBJECT: u64 = 910_011;
    let fx = fixture("bua9-script-committed-fixture").await;

    assert_eq!(
        commit_run(&fx, LAYER, OBJECT, MAPPING_ABSENT).await,
        ["COMMITTED"]
    );

    assert_eq!(
        mapping(&fx, LAYER).await,
        Some(OBJECT.to_string()),
        "the flip"
    );
    assert!(indexed(&fx, LAYER).await, "the index member");
    assert_eq!(claim(&fx, OBJECT).await.as_deref(), Some(CLAIM_COMMITTED));
    let ttl = claim_pttl(&fx, OBJECT).await;
    assert!(
        ttl > 0 && ttl <= COMMITTED_CLAIM_TTL_MS as i64,
        "the committed claim carries the TTL the run passed; got PTTL {ttl}"
    );
}

/// Test 8: the veto claims an unmapped, unclaimed object `vetoed` with its
/// 24 h TTL, and answers `MAPPED` (the object is the mapping) or
/// `CLAIMED <value>` (a claim exists, kept as it is) writing nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_veto_script_vetoes_only_an_unclaimed_unmapped_object() {
    const LAYER: &str = "bua9-veto";
    const UNCLAIMED: u64 = 910_021;
    const MAPPED: u64 = 910_022;
    const COMMITTED: u64 = 910_023;
    let fx = fixture("bua9-veto-fixture").await;

    assert_eq!(veto_run(&fx, LAYER, UNCLAIMED).await, ["VETOED"]);
    assert_eq!(claim(&fx, UNCLAIMED).await.as_deref(), Some(CLAIM_VETOED));
    let ttl = claim_pttl(&fx, UNCLAIMED).await;
    assert!(
        ttl > 0 && ttl <= VETOED_CLAIM_TTL_MS,
        "the vetoed claim carries the 24 h TTL; got PTTL {ttl}"
    );

    set_mapping(&fx, LAYER, &MAPPED.to_string()).await;
    assert_eq!(veto_run(&fx, LAYER, MAPPED).await, ["MAPPED"]);
    assert_eq!(claim(&fx, MAPPED).await, None, "MAPPED writes nothing");

    assert_eq!(
        commit_run(&fx, "bua9-veto-other", COMMITTED, MAPPING_ABSENT).await,
        ["COMMITTED"],
        "precondition: the object is claimed committed"
    );
    assert_eq!(
        veto_run(&fx, LAYER, COMMITTED).await,
        ["CLAIMED", CLAIM_COMMITTED]
    );
    assert_eq!(
        claim(&fx, COMMITTED).await.as_deref(),
        Some(CLAIM_COMMITTED),
        "the committed claim is kept"
    );
}

/// Test 29 (A10): a veto that answers `CLAIMED` keeps the existing claim's
/// expiry as it was, for a `committed` and a `vetoed` claim alike. The veto
/// passes the 24 h `vetoed` TTL, far longer than either planted claim's, so a
/// script that touched the claim's expiry before its `CLAIMED` check (the
/// out-of-memory opener placed first) would push the `PTTL` up.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_veto_script_answering_claimed_keeps_the_existing_claims_ttl() {
    const LAYER: &str = "bua9-veto-ttl";
    const COMMITTED: u64 = 910_061;
    const VETOED: u64 = 910_062;
    /// The planted `vetoed` claim's TTL: well under the 24 h the veto passes.
    const PLANTED_VETOED_TTL_MS: u64 = 300_000;
    let fx = fixture("bua9-veto-ttl-fixture").await;
    assert_eq!(
        commit_run(&fx, "bua9-veto-ttl-other", COMMITTED, MAPPING_ABSENT).await,
        ["COMMITTED"],
        "precondition: the object is claimed committed"
    );
    let _: () = fx
        .tv
        .client()
        .pool
        .set(
            fx.tv
                .key_schema()
                .create_layer_claim(&fx.session_id, VETOED),
            CLAIM_VETOED,
            Some(Expiration::PX(PLANTED_VETOED_TTL_MS as i64)),
            None,
            false,
        )
        .await
        .expect("SET of the vetoed claim must succeed");

    for (object, value, planted_ms) in [
        (COMMITTED, CLAIM_COMMITTED, COMMITTED_CLAIM_TTL_MS),
        (VETOED, CLAIM_VETOED, PLANTED_VETOED_TTL_MS),
    ] {
        let before = claim_pttl(&fx, object).await;
        assert!(
            before > 0 && before <= planted_ms as i64,
            "precondition: the {value} claim carries its planted TTL; got PTTL {before}"
        );

        assert_eq!(veto_run(&fx, LAYER, object).await, ["CLAIMED", value]);

        let after = claim_pttl(&fx, object).await;
        assert!(
            after > 0 && after <= before,
            "the veto must not rewrite the {value} claim's expiry: PTTL {before} before, {after} \
             after"
        );
        assert_eq!(claim(&fx, object).await.as_deref(), Some(value));
    }
}

/// The veto's two refusals of a key holding a foreign Valkey type: a
/// wrong-typed layer mapping or commit claim is answered with a
/// `WRONGTYPE_KEY:` error naming the key's role — the tag the sweep's callers
/// classify on — and the veto writes nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_veto_script_refuses_a_wrong_typed_mapping_or_claim_writing_nothing() {
    const MAPPING_LAYER: &str = "bua9-veto-wrongtype-mapping";
    const CLAIM_LAYER: &str = "bua9-veto-wrongtype-claim";
    const OBJECT: u64 = 910_071;
    let fx = fixture("bua9-veto-wrongtype-fixture").await;

    plant_hash(
        &fx,
        fx.tv
            .key_schema()
            .layer_mapping(&fx.session_id, MAPPING_LAYER),
    )
    .await;
    let mapping_refused = veto_try(&fx, MAPPING_LAYER, OBJECT).await;
    let after_mapping = claim(&fx, OBJECT).await;
    plant_hash(
        &fx,
        fx.tv
            .key_schema()
            .create_layer_claim(&fx.session_id, OBJECT),
    )
    .await;
    let claim_refused = veto_try(&fx, CLAIM_LAYER, OBJECT).await;

    for (refused, role) in [
        (&mapping_refused, "(layer mapping)"),
        (&claim_refused, "(create-layer commit claim)"),
    ] {
        let details = refused
            .as_ref()
            .err()
            .map(|e| e.details().to_owned())
            .unwrap_or_default();
        assert!(
            details.starts_with("WRONGTYPE_KEY:") && details.contains(role),
            "a wrong-typed {role} must be refused with the WRONGTYPE_KEY tag; got {refused:?}"
        );
    }
    assert_eq!(
        after_mapping, None,
        "the refusal of a wrong-typed mapping writes no claim"
    );
}

/// The veto on a session whose deletion marker exists answers
/// `SESSION_DELETED` and writes nothing: no commit can map a layer of a
/// deleted session, so no veto is needed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_veto_script_answers_session_deleted_writing_nothing() {
    const LAYER: &str = "bua9-veto-session-deleted";
    const OBJECT: u64 = 910_081;
    let fx = fixture("bua9-veto-session-deleted-fixture").await;
    mark_session_deleted(&fx).await;

    assert_eq!(veto_run(&fx, LAYER, OBJECT).await, ["SESSION_DELETED"]);
    assert_eq!(claim(&fx, OBJECT).await, None, "no claim is written");
}

/// The commit script refuses a wrong-typed commit claim with a
/// `WRONGTYPE_KEY:` error naming the claim — the tag `CreateLayer`
/// classifies as a keyspace refusal — before any write: no mapping, no
/// layers-index member.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_commit_script_refuses_a_wrong_typed_claim_writing_nothing() {
    const LAYER: &str = "bua9-commit-wrongtype-claim";
    const OBJECT: u64 = 910_091;
    let fx = fixture("bua9-commit-wrongtype-claim-fixture").await;
    plant_hash(
        &fx,
        fx.tv
            .key_schema()
            .create_layer_claim(&fx.session_id, OBJECT),
    )
    .await;

    let refused = commit_try(&fx, LAYER, OBJECT, MAPPING_ABSENT).await;

    let details = refused
        .as_ref()
        .err()
        .map(|e| e.details().to_owned())
        .unwrap_or_default();
    assert!(
        details.starts_with("WRONGTYPE_KEY:") && details.contains("(create-layer commit claim)"),
        "a wrong-typed claim must be refused with the WRONGTYPE_KEY tag; got {refused:?}"
    );
    assert_eq!(mapping(&fx, LAYER).await, None, "no mapping written");
    assert!(!indexed(&fx, LAYER).await, "no layers-index member written");
}

/// Arm (iii) on a session whose deletion marker exists: the veto answers
/// `SESSION_DELETED`, and the sweep retires the unmapped object without
/// writing a claim.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn orphan_sweep_retires_an_unmapped_object_of_a_deleted_session_without_a_claim() {
    const ORPHAN: &str = "910101";
    let fx = fixture("bua9-sweep-session-deleted").await;
    plant_orphan_header(&fx, ORPHAN, "bua9-sweep-session-deleted-orphan").await;
    mark_session_deleted(&fx).await;
    fx.tv
        .client()
        .set_orphan_sweep_grace_for_test(Duration::ZERO);

    sweep(&fx, plain_store(&fx)).await;

    assert!(
        !header_exists(&fx, ORPHAN).await,
        "an unmapped object of a deleted session is retired"
    );
    assert_eq!(
        claim(&fx, ORPHAN.parse().expect("a decimal object id")).await,
        None,
        "SESSION_DELETED writes no claim"
    );
}

/// Arm (iii) when the veto is refused `WRONGTYPE_KEY` (the object's claim
/// key holds a foreign type): the sweep leaves the object this pass — it
/// cannot prove no commit will map it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn orphan_sweep_leaves_an_unmapped_object_whose_veto_meets_a_wrong_typed_claim() {
    const ORPHAN: &str = "910111";
    let fx = fixture("bua9-sweep-wrongtype-claim").await;
    plant_orphan_header(&fx, ORPHAN, "bua9-sweep-wrongtype-claim-orphan").await;
    plant_hash(
        &fx,
        fx.tv
            .key_schema()
            .create_layer_claim(&fx.session_id, ORPHAN.parse().expect("a decimal object id")),
    )
    .await;
    fx.tv
        .client()
        .set_orphan_sweep_grace_for_test(Duration::ZERO);

    sweep(&fx, plain_store(&fx)).await;

    assert!(
        header_exists(&fx, ORPHAN).await,
        "an object whose veto was refused is left for a later pass"
    );
}

/// A11: an operation-id `CreateLayer` writes its operation id and request
/// digest on its new object's header — the operation record the orphan
/// sweep's completion of its fence writes is built from them — and a create
/// without an id writes neither.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_header_carries_the_operation_id_and_digest_only_for_a_create_with_one() {
    const LAYER: &str = "bua9-header-op";
    let fx = fixture("bua9-header-op-fixture").await;
    let op = create_op(LAYER, PAYLOAD);
    let digest = op.digest().to_owned();

    let with_id = create_op_within(&plain_store(&fx), &fx, LAYER, Some(op), CALL_BUDGET)
        .await
        .expect("the operation-id create must commit")
        .object_id
        .to_string();

    assert_eq!(
        (
            header_field(&fx, &with_id, HEADER_CREATE_OP_ID).await,
            header_field(&fx, &with_id, HEADER_CREATE_OP_DIGEST).await,
        ),
        (Some(OPERATION_ID.to_owned()), Some(digest)),
        "a create carrying an operation id records it and its digest on its object"
    );
    let without_id = fx.object_id.to_string();
    assert!(
        header_exists(&fx, &without_id).await,
        "precondition: the fixture's object, created with no operation id, exists"
    );
    assert_eq!(
        (
            header_field(&fx, &without_id, HEADER_CREATE_OP_ID).await,
            header_field(&fx, &without_id, HEADER_CREATE_OP_DIGEST).await,
        ),
        (None, None),
        "a create without an operation id writes neither column"
    );
}

/// Test 19: X's commit lands, a create replaces X with B, and X's commit runs
/// again planned against B. The claim, read before the comparison, answers
/// `COMMITTED_EARLIER` naming B, and the run writes nothing: no object is
/// mapped twice. (Comparing first would flip the layer back to X, an object
/// that is being retired.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_commit_script_never_maps_an_object_a_run_already_mapped() {
    const LAYER: &str = "bua9-script-twice";
    const E: &str = "5001";
    const X: u64 = 910_031;
    const B: &str = "5003";
    let fx = fixture("bua9-script-twice-fixture").await;
    set_mapping(&fx, LAYER, E).await;
    assert_eq!(
        commit_run(&fx, LAYER, X, &mapping_value(E)).await,
        ["COMMITTED"],
        "precondition: X's commit lands"
    );
    set_mapping(&fx, LAYER, B).await;

    assert_eq!(
        commit_run(&fx, LAYER, X, &mapping_value(B)).await,
        ["COMMITTED_EARLIER".to_owned(), mapping_value(B)],
    );
    assert_eq!(
        mapping(&fx, LAYER).await.as_deref(),
        Some(B),
        "the mapping still names B"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 21 / 11 / 13 / 26 — seam-free
// ═════════════════════════════════════════════════════════════════════════════

/// Tests 21 and 10 (its storage half): a `DeleteLayer` that stopped after
/// removing the mapping leaves a committed object no mapping names, with no
/// `deleted` on its stream. One pass at grace 0 retires it: the stream gets
/// `deleted` and an expiry, the header goes. The veto finds the object's
/// `committed` claim and keeps it, so a commit run of the object still
/// answers `COMMITTED_EARLIER` rather than `VETOED`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn orphan_sweep_retires_a_committed_unmapped_object_and_keeps_its_committed_claim() {
    let fx = fixture("bua9-veto-keeps-committed").await;
    let x = fx.object_id;
    assert_eq!(
        claim(&fx, x).await.as_deref(),
        Some(CLAIM_COMMITTED),
        "precondition: the create claimed its object committed"
    );
    delete_mapping(&fx, fx.layer).await;
    fx.tv
        .client()
        .set_orphan_sweep_grace_for_test(Duration::ZERO);

    sweep(&fx, plain_store(&fx)).await;

    assert!(
        !header_exists(&fx, &x.to_string()).await,
        "the pass retires the object"
    );
    assert_eq!(
        deleted_entries(&fx, x).await,
        1,
        "the retirement wakes the stream before it deletes anything"
    );
    assert!(stream_pttl(&fx, x).await > 0, "the stream expires");
    assert_eq!(
        claim(&fx, x).await.as_deref(),
        Some(CLAIM_COMMITTED),
        "the veto keeps a committed claim"
    );
    assert_eq!(
        commit_run(&fx, fx.layer, x, MAPPING_ABSENT).await,
        ["COMMITTED_EARLIER", MAPPING_ABSENT],
        "no commit can map the retired object again"
    );
}

/// A proxy rule answering every `EVALSHA` of the script with SHA `sha` with
/// the store's out-of-memory refusal.
fn script_refused_out_of_memory(sha: String) -> ProxyRule {
    Arc::new(move |args: &[Vec<u8>]| {
        (args.first()?.eq_ignore_ascii_case(b"EVALSHA")
            && args.get(1).is_some_and(|a| a.as_slice() == sha.as_bytes()))
        .then(|| {
            ProxyAction::Eval(error_reply_script(
                "OOM command not allowed when used memory > 'maxmemory'.",
            ))
        })
    })
}

/// Test 11: the sweep's wake-up refused out of memory (a store that does not
/// follow the documented rule) still deletes the orphan, since deleting keys
/// is what frees the store, and — like every store write refused `-OOM` —
/// requests the GC's out-of-memory collection (b.pa1). The refusal is staged
/// by a proxy answering the wake script's `EVALSHA` with `-OOM`; the sweep's
/// client is the only client through it, and its signal is armed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn orphan_sweep_retires_an_orphan_whose_wake_up_the_store_refused_out_of_memory() {
    const ORPHAN: &str = "910041";
    let fx = fixture("bua9-sweep-oom").await;
    // An orphan the sweep's arm (iii) takes: it names a session and a layer
    // whose mapping does not name it, and carries no `created_at`, so it
    // counts as older than any grace.
    plant_orphan_header(&fx, ORPHAN, "bua9-sweep-oom-orphan").await;
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        script_refused_out_of_memory(RETIRED_OBJECT_WAKE_SCRIPT.sha1().to_string()),
    )
    .await;
    let swept_through = client_through(&proxy).await;
    arm_oom_signal(&fx, &swept_through);

    sweep_one_pass(
        &swept_through,
        fx.tv.key_schema(),
        fx.ext.clone(),
        Arc::new(LayerStore::new(
            swept_through.clone(),
            fx.tv.key_schema().clone(),
            4096,
        )),
    )
    .await;

    assert!(
        proxy.hits() >= 1,
        "precondition: the sweep's wake-up met the staged out-of-memory refusal"
    );
    assert!(
        !header_exists(&fx, ORPHAN).await,
        "an out-of-memory wake-up refusal must not keep the orphan: deleting it is what frees \
         the store"
    );
    assert_eq!(
        oom_request_label(&fx).await,
        OOM_REQUESTER,
        "the refused wake-up requests the GC's out-of-memory collection, naming the sweep's \
         replica"
    );
}

/// The sweep's veto of an unmapped object refused out of memory: the pass
/// leaves the object for a later pass (it retires an unmapped object only
/// after its veto answers `VETOED`, `CLAIMED` or `SESSION_DELETED`), writes no
/// claim, and requests the GC's out-of-memory collection (b.pa1). Staged as
/// test 11 is, on the veto script's `EVALSHA`. A store that follows the
/// veto's opener rule never refuses it (A10); this is the store that does
/// not.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn orphan_sweep_whose_veto_the_store_refused_out_of_memory_leaves_the_object_and_requests_an_oom_collection(
) {
    const ORPHAN: u64 = 910_042;
    let fx = fixture("bua9-veto-oom").await;
    plant_orphan_header(&fx, &ORPHAN.to_string(), "bua9-veto-oom-orphan").await;
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        script_refused_out_of_memory(CREATE_LAYER_VETO_SCRIPT.sha1().to_string()),
    )
    .await;
    let swept_through = client_through(&proxy).await;
    arm_oom_signal(&fx, &swept_through);

    sweep_one_pass(
        &swept_through,
        fx.tv.key_schema(),
        fx.ext.clone(),
        Arc::new(LayerStore::new(
            swept_through.clone(),
            fx.tv.key_schema().clone(),
            4096,
        )),
    )
    .await;

    assert!(
        proxy.hits() >= 1,
        "precondition: the sweep's veto met the staged out-of-memory refusal"
    );
    assert!(
        header_exists(&fx, &ORPHAN.to_string()).await,
        "an object whose veto was refused is left for a later pass"
    );
    assert_eq!(
        claim(&fx, ORPHAN).await,
        None,
        "the refused veto wrote no claim"
    );
    assert_eq!(
        oom_request_label(&fx).await,
        OOM_REQUESTER,
        "the refused veto requests the GC's out-of-memory collection, naming the sweep's replica"
    );
}

/// Test 13: an overwrite of a layer whose mapping holds a value that names no
/// object (an out-of-band write) answers OK and maps the new object; it fences
/// nothing, so no header is created for the bogus value.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_overwrite_of_a_non_numeric_mapping_answers_ok_and_writes_no_ghost_header() {
    const CORRUPT: &str = "not-an-object-id";
    let fx = fixture("bua9-corrupt-mapping").await;
    set_mapping(&fx, fx.layer, CORRUPT).await;

    let created = create(&plain_store(&fx), &fx, fx.layer)
        .await
        .expect("an overwrite of a non-numeric mapping must answer OK");

    assert_eq!(
        mapping(&fx, fx.layer).await,
        Some(created.object_id.to_string())
    );
    assert!(
        !header_exists(&fx, CORRUPT).await,
        "the bogus value names no object: no fence may create a ghost header for it"
    );
}

/// Test 26 (absent stream): the wake script on a stream that is gone
/// re-creates it with exactly one entry, `payload=deleted` — the field and
/// value every reader matches (`NOTIFICATION_PAYLOAD_FIELD`,
/// `DELETED_STREAM_PAYLOAD`), so a subscriber still blocked on it wakes — and
/// gives it an expiry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retired_object_wake_script_recreates_an_absent_stream_with_one_deleted_entry_and_an_expiry(
) {
    const GONE: u64 = 910_051;
    let fx = fixture("bua9-wake-absent").await;
    assert_eq!(
        stream_pttl(&fx, GONE).await,
        -2,
        "precondition: the stream is absent"
    );

    let _entry_id: String = RETIRED_OBJECT_WAKE_SCRIPT
        .evalsha_with_reload(
            fx.tv.client().pool.next(),
            vec![fx.tv.key_schema().layer_stream(GONE)],
            vec![STREAM_GRACE_MS.to_string()],
        )
        .await
        .expect("the wake script must run");

    assert_eq!(
        stream_entries(&fx, GONE).await,
        [HashMap::from([(
            NOTIFICATION_PAYLOAD_FIELD.to_owned(),
            DELETED_STREAM_PAYLOAD.to_owned()
        )])],
        "one entry, written with the readers' own field and value"
    );
    let ttl = stream_pttl(&fx, GONE).await;
    assert!(
        ttl > 0 && ttl <= STREAM_GRACE_MS,
        "the re-created stream expires within the grace; got PTTL {ttl}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The seam-driven scenarios
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
pub(crate) mod seams {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use fred::clients::Client;
    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use fred::interfaces::SortedSetsInterface;
    use futures::FutureExt;
    use tokio_util::task::TaskTracker;

    use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
    use live_edit::checkpoint::{CheckpointStore, SessionCheckpoint, SessionCheckpointWriter};
    use live_edit::observability::metrics::VALKEY_POOL_TXN;
    use live_edit::storage::layer::{CreateCommitFault, OperationOutcome, OverwriteCleanupProbe};
    use live_edit::storage::test_support::relay::RelayedUpdates;
    use live_edit::storage::test_support::{command_timeout_row, command_timeouts_only_on};
    use live_edit::storage::watched_txn::WatchWindowHook;

    use crate::common::owned_delta::{delta_with_set_field_external, delta_with_set_field_inline};
    use crate::grpc::log_capture::{capture, purge_keys, ThreadCapture};
    use crate::storage::b_qpf_tests::{
        ext_hash, ext_hash_vec, pending_members, pre_store, retirement_member, EXT_SIZE, FIELD_A,
        SECTION_ID,
    };

    use super::*;

    type Faults = Arc<Mutex<VecDeque<CreateCommitFault>>>;
    type WakeFaults = Arc<Mutex<VecDeque<FredError>>>;

    /// The ERROR an overwrite logs when its runs are spent with its fence
    /// still mapped, and the opening words of its `consequence`. Restated:
    /// the line is the operator contract.
    const ERROR_BUDGET_SPENT: &str =
        "CreateLayer overwrite commit failed and its run budget is spent";
    const NOT_LANDED_CONSEQUENCE: &str =
        "the replaced object stays mapped and marked superseded, so every UpdateLayer to this \
         layer fails FAILED_PRECONDITION until the replacement completes";

    /// The `UNAVAILABLE` messages of a vetoed create and of a spent re-plan
    /// budget.
    const VETOED_MESSAGE: &str =
        "CreateLayer could not commit: its new object was retired before the commit ran";
    const REPLAN_BUDGET_MESSAGE: &str =
        "CreateLayer could not commit: the layer was replaced concurrently too many times";

    /// The WARN arm (i) logs when it leaves a mapped object whose successor
    /// was already committed once.
    const WARN_ARM_I_LEAVES_LIVE: &str =
        "a CreateLayer fenced this mapped object to replace it, and the object that was to \
         replace it was already committed once and cannot be mapped again, yet the layer names \
         this object; leaving it, since it is the layer's live object";

    /// The out-of-memory refusal's reply, as the store words it.
    const OOM_REPLY: &str = "OOM command not allowed when used memory > 'maxmemory'.";

    /// How many wake-up attempts one overwrite cleanup makes: the tail's
    /// inline one and its body's three retries.
    const WAKE_ATTEMPTS: usize = 4;

    fn io(script_ran: bool) -> CreateCommitFault {
        CreateCommitFault {
            error: FredError::new(FredErrorKind::IO, "bua9: connection reset"),
            script_ran,
        }
    }

    fn faults(list: impl IntoIterator<Item = CreateCommitFault>) -> Faults {
        Arc::new(Mutex::new(list.into_iter().collect()))
    }

    fn wake_faults(n: usize) -> WakeFaults {
        Arc::new(Mutex::new(
            (0..n)
                .map(|_| FredError::new(FredErrorKind::IO, "bua9: wake-up lost"))
                .collect(),
        ))
    }

    fn remaining<T>(queue: &Arc<Mutex<VecDeque<T>>>) -> usize {
        queue.lock().unwrap().len()
    }

    /// A finisher that gives up before its first run: the state the orphan
    /// sweep's completion of a stranded overwrite is for.
    fn no_finisher(store: LayerStore) -> LayerStore {
        store.with_create_commit_finisher_timing_for_test(
            Duration::from_millis(1),
            Duration::from_millis(1),
            Duration::ZERO,
        )
    }

    /// One field update of `object_id` on the fixture's layer.
    async fn update(
        store: &LayerStore,
        fx: &Fx,
        object_id: u64,
    ) -> Result<(u64, u64), ServiceError> {
        tokio::time::timeout(
            CALL_BUDGET,
            store.update_layer_relayed(
                &fx.session_id,
                fx.layer,
                object_id,
                Some(delta_with_set_field_inline(
                    SECTION_ID,
                    FIELD_A,
                    vec![0x5A],
                    0,
                )),
                "bua9-writer",
                1_000,
            ),
        )
        .await
        .expect("update_layer did not settle within the call budget")
    }

    fn assert_refused_superseded(result: &Result<(u64, u64), ServiceError>, what: &str) {
        assert!(
            matches!(result, Err(ServiceError::FailedPrecondition(_))),
            "{what}: the update must be refused FAILED_PRECONDITION; got {result:?}"
        );
    }

    async fn wait_cleanup(store: &LayerStore) {
        store.wait_for_cleanup(CLEANUP_BUDGET).await;
    }

    /// Release the overwrite cleanup body held on `gate` (the store's
    /// `with_overwrite_cleanup_gate_for_test`), within [`SETTLE_BUDGET`].
    async fn release(gate: &tokio::sync::Barrier) {
        tokio::time::timeout(SETTLE_BUDGET, gate.wait())
            .await
            .expect("the held overwrite cleanup must reach its gate");
    }

    /// Poll `check` until it holds, failing after [`SETTLE_BUDGET`].
    async fn settle<F, Fut>(what: &str, mut check: F)
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = bool>,
    {
        let deadline = tokio::time::Instant::now() + SETTLE_BUDGET;
        while !check().await {
            assert!(
                tokio::time::Instant::now() < deadline,
                "{what} did not happen within {SETTLE_BUDGET:?}"
            );
            tokio::time::sleep(POLL).await;
        }
    }

    /// The object-id counter's value: the last object allocated, which is the
    /// call's own when nothing else allocated since.
    async fn last_allocated(fx: &Fx) -> u64 {
        let counter: String = fx
            .tv
            .client()
            .pool
            .get(fx.tv.key_schema().counter_object_id())
            .await
            .expect("GET of the object-id counter must succeed");
        counter.parse().expect("the counter is a decimal object id")
    }

    /// The successor E's fence names: the object the stuck call materialized.
    async fn fenced_successor(fx: &Fx) -> u64 {
        header_field(fx, &fx.object_id.to_string(), "superseded_by")
            .await
            .expect("precondition: the call fenced the old object, naming its successor")
            .parse()
            .expect("superseded_by is a decimal object id")
    }

    /// An overwrite of the fixture's layer whose two commit runs both fail
    /// before their script runs, and whose finisher gives up at once: the old
    /// object `E` stays mapped, fenced and naming the call's object `X`, which
    /// is materialized and unmapped. Returns `X`.
    async fn stranded_overwrite(fx: &Fx) -> u64 {
        stranded_overwrite_with(fx, None).await
    }

    /// [`stranded_overwrite`], the call carrying `operation`.
    async fn stranded_overwrite_with(fx: &Fx, operation: Option<CreateOperation>) -> u64 {
        let injected = faults([io(false), io(false)]);
        let store = no_finisher(
            plain_store(fx).with_create_layer_commit_fault_injector_for_test(Arc::clone(&injected)),
        );
        let refused = create_op_within(&store, fx, fx.layer, operation, CALL_BUDGET).await;
        assert!(
            matches!(refused, Err(ServiceError::Unavailable(_))),
            "precondition: both runs failed and the call gave up; got {refused:?}"
        );
        assert_eq!(remaining(&injected), 0, "precondition: two runs dispatched");
        wait_cleanup(&store).await;
        assert_eq!(
            mapping(fx, fx.layer).await,
            Some(fx.object_id.to_string()),
            "precondition: the old object is still mapped"
        );
        fenced_successor(fx).await
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 1 / 3 / 25 / 26 — the overwrite tail's wake-up
    // ─────────────────────────────────────────────────────────────────────────

    /// Test 1: the overwrite's inline wake-up fails once. The call answers OK
    /// — the commit point passed — and enqueues the session checkpoint, which
    /// names the new object. The cleanup retries the wake-up: the old stream
    /// gets `deleted`, and only then is the old header deleted.
    ///
    /// Before b.ua9 this answered `UNAVAILABLE` after the flip, appended no
    /// `deleted` and enqueued no checkpoint.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_overwrite_whose_wake_up_fails_once_answers_ok_and_its_cleanup_retries_the_wake_up(
    ) {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only("test-ua9"));
        let checkpoints: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let tracker = TaskTracker::new();
        let writer = Arc::new(SessionCheckpointWriter::new(
            Arc::clone(&checkpoints),
            tv.client().clone(),
            tv.key_schema().clone(),
            tracker.clone(),
            Arc::clone(&metrics),
            "test-ua9".to_owned(),
            true,
        ));
        let fx = fixture_on(tv, "bua9-wake-once").await;
        let wake = wake_faults(1);
        let store = plain_store(&fx)
            .with_checkpoint_writer(writer)
            .with_create_layer_overwrite_xadd_fault_injector_for_test(Arc::clone(&wake));

        let created = create(&store, &fx, fx.layer)
            .await
            .expect("an overwrite whose wake-up failed must answer OK: its commit landed");
        wait_cleanup(&store).await;
        crate::checkpoint::helpers::wait_for_writer_idle(&tracker, 10).await;

        assert_eq!(
            remaining(&wake),
            0,
            "precondition: the inline wake-up failed"
        );
        let blob = checkpoints
            .get_session_checkpoint(&fx.session_id)
            .await
            .expect("get_session_checkpoint must succeed")
            .expect("the overwrite must enqueue a session checkpoint");
        assert_eq!(
            SessionCheckpoint::from_bytes(&blob.bytes)
                .expect("the session checkpoint must decode")
                .layer_mappings
                .get(fx.layer)
                .copied(),
            Some(created.object_id),
            "the session checkpoint names the new object"
        );
        assert_eq!(
            deleted_entries(&fx, fx.object_id).await,
            1,
            "the cleanup's retry appends the wake-up the inline attempt lost"
        );
        assert!(
            !header_exists(&fx, &fx.object_id.to_string()).await,
            "once a wake-up landed the cleanup deletes the old object"
        );
    }

    /// Test 3: every wake-up attempt fails (the inline one and the cleanup's
    /// three retries). The call still answers OK, and the cleanup deletes
    /// nothing: the old header and stream are kept, the stream with no expiry.
    /// One sweep pass at grace 0 then retires the object through its overwrite
    /// tail: `deleted` and an expiry on the stream, the header gone, its one
    /// external value released exactly once, and the object registered dead.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_overwrite_whose_every_wake_up_fails_keeps_the_old_object_until_one_sweep_pass_retires_it(
    ) {
        let fx = fixture("bua9-wake-never").await;
        let e = fx.object_id;
        let h = ext_hash(0xA3);
        pre_store(&fx.ext, &[h]).await;
        tokio::time::timeout(
            CALL_BUDGET,
            plain_store(&fx).update_layer_relayed(
                &fx.session_id,
                fx.layer,
                e,
                Some(delta_with_set_field_external(
                    SECTION_ID,
                    FIELD_A,
                    ext_hash_vec(0xA3),
                    EXT_SIZE,
                    1,
                    0,
                )),
                "bua9-writer",
                1_000,
            ),
        )
        .await
        .expect("the precondition update did not settle within the call budget")
        .expect("precondition: the old object references one external value");
        let wake = wake_faults(WAKE_ATTEMPTS);
        let store = plain_store(&fx)
            .with_create_layer_overwrite_xadd_fault_injector_for_test(Arc::clone(&wake));

        create(&store, &fx, fx.layer)
            .await
            .expect("an overwrite whose every wake-up failed must answer OK");
        wait_cleanup(&store).await;

        assert_eq!(
            remaining(&wake),
            0,
            "precondition: all four attempts were made"
        );
        assert!(
            header_exists(&fx, &e.to_string()).await,
            "no wake-up landed: the cleanup deletes nothing"
        );
        assert_eq!(
            (deleted_entries(&fx, e).await, stream_pttl(&fx, e).await),
            (0, -1),
            "the stream is kept, unwoken and with no expiry"
        );
        assert!(
            pending_members(&fx.tv).await.is_empty(),
            "nothing is released before the wake-up lands"
        );

        fx.tv
            .client()
            .set_orphan_sweep_grace_for_test(Duration::ZERO);
        sweep(&fx, plain_store(&fx).with_dead_object_registration(true)).await;

        assert_eq!(
            deleted_entries(&fx, e).await,
            1,
            "the pass wakes the stream"
        );
        assert!(stream_pttl(&fx, e).await > 0, "the stream expires");
        assert!(
            !header_exists(&fx, &e.to_string()).await,
            "the pass deletes the old header"
        );
        assert_eq!(
            pending_members(&fx.tv).await,
            vec![retirement_member(h, e)],
            "the old object's one reference is released exactly once"
        );
        let dead: Option<f64> = fx
            .tv
            .client()
            .pool
            .zscore(
                fx.tv.key_schema().checkpoint_reconcile_dead(),
                e.to_string(),
            )
            .await
            .expect("ZSCORE of the dead-object set must succeed");
        assert!(
            dead.is_some(),
            "the retired object is registered for checkpoint reclamation"
        );
    }

    /// Test 25: the inline wake-up is refused out of memory (the injected
    /// error reads `OOM …`, as a store that does not follow the documented rule
    /// would answer). The cleanup retires the replaced object anyway — deleting
    /// keys is what frees a full store — with one WARN naming the out-of-memory
    /// refusal, and makes no retry; and the refusal, like every store write
    /// refused `-OOM`, requests the GC's out-of-memory collection (b.pa1).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_overwrite_whose_wake_up_the_store_refused_out_of_memory_still_retires_the_old_object(
    ) {
        let fx = fixture("bua9-wake-oom").await;
        let wake: WakeFaults = Arc::new(Mutex::new(VecDeque::from([
            FredError::new(FredErrorKind::Unknown, OOM_REPLY),
            FredError::new(FredErrorKind::IO, "bua9: a retry nobody should make"),
        ])));
        let store = plain_store(&fx)
            .with_create_layer_overwrite_xadd_fault_injector_for_test(Arc::clone(&wake));
        arm_oom_signal(&fx, fx.tv.client());

        let (created, captured) = capture(create(&store, &fx, fx.layer)).await;
        created.expect("an overwrite whose wake-up was refused out of memory answers OK");
        assert_eq!(
            oom_request_label(&fx).await,
            OOM_REQUESTER,
            "the refused wake-up requests the GC's out-of-memory collection"
        );
        wait_cleanup(&store).await;

        assert_eq!(
            remaining(&wake),
            1,
            "the out-of-memory refusal ends the wake-up attempts: no retry"
        );
        let oom_warnings: Vec<_> = captured
            .events
            .iter()
            .filter(|event| {
                event["level"] == "WARN"
                    && event["fields"]["message"]
                        .as_str()
                        .is_some_and(|m| m.contains("out of memory"))
            })
            .collect();
        assert_eq!(
            oom_warnings.len(),
            1,
            "one WARN names the out-of-memory refusal; full capture:\n{}",
            captured.text
        );
        assert!(
            !header_exists(&fx, &fx.object_id.to_string()).await,
            "the cleanup retires the replaced object anyway"
        );
    }

    /// Test 26 (present stream): after the tail's one inline wake-up — and
    /// before its cleanup body has run any step (held on the cleanup gate) —
    /// the old stream holds exactly one `deleted` entry and already has an
    /// expiry. The wake-up sets it; the cleanup's own expiry step has not run.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_overwrite_inline_wake_up_leaves_one_deleted_entry_and_an_expiry_before_the_cleanup_runs(
    ) {
        let fx = fixture("bua9-wake-shape").await;
        let gate = Arc::new(tokio::sync::Barrier::new(2));
        let store = plain_store(&fx).with_overwrite_cleanup_gate_for_test(Arc::clone(&gate));

        create(&store, &fx, fx.layer)
            .await
            .expect("the overwrite must commit");

        let (deleted, ttl) = (
            deleted_entries(&fx, fx.object_id).await,
            stream_pttl(&fx, fx.object_id).await,
        );
        release(&gate).await;
        wait_cleanup(&store).await;
        assert_eq!(deleted, 1, "one wake-up, one `deleted` entry");
        assert!(
            ttl > 0 && ttl <= STREAM_GRACE_MS,
            "the wake-up itself gives the stream its expiry; got PTTL {ttl}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // b.pa1 — a failed commit run is observed where it is received
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Every run of the commit script goes through one dispatch, which observes
    // a failed run's error there (a command timeout counted on the txn pool, a
    // store write refused `-OOM` noted on the store out-of-memory signal),
    // because its caller may run the commit again, re-plan or give up without
    // ever answering the error.

    /// An overwrite's first commit run refused out of memory (an injected
    /// fault, before the script runs), its second run landing: the call
    /// answers OK, so no mapper ever sees the refusal, and the refusal still
    /// requests the GC's out-of-memory collection.
    ///
    /// This pins only the dispatch's injected-fault branch, which observes the
    /// queued error on its own line, so that a fault injected here is
    /// received as a real one would be. The production path, a run the store
    /// itself refuses `-OOM`, is pinned by `store_write_refusal_test.rs`'s
    /// `create_layer_overwrite_commit_run_refused_out_of_memory_then_landed_requests_an_oom_collection`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_overwrite_whose_refused_out_of_memory_commit_run_is_retried_and_lands_requests_an_oom_collection(
    ) {
        let fx = fixture("bua9-commit-oom").await;
        let injected = faults([CreateCommitFault {
            error: FredError::new(FredErrorKind::Unknown, OOM_REPLY),
            script_ran: false,
        }]);
        let store = plain_store(&fx)
            .with_create_layer_commit_fault_injector_for_test(Arc::clone(&injected));
        arm_oom_signal(&fx, fx.tv.client());

        let created = create(&store, &fx, fx.layer)
            .await
            .expect("an overwrite whose second commit run lands answers OK");

        assert_eq!(
            remaining(&injected),
            0,
            "precondition: the refused run dispatched"
        );
        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(created.object_id.to_string()),
            "precondition: the second run landed"
        );
        assert_eq!(
            oom_request_label(&fx).await,
            OOM_REQUESTER,
            "the refused run requests the GC's out-of-memory collection although the call \
             answered OK"
        );
        wait_cleanup(&store).await;
    }

    /// An overwrite whose two commit runs both time out (injected, before the
    /// script runs) and that gives up counts exactly two command timeouts on
    /// the txn pool: one per failed run, by the dispatch, and none again when
    /// the call maps the error it gives up with. Its finisher gives up before
    /// running, so no further run is dispatched.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_overwrite_whose_two_commit_runs_time_out_counts_each_timeout_once() {
        let fx = fixture("bua9-commit-timeouts").await;
        let metrics = Arc::new(crate::common::metrics_scrape_only("bua9-commit-timeouts"));
        fx.tv
            .client()
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let timed_out = || CreateCommitFault {
            error: FredError::new(FredErrorKind::Timeout, "bua9: commit run timed out"),
            script_ran: false,
        };
        let injected = faults([timed_out(), timed_out()]);
        let store = no_finisher(
            plain_store(&fx)
                .with_create_layer_commit_fault_injector_for_test(Arc::clone(&injected)),
        );

        let refused = create(&store, &fx, fx.layer).await;
        wait_cleanup(&store).await;

        assert!(
            matches!(refused, Err(ServiceError::Unavailable(_))),
            "precondition: both runs failed and the call gave up; got {refused:?}"
        );
        assert_eq!(remaining(&injected), 0, "precondition: two runs dispatched");
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on(VALKEY_POOL_TXN, 2),
            "each failed run's timeout is counted once, where the dispatch received it; the \
             give-up's mapping counts it no second time"
        );
    }

    /// (b.pa1) The orphan sweep's completion of a stranded overwrite refused
    /// out of memory: the pass leaves the overwrite for a later pass (the old
    /// object still mapped, its successor kept) and the refusal requests the
    /// GC's out-of-memory collection. Staged with a proxy answering the commit
    /// script's `EVALSHA` with `-OOM`; the sweep's client is the only client
    /// through it, and its signal is armed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn orphan_sweep_completion_refused_out_of_memory_leaves_the_overwrite_and_requests_an_oom_collection(
    ) {
        let fx = fixture("bua9-complete-oom").await;
        let x = stranded_overwrite(&fx).await;
        let proxy = ScriptReplyProxy::start(
            test_valkey_addr(),
            script_refused_out_of_memory(CREATE_LAYER_COMMIT_SCRIPT.sha1().to_string()),
        )
        .await;
        let swept_through = client_through(&proxy).await;
        swept_through.set_orphan_sweep_grace_for_test(Duration::ZERO);
        arm_oom_signal(&fx, &swept_through);

        sweep_one_pass(
            &swept_through,
            fx.tv.key_schema(),
            fx.ext.clone(),
            Arc::new(LayerStore::new(
                swept_through.clone(),
                fx.tv.key_schema().clone(),
                4096,
            )),
        )
        .await;

        assert!(
            proxy.hits() >= 1,
            "precondition: the sweep's completion met the staged out-of-memory refusal"
        );
        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(fx.object_id.to_string()),
            "the refused completion leaves the old object mapped for a later pass"
        );
        assert!(
            header_exists(&fx, &x.to_string()).await,
            "the successor is kept for a later pass"
        );
        assert_eq!(
            oom_request_label(&fx).await,
            OOM_REQUESTER,
            "the refused completion requests the GC's out-of-memory collection"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 4 / 5 / 9 / 23 — a call that gives up: the finisher and the sweep
    // ─────────────────────────────────────────────────────────────────────────

    /// Test 4: both of the call's runs fail before their script runs and the
    /// check still names the old object: the call answers `UNAVAILABLE` and
    /// logs one ERROR whose consequence says the replacement completes later.
    /// Its finisher's next run lands: the mapping names the new object, the
    /// old stream gets `deleted`, and an `UpdateLayer` to the new object
    /// commits.
    ///
    /// Before b.ua9 nothing ran after the call gave up: the old object stayed
    /// mapped and superseded until a client re-send.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_overwrite_whose_commit_runs_fail_answers_unavailable_and_its_finisher_completes_it(
    ) {
        let fx = fixture("bua9-finisher").await;
        let injected = faults([io(false), io(false)]);
        let store = plain_store(&fx)
            .with_create_layer_commit_fault_injector_for_test(Arc::clone(&injected))
            .with_create_commit_finisher_timing_for_test(
                Duration::from_millis(10),
                Duration::from_millis(10),
                SETTLE_BUDGET,
            );

        let (refused, captured) = capture(create(&store, &fx, fx.layer)).await;

        assert!(
            matches!(refused, Err(ServiceError::Unavailable(_))),
            "both runs failed: the call answers UNAVAILABLE; got {refused:?}"
        );
        assert_eq!(remaining(&injected), 0, "precondition: two runs dispatched");
        let errors: Vec<_> = captured
            .events
            .iter()
            .filter(|event| event["level"] == "ERROR")
            .collect();
        assert_eq!(errors.len(), 1, "full capture:\n{}", captured.text);
        assert_eq!(errors[0]["fields"]["message"], ERROR_BUDGET_SPENT);
        assert!(
            errors[0]["fields"]["consequence"]
                .as_str()
                .is_some_and(|c| c.starts_with(NOT_LANDED_CONSEQUENCE)),
            "{}",
            errors[0]
        );

        // Read off the counter, not the old header's fence: the finisher may
        // already have retired the old object.
        let x = last_allocated(&fx).await;
        settle("the finisher's landing run", || async {
            mapping(&fx, fx.layer).await == Some(x.to_string())
        })
        .await;
        settle("the finisher's overwrite tail", || async {
            deleted_entries(&fx, fx.object_id).await == 1
        })
        .await;
        wait_cleanup(&store).await;
        update(&plain_store(&fx), &fx, x)
            .await
            .expect("an UpdateLayer to the new object must commit");
    }

    /// Test 5: a crash between the fence and the flip, staged as a call whose
    /// runs both fail and whose finisher is out of budget. Before a pass, an
    /// `UpdateLayer` of the layer's object is refused `FAILED_PRECONDITION`;
    /// one sweep pass at grace 0 completes the overwrite, and the update of
    /// the layer's object then commits. This pins the end of the write-refusal
    /// window.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn orphan_sweep_completes_an_overwrite_stranded_after_its_finisher_ran_out_of_budget() {
        let fx = fixture("bua9-stranded").await;
        let x = stranded_overwrite(&fx).await;
        assert_refused_superseded(
            &update(&plain_store(&fx), &fx, fx.object_id).await,
            "before a pass, the layer's mapped object is fenced",
        );

        fx.tv
            .client()
            .set_orphan_sweep_grace_for_test(Duration::ZERO);
        sweep(&fx, plain_store(&fx)).await;

        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(x.to_string()),
            "the pass maps the layer to the fence's successor"
        );
        update(&plain_store(&fx), &fx, x)
            .await
            .expect("after one pass the layer takes writes again");
        assert!(
            !header_exists(&fx, &fx.object_id.to_string()).await,
            "the pass runs the replaced object's tail"
        );
    }

    /// Test 9: the sweep does not veto or retire an unmapped object that the
    /// mapped object's fence names as its successor. Arm (i)'s completion of
    /// the stranded overwrite is made to fail (one injected commit fault on
    /// the sweep's store), so whichever order the pass visits the two objects
    /// in, the successor reaches arm (iii) unmapped: it must keep its header
    /// and get no claim.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn orphan_sweep_neither_vetoes_nor_retires_the_pending_successor_of_a_mapped_fenced_object(
    ) {
        let fx = fixture("bua9-pending-successor").await;
        let x = stranded_overwrite(&fx).await;
        let completion = faults([io(false)]);

        fx.tv
            .client()
            .set_orphan_sweep_grace_for_test(Duration::ZERO);
        sweep(
            &fx,
            plain_store(&fx)
                .with_create_layer_commit_fault_injector_for_test(Arc::clone(&completion)),
        )
        .await;

        assert!(
            header_exists(&fx, &x.to_string()).await,
            "the pending successor must not be retired"
        );
        assert_eq!(
            claim(&fx, x).await,
            None,
            "the pending successor must not be vetoed"
        );
        // Checked after the two above: a sweep that retired the successor
        // first leaves arm (i) no successor to commit, so the fault stays.
        assert_eq!(
            remaining(&completion),
            0,
            "precondition: arm (i) tried, and failed, to complete the overwrite"
        );
        assert_eq!(mapping(&fx, fx.layer).await, Some(fx.object_id.to_string()));
    }

    /// Test 23: arm (i) never retires a mapped object. Staged: the old object
    /// mapped and fenced naming X, X materialized and past the grace, and X's
    /// claim `committed` (X was mapped once, and the layer names the old
    /// object again — no known path does this). The pass's commit run answers
    /// `COMMITTED_EARLIER` naming the old object: the old object stays mapped
    /// and whole, with one WARN.
    ///
    /// Current-thread, so the sweep task's WARN reaches the thread-local
    /// capture; the fixture's keys are purged at the end, since `TestValkey`'s
    /// own cleanup needs a multi-thread runtime.
    #[tokio::test]
    async fn orphan_sweep_never_retires_a_mapped_object_whose_fenced_successor_was_already_committed(
    ) {
        let capture = ThreadCapture::start();
        let fx = fixture("bua9-arm-i-live").await;
        let x = stranded_overwrite(&fx).await;
        let _: () = fx
            .tv
            .client()
            .pool
            .set(
                fx.tv.key_schema().create_layer_claim(&fx.session_id, x),
                CLAIM_COMMITTED,
                None,
                None,
                false,
            )
            .await
            .expect("SET of the committed claim must succeed");

        fx.tv
            .client()
            .set_orphan_sweep_grace_for_test(Duration::ZERO);
        sweep(&fx, plain_store(&fx)).await;
        let captured = capture.snapshot();
        let mapped = mapping(&fx, fx.layer).await;
        let old_seq = header_field(&fx, &fx.object_id.to_string(), "seq").await;
        purge_keys(&fx.tv).await;

        assert_eq!(
            mapped,
            Some(fx.object_id.to_string()),
            "the layer still names the old object"
        );
        assert_eq!(
            old_seq.as_deref(),
            Some("1"),
            "the mapped object's keys are intact"
        );
        assert_eq!(
            captured.with_message(WARN_ARM_I_LEAVES_LIVE).len(),
            1,
            "one WARN; full capture:\n{}",
            captured.text
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 6 / 7 / 12 — concurrent creates, a slow create, the re-plan budget
    // ─────────────────────────────────────────────────────────────────────────

    /// Test 6 (c1): creates A and B read the same mapping E. A is paused at
    /// its pre-commit hook (after its fence); B, through a second store, runs
    /// to completion; then A resumes. A's run finds the mapping moved to B's
    /// object and re-plans: it fences B's object, maps its own, and retires
    /// B's. The final mapping is A's object; B's object is fenced naming A's,
    /// woken, refused to writers, and cleaned.
    ///
    /// Before b.ua9 A's flip was unconditional: B's object was never fenced or
    /// woken, and its writers kept committing to an object no mapping named.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_creates_on_one_layer_fence_wake_and_clean_every_replaced_object() {
        let fx = Arc::new(fixture("bua9-concurrent").await);
        let b_store = Arc::new(plain_store(&fx));
        let b_object: Arc<Mutex<Option<u64>>> = Arc::new(Mutex::new(None));
        let hook: WatchWindowHook = {
            let (fx, b_store, b_object) = (fx.clone(), b_store.clone(), b_object.clone());
            Arc::new(move |_client: &Client| {
                let (fx, b_store, b_object) = (fx.clone(), b_store.clone(), b_object.clone());
                async move {
                    let b = create(&b_store, &fx, fx.layer)
                        .await
                        .expect("the competing create B must commit — it is uncontended");
                    *b_object.lock().unwrap() = Some(b.object_id);
                }
                .boxed()
            })
        };
        let gate = Arc::new(tokio::sync::Barrier::new(2));
        let a_store = plain_store(&fx)
            .with_create_layer_pre_commit_hook_for_test(hook)
            .with_overwrite_cleanup_gate_for_test(Arc::clone(&gate));

        let a = create_within(&a_store, &fx, fx.layer, HOOKED_CALL_BUDGET)
            .await
            .expect("A must commit after re-planning against B's object");
        let b = b_object
            .lock()
            .unwrap()
            .expect("precondition: the hook ran and B committed");
        let b_str = b.to_string();

        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(a.object_id.to_string()),
            "the later commit wins"
        );
        assert_eq!(
            (
                header_field(&fx, &b_str, "superseded").await.as_deref(),
                header_field(&fx, &b_str, "superseded_by").await
            ),
            (Some("1"), Some(a.object_id.to_string())),
            "A's re-plan fences B's object, naming its own"
        );
        assert_eq!(deleted_entries(&fx, b).await, 1, "B's object is woken");
        assert_refused_superseded(
            &update(&plain_store(&fx), &fx, b).await,
            "B's object, replaced",
        );
        release(&gate).await;
        wait_cleanup(&a_store).await;
        wait_cleanup(&b_store).await;
        assert!(!header_exists(&fx, &b_str).await, "B's object is cleaned");
        assert!(
            !header_exists(&fx, &fx.object_id.to_string()).await,
            "E, which B replaced, is cleaned too"
        );
    }

    /// A create of a NEW layer whose pre-commit hook runs one sweep pass, as a
    /// create paused between materializing its object and its commit would
    /// meet one. Returns the call's answer and its object (the counter's
    /// value: nothing else allocates).
    async fn slow_create_meeting_a_sweep(
        fx: &Arc<Fx>,
        layer: &'static str,
    ) -> (Result<CreateCommit, ServiceError>, u64) {
        let hook: WatchWindowHook = {
            let fx = fx.clone();
            Arc::new(move |_client: &Client| {
                let fx = fx.clone();
                async move { sweep(&fx, plain_store(&fx)).await }.boxed()
            })
        };
        let store = plain_store(fx).with_create_layer_pre_commit_hook_for_test(hook);
        let answer = create_within(&store, fx, layer, HOOKED_CALL_BUDGET).await;
        (answer, last_allocated(fx).await)
    }

    /// Test 7 (c2), past the grace: the pass vetoes the slow create's unmapped
    /// object and retires it, and the create's commit then answers `VETOED`:
    /// `UNAVAILABLE`, and the layer is never mapped to the deleted object.
    ///
    /// Before b.ua9 the sweep had no grace and no veto: it retired the object,
    /// and the create mapped the layer to it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn orphan_sweep_vetoes_and_retires_a_slow_creates_object_past_the_grace_and_the_create_answers_unavailable(
    ) {
        const LAYER: &str = "bua9-slow-vetoed";
        let fx = Arc::new(fixture("bua9-slow-vetoed-fixture").await);
        fx.tv
            .client()
            .set_orphan_sweep_grace_for_test(Duration::ZERO);

        let (answer, x) = slow_create_meeting_a_sweep(&fx, LAYER).await;

        match answer {
            Err(ServiceError::Unavailable(message)) => assert_eq!(message, VETOED_MESSAGE),
            other => panic!("a vetoed create must answer UNAVAILABLE; got {other:?}"),
        }
        assert_eq!(
            mapping(&fx, LAYER).await,
            None,
            "no mapping names a deleted object"
        );
        assert!(
            !header_exists(&fx, &x.to_string()).await,
            "the pass retired the object"
        );
        assert_eq!(claim(&fx, x).await.as_deref(), Some(CLAIM_VETOED));
    }

    /// Test 7 (c2), within the grace: the same pass leaves the slow create's
    /// object alone, and the create commits to it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn orphan_sweep_leaves_a_slow_creates_object_alone_within_the_grace_and_the_create_commits(
    ) {
        const LAYER: &str = "bua9-slow-graced";
        let fx = Arc::new(fixture("bua9-slow-graced-fixture").await);

        let (answer, x) = slow_create_meeting_a_sweep(&fx, LAYER).await;

        let created = answer.expect("a create within the grace must commit");
        assert_eq!(created.object_id, x);
        assert_eq!(mapping(&fx, LAYER).await, Some(x.to_string()));
        assert!(header_exists(&fx, &x.to_string()).await);
    }

    /// Test 12: the mapping moves under every run. A proxy answers each commit
    /// `EVALSHA` with a script that moves the mapping to a fresh value and
    /// answers `MAPPING_CHANGED` with it, as a concurrent create landing just
    /// before each run would. The call re-plans (fencing each value it plans
    /// against) until its run budget is spent, then answers `UNAVAILABLE` and
    /// leaves the current mapping unfenced.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_whose_mapping_moves_under_every_run_answers_unavailable_and_fences_no_current_mapping(
    ) {
        /// The run budget of one call, `CREATE_COMMIT_RUNS_PER_CALL`.
        const RUNS_PER_CALL: usize = 4;
        /// The first value the proxy moves the mapping to is this plus 1.
        const MOVED_BASE: u64 = 920_000;
        let fx = fixture("bua9-replan-budget").await;
        let mapping_key = fx.tv.key_schema().layer_mapping(&fx.session_id, fx.layer);
        let counter_key = format!("{}bua9-replan-counter", fx.tv.prefix());
        let move_mapping = format!(
            "local v = tostring({MOVED_BASE} + redis.call('INCR', '{counter_key}')) \
             redis.call('SET', '{mapping_key}', v) \
             return {{'MAPPING_CHANGED', 'value:' .. v}}"
        );
        let sha = CREATE_LAYER_COMMIT_SCRIPT.sha1().to_owned();
        let rule: ProxyRule = Arc::new(move |args: &[Vec<u8>]| {
            (args.first()?.eq_ignore_ascii_case(b"EVALSHA")
                && args.get(1).is_some_and(|a| a.as_slice() == sha.as_bytes()))
            .then(|| ProxyAction::Eval(move_mapping.clone()))
        });
        let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
        let store = LayerStore::new(
            client_through(&proxy).await,
            fx.tv.key_schema().clone(),
            4096,
        );

        let refused = create(&store, &fx, fx.layer).await;

        match refused {
            Err(ServiceError::Unavailable(message)) => {
                assert_eq!(message, REPLAN_BUDGET_MESSAGE)
            }
            other => panic!("a spent re-plan budget answers UNAVAILABLE; got {other:?}"),
        }
        assert_eq!(
            proxy.hits(),
            RUNS_PER_CALL,
            "one commit run per budgeted run"
        );
        let last = (MOVED_BASE + RUNS_PER_CALL as u64).to_string();
        assert_eq!(mapping(&fx, fx.layer).await.as_deref(), Some(last.as_str()));
        assert_eq!(
            header_field(&fx, &last, "superseded").await,
            None,
            "the value the call gave up on is left unfenced"
        );
        let planned = (MOVED_BASE + RUNS_PER_CALL as u64 - 1).to_string();
        assert_eq!(
            header_field(&fx, &planned, "superseded").await.as_deref(),
            Some("1"),
            "control: each value the call did plan against was fenced first"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 14 / 28 — one cleanup body at a time
    // ─────────────────────────────────────────────────────────────────────────

    /// Test 14 (A1): one pass over K = 2 overwrite-fenced orphans runs their
    /// cleanup bodies one at a time. Each body holds inside the probe for a
    /// while before it does anything, so a pass that spawned the bodies rather
    /// than awaiting them would have both in flight at once.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn orphan_sweep_runs_the_cleanup_bodies_of_its_overwrite_fenced_orphans_one_at_a_time() {
        /// Long enough that a spawned second body enters while the first
        /// still holds; it is paid once per body.
        const HOLD: Duration = Duration::from_millis(200);
        const OTHER_LAYER: &str = "bua9-serial-other";
        let fx = fixture("bua9-serial").await;
        let other = create(&plain_store(&fx), &fx, OTHER_LAYER)
            .await
            .expect("the second layer must be created")
            .object_id;
        let wake = wake_faults(2 * WAKE_ATTEMPTS);
        let store = plain_store(&fx)
            .with_create_layer_overwrite_xadd_fault_injector_for_test(Arc::clone(&wake));
        for layer in [fx.layer, OTHER_LAYER] {
            create(&store, &fx, layer)
                .await
                .expect("each overwrite answers OK");
        }
        wait_cleanup(&store).await;
        assert_eq!(remaining(&wake), 0, "precondition: no wake-up landed");
        for object in [fx.object_id, other] {
            assert!(
                header_exists(&fx, &object.to_string()).await,
                "precondition: {object} is an overwrite-fenced orphan"
            );
        }

        let probe = Arc::new(OverwriteCleanupProbe::new(HOLD));
        fx.tv
            .client()
            .set_orphan_sweep_grace_for_test(Duration::ZERO);
        sweep(
            &fx,
            plain_store(&fx).with_overwrite_cleanup_probe_for_test(Arc::clone(&probe)),
        )
        .await;

        assert_eq!(
            probe.entered(),
            2,
            "precondition: both bodies ran in the pass"
        );
        assert_eq!(
            probe.max_in_flight(),
            1,
            "the sweep awaits each cleanup body before the next"
        );
    }

    /// Test 28 (A9): during a healthy overwrite whose spawned cleanup is still
    /// running (held on the cleanup gate, inside the probe), the replaced
    /// object is unmapped and fenced, so a sweep pass reaches it. At the
    /// default grace the pass leaves it — its successor is young — and runs no
    /// second body (max in flight stays 1). At grace 0 the same pass does
    /// (max in flight reaches 2). The grace is lowered only after the call
    /// returned.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn orphan_sweep_runs_no_second_cleanup_body_during_a_healthy_overwrite_within_the_grace()
    {
        let fx = fixture("bua9-no-second-body").await;
        let probe = Arc::new(OverwriteCleanupProbe::new(Duration::ZERO));
        let gate = Arc::new(tokio::sync::Barrier::new(2));
        let store = plain_store(&fx)
            .with_overwrite_cleanup_probe_for_test(Arc::clone(&probe))
            .with_overwrite_cleanup_gate_for_test(Arc::clone(&gate));
        create(&store, &fx, fx.layer)
            .await
            .expect("the overwrite must commit");
        settle("the spawned cleanup entering its probe", || async {
            probe.entered() == 1
        })
        .await;
        let sweeper = || plain_store(&fx).with_overwrite_cleanup_probe_for_test(Arc::clone(&probe));

        sweep(&fx, sweeper()).await;
        let within_grace = (probe.entered(), probe.max_in_flight());
        fx.tv
            .client()
            .set_orphan_sweep_grace_for_test(Duration::ZERO);
        sweep(&fx, sweeper()).await;
        let past_grace = (probe.entered(), probe.max_in_flight());
        release(&gate).await;
        wait_cleanup(&store).await;

        assert_eq!(
            within_grace,
            (1, 1),
            "within its successor's grace the replaced object is left to its own cleanup"
        );
        assert_eq!(
            past_grace,
            (2, 2),
            "control: past the grace the same pass runs a second body beside the held one"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 15–18 / 22 — a landed commit is answered as committed
    // ─────────────────────────────────────────────────────────────────────────

    /// What another writer does to the layer inside a hook: replace it with a
    /// create B (through its own store), or delete it.
    #[derive(Clone, Copy)]
    enum Then {
        Replace,
        Delete,
    }

    /// The other writer's act. Returns B's object for a replacement.
    async fn act(fx: &Fx, store: &LayerStore, then: Then) -> Option<u64> {
        match then {
            Then::Replace => Some(
                create(store, fx, fx.layer)
                    .await
                    .expect("the competing create B must commit")
                    .object_id,
            ),
            Then::Delete => {
                tokio::time::timeout(
                    CALL_BUDGET,
                    store.delete_layer(&fx.session_id, fx.layer, fx.ext.clone()),
                )
                .await
                .expect("DeleteLayer did not settle within the call budget")
                .expect("the competing DeleteLayer must succeed");
                None
            }
        }
    }

    /// Where the hook leaves what the other writer did.
    type Acted = Arc<Mutex<Option<Option<u64>>>>;

    fn acted(slot: &Acted) -> Option<u64> {
        slot.lock()
            .unwrap()
            .expect("precondition: the hook ran the other writer")
    }

    /// Tests 15 and 17, the definitive path: overwrite A of E (no operation
    /// id) is paused at its pre-commit hook, after its fence. In the hook one
    /// sweep pass at grace 0 completes the stranded fence — mapping the layer
    /// to A's object X, standing in for A's own run landing with its reply
    /// lost — and then `then` replaces X with B, or deletes the layer. A then
    /// runs: X's claim answers `COMMITTED_EARLIER`, and A answers OK with X.
    async fn landed_by_the_sweep_then(
        name: &'static str,
        then: Then,
    ) -> (Arc<Fx>, CreateCommit, Option<u64>, Vec<Arc<LayerStore>>) {
        let fx = Arc::new(fixture(name).await);
        fx.tv
            .client()
            .set_orphan_sweep_grace_for_test(Duration::ZERO);
        let other = Arc::new(plain_store(&fx));
        let sweeper = Arc::new(plain_store(&fx));
        let slot: Acted = Arc::new(Mutex::new(None));
        let hook: WatchWindowHook = {
            let (fx, other, sweeper, slot) =
                (fx.clone(), other.clone(), sweeper.clone(), slot.clone());
            Arc::new(move |_client: &Client| {
                let (fx, other, sweeper, slot) =
                    (fx.clone(), other.clone(), sweeper.clone(), slot.clone());
                async move {
                    sweep_one_pass(fx.tv.client(), fx.tv.key_schema(), fx.ext.clone(), sweeper)
                        .await;
                    *slot.lock().unwrap() = Some(act(&fx, &other, then).await);
                }
                .boxed()
            })
        };
        let a_store = Arc::new(plain_store(&fx).with_create_layer_pre_commit_hook_for_test(hook));

        let a = create_within(&a_store, &fx, fx.layer, HOOKED_CALL_BUDGET)
            .await
            .expect("A's commit landed: A must answer OK");
        let b = acted(&slot);
        (fx, a, b, vec![a_store, other])
    }

    /// Tests 16 and 18, the transient path: overwrite A of E's first run lands
    /// and its reply is lost (`script_ran: true`). In the post-failed-run hook,
    /// before A's check, `then` replaces A's object X with B or deletes the
    /// layer. A's check reads the claim `committed` and answers OK with X.
    async fn landed_with_a_lost_reply_then(
        name: &'static str,
        then: Then,
    ) -> (Arc<Fx>, CreateCommit, Option<u64>, Vec<Arc<LayerStore>>) {
        let fx = Arc::new(fixture(name).await);
        let other = Arc::new(plain_store(&fx));
        let slot: Acted = Arc::new(Mutex::new(None));
        let hook: WatchWindowHook = {
            let (fx, other, slot) = (fx.clone(), other.clone(), slot.clone());
            Arc::new(move |_client: &Client| {
                let (fx, other, slot) = (fx.clone(), other.clone(), slot.clone());
                async move {
                    *slot.lock().unwrap() = Some(act(&fx, &other, then).await);
                }
                .boxed()
            })
        };
        let injected = faults([io(true)]);
        let a_store = Arc::new(
            plain_store(&fx)
                .with_create_layer_commit_fault_injector_for_test(Arc::clone(&injected))
                .with_create_layer_post_failed_run_hook_for_test(hook),
        );

        let a = create_within(&a_store, &fx, fx.layer, HOOKED_CALL_BUDGET)
            .await
            .expect("A's first run landed: A must answer OK");
        assert_eq!(
            remaining(&injected),
            0,
            "precondition: A's first run lost its reply"
        );
        let b = acted(&slot);
        (fx, a, b, vec![a_store, other])
    }

    /// The assertions of 15 and 16: B, which replaced A's landed object, stays
    /// the layer's live object.
    async fn assert_replacement_live(
        fx: &Fx,
        a: CreateCommit,
        b: Option<u64>,
        stores: Vec<Arc<LayerStore>>,
    ) {
        let b = b.expect("precondition: B replaced the layer");
        assert_ne!(
            a.object_id, b,
            "precondition: A and B are different objects"
        );
        for store in &stores {
            wait_cleanup(store).await;
        }
        assert_eq!(
            mapping(fx, fx.layer).await,
            Some(b.to_string()),
            "the mapping still names B"
        );
        assert_eq!(
            header_field(fx, &b.to_string(), "superseded").await,
            None,
            "B is not fenced: A must not re-plan against it"
        );
        assert!(
            header_exists(fx, &b.to_string()).await,
            "B's keys are intact"
        );
        update(&plain_store(fx), fx, b)
            .await
            .expect("an UpdateLayer to B must commit");
    }

    /// The assertions of 17 and 18: the layer a `DeleteLayer` removed stays
    /// deleted.
    async fn assert_layer_stays_deleted(fx: &Fx, a: CreateCommit, stores: Vec<Arc<LayerStore>>) {
        for store in &stores {
            wait_cleanup(store).await;
        }
        assert_eq!(
            mapping(fx, fx.layer).await,
            None,
            "the mapping stays absent"
        );
        assert!(
            !indexed(fx, fx.layer).await,
            "the layers-index entry stays absent"
        );
        assert!(
            !header_exists(fx, &a.object_id.to_string()).await,
            "the deleted layer's object X has no header"
        );
    }

    /// Test 15: landed (by the sweep's completion), then replaced by B. A
    /// answers OK with X, and B stays the layer: unfenced, writable, whole.
    ///
    /// Before the commit claim, A's run found the mapping moved, re-planned
    /// against B, fenced it, mapped X back and retired B.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_layer_whose_commit_the_sweep_completed_and_another_create_replaced_answers_ok_and_leaves_the_replacement_live(
    ) {
        let (fx, a, b, stores) =
            landed_by_the_sweep_then("bua9-landed-replaced", Then::Replace).await;
        assert_replacement_live(&fx, a, b, stores).await;
    }

    /// Test 16: the same through the transient path: A's first run landed with
    /// its reply lost, B replaced X before A's check. A answers OK with X, and
    /// B stays the layer.
    ///
    /// Before the commit claim, the check read the mapping B and re-planned.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_layer_whose_lost_reply_run_landed_before_another_create_replaced_it_answers_ok_and_leaves_the_replacement_live(
    ) {
        let (fx, a, b, stores) =
            landed_with_a_lost_reply_then("bua9-lost-reply-replaced", Then::Replace).await;
        assert_replacement_live(&fx, a, b, stores).await;
    }

    /// Test 17 (Observation 3): landed (by the sweep), then the layer deleted.
    /// A answers OK with X, and the layer stays deleted: no mapping, no index
    /// entry, no header for X.
    ///
    /// Before the commit claim, A re-planned as a new layer and mapped the
    /// layer to the deleted X.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_layer_whose_commit_the_sweep_completed_before_a_delete_layer_answers_ok_and_the_layer_stays_deleted(
    ) {
        let (fx, a, _, stores) =
            landed_by_the_sweep_then("bua9-landed-deleted", Then::Delete).await;
        assert_layer_stays_deleted(&fx, a, stores).await;
    }

    /// Test 18: the same through the transient path. A answers OK, and the
    /// mapping stays absent.
    ///
    /// Before the commit claim, the check read an absent mapping and answered
    /// `UNAVAILABLE`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_layer_whose_lost_reply_run_landed_before_a_delete_layer_answers_ok_and_the_layer_stays_deleted(
    ) {
        let (fx, a, _, stores) =
            landed_with_a_lost_reply_then("bua9-lost-reply-deleted", Then::Delete).await;
        assert_layer_stays_deleted(&fx, a, stores).await;
    }

    /// Test 22: the finisher after "landed, then replaced". A's two runs fail
    /// before their script runs; between them (the post-failed-run hook) the
    /// test lands X with one commit run built from the session, the layer, E
    /// and X (X read off E's `superseded_by`), and B replaces X. A proxy
    /// answers both of A's checks with the mapping E and no claim — a check
    /// that read the store before the landing — so A gives up and spawns its
    /// finisher. The finisher's run answers `COMMITTED_EARLIER` naming B, and
    /// it runs E's tail: E's stream gets `deleted` and E's header goes, with
    /// no sweep pass.
    ///
    /// Before the finisher recognized `COMMITTED_EARLIER`, it stopped on the
    /// moved mapping and E stayed fenced, unmapped and unwoken until a sweep.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_layer_commit_finisher_runs_the_overwrite_tail_when_its_object_landed_and_was_replaced(
    ) {
        let fx = Arc::new(fixture("bua9-finisher-landed-replaced").await);
        let e = fx.object_id;
        let stale_check = format!("return {{'{e}', false}}");
        let rule: ProxyRule = Arc::new(move |args: &[Vec<u8>]| {
            (args.first()?.eq_ignore_ascii_case(b"MGET")
                && args
                    .iter()
                    .any(|a| a.windows(13).any(|w| w == b":createclaim:")))
            .then(|| ProxyAction::Eval(stale_check.clone()))
        });
        let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
        let other = Arc::new(plain_store(&fx));
        let replaced_by: Acted = Arc::new(Mutex::new(None));
        let hook: WatchWindowHook = {
            let (fx, other, replaced_by) = (fx.clone(), other.clone(), replaced_by.clone());
            Arc::new(move |_client: &Client| {
                let (fx, other, replaced_by) = (fx.clone(), other.clone(), replaced_by.clone());
                async move {
                    let x = fenced_successor(&fx).await;
                    assert_eq!(
                        commit_run(&fx, fx.layer, x, &mapping_value(&fx.object_id.to_string()))
                            .await,
                        ["COMMITTED"],
                        "precondition: the test's run lands X"
                    );
                    *replaced_by.lock().unwrap() = Some(act(&fx, &other, Then::Replace).await);
                }
                .boxed()
            })
        };
        let injected = faults([io(false), io(false)]);
        let a_store = LayerStore::new(
            client_through(&proxy).await,
            fx.tv.key_schema().clone(),
            4096,
        )
        .with_create_layer_commit_fault_injector_for_test(Arc::clone(&injected))
        .with_create_layer_post_failed_run_hook_for_test(hook)
        .with_create_commit_finisher_timing_for_test(
            Duration::from_millis(10),
            Duration::from_millis(10),
            SETTLE_BUDGET,
        );

        let refused = create_within(&a_store, &fx, fx.layer, HOOKED_CALL_BUDGET).await;
        assert!(
            matches!(refused, Err(ServiceError::Unavailable(_))),
            "precondition: A's checks read E, so A gave up; got {refused:?}"
        );
        assert_eq!(
            (remaining(&injected), proxy.hits()),
            (0, 2),
            "precondition: two failed runs, each checked through the stale read"
        );
        let b = acted(&replaced_by).expect("precondition: B replaced X");

        settle("the finisher's run of E's tail", || async {
            deleted_entries(&fx, e).await == 1
        })
        .await;
        wait_cleanup(&a_store).await;
        wait_cleanup(&other).await;
        assert!(
            !header_exists(&fx, &e.to_string()).await,
            "the finisher's tail retires E"
        );
        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(b.to_string()),
            "the finisher never re-plans: B stays the layer"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 30 / 31 — the operation record, after the sweep and on the check
    // ─────────────────────────────────────────────────────────────────────────

    /// The ERROR arm (i) logs when the fence's successor's operation is
    /// recorded for another object. Restated: the line is the operator
    /// contract.
    const ERROR_ARM_I_RECORDED_FOR_ANOTHER: &str =
        "a CreateLayer fenced this mapped object to replace it, but that CreateLayer's \
         operation_id is already recorded for another object, so the replacement cannot be \
         completed: every UpdateLayer to this layer fails FAILED_PRECONDITION until the operation \
         record expires (up to 24 hours after it was written), after which the orphan sweep \
         completes the replacement, or until a CreateLayer replaces the layer";

    /// The ERROR a call logs when its commit is found landed while the
    /// mapping still names the object it replaced.
    const ERROR_LANDED_BUT_OLD_STILL_MAPPED: &str =
        "CreateLayer found its commit landed while the layer mapping still names the object it \
         replaced; that object is not retired";

    /// An operation-id overwrite E → X strands (both runs fail, no finisher)
    /// and one sweep pass at grace 0 completes it (arm (i)). Returns the
    /// fixture, X, and the operation record as the pass left it.
    async fn op_overwrite_completed_by_the_sweep(name: &'static str) -> (Fx, u64, Option<String>) {
        let fx = fixture(name).await;
        let x = stranded_overwrite_with(&fx, Some(create_op(fx.layer, PAYLOAD))).await;
        assert_eq!(
            operation_record(&fx).await,
            None,
            "precondition: no run of the call landed, so nothing is recorded"
        );

        fx.tv
            .client()
            .set_orphan_sweep_grace_for_test(Duration::ZERO);
        sweep(&fx, plain_store(&fx)).await;

        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(x.to_string()),
            "precondition: the pass completed the overwrite"
        );
        let record = operation_record(&fx).await;
        (fx, x, record)
    }

    /// Test 30 (a) (A11): after the sweep completed an operation-id overwrite
    /// E → X and another participant edited X, a re-send of the same request
    /// with the same operation id is answered `ALREADY_COMMITTED` with X's
    /// object id, and changes nothing: X stays the layer, the edit kept. The
    /// record the completion wrote is byte-identical to the one the call's own
    /// landing against E would have written: `<X>:<E>:<digest>`.
    ///
    /// Before A11 the completion wrote no record, so the re-send replaced X
    /// with a fresh copy of its upload and the edit was lost.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn orphan_sweep_completed_operation_id_overwrite_answers_a_same_id_resend_already_committed_and_keeps_later_edits(
    ) {
        let (fx, x, record) = op_overwrite_completed_by_the_sweep("bua9-sweep-record-resend").await;
        update(&plain_store(&fx), &fx, x)
            .await
            .expect("another participant's edit of X must commit");

        let resent = create_op_within(
            &plain_store(&fx),
            &fx,
            fx.layer,
            Some(create_op(fx.layer, PAYLOAD)),
            CALL_BUDGET,
        )
        .await;

        assert!(
            matches!(
                resent,
                Ok(CreateCommit {
                    object_id,
                    outcome: OperationOutcome::AlreadyCommitted,
                }) if object_id == x
            ),
            "the re-send must be answered ALREADY_COMMITTED with X ({x}); got {resent:?}"
        );
        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(x.to_string()),
            "X is still the layer"
        );
        assert_eq!(
            header_field(&fx, &x.to_string(), "seq").await.as_deref(),
            Some("2"),
            "the other participant's edit survives"
        );
        assert_eq!(
            record,
            Some(format!(
                "{x}:{}:{}",
                fx.object_id,
                create_op(fx.layer, PAYLOAD).digest()
            )),
            "the completion writes the record the call's own landing against E would have"
        );
    }

    /// Test 30 (b) (A11): after the sweep completed an operation-id overwrite,
    /// a DIFFERENT request carrying the same operation id is refused
    /// `ALREADY_EXISTS`, and X stays the layer.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn orphan_sweep_completed_operation_id_overwrite_refuses_a_different_request_with_the_same_id(
    ) {
        let (fx, x, _) = op_overwrite_completed_by_the_sweep("bua9-sweep-record-conflict").await;

        let different = create_op_within(
            &plain_store(&fx),
            &fx,
            fx.layer,
            Some(create_op(fx.layer, OTHER_PAYLOAD)),
            CALL_BUDGET,
        )
        .await;

        assert!(
            matches!(different, Err(ServiceError::AlreadyExists(_))),
            "a different request under the same operation id must be refused ALREADY_EXISTS; got \
             {different:?}"
        );
        assert_eq!(mapping(&fx, fx.layer).await, Some(x.to_string()));
    }

    /// Test 30 (c) (A11, the re-plan variant): an operation-id overwrite of E
    /// re-plans onto E2 (another create replaced E between its failed first
    /// run and its check), fences E2 naming its object X, and strands there.
    /// One sweep pass completes E2 → X and writes `<X>:<E2>:<digest>` — the
    /// record the call's own landing against E2 would have written, not one
    /// naming the E it first planned against.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn orphan_sweep_completion_of_a_re_planned_operation_id_overwrite_records_the_re_planned_object(
    ) {
        let fx = Arc::new(fixture("bua9-sweep-record-replan").await);
        let op = create_op(fx.layer, PAYLOAD);
        let digest = op.digest().to_owned();
        let other = Arc::new(plain_store(&fx));
        let e2_slot: Acted = Arc::new(Mutex::new(None));
        let hook: WatchWindowHook = {
            let (fx, other, e2_slot) = (fx.clone(), other.clone(), e2_slot.clone());
            Arc::new(move |_client: &Client| {
                let (fx, other, e2_slot) = (fx.clone(), other.clone(), e2_slot.clone());
                async move {
                    *e2_slot.lock().unwrap() = Some(act(&fx, &other, Then::Replace).await);
                }
                .boxed()
            })
        };
        // The first run fails (the hook then moves the layer to E2); the two
        // runs against E2 fail too, and the finisher gives up at once.
        let injected = faults([io(false), io(false), io(false)]);
        let a_store = no_finisher(
            plain_store(&fx)
                .with_create_layer_commit_fault_injector_for_test(Arc::clone(&injected))
                .with_create_layer_post_failed_run_hook_for_test(hook),
        );

        let refused = create_op_within(&a_store, &fx, fx.layer, Some(op), HOOKED_CALL_BUDGET).await;
        assert!(
            matches!(refused, Err(ServiceError::Unavailable(_))),
            "precondition: every run failed and the call gave up; got {refused:?}"
        );
        assert_eq!(
            remaining(&injected),
            0,
            "precondition: three runs dispatched"
        );
        let e2 = acted(&e2_slot).expect("precondition: the other create replaced E with E2");
        wait_cleanup(&a_store).await;
        wait_cleanup(&other).await;
        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(e2.to_string()),
            "precondition: E2 is still mapped"
        );
        let x: u64 = header_field(&fx, &e2.to_string(), "superseded_by")
            .await
            .expect("precondition: the call re-planned and fenced E2")
            .parse()
            .expect("superseded_by is a decimal object id");

        fx.tv
            .client()
            .set_orphan_sweep_grace_for_test(Duration::ZERO);
        sweep(&fx, plain_store(&fx)).await;

        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(x.to_string()),
            "precondition: the pass completed the overwrite"
        );
        assert_eq!(
            operation_record(&fx).await,
            Some(format!("{x}:{e2}:{digest}")),
            "the completion records the object the call re-planned against: <X>:<E2>:<digest>"
        );
    }

    /// Test 31: a duplicate B (the same request, the same operation id) runs
    /// while the original A is in flight. B's first run fails before its
    /// script runs; before B's check, A lands its own object XA over E. B's
    /// check reads A's operation record in the same `MGET` as the mapping and
    /// decides from it first: B answers `ALREADY_COMMITTED` with XA and fences
    /// nothing. XA stays the layer, unfenced, and takes writes.
    ///
    /// Without the record in the check, B saw the mapping moved to XA,
    /// re-planned, and fenced XA naming its own object; its next run met A's
    /// record and stopped, leaving the layer's live object fenced: every
    /// `UpdateLayer` refused until the record expired, 24 h later.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_layer_duplicate_whose_check_finds_the_original_recorded_answers_from_the_record_and_fences_nothing(
    ) {
        let fx = Arc::new(fixture("bua9-dup-resend").await);
        let original = Arc::new(plain_store(&fx));
        let a_slot: Arc<Mutex<Option<CreateCommit>>> = Arc::new(Mutex::new(None));
        let hook: WatchWindowHook = {
            let (fx, original, a_slot) = (fx.clone(), original.clone(), a_slot.clone());
            Arc::new(move |_client: &Client| {
                let (fx, original, a_slot) = (fx.clone(), original.clone(), a_slot.clone());
                async move {
                    let a = create_op_within(
                        &original,
                        &fx,
                        fx.layer,
                        Some(create_op(fx.layer, PAYLOAD)),
                        CALL_BUDGET,
                    )
                    .await
                    .expect("the original A must commit — it is uncontended");
                    *a_slot.lock().unwrap() = Some(a);
                }
                .boxed()
            })
        };
        let injected = faults([io(false)]);
        let b_store = plain_store(&fx)
            .with_create_layer_commit_fault_injector_for_test(Arc::clone(&injected))
            .with_create_layer_post_failed_run_hook_for_test(hook);

        let b = create_op_within(
            &b_store,
            &fx,
            fx.layer,
            Some(create_op(fx.layer, PAYLOAD)),
            HOOKED_CALL_BUDGET,
        )
        .await;
        let a = a_slot
            .lock()
            .unwrap()
            .expect("precondition: the hook ran and A committed");
        wait_cleanup(&b_store).await;
        wait_cleanup(&original).await;
        let xa = a.object_id.to_string();

        assert_eq!(
            (a.outcome, remaining(&injected)),
            (OperationOutcome::Applied, 0),
            "precondition: A landed its own commit, and B's first run failed"
        );
        assert!(
            matches!(
                b,
                Ok(CreateCommit {
                    object_id,
                    outcome: OperationOutcome::AlreadyCommitted,
                }) if object_id == a.object_id
            ),
            "B must be answered from A's record: ALREADY_COMMITTED with XA ({xa}); got {b:?}"
        );
        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(xa.clone()),
            "XA is the layer"
        );
        assert_eq!(
            (
                header_field(&fx, &xa, "superseded").await,
                header_field(&fx, &xa, "superseded_by").await,
            ),
            (None, None),
            "B must not fence A's live object"
        );
        update(&plain_store(&fx), &fx, a.object_id)
            .await
            .expect("an UpdateLayer to the layer's object must commit");
    }

    /// Test 31's ERROR (path (b)): the cross-layer misuse. A `CreateLayer`
    /// carrying an operation id fences the mapped E naming its object X and
    /// strands; then a create of ANOTHER layer reusing that operation id
    /// commits, recording it for its own object. One sweep pass cannot
    /// complete E → X — the record is checked first — and logs one ERROR
    /// naming the layer and the operation id; E stays mapped and whole.
    ///
    /// Current-thread, so the sweep task's ERROR reaches the thread-local
    /// capture; the fixture's keys are purged at the end, since `TestValkey`'s
    /// own cleanup needs a multi-thread runtime.
    #[tokio::test]
    async fn orphan_sweep_logs_one_error_and_leaves_a_fence_whose_operation_id_another_layer_recorded(
    ) {
        const OTHER_LAYER: &str = "bua9-cross-layer-other";
        let capture = ThreadCapture::start();
        let fx = fixture("bua9-cross-layer").await;
        let x = stranded_overwrite_with(&fx, Some(create_op(fx.layer, PAYLOAD))).await;
        let other = create_op_within(
            &plain_store(&fx),
            &fx,
            OTHER_LAYER,
            Some(create_op(OTHER_LAYER, PAYLOAD)),
            CALL_BUDGET,
        )
        .await
        .expect("the other layer's create, reusing the operation id, must commit");
        let record = operation_record(&fx).await;

        fx.tv
            .client()
            .set_orphan_sweep_grace_for_test(Duration::ZERO);
        sweep(&fx, plain_store(&fx)).await;
        let captured = capture.snapshot();
        let mapped = mapping(&fx, fx.layer).await;
        let e_seq = header_field(&fx, &fx.object_id.to_string(), "seq").await;
        let x_kept = header_exists(&fx, &x.to_string()).await;
        purge_keys(&fx.tv).await;

        assert!(
            record
                .as_deref()
                .is_some_and(|r| r.starts_with(&format!("{}:", other.object_id))),
            "precondition: the operation is recorded for the other layer's object; got {record:?}"
        );
        assert_eq!(
            mapped,
            Some(fx.object_id.to_string()),
            "E stays the layer's mapping"
        );
        assert_eq!(e_seq.as_deref(), Some("1"), "E's keys are intact");
        assert!(x_kept, "X, the fence's pending successor, is not retired");
        let errors = captured.with_message(ERROR_ARM_I_RECORDED_FOR_ANOTHER);
        assert_eq!(
            errors.len(),
            1,
            "one ERROR per pass; full capture:\n{}",
            captured.text
        );
        assert_eq!(errors[0]["level"], "ERROR");
        assert_eq!(errors[0]["fields"]["layer_name"], fx.layer);
        assert_eq!(errors[0]["fields"]["operation_id"], OPERATION_ID);
    }

    /// The check after a failed run finds the call's OWN operation record
    /// while the mapping names the object the call replaced — a state no
    /// known path produces, staged out of band before the check: the call's
    /// first run lands (reply lost), then the mapping is set back to E and
    /// the call's `committed` claim removed, so only the record shows the
    /// landing. The call answers OK, and since the check read the mapping in
    /// the same `MGET`, its tail guard sees E still mapped: one ERROR, and E
    /// is neither woken nor retired.
    ///
    /// This pins the check's own-record verdict carrying the mapping it read;
    /// the decision table's own-record row alone carries none (a record does
    /// not show the mapping), and the tail would retire the layer's live
    /// object.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_check_finding_its_own_record_while_the_replaced_object_is_still_mapped_retires_nothing(
    ) {
        let fx = Arc::new(fixture("bua9-own-record-old-mapped").await);
        let hook: WatchWindowHook = {
            let fx = fx.clone();
            Arc::new(move |_client: &Client| {
                let fx = fx.clone();
                async move {
                    let x = fenced_successor(&fx).await;
                    set_mapping(&fx, fx.layer, &fx.object_id.to_string()).await;
                    let removed: i64 = fx
                        .tv
                        .client()
                        .pool
                        .del(fx.tv.key_schema().create_layer_claim(&fx.session_id, x))
                        .await
                        .expect("DEL of the commit claim must succeed");
                    assert_eq!(
                        removed, 1,
                        "precondition: the first run claimed X committed"
                    );
                }
                .boxed()
            })
        };
        let injected = faults([io(true)]);
        let store = plain_store(&fx)
            .with_create_layer_commit_fault_injector_for_test(Arc::clone(&injected))
            .with_create_layer_post_failed_run_hook_for_test(hook);

        let (created, captured) = capture(create_op_within(
            &store,
            &fx,
            fx.layer,
            Some(create_op(fx.layer, PAYLOAD)),
            CALL_BUDGET,
        ))
        .await;
        wait_cleanup(&store).await;

        let created = created.expect("the call's first run landed: it answers OK");
        assert_eq!(
            (created.outcome, remaining(&injected)),
            (OperationOutcome::Applied, 0),
            "precondition: the first run landed with its reply lost"
        );
        assert_eq!(
            captured
                .with_message(ERROR_LANDED_BUT_OLD_STILL_MAPPED)
                .len(),
            1,
            "full capture:\n{}",
            captured.text
        );
        assert!(
            header_exists(&fx, &fx.object_id.to_string()).await,
            "the still-mapped object is not retired"
        );
        assert_eq!(
            deleted_entries(&fx, fx.object_id).await,
            0,
            "the still-mapped object's stream is not woken"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Arm (i)'s other branches; the re-plan as a new layer
    // ─────────────────────────────────────────────────────────────────────────

    /// The ERROR arm (i) logs when the fence's successor is gone.
    const ERROR_ARM_I_SUCCESSOR_GONE: &str =
        "a CreateLayer fenced this mapped object to replace it, and the object that was to \
         replace it is gone, so the replacement cannot be completed: every UpdateLayer to this \
         layer fails FAILED_PRECONDITION until a CreateLayer of it succeeds";

    /// The INFO a call logs when it re-plans as a new layer.
    const INFO_REPLAN_AS_NEW_LAYER: &str = "CreateLayer re-plans: the layer was deleted since \
                                            this call read its mapping; committing it as a new \
                                            layer";

    /// Arm (i) at the default grace: a stranded overwrite whose successor X
    /// is younger than the grace is left to its call or finisher this pass —
    /// E stays mapped, X gets no claim, and both headers are intact.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn orphan_sweep_leaves_a_stranded_overwrite_alone_while_its_successor_is_within_the_grace(
    ) {
        let fx = fixture("bua9-arm-i-graced").await;
        let x = stranded_overwrite(&fx).await;

        sweep(&fx, plain_store(&fx)).await;

        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(fx.object_id.to_string()),
            "the pass must not complete the overwrite within the successor's grace"
        );
        assert_eq!(claim(&fx, x).await, None, "X gets no claim");
        assert!(
            header_exists(&fx, &fx.object_id.to_string()).await
                && header_exists(&fx, &x.to_string()).await,
            "both headers are intact"
        );
    }

    /// Arm (i) meets `COMMITTED_EARLIER` with the layer no longer naming E:
    /// the successor was mapped once and the layer has moved on since, so
    /// only E's tail is missing, and the pass runs it, awaited — when the
    /// pass returns, E's stream has `deleted` and E's header is gone. Staged
    /// as test 11 is: a proxy on the sweep's own client answers the commit
    /// `EVALSHA` with `COMMITTED_EARLIER` naming a third object.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn orphan_sweep_runs_the_tail_of_a_stranded_overwrite_whose_successor_committed_earlier_and_moved_on(
    ) {
        /// The object the staged reply says the layer names now.
        const MOVED_ON_TO: &str = "987654329";
        let fx = fixture("bua9-arm-i-committed-earlier").await;
        stranded_overwrite(&fx).await;
        let sha = CREATE_LAYER_COMMIT_SCRIPT.sha1().to_owned();
        let reply = format!("return {{'COMMITTED_EARLIER', 'value:{MOVED_ON_TO}'}}");
        let rule: ProxyRule = Arc::new(move |args: &[Vec<u8>]| {
            (args.first()?.eq_ignore_ascii_case(b"EVALSHA")
                && args.get(1).is_some_and(|a| a.as_slice() == sha.as_bytes()))
            .then(|| ProxyAction::Eval(reply.clone()))
        });
        let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
        let swept_through = client_through(&proxy).await;
        swept_through.set_orphan_sweep_grace_for_test(Duration::ZERO);

        sweep_one_pass(
            &swept_through,
            fx.tv.key_schema(),
            fx.ext.clone(),
            Arc::new(LayerStore::new(
                swept_through.clone(),
                fx.tv.key_schema().clone(),
                4096,
            )),
        )
        .await;

        assert_eq!(
            proxy.hits(),
            1,
            "precondition: arm (i)'s commit run met the staged COMMITTED_EARLIER"
        );
        assert_eq!(
            (
                deleted_entries(&fx, fx.object_id).await,
                header_exists(&fx, &fx.object_id.to_string()).await
            ),
            (1, false),
            "the pass runs E's tail and awaits it: E is woken and retired when the pass returns"
        );
    }

    /// Arm (i) when the fence's successor X is gone: nothing can be mapped in
    /// E's place, so the pass leaves E mapped and whole and logs one ERROR.
    ///
    /// Current-thread, so the sweep task's ERROR reaches the thread-local
    /// capture; the fixture's keys are purged at the end.
    #[tokio::test]
    async fn orphan_sweep_logs_one_error_and_leaves_a_stranded_overwrite_whose_successor_is_gone() {
        let capture = ThreadCapture::start();
        let fx = fixture("bua9-arm-i-successor-gone").await;
        let x = stranded_overwrite(&fx).await;
        let removed: i64 = fx
            .tv
            .client()
            .pool
            .del(fx.tv.key_schema().obj_header(&x.to_string()))
            .await
            .expect("DEL of X's header must succeed");
        assert_eq!(removed, 1, "precondition: X's header existed");

        fx.tv
            .client()
            .set_orphan_sweep_grace_for_test(Duration::ZERO);
        sweep(&fx, plain_store(&fx)).await;
        let captured = capture.snapshot();
        let mapped = mapping(&fx, fx.layer).await;
        let e_seq = header_field(&fx, &fx.object_id.to_string(), "seq").await;
        purge_keys(&fx.tv).await;

        assert_eq!(
            mapped,
            Some(fx.object_id.to_string()),
            "E stays the layer's mapping"
        );
        assert_eq!(e_seq.as_deref(), Some("1"), "E's keys are intact");
        assert_eq!(
            captured.with_message(ERROR_ARM_I_SUCCESSOR_GONE).len(),
            1,
            "one ERROR; full capture:\n{}",
            captured.text
        );
    }

    /// Arm (i) drops an operation id from the successor's header that would
    /// not make a valid record key (here `bad:id`, refused by the
    /// operation-id validation): the pass still completes the overwrite, and
    /// writes no record under that id.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn orphan_sweep_completes_a_stranded_overwrite_without_a_record_for_an_invalid_header_operation_id(
    ) {
        const INVALID_ID: &str = "bad:id";
        let fx = fixture("bua9-arm-i-invalid-op").await;
        let x = stranded_overwrite_with(&fx, Some(create_op(fx.layer, PAYLOAD))).await;
        let _: () = fx
            .tv
            .client()
            .pool
            .hset(
                fx.tv.key_schema().obj_header(&x.to_string()),
                [(HEADER_CREATE_OP_ID, INVALID_ID)],
            )
            .await
            .expect("HSET of the invalid operation id must succeed");

        fx.tv
            .client()
            .set_orphan_sweep_grace_for_test(Duration::ZERO);
        sweep(&fx, plain_store(&fx)).await;

        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(x.to_string()),
            "precondition: the pass completed the overwrite"
        );
        let invalid_record: Option<String> = fx
            .tv
            .client()
            .pool
            .get(
                fx.tv
                    .key_schema()
                    .create_layer_operation(&fx.session_id, INVALID_ID),
            )
            .await
            .expect("GET of the operation record must succeed");
        assert_eq!(
            invalid_record, None,
            "no record may be written under an operation id the validation refuses"
        );
    }

    /// The re-plan as a new layer: an overwrite's first run fails before its
    /// script runs, and before its check a `DeleteLayer` (through another
    /// store) removes the layer. No run of the call landed, so the check
    /// reads the mapping moved to absent and the call re-plans as a new
    /// layer: OK, the mapping names the call's object, the layer is in the
    /// index, and the re-plan INFO is logged once.
    ///
    /// Before b.ua9 the transient path gave up on an absent mapping and the
    /// call answered `UNAVAILABLE`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_layer_overwrite_whose_layer_a_delete_layer_removed_before_its_check_re_plans_as_a_new_layer(
    ) {
        let fx = Arc::new(fixture("bua9-replan-new-layer").await);
        let other = Arc::new(plain_store(&fx));
        let slot: Acted = Arc::new(Mutex::new(None));
        let hook: WatchWindowHook = {
            let (fx, other, slot) = (fx.clone(), other.clone(), slot.clone());
            Arc::new(move |_client: &Client| {
                let (fx, other, slot) = (fx.clone(), other.clone(), slot.clone());
                async move {
                    *slot.lock().unwrap() = Some(act(&fx, &other, Then::Delete).await);
                }
                .boxed()
            })
        };
        let injected = faults([io(false)]);
        let store = plain_store(&fx)
            .with_create_layer_commit_fault_injector_for_test(Arc::clone(&injected))
            .with_create_layer_post_failed_run_hook_for_test(hook);

        let (created, captured) =
            capture(create_within(&store, &fx, fx.layer, HOOKED_CALL_BUDGET)).await;
        wait_cleanup(&store).await;
        wait_cleanup(&other).await;

        acted(&slot);
        assert_eq!(
            remaining(&injected),
            0,
            "precondition: the first run failed"
        );
        let created = created.expect("the call must re-plan as a new layer and commit");
        assert_eq!(
            mapping(&fx, fx.layer).await,
            Some(created.object_id.to_string()),
            "the layer maps the call's object"
        );
        assert!(indexed(&fx, fx.layer).await, "the layer is in the index");
        assert_eq!(
            captured.with_message(INFO_REPLAN_AS_NEW_LAYER).len(),
            1,
            "full capture:\n{}",
            captured.text
        );
    }
}
