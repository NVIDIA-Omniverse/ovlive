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

//! The paged teardown of a retiring object on its routes (b.fsz MR-D4, D4-T7
//! and D4-T17; `storage::retirement::Teardown`): the first `UNLINK` holds
//! `obj:hdr`, `obj:ext` and every other fixed key, the row pages follow, the
//! five indices go last; on the overwrite and sweep routes a page that fails
//! transiently after the first `UNLINK` is retried in place, since nothing
//! finds the object again once its header is gone; `DeleteLayer` retries the
//! whole step from its retry loop instead, from the indices still standing.
//!
//! | Test | Design row | What it pins |
//! |---|---|---|
//! | [`a_delete_whose_first_unlink_fails_keeps_the_header_and_obj_ext_with_its_stream_woken_and_expiring`] | D4-T7 | a fault on the first `UNLINK` keeps the header and `obj:ext` together, after the stream was woken and given its expiry |
//! | [`a_delete_whose_later_page_fails_once_retries_and_leaves_only_its_expiring_stream`] | D4-T7 | `DeleteLayer`, a page failing after the first `UNLINK`: the retry loop's re-run pages from the indices and leaves only the TTL'd stream (and the release carrier keys, which outlive a teardown by design) |
//! | [`an_overwrite_body_retries_a_failed_later_page_in_place_and_removes_every_key`] | D4-T17 | the overwrite body: a transient failure on the teardown's second command after the first `UNLINK` still ends with every `obj:*` key gone |
//! | [`a_sweep_retirement_retries_a_failed_later_page_in_place_and_removes_every_key`] | D4-T17 | the orphan sweep's generic arm, the same |
//! | [`an_overwrite_body_retries_a_later_page_retirement_page_retries_times_then_gives_up`] | D4-T17 | the in-place retry's bound: `RETIREMENT_PAGE_RETRIES` failures in a row are retried and the teardown finishes; one more and the body gives up, every fault taken, rows left |
//! | [`a_delete_whose_first_index_page_read_fails_once_retries_and_removes_every_key`] | D4-T7 | b.u96's site, the first index-page read (`SSCAN obj:secs`): `DeleteLayer`'s retry loop re-runs and removes every key |
//! | [`a_delete_whose_teardown_runs_out_of_retries_after_its_first_unlink_warns_naming_the_object_and_the_key`] | F2 | a teardown out of transient retries after its first `UNLINK`: one WARN per failed attempt naming the object and the failed command's full key; the client text stays plain |
//! | [`a_teardown_failure_reads_the_same_on_every_route`] | F9 | `DeleteLayer`, the overwrite body and the sweep report a teardown failure as `<op> on <key>` beside their object field, the key once |
//! | [`a_delete_whose_teardown_is_refused_out_of_memory_after_its_first_unlink_warns_naming_the_object_and_the_key`] | F2 | the store-write-refusal class: one WARN naming the object and the full key, `UNAVAILABLE` with the plain label, never retried |
//! | [`a_teardown_on_an_objects_page_widens_to_the_full_page_before_reading_its_second_rank_page`] | F3 | the teardown's own rank paging, no walk before it: widened to `W` before the read past the object's page |
//!
//! # Worst case per test
//!
//! Every wait for a condition is bounded by `BUDGET` (20 s), as in
//! `retirement_budget_test.rs`: a test's worst case is its number of such
//! waits × `BUDGET`, [`Rig::sweep_until`] counting two (its condition and its
//! join), and each stays inside nextest's 120 s hang-killer. The in-place
//! retry's backoff (100 ms doubling to 1 s per retry) is not a wait for a
//! condition and adds at most a few seconds.
//!
//! | Test | Bounded waits | Worst case |
//! |---|---|---|
//! | `a_delete_whose_first_unlink_fails_…` | 0 | — |
//! | `a_delete_whose_later_page_fails_once_…` | 0 | — |
//! | `an_overwrite_body_retries_a_failed_later_page_…` | 2 | 40 s |
//! | `a_sweep_retirement_retries_a_failed_later_page_…` | 2 | 40 s |
//! | `an_overwrite_body_retries_a_later_page_retirement_page_retries_times_…` | 2 | 40 s |
//! | `a_delete_whose_first_index_page_read_fails_once_…` | 0 | — |
//! | `a_delete_whose_teardown_runs_out_of_retries_…` | 0 | — |
//! | `a_teardown_failure_reads_the_same_on_every_route` | 4 | 80 s |
//! | `a_delete_whose_teardown_is_refused_out_of_memory_…` | 0 | — |
//! | `a_teardown_on_an_objects_page_widens_…` | 0 | — |
//!
//! Elsewhere: the first `UNLINK`'s contents and order, the last's, and that
//! the pieces add up to `KeySchema::object_cleanup_keys` with no stream key —
//! the in-crate `storage::keys` tests (section G); a void-token carrier
//! settling `pay` once a teardown stopped after its first `UNLINK` — the
//! in-crate `storage::orphan_sweep` tests; the teardown's heap per page —
//! `tests/upload_memory_envelope`.
//!
//! The objects the later-page tests retire hold sections under the root,
//! fields and time samples ([`populated`]), so the sections, fields and
//! time-samples pages have members to remove. Requires Valkey
//! (`TEST_VALKEY_URL`) and the `test-fault-injection` feature: the module is
//! compiled out without it.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fred::interfaces::KeysInterface;
use ordered_float::OrderedFloat;

use live_edit::domain::types::{Field, FieldValue, ObjectState, Section, TimeSample};
use live_edit::error::ServiceError;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::retirement::RetirementFaults;

use crate::storage::retirement_budget_test::{eventually, make_foreign, Rig, BUDGET, R_REF};

/// Sections the [`populated`] state holds, each with a field and a time
/// sample.
const SECTIONS: u64 = 40;

/// A state with rows on every index: [`SECTIONS`] sections under the root,
/// each with an inline field and an inline time sample.
fn populated() -> ObjectState {
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        next_set_order: SECTIONS + 1,
        ..Default::default()
    };
    let ids: Vec<u64> = (1..=SECTIONS).collect();
    for &id in &ids {
        state.sections.insert(
            id,
            Section {
                section_id: id,
                parent_id: 0,
                name: format!("s{id}"),
                section_type: 1,
                section_order: u32::try_from(id).expect("few sections"),
                first_version: 1,
                last_version: 1,
            },
        );
        state.fields.insert(
            (id, "f".to_owned()),
            Field {
                section_id: id,
                key_name: "f".to_owned(),
                value: FieldValue::Inline(vec![1, 2, 3]),
                set_order: id,
                authoring_set_order: id,
                version: 1,
            },
        );
        state.time_samples.insert(
            (id, OrderedFloat(1.0)),
            TimeSample {
                section_id: id,
                time: OrderedFloat(1.0),
                value: FieldValue::Inline(vec![4]),
                version: 1,
            },
        );
    }
    state.children.insert(0, ids);
    state
}

/// Create `layer` through `ls` with the [`populated`] state.
async fn populated_layer(rig: &Rig, ls: &LayerStore, layer: &str) -> u64 {
    ls.create_layer(
        &rig.session_id,
        layer,
        "teardown-tester",
        &populated(),
        rig.ext.clone(),
    )
    .await
    .expect("create_layer must succeed")
}

/// A script of later-command faults: `before` commands run, then one
/// transient failure.
fn fail_later_command(before: usize) -> RetirementFaults {
    let mut script: VecDeque<Option<fred::error::Error>> = (0..before).map(|_| None).collect();
    script.push_back(Some(fred::error::Error::new(
        fred::error::ErrorKind::Timeout,
        "injected teardown page timeout",
    )));
    Arc::new(Mutex::new(script))
}

/// The remaining time to live of `oid`'s stream, in ms (`PTTL`: -2 when
/// absent, -1 when it has no expiry).
async fn stream_pttl(rig: &Rig, oid: u64) -> i64 {
    rig.tv
        .client()
        .pool
        .pttl(rig.tv.key_schema().layer_stream(oid))
        .await
        .expect("PTTL must succeed")
}

/// D4-T7: a `DeleteLayer` whose teardown's first `UNLINK` fails (its retry
/// budget one attempt) keeps `obj:hdr` and `obj:ext` — both, together, so
/// the object is found again and the settle's liveness gate is not misled —
/// and by then its stream already carries `deleted` and an expiry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_whose_first_unlink_fails_keeps_the_header_and_obj_ext_with_its_stream_woken_and_expiring(
) {
    let rig = Rig::new("teardown-first").await;
    let first = Arc::new(Mutex::new(VecDeque::from([fred::error::Error::new(
        fred::error::ErrorKind::IO,
        "injected first UNLINK failure",
    )])));
    let ls = rig
        .store()
        .with_update_layer_retry_max_attempts(1)
        .with_cleanup_old_object_fault_injector_for_test(Arc::clone(&first));
    let oid = rig.layer(&ls, "first-unlink", 1, 3).await;

    let result = ls
        .delete_layer(&rig.session_id, "first-unlink", rig.ext.clone())
        .await;

    assert!(
        matches!(result, Err(ServiceError::Aborted(_))),
        "fixture: the injected failure ends the one attempt; got {result:?}"
    );
    assert!(
        first.lock().unwrap().is_empty(),
        "fixture: the first UNLINK took it"
    );
    let keys = rig.tv.key_schema();
    assert!(
        rig.header_exists(oid).await && rig.exists(&keys.obj_ext_index(&oid.to_string())).await,
        "the header and obj:ext stand, together"
    );
    assert_eq!(
        rig.deleted_entries(oid).await,
        1,
        "the stream was woken first"
    );
    assert!(
        stream_pttl(&rig, oid).await > 0,
        "the stream was given its expiry first"
    );
}

/// D4-T7: a `DeleteLayer` whose teardown fails transiently on a page after
/// the first `UNLINK` re-runs its retry-loop body, which pages again from the
/// indices (they go last), and ends with nothing of the object left but its
/// expiring stream (and any release carrier keys, which outlive a teardown).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_whose_later_page_fails_once_retries_and_leaves_only_its_expiring_stream() {
    let rig = Rig::new("teardown-delete-retry").await;
    let faults = fail_later_command(1);
    let ls = rig
        .store()
        .with_retirement_teardown_faults_for_test(Arc::clone(&faults));
    let oid = populated_layer(&rig, &ls, "delete-retry").await;

    ls.delete_layer(&rig.session_id, "delete-retry", rig.ext.clone())
        .await
        .expect("the retry loop re-runs the teardown and the delete succeeds");

    assert!(
        faults.lock().unwrap().is_empty(),
        "fixture: the fault was taken"
    );
    assert_eq!(
        rig.object_keys(oid).await,
        Vec::<String>::new(),
        "nothing of the object is left"
    );
    assert!(stream_pttl(&rig, oid).await > 0, "but its stream, expiring");
}

/// D4-T17: the overwrite body's teardown, its second command after the first
/// `UNLINK` failing transiently once, retries that command in place and ends
/// with every `obj:*` key of the replaced object gone. Without the in-place
/// retry the body would WARN and stop with the header already gone,
/// stranding the rows and indices for good.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_overwrite_body_retries_a_failed_later_page_in_place_and_removes_every_key() {
    let rig = Rig::new("teardown-overwrite-retry").await;
    let faults = fail_later_command(1);
    let ls = rig
        .store()
        .with_retirement_teardown_faults_for_test(Arc::clone(&faults));
    let old = populated_layer(&rig, &ls, "overwritten").await;

    rig.layer(&ls, "overwritten", 2, 0).await;

    eventually("the body took the fault", || async {
        faults.lock().unwrap().is_empty()
    })
    .await;
    rig.until_object_gone(old).await;
}

/// D4-T17: the orphan sweep's generic arm, the same: its teardown's second
/// command after the first `UNLINK` fails transiently once, is retried in
/// place, and every `obj:*` key of the orphan is gone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sweep_retirement_retries_a_failed_later_page_in_place_and_removes_every_key() {
    use fred::interfaces::SetsInterface;

    let rig = Rig::new("teardown-sweep-retry").await;
    let ls = rig.store();
    let oid = populated_layer(&rig, &ls, "orphaned").await;
    let keys = rig.tv.key_schema();
    let _: i64 = rig
        .tv
        .client()
        .pool
        .del(keys.layer_mapping(&rig.session_id, "orphaned"))
        .await
        .expect("DEL mapping must succeed");
    let _: i64 = rig
        .tv
        .client()
        .pool
        .srem(
            keys.layers_index(&rig.session_id),
            live_edit::storage::keys::hex_encode("orphaned"),
        )
        .await
        .expect("SREM layers index must succeed");
    let faults = fail_later_command(1);

    rig.sweep_until(
        rig.store()
            .with_orphan_sweep_grace(Duration::ZERO)
            .with_retirement_teardown_faults_for_test(Arc::clone(&faults)),
        "the sweep removed every key of the orphan",
        || async { rig.object_keys(oid).await.is_empty() },
    )
    .await;

    assert!(
        faults.lock().unwrap().is_empty(),
        "fixture: the fault was taken"
    );
}

/// A script of later-command faults: `before` commands run, then `failures`
/// transient failures in a row — the in-place retry takes one per attempt of
/// the same command.
fn fail_later_command_times(before: usize, failures: u32) -> RetirementFaults {
    let mut script: VecDeque<Option<fred::error::Error>> = (0..before).map(|_| None).collect();
    for _ in 0..failures {
        script.push_back(Some(fred::error::Error::new(
            fred::error::ErrorKind::Timeout,
            "injected teardown page timeout",
        )));
    }
    Arc::new(Mutex::new(script))
}

/// D4-T17, the in-place retry's bound: on the overwrite body a later command
/// failing transiently `RETIREMENT_PAGE_RETRIES` times in a row is retried
/// that often and the teardown finishes; failing once more than that
/// (`RETIREMENT_PAGE_RETRIES + 1`), the body gives up after the last retry —
/// every fault taken, and the rows the stopped teardown had not reached left
/// behind its removed header.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_overwrite_body_retries_a_later_page_retirement_page_retries_times_then_gives_up() {
    use live_edit::storage::retirement::RETIREMENT_PAGE_RETRIES;
    for (failures, finishes) in [
        (RETIREMENT_PAGE_RETRIES, true),
        (RETIREMENT_PAGE_RETRIES + 1, false),
    ] {
        let rig = Rig::new("teardown-retry-bound").await;
        let faults = fail_later_command_times(1, failures);
        let ls = rig
            .store()
            .with_retirement_teardown_faults_for_test(Arc::clone(&faults));
        let old = populated_layer(&rig, &ls, "bounded").await;

        rig.layer(&ls, "bounded", 2, 0).await;
        tokio::time::timeout(BUDGET, ls.wait_for_cleanup(BUDGET * 2))
            .await
            .expect("the body ends");

        assert!(
            faults.lock().unwrap().is_empty(),
            "{failures} failures: every fault was taken, one per attempt"
        );
        assert!(
            !rig.header_exists(old).await,
            "{failures} failures: the first UNLINK ran"
        );
        let left = rig.object_keys(old).await;
        if finishes {
            assert_eq!(
                left,
                Vec::<String>::new(),
                "{failures} failures: retried in place, the teardown finished"
            );
        } else {
            assert!(
                !left.is_empty(),
                "{failures} failures: past the bound the body gave up, leaving the rows it had \
                 not reached"
            );
        }
    }
}

/// b.u96's original site, the teardown's first index-page read: a
/// `DeleteLayer` whose `SSCAN obj:secs` (the first command after the first
/// `UNLINK`) fails transiently once re-runs its retry-loop body, pages again
/// from the indices, and removes every key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_whose_first_index_page_read_fails_once_retries_and_removes_every_key() {
    let rig = Rig::new("teardown-index-read").await;
    let faults = fail_later_command(0);
    let ls = rig
        .store()
        .with_retirement_teardown_faults_for_test(Arc::clone(&faults));
    let oid = populated_layer(&rig, &ls, "index-read").await;

    ls.delete_layer(&rig.session_id, "index-read", rig.ext.clone())
        .await
        .expect("the retry loop re-runs the teardown and the delete succeeds");

    assert!(
        faults.lock().unwrap().is_empty(),
        "fixture: the index read took the fault"
    );
    assert_eq!(
        rig.object_keys(oid).await,
        Vec::<String>::new(),
        "nothing of the object is left"
    );
}

/// The message of `DeleteLayer`'s teardown-failure WARN
/// (`storage::layer::cleanup_old_object`).
const DELETE_TEARDOWN_FAILED: &str = "could not delete a deleted layer's object keys; the delete \
     retries a transient failure. If the delete still fails once the object's header is gone, the \
     keys not yet deleted are left behind, and nothing deletes them";

/// The message of the overwrite body's teardown-failure WARN.
const OVERWRITE_TEARDOWN_FAILED: &str = "failed to clean up old object keys";

/// The message of the orphan sweep's failed-retirement WARN.
const SWEEP_RETIREMENT_FAILED: &str = "failed to clean up orphaned object";

/// The failed command as every route reports it
/// (`TeardownFailure::label`): `<op> on <key>`.
fn reported(op: &str, key: &str) -> String {
    format!("{op} on {key}")
}

/// F2: a `DeleteLayer` whose teardown runs out of transient retries after
/// its first `UNLINK` — its retry budget two attempts, the first index-page
/// read (`SSCAN obj:secs`) failing transiently on both — strands the rows
/// and indices with no header and no mapping. Each failed attempt WARNs once,
/// naming the object and the failed command with its full key, so an
/// operator can find the stranded keys; the error the client sees keeps the
/// plain command label, with no key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_whose_teardown_runs_out_of_retries_after_its_first_unlink_warns_naming_the_object_and_the_key(
) {
    use crate::grpc::log_capture::capture;

    let rig = Rig::new("teardown-exhausted").await;
    let faults = fail_later_command_times(0, 2);
    let ls = rig
        .store()
        .with_update_layer_retry_max_attempts(2)
        .with_retirement_teardown_faults_for_test(Arc::clone(&faults));
    let oid = populated_layer(&rig, &ls, "exhausted").await;
    let secs = rig.tv.key_schema().obj_sections_index(&oid.to_string());

    let (result, captured) =
        capture(ls.delete_layer(&rig.session_id, "exhausted", rig.ext.clone())).await;

    match &result {
        Err(ServiceError::Aborted(text)) => {
            assert!(
                text.contains("SSCAN obj sections index") && !text.contains(&secs),
                "the client-facing text keeps the plain command label; got {text:?}"
            );
        }
        other => panic!("both attempts failed transiently: retries exhausted; got {other:?}"),
    }
    assert!(
        faults.lock().unwrap().is_empty(),
        "fixture: each attempt took its fault"
    );
    let warns = captured.with_message(DELETE_TEARDOWN_FAILED);
    assert_eq!(
        warns.len(),
        2,
        "one WARN per failed teardown attempt; full capture:\n{}",
        captured.text
    );
    for warn in warns {
        assert_eq!(warn["level"], "WARN", "full event: {warn}");
        assert_eq!(warn["fields"]["object_id"], oid.to_string(), "the object");
        assert_eq!(
            warn["fields"]["failure"],
            reported("SSCAN obj sections index", &secs),
            "the failed command and its full key"
        );
    }
    assert!(
        !rig.header_exists(oid).await && rig.exists(&secs).await,
        "fixture: the header went with the first UNLINK, and the indices are stranded"
    );
}

/// F9: one teardown failure reads the same on every route. A foreign
/// `obj:secs` (a string) is counted by each route's size read under
/// `key_family="sections_index"` and WARNed there, and the route continues at
/// the full page; this test asserts neither, only the teardown lines. The
/// teardown's own index-page reads count nothing: the foreign index stops the
/// teardown of a `DeleteLayer`, of an overwrite's cleanup body and of the
/// orphan sweep's generic arm at its first index-page read, after the first
/// `UNLINK` took the header, and each route's
/// line reports it as `SSCAN obj sections index on <key>`
/// (`TeardownFailure::label`) once, beside the route's own object field and
/// with no separate `key` field: `DeleteLayer`'s and the overwrite body's
/// WARN in their `failure` field, the sweep's in its error (the sweep's
/// `SweepError` wraps it). `DeleteLayer`'s `INTERNAL` detail keeps the plain
/// command label. The overwrite body and the sweep log from spawned tasks, so
/// the test runs on a current-thread runtime under a thread-local capture.
#[tokio::test]
async fn a_teardown_failure_reads_the_same_on_every_route() {
    use crate::grpc::log_capture::{assert_field_absent, purge_keys, ThreadCapture};
    use fred::interfaces::SetsInterface;

    let capture = ThreadCapture::start();
    let rig = Rig::new("teardown-one-format").await;
    let secs_of = |oid: u64| rig.tv.key_schema().obj_sections_index(&oid.to_string());

    // DeleteLayer.
    let ls = rig.store().with_update_layer_retry_max_attempts(1);
    let deleted = populated_layer(&rig, &ls, "foreign-delete").await;
    make_foreign(&rig, &secs_of(deleted)).await;
    let result = ls
        .delete_layer(&rig.session_id, "foreign-delete", rig.ext.clone())
        .await;
    match &result {
        Err(ServiceError::Internal(detail)) => assert!(
            detail.contains("SSCAN obj sections index") && !detail.contains(&secs_of(deleted)),
            "the INTERNAL detail keeps the plain command label; got {detail:?}"
        ),
        other => panic!("a foreign obj:secs is a permanent failure: INTERNAL; got {other:?}"),
    }

    // The overwrite body, released from its gate once its old index is foreign.
    let gate = Arc::new(tokio::sync::Barrier::new(2));
    let ls = rig
        .store()
        .with_overwrite_cleanup_gate_for_test(Arc::clone(&gate));
    let replaced = populated_layer(&rig, &ls, "foreign-overwrite").await;
    rig.layer(&ls, "foreign-overwrite", 2, 0).await;
    make_foreign(&rig, &secs_of(replaced)).await;
    tokio::time::timeout(BUDGET, gate.wait())
        .await
        .expect("the body is parked on its gate");
    eventually("the overwrite body's WARN", || async {
        !capture
            .snapshot()
            .with_message(OVERWRITE_TEARDOWN_FAILED)
            .is_empty()
    })
    .await;

    // The orphan sweep's generic arm.
    let orphan = populated_layer(&rig, &rig.store(), "foreign-orphan").await;
    let keys = rig.tv.key_schema();
    let _: i64 = rig
        .tv
        .client()
        .pool
        .del(keys.layer_mapping(&rig.session_id, "foreign-orphan"))
        .await
        .expect("DEL mapping must succeed");
    let _: i64 = rig
        .tv
        .client()
        .pool
        .srem(
            keys.layers_index(&rig.session_id),
            live_edit::storage::keys::hex_encode("foreign-orphan"),
        )
        .await
        .expect("SREM layers index must succeed");
    make_foreign(&rig, &secs_of(orphan)).await;
    rig.sweep_until(
        rig.store().with_orphan_sweep_grace(Duration::ZERO),
        "the sweep's WARN",
        || async {
            !capture
                .snapshot()
                .with_message(SWEEP_RETIREMENT_FAILED)
                .is_empty()
        },
    )
    .await;

    let captured = capture.snapshot();
    let delete_warn = captured.only(DELETE_TEARDOWN_FAILED);
    assert_eq!(delete_warn["fields"]["object_id"], deleted.to_string());
    assert_eq!(
        delete_warn["fields"]["failure"],
        reported("SSCAN obj sections index", &secs_of(deleted))
    );
    let overwrite_warn = captured.only(OVERWRITE_TEARDOWN_FAILED);
    assert_eq!(
        overwrite_warn["fields"]["old_object_id"],
        replaced.to_string()
    );
    assert_eq!(
        overwrite_warn["fields"]["failure"],
        reported("SSCAN obj sections index", &secs_of(replaced))
    );
    let sweep_warn = captured.only(SWEEP_RETIREMENT_FAILED);
    assert_eq!(sweep_warn["fields"]["object_id"], orphan.to_string());
    let sweep_error = sweep_warn["fields"]["error"]
        .as_str()
        .expect("the sweep's error is text");
    assert!(
        sweep_error.starts_with(&format!(
            "orphan sweep error: {} failed (",
            reported("SSCAN obj sections index", &secs_of(orphan))
        )),
        "the sweep's error carries the same report; got {sweep_error:?}"
    );
    for warn in [delete_warn, overwrite_warn, sweep_warn] {
        assert_field_absent(warn, "key", "the key is named once, in the report");
    }
    purge_keys(&rig.tv).await;
}

/// F2, the store-write-refusal class: a `DeleteLayer` whose teardown's first
/// index-page read after its first `UNLINK` (`SSCAN obj:secs`) is answered
/// `-OOM` WARNs once, naming the object and the failed command with its full
/// key, and answers `UNAVAILABLE` with the plain command label. A store write
/// refusal is never retried in process, so with a retry budget of two
/// attempts there is still one attempt and one WARN. With F2's transient
/// case and F9's permanent one, each failure class reaches the same line.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_whose_teardown_is_refused_out_of_memory_after_its_first_unlink_warns_naming_the_object_and_the_key(
) {
    use crate::grpc::log_capture::capture;

    let rig = Rig::new("teardown-oom").await;
    let faults: RetirementFaults =
        Arc::new(Mutex::new(VecDeque::from([Some(fred::error::Error::new(
            fred::error::ErrorKind::Unknown,
            "OOM command not allowed when used memory > 'maxmemory'.",
        ))])));
    let ls = rig
        .store()
        .with_update_layer_retry_max_attempts(2)
        .with_retirement_teardown_faults_for_test(Arc::clone(&faults));
    let oid = populated_layer(&rig, &ls, "refused").await;
    let secs = rig.tv.key_schema().obj_sections_index(&oid.to_string());

    let (result, captured) =
        capture(ls.delete_layer(&rig.session_id, "refused", rig.ext.clone())).await;

    match &result {
        Err(ServiceError::Unavailable(text)) => assert!(
            text.contains("SSCAN obj sections index") && !text.contains(&secs),
            "the client-facing text keeps the plain command label; got {text:?}"
        ),
        other => panic!("a store write refusal answers UNAVAILABLE; got {other:?}"),
    }
    assert!(
        faults.lock().unwrap().is_empty(),
        "fixture: the index read took the refusal"
    );
    let warn = captured.only(DELETE_TEARDOWN_FAILED);
    assert_eq!(warn["level"], "WARN", "full event: {warn}");
    assert_eq!(warn["fields"]["object_id"], oid.to_string(), "the object");
    assert_eq!(
        warn["fields"]["failure"],
        reported("SSCAN obj sections index", &secs),
        "the failed command and its full key"
    );
    assert!(
        !rig.header_exists(oid).await && rig.exists(&secs).await,
        "fixture: the header went with the first UNLINK, and the indices are stranded"
    );
}

/// F3, the teardown's own rank-page widening, where no walk widened first (a
/// release already scheduled skips the walk; a walk that failed before
/// paging never widened): a teardown reading at an object's one-member page,
/// whose `obj:flds` holds three full pages — as an index that grew after the
/// size read leaves it — widens its reservation to the full page before it
/// reads the second page. The test runs `Teardown` on a reservation it holds
/// itself, with the second `obj:flds` `ZRANGE` scripted to fail, so the
/// teardown stops at that read: the reservation already holds the full page,
/// which it can only have taken before the read was issued. A teardown that
/// read past the object's page without widening, or widened only after the
/// read, leaves the one-member page.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_teardown_on_an_objects_page_widens_to_the_full_page_before_reading_its_second_rank_page()
{
    use fred::interfaces::SortedSetsInterface;
    use live_edit::storage::keys::KeySchema;
    use live_edit::storage::retirement::{RetirementPaging, RetirementReservation, Teardown};

    let rig = Rig::new("teardown-widen").await;
    let ls = rig.store();
    let oid = rig.layer(&ls, "widen", 1, 0).await;
    let (keys, oid_str) = (rig.tv.key_schema(), oid.to_string());
    let full = RetirementPaging::for_allowance(R_REF);
    for i in 0..3 * full.page_len() {
        let _: i64 = rig
            .tv
            .client()
            .pool
            .zadd(
                keys.obj_fields_index(&oid_str),
                None,
                None,
                false,
                false,
                (i as f64, KeySchema::field_member("1", &format!("w{i}"))),
            )
            .await
            .expect("ZADD must succeed");
    }
    let own = full.for_rows(1);
    let reservation = RetirementReservation::reserve(ls.retirement_budget(), full, own, 0)
        .await
        .expect("the allowance is open and empty");
    assert!(
        reservation.bytes() < full.page_bytes(),
        "fixture: the reservation holds the object's page, below the full page"
    );
    // After the first UNLINK: the three set indices' `SSCAN` steps (each
    // empty, so no `UNLINK` follows), the first `obj:flds` page and its one
    // row's `UNLINK`, then the second page's `ZRANGE`.
    let script: RetirementFaults = Arc::new(Mutex::new(VecDeque::from([
        None,
        None,
        None,
        None,
        None,
        Some(fred::error::Error::new(
            fred::error::ErrorKind::Protocol,
            "injected second-page failure",
        )),
    ])));

    let failure = Teardown::new(&rig.tv.client().pool, keys, &oid_str, full)
        .reserved(Some(&*reservation))
        .with_faults(None, Some(Arc::clone(&script)))
        .run()
        .await
        .expect_err("the scripted failure stops the teardown");

    assert!(
        script.lock().unwrap().is_empty(),
        "fixture: every scripted command was issued"
    );
    assert_eq!(
        (failure.op, failure.key.clone()),
        ("ZRANGE obj fields index", keys.obj_fields_index(&oid_str)),
        "fixture: the teardown stopped at its second obj:flds page"
    );
    assert_eq!(
        (reservation.page(), reservation.bytes()),
        (full, full.page_bytes()),
        "the teardown widened to the full page before the read that failed"
    );
}
