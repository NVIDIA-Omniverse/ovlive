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

//! The restore writer's session-scoped writes end to end
//! (`live_edit::dr::job::execute::session`; SDD §22.22 *The restore writer*,
//! SR-9.4, SR-9.5, SR-9.10): moves to a successor, abandoned deletes, session
//! restores, and defer-51, through the Job's frame against a real store and
//! a scripted backup store.
//!
//! - **W7, a move**: a layer mapped to a fenced object names its complete
//!   successor afterwards, and reads back the successor's content; the
//!   fenced object is left as it was. A kill before the move finds the
//!   successor already restored (moves run after the objects stage) and
//!   converges.
//! - **W9, an abandoned delete**: with the deletion marker as evidence, the
//!   session's objects are registered dead, wiped (streams included),
//!   unmapped, every layers-index member removed, their path-index members
//!   removed and the session backup deleted; a kill after each step leaves
//!   exactly the steps before it done (checked at each point) and converges,
//!   and one after the unmap still leaves them registered (registration
//!   comes first).
//! - **W10**: a session backup's mapping to a dead object is not written.
//! - **I-51 (defer-51)**: a surviving mapping to a dead object that the
//!   session backup's newer mapping replaces is not held; the newer mapping
//!   is written and its object restored; the compare-and-delete unmapping
//!   (Task 10's only way to remove a held mapping) leaves the live mapping.
//! - **Rule 2 over a surviving record**: a backup mapping newer than the
//!   surviving one is written into a session whose record survives, so the
//!   Job's record step is the unconditional one, not on-access recovery's
//!   conditional script.
//! - **A session killed after its path index** (Step A) is restored with
//!   its record (Step B) by the re-run.
//!
//! Every run uses a per-test key prefix (`TestValkey`). Waits: each test's
//! body runs under `writer_support`'s one-test budget (its module docs,
//! *Time*).

use std::sync::Arc;

use fred::interfaces::HashesInterface;
use tonic::Request;

use live_edit::dr::job::execute::KillPoint;
use live_edit::dr::job::plan::model::HeldReason;
use live_edit::grpc::proto::ListSessionsRequest;
use live_edit::grpc::session::handle_list_sessions;
use live_edit::grpc::BearerToken;
use live_edit::storage::dr_write::{unmap_layer_if, UnmapOutcome};
use live_edit::storage::keys::hex_encode;
use live_edit::storage::test_support::TestValkey;

use super::writer_support::*;
use crate::grpc::helpers::{build_service, AllowAllPermissions};

// ═════════════════════════════════════════════════════════════════════════════
// W7: a move to the successor
// ═════════════════════════════════════════════════════════════════════════════

/// W7's fenced object and its successor, whose newer backup is at
/// [`SUCCESSOR_BACKUP`].
const FENCED: u64 = 701;
const SUCCESSOR: u64 = 702;
const SUCCESSOR_BACKUP: u64 = 25;

/// W7's fixture: layer `m.usda` of session "s7" mapped to [`FENCED`],
/// retiring with its complete successor [`SUCCESSOR`] of the same layer,
/// whose newer backup is listed. Returns the successor's base and the
/// fenced object's header as it was.
async fn moved_fixture(
    tv: &TestValkey,
    store: &Arc<WriterStore>,
) -> (bytes::Bytes, std::collections::HashMap<String, String>) {
    let successor = base(2, 0x72, &[]);
    put_session(
        store.as_ref(),
        &session_backup("s7", &[("m.usda", FENCED)], &[]),
    )
    .await;
    put_base(
        store.as_ref(),
        SUCCESSOR,
        "s7",
        "m.usda",
        SUCCESSOR_BACKUP,
        successor.clone(),
    )
    .await;
    put_counters(store.as_ref()).await;
    seed_session_record(tv, "s7").await;
    seed_complete(
        tv,
        FENCED,
        "s7",
        "m.usda",
        10,
        &[("superseded", "1"), ("superseded_by", "702")],
    )
    .await;
    seed_complete(tv, SUCCESSOR, "s7", "m.usda", 20, &[]).await;
    seed_mapping(tv, "s7", "m.usda", FENCED).await;
    (successor, header(tv, FENCED).await)
}

/// What W7's run left: the mapping names the successor, whose content is
/// its newer backup's; the fenced object's header is untouched.
async fn assert_moved(
    tv: &TestValkey,
    successor: &bytes::Bytes,
    fenced_before: &std::collections::HashMap<String, String>,
) {
    assert_eq!(
        get(tv, tv.key_schema().layer_mapping("s7", "m.usda")).await,
        Some(SUCCESSOR.to_string()),
        "the layer names the successor"
    );
    assert_restored(
        tv,
        SUCCESSOR,
        &expected_state(
            SUCCESSOR,
            SUCCESSOR_BACKUP,
            successor,
            &[],
            SUCCESSOR_BACKUP + BAND,
        ),
    )
    .await;
    assert_eq!(
        &header(tv, FENCED).await,
        fenced_before,
        "the fenced object is left to the orphan sweep"
    );
}

/// W7. The move is recorded `(701, 702)` and lands after 702's restore.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_moves_a_layer_from_its_fenced_object_to_the_complete_successor() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let (successor, fenced_before) = moved_fixture(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let (_plan, report, _captured) = run_to_the_end(&tv, &config, &store).await;

        assert_eq!(report.remapped, vec![(FENCED, SUCCESSOR)]);
        assert_moved(&tv, &successor, &fenced_before).await;
    })
    .await;
}

/// W7, killed before the move: the successor is already restored from its
/// newer backup (the moves stage runs after the objects stage, so a move
/// never names a successor still being written), and the mapping still
/// names the fenced object, so the re-run moves it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_killed_before_a_move_converges() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let (successor, fenced_before) = moved_fixture(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        run_to_the_kill(&tv, &config, &store, |b| {
            b.with_kill_point_for_test(KillPoint::RemapBefore(Some(FENCED)))
        })
        .await;
        assert_eq!(
            header(&tv, SUCCESSOR).await.get("seq").cloned(),
            Some((SUCCESSOR_BACKUP + BAND).to_string()),
            "at the kill: the successor was restored before the move"
        );
        assert_eq!(
            get(&tv, tv.key_schema().layer_mapping("s7", "m.usda")).await,
            Some(FENCED.to_string()),
            "at the kill: the mapping still names the fenced object"
        );
        let (_plan, report, _captured) = run_to_the_end(&tv, &config, &store).await;

        assert_eq!(report.remapped, vec![(FENCED, SUCCESSOR)]);
        assert_moved(&tv, &successor, &fenced_before).await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// W9: an abandoned delete
// ═════════════════════════════════════════════════════════════════════════════

/// The deleted session.
const DELETED_SESSION: &str = "sd";
/// Its objects: two mapped, one named only by its header.
const DELETED_OBJECTS: [u64; 3] = [901, 902, 903];
/// The path-names member its backup lists.
const NAMES_MEMBER: &str = "sd-name\u{0}sd";
/// Another session's member of the same path index, which stays.
const OTHER_MEMBER: &str = "s-other";
/// The layer names its layers index holds a member for: the two mapped
/// layers, `d3.usda` (named only by object 903's header), and `d4.usda`
/// (named by nothing else).
const INDEXED_LAYERS: [&str; 4] = ["d1.usda", "d2.usda", "d3.usda", "d4.usda"];

/// W9's fixture: session [`DELETED_SESSION`], whose deletion marker
/// survived, with layers `d1.usda` → 901 and `d2.usda` → 902 mapped, a third
/// object 903 whose header names it as `d3.usda` with no mapping, each with
/// rows and a stream; a layers-index member for each of
/// [`INDEXED_LAYERS`], `d4.usda`'s with no mapping and no header naming it;
/// its path-index members, next to another session's; and its listed
/// session backup.
///
/// The service never leaves this store: `DeleteSession` deletes the layers
/// index in the transaction that writes the marker. The fixture is a store
/// edited from outside the service (or a kill no service code path makes):
/// it pins that the Job removes every layers-index member of a deleted
/// session whatever names it — a mapping, a header, or the index alone —
/// so a member that only the index carries never outlives the delete.
async fn deleted_fixture(tv: &TestValkey, store: &Arc<WriterStore>) {
    put_session(
        store.as_ref(),
        &session_backup(
            DELETED_SESSION,
            &[("d1.usda", 901), ("d2.usda", 902)],
            &[NAMES_MEMBER],
        ),
    )
    .await;
    put_counters(store.as_ref()).await;
    seed_deletion_marker(tv, DELETED_SESSION).await;
    for (oid, layer) in [(901, "d1.usda"), (902, "d2.usda"), (903, "d3.usda")] {
        seed_complete(tv, oid, DELETED_SESSION, layer, 10, &[]).await;
        seed_residue(tv, oid).await;
        xadd(tv, oid, "10-0").await;
    }
    seed_mapping(tv, DELETED_SESSION, "d1.usda", 901).await;
    seed_mapping(tv, DELETED_SESSION, "d2.usda", 902).await;
    for layer in ["d3.usda", "d4.usda"] {
        sadd(
            tv,
            tv.key_schema().layers_index(DELETED_SESSION),
            &hex_encode(layer),
        )
        .await;
    }
    assert_eq!(
        smembers(tv, tv.key_schema().layers_index(DELETED_SESSION)).await,
        INDEXED_LAYERS.iter().map(|l| hex_encode(l)).collect(),
        "fixture: a member per indexed layer"
    );
    let path = base_path_of(DELETED_SESSION);
    sadd(tv, tv.key_schema().path_index(&path), DELETED_SESSION).await;
    sadd(tv, tv.key_schema().path_index(&path), OTHER_MEMBER).await;
    sadd(tv, tv.key_schema().path_names_index(&path), NAMES_MEMBER).await;
}

/// Whether the deleted session's two mappings are each still there.
async fn mappings_left(tv: &TestValkey) -> [bool; 2] {
    [
        exists(
            tv,
            tv.key_schema().layer_mapping(DELETED_SESSION, "d1.usda"),
        )
        .await,
        exists(
            tv,
            tv.key_schema().layer_mapping(DELETED_SESSION, "d2.usda"),
        )
        .await,
    ]
}

/// Whether the deleted session's member of its path index is still there.
async fn path_member_left(tv: &TestValkey) -> bool {
    smembers(
        tv,
        tv.key_schema().path_index(&base_path_of(DELETED_SESSION)),
    )
    .await
    .contains(DELETED_SESSION)
}

/// The finished delete: the objects are registered dead and nothing of
/// them is left, streams included; no mapping and no layers-index member
/// remains; the session's path-index members are gone and the other
/// session's stays; its backup is deleted.
async fn assert_delete_finished(tv: &TestValkey, store: &Arc<WriterStore>) {
    assert_eq!(
        dead_set(tv).await,
        DELETED_OBJECTS.into_iter().collect(),
        "every object of the deleted session is registered dead"
    );
    for oid in DELETED_OBJECTS {
        assert_eq!(
            object_keys(tv, oid).await,
            Vec::<String>::new(),
            "object {oid}"
        );
    }
    assert_eq!(mappings_left(tv).await, [false, false], "no mapping left");
    let layers_index = smembers(tv, tv.key_schema().layers_index(DELETED_SESSION)).await;
    assert!(
        layers_index.is_empty(),
        "no layers-index member of the deleted session remains, d3.usda's ({}) and d4.usda's \
         ({}) included: {layers_index:?}",
        hex_encode("d3.usda"),
        hex_encode("d4.usda")
    );
    let path = base_path_of(DELETED_SESSION);
    assert_eq!(
        smembers(tv, tv.key_schema().path_index(&path)).await,
        [OTHER_MEMBER.to_owned()].into_iter().collect()
    );
    assert!(smembers(tv, tv.key_schema().path_names_index(&path))
        .await
        .is_empty());
    assert!(
        !store.holds_session(DELETED_SESSION).await,
        "the session backup is deleted"
    );
    assert!(
        !exists(tv, tv.key_schema().session_meta(DELETED_SESSION)).await,
        "no session record is written"
    );
}

/// W9. The Job finishes the delete in one run.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_finishes_an_abandoned_delete_registering_its_objects_dead() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        deleted_fixture(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let (_plan, report, _captured) = run_to_the_end(&tv, &config, &store).await;

        assert_eq!(report.deletes_completed, vec![DELETED_SESSION.to_owned()]);
        assert!(report.objects.is_empty(), "nothing of it is restored");
        assert_delete_finished(&tv, &store).await;
    })
    .await;
}

/// What a delete killed at `point` left: every step before the point done
/// and none after it. Registration, then the wipe, then the unmap, then the
/// path members, then the session backup.
async fn assert_delete_killed_at(tv: &TestValkey, store: &Arc<WriterStore>, point: &KillPoint) {
    let mut objects_left = Vec::new();
    for oid in DELETED_OBJECTS {
        objects_left.push(!object_keys(tv, oid).await.is_empty());
    }
    let state = (
        dead_set(tv).await == DELETED_OBJECTS.into_iter().collect(),
        objects_left,
        mappings_left(tv).await,
        path_member_left(tv).await,
        store.holds_session(DELETED_SESSION).await,
    );
    let want = match point {
        KillPoint::DeleteAfterRegister(_) => (true, vec![true; 3], [true; 2], true, true),
        KillPoint::DeleteAfterWipe(_) => (true, vec![false; 3], [true; 2], true, true),
        KillPoint::DeleteAfterUnmap(_) => (true, vec![false; 3], [false; 2], true, true),
        KillPoint::DeletePathMembersDone(_) => (true, vec![false; 3], [false; 2], false, true),
        other => panic!("not an abandoned delete's kill point: {other:?}"),
    };
    assert_eq!(
        state, want,
        "{point:?}: (registered, object keys left, mappings left, path member left, session \
         backup left) at the kill"
    );
}

/// Kill the delete at `point`, check what the kill left, run the Job again,
/// and check it finished.
async fn delete_kill_then_converge(point: KillPoint) {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        deleted_fixture(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        run_to_the_kill(&tv, &config, &store, |b| {
            b.with_kill_point_for_test(point.clone())
        })
        .await;
        assert_delete_killed_at(&tv, &store, &point).await;
        let (_plan, report, _captured) = run_to_the_end(&tv, &config, &store).await;

        assert_eq!(report.deletes_completed, vec![DELETED_SESSION.to_owned()]);
        assert_delete_finished(&tv, &store).await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_delete_killed_after_the_registration_converges() {
    delete_kill_then_converge(KillPoint::DeleteAfterRegister(Some(
        DELETED_SESSION.to_owned(),
    )))
    .await;
}

/// Also the pin that a name only a header carried is not lost: the wipe
/// removed object 903's header, the only thing naming `d3.usda` besides
/// the layers index, so the re-run finds that name in the index itself.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_delete_killed_after_the_wipe_converges() {
    delete_kill_then_converge(KillPoint::DeleteAfterWipe(Some(DELETED_SESSION.to_owned()))).await;
}

/// PD-1: killed after the unmap, the re-run's plan names none of the
/// objects any more; they are registered all the same, because the
/// registration came first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_delete_killed_after_the_unmap_converges_with_its_objects_registered() {
    delete_kill_then_converge(KillPoint::DeleteAfterUnmap(Some(
        DELETED_SESSION.to_owned(),
    )))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_delete_killed_after_the_path_members_converges() {
    delete_kill_then_converge(KillPoint::DeletePathMembersDone(Some(
        DELETED_SESSION.to_owned(),
    )))
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// W10 and I-51: mappings to dead objects
// ═════════════════════════════════════════════════════════════════════════════

/// W10. After a full wipe, a session backup mapping one layer to a dead
/// object and another to a live one: the session is restored with the live
/// mapping only.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_does_not_write_a_backup_mapping_to_a_dead_object() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        put_session(
            store.as_ref(),
            &session_backup("s10", &[("dead.usda", 1001), ("live.usda", 1002)], &[]),
        )
        .await;
        put_base(
            store.as_ref(),
            1001,
            "s10",
            "dead.usda",
            5,
            base(1, 0x10, &[]),
        )
        .await;
        put_base(
            store.as_ref(),
            1002,
            "s10",
            "live.usda",
            6,
            base(1, 0x11, &[]),
        )
        .await;
        put_counters(store.as_ref()).await;
        seed_dead(&tv, 1001).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let (_plan, report, _captured) = run_to_the_end(&tv, &config, &store).await;

        assert_eq!(report.sessions_restored, 1);
        assert!(!exists(&tv, tv.key_schema().layer_mapping("s10", "dead.usda")).await);
        assert!(
            object_keys(&tv, 1001).await.is_empty(),
            "the dead object is not restored"
        );
        assert_eq!(
            get(&tv, tv.key_schema().layer_mapping("s10", "live.usda")).await,
            Some("1002".to_owned())
        );
        assert!(exists(&tv, tv.key_schema().session_meta("s10")).await);
    })
    .await;
}

/// I-51 (defer-51). A surviving mapping `L → 1101` names a dead object with
/// no header; the session backup maps `L` to the newer, live 1102, whose
/// backup is listed. The plan holds no mapping to a deleted object (it
/// reports the replacement instead), Step B writes `L → 1102` and 1102 is
/// restored complete; Task 10's compare-and-delete of the held mapping, were
/// it run, answers `Other(1102)` and leaves `L → 1102` in place.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_replaces_a_dead_mapping_with_the_newer_backup_mapping_and_holds_nothing() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let replacement = base(2, 0x51, &[]);
        put_session(
            store.as_ref(),
            &session_backup("s51", &[("l.usda", 1102)], &[]),
        )
        .await;
        put_base(
            store.as_ref(),
            1102,
            "s51",
            "l.usda",
            8,
            replacement.clone(),
        )
        .await;
        put_counters(store.as_ref()).await;
        seed_session_record(&tv, "s51").await;
        seed_mapping(&tv, "s51", "l.usda", 1101).await;
        seed_dead(&tv, 1101).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let (plan, report, _captured) = run_to_the_end(&tv, &config, &store).await;

        assert!(
            !plan
                .held
                .iter()
                .any(|h| matches!(h.reason, HeldReason::MappedToDeletedObject { .. })),
            "no hold on the replaced mapping: {:?}",
            plan.held
        );
        assert_eq!(report.sessions_restored, 1, "Step B ran");
        let mapping = tv.key_schema().layer_mapping("s51", "l.usda");
        assert_eq!(get(&tv, mapping.clone()).await, Some("1102".to_owned()));
        assert_restored(
            &tv,
            1102,
            &expected_state(1102, 8, &replacement, &[], 8 + BAND),
        )
        .await;

        let outcome = unmap_layer_if(tv.client(), tv.key_schema(), "s51", "l.usda", Some(1101))
            .await
            .expect("the unmapping answers");
        assert_eq!(outcome, UnmapOutcome::Other(1102));
        assert_eq!(
            get(&tv, mapping).await,
            Some("1102".to_owned()),
            "the live mapping stays"
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Rule 2 over a surviving record: the Job's unconditional record step
// ═════════════════════════════════════════════════════════════════════════════

/// Rule 2 over a surviving record (fail-on-old). The session's record
/// survives, and so does its mapping `L → 1201` to a complete object; its
/// listed backup maps `L` to the newer 1202. The Job writes `L → 1202` and
/// restores 1202. Its record step is the applier's unconditional
/// transaction: on-access recovery's conditional script would find the
/// surviving record and write nothing. With the Job's record step sent in
/// the conditional mode, the mapping stayed `L → 1201`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_writes_a_newer_backup_mapping_into_a_session_whose_record_survives() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let newer = base(2, 0x22, &[]);
        put_session(
            store.as_ref(),
            &session_backup("s-r2", &[("r.usda", 1202)], &[]),
        )
        .await;
        put_base(store.as_ref(), 1202, "s-r2", "r.usda", 8, newer.clone()).await;
        put_counters(store.as_ref()).await;
        seed_session_record(&tv, "s-r2").await;
        seed_complete(&tv, 1201, "s-r2", "r.usda", 5, &[]).await;
        seed_mapping(&tv, "s-r2", "r.usda", 1201).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let (_plan, report, _captured) = run_to_the_end(&tv, &config, &store).await;

        assert_eq!(
            get(&tv, tv.key_schema().layer_mapping("s-r2", "r.usda")).await,
            Some("1202".to_owned()),
            "the Job writes the backup's newer mapping over the surviving one"
        );
        assert_eq!(report.sessions_restored, 1, "Step B ran");
        assert_restored(&tv, 1202, &expected_state(1202, 8, &newer, &[], 8 + BAND)).await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// A session killed between its two steps
// ═════════════════════════════════════════════════════════════════════════════

/// Killed after a session's path index (Step A) and before its record (Step
/// B), the store has a path member with no record; the re-run restores the
/// record, and ListSessions then lists the session.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_session_killed_after_its_path_index_converges() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        put_session(
            store.as_ref(),
            &session_backup("s14", &[("p.usda", 1401)], &[]),
        )
        .await;
        put_base(store.as_ref(), 1401, "s14", "p.usda", 3, base(1, 0x14, &[])).await;
        put_counters(store.as_ref()).await;
        let config = writer_config();
        hold(&tv, &config).await;
        let path = base_path_of("s14");

        run_to_the_kill(&tv, &config, &store, |b| {
            b.with_kill_point_for_test(KillPoint::SessionAfterStepA(Some("s14".to_owned())))
        })
        .await;
        assert!(
            smembers(&tv, tv.key_schema().path_index(&path))
                .await
                .contains("s14"),
            "fixture: Step A landed"
        );
        assert!(
            !exists(&tv, tv.key_schema().session_meta("s14")).await,
            "fixture: Step B did not"
        );
        let (_plan, report, _captured) = run_to_the_end(&tv, &config, &store).await;

        assert_eq!(report.sessions_restored, 1);
        let meta: std::collections::HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(tv.key_schema().session_meta("s14"))
            .await
            .expect("HGETALL");
        assert!(!meta.is_empty(), "the record is restored");
        assert!(tv.remove_hold_marker().await);
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let mut request = Request::new(ListSessionsRequest { path });
        request
            .extensions_mut()
            .insert(BearerToken::new("test-token"));
        let listed = handle_list_sessions(&svc, request)
            .await
            .expect("ListSessions answers")
            .into_inner();
        assert_eq!(
            listed
                .sessions
                .iter()
                .map(|s| s.session_id.as_str())
                .collect::<Vec<_>>(),
            vec!["s14"]
        );
    })
    .await;
}
