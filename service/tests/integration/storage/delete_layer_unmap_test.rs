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

//! `DeleteLayer`'s conditional unmap (SDD §7.11 step 2; `layer_unmap.lua`,
//! `LayerStore::delete_layer_target`): the delete removes the layer mapping
//! and its layers-index member in one atomic step, only while the mapping
//! still names the object the delete resolved; each send is written to the
//! wire at most once; once a send settled, no re-run of the retry loop sends
//! it again; and a send that follows one whose outcome is unknown never
//! retargets the delete. So a same-name `CreateLayer` that commits after the
//! unmap keeps its mapping and its member, however the rest of the delete
//! fails and re-runs.
//!
//! The replacement `Y` of the deleted object `X` is committed by a second
//! store on the same client (a second replica), from the unmap's test hook
//! (`LayerStore::with_layer_unmap_hook_for_test`) or while the retirement is
//! parked on its reference walk. The tests with no `Y` write the mapping or
//! the layers index straight to the store from the same hook, just before
//! the first send (and, in one, the second) ([`Staged`]), standing in for an
//! out-of-band write.
//!
//! | Test | Design row | What it pins |
//! |---|---|---|
//! | [`a_delete_whose_teardown_fails_after_a_same_name_create_keeps_the_replacements_mapping_and_member`] | D1 | `Y` commits right after the unmap landed, then the teardown fails once and the loop re-runs: `OK`, `Y`'s mapping, member and header intact, `X` gone, one unmap send |
//! | [`a_same_name_create_committing_during_a_deletes_retirement_stays_indexed_and_in_the_session_checkpoint`] | D2 | `Y` commits while the delete retires `X`, no fault: `Y`'s member stays and the session checkpoint the delete writes maps the layer to `Y` |
//! | [`a_replacement_committed_before_the_first_unmap_send_is_retargeted_and_deleted`] | policy 9 | `Y` commits between the mapping re-read and the first send: the delete retargets onto `Y` and deletes it, one send per target, with its INFO line |
//! | [`a_send_after_an_unmap_in_doubt_that_finds_another_object_keeps_it_and_finishes_the_delete`] | P1 | the first send lands but answers a transient error, then `Y` commits: the next send finds `Y` and does not retarget; `Y` survives, `X` is gone, with the in-doubt INFO line |
//! | [`a_delete_whose_mapping_is_gone_by_its_unmap_removes_the_member_and_retires_its_object`] | I1 | the `ABSENT` branch: the mapping deleted out of band before the first send, the member left: `OK`, the member removed, `X` retired |
//! | [`a_delete_whose_unmap_cannot_use_its_mapping_or_index_answers_internal_and_retires_nothing`] | I1 | a mapping naming no object id, a wrong-typed mapping, a wrong-typed layers index, each staged before the first send: `INTERNAL` labelled `unmap layer`, one send, both keys as staged, nothing of `X` retired |
//! | [`a_delete_whose_mapping_spells_its_object_non_canonically_unmaps_it_with_one_re_send`] | I1 | the mapping names `X` with a leading zero: one re-send expecting the stored text unmaps it, `OK`, the mapping and member gone, `X` retired, exactly two sends at one attempt; bounded, so a loop fails |
//! | [`a_delete_whose_re_send_meets_a_second_non_canonical_spelling_answers_internal_after_two_sends`] | I1 | the mapping respelled again before the re-send: `INTERNAL` naming the second spelling, exactly two sends, the mapping as staged, the member kept, nothing retired |
//! | [`an_unmap_whose_reply_is_lost_is_never_re_sent_by_the_client`] | I1 | a real lost reply (the proxy closes the connection once the first unmap's reply arrived): fred re-sends nothing, every unmap on the wire is one of the delete's own sends, each at one attempt |
//!
//! Elsewhere: the unmap refused `-MISCONF` (plain `UNAVAILABLE` labelled
//! `unmap layer`, nothing written) — `store_write_refusal_test.rs`; on a
//! full store — that file's disruptive full-store test; a teardown stopped
//! by a foreign index, which leaves the layer unmapped and its member removed
//! — `retirement_budget_test.rs` (D4-T15); the script's source hash and its
//! slot-migration reason — `script_migration_guard_test.rs`.
//!
//! # Worst case per test
//!
//! Every wait for a condition is bounded by `BUDGET` (20 s), as in
//! `retirement_budget_test.rs`, whose [`Rig`] this module uses: a test's
//! worst case is its number of such waits × `BUDGET`, plus 15 s per `Rig`
//! for its fixture's drop (5 s for the first unanswered step of the key
//! purge, then 10 s for its client's quit). The purge's answered round trips
//! come on top. The deletes under test are not bounded: a hang there is the
//! hang-killer's to report. The largest row, 75 s, stays inside nextest's
//! 120 s hang-killer.
//!
//! | Test | Bounded waits | Rigs | Worst case |
//! |---|---|---|---|
//! | `a_delete_whose_teardown_fails_after_…` | 0 | 1 | 15 s |
//! | `a_same_name_create_committing_during_…` | 3 | 1 | 75 s |
//! | `a_replacement_committed_before_…` | 1 | 1 | 35 s |
//! | `a_send_after_an_unmap_in_doubt_…` | 0 | 1 | 15 s |
//! | `a_delete_whose_mapping_is_gone_…` | 0 | 1 | 15 s |
//! | `a_delete_whose_unmap_cannot_use_…` | 0 | 3 (one per case, in turn) | 45 s |
//! | `a_delete_whose_mapping_spells_…` | 1 | 1 | 35 s |
//! | `a_delete_whose_re_send_meets_…` | 1 | 1 | 35 s |
//! | `an_unmap_whose_reply_is_lost_…` | 0 | 1 | 15 s |
//!
//! Requires Valkey (`TEST_VALKEY_URL`) and the `test-fault-injection`
//! feature: the module is compiled out without it.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use futures::future::BoxFuture;
use serde_json::json;
use tokio_util::task::TaskTracker;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{CheckpointStore, SessionCheckpoint, SessionCheckpointWriter};
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::storage::layer::{LayerStore, LayerUnmapHook, LayerUnmapHookPoint};
use live_edit::storage::lua::LAYER_UNMAP_SCRIPT;
use live_edit::storage::test_support::TestValkey;

use crate::grpc::log_capture::capture;
use crate::storage::retirement_budget_test::{hold_walk_at_obj_ext, parked, Rig, BUDGET};
use crate::storage::snapshot_read_support::{
    test_valkey_addr, ConnProxyRule, ProxyAction, ScriptCalls, ScriptReplyProxy,
};

/// The layer every test deletes and re-creates.
const LAYER: &str = "unmapped";

/// The INFO line of a delete retargeted onto a replacement it found
/// (`LayerStore::delete_layer_inner`).
const RETARGET_INFO: &str =
    "the layer was replaced before its delete unmapped it; deleting the replacement";

/// The INFO line of a send after one in doubt that finds another object
/// (`LayerStore::delete_layer_target`).
const IN_DOUBT_INFO: &str = "the layer names another object after an earlier unmap of this \
     delete whose outcome is unknown; finishing the delete of its object and keeping the other \
     one";

/// Every point the unmap hook was called at, in order.
type Points = Arc<Mutex<Vec<LayerUnmapHookPoint>>>;

/// The future a hook action answers with.
type HookAnswer = BoxFuture<'static, Option<fred::error::Error>>;

/// A hook that records every point it is called at in `points` and, the
/// first time `at` matches one, runs `action` and answers what it returns;
/// every other call answers `None` (the send goes ahead, its reply kept).
fn hook_once(
    points: &Points,
    at: fn(LayerUnmapHookPoint) -> bool,
    action: impl Fn() -> HookAnswer + Send + Sync + 'static,
) -> LayerUnmapHook {
    let points = Arc::clone(points);
    let fired = AtomicBool::new(false);
    Arc::new(move |point| {
        points.lock().unwrap().push(point);
        if at(point) && !fired.swap(true, Ordering::SeqCst) {
            action()
        } else {
            Box::pin(async { None })
        }
    })
}

/// A hook action: commit [`LAYER`] through `replica` (a same-name
/// `CreateLayer` on a second replica), record its object in `committed`, and
/// answer `answer()`.
fn commit_replacement(
    rig: &Rig,
    replica: &Arc<LayerStore>,
    committed: &Arc<OnceLock<u64>>,
    answer: fn() -> Option<fred::error::Error>,
) -> impl Fn() -> HookAnswer + Send + Sync + 'static {
    let (replica, committed) = (Arc::clone(replica), Arc::clone(committed));
    let (session_id, ext) = (rig.session_id.clone(), rig.ext.clone());
    move || {
        let (replica, committed) = (Arc::clone(&replica), Arc::clone(&committed));
        let (session_id, ext) = (session_id.clone(), ext.clone());
        Box::pin(async move {
            let y = replica
                .create_layer(&session_id, LAYER, "unmap-tester", &empty(), ext)
                .await
                .expect("the replacement's CreateLayer must succeed");
            committed
                .set(y)
                .expect("fixture: the replacement is committed once");
            answer()
        })
    }
}

/// The empty layer state every create here commits.
fn empty() -> ObjectState {
    ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        ..Default::default()
    }
}

/// A transient error standing in for a reply lost after the script ran.
fn lost_reply() -> Option<fred::error::Error> {
    Some(fred::error::Error::new(
        fred::error::ErrorKind::IO,
        "injected: the unmap's reply was lost",
    ))
}

/// The sends the hook saw: its `BeforeSend` points.
fn sends(points: &Points) -> Vec<LayerUnmapHookPoint> {
    points
        .lock()
        .unwrap()
        .iter()
        .copied()
        .filter(|p| matches!(p, LayerUnmapHookPoint::BeforeSend { .. }))
        .collect()
}

/// D1: `X`'s delete unmaps the layer; right after the send, a second replica
/// commits the same name as `Y`; then `X`'s teardown fails once (its first
/// `UNLINK`, transiently), so the retry loop re-runs its body. The re-run
/// must not unmap again: an unconditional mapping `DEL` re-run there, or a
/// layers-index `SREM` after the teardown, removes `Y`'s mapping and member,
/// and `Y`'s client was answered OK. Once the delete answers OK, the layer
/// maps to `Y`, the index holds its member, `Y`'s header stands, and nothing
/// of `X` is left but its expiring stream.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_whose_teardown_fails_after_a_same_name_create_keeps_the_replacements_mapping_and_member(
) {
    let rig = Rig::new("unmap-settled-rerun").await;
    let replica = Arc::new(rig.store());
    let committed = Arc::new(OnceLock::new());
    let points = Points::default();
    let first_unlink = Arc::new(Mutex::new(VecDeque::from([fred::error::Error::new(
        fred::error::ErrorKind::IO,
        "injected first UNLINK failure",
    )])));
    let deleter = rig
        .store()
        .with_cleanup_old_object_fault_injector_for_test(Arc::clone(&first_unlink))
        .with_layer_unmap_hook_for_test(hook_once(
            &points,
            |p| p == LayerUnmapHookPoint::AfterSend { send: 1 },
            commit_replacement(&rig, &replica, &committed, || None),
        ));
    let x = rig.layer(&rig.store(), LAYER, 1, 0).await;

    deleter
        .delete_layer(&rig.session_id, LAYER, rig.ext.clone())
        .await
        .expect("the retry loop re-runs the failed teardown and the delete answers OK");

    assert!(
        first_unlink.lock().unwrap().is_empty(),
        "fixture: the teardown took its fault, so the loop re-ran"
    );
    let y = *committed
        .get()
        .expect("fixture: the replacement committed from the hook");
    assert_eq!(
        rig.mapping(LAYER).await,
        Some(y.to_string()),
        "the replacement keeps its mapping"
    );
    assert!(
        rig.indexed(LAYER).await,
        "the replacement keeps its layers-index member"
    );
    assert!(
        rig.header_exists(y).await,
        "the replacement's object stands"
    );
    assert_eq!(
        rig.object_keys(x).await,
        Vec::<String>::new(),
        "nothing of the deleted object is left"
    );
    assert_eq!(
        sends(&points).len(),
        1,
        "the re-run sent no unmap: the first one settled"
    );
}

/// D2: `X`'s delete has unmapped the layer and is retiring `X` (parked on its
/// reference walk) when a second replica commits the same name as `Y`. No
/// fault: the delete then runs to OK. `Y`'s layers-index member stays, so the
/// session checkpoint the delete writes on success — built from the index by
/// `read_session_state`, which is private to `checkpoint::session`, so the
/// written checkpoint is how this test reads it — maps the layer to `Y`. A
/// layers-index `SREM` after the teardown would have removed `Y`'s member,
/// dropping it from that checkpoint and from `DeleteSession`'s cascade.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_same_name_create_committing_during_a_deletes_retirement_stays_indexed_and_in_the_session_checkpoint(
) {
    let rig = Rig::new("unmap-during-retirement").await;
    let checkpoints: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let tracker = TaskTracker::new();
    let writer = Arc::new(SessionCheckpointWriter::new(
        Arc::clone(&checkpoints),
        rig.tv.client().clone(),
        rig.tv.key_schema().clone(),
        tracker.clone(),
        Arc::clone(&rig.metrics),
        "unmap-tester".to_owned(),
        true,
    ));
    let deleter = rig.store().with_checkpoint_writer(writer);
    // References on `obj:ext`, so the retirement walks them and parks there.
    rig.layer(&rig.store(), LAYER, 1, 3).await;
    let walk = hold_walk_at_obj_ext(&rig.injector());

    let delete = deleter.delete_layer(&rig.session_id, LAYER, rig.ext.clone());
    tokio::pin!(delete);
    tokio::select! {
        result = &mut delete => panic!("the delete must park on its reference walk; got {result:?}"),
        () = parked(&walk) => {}
    }
    assert!(
        rig.mapping(LAYER).await.is_none(),
        "fixture: the mapping went before the retirement"
    );
    let y = rig.layer(&rig.store(), LAYER, 2, 0).await;
    walk.release();
    tokio::time::timeout(BUDGET, delete)
        .await
        .expect("the delete answers once its walk is released")
        .expect("the delete answers OK");

    assert_eq!(
        rig.mapping(LAYER).await,
        Some(y.to_string()),
        "the replacement keeps its mapping"
    );
    assert!(
        rig.indexed(LAYER).await,
        "the replacement keeps its layers-index member"
    );
    tracker.close();
    tokio::time::timeout(BUDGET, tracker.wait())
        .await
        .expect("the session checkpoint write the delete enqueued finishes");
    let blob = checkpoints
        .get_session_checkpoint(&rig.session_id)
        .await
        .expect("the in-memory store reads")
        .expect("the delete wrote a session checkpoint");
    let checkpoint = SessionCheckpoint::from_bytes(&blob.bytes).expect("a session checkpoint");
    assert_eq!(
        checkpoint.layer_mappings.get(LAYER),
        Some(&y),
        "the session checkpoint maps the layer to the replacement; got {:?}",
        checkpoint.layer_mappings
    );
}

/// Policy 9 kept: a second replica replaces `X` with `Y` between the
/// delete's mapping re-read and its first unmap send. That send finds `Y`,
/// writes nothing, and no earlier send can have landed, so the delete
/// retargets onto `Y` and deletes it, logging the retarget at INFO. The
/// layer is gone, `Y` with it, and `X` is retired by the replica's
/// overwrite.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_replacement_committed_before_the_first_unmap_send_is_retargeted_and_deleted() {
    let rig = Rig::new("unmap-retarget").await;
    let replica = Arc::new(rig.store());
    let committed = Arc::new(OnceLock::new());
    let points = Points::default();
    let deleter = rig.store().with_layer_unmap_hook_for_test(hook_once(
        &points,
        |p| matches!(p, LayerUnmapHookPoint::BeforeSend { send: 1, .. }),
        commit_replacement(&rig, &replica, &committed, || None),
    ));
    let x = rig.layer(&rig.store(), LAYER, 1, 0).await;

    let (answer, captured) =
        capture(deleter.delete_layer(&rig.session_id, LAYER, rig.ext.clone())).await;

    answer.expect("the delete answers OK, retargeted onto the replacement");
    let y = *committed
        .get()
        .expect("fixture: the replacement committed from the hook");
    assert_eq!(rig.mapping(LAYER).await, None, "the layer is gone");
    assert!(!rig.indexed(LAYER).await, "its member is gone");
    assert_eq!(
        replica.wait_for_cleanup(BUDGET).await,
        0,
        "the replica's overwrite retires the replaced object"
    );
    assert_eq!(
        (rig.object_keys(y).await, rig.object_keys(x).await),
        (Vec::<String>::new(), Vec::<String>::new()),
        "the replacement was deleted, and the replaced object retired"
    );
    let info = captured.only(RETARGET_INFO);
    assert_eq!(info["level"], "INFO", "full event: {info}");
    assert_eq!(info["fields"]["object_id"], json!(x), "full event: {info}");
    assert_eq!(
        info["fields"]["replacement_object_id"],
        json!(y),
        "full event: {info}"
    );
    assert_eq!(
        sends(&points).len(),
        2,
        "one send per target: the first found the replacement, the second unmapped it"
    );
}

/// P1: the delete's first unmap send lands, but its reply is lost (a
/// transient error in its place), and a second replica then commits the same
/// name as `Y`. The re-run's send finds `Y`; it cannot tell `Y` from a
/// replacement that committed before the delete, since its own first send
/// may have unmapped `X`, so it does not retarget: the delete finishes
/// retiring `X` and keeps `Y`, logging so at INFO. `Y`'s mapping, member and
/// header stand; nothing of `X` is left.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_send_after_an_unmap_in_doubt_that_finds_another_object_keeps_it_and_finishes_the_delete()
{
    let rig = Rig::new("unmap-in-doubt").await;
    let replica = Arc::new(rig.store());
    let committed = Arc::new(OnceLock::new());
    let points = Points::default();
    let deleter = rig.store().with_layer_unmap_hook_for_test(hook_once(
        &points,
        |p| p == LayerUnmapHookPoint::AfterSend { send: 1 },
        commit_replacement(&rig, &replica, &committed, lost_reply),
    ));
    let x = rig.layer(&rig.store(), LAYER, 1, 0).await;

    let (answer, captured) =
        capture(deleter.delete_layer(&rig.session_id, LAYER, rig.ext.clone())).await;

    answer.expect("the delete re-runs after the lost reply and answers OK");
    let y = *committed
        .get()
        .expect("fixture: the replacement committed from the hook");
    assert_eq!(
        rig.mapping(LAYER).await,
        Some(y.to_string()),
        "the replacement keeps its mapping"
    );
    assert!(
        rig.indexed(LAYER).await,
        "the replacement keeps its layers-index member"
    );
    assert!(
        rig.header_exists(y).await,
        "the replacement's object stands"
    );
    assert_eq!(
        rig.object_keys(x).await,
        Vec::<String>::new(),
        "the delete finished retiring its own object"
    );
    let info = captured.only(IN_DOUBT_INFO);
    assert_eq!(info["level"], "INFO", "full event: {info}");
    assert_eq!(info["fields"]["object_id"], json!(x), "full event: {info}");
    assert_eq!(
        info["fields"]["replacement_object_id"],
        json!(y),
        "full event: {info}"
    );
    assert!(
        captured.with_message(RETARGET_INFO).is_empty(),
        "no retarget; full capture:\n{}",
        captured.text
    );
    assert_eq!(
        sends(&points),
        [
            LayerUnmapHookPoint::BeforeSend {
                send: 1,
                max_attempts: 1
            },
            LayerUnmapHookPoint::BeforeSend {
                send: 2,
                max_attempts: 1
            },
        ],
        "two sends of the one target: the lost reply, then the one that found the replacement"
    );
}

/// What a hook action writes straight to the store just before one of the
/// delete's unmap sends (the first one comes after its mapping re-read),
/// standing in for an out-of-band write.
#[derive(Clone, Copy, Debug)]
enum Staged {
    /// The mapping deleted; the layers-index member left.
    MappingAbsent,
    /// The mapping set to a value that is no object id.
    MappingNotAnObjectId,
    /// The mapping set to the deleted object's id with a leading zero.
    MappingNonCanonical,
    /// The mapping set to the deleted object's id with two leading zeros.
    MappingNonCanonicalAgain,
    /// The mapping replaced by a hash naming the deleted object.
    MappingWrongType,
    /// The layers index replaced by a string.
    IndexWrongType,
}

/// [`Staged::MappingNotAnObjectId`]'s value.
const NOT_AN_OBJECT_ID: &str = "not-an-object-id";

/// [`Staged::IndexWrongType`]'s value.
const NOT_A_SET: &str = "not-a-set";

impl Staged {
    async fn stage(self, pool: &fred::clients::Pool, mapping: &str, index: &str, x: u64) {
        use fred::interfaces::{HashesInterface, KeysInterface};
        match self {
            Self::MappingAbsent => {
                let _: i64 = pool.del(mapping).await.expect("DEL of the mapping");
            }
            Self::MappingNotAnObjectId => {
                let _: () = pool
                    .set(mapping, NOT_AN_OBJECT_ID, None, None, false)
                    .await
                    .expect("SET of the mapping");
            }
            Self::MappingNonCanonical => {
                let _: () = pool
                    .set(mapping, format!("0{x}"), None, None, false)
                    .await
                    .expect("SET of the mapping");
            }
            Self::MappingNonCanonicalAgain => {
                let _: () = pool
                    .set(mapping, format!("00{x}"), None, None, false)
                    .await
                    .expect("SET of the mapping");
            }
            Self::MappingWrongType => {
                let _: i64 = pool.del(mapping).await.expect("DEL of the mapping");
                let _: () = pool
                    .hset(mapping, [("object_id", x.to_string())])
                    .await
                    .expect("HSET of the mapping");
            }
            Self::IndexWrongType => {
                let _: i64 = pool.del(index).await.expect("DEL of the layers index");
                let _: () = pool
                    .set(index, NOT_A_SET, None, None, false)
                    .await
                    .expect("SET of the layers index");
            }
        }
    }

    /// Assert the mapping and the layers index are as staged: the unmap
    /// wrote neither.
    async fn assert_left_as_staged(self, rig: &Rig, x: u64) {
        use fred::interfaces::{HashesInterface, KeysInterface};
        let pool = &rig.tv.client().pool;
        let keys = rig.tv.key_schema();
        let mapping = keys.layer_mapping(&rig.session_id, LAYER);
        match self {
            Self::MappingAbsent | Self::MappingNonCanonical => {
                unreachable!("{self:?} is unmapped, not refused")
            }
            Self::MappingNotAnObjectId | Self::MappingNonCanonicalAgain => {
                let want = match self {
                    Self::MappingNotAnObjectId => NOT_AN_OBJECT_ID.to_owned(),
                    _ => format!("00{x}"),
                };
                assert_eq!(
                    rig.mapping(LAYER).await,
                    Some(want),
                    "{self:?}: the mapping"
                );
                assert!(rig.indexed(LAYER).await, "{self:?}: the member");
            }
            Self::MappingWrongType => {
                let fields: std::collections::HashMap<String, String> = pool
                    .hgetall(&mapping)
                    .await
                    .expect("HGETALL of the mapping");
                assert_eq!(
                    fields,
                    std::collections::HashMap::from([("object_id".to_owned(), x.to_string())]),
                    "{self:?}: the mapping"
                );
                assert!(rig.indexed(LAYER).await, "{self:?}: the member");
            }
            Self::IndexWrongType => {
                assert_eq!(
                    rig.mapping(LAYER).await,
                    Some(x.to_string()),
                    "{self:?}: the mapping"
                );
                let index: Option<String> = pool
                    .get(keys.layers_index(&rig.session_id))
                    .await
                    .expect("GET of the layers index");
                assert_eq!(index.as_deref(), Some(NOT_A_SET), "{self:?}: the index");
            }
        }
    }
}

/// A store whose unmap hook stages `staged` just before the first send (and
/// records every hook point in `points`), on `rig`'s client.
fn staging_store(rig: &Rig, x: u64, staged: Staged, points: &Points) -> LayerStore {
    staging_store_at(rig, x, &[(1, staged)], points)
}

/// A store whose unmap hook stages each `(send, staged)` once, just before
/// that numbered send (and records every hook point in `points`), on `rig`'s
/// client.
fn staging_store_at(rig: &Rig, x: u64, staged: &[(u32, Staged)], points: &Points) -> LayerStore {
    let pool = rig.tv.client().pool.clone();
    let mapping = rig.tv.key_schema().layer_mapping(&rig.session_id, LAYER);
    let index = rig.tv.key_schema().layers_index(&rig.session_id);
    let points = Arc::clone(points);
    let staged: Vec<(u32, Staged, AtomicBool)> = staged
        .iter()
        .map(|&(at, s)| (at, s, AtomicBool::new(false)))
        .collect();
    rig.store()
        .with_layer_unmap_hook_for_test(Arc::new(move |point| {
            points.lock().unwrap().push(point);
            let stage = match point {
                LayerUnmapHookPoint::BeforeSend { send, .. } => staged
                    .iter()
                    .find(|(at, _, fired)| *at == send && !fired.swap(true, Ordering::SeqCst))
                    .map(|&(_, s, _)| s),
                LayerUnmapHookPoint::AfterSend { .. } => None,
            };
            let (pool, mapping, index) = (pool.clone(), mapping.clone(), index.clone());
            Box::pin(async move {
                if let Some(stage) = stage {
                    stage.stage(&pool, &mapping, &index, x).await;
                }
                None
            })
        }))
}

/// Assert the delete retired nothing of `x`: its header stands and its stream
/// carries no `deleted` entry (the wake-up follows the unmap).
async fn assert_nothing_retired(rig: &Rig, x: u64, context: &str) {
    assert!(
        rig.header_exists(x).await,
        "{context}: the object's header stands"
    );
    assert_eq!(
        rig.deleted_entries(x).await,
        0,
        "{context}: the object's stream was not woken"
    );
}

/// The unmap's `ABSENT` branch: the mapping is gone by the first send (an
/// out-of-band `DEL`, the member left), so the script removes the member and
/// writes nothing else; the delete answers OK and retires its object. No
/// member is left that no mapping names, for a session checkpoint to drop or
/// `DeleteSession`'s cascade to meet.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_whose_mapping_is_gone_by_its_unmap_removes_the_member_and_retires_its_object() {
    let rig = Rig::new("unmap-absent").await;
    let x = rig.layer(&rig.store(), LAYER, 1, 0).await;
    let points = Points::default();
    let deleter = staging_store(&rig, x, Staged::MappingAbsent, &points);

    deleter
        .delete_layer(&rig.session_id, LAYER, rig.ext.clone())
        .await
        .expect("the delete of a layer unmapped under it answers OK");

    assert_eq!(
        sends(&points).len(),
        1,
        "fixture: one send, after the staging"
    );
    assert_eq!(rig.mapping(LAYER).await, None, "the mapping stays gone");
    assert!(
        !rig.indexed(LAYER).await,
        "the unmap removed the member no mapping names"
    );
    assert_eq!(
        rig.object_keys(x).await,
        Vec::<String>::new(),
        "the delete retired its object"
    );
}

/// The unmap meets a mapping or a layers index it cannot use, each staged
/// out of band just before the first send: a mapping naming no object id
/// (the script answers `OTHER` with a value that is not a `u64`), a
/// wrong-typed mapping (its `GET` fails) and a wrong-typed layers index (its
/// `SREM` fails, the script's first write). Each answers `INTERNAL` labelled
/// `unmap layer`, with the mapping and the index as staged and nothing of the
/// object retired: the unmap is the delete's first write.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_whose_unmap_cannot_use_its_mapping_or_index_answers_internal_and_retires_nothing()
{
    for staged in [
        Staged::MappingNotAnObjectId,
        Staged::MappingWrongType,
        Staged::IndexWrongType,
    ] {
        let rig = Rig::new("unmap-unusable").await;
        let x = rig.layer(&rig.store(), LAYER, 1, 0).await;
        let points = Points::default();
        let deleter = staging_store(&rig, x, staged, &points);

        let answer = deleter
            .delete_layer(&rig.session_id, LAYER, rig.ext.clone())
            .await;

        match &answer {
            Err(ServiceError::Internal(message)) => assert!(
                message.contains("unmap layer"),
                "{staged:?}: the error names the unmap; got {message:?}"
            ),
            other => panic!("{staged:?}: INTERNAL; got {other:?}"),
        }
        assert_eq!(sends(&points).len(), 1, "{staged:?}: one send, not retried");
        staged.assert_left_as_staged(&rig, x).await;
        assert_nothing_retired(&rig, x, &format!("{staged:?}")).await;
    }
}

/// The two sends a non-canonical spelling costs: the first, which finds it,
/// and the one re-send expecting the stored text, each at one attempt.
const TWO_SENDS: [LayerUnmapHookPoint; 2] = [
    LayerUnmapHookPoint::BeforeSend {
        send: 1,
        max_attempts: 1,
    },
    LayerUnmapHookPoint::BeforeSend {
        send: 2,
        max_attempts: 1,
    },
];

/// The mapping names the deleted object, spelled with a leading zero (an
/// out-of-band write just before the first send). The script compares text,
/// so the first send answers `OTHER` with the stored text, which reads as
/// the deleted object itself. Neither a retarget onto it (which would
/// re-enter the delete for the same object round after round) nor a refusal
/// (which would leave the layer undeletable) follows: the delete sends the
/// unmap once more expecting the stored text, which unmaps the layer, and
/// answers OK with the mapping and member gone and its object retired,
/// after exactly two sends. Bounded by `BUDGET`, so a delete that loops
/// fails the test instead of hanging it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_whose_mapping_spells_its_object_non_canonically_unmaps_it_with_one_re_send() {
    let rig = Rig::new("unmap-non-canonical").await;
    let x = rig.layer(&rig.store(), LAYER, 1, 0).await;
    let points = Points::default();
    let deleter = staging_store(&rig, x, Staged::MappingNonCanonical, &points);

    let answer = tokio::time::timeout(
        BUDGET,
        deleter.delete_layer(&rig.session_id, LAYER, rig.ext.clone()),
    )
    .await
    .expect("the delete answers rather than retargeting onto its own object round after round");

    assert!(
        answer.is_ok(),
        "the re-send unmaps the layer and the delete answers OK; got {answer:?}"
    );
    assert_eq!(
        (rig.mapping(LAYER).await, rig.indexed(LAYER).await),
        (None, false),
        "the re-send removed the mapping and the layers-index member"
    );
    assert_eq!(
        rig.object_keys(x).await,
        Vec::<String>::new(),
        "the delete retired its object"
    );
    assert_eq!(
        sends(&points),
        TWO_SENDS,
        "the first send and one re-send, each at one attempt"
    );
}

/// The re-send meets a second non-canonical spelling of the deleted object
/// (the mapping rewritten to two leading zeros just before it). There is no
/// third send: the delete answers `INTERNAL` labelled `unmap layer`, naming
/// the second spelling, after exactly two sends, with the mapping as staged,
/// the member kept and nothing of the object retired. Bounded by `BUDGET`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_whose_re_send_meets_a_second_non_canonical_spelling_answers_internal_after_two_sends(
) {
    let rig = Rig::new("unmap-non-canonical-again").await;
    let x = rig.layer(&rig.store(), LAYER, 1, 0).await;
    let points = Points::default();
    let deleter = staging_store_at(
        &rig,
        x,
        &[
            (1, Staged::MappingNonCanonical),
            (2, Staged::MappingNonCanonicalAgain),
        ],
        &points,
    );

    let answer = tokio::time::timeout(
        BUDGET,
        deleter.delete_layer(&rig.session_id, LAYER, rig.ext.clone()),
    )
    .await
    .expect("the delete answers rather than sending on round after round");

    match &answer {
        Err(ServiceError::Internal(message)) => assert!(
            message.contains("unmap layer") && message.contains(&format!("\"00{x}\"")),
            "the error names the unmap and the second spelling; got {message:?}"
        ),
        other => panic!("INTERNAL; got {other:?}"),
    }
    assert_eq!(sends(&points), TWO_SENDS, "two sends, no third");
    Staged::MappingNonCanonicalAgain
        .assert_left_as_staged(&rig, x)
        .await;
    assert_nothing_retired(&rig, x, "second non-canonical spelling").await;
}

/// I1, written at most once: a real lost reply. The proxy forwards the
/// delete's first unmap, lets the script run (it is loaded first), and closes
/// the connection once
/// its reply arrives, without delivering it. The client must not re-send
/// the script behind the delete's back — a re-sent run could follow a
/// same-name `CreateLayer` and be read as a first run that found a
/// replacement — so the send fails, and the delete's own re-run sends it
/// again. Every unmap call the proxy saw is therefore one of the delete's
/// own sends, each sent at one attempt, and the delete answers OK with the
/// layer gone. At fred's default attempts the client re-sends the first call
/// itself: the proxy sees two calls for the delete's one send.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unmap_whose_reply_is_lost_is_never_re_sent_by_the_client() {
    const POOL_SIZE: u32 = 4;
    let calls = Arc::new(ScriptCalls::new(LAYER_UNMAP_SCRIPT.sha1().to_string()));
    let mapping_key: Arc<Mutex<Option<String>>> = Arc::default();
    let closed = Arc::new(AtomicBool::new(false));
    let rule: ConnProxyRule = {
        let (calls, mapping_key, closed) = (
            Arc::clone(&calls),
            Arc::clone(&mapping_key),
            Arc::clone(&closed),
        );
        Arc::new(move |_conn: usize, args: &[Vec<u8>]| {
            let call = calls.see(args)?;
            let key = mapping_key.lock().unwrap().clone()?;
            (call == 0
                && args.iter().any(|a| a.as_slice() == key.as_bytes())
                && !closed.swap(true, Ordering::SeqCst))
            .then_some(ProxyAction::CloseAfterReply)
        })
    };
    let proxy = ScriptReplyProxy::start_rewriting_replies(test_valkey_addr(), rule).await;
    let rig = Rig::on(
        TestValkey::setup_with_url(proxy.url(), POOL_SIZE).await,
        "unmap-lost-reply",
    )
    .await;
    let points = Points::default();
    let deleter = rig.store().with_layer_unmap_hook_for_test(hook_once(
        &points,
        |_| false,
        || Box::pin(async { None }),
    ));
    rig.layer(&rig.store(), LAYER, 1, 0).await;
    // Loaded up front, so the first call runs rather than meeting `NOSCRIPT`
    // (a concurrent `SCRIPT FLUSH` may still empty the cache: the counts
    // below hold either way).
    LAYER_UNMAP_SCRIPT
        .load(rig.tv.client().pool.next())
        .await
        .expect("SCRIPT LOAD of the unmap");
    *mapping_key.lock().unwrap() = Some(rig.tv.key_schema().layer_mapping(&rig.session_id, LAYER));

    deleter
        .delete_layer(&rig.session_id, LAYER, rig.ext.clone())
        .await
        .expect("the delete re-runs after the lost reply and answers OK");

    assert!(
        closed.load(Ordering::SeqCst),
        "fixture: the proxy lost the first unmap's reply"
    );
    let sends = sends(&points);
    assert_eq!(
        sends,
        [
            LayerUnmapHookPoint::BeforeSend {
                send: 1,
                max_attempts: 1
            },
            LayerUnmapHookPoint::BeforeSend {
                send: 2,
                max_attempts: 1
            },
        ],
        "the lost reply failed the first send, and the delete sent again, each at one attempt"
    );
    assert_eq!(
        calls.calls(),
        sends.len(),
        "every unmap call on the wire is one of the delete's own sends: the client re-sent none"
    );
    assert_eq!(rig.mapping(LAYER).await, None, "the layer is gone");
    assert!(!rig.indexed(LAYER).await, "its member is gone");
}
