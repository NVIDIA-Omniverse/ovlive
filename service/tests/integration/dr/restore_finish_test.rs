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

//! The restore Job's finish end to end (`live_edit::dr::job::finish`,
//! `live_edit::dr::job::restore`; SDD §22.22, SR-9.8, SR-9.9, SR-14.6): the
//! Job run through its frame against a real store and a scripted backup
//! store, from its preconditions to its exit code, with the authoritative
//! recount, the restore summary and the un-hold — the real one, no test kill
//! point before it unless a test names one.
//!
//! One named test per Epic restore acceptance criterion this Task closes;
//! each test's doc line starts with its ID. E6, E8 and E10 are end to end in
//! Tasks 6 and 8 (the restore writer's and planner's tests).
//!
//! - **E1**, a full wipe: exit 0, every layer as backed up, every value's
//!   count its references, the counter above every term, the marker gone.
//! - **E2**, a partial wipe: only what was lost is restored; a newer
//!   survivor is kept as it is.
//! - **E3**, a big layer through the writer's batch seam: several
//!   transactions, then the finish (the 100,001-entry layer itself is W2's).
//! - **E4**, kills: mid-recount, before the un-hold and after it, each
//!   converging (Task 8's kill points converge in `restore_write_test` and
//!   `restore_sessions_test`).
//! - **E5**, a new incident recorded before the un-hold: exit 4, nothing
//!   deleted.
//! - **E7**, stays held: exit 3 with the restorable layers written and the
//!   counts recounted; with the missing value supplied, the next run exits 0.
//! - **E9**, a lost `{hash}` slot: its references counted, with no data.
//! - **E-catchup**, a value only the backup store holds: its first catch-up
//!   fetches it and leaves its count, with no stamp.
//! - **E-rerun**, a run after a successful one: exit 0, nothing written.
//! - **E-ids**, a layer created after the un-hold gets an id above every term.
//! - **E-summary**, the summary lines' fields equal the writer's report, and
//!   the fetch count the number of `Backup` verdicts.
//! - **PM S2**, held replicas exit on the un-hold, and a rebuilt server takes
//!   an update on a kept survivor whose retirement marks were cleared.
//! - **defer-6(c)**, a StorageApi backup store answering 404 for an empty
//!   prefix: three full runs, each exit 0.
//! - **defer-62**, killed after an abandoned delete's wipe: the re-run's
//!   counter clears every dead-set id.
//! - **ADD 9 (a), (b)**: a recount that fails on a corrupt record, and a
//!   write after the recount, each leave the marker (the single-deleter scan
//!   is `tests/misc/hold_unhold_scan.rs`); a recount whose walk is not shown
//!   complete — a header one of its two object walks skips, or an `obj:ext`
//!   field its counted walk skips (b.f3t, through the walk seam) — exits 1,
//!   leaves the marker and sets no count, and its re-run exits 0; one whose
//!   walks read a placeholder or changing run_id runs on and exits 0.
//! - **The un-hold's refusals**: an object-id counter that is not a decimal
//!   integer, and a counter or marker of the wrong type, each exit 3 and
//!   clear nothing (the script's other answers are pinned against the store
//!   in `storage::dr_hold`'s tests).
//! - **The counter floor**: the sidecar term and the surviving term each
//!   bind when largest; a floor above the largest allocatable id exits 3;
//!   a counters record that does not parse exits 3 before any write (PD-7).
//!
//! Every run uses a per-test key prefix (`TestValkey`).
//!
//! # Fail-on-old
//!
//! Each pin below was proven red by an in-place perturbation of the code it
//! guards, then green once restored: defer-62 with the counter floor's dead
//! term dropped (the counter lands at 101, below the dead object 5,000);
//! ADD 9 (a) with the object walk passing over an `obj:ext` field it cannot
//! decode (the recount completes and the marker is deleted); ADD 9 (b) with
//! the finish's check for a write after the recount removed (the marker is
//! deleted); the non-decimal counter with `dr_unhold.lua`'s arm for it
//! replaced by the script error it had (exit 1); the key of the wrong type
//! with the finish's `WRONGTYPE` arm removed (exit 1); the recount's skipped
//! members with `scan_guard::keyspace_walks_agree` and `check_count`
//! answering `Ok` (the recount completes and the Job exits 0); the run_id
//! cases with `KeyspaceWalk::start_job` reading its `INFO server` witness
//! again (exit 3, or exit 1 for the change).
//!
//! # Time
//!
//! Every test runs under `writer_support`'s one-test budget (its module
//! docs, *Time*): 90 s plus the store clean-up, inside nextest's 120 s kill.
//! A healthy test of these fixtures took under half a second on the remote
//! build host (2026-10-21; each Job run bounded by `RUN_BUDGET`, 30 s, and
//! every run of a test, however many it makes, by the one test budget), PM
//! S2 about 4 s: it also waits for each held
//! replica's exit, one poll and its jitter, within
//! `hold_support::EXIT_DEADLINE` (20 s), inside the same budget.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface, ListInterface, SortedSetsInterface};
use futures::StreamExt;
use tonic::Request;

use live_edit::checkpoint::CheckpointStore;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::dr::job::execute::{KillPoint, WriteKind};
use live_edit::dr::job::finish::{FinishRecord, MAX_OBJECT_ID};
use live_edit::dr::job::plan::model::{CoverageReport, HashVerdict, ObjectAction, PlanOutcome};
use live_edit::dr::job::{run_job, JobExit, JobMode, RestoreBody};
use live_edit::grpc::layer::handle_update_layer;
use live_edit::grpc::proto::{subscribe_read_layer_response, SubscribeReadLayerRequest};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::storage::dr_hold::UnholdOutcome;
use live_edit::storage::external_values::ValkeyExternalValueStore;
use live_edit::storage::keys::{sha256_to_hex, KeySchema};
use live_edit::storage::test_support::{
    block_on_spawned, write_hold_marker, HoldMarkerSeed, TestValkey,
};

use super::hold_support::{startup_held, HeldModeRun, EXIT_DEADLINE};
use super::support::{keyspace, keyspace_writes, WriteRules};
use super::writer_support::*;
use crate::checkpoint::adapter_storage_api_test::{
    build_fake_backed_adapter, mount_fake_storage_routes, FakeStorageState,
};
use crate::grpc::helpers::{
    build_service, create_session_and_layer, make_request, make_update_data, make_update_header,
    streaming_update, AllowAllPermissions,
};

// ═════════════════════════════════════════════════════════════════════════════
// Shared fixtures and readers
// ═════════════════════════════════════════════════════════════════════════════

/// How the un-hold's INFO (text 9) starts.
const UNHELD_LINE: &str = "the restore Job cleared the hold for incident";
/// How the corrupt-record recount's ERROR (text 4) starts.
const CORRUPT_RECOUNT_LINE: &str =
    "the restore Job cannot recount the references of the large values: the backing store holds \
     a record it cannot read";
/// How the write-after-recount refusal (text 13) starts.
const WRITE_AFTER_RECOUNT_LINE: &str =
    "the restore Job did not clear the hold: a write ran after its recount";
/// How the different-incident refusal (text 11) starts.
const OTHER_INCIDENT_LINE: &str = "held under a different incident";
/// How a restored layer's summary line (text 5) starts.
const LAYER_SUMMARY_LINE: &str = "layer ";
/// How the summary's totals line (text 7) starts.
const TOTALS_LINE: &str = "the restore Job's summary:";
/// The incident E5 records while the Job runs.
const OTHER_INCIDENT: &str = "20261020t1300-newer";
/// How the un-hold's ERROR on a non-decimal object-id counter starts.
const COUNTER_UNREADABLE_LINE: &str =
    "the restore Job cannot clear the hold for incident 20261020t1200-restore: the object-id \
     counter";
/// How the un-hold's ERROR on a key of the wrong type starts.
const UNHOLD_WRONGTYPE_LINE: &str =
    "the restore Job cannot clear the hold for incident 20261020t1200-restore: one of the keys \
     it reads to clear it holds the wrong type of value";
/// How the counter floor's ERROR above the largest allocatable id (text 14)
/// starts.
const FLOOR_ABOVE_MAX_LINE: &str = "the restore Job cannot clear the hold: the object-id counter \
                                    would pass 9007199254740991";
/// The ERROR of a counters record that does not parse (PD-7, text 15).
const SIDECAR_UNREADABLE_LINE: &str =
    "the restore Job wrote nothing and exits 3; the service stays held: the backup store's \
     counters record cannot be read";

/// What a run's finish observer recorded.
type Finished = Arc<Mutex<Vec<FinishRecord>>>;
/// What a run's plan observer recorded: the finalised coverage report.
type Covered = Arc<Mutex<Vec<CoverageReport>>>;

/// The restore body with `seams` applied, recording what the writer did,
/// the finalised coverage report (unless `seams` sets a plan observer of
/// its own) and what the finish reached.
fn finishing_body(
    seams: impl FnOnce(RestoreBody) -> RestoreBody,
) -> (RestoreBody, Observed, Covered, Finished) {
    let finished: Finished = Arc::new(Mutex::new(Vec::new()));
    let covered: Covered = Arc::new(Mutex::new(Vec::new()));
    let (finish_sink, cover_sink) = (Arc::clone(&finished), Arc::clone(&covered));
    // The seams go on last, so a test's own plan observer replaces this one.
    let (body, observed) = observed_body(|b| {
        seams(
            b.with_finish_observer(Arc::new(move |record| {
                finish_sink.lock().unwrap().push(record.clone());
            }))
            .with_plan_observer(Arc::new(move |_plan, report| {
                cover_sink.lock().unwrap().push(report.clone());
            })),
        )
    });
    (body, observed, covered, finished)
}

/// The one record the finish observer got.
fn the_finish(finished: &Finished) -> FinishRecord {
    let finished = finished.lock().unwrap();
    assert_eq!(finished.len(), 1, "the finish was reached once");
    finished[0].clone()
}

/// The one finalised coverage report.
fn the_coverage(covered: &Covered) -> CoverageReport {
    let covered = covered.lock().unwrap();
    assert_eq!(covered.len(), 1, "the coverage check was finalised once");
    covered[0].clone()
}

/// Run the Job once with `seams` and no kill point: its exit, its log and
/// what its observers recorded.
async fn run_once(
    tv: &TestValkey,
    config: &live_edit::config::Config,
    store: &Arc<WriterStore>,
    seams: impl FnOnce(RestoreBody) -> RestoreBody,
) -> (
    JobExit,
    crate::grpc::log_capture::Captured,
    Observed,
    Covered,
    Finished,
) {
    let (body, observed, covered, finished) = finishing_body(seams);
    let (exit, captured) = run(tv, config, store, body).await;
    (exit, captured, observed, covered, finished)
}

/// The one captured line whose message starts with `prefix`.
fn the_line<'c>(
    captured: &'c crate::grpc::log_capture::Captured,
    prefix: &str,
) -> &'c serde_json::Value {
    let found: Vec<&serde_json::Value> = captured
        .events
        .iter()
        .filter(|e| {
            e["fields"]["message"]
                .as_str()
                .is_some_and(|m| m.starts_with(prefix))
        })
        .collect();
    assert_eq!(
        found.len(),
        1,
        "one line starting {prefix:?}:\n{}",
        captured.text
    );
    found[0]
}

/// Whether the hold marker exists.
async fn marker_held(tv: &TestValkey) -> bool {
    exists(tv, tv.key_schema().dr_hold_marker()).await
}

/// The object-id counter, `None` when absent.
async fn counter(tv: &TestValkey) -> Option<u64> {
    get(tv, tv.key_schema().counter_object_id())
        .await
        .map(|v| v.parse().expect("a decimal counter"))
}

/// `hash`'s `ref_count`, `None` when its meta is absent.
async fn ref_count(tv: &TestValkey, hash: &[u8; 32]) -> Option<u64> {
    let raw: Option<String> = tv
        .client()
        .pool
        .hget(tv.key_schema().ext_meta(&sha256_to_hex(hash)), "ref_count")
        .await
        .expect("HGET ref_count");
    raw.map(|v| v.parse().expect("a decimal count"))
}

/// Whether `hash`'s meta carries a `last_store_ms` stamp.
async fn stamped(tv: &TestValkey, hash: &[u8; 32]) -> bool {
    tv.client()
        .pool
        .hexists(
            tv.key_schema().ext_meta(&sha256_to_hex(hash)),
            "last_store_ms",
        )
        .await
        .expect("HEXISTS last_store_ms")
}

/// Whether Valkey holds `hash`'s bytes.
async fn data_held(tv: &TestValkey, hash: &[u8; 32]) -> bool {
    exists(tv, tv.key_schema().ext_data(&sha256_to_hex(hash))).await
}

/// Put `bytes` into the backup store as the value `hash`.
async fn put_value(store: &Arc<WriterStore>, hash: &[u8; 32], bytes: &[u8]) {
    store
        .put_ext_blob(&sha256_to_hex(hash), Bytes::from(bytes.to_vec()))
        .await
        .expect("the value put succeeds");
}

/// A value of `len` bytes, told apart by `tag`, and its content hash.
fn value_of(tag: u8, len: usize) -> ([u8; 32], Vec<u8>) {
    let bytes: Vec<u8> = (0..len)
        .map(|i| (i as u8).wrapping_mul(17).wrapping_add(tag))
        .collect();
    (
        live_edit::storage::external_values::ext_content_hash(&bytes),
        bytes,
    )
}

/// A bare hash, told apart by `tag` and `i`, for a value whose bytes the
/// test never reads.
fn hash_of(tag: u8, i: u8) -> [u8; 32] {
    let mut hash = [tag; 32];
    hash[0] = i;
    hash
}

/// One field row naming `hash`, its index member and an `obj:ext` field
/// counting it, on the surviving object `object_id`: the references the
/// recount's object walk reads.
async fn seed_survivor_value(tv: &TestValkey, object_id: u64, hash: &[u8; 32]) {
    let oid = object_id.to_string();
    let hex = sha256_to_hex(hash);
    let member = KeySchema::field_member("2", "default");
    hset(
        tv,
        tv.key_schema().obj_field_from_member(&oid, &member),
        &[("hash", hex.as_str()), ("size", "4096")],
    )
    .await;
    let _: i64 = tv
        .client()
        .pool
        .zadd(
            tv.key_schema().obj_fields_index(&oid),
            None,
            None,
            false,
            false,
            (1.0, member.as_str()),
        )
        .await
        .expect("ZADD obj:flds");
    hset(
        tv,
        tv.key_schema().obj_ext_index(&oid),
        &[(hex.as_str(), "1")],
    )
    .await;
}

/// The partial-wipe fixture (I2's shape): session [`PARTIAL`] with
/// `kept.usda` → [`KEPT`], a survivor at seq 50 newer than its backup at 40,
/// and `lost.usda` → [`LOST`], whose header the incident lost, backed up at
/// 30; the session record and both mappings survive.
async fn partial_wipe(tv: &TestValkey, store: &Arc<WriterStore>) -> Bytes {
    let lost = base(2, 0x02, &[]);
    put_session(
        store.as_ref(),
        &session_backup(PARTIAL, &[("kept.usda", KEPT), ("lost.usda", LOST)], &[]),
    )
    .await;
    put_base(
        store.as_ref(),
        KEPT,
        PARTIAL,
        "kept.usda",
        40,
        base(1, 0x01, &[]),
    )
    .await;
    put_base(store.as_ref(), LOST, PARTIAL, "lost.usda", 30, lost.clone()).await;
    put_counters(store.as_ref()).await;
    seed_session_record(tv, PARTIAL).await;
    seed_complete(tv, KEPT, PARTIAL, "kept.usda", 50, &[]).await;
    seed_mapping(tv, PARTIAL, "kept.usda", KEPT).await;
    seed_mapping(tv, PARTIAL, "lost.usda", LOST).await;
    lost
}

/// The partial-wipe session, its kept survivor and its lost layer.
const PARTIAL: &str = "s-partial";
const KEPT: u64 = 101;
const LOST: u64 = 102;

/// A unary request carrying the test bearer.
fn bearer<T>(message: T) -> Request<T> {
    let mut request = Request::new(message);
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

// ═════════════════════════════════════════════════════════════════════════════
// E1: a full wipe
// ═════════════════════════════════════════════════════════════════════════════

/// The full-wipe fixture: sessions `s-a` (`a1.usda` → 101 and `a2.usda` →
/// 102) and `s-b` (`b1.usda` → 201), every layer backed up. 101 and 201
/// reference a value only the backup store holds (`shared`), 102 another
/// (`own`), and 201's tail carries a third (`carried`), which the writer
/// backfills into Valkey.
struct FullWipe {
    bases: [(u64, &'static str, &'static str, Bytes, u64); 3],
    b1_tail: Bytes,
    b1_head: u64,
    shared: [u8; 32],
    own: [u8; 32],
    carried: [u8; 32],
}

async fn full_wipe(store: &Arc<WriterStore>) -> FullWipe {
    let (shared, shared_bytes) = value_of(0x51, 64);
    let (own, own_bytes) = value_of(0x52, 64);
    let (carried_hash, carried_bytes) = carried(b"a value only b1's tail carries");
    let a1 = base(3, 0xA1, &[shared]);
    let a2 = base(1, 0xA2, &[own]);
    let b1 = base(2, 0xB1, &[shared]);
    let (b1_tail, b1_head) = tail(201, 9, &[(carried_hash, carried_bytes)]);
    put_session(
        store.as_ref(),
        &session_backup("s-a", &[("a1.usda", 101), ("a2.usda", 102)], &[]),
    )
    .await;
    put_session(
        store.as_ref(),
        &session_backup("s-b", &[("b1.usda", 201)], &[]),
    )
    .await;
    put_base(store.as_ref(), 101, "s-a", "a1.usda", 5, a1.clone()).await;
    put_base(store.as_ref(), 102, "s-a", "a2.usda", 7, a2.clone()).await;
    put_base(store.as_ref(), 201, "s-b", "b1.usda", 9, b1.clone()).await;
    put_tail(
        store.as_ref(),
        201,
        "s-b",
        "b1.usda",
        9,
        b1_head,
        b1_tail.clone(),
    )
    .await;
    put_value(store, &shared, &shared_bytes).await;
    put_value(store, &own, &own_bytes).await;
    put_counters(store.as_ref()).await;
    FullWipe {
        bases: [
            (101, "s-a", "a1.usda", a1, 5),
            (102, "s-a", "a2.usda", a2, 7),
            (201, "s-b", "b1.usda", b1, 9),
        ],
        b1_tail,
        b1_head,
        shared,
        own,
        carried: carried_hash,
    }
}

/// **E1.** After a full wipe, with every session and layer backed up, the
/// Job writes every layer as its backup decodes, recounts, logs its summary
/// and clears the hold: exit 0, the marker gone. Every value's count is its
/// references (nothing is owed after a full wipe): `shared` 2, `own` 1,
/// `carried` 1; the two only the backup store holds have no data (fetched
/// on first read). The counter is the floor, above every term: the listed
/// ids and the sidecar plus the margin, and every restored id plus 1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e1_restore_job_after_a_full_wipe_restores_recounts_and_clears_the_hold() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let fixture = full_wipe(&store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let (exit, captured, observed, _covered, finished) =
            run_once(&tv, &config, &store, |b| b).await;

        assert_eq!(exit, JobExit::Done, "{}", captured.text);
        assert!(!marker_held(&tv).await, "the hold is cleared");
        assert_eq!(lines_starting(&captured, UNHELD_LINE), 1);
        let (_plan, report) = the_execution(&observed);
        for (oid, _session, _layer, base_bytes, base_seq) in &fixture.bases {
            let tails = if *oid == 201 {
                vec![fixture.b1_tail.clone()]
            } else {
                vec![]
            };
            let head = if *oid == 201 {
                fixture.b1_head
            } else {
                *base_seq
            };
            assert_restored(
                &tv,
                *oid,
                &expected_state(*oid, *base_seq, base_bytes, &tails, head + BAND),
            )
            .await;
            assert!(report.objects.contains_key(oid), "object {oid} written");
        }
        assert_eq!(ref_count(&tv, &fixture.shared).await, Some(2));
        assert_eq!(ref_count(&tv, &fixture.own).await, Some(1));
        assert_eq!(ref_count(&tv, &fixture.carried).await, Some(1));
        assert!(!data_held(&tv, &fixture.shared).await && !data_held(&tv, &fixture.own).await);
        assert!(
            data_held(&tv, &fixture.carried).await,
            "the backfill stored it"
        );

        let record = the_finish(&finished);
        let floor = record.floor.expect("the floor was computed");
        let margin = config.checkpoint_counter_safety_margin;
        assert_eq!(floor.listed, Some(201 + margin));
        assert_eq!(floor.sidecar, Some(margin), "the counters record reads 0");
        assert_eq!(
            floor.surviving,
            Some(202),
            "the restored headers' largest id + 1"
        );
        assert_eq!(floor.floor, 201 + margin);
        assert_eq!(counter(&tv).await, Some(floor.floor));
        assert_eq!(
            record.unhold,
            Some(UnholdOutcome::Unheld {
                old: 0,
                new: floor.floor
            })
        );
    })
    .await;
}

/// **E-rerun.** A run after a successful one finds the store not held:
/// exit 0, and it writes nothing — every key as it was, and nothing put to
/// the backup store.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e_rerun_restore_job_after_a_successful_run_exits_0_and_writes_nothing() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        full_wipe(&store).await;
        let config = writer_config();
        hold(&tv, &config).await;
        let (exit, captured, ..) = run_once(&tv, &config, &store, |b| b).await;
        assert_eq!(exit, JobExit::Done, "fixture: {}", captured.text);
        let before = keyspace(&tv).await;
        let puts_before = store.ext_puts.lock().unwrap().len();

        let (exit, captured, observed, ..) = run_once(&tv, &config, &store, |b| b).await;

        assert_eq!(exit, JobExit::NothingToDo, "{}", captured.text);
        assert!(observed.lock().unwrap().is_empty(), "no writer ran");
        assert_eq!(
            keyspace_writes(&before, &keyspace(&tv).await, WriteRules::STRICT),
            Vec::<String>::new()
        );
        assert_eq!(store.ext_puts.lock().unwrap().len(), puts_before);
    })
    .await;
}

/// **E-ids.** A layer created after the un-hold gets an object id above
/// every counter term: the sidecar's and the listed ids' (with the margin),
/// and every restored id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e_ids_a_layer_created_after_the_unhold_gets_an_id_above_every_term() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        full_wipe(&store).await;
        let config = writer_config();
        hold(&tv, &config).await;
        let (exit, captured, _observed, _covered, finished) =
            run_once(&tv, &config, &store, |b| b).await;
        assert_eq!(exit, JobExit::Done, "fixture: {}", captured.text);
        let floor = the_finish(&finished).floor.expect("the floor");

        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let (_session, oid) = create_session_and_layer(&tv, &svc, "after", "new.usda").await;

        for (term, value) in floor.terms() {
            assert!(
                oid > value,
                "the new id {oid} is above the {term} term {value}"
            );
        }
        assert!(oid > 201, "and above every restored id");
    })
    .await;
}

/// **E-summary.** The summary's per-layer lines carry what the writer
/// reported for each restored layer — its seeded `seq`, its band and its
/// transaction count — and where its values are: 101 and 102 each one
/// value only the backup store holds (`shared`, `own`), 201 that one
/// (`shared`) and the one its tail carried, which the writer stored
/// (`carried`). The totals line counts 3 layers restored and 3 values, its
/// fetch count is the number of final coverage verdicts `Backup` (here 2:
/// `shared` and `own`), and the recount created those two values' metas.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e_summary_lines_carry_the_writers_report_and_the_backup_only_count() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        full_wipe(&store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let (exit, captured, observed, covered, _finished) =
            run_once(&tv, &config, &store, |b| b).await;

        assert_eq!(exit, JobExit::Done, "{}", captured.text);
        let (_plan, report) = the_execution(&observed);
        let layers: Vec<&serde_json::Value> = captured
            .events
            .iter()
            .filter(|e| {
                e["fields"]["message"]
                    .as_str()
                    .is_some_and(|m| m.starts_with(LAYER_SUMMARY_LINE))
                    && e["fields"]["seeded_seq"].is_u64()
            })
            .collect();
        assert_eq!(
            layers.len(),
            report.objects.len(),
            "one line per restored layer"
        );
        for line in layers {
            let oid = line["fields"]["object_id"].as_u64().expect("object_id");
            let write = &report.objects[&oid];
            let fields = &line["fields"];
            assert_eq!(
                (
                    fields["seeded_seq"].as_u64(),
                    fields["band"].as_u64(),
                    fields["batches"].as_u64()
                ),
                (
                    Some(write.seeded_seq),
                    Some(write.band),
                    Some(write.batches.len() as u64)
                ),
                "object {oid}: the summary line is the writer's report"
            );
            let expected = match oid {
                201 => (1, 1),
                _ => (0, 1),
            };
            assert_eq!(
                (
                    fields["values_in_store"].as_u64(),
                    fields["values_backup_only"].as_u64(),
                    fields["values_uncovered"].as_u64(),
                    fields["values_unknown"].as_u64()
                ),
                (Some(expected.0), Some(expected.1), Some(0), Some(0)),
                "object {oid}: its values in the store and only in the backup store"
            );
        }
        let backup_verdicts = the_coverage(&covered)
            .verdicts
            .values()
            .filter(|v| matches!(v, HashVerdict::Backup))
            .count() as u64;
        assert_eq!(
            backup_verdicts, 2,
            "fixture: two values only the backup store holds"
        );
        let totals: Vec<&serde_json::Value> = captured
            .events
            .iter()
            .filter(|e| {
                e["fields"]["message"]
                    .as_str()
                    .is_some_and(|m| m.starts_with(TOTALS_LINE))
            })
            .collect();
        assert_eq!(totals.len(), 1, "one totals line");
        let fields = &totals[0]["fields"];
        assert_eq!(
            fields["fetch_on_first_read"].as_u64(),
            Some(backup_verdicts)
        );
        assert_eq!(
            (
                fields["restored"].as_u64(),
                fields["values"].as_u64(),
                fields["recount_values"].as_u64(),
                fields["recount_created"].as_u64()
            ),
            (Some(3), Some(3), Some(3), Some(2)),
            "3 layers restored, 3 values; the recount set 3 and created the two backup-only \
             values' metas"
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// E2, E3: a partial wipe, a big layer
// ═════════════════════════════════════════════════════════════════════════════

/// **E2.** A partial wipe: only the lost layer is restored, the newer
/// survivor is kept with its header untouched, and the hold is cleared.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2_restore_job_after_a_partial_wipe_restores_only_the_lost_layer_and_clears_the_hold() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let lost = partial_wipe(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;
        let kept_before = header(&tv, KEPT).await;

        let (exit, captured, observed, ..) = run_once(&tv, &config, &store, |b| b).await;

        assert_eq!(exit, JobExit::Done, "{}", captured.text);
        let (plan, report) = the_execution(&observed);
        assert_eq!(
            report.objects.keys().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([LOST]),
            "only the lost layer is written"
        );
        assert!(plan
            .objects
            .iter()
            .any(|o| o.object_id == KEPT && matches!(o.action, ObjectAction::Keep)));
        assert_eq!(header(&tv, KEPT).await, kept_before, "the survivor is kept");
        assert_restored(&tv, LOST, &expected_state(LOST, 30, &lost, &[], 30 + BAND)).await;
        assert!(!marker_held(&tv).await);
    })
    .await;
}

/// **E3.** A layer the writer writes in several transactions (its batch
/// seam at 12 commands), then the finish: exit 0, its content as backed
/// up, the hold cleared. (The 100,001-entry layer at the production batch
/// size is W2's, `restore_write_test`.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e3_restore_job_writes_a_layer_in_several_batches_then_clears_the_hold() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let big = base(40, 0xE3, &[]);
        put_session(
            store.as_ref(),
            &session_backup("s-big", &[("big.usda", 301)], &[]),
        )
        .await;
        put_base(store.as_ref(), 301, "s-big", "big.usda", 11, big.clone()).await;
        put_counters(store.as_ref()).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let (exit, captured, observed, ..) =
            run_once(&tv, &config, &store, |b| b.with_batch_commands_for_test(12)).await;

        assert_eq!(exit, JobExit::Done, "{}", captured.text);
        let (_plan, report) = the_execution(&observed);
        assert!(
            report.objects[&301].batches.len() > 2,
            "several transactions: {:?}",
            report.objects[&301].batches
        );
        assert_restored(&tv, 301, &expected_state(301, 11, &big, &[], 11 + BAND)).await;
        assert!(!marker_held(&tv).await);
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// E4: kills in the finish
// ═════════════════════════════════════════════════════════════════════════════

/// How many values E4's layer references: more than one recount group (64),
/// so the mid-recount kill point fires.
const MANY: u8 = 70;

/// E4's fixture: one layer referencing [`MANY`] values, each only in the
/// backup store, so the recount creates [`MANY`] metas in two groups.
async fn many_values(store: &Arc<WriterStore>) -> Vec<[u8; 32]> {
    let hashes: Vec<[u8; 32]> = (0..MANY).map(|i| hash_of(0xE4, i)).collect();
    put_session(
        store.as_ref(),
        &session_backup("s-many", &[("many.usda", 401)], &[]),
    )
    .await;
    put_base(
        store.as_ref(),
        401,
        "s-many",
        "many.usda",
        3,
        base(1, 0xE4, &hashes),
    )
    .await;
    for hash in &hashes {
        put_value(store, hash, b"backup-only").await;
    }
    put_counters(store.as_ref()).await;
    hashes
}

/// How many of `hashes` have a meta, and whether each that has one counts 1.
async fn metas_set(tv: &TestValkey, hashes: &[[u8; 32]]) -> usize {
    let mut set = 0;
    for hash in hashes {
        if let Some(count) = ref_count(tv, hash).await {
            assert_eq!(count, 1, "a set count is exact");
            set += 1;
        }
    }
    set
}

/// What an uninterrupted run leaves, after a converging re-run: every count
/// set, the counter at the floor, the marker gone.
async fn assert_converged(
    tv: &TestValkey,
    hashes: &[[u8; 32]],
    config: &live_edit::config::Config,
) {
    assert_eq!(metas_set(tv, hashes).await, hashes.len(), "every count set");
    assert!(!marker_held(tv).await, "the hold is cleared");
    assert_eq!(
        counter(tv).await,
        Some(401 + config.checkpoint_counter_safety_margin),
        "the counter at the floor (listed 401 + margin)"
    );
}

/// **E4, killed mid-recount** (after its first group of 64 count sets): exit
/// 1, some counts set and exact, none past, the marker held and the counter
/// untouched; the re-run recounts from scratch and clears the hold.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e4_restore_job_killed_mid_recount_converges() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let hashes = many_values(&store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let (exit, captured, _observed, _covered, finished) = run_once(&tv, &config, &store, |b| {
            b.with_kill_point_for_test(KillPoint::RecountAfterFirstChunk)
        })
        .await;

        assert_eq!(exit, JobExit::Failed, "{}", captured.text);
        assert_eq!(lines_starting(&captured, KILL_LINE), 1);
        assert_eq!(the_finish(&finished).recount, None, "no receipt");
        let set = metas_set(&tv, &hashes).await;
        assert!(0 < set && set < hashes.len(), "partly set: {set}");
        assert!(marker_held(&tv).await);
        assert_eq!(counter(&tv).await, None);

        let (exit, captured, ..) = run_once(&tv, &config, &store, |b| b).await;
        assert_eq!(exit, JobExit::Done, "the re-run: {}", captured.text);
        assert_converged(&tv, &hashes, &config).await;
    })
    .await;
}

/// **E4, killed before the un-hold** (after the recount and the summary):
/// exit 1, every count exact, the marker held and the counter untouched;
/// the re-run recounts to the same values and clears the hold.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e4_restore_job_killed_before_the_unhold_converges() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let hashes = many_values(&store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let (exit, captured, ..) = run_once(&tv, &config, &store, |b| {
            b.with_kill_point_for_test(KillPoint::BeforeUnhold)
        })
        .await;

        assert_eq!(exit, JobExit::Failed, "{}", captured.text);
        assert_eq!(lines_starting(&captured, KILL_LINE), 1);
        assert_eq!(metas_set(&tv, &hashes).await, hashes.len());
        assert!(marker_held(&tv).await);
        assert_eq!(counter(&tv).await, None);

        let (exit, captured, ..) = run_once(&tv, &config, &store, |b| b).await;
        assert_eq!(exit, JobExit::Done, "the re-run: {}", captured.text);
        assert_converged(&tv, &hashes, &config).await;
    })
    .await;
}

/// **E4, killed after the un-hold** (the script ran; its line not logged):
/// exit 1 with the hold already cleared and the counter raised; the re-run
/// finds the store not held, exits 0 and writes nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e4_restore_job_killed_after_the_unhold_converges() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let hashes = many_values(&store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let (exit, captured, ..) = run_once(&tv, &config, &store, |b| {
            b.with_kill_point_for_test(KillPoint::AfterUnhold)
        })
        .await;

        assert_eq!(exit, JobExit::Failed, "{}", captured.text);
        assert_eq!(lines_starting(&captured, KILL_LINE), 1);
        assert_eq!(
            lines_starting(&captured, UNHELD_LINE),
            0,
            "its line is lost"
        );
        assert_converged(&tv, &hashes, &config).await;
        let before = keyspace(&tv).await;

        let (exit, captured, ..) = run_once(&tv, &config, &store, |b| b).await;
        assert_eq!(exit, JobExit::NothingToDo, "the re-run: {}", captured.text);
        assert_eq!(
            keyspace_writes(&before, &keyspace(&tv).await, WriteRules::STRICT),
            Vec::<String>::new()
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// E5, E7, E9
// ═════════════════════════════════════════════════════════════════════════════

/// **E5.** A new incident recorded while the Job runs (here once its
/// coverage check is finalised, before the recount): the un-hold script
/// finds the marker naming it, writes nothing and the Job exits 4. The
/// marker stays, naming the new incident, and the counter is untouched.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e5_restore_job_whose_incident_changed_before_the_unhold_exits_4_and_deletes_nothing() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        partial_wipe(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;
        let (client, keys) = (tv.client().clone(), tv.key_schema().clone());

        let (exit, captured, _observed, _covered, finished) =
            run_once(&tv, &config, &store, move |b| {
                b.with_plan_observer(Arc::new(move |_plan, _report| {
                    let (client, keys) = (client.clone(), keys.clone());
                    block_on_spawned(&tokio::runtime::Handle::current(), async move {
                        write_hold_marker(
                            &client,
                            &keys,
                            &HoldMarkerSeed::default().with_incident(OTHER_INCIDENT),
                        )
                        .await;
                    })
                    .expect("the new incident is recorded");
                }))
            })
            .await;

        assert_eq!(exit, JobExit::DifferentIncident, "{}", captured.text);
        assert_eq!(lines_starting(&captured, OTHER_INCIDENT_LINE), 1);
        assert_eq!(
            the_finish(&finished).unhold,
            Some(UnholdOutcome::OtherIncident {
                held: OTHER_INCIDENT.to_owned()
            })
        );
        let marker: Option<String> = tv
            .client()
            .pool
            .hget(tv.key_schema().dr_hold_marker(), "incident")
            .await
            .expect("HGET incident");
        assert_eq!(marker.as_deref(), Some(OTHER_INCIDENT), "the marker stays");
        assert_eq!(counter(&tv).await, None, "the counter is untouched");
    })
    .await;
}

/// **E7.** A layer referencing a value no store holds stays held: the Job
/// still writes both layers and recounts (the restorable value's count and
/// the missing one's, from the layers that name them), then exits 3 with
/// the marker kept. With the missing value supplied to the backup store,
/// the next run exits 0 and clears the hold.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e7_restore_job_stays_held_on_an_uncovered_value_then_exits_0_once_it_is_supplied() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let (present, present_bytes) = value_of(0x71, 64);
        let (missing, missing_bytes) = value_of(0x72, 64);
        put_session(
            store.as_ref(),
            &session_backup("s-7", &[("ok.usda", 701), ("gap.usda", 702)], &[]),
        )
        .await;
        put_base(
            store.as_ref(),
            701,
            "s-7",
            "ok.usda",
            4,
            base(1, 0x71, &[present]),
        )
        .await;
        put_base(
            store.as_ref(),
            702,
            "s-7",
            "gap.usda",
            6,
            base(1, 0x72, &[missing]),
        )
        .await;
        put_value(&store, &present, &present_bytes).await;
        put_counters(store.as_ref()).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let (exit, captured, observed, _covered, finished) =
            run_once(&tv, &config, &store, |b| b).await;

        assert_eq!(exit, JobExit::StaysHeld, "{}", captured.text);
        assert_eq!(lines_starting(&captured, STAYS_HELD_LINE), 1);
        let record = the_finish(&finished);
        assert_eq!(record.outcome, PlanOutcome::StaysHeld);
        assert!(record.recount.is_some(), "it recounted");
        assert_eq!(record.unhold, None);
        let (_plan, report) = the_execution(&observed);
        assert_eq!(
            report.objects.keys().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([701, 702]),
            "both layers written"
        );
        assert_eq!(ref_count(&tv, &present).await, Some(1));
        assert_eq!(ref_count(&tv, &missing).await, Some(1));
        assert!(marker_held(&tv).await);
        assert_eq!(counter(&tv).await, None);

        put_value(&store, &missing, &missing_bytes).await;
        let (exit, captured, ..) = run_once(&tv, &config, &store, |b| b).await;

        assert_eq!(
            exit,
            JobExit::Done,
            "with the value supplied: {}",
            captured.text
        );
        assert!(!marker_held(&tv).await);
        assert_eq!(
            ref_count(&tv, &missing).await,
            Some(1),
            "the same exact count"
        );
    })
    .await;
}

/// **E9.** A value whose `{hash}` slot the incident lost, while two kept
/// survivors still name it (rows and `obj:ext`) and the backup store holds
/// its copy: the Job exits 0, and its recount gives the value a meta at 2,
/// one per survivor, with no data — the meta that keeps the collect after
/// the un-hold from deleting the copy (T1, `checkpoint::checkpointer`'s
/// `reconcile_tests`) and that its first read fetches against (E-catchup).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e9_restore_job_counts_a_value_whose_slot_was_lost_with_no_data() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        partial_wipe(&tv, &store).await;
        let (lost_slot, bytes) = value_of(0x91, 64);
        const SECOND: u64 = 103;
        put_session(
            store.as_ref(),
            &session_backup(
                PARTIAL,
                &[
                    ("kept.usda", KEPT),
                    ("lost.usda", LOST),
                    ("kept2.usda", SECOND),
                ],
                &[],
            ),
        )
        .await;
        put_base(
            store.as_ref(),
            SECOND,
            PARTIAL,
            "kept2.usda",
            40,
            base(1, 0x03, &[]),
        )
        .await;
        seed_complete(&tv, SECOND, PARTIAL, "kept2.usda", 50, &[]).await;
        seed_mapping(&tv, PARTIAL, "kept2.usda", SECOND).await;
        for oid in [KEPT, SECOND] {
            seed_survivor_value(&tv, oid, &lost_slot).await;
        }
        put_value(&store, &lost_slot, &bytes).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let (exit, captured, ..) = run_once(&tv, &config, &store, |b| b).await;

        assert_eq!(exit, JobExit::Done, "{}", captured.text);
        assert_eq!(
            ref_count(&tv, &lost_slot).await,
            Some(2),
            "one per survivor"
        );
        assert!(!data_held(&tv, &lost_slot).await, "with no data");
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// E-catchup
// ═════════════════════════════════════════════════════════════════════════════

/// **E-catchup.** After a full wipe, a value only the backup store holds,
/// named by two restored layers, is in Valkey as a meta at 2 (one per
/// layer, the recount's exact count) with no data. The first catch-up
/// (`SubscribeReadLayer` from nothing) of one layer fetches it from the
/// backup store through `recover_value` and delivers it; its count stays
/// 2, its meta gets no `last_store_ms` stamp, and its bytes are now stored.
///
/// Fail-on-old (in-place perturbation: `store_recovered_values` on the real
/// store writing back in the ordinary store mode instead of the recover
/// mode, red then green once restored): the write-back pays `+1` and a
/// fresh stamp, so the count reads 3 and the stamp is there.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e_catchup_a_backup_only_values_first_catch_up_fetches_it_and_keeps_its_count() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let (hash, bytes) = value_of(0xC1, 4096);
        put_session(
            store.as_ref(),
            &session_backup("s-c", &[("c.usda", 501), ("d.usda", 502)], &[]),
        )
        .await;
        put_base(
            store.as_ref(),
            501,
            "s-c",
            "c.usda",
            8,
            base(1, 0xC1, &[hash]),
        )
        .await;
        put_base(
            store.as_ref(),
            502,
            "s-c",
            "d.usda",
            8,
            base(1, 0xC2, &[hash]),
        )
        .await;
        put_value(&store, &hash, &bytes).await;
        put_counters(store.as_ref()).await;
        let config = writer_config();
        hold(&tv, &config).await;
        let (exit, captured, ..) = run_once(&tv, &config, &store, |b| b).await;
        assert_eq!(exit, JobExit::Done, "fixture: {}", captured.text);
        assert_eq!(
            (
                ref_count(&tv, &hash).await,
                data_held(&tv, &hash).await,
                stamped(&tv, &hash).await
            ),
            (Some(2), false, false),
            "fixture: a meta at 2, one per layer, without data or a stamp"
        );

        let base_svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let ext = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
            .with_metrics(Arc::clone(&base_svc.metrics))
            .with_checkpoint_store(Arc::clone(&store) as Arc<dyn CheckpointStore>);
        let svc = LiveEditService {
            ext_value_store: Arc::new(ext),
            ..base_svc
        };
        let carried = first_catch_up_values(&svc, "s-c", "c.usda").await;

        assert!(carried.contains(&bytes), "the catch-up delivers the value");
        assert_eq!(ref_count(&tv, &hash).await, Some(2), "its count is kept");
        assert!(
            !stamped(&tv, &hash).await,
            "the recover write-back writes no `last_store_ms` stamp"
        );
        assert!(data_held(&tv, &hash).await, "and its bytes are stored");
    })
    .await;
}

/// The external-value parts of the first content message a catch-up from
/// nothing delivers on `layer`.
async fn first_catch_up_values(svc: &LiveEditService, session: &str, layer: &str) -> Vec<Vec<u8>> {
    let mut stream = handle_subscribe_read_layer(
        svc,
        bearer(SubscribeReadLayerRequest {
            session_id: session.to_owned(),
            layer_name: layer.to_owned(),
            object_id: 0,
            sequence_num: 0,
            client_id: "restore-finish-subscriber".to_owned(),
        }),
    )
    .await
    .expect("SubscribeReadLayer opens")
    .into_inner();
    let first = tokio::time::timeout(RUN_BUDGET, stream.next())
        .await
        .expect("a message within the budget")
        .expect("the stream sends a message")
        .expect("a header, not a status");
    let header = match first.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(header)) => header,
        other => panic!("expected a header, got {other:?}"),
    };
    let mut body = Vec::new();
    while (body.len() as u64) < header.total_size {
        let next = tokio::time::timeout(RUN_BUDGET, stream.next())
            .await
            .expect("a chunk within the budget")
            .expect("the stream sends the body")
            .expect("a chunk, not a status");
        match next.response_type {
            Some(subscribe_read_layer_response::ResponseType::Chunk(chunk)) => {
                body.extend_from_slice(&chunk.data)
            }
            other => panic!("expected a body chunk, got {other:?}"),
        }
    }
    OmpfContainer::parse(&body)
        .expect("the content parses as OMPF")
        .external_values
        .iter()
        .map(|part| part.to_vec())
        .collect()
}

// ═════════════════════════════════════════════════════════════════════════════
// PM S2: held replicas exit on the un-hold
// ═════════════════════════════════════════════════════════════════════════════

/// The flags a crash left on PM S2's survivor.
const FLAGS: [&str; 3] = ["superseded", "superseded_by", "ext_gc_scheduled"];

/// **PM S2.** Two replicas run held mode while the Job restores a kept
/// survivor whose retirement marks a crash left set (W6's shape); its
/// summary names the survivor's flag clear (text 6). The Job's
/// real un-hold (exit 0) makes each held replica call its exit seam with 0
/// within `EXIT_DEADLINE`; a server built again then takes an `UpdateLayer`
/// on the survivor, whose marks the Job cleared.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pm_s2_held_replicas_exit_on_the_unhold_and_a_rebuilt_server_takes_an_update() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let (session, oid) = create_session_and_layer(&tv, &svc, "pm-s2", "c.usda").await;
        drop(svc);
        let header_key = tv.key_schema().obj_header(&oid.to_string());
        let seq: u64 = header(&tv, oid).await["seq"].parse().expect("a seq");
        hset(
            &tv,
            header_key,
            &[
                ("superseded", "1"),
                ("superseded_by", "999999"),
                ("ext_gc_scheduled", "1"),
            ],
        )
        .await;
        put_session(
            store.as_ref(),
            &session_backup(&session, &[("c.usda", oid)], &[]),
        )
        .await;
        put_counters(store.as_ref()).await;
        let config = writer_config();
        hold(&tv, &config).await;
        let mut pods = Vec::new();
        for _ in 0..2 {
            let startup = startup_held(&tv).await;
            pods.push(HeldModeRun::start(tv.client().clone(), tv.key_schema(), startup).await);
        }

        let (exit, captured, observed, ..) = run_once(&tv, &config, &store, |b| b).await;

        assert_eq!(exit, JobExit::Done, "{}", captured.text);
        let (_plan, report) = the_execution(&observed);
        let write = &report.objects[&oid];
        assert_eq!(write.kind, WriteKind::FlagsCleared);
        assert_eq!(
            lines(
                &captured,
                &format!(
                    "layer c.usda of session {session} (object {oid}): kept its surviving copy \
                     and cleared its retirement marks; resumes at sequence {} (band {})",
                    write.seeded_seq, write.band
                )
            )
            .len(),
            1,
            "the summary's flag-clear line (text 6):\n{}",
            captured.text
        );
        for (i, pod) in pods.iter_mut().enumerate() {
            assert_eq!(
                pod.next_exit(EXIT_DEADLINE, &format!("held replica {i}"))
                    .await,
                0,
                "held replica {i} exits 0 to restart normally"
            );
        }
        let h = header(&tv, oid).await;
        for flag in FLAGS {
            assert!(!h.contains_key(flag), "{flag} cleared: {h:?}");
        }
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let ompf = crate::common::owned_delta::delta_to_ompf_bytes(
            &crate::common::owned_delta::empty_delta(),
            &[],
        );
        handle_update_layer(
            &svc,
            make_request(streaming_update(vec![
                make_update_header(&session, "c.usda", oid),
                make_update_data(ompf),
            ])),
        )
        .await
        .unwrap_or_else(|s| panic!("an UpdateLayer commits on the cleared survivor: {s}"));
        assert_eq!(
            header(&tv, oid).await["seq"],
            (seq + BAND + 1).to_string(),
            "the update committed at the next seq"
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// defer-6(c): a StorageApi backup store answering 404 for an empty prefix
// ═════════════════════════════════════════════════════════════════════════════

/// **defer-6(c)**, PM S6. Over the StorageApi fake, which answers an
/// Enumerate of a prefix holding nothing with 404 as storage-api does, three
/// full runs each reach exit 0 and clear the hold: (i) session backups with
/// no layer backup; (ii) layer backups with no session backup (the session
/// record surviving); (iii) the counters record alone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn defer_6c_restore_job_on_a_storage_api_store_with_an_empty_prefix_exits_0() {
    within_test_budget(async {
        for case in [
            "(i) sessions without layers",
            "(ii) layers without sessions",
            "(iii) the counters record only",
        ] {
            let tv = TestValkey::setup().await;
            let server = wiremock::MockServer::start().await;
            let state: FakeStorageState = Arc::new(Mutex::new(Default::default()));
            mount_fake_storage_routes(&server, &state).await;
            let (adapter, _token_dir) = build_fake_backed_adapter(&server, "restore-finish-6c");
            let store: Arc<dyn CheckpointStore> = Arc::new(adapter);
            put_counters(store.as_ref()).await;
            match case {
                "(i) sessions without layers" => {
                    put_session(store.as_ref(), &session_backup("s-i", &[], &[])).await;
                }
                "(ii) layers without sessions" => {
                    put_base(store.as_ref(), 801, "s-ii", "l.usda", 5, base(1, 0x80, &[])).await;
                    seed_session_record(&tv, "s-ii").await;
                    seed_mapping(&tv, "s-ii", "l.usda", 801).await;
                }
                _ => {}
            }
            let config = writer_config();
            hold(&tv, &config).await;
            let (body, _observed, _covered, finished) = finishing_body(|b| b);
            let metrics = metrics();

            let (exit, captured) = captured_within(
                RUN_BUDGET,
                run_job(
                    context(&tv, JobMode::Restore, config, store, &metrics),
                    body,
                ),
            )
            .await;

            assert_eq!(exit, JobExit::Done, "{case}: {}", captured.text);
            assert_eq!(
                the_finish(&finished).outcome,
                PlanOutcome::Writable,
                "{case}"
            );
            assert!(!marker_held(&tv).await, "{case}: the hold is cleared");
        }
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// defer-62: the dead set's counter term
// ═════════════════════════════════════════════════════════════════════════════

/// The abandoned delete's object: above every listed id, with no listed
/// backup of its own.
const DEAD: u64 = 5_000;

/// **defer-62** (fail-on-old). With a margin of 1, no counter, a counters
/// record of 10 and listed ids at most 100, an abandoned delete's object
/// [`DEAD`] (no listed backup) is registered dead and wiped, then the run is
/// killed (`DeleteAfterWipe`): exit 1, [`DEAD`] on the dead set, its header
/// gone. The re-run exits 0, and the counter clears every dead-set id: it
/// is [`DEAD`] + 1, the dead term (the others are 11, 101 and 101). With the
/// dead term dropped the counter lands at 101, below [`DEAD`]: a later
/// layer could take a dead id, and the reconcile reap its backups.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn defer_62_restore_job_killed_after_a_deletes_wipe_raises_the_counter_past_every_dead_id() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        put_session(
            store.as_ref(),
            &session_backup("s-live", &[("l.usda", 100)], &[]),
        )
        .await;
        put_base(
            store.as_ref(),
            100,
            "s-live",
            "l.usda",
            3,
            base(1, 0x62, &[]),
        )
        .await;
        put_session(
            store.as_ref(),
            &session_backup("sd", &[("d1.usda", DEAD)], &[]),
        )
        .await;
        store
            .put_counters_sidecar(Bytes::from_static(
                b"{\"max_object_id_observed\":10,\"updated_at\":\"2026-10-20T12:00:00Z\"}",
            ))
            .await
            .expect("the counters record put succeeds");
        seed_deletion_marker(&tv, "sd").await;
        seed_complete(&tv, DEAD, "sd", "d1.usda", 10, &[]).await;
        seed_mapping(&tv, "sd", "d1.usda", DEAD).await;
        let mut config = writer_config();
        config.checkpoint_counter_safety_margin = 1;
        hold(&tv, &config).await;

        run_to_the_kill(&tv, &config, &store, |b| {
            b.with_kill_point_for_test(KillPoint::DeleteAfterWipe(Some("sd".to_owned())))
        })
        .await;
        assert_eq!(
            dead_set(&tv).await,
            BTreeSet::from([DEAD]),
            "at the kill: registered"
        );
        assert!(
            header(&tv, DEAD).await.is_empty(),
            "at the kill: its header wiped"
        );
        assert_eq!(counter(&tv).await, None, "fixture: no counter");

        let (exit, captured, _observed, _covered, finished) =
            run_once(&tv, &config, &store, |b| b).await;

        assert_eq!(exit, JobExit::Done, "the re-run: {}", captured.text);
        let counter = counter(&tv).await.expect("the un-hold set the counter");
        assert!(
            counter > DEAD,
            "the counter {counter} clears every dead-set id (the largest is {DEAD})"
        );
        assert_eq!(counter, DEAD + 1);
        assert_eq!(
            the_finish(&finished).floor.and_then(|f| f.dead),
            Some(DEAD + 1)
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// ADD 9: no un-hold without a completed recount, nor after a write since
// ═════════════════════════════════════════════════════════════════════════════

/// **ADD 9 (a), PD-4** (fail-on-old). A writable plan whose kept survivor's
/// `obj:ext` holds a field that is not a hash (the planner reads the index
/// leniently): the recount cannot read it and fails, so the Job logs the
/// corrupt-record ERROR, whose `key` names the object and its `obj:ext`
/// ahead of the field, and exits 3 with no receipt and no un-hold — the
/// marker present, the counter untouched, no un-hold line. With the walk
/// passing over the field instead, the recount completes and the marker is
/// deleted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn add9_a_restore_job_whose_recount_meets_a_corrupt_record_exits_3_and_keeps_the_marker() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        partial_wipe(&tv, &store).await;
        hset(
            &tv,
            tv.key_schema().obj_ext_index(&KEPT.to_string()),
            &[("not-a-hash", "1")],
        )
        .await;
        let config = writer_config();
        hold(&tv, &config).await;

        let (exit, captured, _observed, _covered, finished) =
            run_once(&tv, &config, &store, |b| b).await;

        assert!(
            marker_held(&tv).await,
            "the marker stays: {}",
            captured.text
        );
        assert_eq!(exit, JobExit::StaysHeld);
        assert_eq!(
            the_line(&captured, CORRUPT_RECOUNT_LINE)["fields"]["key"],
            format!(
                "{} (object {KEPT}): not-a-hash",
                tv.key_schema().obj_ext_index(&KEPT.to_string())
            ),
            "the line names the object and its key ahead of the field it cannot read"
        );
        assert_eq!(lines_starting(&captured, UNHELD_LINE), 0);
        let record = the_finish(&finished);
        assert_eq!(
            record.outcome,
            PlanOutcome::Writable,
            "the plan was writable"
        );
        assert_eq!((record.recount, record.unhold), (None, None));
        assert_eq!(counter(&tv).await, None, "the counter is untouched");
    })
    .await;
}

/// **ADD 9 (b)** (fail-on-old). A write unit run between the recount and
/// the un-hold (`with_write_after_recount_for_test`): the finish refuses the
/// un-hold — one ERROR (text 13), exit 1 — the marker stays and the counter
/// is untouched. With the check removed, the marker is deleted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn add9_b_restore_job_with_a_write_after_its_recount_exits_1_and_keeps_the_marker() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        partial_wipe(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let (exit, captured, _observed, _covered, finished) = run_once(&tv, &config, &store, |b| {
            b.with_write_after_recount_for_test()
        })
        .await;

        assert!(
            marker_held(&tv).await,
            "the marker stays: {}",
            captured.text
        );
        assert_eq!(exit, JobExit::Failed);
        assert_eq!(lines_starting(&captured, WRITE_AFTER_RECOUNT_LINE), 1);
        assert_eq!(the_finish(&finished).unhold, None);
        assert_eq!(counter(&tv).await, None, "the counter is untouched");
    })
    .await;
}

/// The walk seam's faults armed once the plan is made, so they reach only
/// the walks the recount then starts (the faults' key or pattern carries the
/// test's prefix, and the planner's own walks ran before the plan): the
/// guards, held by the caller until the run ends.
#[cfg(feature = "test-fault-injection")]
type ArmedAfterPlan = Arc<Mutex<Vec<live_edit::storage::scan_guard::ArmedWalkFault>>>;

/// `seams`' body with `arm`'s faults armed when the plan observer fires,
/// after the planner's walks and before the recount's.
#[cfg(feature = "test-fault-injection")]
fn arming_after_plan(
    arm: impl Fn() -> Vec<live_edit::storage::scan_guard::ArmedWalkFault> + Send + Sync + 'static,
) -> (ArmedAfterPlan, impl FnOnce(RestoreBody) -> RestoreBody) {
    let armed: ArmedAfterPlan = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&armed);
    let seams = move |b: RestoreBody| {
        b.with_plan_observer(Arc::new(move |_plan, _report| {
            sink.lock().unwrap().extend(arm());
        }))
    };
    (armed, seams)
}

/// Drop `key` from walk number `walk` of `pattern` (counted from when the
/// faults are armed), whichever step returns it: a keyspace walk of a shared
/// store answers the test's key at a step no test can know in advance, so
/// the seam's `DropMember` names the key on every step of the walk
/// (`step: None`). A drop that never fires leaves the walks agreeing, and
/// the test that expects them to disagree fails.
#[cfg(feature = "test-fault-injection")]
fn drop_key(
    pattern: &str,
    walk: usize,
    key: &str,
) -> Vec<live_edit::storage::scan_guard::ArmedWalkFault> {
    use live_edit::storage::scan_guard::{arm_walk_fault_for_test, WalkFault};
    vec![arm_walk_fault_for_test(
        pattern,
        WalkFault::DropMember {
            walk,
            step: None,
            member: Some(key.to_owned()),
        },
    )]
}

/// How the recount's ERROR on a store fault or a walk not shown complete
/// starts (text 3).
#[cfg(feature = "test-fault-injection")]
const RECOUNT_FAILED_LINE: &str =
    "the restore Job could not recount the references of the large values";

/// **The recount's completeness proofs** (b.f3t T2; fail-on-old). The
/// restore of E4's 70 backup-only values, with one member skipped once the
/// plan is made (the walk seam's `DropMember`): the restored object's header
/// from the recount's first walk of the object headers, then from its
/// second, then a field of its `obj:ext` from the counted `HSCAN` walk. Each
/// run logs the recount's ERROR, saying the backing store changed while the
/// Job read it, naming the walk and what it compared, and that it set no
/// count and exits 1 so Kubernetes retries it; the Job exits 1 with no
/// receipt and no un-hold, the marker present, the counter untouched and no
/// count set. A re-run with the seam disarmed exits 0 with every count
/// exact. Needs `test-fault-injection` (the seam).
///
/// Fail-on-old (in-place perturbations, red then green once restored): with
/// `scan_guard::keyspace_walks_agree` answering `Ok` whatever the two sets,
/// both header cases' recounts complete and the Job exits 0; with
/// `scan_guard::check_count` answering `Ok` whatever the counts, the
/// `obj:ext` case's does (the two run together, each case caught only by
/// its own walk's check).
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_recount_whose_walk_skips_a_member_exits_1_keeps_the_marker_and_reruns() {
    use live_edit::storage::scan_guard::{arm_walk_fault_for_test, WalkFault};
    within_test_budget(async {
        let keys = |tv: &TestValkey| tv.key_schema().clone();
        // (what, how the skip is armed once the plan is made, what the
        // ERROR names)
        type Arm = fn(&KeySchema) -> Vec<live_edit::storage::scan_guard::ArmedWalkFault>;
        type Names = fn(&KeySchema) -> String;
        let cases: [(&str, Arm, Names); 3] = [
            (
                "the first walk of the object headers",
                |keys| drop_key(&keys.obj_header_scan_pattern(), 0, &keys.obj_header("401")),
                |keys| {
                    format!(
                        "two SCAN walks of {} returned different keys: 0 and 1 distinct keys",
                        keys.obj_header_scan_pattern()
                    )
                },
            ),
            (
                "the second walk of the object headers",
                |keys| drop_key(&keys.obj_header_scan_pattern(), 1, &keys.obj_header("401")),
                |keys| {
                    format!(
                        "two SCAN walks of {} returned different keys: 1 and 0 distinct keys",
                        keys.obj_header_scan_pattern()
                    )
                },
            ),
            (
                "the counted walk of obj:ext",
                |keys| {
                    vec![arm_walk_fault_for_test(
                        &keys.obj_ext_index("401"),
                        WalkFault::DropMember {
                            walk: 0,
                            step: Some(0),
                            member: None,
                        },
                    )]
                },
                |keys| {
                    format!(
                        "the HSCAN walk of {} returned {} members, and HLEN read {}",
                        keys.obj_ext_index("401"),
                        MANY - 1,
                        MANY
                    )
                },
            ),
        ];
        // A case whose Job did not exit 1 is collected, so one run names
        // every case a regression reaches; the rest are asserted at once.
        let mut not_failed: Vec<String> = Vec::new();
        for (what, arm, names) in cases {
            let tv = TestValkey::setup().await;
            let store = WriterStore::new();
            let hashes = many_values(&store).await;
            let config = writer_config();
            hold(&tv, &config).await;
            let (armed, seams) = arming_after_plan({
                let keys = keys(&tv);
                move || arm(&keys)
            });

            let (exit, captured, _observed, _covered, finished) =
                run_once(&tv, &config, &store, seams).await;
            armed.lock().unwrap().clear();

            if exit != JobExit::Failed {
                not_failed.push(format!("{what}: exit {exit:?}, not 1"));
                continue;
            }
            let line = the_line(&captured, RECOUNT_FAILED_LINE);
            assert_eq!(line["level"], "ERROR", "{what}");
            let message = line["fields"]["message"].as_str().unwrap_or_default();
            for part in [
                "the backing store changed while the Job read it".to_owned(),
                names(tv.key_schema()),
                "it set no count and exits 1 so Kubernetes retries it".to_owned(),
            ] {
                assert!(
                    message.contains(&part),
                    "{what}: the line names {part:?}: {message}"
                );
            }
            assert_eq!(lines_starting(&captured, CORRUPT_RECOUNT_LINE), 0, "{what}");
            let record = the_finish(&finished);
            assert_eq!((record.recount, record.unhold), (None, None), "{what}");
            assert!(marker_held(&tv).await, "{what}: the marker stays");
            assert_eq!(counter(&tv).await, None, "{what}: the counter is untouched");
            assert_eq!(metas_set(&tv, &hashes).await, 0, "{what}: no count is set");

            let (exit, captured, ..) = run_once(&tv, &config, &store, |b| b).await;
            assert_eq!(exit, JobExit::Done, "{what}: the re-run: {}", captured.text);
            assert_converged(&tv, &hashes, &config).await;
        }
        assert!(
            not_failed.is_empty(),
            "a skipped member exits 1:\n{}",
            not_failed.join("\n")
        );
    })
    .await;
}

/// **The recount on a store with no usable or a changing run_id** (b.f3t
/// T6; b.gns test 9's Job consumer before). The recount reads no
/// serving-process witness, so the same restore with every walk of the
/// recount forged, once the plan is made, to read a placeholder run_id
/// (`"0"`), no `run_id` line, or a change of run_id (the walk seam's `RunId`
/// fault on the first witness read, or a change at the second) exits 0
/// with every count exact and the hold cleared. Needs `test-fault-injection`
/// (the seam).
///
/// Fail-on-old (in-place perturbation, red then green once restored): with
/// `KeyspaceWalk::start_job` beginning a guarded walk again (its
/// `INFO server` witness put back), the placeholder cases exit 3 and the
/// change exits 1.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_recount_on_a_store_whose_run_id_is_unusable_or_changes_clears_the_hold() {
    use live_edit::storage::scan_guard::{
        arm_walk_fault_for_test, forge_process_change_for_test, ArmedWalkFault, WalkFault,
    };
    within_test_budget(async {
        type Forge = fn(&str) -> ArmedWalkFault;
        let cases: [(&str, Forge); 3] = [
            ("run_id \"0\"", |target| {
                arm_walk_fault_for_test(
                    target,
                    WalkFault::RunId {
                        read: 0,
                        run_id: Some("0".to_owned()),
                    },
                )
            }),
            ("no run_id line", |target| {
                arm_walk_fault_for_test(
                    target,
                    WalkFault::RunId {
                        read: 0,
                        run_id: None,
                    },
                )
            }),
            ("a change of run_id", |target| {
                forge_process_change_for_test(target, 1)
            }),
        ];
        let mut not_done: Vec<String> = Vec::new();
        for (case, forge) in cases {
            let tv = TestValkey::setup().await;
            let store = WriterStore::new();
            let hashes = many_values(&store).await;
            let config = writer_config();
            hold(&tv, &config).await;
            let keys = tv.key_schema();
            // Every walk of the recount: the objects, the restored object's
            // value indices, the carriers, `pending` and `inflight`.
            let targets = [
                keys.obj_header_scan_pattern(),
                keys.obj_fields_index("401"),
                keys.obj_time_samples_index("401"),
                keys.obj_ext_index("401"),
                format!("{}{{*}}:obj:extret", keys.prefix()),
                keys.ext_pending(),
                keys.ext_inflight(),
            ];
            let (armed, seams) = arming_after_plan(move || {
                targets
                    .iter()
                    .map(|target| forge(target.as_str()))
                    .collect()
            });

            let (exit, captured, ..) = run_once(&tv, &config, &store, seams).await;
            armed.lock().unwrap().clear();

            // A case whose Job did not exit 0 is collected, so one run names
            // every case a regression reaches.
            if exit != JobExit::Done {
                let errors: Vec<&str> = captured
                    .events
                    .iter()
                    .filter(|e| e["level"] == "ERROR")
                    .filter_map(|e| e["fields"]["message"].as_str())
                    .collect();
                not_done.push(format!("{case}: exit {}, not 0: {errors:?}", exit.code()));
                continue;
            }
            assert_converged(&tv, &hashes, &config).await;
        }
        assert!(not_done.is_empty(), "{}", not_done.join("\n"));
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The un-hold's refusals: an unreadable counter, a key of the wrong type
// ═════════════════════════════════════════════════════════════════════════════

/// Set `key` to the string `value`.
async fn set_string(tv: &TestValkey, key: String, value: &str) {
    let _: () = tv
        .client()
        .pool
        .set(key, value, None, None, false)
        .await
        .expect("SET must succeed");
}

/// **An object-id counter that is not a decimal integer** at the un-hold
/// (fail-on-old). A writable plan over a counter holding `"abc"`: the
/// un-hold script answers `counter_unreadable` before any write, and the
/// Job logs one ERROR naming the counter's key and value and exits 3 — the
/// marker present, the counter still `"abc"`, no un-hold line. Fail-on-old
/// (in-place perturbation of `dr_unhold.lua`, red then green once
/// restored): with the arm replaced by the `redis.error_reply` it had, the
/// Job exits 1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unhold_on_a_non_decimal_counter_exits_3_and_leaves_the_counter_and_the_marker() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        partial_wipe(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;
        let counter_key = tv.key_schema().counter_object_id();
        set_string(&tv, counter_key.clone(), "abc").await;

        let (exit, captured, _observed, _covered, finished) =
            run_once(&tv, &config, &store, |b| b).await;

        assert_eq!(exit, JobExit::StaysHeld, "{}", captured.text);
        let line = the_line(&captured, COUNTER_UNREADABLE_LINE);
        assert_eq!(line["level"], "ERROR");
        assert!(
            line["fields"]["message"]
                .as_str()
                .is_some_and(|m| m.contains(&format!("{counter_key} holds \"abc\""))),
            "the line names the counter's key and value: {line}"
        );
        assert_eq!(
            the_finish(&finished).unhold,
            Some(UnholdOutcome::CounterUnreadable {
                value: "abc".to_owned()
            })
        );
        assert!(marker_held(&tv).await, "the marker stays");
        assert_eq!(get(&tv, counter_key).await.as_deref(), Some("abc"));
        assert_eq!(lines_starting(&captured, UNHELD_LINE), 0);
    })
    .await;
}

/// **A key the un-hold reads of the wrong type** (fail-on-old): the object-id
/// counter a HASH from before the run, or the marker turned into a LIST
/// while the Job runs (once its coverage check is finalised). The script
/// fails with the server's `WRONGTYPE` before any write, and the Job logs
/// one ERROR naming the lease's and the counter's keys and exits 3, not
/// retrying — the marker's key still standing, the counter as it was.
/// Fail-on-old (in-place perturbation of `dr::job::finish::unhold`, red then
/// green once restored): with its `WRONGTYPE` arm removed, the error is
/// fatal and the Job exits 1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unhold_on_a_key_of_the_wrong_type_exits_3_and_clears_nothing() {
    within_test_budget(async {
        for wrong in ["the counter", "the marker"] {
            let tv = TestValkey::setup().await;
            let store = WriterStore::new();
            partial_wipe(&tv, &store).await;
            let config = writer_config();
            hold(&tv, &config).await;
            let keys = tv.key_schema().clone();
            if wrong == "the counter" {
                hset(&tv, keys.counter_object_id(), &[("not", "a counter")]).await;
            }
            let client = tv.client().clone();
            let marker = keys.dr_hold_marker();

            let (exit, captured, _observed, _covered, finished) =
                run_once(&tv, &config, &store, move |b| {
                    if wrong != "the marker" {
                        return b;
                    }
                    b.with_plan_observer(Arc::new(move |_plan, _report| {
                        let (client, marker) = (client.clone(), marker.clone());
                        block_on_spawned(&tokio::runtime::Handle::current(), async move {
                            let _: i64 = client.pool.del(&marker).await.expect("DEL");
                            let _: i64 = client.pool.rpush(&marker, "x").await.expect("RPUSH");
                        })
                        .expect("the marker is retyped");
                    }))
                })
                .await;

            assert_eq!(exit, JobExit::StaysHeld, "{wrong}: {}", captured.text);
            let line = the_line(&captured, UNHOLD_WRONGTYPE_LINE);
            assert_eq!(line["level"], "ERROR", "{wrong}");
            assert_eq!(line["fields"]["lease_key"], keys.dr_job_lease(), "{wrong}");
            assert_eq!(
                line["fields"]["counter_key"],
                keys.counter_object_id(),
                "{wrong}"
            );
            assert_eq!(the_finish(&finished).unhold, None, "{wrong}: no answer");
            assert!(marker_held(&tv).await, "{wrong}: the marker's key stands");
            let counter_type: String = tv
                .client()
                .pool
                .r#type(keys.counter_object_id())
                .await
                .expect("TYPE");
            assert_eq!(
                counter_type,
                if wrong == "the counter" {
                    "hash"
                } else {
                    "none"
                },
                "{wrong}: the counter as it was"
            );
            assert_eq!(lines_starting(&captured, UNHELD_LINE), 0, "{wrong}");
        }
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The counter floor's terms, its ceiling, and the counters record (PD-7)
// ═════════════════════════════════════════════════════════════════════════════

/// The floor fixture: session `s-floor` with `l.usda` → 100, lost and backed
/// up at seq 3, and a counters record reading `sidecar_max`.
async fn floor_fixture(store: &Arc<WriterStore>, sidecar_max: u64) {
    put_session(
        store.as_ref(),
        &session_backup("s-floor", &[("l.usda", 100)], &[]),
    )
    .await;
    put_base(
        store.as_ref(),
        100,
        "s-floor",
        "l.usda",
        3,
        base(1, 0xF1, &[]),
    )
    .await;
    store
        .put_counters_sidecar(Bytes::from(format!(
            "{{\"max_object_id_observed\":{sidecar_max},\"updated_at\":\"2026-10-20T12:00:00Z\"}}"
        )))
        .await
        .expect("the counters record put succeeds");
}

/// **The sidecar term binds.** With a margin of 1, a counters record of
/// 9,000 and listed ids at most 100, the floor is the sidecar term, 9,001,
/// above the listed and surviving terms (101 each); the counter is set to
/// it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_counter_floor_is_the_sidecar_term_when_it_is_the_largest() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        floor_fixture(&store, 9_000).await;
        let mut config = writer_config();
        config.checkpoint_counter_safety_margin = 1;
        hold(&tv, &config).await;

        let (exit, captured, _observed, _covered, finished) =
            run_once(&tv, &config, &store, |b| b).await;

        assert_eq!(exit, JobExit::Done, "{}", captured.text);
        let floor = the_finish(&finished).floor.expect("the floor was computed");
        assert_eq!(
            (floor.sidecar, floor.listed, floor.surviving, floor.dead),
            (Some(9_001), Some(101), Some(101), None)
        );
        assert_eq!(floor.floor, 9_001);
        assert_eq!(counter(&tv).await, Some(9_001));
    })
    .await;
}

/// **The surviving term binds.** With a margin of 1, a counters record of 0
/// and listed ids at most 100, a kept survivor no backup lists, object
/// 70,000 (its session's record and mapping surviving), makes the floor the
/// surviving term, 70,001; the counter is set to it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_counter_floor_is_the_surviving_term_when_it_is_the_largest() {
    const SURVIVOR: u64 = 70_000;
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        floor_fixture(&store, 0).await;
        seed_session_record(&tv, "s-kept").await;
        seed_complete(&tv, SURVIVOR, "s-kept", "k.usda", 50, &[]).await;
        seed_mapping(&tv, "s-kept", "k.usda", SURVIVOR).await;
        let mut config = writer_config();
        config.checkpoint_counter_safety_margin = 1;
        hold(&tv, &config).await;

        let (exit, captured, _observed, _covered, finished) =
            run_once(&tv, &config, &store, |b| b).await;

        assert_eq!(exit, JobExit::Done, "{}", captured.text);
        let floor = the_finish(&finished).floor.expect("the floor was computed");
        assert_eq!(
            (floor.sidecar, floor.listed, floor.surviving, floor.dead),
            (Some(1), Some(101), Some(SURVIVOR + 1), None)
        );
        assert_eq!(floor.floor, SURVIVOR + 1);
        assert_eq!(counter(&tv).await, Some(SURVIVOR + 1));
    })
    .await;
}

/// **A floor above the largest allocatable id** (text 14). A counters record
/// at `MAX_OBJECT_ID` with a margin of 1 makes the sidecar term 2^53, past
/// the largest id the service can allocate: the Job logs one ERROR naming
/// the term and its value, and exits 3 with no un-hold — the marker present
/// and the counter untouched.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_counter_floor_above_the_largest_allocatable_id_exits_3_and_clears_nothing() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        floor_fixture(&store, MAX_OBJECT_ID).await;
        let mut config = writer_config();
        config.checkpoint_counter_safety_margin = 1;
        hold(&tv, &config).await;

        let (exit, captured, _observed, _covered, finished) =
            run_once(&tv, &config, &store, |b| b).await;

        assert_eq!(exit, JobExit::StaysHeld, "{}", captured.text);
        let line = the_line(&captured, FLOOR_ABOVE_MAX_LINE);
        assert_eq!(line["level"], "ERROR");
        assert_eq!(line["fields"]["term"], "sidecar");
        assert_eq!(line["fields"]["value"], MAX_OBJECT_ID + 1);
        let record = the_finish(&finished);
        assert!(record.recount.is_some(), "it recounted");
        assert_eq!((record.floor, record.unhold), (None, None));
        assert!(marker_held(&tv).await, "the marker stays");
        assert_eq!(counter(&tv).await, None, "the counter is untouched");
    })
    .await;
}

/// **PD-7, a counters record that does not parse** (text 15): a record
/// with no `max_object_id_observed` (`{"counters":{}}`). The Job logs one
/// ERROR and exits 3 before any write: no writer runs, the finish is not
/// reached, and the store is as it was before the run.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_counters_record_that_does_not_parse_exits_3_before_any_write() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        partial_wipe(&tv, &store).await;
        store
            .put_counters_sidecar(Bytes::from_static(b"{\"counters\":{}}"))
            .await
            .expect("the counters record put succeeds");
        let config = writer_config();
        hold(&tv, &config).await;
        let before = keyspace(&tv).await;

        let (exit, captured, observed, _covered, finished) =
            run_once(&tv, &config, &store, |b| b).await;

        assert_eq!(exit, JobExit::StaysHeld, "{}", captured.text);
        assert_eq!(
            the_line(&captured, SIDECAR_UNREADABLE_LINE)["level"],
            "ERROR"
        );
        assert!(observed.lock().unwrap().is_empty(), "no writer ran");
        assert!(
            finished.lock().unwrap().is_empty(),
            "the finish was not reached"
        );
        assert_eq!(
            keyspace_writes(&before, &keyspace(&tv).await, WriteRules::STRICT),
            Vec::<String>::new(),
            "nothing written"
        );
    })
    .await;
}
