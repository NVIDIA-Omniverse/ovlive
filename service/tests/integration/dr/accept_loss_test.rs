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

//! The accept-loss Job end to end (`live_edit::dr::job::accept_loss`; SDD
//! SR-10, PRD FR7): the Job run through its frame against a real store and a
//! scripted backup store, from its preconditions to its exit code, with its
//! restore phase, its lost set, its discards, the recount and the un-hold.
//!
//! One named test per item of the Task's test shape; each test's doc line
//! starts with its ID. The pure halves are unit tests beside the code: the
//! lost set and the discard check (LS1–LS3, P51-b, P55-b, PD-1 and option E,
//! `dr::job::plan::lost`), the rows a removal deletes (LS4,
//! `dr::job::plan::report`), the over-limit step-down (L1-AL,
//! `dr::job::plan::lineage`), the removal's batch limits, its header
//! transaction's bound and the value removal's fence
//! (`storage::layer::dr_restore`), the command line (option E,
//! `dr::job::args`), P51-d, P55-a (D0's stop, and D2's on an object another
//! writer registered before the run) and D3's, D5's and D6's steps no
//! fixture here reaches, on hand-built lost sets — D5's walk over
//! table-encoded value indices, the order of its writes and its admission
//! among them (`dr::job::execute::discard`), the cursor pager it walks by
//! (`tests/proptests.rs`, `cursor_pager`), the session backup's rewrite leaving out
//! an over-limit layer (`checkpoint::session`), D1's read
//! of the members the dead-object set lacks
//! (`storage::external_values::recon_absent`), and P55-c and the body's
//! discard check (AL10; `dr::job::accept_loss`).
//!
//! - **A1**, a value in no store: the layer's catch-up fails before the run
//!   and succeeds after it; the row and the `obj:ext` member are gone and the
//!   header is banded.
//! - **A2**, a lost layer with no backup: unmapped, registered dead, its
//!   session backup rewritten; un-held.
//! - **A3**, a lost layer with a usable backup: restored, never discarded.
//! - **A4**, the `Disabled` and `InMemory` adapters: no backup-store call; the
//!   lost layer discarded and registered (PD-5).
//! - **A5**, defer-6(c) on the StorageApi fake (three shapes, exit 0), and a
//!   backup store at the wrong address (exit 3, nothing written).
//! - **A6**, an Enumerate route answering 404: the canary keeps the service
//!   held, nothing written.
//! - **A7**, **A7b**, **A8**, PD-3: the backup store is read only at or
//!   after `T_safe` — every time-switched call's first one included: a
//!   layer listing empty before it, a session backup listed only from it
//!   (beside another session's, so the listing is never empty) and a value
//!   absent before it are not acted on.
//! - **A9**, P51-c: a deleted mapping Step B replaces is never discarded.
//! - **A10**, P55-a across a retry (B1): an object completed after the plan
//!   is never registered, so the retry keeps it.
//! - **A10b**, **A10c**, **A10d**, P55-a in the window after the
//!   registration (B1): an object completed after D1 registered it stops
//!   the run at D2 with AL9, which removes every object the run registered
//!   and has not wiped (the completed one included, so the retry keeps it)
//!   and logs the companion WARN; an object already wiped stays registered,
//!   and so does one another writer registered before the run.
//! - **A11**, PD-1: an object no read could attribute refuses every discard;
//!   with the second confirm it is reported and never registered.
//! - **A12**, PD-2: a brownout refuses every discard (exit 3, counts exact);
//!   the re-run after it exits 0. **A12b**: the same brownout with the second
//!   confirm discards the unconfirmed value, WARNs it and counts it apart.
//! - **Option E, ruling 2**: a layer whose newer backups are unknown keeps
//!   its survivor under the second confirm. **Option E, the totals**: AL4
//!   and AL11 count the unconfirmed items discarded apart from those kept.
//! - **PD-2, a session backup's read failing**: a record-less session's
//!   backup read failing is unconfirmed, discarded only under the second
//!   confirm; a live session's makes its rewrite unconfirmed.
//! - **PD-4 / S1**: a layer whose newest full backup is over the per-delta
//!   entry limit is restored from an older valid one; with nothing older
//!   usable (only over-limit backups, or one beside a broken older one) it is
//!   unmapped with its backups kept and never registered (AL13), and a later
//!   restore reports them unreferenced.
//! - **A13**, ADD 9: a write after the recount, and a recount killed
//!   part-way, each leave the marker; the re-run converges.
//! - **A14**, defer-65: the un-hold's `lease_lost` exits 1, its in-doubt
//!   `already_unheld` exits 0.
//! - **A15**, kills converge: at each discard kill point the re-run leaves
//!   the store as an uninterrupted run does — the lost layers' registration,
//!   wipe and unmap; a lost session's registration and wipe (on the
//!   convergence fixture, and on A16's under a listing and under
//!   `Disabled`), its unmap and its backup's delete; a value removal after
//!   its first row batch, and after its row and `obj:ext` batches before
//!   the banded header; a
//!   session backup's rewrite; and, on the over-limit fixture, the
//!   registration beside an over-limit layer, D3's early rewrite of its
//!   session backup, its unmap and its session backup's rewrite, each
//!   re-run taking no Step B, so the session record keeps its name — and a
//!   restore after the registration holds and maps nothing to a registered
//!   object. Every kill point the discards have is covered.
//! - **A16**, PD-6: a lost session with a complete layer (a non-empty
//!   listing; and `Disabled`) is registered, wiped and unmapped.
//! - **A17**: the accept-loss lines' fields and words are the discard
//!   report's; the value removal leaves no row naming a removed value, and
//!   the layer's catch-up succeeds.
//!
//! Every run uses a per-test key prefix (`TestValkey`).
//!
//! # Fail-on-old
//!
//! Each pin below was proven red by an in-place perturbation of the code it
//! guards, then green once restored: A1 with the value removal (D5) skipped
//! (the row stays); A7 and A8 with the planner's `T_safe` wait for
//! accept-loss dropped (the first listing and the first probe come before
//! `T_safe`), and A7b likewise (the session is discarded as lost); P51-c
//! with the planner's defer-51 fix and both plan checks' rule 2 removed (the
//! stale item reaches the unmap, which leaves the mapping as the CAS finds
//! it, but is a lost layer); A10 with D0's header read moved below the
//! registration (the object is registered) and with D0 removed (D2 stops the
//! run and logs AL9's companion); A10b with D2's removal removing nothing
//! (the retry wipes the completed header unread); A10c with the removal
//! limited to AL9's own object (the companion names only it); A10d with D1
//! taking every member as added (the companion reports the other writer's
//! registration removed, not kept); A11 with
//! the unattributed object discarded as a lost layer (the run no longer
//! refuses); option E's ruling 2 with the survivor's layer discarded under
//! the second confirm (the discard check refuses it, exit 1); option E's
//! totals with every unconfirmed item counted as discarded (AL4 counts 3
//! and 0, not 1 and 2); A12 with the refusal gate dropped (the run exits
//! 0); PD-2's session read with a failed read taken as an undecodable
//! backup (the session is discarded, exit 0); PD-4 with the lineage walk
//! run under restore's policy (the run stays held); S1 with the over-limit
//! layer judged a lost layer like any other (it is registered, with AL5 and
//! no AL13); A13 (a) with the finish's check for a write after the recount
//! removed (the marker is deleted); A14 with `lease_lost` mapped to exit 0
//! and `already_unheld` to exit 1; A15 after the unmap with the
//! registration moved after the unmap (the object is never registered by
//! the killed run, so the re-run restores the stale session backup's
//! mapping of it and its record from that backup: the store differs from an
//! uninterrupted run's); A15 on an over-limit layer at D3's early rewrite,
//! its unmap and its session backup's rewrite with the early rewrite made a
//! no-op (the killed run's session backup still names the layer, and, past
//! that check, the re-run after the unmap and before the session rewrite
//! takes Step B for the session);
//! A15 before the banded header with every
//! `HDEL obj:ext` in the batches before the header and none in the header
//! transaction (the killed run leaves no hash in `obj:ext`).
//!
//! # Time
//!
//! Every test runs under `writer_support`'s one-test budget (its module
//! docs, *Time*): 90 s plus the store clean-up, inside nextest's 120 s kill,
//! each Job run bounded by `RUN_BUDGET` (30 s). A healthy run of these
//! fixtures takes about a second; A6, A7, A7b and A8 also wait about 2 s for
//! `T_safe` (`T_SAFE_IN`), and the over-limit fixtures build a full backup of
//! about 100,000 entries. A15's tests make three runs each (an uninterrupted
//! reference, the killed run and its re-run); A10, A10b, A12, A13 (b),
//! PD-2's session tests and S1 (i) two.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::{
    HashesInterface, KeysInterface, ListInterface, SetsInterface, SortedSetsInterface,
    StreamsInterface,
};
use futures::StreamExt;
use tonic::Request;

use live_edit::checkpoint::{CheckpointAdapter, CheckpointStore, SessionCheckpoint};
use live_edit::config::Config;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::dr::job::execute::discard::DiscardReport;
use live_edit::dr::job::execute::KillPoint;
use live_edit::dr::job::finish::FinishRecord;
use live_edit::dr::job::plan::lost::LostSet;
use live_edit::dr::job::plan::model::{
    Advisory, CoverageReport, PlanOutcome, RestorePlan, SessionAction,
};
use live_edit::dr::job::plan::report::{hash_prefix, DiscardPlanCounts};
use live_edit::dr::job::quiet::{not_found_cache_lifetime, QuietPeriod, StoreDeadlines};
use live_edit::dr::job::{run_job, AcceptLossBody, JobExit, JobMode, RestoreBody};
use live_edit::grpc::layer::handle_update_layer;
use live_edit::grpc::proto::{subscribe_read_layer_response, SubscribeReadLayerRequest};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::storage::dr_hold::UnholdOutcome;
use live_edit::storage::dr_job::store_time_ms;
use live_edit::storage::external_values::ext_content_hash;
use live_edit::storage::keys::{hex_encode, sha256_to_hex, KeySchema};
use live_edit::storage::test_support::{block_on_spawned, HoldMarkerSeed, TestValkey};
use oodf_ffi::{OwnedDelta, SetField, TimeSample};

use super::support::{keyspace, keyspace_writes, WriteRules};
use super::writer_support::*;
use crate::checkpoint::adapter_storage_api_test::{
    build_fake_backed_adapter, mount_fake_storage_routes, FakeStorageState,
};
use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::grpc::helpers::{
    build_service, create_session_and_layer, make_request, make_update_data, make_update_header,
    streaming_update, AllowAllPermissions,
};
use crate::grpc::log_capture::Captured;

// ═════════════════════════════════════════════════════════════════════════════
// The accept-loss lines
// ═════════════════════════════════════════════════════════════════════════════

/// How the INFO of a test kill point starts.
const AL_KILL_LINE: &str = "the accept-loss Job stopped at its test kill point";
/// How AL3, the refusal on an unconfirmed item (ERROR, exit 3), starts.
const AL3_REFUSAL: &str = "the accept-loss Job restored everything it could and recounted the \
                           references of the large values, but discarded nothing";
/// How AL4, the discard plan (INFO), starts.
const AL4_PLAN: &str = "the accept-loss Job discards ";
/// How AL5, one lost layer discarded (WARN), starts.
const AL5_LAYER: &str = "the accept-loss Job discarded layer ";
/// How AL6, one lost session discarded (WARN), starts.
const AL6_SESSION: &str = "the accept-loss Job discarded session ";
/// How AL7, one layer's values removed (WARN), starts.
const AL7_VALUES: &str = "the accept-loss Job removed ";
/// How AL8, a mapping left as found (WARN), starts.
const AL8_LEFT: &str = "the accept-loss Job left layer ";
/// How AL9, a lost layer's object found complete (ERROR, exit 1), starts.
const AL9_COMPLETE: &str = "the accept-loss Job found object ";
/// How AL9's companion (WARN) starts when D2's stop removed objects this run
/// registered from the dead-object set.
const AL9_REMOVED: &str = "the accept-loss Job removed objects ";
/// How AL9's companion (WARN) starts when D2's stop removed nothing.
const AL9_REMOVED_NOTHING: &str = "the accept-loss Job removed nothing from the dead-object set";
/// How AL11, the accept-loss totals (INFO), starts.
const AL11_SUMMARY: &str = "the accept-loss Job's summary: discarded ";
/// How AL12, one unconfirmed item discarded under the second confirm
/// (WARN), starts.
const AL12_UNCONFIRMED: &str = "the accept-loss Job could not confirm ";
/// How the un-hold's INFO (text 9) starts.
const UNHELD_LINE: &str = "the accept-loss Job cleared the hold for incident";
/// How the write-after-recount refusal (text 13) starts.
const WRITE_AFTER_RECOUNT_LINE: &str =
    "the accept-loss Job did not clear the hold: a write ran after its recount";
/// How the lost-lease ERROR at the un-hold (text 12) starts.
const LEASE_LOST_LINE: &str =
    "the accept-loss Job's lease was no longer its own when it went to clear the hold";
/// Why accept-loss waits for `T_safe` before it plans (PD-3).
const PD3_WAIT: &str = "the accept-loss Job reads the backup store only once \
                        checkpoint.notFoundCacheSecs has passed after the quiet period, so every \
                        absence it discards on is authoritative";
/// The not-found cache lifetime A6, A7 and A8 run with, so `T_safe` is this
/// far past the quiet period's end.
const T_SAFE_IN: Duration = Duration::from_secs(2);

// ═════════════════════════════════════════════════════════════════════════════
// Running the Job
// ═════════════════════════════════════════════════════════════════════════════

/// What one accept-loss run left: its exit, its log, and what its observers
/// recorded.
struct AcceptLossRun {
    exit: JobExit,
    captured: Captured,
    /// Each lost set and what its discards did (the discard observer).
    discards: Vec<(LostSet, DiscardReport)>,
    /// What the finish reached (the finish observer).
    finishes: Vec<FinishRecord>,
    /// The plan and its finalised coverage report (the plan observer).
    plans: Vec<(RestorePlan, CoverageReport)>,
}

impl AcceptLossRun {
    /// The one lost set and discard report.
    fn the_discard(&self) -> &(LostSet, DiscardReport) {
        assert_eq!(
            self.discards.len(),
            1,
            "the discards ran once; capture:\n{}",
            self.captured.text
        );
        &self.discards[0]
    }

    /// The one finish record.
    fn the_finish(&self) -> &FinishRecord {
        assert_eq!(self.finishes.len(), 1, "the finish was reached once");
        &self.finishes[0]
    }

    /// The one plan.
    fn the_plan(&self) -> &(RestorePlan, CoverageReport) {
        assert_eq!(self.plans.len(), 1, "the plan was finalised once");
        &self.plans[0]
    }

    /// The captured lines whose message starts with `prefix`.
    fn lines(&self, prefix: &str) -> Vec<&serde_json::Value> {
        self.captured
            .events
            .iter()
            .filter(|e| {
                e["fields"]["message"]
                    .as_str()
                    .is_some_and(|m| m.starts_with(prefix))
            })
            .collect()
    }

    /// The one captured line whose message starts with `prefix`.
    fn line(&self, prefix: &str) -> &serde_json::Value {
        let found = self.lines(prefix);
        assert_eq!(
            found.len(),
            1,
            "one line starting {prefix:?}; capture:\n{}",
            self.captured.text
        );
        found[0]
    }
}

/// Run the accept-loss Job on `tv` with `config`, reading backups from
/// `store`, with the second confirm when `unconfirmed`, and `seams` applied
/// to its body after the recording observers (so a test's own plan observer
/// replaces the recording one).
async fn accept_loss_with(
    tv: &TestValkey,
    config: &Config,
    store: Arc<dyn CheckpointStore>,
    unconfirmed: bool,
    seams: impl FnOnce(AcceptLossBody) -> AcceptLossBody,
) -> AcceptLossRun {
    type Sink<T> = Arc<Mutex<Vec<T>>>;
    let discards: Sink<(LostSet, DiscardReport)> = Arc::new(Mutex::new(Vec::new()));
    let finishes: Sink<FinishRecord> = Arc::new(Mutex::new(Vec::new()));
    let plans: Sink<(RestorePlan, CoverageReport)> = Arc::new(Mutex::new(Vec::new()));
    let body = {
        let (d, f, p) = (
            Arc::clone(&discards),
            Arc::clone(&finishes),
            Arc::clone(&plans),
        );
        seams(
            AcceptLossBody::new(unconfirmed)
                .with_discard_observer(Arc::new(move |lost, report| {
                    d.lock().unwrap().push((lost.clone(), report.clone()));
                }))
                .with_finish_observer(Arc::new(move |record| {
                    f.lock().unwrap().push(record.clone());
                }))
                .with_plan_observer(Arc::new(move |plan, report| {
                    p.lock().unwrap().push((plan.clone(), report.clone()));
                })),
        )
    };
    let metrics = metrics();
    let mut ctx = context(tv, JobMode::AcceptLoss, config.clone(), store, &metrics);
    ctx.args.discard_unconfirmed = unconfirmed;
    let (exit, captured) = captured_within(RUN_BUDGET, run_job(ctx, body)).await;
    let discards = std::mem::take(&mut *discards.lock().unwrap());
    let finishes = std::mem::take(&mut *finishes.lock().unwrap());
    let plans = std::mem::take(&mut *plans.lock().unwrap());
    AcceptLossRun {
        exit,
        captured,
        discards,
        finishes,
        plans,
    }
}

/// [`accept_loss_with`] with no seam and no second confirm.
async fn accept_loss(tv: &TestValkey, config: &Config, store: &Arc<WriterStore>) -> AcceptLossRun {
    accept_loss_with(tv, config, Arc::clone(store) as _, false, |b| b).await
}

/// [`accept_loss_with`] with the second confirm and no seam.
async fn accept_loss_discarding_unconfirmed(
    tv: &TestValkey,
    config: &Config,
    store: &Arc<WriterStore>,
) -> AcceptLossRun {
    accept_loss_with(tv, config, Arc::clone(store) as _, true, |b| b).await
}

/// Hold the store under [`INCIDENT`] with its quiet period over now, as
/// `writer_support::hold` does; returns `T_safe` on the store's clock, the
/// quiet period's end plus `config`'s not-found cache lifetime, as the frame
/// computes it.
async fn hold_returning_t_safe(tv: &TestValkey, config: &Config) -> u64 {
    let quiet = QuietPeriod::from_config(config);
    let q = u64::try_from(quiet.total().as_millis()).expect("Q fits");
    let now = store_time_ms(tv.client(), tv.key_schema())
        .await
        .expect("TIME");
    let detected_at = now - q;
    tv.write_hold_marker(
        &HoldMarkerSeed::default()
            .with_incident(INCIDENT)
            .detected_at(detected_at),
    )
    .await;
    StoreDeadlines::new(detected_at, quiet.total(), not_found_cache_lifetime(config)).t_safe_ms
}

/// How many `waiting {n}s on the store's clock: {why}` lines name `why`.
fn waits_for(captured: &Captured, why: &str) -> usize {
    captured
        .events
        .iter()
        .filter(|e| {
            e["fields"]["message"].as_str().is_some_and(|m| {
                m.starts_with("waiting ")
                    && m.contains("s on the store's clock: ")
                    && m.ends_with(why)
            })
        })
        .count()
}

// ═════════════════════════════════════════════════════════════════════════════
// Store readers
// ═════════════════════════════════════════════════════════════════════════════

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

/// The object `layer` of `session` is mapped to, as stored.
async fn mapping(tv: &TestValkey, session: &str, layer: &str) -> Option<String> {
    get(tv, tv.key_schema().layer_mapping(session, layer)).await
}

/// Whether `session`'s layers index names `layer`.
async fn indexed(tv: &TestValkey, session: &str, layer: &str) -> bool {
    smembers(tv, tv.key_schema().layers_index(session))
        .await
        .contains(&hex_encode(layer))
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

/// Whether `object_id`'s `obj:ext` names `hash`.
async fn ext_indexed(tv: &TestValkey, object_id: u64, hash: &[u8; 32]) -> bool {
    let n: i64 = tv
        .client()
        .pool
        .hexists(
            tv.key_schema().obj_ext_index(&object_id.to_string()),
            sha256_to_hex(hash),
        )
        .await
        .expect("HEXISTS");
    n > 0
}

/// The key of `object_id`'s field row `(section, key)`.
fn field_key(tv: &TestValkey, object_id: u64, section: &str, key: &str) -> String {
    tv.key_schema().obj_field_from_member(
        &object_id.to_string(),
        &KeySchema::field_member(section, key),
    )
}

/// `session`'s backup in `store`, decoded; `None` when absent.
async fn session_backup_of(store: &WriterStore, session: &str) -> Option<SessionCheckpoint> {
    store
        .get_session_checkpoint(session)
        .await
        .expect("the in-memory store answers")
        .map(|blob| SessionCheckpoint::from_bytes(&blob.bytes).expect("the backup decodes"))
}

/// A value of `len` bytes, told apart by `tag`, and its content hash.
fn value_of(tag: u8, len: usize) -> ([u8; 32], Vec<u8>) {
    let bytes: Vec<u8> = (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(tag))
        .collect();
    (ext_content_hash(&bytes), bytes)
}

/// Put `bytes` into the backup store as the value `hash`.
async fn put_value(store: &WriterStore, hash: &[u8; 32], bytes: &[u8]) {
    store
        .put_ext_blob(&sha256_to_hex(hash), Bytes::from(bytes.to_vec()))
        .await
        .expect("the value put succeeds");
}

// ═════════════════════════════════════════════════════════════════════════════
// Fixtures
// ═════════════════════════════════════════════════════════════════════════════

/// The lost-layer fixture's session, its kept survivor and its lost layer.
const LS: &str = "s-al";
const LS_KEPT: u64 = 101;
const LS_LOST: u64 = 102;

/// The lost-layer fixture: session [`LS`], its record surviving, with
/// `kept.usda` → [`LS_KEPT`], a complete survivor at seq 50 backed up at 40,
/// and `lost.usda` → [`LS_LOST`], whose header the incident lost and which
/// has no backup; its session backup maps both. The counters record is put.
async fn lost_layer_fixture(tv: &TestValkey, store: &WriterStore) {
    put_session(
        store,
        &session_backup(LS, &[("kept.usda", LS_KEPT), ("lost.usda", LS_LOST)], &[]),
    )
    .await;
    put_base(store, LS_KEPT, LS, "kept.usda", 40, base(1, 0x01, &[])).await;
    put_counters(store).await;
    seed_session_record(tv, LS).await;
    seed_complete(tv, LS_KEPT, LS, "kept.usda", 50, &[]).await;
    seed_mapping(tv, LS, "kept.usda", LS_KEPT).await;
    seed_mapping(tv, LS, "lost.usda", LS_LOST).await;
}

/// The layer the service-made fixtures write their values on.
const VALUE_LAYER: &str = "v.usda";

/// An external-value field row on the layer's root (section 1): `key` names
/// `hash`, the `index`-th (1-based) value part of the upload.
fn ext_field(key: &str, hash: &[u8; 32], len: usize, index: u32, set_order: u64) -> SetField {
    SetField {
        section_id: 1,
        key_name: key.to_owned(),
        value_or_ext_hash: hash.to_vec(),
        ext_value_size: len as u64,
        ext_value_index: index,
        set_order,
    }
}

/// Commit `delta` with `parts` on `layer` of `session` (object `object_id`)
/// through `UpdateLayer`.
async fn update(
    svc: &LiveEditService,
    session: &str,
    object_id: u64,
    delta: &OwnedDelta,
    parts: &[&[u8]],
) {
    handle_update_layer(
        svc,
        make_request(streaming_update(vec![
            make_update_header(session, VALUE_LAYER, object_id),
            make_update_data(delta_to_ompf_bytes(delta, parts)),
        ])),
    )
    .await
    .unwrap_or_else(|s| panic!("the update commits: {s}"));
}

/// Delete `hash`'s bytes from Valkey, as an incident that lost its `{hash}`
/// slot does; its meta and every reference stay.
async fn lose_bytes(tv: &TestValkey, hash: &[u8; 32]) {
    let _: i64 = tv
        .client()
        .pool
        .del(tv.key_schema().ext_data(&sha256_to_hex(hash)))
        .await
        .expect("DEL ext data");
}

/// A session named `name` and its layer [`VALUE_LAYER`], made through the
/// service, whose root's field `ext` names a 64-byte value told apart by
/// `tag`, the value's bytes then lost from Valkey. Returns the session, the
/// object, and the value's hash and bytes.
async fn layer_with_a_lost_value(
    tv: &TestValkey,
    name: &str,
    tag: u8,
) -> (String, u64, [u8; 32], Vec<u8>) {
    let svc = build_service(tv, Arc::new(AllowAllPermissions));
    let (session, object_id) = create_session_and_layer(tv, &svc, name, VALUE_LAYER).await;
    let (hash, bytes) = value_of(tag, 64);
    let delta = OwnedDelta {
        set_fields: vec![ext_field("ext", &hash, bytes.len(), 1, 1)],
        ..OwnedDelta::default()
    };
    update(&svc, &session, object_id, &delta, &[&bytes]).await;
    assert!(
        ext_indexed(tv, object_id, &hash).await,
        "fixture: the update indexed the value"
    );
    lose_bytes(tv, &hash).await;
    (session, object_id, hash, bytes)
}

/// The external-value parts of the first content a catch-up from nothing
/// delivers on [`VALUE_LAYER`] of `session`, or the status it ends with.
async fn catch_up_values(
    svc: &LiveEditService,
    session: &str,
) -> Result<Vec<Vec<u8>>, tonic::Status> {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session.to_owned(),
        layer_name: VALUE_LAYER.to_owned(),
        object_id: 0,
        sequence_num: 0,
        client_id: "accept-loss-subscriber".to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    let mut stream = handle_subscribe_read_layer(svc, request)
        .await?
        .into_inner();
    // One deadline over the whole read, however many chunks it takes.
    tokio::time::timeout(RUN_BUDGET, async move {
        let first = stream.next().await.expect("the stream sends a message")?;
        let header = match first.response_type {
            Some(subscribe_read_layer_response::ResponseType::Header(header)) => header,
            other => panic!("expected a header, got {other:?}"),
        };
        let mut body = Vec::new();
        while (body.len() as u64) < header.total_size {
            let next = stream.next().await.expect("the stream sends the body")?;
            match next.response_type {
                Some(subscribe_read_layer_response::ResponseType::Chunk(chunk)) => {
                    body.extend_from_slice(&chunk.data)
                }
                other => panic!("expected a body chunk, got {other:?}"),
            }
        }
        Ok(OmpfContainer::parse(&body)
            .expect("the content parses as OMPF")
            .external_values
            .iter()
            .map(|part| part.to_vec())
            .collect())
    })
    .await
    .expect("the catch-up ends within the budget")
}

// ═════════════════════════════════════════════════════════════════════════════
// A1: a value in no store
// ═════════════════════════════════════════════════════════════════════════════

/// **A1** (AC1; fail-on-old). A kept layer referencing a value whose bytes
/// are in no store: before the run its catch-up from nothing fails
/// `INTERNAL`. The accept-loss Job removes the value in place and un-holds
/// (exit 0, the marker gone): the field row and the `obj:ext` member are
/// gone, the header's `seq` is the stream's floor F plus the band, and its
/// `restore_seq_bump` grew by the band; the layer's catch-up then succeeds,
/// carrying no value. With the value removal skipped, the row stays.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a1_accept_loss_removes_a_value_in_no_store_so_the_layers_catch_up_succeeds() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        put_counters(store.as_ref()).await;
        let (session, object_id, hash, _bytes) = layer_with_a_lost_value(&tv, "a1", 0xA1).await;
        let before = catch_up_values(&build_service(&tv, Arc::new(AllowAllPermissions)), &session)
            .await
            .expect_err("fixture: the catch-up fails before the run");
        assert_eq!(before.code(), tonic::Code::Internal, "{before}");
        let prior = header(&tv, object_id).await;
        let prior_seq: u64 = prior["seq"].parse().expect("a seq");
        let prior_bump: u64 = prior
            .get("restore_seq_bump")
            .map_or(0, |b| b.parse().expect("a bump"));
        let config = writer_config();
        hold(&tv, &config).await;

        let run = accept_loss(&tv, &config, &store).await;

        assert_eq!(run.exit, JobExit::Done, "{}", run.captured.text);
        assert!(
            !exists(&tv, field_key(&tv, object_id, "1", "ext")).await,
            "the field row is removed"
        );
        assert!(
            !ext_indexed(&tv, object_id, &hash).await,
            "the obj:ext member is removed"
        );
        assert!(!marker_held(&tv).await, "the hold is cleared");
        let (_lost, report) = run.the_discard();
        let record = &report.values[&object_id];
        assert_eq!(record.hashes, BTreeSet::from([hash]));
        assert_eq!(record.rows, 1);
        assert_eq!(record.prior_seq, prior_seq);
        assert!(record.floor >= prior_seq, "F is at least S: {record:?}");
        assert_eq!(record.band, BAND);
        let after = header(&tv, object_id).await;
        assert_eq!(
            after["seq"],
            (record.floor + record.band).to_string(),
            "seq = F + band"
        );
        assert_eq!(record.seeded_seq, record.floor + record.band);
        assert_eq!(
            after["restore_seq_bump"],
            (prior_bump + record.band).to_string(),
            "restore_seq_bump += band"
        );
        let values = catch_up_values(&build_service(&tv, Arc::new(AllowAllPermissions)), &session)
            .await
            .expect("the catch-up succeeds after the run");
        assert!(values.is_empty(), "and carries no value: {values:?}");
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// A2, A3: a lost layer with no backup, one with a usable backup
// ═════════════════════════════════════════════════════════════════════════════

/// **A2** (AC2). A lost layer with no backup: its mapping and its
/// layers-index member are removed, its object is on the dead-object set,
/// its session's backup is rewritten without it (the kept layer still
/// mapped), and the hold is cleared (exit 0), with one AL5 WARN naming it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a2_accept_loss_discards_a_lost_layer_with_no_backup_and_rewrites_its_session_backup() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        lost_layer_fixture(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let run = accept_loss(&tv, &config, &store).await;

        assert_eq!(run.exit, JobExit::Done, "{}", run.captured.text);
        assert_eq!(mapping(&tv, LS, "lost.usda").await, None, "unmapped");
        assert!(!indexed(&tv, LS, "lost.usda").await, "out of the index");
        assert_eq!(
            mapping(&tv, LS, "kept.usda").await.as_deref(),
            Some("101"),
            "the kept layer stays"
        );
        assert!(indexed(&tv, LS, "kept.usda").await);
        assert_eq!(dead_set(&tv).await, BTreeSet::from([LS_LOST]));
        let backup = session_backup_of(&store, LS)
            .await
            .expect("the session backup");
        assert_eq!(
            backup.layer_mappings,
            BTreeMap::from([("kept.usda".to_owned(), LS_KEPT)]),
            "the session backup is rewritten without the lost layer"
        );
        let (_lost, report) = run.the_discard();
        assert_eq!(
            report.layers_discarded,
            vec![(LS.to_owned(), "lost.usda".to_owned(), LS_LOST)]
        );
        assert_eq!(report.rewritten, vec![LS.to_owned()]);
        assert_eq!(run.line(AL5_LAYER)["level"], "WARN");
        assert!(!marker_held(&tv).await, "un-held");
        assert_eq!(run.lines(UNHELD_LINE).len(), 1);
    })
    .await;
}

/// **A3** (AC3). A lost layer whose backup is usable, and no restore ran
/// before: accept-loss restores it as its backup decodes and discards
/// nothing — its mapping kept, nothing registered — and clears the hold.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a3_accept_loss_restores_a_lost_layer_with_a_usable_backup_and_discards_nothing() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        lost_layer_fixture(&tv, &store).await;
        let lost = base(2, 0x03, &[]);
        put_base(store.as_ref(), LS_LOST, LS, "lost.usda", 30, lost.clone()).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let run = accept_loss(&tv, &config, &store).await;

        assert_eq!(run.exit, JobExit::Done, "{}", run.captured.text);
        assert_restored(
            &tv,
            LS_LOST,
            &expected_state(LS_LOST, 30, &lost, &[], 30 + BAND),
        )
        .await;
        let (lost_set, report) = run.the_discard();
        assert!(lost_set.is_empty(), "nothing is lost: {lost_set:?}");
        assert_eq!(*report, DiscardReport::default());
        assert_eq!(mapping(&tv, LS, "lost.usda").await.as_deref(), Some("102"));
        assert!(dead_set(&tv).await.is_empty(), "nothing registered");
        assert!(!marker_held(&tv).await);
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// A4: no backup store
// ═════════════════════════════════════════════════════════════════════════════

/// **A4** (AC4, PD-5). With the `Disabled` and with the `InMemory` adapter
/// the Job never calls the backup store it was handed; the lost layer
/// (held `NoBackupStore`) is discarded and its object registered on the
/// dead-object set all the same, no session backup is rewritten, and the
/// hold is cleared (exit 0).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a4_accept_loss_with_no_backup_store_discards_and_registers_the_lost_layer_unaided() {
    within_test_budget(async {
        for adapter in [CheckpointAdapter::Disabled, CheckpointAdapter::InMemory] {
            let tv = TestValkey::setup().await;
            let store = WriterStore::new();
            seed_session_record(&tv, LS).await;
            seed_complete(&tv, LS_KEPT, LS, "kept.usda", 50, &[]).await;
            seed_mapping(&tv, LS, "kept.usda", LS_KEPT).await;
            seed_mapping(&tv, LS, "lost.usda", LS_LOST).await;
            let mut config = crate::common::test_config();
            config.checkpoint_adapter = adapter;
            hold(&tv, &config).await;
            let calls_before = store.calls();

            let run = accept_loss(&tv, &config, &store).await;

            assert_eq!(
                run.exit,
                JobExit::Done,
                "{adapter:?}: {}",
                run.captured.text
            );
            assert_eq!(
                store.calls(),
                calls_before,
                "{adapter:?}: no call of the backup store"
            );
            assert_eq!(
                dead_set(&tv).await,
                BTreeSet::from([LS_LOST]),
                "{adapter:?}: registered on every adapter"
            );
            assert_eq!(mapping(&tv, LS, "lost.usda").await, None, "{adapter:?}");
            let (_lost, report) = run.the_discard();
            assert_eq!(report.layers_discarded.len(), 1, "{adapter:?}");
            assert!(
                report.rewritten.is_empty(),
                "{adapter:?}: no backup to rewrite"
            );
            assert!(!marker_held(&tv).await, "{adapter:?}");
        }
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// A5, A6: the StorageApi fake
// ═════════════════════════════════════════════════════════════════════════════

/// A StorageApi adapter over the stateful fake (which answers an Enumerate
/// of an empty prefix with 404, as storage-api does), with the fake's
/// server and the adapter's token directory, both kept alive by the caller.
async fn fake_storage_api(
    name: &str,
) -> (
    Arc<dyn CheckpointStore>,
    wiremock::MockServer,
    tempfile::TempDir,
) {
    let server = wiremock::MockServer::start().await;
    let state: FakeStorageState = Arc::new(Mutex::new(Default::default()));
    mount_fake_storage_routes(&server, &state).await;
    let (adapter, token_dir) = build_fake_backed_adapter(&server, name);
    (Arc::new(adapter), server, token_dir)
}

/// **A5** (AC5, defer-6(c)). Over the StorageApi fake, three stores each
/// reach exit 0 with the hold cleared: (i) session backups with no layer
/// backup; (ii) a layer backup with no session backup (its session's record
/// surviving), restored; (iii) the counters record alone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a5_accept_loss_on_a_storage_api_store_with_an_empty_prefix_exits_0() {
    within_test_budget(async {
        for case in [
            "(i) sessions without layers",
            "(ii) layers without sessions",
            "(iii) the counters record only",
        ] {
            let tv = TestValkey::setup().await;
            let (store, _server, _token_dir) = fake_storage_api("accept-loss-6c").await;
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

            let run = accept_loss_with(&tv, &config, store, false, |b| b).await;

            assert_eq!(run.exit, JobExit::Done, "{case}: {}", run.captured.text);
            let (lost, _report) = run.the_discard();
            assert!(lost.is_empty(), "{case}: nothing is lost: {lost:?}");
            assert!(!marker_held(&tv).await, "{case}: the hold is cleared");
            assert!(dead_set(&tv).await.is_empty(), "{case}");
        }
    })
    .await;
}

/// **A5**, a backup store at the wrong address (defer-6(c)'s other side): a
/// StorageApi store holding only the counters record, while the store holds
/// a stamped survivor its listing does not show. The listing contradicts the
/// store: the Job writes nothing and exits 3 — the lost layer beside it is
/// not discarded, nothing registered, the store as it was.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a5_accept_loss_on_a_backup_store_at_the_wrong_address_exits_3_and_discards_nothing() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let (store, _server, _token_dir) = fake_storage_api("accept-loss-wrong").await;
        put_counters(store.as_ref()).await;
        seed_session_record(&tv, "s-w").await;
        seed_complete(
            &tv,
            851,
            "s-w",
            "k.usda",
            10,
            &[("last_checkpoint_seq", "7")],
        )
        .await;
        seed_mapping(&tv, "s-w", "k.usda", 851).await;
        seed_mapping(&tv, "s-w", "lost.usda", 852).await;
        let config = writer_config();
        hold(&tv, &config).await;
        let before = keyspace(&tv).await;

        let run = accept_loss_with(&tv, &config, store, false, |b| b).await;

        assert_eq!(run.exit, JobExit::StaysHeld, "{}", run.captured.text);
        let refusal = run.line(
            "the accept-loss Job wrote nothing and exits 3; the service stays held: the backup \
             store's listing contradicts the store",
        );
        assert_eq!(refusal["level"], "ERROR");
        assert!(
            run.discards.is_empty() && run.plans.is_empty(),
            "no body ran"
        );
        assert_eq!(
            keyspace_writes(&before, &keyspace(&tv).await, WriteRules::STRICT),
            Vec::<String>::new(),
            "nothing written"
        );
    })
    .await;
}

/// **A6** (AC6). A StorageApi store whose Enumerate route answers 404 while
/// its by-address routes work lists empty; at `T_safe` the frame's
/// list-route canary keeps the service held: exit 3 with the route named,
/// the body never run, and the lost layer not discarded — the store as it
/// was.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a6_accept_loss_whose_listing_route_answers_404_exits_3_and_discards_nothing() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let (store, server, _token_dir) = fake_storage_api("accept-loss-404").await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path_regex(
                r"^/v1alpha/fileobject/data-objects/",
            ))
            .respond_with(wiremock::ResponseTemplate::new(404))
            .with_priority(1)
            .mount(&server)
            .await;
        put_counters(store.as_ref()).await;
        put_base(store.as_ref(), 811, "s-6", "l.usda", 5, base(1, 0x81, &[])).await;
        seed_session_record(&tv, "s-6").await;
        seed_mapping(&tv, "s-6", "l.usda", 811).await;
        let mut config = writer_config();
        config.checkpoint_storage_api_not_found_cache_secs = T_SAFE_IN.as_secs();
        hold(&tv, &config).await;
        let before = keyspace(&tv).await;

        let run = accept_loss_with(&tv, &config, store, false, |b| b).await;

        assert_eq!(run.exit, JobExit::StaysHeld, "{}", run.captured.text);
        assert_eq!(
            run.line(&format!(
                "the accept-loss Job exits 3 and the service stays held: {}",
                live_edit::dr::job::absence::LIST_ROUTE_CAUSE
            ))["level"],
            "ERROR"
        );
        assert!(
            run.discards.is_empty() && run.plans.is_empty(),
            "no body ran"
        );
        assert_eq!(
            keyspace_writes(&before, &keyspace(&tv).await, WriteRules::STRICT),
            Vec::<String>::new(),
            "nothing written"
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// A7, A8: PD-3, the backup store read only at or after T_safe
// ═════════════════════════════════════════════════════════════════════════════

/// **A7** (AC7a, PD-3; fail-on-old). A lost layer whose backup the listing
/// shows only from `T_safe` on (it lists nothing before): the accept-loss
/// Job waits for `T_safe` before its first listing (logging why), so it
/// restores the layer and discards nothing — no re-listing needed; the
/// first call of every time-switched backup-store method came at or after
/// `T_safe`. With the wait dropped, the first listing comes before
/// `T_safe`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a7_accept_loss_lists_only_at_t_safe_so_a_backup_listed_from_then_on_is_restored() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        store.use_the_store_clock(&tv);
        let backup = base(2, 0x07, &[]);
        put_base(store.as_ref(), 701, "s-7", "l7.usda", 30, backup.clone()).await;
        put_counters(store.as_ref()).await;
        seed_session_record(&tv, "s-7").await;
        seed_mapping(&tv, "s-7", "l7.usda", 701).await;
        let mut config = writer_config();
        config.checkpoint_storage_api_not_found_cache_secs = T_SAFE_IN.as_secs();
        let t_safe = hold_returning_t_safe(&tv, &config).await;
        store.list_nothing_until(t_safe);

        let run = accept_loss(&tv, &config, &store).await;

        assert_eq!(run.exit, JobExit::Done, "{}", run.captured.text);
        assert_restored(&tv, 701, &expected_state(701, 30, &backup, &[], 30 + BAND)).await;
        assert!(dead_set(&tv).await.is_empty(), "nothing discarded");
        let first = store.first_listing_at_ms().expect("the store was listed");
        assert!(
            first >= t_safe,
            "the first listing, at {first} ms, is at or after T_safe at {t_safe} ms"
        );
        assert_first_calls_at_or_after(&store, t_safe, "list_layer_artifacts");
        let (plan, _report) = run.the_plan();
        assert!(plan.epoch.at_or_after_t_safe);
        assert!(!plan.epoch.relisted_at_t_safe, "no re-listing was needed");
        assert!(
            waits_for(&run.captured, PD3_WAIT) >= 1,
            "the wait names PD-3:\n{}",
            run.captured.text
        );
    })
    .await;
}

/// Assert the first call of every time-switched method of `store` that was
/// called came at or after `t_safe` on the store's clock, and that
/// `expected` was among them.
fn assert_first_calls_at_or_after(store: &WriterStore, t_safe: u64, expected: &str) {
    let first_calls = store.first_calls_at_ms();
    assert!(
        first_calls.contains_key(expected),
        "{expected} was called: {first_calls:?}"
    );
    for (method, at) in &first_calls {
        assert!(
            *at >= t_safe,
            "{method}'s first call, at {at} ms, is at or after T_safe at {t_safe} ms"
        );
    }
}

/// The session A7b's backup store lists only from `T_safe` on, and its
/// complete layer's object.
const A7B_SESSION: &str = "s-7b";
const A7B_OBJECT: u64 = 702;

/// **A7b** (PD-3's motivating case; fail-on-old). A session with no record,
/// a complete layer and a backup the session listing shows only from
/// `T_safe` on, beside another session's backup listed throughout (so the
/// listing is never empty): the accept-loss Job lists only at or after
/// `T_safe`, finds the backup, restores the session — its record back, its
/// layer still mapped — and registers nothing. Every time-switched call
/// came at or after `T_safe`. With the wait dropped, the session's absence
/// from the first listing is trusted: it is discarded as lost and its
/// object registered.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a7b_accept_loss_lists_sessions_only_at_t_safe_so_a_session_listed_from_then_on_is_restored(
) {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        store.use_the_store_clock(&tv);
        put_session(store.as_ref(), &session_backup("s-7o", &[], &[])).await;
        put_session(
            store.as_ref(),
            &session_backup(A7B_SESSION, &[("l.usda", A7B_OBJECT)], &[]),
        )
        .await;
        put_counters(store.as_ref()).await;
        seed_complete(&tv, A7B_OBJECT, A7B_SESSION, "l.usda", 10, &[]).await;
        seed_mapping(&tv, A7B_SESSION, "l.usda", A7B_OBJECT).await;
        let mut config = writer_config();
        config.checkpoint_storage_api_not_found_cache_secs = T_SAFE_IN.as_secs();
        let t_safe = hold_returning_t_safe(&tv, &config).await;
        store.hide_session_until(A7B_SESSION, t_safe);

        let run = accept_loss(&tv, &config, &store).await;

        assert_eq!(run.exit, JobExit::Done, "{}", run.captured.text);
        assert!(dead_set(&tv).await.is_empty(), "nothing registered");
        assert!(
            exists(&tv, tv.key_schema().session_meta(A7B_SESSION)).await,
            "the session's record is restored"
        );
        assert_eq!(
            mapping(&tv, A7B_SESSION, "l.usda").await,
            Some(A7B_OBJECT.to_string()),
            "its layer stays mapped"
        );
        let (lost, _report) = run.the_discard();
        assert!(lost.sessions.is_empty(), "no lost session: {lost:?}");
        assert_first_calls_at_or_after(&store, t_safe, "list_session_checkpoint_ids");
    })
    .await;
}

/// **A8** (AC7b, PD-3; fail-on-old). A kept layer's value whose bytes only
/// the backup store holds, probing absent before `T_safe` and present from
/// it on: the accept-loss Job's first probe is at or after `T_safe`, so the
/// value is covered and keeps its attribute — its row and its `obj:ext`
/// member stay, nothing is removed; the first call of every time-switched
/// backup-store method came at or after `T_safe`. The layer's older backup
/// is listed, so the listing is not empty and nothing else makes the
/// planner wait for `T_safe`. With accept-loss's wait dropped, the first
/// probe comes before `T_safe`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a8_accept_loss_probes_only_at_t_safe_so_a_value_present_from_then_on_is_kept() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        store.use_the_store_clock(&tv);
        put_counters(store.as_ref()).await;
        let (session, object_id, hash, bytes) = layer_with_a_lost_value(&tv, "a8", 0xA8).await;
        put_value(&store, &hash, &bytes).await;
        put_base(
            store.as_ref(),
            object_id,
            &session,
            VALUE_LAYER,
            1,
            base(1, 0xA8, &[]),
        )
        .await;
        let mut config = writer_config();
        config.checkpoint_storage_api_not_found_cache_secs = T_SAFE_IN.as_secs();
        let t_safe = hold_returning_t_safe(&tv, &config).await;
        store.value_absent_until(&hash, t_safe);

        let run = accept_loss(&tv, &config, &store).await;

        assert_eq!(run.exit, JobExit::Done, "{}", run.captured.text);
        assert!(
            exists(&tv, field_key(&tv, object_id, "1", "ext")).await,
            "the attribute keeps its value"
        );
        assert!(ext_indexed(&tv, object_id, &hash).await);
        let (_lost, report) = run.the_discard();
        assert!(report.values.is_empty(), "nothing removed: {report:?}");
        let first = store.first_probe_at_ms().expect("the value was probed");
        assert!(
            first >= t_safe,
            "the first probe, at {first} ms, is at or after T_safe at {t_safe} ms"
        );
        assert_first_calls_at_or_after(&store, t_safe, "ext_blob_exists");
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// A9: P51-c, a deleted mapping Step B replaces
// ═════════════════════════════════════════════════════════════════════════════

/// **A9, P51-c** (fail-on-old). A surviving mapping `L → X` to a deleted
/// object, and a session backup mapping `L` to the newer, listed `Y`: the
/// restore phase writes `L → Y` and restores `Y`, and accept-loss discards
/// nothing for `L` — no lost layer, no mapping left as found, no AL8 — so
/// `L → Y` stays, only `X` is on the dead-object set, and the run exits 0.
/// With the planner's defer-51 fix and both plan checks' rule 2 removed,
/// the stale item reaches the unmap, which leaves `L → Y` and logs AL8.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a9_p51c_accept_loss_never_discards_a_deleted_mapping_step_b_replaces() {
    const X: u64 = 5_101;
    const Y: u64 = 5_102;
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let newer = base(1, 0x51, &[]);
        put_session(
            store.as_ref(),
            &session_backup("s-51", &[("l.usda", Y)], &[]),
        )
        .await;
        put_base(store.as_ref(), Y, "s-51", "l.usda", 30, newer.clone()).await;
        put_counters(store.as_ref()).await;
        seed_session_record(&tv, "s-51").await;
        seed_mapping(&tv, "s-51", "l.usda", X).await;
        seed_dead(&tv, X).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let run = accept_loss(&tv, &config, &store).await;

        assert_eq!(run.exit, JobExit::Done, "{}", run.captured.text);
        let (lost, report) = run.the_discard();
        assert!(
            lost.layers.is_empty(),
            "no lost layer for l.usda: {:?}",
            lost.layers
        );
        assert!(
            report.left.is_empty(),
            "no mapping left as found: {report:?}"
        );
        assert!(run.lines(AL8_LEFT).is_empty(), "no AL8");
        assert_eq!(
            mapping(&tv, "s-51", "l.usda").await,
            Some(Y.to_string()),
            "L → Y stays"
        );
        assert_restored(&tv, Y, &expected_state(Y, 30, &newer, &[], 30 + BAND)).await;
        assert_eq!(
            dead_set(&tv).await,
            BTreeSet::from([X]),
            "Y is never registered"
        );
    })
    .await;
}

/// **A10, P55-a across a retry** (B1; fail-on-old). A writer that does not
/// honour the hold completes the lost layer's object after the plan found
/// it lost (the plan observer writes its header, as such a writer would,
/// before any discard): the Job reads that header before it registers
/// anything (D0), logs AL9 and exits 1 with nothing written — the header and
/// the mapping as the writer left them, the dead-object set empty, and no
/// AL9 companion WARN, which only D2's stop after a registration logs. The
/// automatic retry, the writer still active, plans the complete, mapped
/// object as a survivor and exits 0: its header, its mapping and the
/// dead-object set are unchanged after it. With D0's loop moved below the
/// registration, the first run leaves the object registered; with D0's loop
/// removed, D2's re-read stops the first run after registering the object
/// and removing it again, and the companion WARN is logged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a10_p55a_accept_loss_stopped_by_a_completed_object_registers_nothing_so_its_retry_keeps_it(
) {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        lost_layer_fixture(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;
        let rogue_writer = || {
            let client = tv.client().clone();
            let key = tv.key_schema().obj_header(&LS_LOST.to_string());
            move |body: AcceptLossBody| {
                body.with_plan_observer(Arc::new(move |_plan, _report| {
                    let (client, key) = (client.clone(), key.clone());
                    block_on_spawned(&tokio::runtime::Handle::current(), async move {
                        let _: i64 = client
                            .pool
                            .hset(
                                key,
                                [
                                    ("seq", "20"),
                                    ("created_by", "user"),
                                    ("session_id", LS),
                                    ("layer_name", "lost.usda"),
                                ],
                            )
                            .await
                            .expect("the rogue writer's HSET");
                    })
                    .expect("the rogue writer's write ran");
                }))
            }
        };

        let run =
            accept_loss_with(&tv, &config, Arc::clone(&store) as _, false, rogue_writer()).await;

        assert_eq!(run.exit, JobExit::Failed, "{}", run.captured.text);
        let al9 = run.line(AL9_COMPLETE);
        assert_eq!(al9["level"], "ERROR");
        assert_eq!(al9["fields"]["object_id"], LS_LOST);
        assert!(dead_set(&tv).await.is_empty(), "nothing registered");
        assert!(
            run.lines(AL9_REMOVED).is_empty() && run.lines(AL9_REMOVED_NOTHING).is_empty(),
            "no AL9 companion: D0 stopped the run, before any registration; capture:\n{}",
            run.captured.text
        );
        assert_eq!(
            mapping(&tv, LS, "lost.usda").await,
            Some(LS_LOST.to_string()),
            "the mapping is kept"
        );
        let completed = header(&tv, LS_LOST).await;
        assert_eq!(completed["seq"], "20", "the header is kept");
        assert!(marker_held(&tv).await);

        let retry =
            accept_loss_with(&tv, &config, Arc::clone(&store) as _, false, rogue_writer()).await;

        assert_eq!(
            retry.exit,
            JobExit::Done,
            "the retry plans the completed object as a survivor: {}",
            retry.captured.text
        );
        assert_eq!(
            header(&tv, LS_LOST).await,
            completed,
            "the retry keeps the header: {}",
            retry.captured.text
        );
        assert_eq!(
            mapping(&tv, LS, "lost.usda").await,
            Some(LS_LOST.to_string()),
            "and the mapping"
        );
        assert!(dead_set(&tv).await.is_empty(), "and registers nothing");
        assert!(run.lines(AL5_LAYER).is_empty() && retry.lines(AL5_LAYER).is_empty());
    })
    .await;
}

/// What the after-register hook saw: the objects D1 registered, per call.
type Registered = Arc<Mutex<Vec<Vec<u64>>>>;

/// A writer that does not honour the hold, acting in the window D0 does not
/// cover: the body with a hook, run after the discards' registration (D1)
/// and before their header re-read (D2), that completes `object_id`'s header
/// as `layer` of `session` at seq 20 and records what D1 registered into
/// `registered`.
fn completes_after_register(
    tv: &TestValkey,
    object_id: u64,
    session: &'static str,
    layer: &'static str,
    registered: &Registered,
) -> impl FnOnce(AcceptLossBody) -> AcceptLossBody {
    let client = tv.client().clone();
    let key = tv.key_schema().obj_header(&object_id.to_string());
    let registered = Arc::clone(registered);
    move |body| {
        body.with_after_discard_register_for_test(Arc::new(move |objects: &[u64]| {
            registered.lock().unwrap().push(objects.to_vec());
            let (client, key) = (client.clone(), key.clone());
            block_on_spawned(&tokio::runtime::Handle::current(), async move {
                let _: i64 = client
                    .pool
                    .hset(
                        key,
                        [
                            ("seq", "20"),
                            ("created_by", "user"),
                            ("session_id", session),
                            ("layer_name", layer),
                        ],
                    )
                    .await
                    .expect("the rogue writer's HSET");
            })
            .expect("the rogue writer's write ran");
        }))
    }
}

/// **A10b, P55-a in the late window** (B1; fail-on-old). A writer that does
/// not honour the hold completes the lost layer's object after the discards
/// registered it (D1) and before their header re-read (D2): the Job logs
/// AL9 and, before exiting 1, removes the object from the dead-object set
/// and logs AL9's companion WARN saying so; the header and the mapping stay
/// as the writer left them. The automatic retry plans the complete, mapped
/// object as a survivor and exits 0, its header and mapping kept and the
/// dead-object set still empty. With D2's removal removing nothing, the
/// object stays registered after the first run and the retry wipes its
/// header unread, as a mapping to a deleted object.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a10b_p55a_accept_loss_stopped_after_its_registration_removes_it_so_its_retry_keeps_the_object(
) {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        lost_layer_fixture(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;
        let registered = Registered::default();

        let run = accept_loss_with(
            &tv,
            &config,
            Arc::clone(&store) as _,
            false,
            completes_after_register(&tv, LS_LOST, LS, "lost.usda", &registered),
        )
        .await;

        assert_eq!(run.exit, JobExit::Failed, "{}", run.captured.text);
        assert_eq!(
            *registered.lock().unwrap(),
            vec![vec![LS_LOST]],
            "the writer completed the header after D1 registered the object"
        );
        let al9 = run.line(AL9_COMPLETE);
        assert_eq!(al9["level"], "ERROR");
        assert_eq!(al9["fields"]["object_id"], LS_LOST);
        let companion = run.line(AL9_REMOVED);
        assert_eq!(companion["level"], "WARN");
        assert_eq!(
            companion["fields"]["message"],
            format!(
                "the accept-loss Job removed objects {LS_LOST} from the dead-object set before \
                 exiting: this run registered them as deleted and wiped none of them, so its \
                 retry reads their headers before it registers them again"
            )
        );
        assert_eq!(
            mapping(&tv, LS, "lost.usda").await,
            Some(LS_LOST.to_string()),
            "the mapping is kept"
        );
        let completed = header(&tv, LS_LOST).await;
        assert_eq!(completed["seq"], "20", "the header is kept");
        assert!(marker_held(&tv).await);
        // Read now, asserted after the retry, so a registration left behind
        // shows what the retry makes of it.
        let dead_after_run = dead_set(&tv).await;

        let retry = accept_loss(&tv, &config, &store).await;

        assert_eq!(
            header(&tv, LS_LOST).await,
            completed,
            "the retry keeps the header, never wiping it: {}",
            retry.captured.text
        );
        assert!(
            dead_after_run.is_empty(),
            "the first run removed its registration: {dead_after_run:?}"
        );
        assert_eq!(
            retry.exit,
            JobExit::Done,
            "the retry plans the completed object as a survivor: {}",
            retry.captured.text
        );
        assert_eq!(
            mapping(&tv, LS, "lost.usda").await,
            Some(LS_LOST.to_string()),
            "and the mapping"
        );
        assert!(dead_set(&tv).await.is_empty(), "and registers nothing");
        assert!(run.lines(AL5_LAYER).is_empty() && retry.lines(AL5_LAYER).is_empty());
    })
    .await;
}

/// A10c's session, its kept survivor, and its lost layers in the order the
/// discards take them (by layer name): `a.usda` → [`A10C_WIPED`], `b.usda`
/// → [`A10C_COMPLETED`], `c.usda` → [`A10C_UNREACHED`].
const A10C: &str = "s-al9";
const A10C_KEPT: u64 = 5_200;
const A10C_WIPED: u64 = 5_201;
const A10C_COMPLETED: u64 = 5_202;
const A10C_UNREACHED: u64 = 5_203;

/// **A10c, D2's removal at a late AL9 is every unwiped registration**
/// (B1; fail-on-old). Three lost layers, all registered at D1; a writer that
/// does not honour the hold completes the second's object after D1: D2
/// wipes the first, stops on the second with AL9, and removes the second
/// and the third — registered, not yet wiped — from the dead-object set,
/// while the first, already wiped, stays registered, as a kill after its
/// wipe would leave it. The companion WARN names the two removed. No
/// mapping is removed (D3 is not reached). With the removal limited to
/// AL9's own object, the third stays on the dead-object set.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a10c_p55a_accept_loss_stopped_after_its_registration_removes_every_object_it_has_not_wiped(
) {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let lost = [
            ("a.usda", A10C_WIPED),
            ("b.usda", A10C_COMPLETED),
            ("c.usda", A10C_UNREACHED),
        ];
        let mut mapped = vec![("kept.usda", A10C_KEPT)];
        mapped.extend(lost);
        put_session(store.as_ref(), &session_backup(A10C, &mapped, &[])).await;
        put_base(
            store.as_ref(),
            A10C_KEPT,
            A10C,
            "kept.usda",
            40,
            base(1, 0x0C, &[]),
        )
        .await;
        put_counters(store.as_ref()).await;
        seed_session_record(&tv, A10C).await;
        seed_complete(&tv, A10C_KEPT, A10C, "kept.usda", 50, &[]).await;
        for (layer, object_id) in mapped {
            seed_mapping(&tv, A10C, layer, object_id).await;
        }
        let config = writer_config();
        hold(&tv, &config).await;
        let registered = Registered::default();

        let run = accept_loss_with(
            &tv,
            &config,
            Arc::clone(&store) as _,
            false,
            completes_after_register(&tv, A10C_COMPLETED, A10C, "b.usda", &registered),
        )
        .await;

        assert_eq!(run.exit, JobExit::Failed, "{}", run.captured.text);
        assert_eq!(
            *registered.lock().unwrap(),
            vec![vec![A10C_WIPED, A10C_COMPLETED, A10C_UNREACHED]],
            "D1 registered all three"
        );
        assert_eq!(
            run.line(AL9_COMPLETE)["fields"]["object_id"],
            A10C_COMPLETED
        );
        let companion = run.line(AL9_REMOVED);
        assert_eq!(companion["level"], "WARN");
        assert_eq!(
            companion["fields"]["message"],
            format!(
                "the accept-loss Job removed objects {A10C_COMPLETED}, {A10C_UNREACHED} from \
                 the dead-object set before exiting: this run registered them as deleted and \
                 wiped none of them, so its retry reads their headers before it registers them \
                 again"
            )
        );
        assert_eq!(
            dead_set(&tv).await,
            BTreeSet::from([A10C_WIPED]),
            "only the wiped object stays registered"
        );
        for (layer, object_id) in lost {
            assert_eq!(
                mapping(&tv, A10C, layer).await,
                Some(object_id.to_string()),
                "{layer}'s mapping is kept"
            );
        }
        assert_eq!(
            header(&tv, A10C_COMPLETED).await["seq"],
            "20",
            "the completed header is kept"
        );
        assert!(marker_held(&tv).await);
    })
    .await;
}

/// **A10d, a registration another writer made stays** (B1; fail-on-old).
/// Another writer registers the lost layer's object as deleted after the
/// plan (the plan observer) and, after the discards' registration (D1),
/// completes its header: D2 logs AL9 and exits 1, and the object stays on
/// the dead-object set, since D1's read found it there and this run's
/// registration added nothing. The companion WARN says that nothing was
/// removed and why; the header and the mapping are kept. With D1 taking
/// every member as added (its read ignored), the object is removed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a10d_p55a_accept_loss_stopped_after_its_registration_keeps_one_another_writer_made() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        lost_layer_fixture(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;
        let registered = Registered::default();
        let completes = completes_after_register(&tv, LS_LOST, LS, "lost.usda", &registered);
        let registers = {
            let client = tv.client().clone();
            let key = tv.key_schema().checkpoint_reconcile_dead();
            move |body: AcceptLossBody| {
                body.with_plan_observer(Arc::new(move |_plan, _report| {
                    let (client, key) = (client.clone(), key.clone());
                    block_on_spawned(&tokio::runtime::Handle::current(), async move {
                        let _: i64 = client
                            .pool
                            .zadd(key, None, None, false, false, (0.0, LS_LOST.to_string()))
                            .await
                            .expect("the other writer's ZADD");
                    })
                    .expect("the other writer's registration ran");
                }))
            }
        };

        let run = accept_loss_with(&tv, &config, Arc::clone(&store) as _, false, move |body| {
            completes(registers(body))
        })
        .await;

        assert_eq!(run.exit, JobExit::Failed, "{}", run.captured.text);
        assert_eq!(
            *registered.lock().unwrap(),
            vec![vec![LS_LOST]],
            "the writer completed the header after D1"
        );
        assert_eq!(run.line(AL9_COMPLETE)["fields"]["object_id"], LS_LOST);
        let companion = run.line(AL9_REMOVED_NOTHING);
        assert_eq!(companion["level"], "WARN");
        assert_eq!(
            companion["fields"]["message"],
            format!(
                "the accept-loss Job removed nothing from the dead-object set before exiting: \
                 object {LS_LOST} stays on it: another writer registered it as deleted before \
                 this run did"
            )
        );
        assert!(run.lines(AL9_REMOVED).is_empty(), "nothing removed");
        assert_eq!(
            dead_set(&tv).await,
            BTreeSet::from([LS_LOST]),
            "the other writer's registration stays"
        );
        assert_eq!(
            mapping(&tv, LS, "lost.usda").await,
            Some(LS_LOST.to_string()),
            "the mapping is kept"
        );
        assert_eq!(
            header(&tv, LS_LOST).await["seq"],
            "20",
            "the header is kept"
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// A11: PD-1, the unattributed object
// ═════════════════════════════════════════════════════════════════════════════

/// The unattributed object: listed in the backup store, every generation
/// failing `Permanent`, under an empty session listing.
const UNATTRIBUTED: u64 = 1_101;
/// The lost layer beside it.
const A11_LOST: u64 = 1_102;

/// A11's fixture: [`UNATTRIBUTED`]'s generation at 30 listed and failing
/// `Permanent` (a 403), no session backup listed, and a lost layer with no
/// backup, `lost.usda` → [`A11_LOST`], in session `s-11` whose record
/// survives.
async fn unattributed_fixture(tv: &TestValkey, store: &WriterStore) {
    put_base(store, UNATTRIBUTED, "s19", "l.usda", 30, base(1, 0x11, &[])).await;
    store.permanent(UNATTRIBUTED, 30);
    put_counters(store).await;
    seed_session_record(tv, "s-11").await;
    seed_mapping(tv, "s-11", "lost.usda", A11_LOST).await;
}

/// **A11** (PD-1; fail-on-old). An object listed in the backup store that no
/// read could attribute to a session and layer is unconfirmed: the Job
/// discards nothing — the lost layer beside it keeps its mapping, the
/// dead-object set stays empty — recounts and exits 3 with AL3, the marker
/// kept. With the object discarded as a lost layer instead, the run does not
/// refuse: it discards and exits 0.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a11_accept_loss_refuses_every_discard_on_an_object_it_cannot_attribute() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        unattributed_fixture(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let run = accept_loss(&tv, &config, &store).await;

        assert_eq!(run.exit, JobExit::StaysHeld, "{}", run.captured.text);
        assert!(
            dead_set(&tv).await.is_empty(),
            "the dead-object set is unchanged"
        );
        assert_eq!(
            mapping(&tv, "s-11", "lost.usda").await,
            Some(A11_LOST.to_string()),
            "nothing discarded"
        );
        let refusal = run.line(AL3_REFUSAL);
        assert_eq!(refusal["level"], "ERROR");
        assert_eq!(refusal["fields"]["unconfirmed"], 1);
        assert!(run.discards.is_empty(), "no discard ran");
        let finish = run.the_finish();
        assert_eq!(finish.outcome, PlanOutcome::StaysHeld);
        assert!(finish.recount.is_some(), "it recounted");
        assert!(marker_held(&tv).await);
    })
    .await;
}

/// **A11 under the second confirm** (option E, ruling 1). The same
/// unattributed object is reported — one AL12 WARN naming it and its error
/// — and never registered dead, so its backups are left; the lost layer
/// beside it is discarded, and the run exits 0.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a11_e_accept_loss_with_the_second_confirm_reports_the_unattributed_object_and_never_registers_it(
) {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        unattributed_fixture(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let run = accept_loss_discarding_unconfirmed(&tv, &config, &store).await;

        assert_eq!(run.exit, JobExit::Done, "{}", run.captured.text);
        assert_eq!(
            dead_set(&tv).await,
            BTreeSet::from([A11_LOST]),
            "the unattributed object is never registered"
        );
        let warn = run.line(AL12_UNCONFIRMED);
        assert_eq!(warn["level"], "WARN");
        assert_eq!(
            warn["fields"]["item"],
            format!("object {UNATTRIBUTED}, listed in the backup store")
        );
        assert!(
            warn["fields"]["error"]
                .as_str()
                .is_some_and(|e| e.contains("403")),
            "the WARN names the error: {warn}"
        );
        assert!(
            !store
                .list_layer_checkpoints(UNATTRIBUTED)
                .await
                .expect("the listing")
                .is_empty(),
            "its backups are left"
        );
        assert_eq!(mapping(&tv, "s-11", "lost.usda").await, None);
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// A12, A12b: PD-2, a backup-store brownout
// ═════════════════════════════════════════════════════════════════════════════

/// The brownout fixture: the lost-layer fixture, and a service-made layer
/// whose value's bytes only the backup store holds, its probe failing
/// `Transient`. Returns the value layer's object and the value.
async fn brownout_fixture(tv: &TestValkey, store: &WriterStore) -> (u64, [u8; 32]) {
    lost_layer_fixture(tv, store).await;
    let (_session, object_id, hash, bytes) = layer_with_a_lost_value(tv, "a12", 0xB1).await;
    put_value(store, &hash, &bytes).await;
    store.fail_probes_of(&hash);
    (object_id, hash)
}

/// **A12** (PD-2; fail-on-old). A value whose every probe fails `Transient`
/// is unconfirmed, so the Job discards nothing — not even the proven-lost
/// layer: its mapping kept, nothing registered, the value's row and
/// `obj:ext` member kept — yet it recounts, the value's count exact at 1,
/// logs AL3 naming 1 item and exits 3, the marker kept. With the fault
/// cleared, the re-run discards the lost layer, keeps the now covered value
/// and exits 0. With the refusal gate dropped, the first run discards and
/// exits 0.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a12_accept_loss_refuses_every_discard_through_a_brownout_then_exits_0_after_it() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let (object_id, hash) = brownout_fixture(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let run = accept_loss(&tv, &config, &store).await;

        assert_eq!(run.exit, JobExit::StaysHeld, "{}", run.captured.text);
        assert_eq!(
            mapping(&tv, LS, "lost.usda").await,
            Some(LS_LOST.to_string()),
            "the proven-lost layer is not discarded"
        );
        assert!(dead_set(&tv).await.is_empty(), "nothing registered");
        assert!(exists(&tv, field_key(&tv, object_id, "1", "ext")).await);
        assert!(ext_indexed(&tv, object_id, &hash).await);
        let refusal = run.line(AL3_REFUSAL);
        assert_eq!(refusal["level"], "ERROR");
        assert_eq!(refusal["fields"]["unconfirmed"], 1);
        assert!(run.discards.is_empty(), "no discard ran");
        assert!(run.the_finish().recount.is_some(), "it recounted");
        assert_eq!(ref_count(&tv, &hash).await, Some(1), "the count is exact");
        assert!(marker_held(&tv).await);

        store.clear_probe_faults();
        let rerun = accept_loss(&tv, &config, &store).await;

        assert_eq!(
            rerun.exit,
            JobExit::Done,
            "the re-run: {}",
            rerun.captured.text
        );
        assert_eq!(mapping(&tv, LS, "lost.usda").await, None);
        assert_eq!(dead_set(&tv).await, BTreeSet::from([LS_LOST]));
        assert!(
            exists(&tv, field_key(&tv, object_id, "1", "ext")).await,
            "the covered value is kept"
        );
        assert!(!marker_held(&tv).await);
    })
    .await;
}

/// **A12b** (PD-2 option E). The same brownout under the second confirm:
/// the unconfirmed value is removed like an uncovered one (its row and
/// `obj:ext` member gone) beside the lost layer's discard; one AL12 WARN
/// names the value and the error it came from; AL4 and AL11 count it apart
/// (1 unconfirmed discarded and none kept, in their fields and verbatim
/// words); and the run exits 0.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a12b_accept_loss_with_the_second_confirm_discards_through_a_brownout_and_counts_it_apart()
{
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let (object_id, hash) = brownout_fixture(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let run = accept_loss_discarding_unconfirmed(&tv, &config, &store).await;

        assert_eq!(run.exit, JobExit::Done, "{}", run.captured.text);
        assert!(
            !exists(&tv, field_key(&tv, object_id, "1", "ext")).await,
            "the unconfirmed value is removed"
        );
        assert!(!ext_indexed(&tv, object_id, &hash).await);
        assert_eq!(mapping(&tv, LS, "lost.usda").await, None);
        assert_eq!(dead_set(&tv).await, BTreeSet::from([LS_LOST]));
        let warn = run.line(AL12_UNCONFIRMED);
        assert_eq!(warn["level"], "WARN");
        assert_eq!(
            warn["fields"]["item"],
            format!(
                "large value {} (referenced by 1 layers)",
                hash_prefix(&hash)
            )
        );
        assert!(
            warn["fields"]["error"]
                .as_str()
                .is_some_and(|e| e.contains("503")),
            "the WARN names the error it came from: {warn}"
        );
        let plan_line = run.line(AL4_PLAN);
        assert_eq!(
            (
                &plan_line["fields"]["unconfirmed"],
                &plan_line["fields"]["unconfirmed_kept"]
            ),
            (&serde_json::json!(1), &serde_json::json!(0))
        );
        assert_eq!(
            plan_line["fields"]["message"],
            "the accept-loss Job discards 1 layers, 0 sessions and 1 large values from 1 layers, \
             and rewrites 1 session backups; 1 layers are registered as deleted; 1 items among \
             these could not be confirmed lost and are discarded because it runs with --confirm \
             discard-unconfirmed-data"
        );
        let summary = run.line(AL11_SUMMARY);
        assert_eq!(
            (
                &summary["fields"]["unconfirmed"],
                &summary["fields"]["unconfirmed_kept"]
            ),
            (&serde_json::json!(1), &serde_json::json!(0))
        );
        assert_eq!(
            summary["fields"]["message"],
            "the accept-loss Job's summary: discarded 1 layers and 0 sessions, removed 1 large \
             values from 1 layers, left 0 mappings as found, registered 1 layers as deleted, \
             deleted 0 session backups and rewrote 1; 1 items among these could not be confirmed \
             lost and were discarded because it ran with --confirm discard-unconfirmed-data"
        );
        let (_lost, report) = run.the_discard();
        assert_eq!((report.unconfirmed, report.unconfirmed_kept), (1, 0));
        assert!(run.lines(AL3_REFUSAL).is_empty(), "no refusal");
        assert!(!marker_held(&tv).await);
    })
    .await;
}

/// **Option E, ruling 2.** A kept layer whose newer backup's fetch fails
/// `Permanent` is held as unknown with its survivor kept; under the second
/// confirm the run reports it (AL12, "the newer backups of …") and keeps
/// the survivor — its mapping and header as they were, nothing registered —
/// and exits 0.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e_accept_loss_with_the_second_confirm_keeps_a_survivor_whose_newer_backups_are_unknown() {
    const SURVIVOR: u64 = 1_701;
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        put_base(
            store.as_ref(),
            SURVIVOR,
            "s-17",
            "k.usda",
            60,
            base(1, 0x17, &[]),
        )
        .await;
        store.permanent(SURVIVOR, 60);
        put_counters(store.as_ref()).await;
        seed_session_record(&tv, "s-17").await;
        seed_complete(&tv, SURVIVOR, "s-17", "k.usda", 50, &[]).await;
        seed_mapping(&tv, "s-17", "k.usda", SURVIVOR).await;
        let config = writer_config();
        hold(&tv, &config).await;
        let header_before = header(&tv, SURVIVOR).await;

        let run = accept_loss_discarding_unconfirmed(&tv, &config, &store).await;

        assert_eq!(run.exit, JobExit::Done, "{}", run.captured.text);
        assert_eq!(
            run.line(AL12_UNCONFIRMED)["fields"]["item"],
            format!("the newer backups of layer k.usda of session s-17 (object {SURVIVOR})")
        );
        assert_eq!(
            mapping(&tv, "s-17", "k.usda").await,
            Some(SURVIVOR.to_string()),
            "the survivor stays mapped"
        );
        assert_eq!(header(&tv, SURVIVOR).await, header_before, "and as it was");
        assert!(dead_set(&tv).await.is_empty(), "nothing registered");
        let (lost, _report) = run.the_discard();
        assert!(lost.layers.is_empty(), "{:?}", lost.layers);
    })
    .await;
}

/// **Option E, the totals** (S2, C57; fail-on-old). Three unconfirmed items
/// under the second confirm: the unattributed object (ruling 1, reported,
/// never registered), a kept layer whose newer backup's fetch fails
/// `Permanent` (ruling 2, its survivor kept), and a value whose probe fails
/// `Transient` (removed). Each gets its AL12 WARN; AL4 and AL11 count the one
/// discarded apart from the two kept, in their fields and their words, and
/// the run exits 0. With every unconfirmed item counted as discarded, AL4
/// says 3 discarded and none kept.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e_accept_loss_counts_the_unconfirmed_items_it_discards_apart_from_those_it_keeps() {
    const SURVIVOR: u64 = 1_702;
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        unattributed_fixture(&tv, &store).await;
        put_base(
            store.as_ref(),
            SURVIVOR,
            "s-17e",
            "k.usda",
            60,
            base(1, 0x18, &[]),
        )
        .await;
        store.permanent(SURVIVOR, 60);
        seed_session_record(&tv, "s-17e").await;
        seed_complete(&tv, SURVIVOR, "s-17e", "k.usda", 50, &[]).await;
        seed_mapping(&tv, "s-17e", "k.usda", SURVIVOR).await;
        let (_session, object_id, hash, bytes) = layer_with_a_lost_value(&tv, "e-s2", 0xE2).await;
        put_value(&store, &hash, &bytes).await;
        store.fail_probes_of(&hash);
        let config = writer_config();
        hold(&tv, &config).await;

        let run = accept_loss_discarding_unconfirmed(&tv, &config, &store).await;

        assert_eq!(run.exit, JobExit::Done, "{}", run.captured.text);
        assert_eq!(run.lines(AL12_UNCONFIRMED).len(), 3, "one AL12 per item");
        assert!(
            !ext_indexed(&tv, object_id, &hash).await,
            "the unconfirmed value is removed"
        );
        assert_eq!(
            mapping(&tv, "s-17e", "k.usda").await,
            Some(SURVIVOR.to_string()),
            "the survivor is kept"
        );
        assert!(
            !dead_set(&tv).await.contains(&UNATTRIBUTED),
            "the unattributed object is never registered"
        );
        let plan_line = run.line(AL4_PLAN);
        assert_eq!(
            (
                &plan_line["fields"]["unconfirmed"],
                &plan_line["fields"]["unconfirmed_kept"]
            ),
            (&serde_json::json!(1), &serde_json::json!(2))
        );
        assert_eq!(
            plan_line["fields"]["message"],
            "the accept-loss Job discards 1 layers, 0 sessions and 1 large values from 1 layers, \
             and rewrites 1 session backups; 1 layers are registered as deleted; 1 items among \
             these could not be confirmed lost and are discarded because it runs with --confirm \
             discard-unconfirmed-data; 2 more could not be confirmed and are left as their WARN \
             says"
        );
        let summary = run.line(AL11_SUMMARY);
        assert_eq!(
            (
                &summary["fields"]["unconfirmed"],
                &summary["fields"]["unconfirmed_kept"]
            ),
            (&serde_json::json!(1), &serde_json::json!(2))
        );
        assert_eq!(
            summary["fields"]["message"],
            "the accept-loss Job's summary: discarded 1 layers and 0 sessions, removed 1 large \
             values from 1 layers, left 0 mappings as found, registered 1 layers as deleted, \
             deleted 0 session backups and rewrote 1; 1 items among these could not be confirmed \
             lost and were discarded because it ran with --confirm discard-unconfirmed-data; 2 \
             more could not be confirmed and were left as their WARN says"
        );
        let (_lost, report) = run.the_discard();
        assert_eq!((report.unconfirmed, report.unconfirmed_kept), (1, 2));
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// PD-2: a session backup whose read fails
// ═════════════════════════════════════════════════════════════════════════════

/// The session whose backup's read fails, and its complete layer's object.
const UNREAD: &str = "s-unread";
const UNREAD_OBJECT: u64 = 1_801;

/// [`UNREAD`]'s fixture: a listed, decodable session backup mapping
/// `u.usda` → [`UNREAD_OBJECT`], whose every read fails `Transient`, and in
/// the store that layer's complete header, its mapping and its
/// layers-index member; its session record survives when `record`.
async fn unreadable_session_fixture(tv: &TestValkey, store: &WriterStore, record: bool) {
    put_session(
        store,
        &session_backup(UNREAD, &[("u.usda", UNREAD_OBJECT)], &[]),
    )
    .await;
    store.fail_session_reads_of(UNREAD);
    put_counters(store).await;
    if record {
        seed_session_record(tv, UNREAD).await;
    }
    seed_complete(tv, UNREAD_OBJECT, UNREAD, "u.usda", 10, &[]).await;
    seed_mapping(tv, UNREAD, "u.usda", UNREAD_OBJECT).await;
}

/// **PD-2, a record-less session's backup read failing** (test review 1;
/// fail-on-old). A session with no record whose listed, decodable backup's
/// read fails `Transient` is unconfirmed, not lost: without the second
/// confirm the Job discards nothing — its layer's header and mapping kept,
/// nothing registered, its backup kept — logs AL3 counting it and exits 3.
/// Run again with the second confirm (the read still failing), it discards
/// the session (AL12 names it and its error; its object registered and
/// wiped, its backup deleted) and exits 0. With a failed read taken as an
/// undecodable backup, the first run discards the session and exits 0.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pd2_accept_loss_discards_a_session_whose_backup_read_failed_only_under_the_second_confirm()
{
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        unreadable_session_fixture(&tv, &store, false).await;
        let config = writer_config();
        hold(&tv, &config).await;
        let header_before = header(&tv, UNREAD_OBJECT).await;

        let run = accept_loss(&tv, &config, &store).await;

        assert_eq!(run.exit, JobExit::StaysHeld, "{}", run.captured.text);
        assert_eq!(run.line(AL3_REFUSAL)["fields"]["unconfirmed"], 1);
        assert!(dead_set(&tv).await.is_empty(), "nothing registered");
        assert_eq!(header(&tv, UNREAD_OBJECT).await, header_before);
        assert_eq!(
            mapping(&tv, UNREAD, "u.usda").await,
            Some(UNREAD_OBJECT.to_string())
        );
        assert!(store.holds_session(UNREAD).await, "its backup is kept");
        assert!(run.discards.is_empty(), "no discard ran");
        assert!(marker_held(&tv).await);

        let rerun = accept_loss_discarding_unconfirmed(&tv, &config, &store).await;

        assert_eq!(rerun.exit, JobExit::Done, "{}", rerun.captured.text);
        let warn = rerun.line(AL12_UNCONFIRMED);
        assert_eq!(warn["fields"]["item"], format!("session {UNREAD}"));
        assert!(
            warn["fields"]["error"]
                .as_str()
                .is_some_and(|e| e.contains("503")),
            "{warn}"
        );
        let (_lost, report) = rerun.the_discard();
        assert_eq!(report.sessions_discarded, vec![UNREAD.to_owned()]);
        assert_eq!(dead_set(&tv).await, BTreeSet::from([UNREAD_OBJECT]));
        assert_eq!(object_keys(&tv, UNREAD_OBJECT).await, Vec::<String>::new());
        assert!(!store.holds_session(UNREAD).await, "its backup is deleted");
        assert!(!marker_held(&tv).await);
    })
    .await;
}

/// **PD-2, a live session's backup read failing** (test review 1). A session
/// whose record survives and whose listed backup's read fails `Transient`
/// has its backup's rewrite unconfirmed: without the second confirm the Job
/// rewrites nothing and exits 3 with AL3 counting it; with it, AL12 names
/// the backup, AL4 counts the rewrite as the one unconfirmed item
/// discarded, the backup is rewritten from the store, and the run exits 0.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pd2_accept_loss_rewrites_a_live_sessions_unreadable_backup_only_under_the_second_confirm()
{
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        unreadable_session_fixture(&tv, &store, true).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let run = accept_loss(&tv, &config, &store).await;

        assert_eq!(run.exit, JobExit::StaysHeld, "{}", run.captured.text);
        assert_eq!(run.line(AL3_REFUSAL)["fields"]["unconfirmed"], 1);
        assert!(run.discards.is_empty(), "no discard ran, so no rewrite");
        assert!(marker_held(&tv).await);

        let rerun = accept_loss_discarding_unconfirmed(&tv, &config, &store).await;

        assert_eq!(rerun.exit, JobExit::Done, "{}", rerun.captured.text);
        assert_eq!(
            rerun.line(AL12_UNCONFIRMED)["fields"]["item"],
            format!("the session backup of session {UNREAD}")
        );
        assert_eq!(rerun.line(AL4_PLAN)["fields"]["unconfirmed"], 1);
        let (_lost, report) = rerun.the_discard();
        assert_eq!(report.rewritten, vec![UNREAD.to_owned()]);
        assert_eq!(report.unconfirmed, 1);
        assert_eq!(
            mapping(&tv, UNREAD, "u.usda").await,
            Some(UNREAD_OBJECT.to_string()),
            "the live layer stays"
        );
        assert!(dead_set(&tv).await.is_empty(), "nothing registered");
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// PD-4: full backups over the per-delta entry limit
// ═════════════════════════════════════════════════════════════════════════════

/// A plain full backup over the per-delta entry limit, as an earlier
/// release wrote one before the chunked format: [`base`] with enough
/// attributes that its `3 × attributes + 2` entries (the root and one
/// section per attribute, two fields each, one time sample) pass
/// `MAX_DELTA_ENTRIES`. Building and storing it takes about a second in a
/// debug build.
fn over_limit_base(tag: u8) -> Bytes {
    let attributes = (live_edit::domain::limits::MAX_DELTA_ENTRIES - 1).div_ceil(3);
    base(attributes, tag, &[])
}

/// The session and object of the over-limit fixtures.
const OL: &str = "s-ol";
const OL_OBJECT: u64 = 1_901;

/// The over-limit fixture: session [`OL`], its record surviving, with
/// `l.usda` → [`OL_OBJECT`], whose header the incident lost and whose
/// newest full backup, at 40, is [`over_limit_base`]; with `broken_older`,
/// an older full backup at 20 that does not decode. Its session backup maps
/// the layer. The record's path-index members are seeded as CreateSession
/// writes them beside the record (its id, and `name\0session` under the
/// record's own name), so a re-run's Step A from a backup rewritten from
/// this store adds nothing an uninterrupted run's store lacks. The counters
/// record is put.
async fn over_limit_fixture(tv: &TestValkey, store: &WriterStore, broken_older: bool) {
    put_session(store, &session_backup(OL, &[("l.usda", OL_OBJECT)], &[])).await;
    put_base(store, OL_OBJECT, OL, "l.usda", 40, over_limit_base(0x40)).await;
    if broken_older {
        put_base(
            store,
            OL_OBJECT,
            OL,
            "l.usda",
            20,
            Bytes::from_static(b"not a full backup"),
        )
        .await;
    }
    put_counters(store).await;
    seed_session_record(tv, OL).await;
    let base_path = base_path_of(OL);
    sadd(tv, tv.key_schema().path_index(&base_path), OL).await;
    sadd(
        tv,
        tv.key_schema().path_names_index(&base_path),
        // The name `seed_session_record` gives the record.
        &KeySchema::path_name_member("session", OL),
    )
    .await;
    seed_mapping(tv, OL, "l.usda", OL_OBJECT).await;
}

/// How AL13, a layer unmapped with its over-limit backups kept (WARN),
/// starts.
const AL13_OVER_LIMIT: &str = "the accept-loss Job removed the mapping of layer ";

/// How the selection's walk of [`OL`]'s layer ends with only the over-limit
/// backup listed, and with the broken older one beside it.
const OVER_LIMIT_EXHAUSTION: &str = "every one of 1 candidate generations was unusable";
const MIXED_EXHAUSTION: &str = "every one of 2 candidate generations was unusable";

/// AL13 for [`OL`]'s layer, whose walk ended `exhaustion` with its only
/// over-limit generation at 40.
fn al13_of_the_over_limit_layer(exhaustion: &str) -> String {
    format!(
        "the accept-loss Job removed the mapping of layer l.usda of session {OL} (object \
         {OL_OBJECT}) but kept its backups and did not register the object as deleted: no listed \
         backup of it could be used ({exhaustion}); the full backups under generation 40 (full \
         backup 40) are over the per-delta entry limit and were written by an earlier release; no \
         mapping or session backup names it any more, so no restore brings it back by itself: to \
         recover it, open a support case with this line (it needs a release whose limit fits \
         those backups)"
    )
}

/// What S1's accept-loss runs assert of [`OL`]'s layer: exit 0; one AL13
/// WARN, `al13`; the layer unmapped and out of the index, its session
/// backup rewritten without it; the object not on the dead-object set and
/// `backups` backups of it still listed; reported as unmapped, not
/// discarded; AL4 and AL11 counting it apart, in their words and their
/// `over_limit` fields.
async fn assert_over_limit_layer_unmapped_and_kept(
    tv: &TestValkey,
    store: &WriterStore,
    run: &AcceptLossRun,
    al13: &str,
    backups: usize,
) {
    assert_eq!(run.exit, JobExit::Done, "{}", run.captured.text);
    let warn = run.line(AL13_OVER_LIMIT);
    assert_eq!(warn["level"], "WARN");
    assert_eq!(warn["fields"]["message"], al13);
    assert_eq!(mapping(tv, OL, "l.usda").await, None, "unmapped");
    assert!(!indexed(tv, OL, "l.usda").await, "out of the index");
    assert!(
        dead_set(tv).await.is_empty(),
        "the object is not registered"
    );
    assert_eq!(
        store
            .list_layer_checkpoints(OL_OBJECT)
            .await
            .expect("the listing")
            .len(),
        backups,
        "its backups are untouched"
    );
    let rewritten = session_backup_of(store, OL)
        .await
        .expect("the session backup");
    assert!(
        rewritten.layer_mappings.is_empty(),
        "the session backup no longer names it: {:?}",
        rewritten.layer_mappings
    );
    let (_lost, report) = run.the_discard();
    assert_eq!(
        report.over_limit_unmapped,
        vec![(OL.to_owned(), "l.usda".to_owned(), OL_OBJECT)]
    );
    assert!(report.layers_discarded.is_empty(), "{report:?}");
    assert_eq!(report.registered, 0);
    assert_eq!(
        run.line(AL4_PLAN)["fields"]["message"],
        "the accept-loss Job discards 0 layers, 0 sessions and 0 large values from 0 layers, and \
         rewrites 1 session backups; 0 layers are registered as deleted; it unmaps 1 more layers \
         whose only backups are over the per-delta entry limit, keeping those backups for a \
         release whose limit fits them"
    );
    assert_eq!(
        run.line(AL4_PLAN)["fields"]["over_limit"],
        1,
        "AL4 counts it"
    );
    assert_eq!(
        run.line(AL11_SUMMARY)["fields"]["message"],
        "the accept-loss Job's summary: discarded 0 layers and 0 sessions, removed 0 large values \
         from 0 layers, left 0 mappings as found, registered 0 layers as deleted, deleted 0 \
         session backups and rewrote 1; unmapped 1 more layers whose only backups are over the \
         per-delta entry limit, keeping those backups for a release whose limit fits them"
    );
    assert_eq!(
        run.line(AL11_SUMMARY)["fields"]["over_limit"],
        1,
        "AL11 counts it"
    );
    assert!(run.lines(AL5_LAYER).is_empty(), "no AL5");
    assert!(!marker_held(tv).await);
}

/// **S1, ruling (d), (i) and (ii)** (fail-on-old). A lost layer whose only
/// backup is a full backup over the per-delta entry limit, with nothing
/// older and no survivor: accept-loss unmaps it and keeps its backup — not
/// registered, not wiped — with AL13 verbatim, and exits 0. (ii) A restore
/// run after it finds the backup listed, unmapped and not deleted: it
/// reports it as an unreferenced backup, holds nothing and exits 0, the
/// backup still listed and nothing registered. With the layer judged a lost
/// layer like any other, the object is registered dead and AL5 logged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn s1_accept_loss_unmaps_a_layer_whose_only_backup_is_over_the_limit_and_keeps_the_backup() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        over_limit_fixture(&tv, &store, false).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let run = accept_loss(&tv, &config, &store).await;

        assert_over_limit_layer_unmapped_and_kept(
            &tv,
            &store,
            &run,
            &al13_of_the_over_limit_layer(OVER_LIMIT_EXHAUSTION),
            1,
        )
        .await;

        hold(&tv, &config).await;
        let (body, observed) = observed_body(|b| b);
        let (exit, captured) = super::writer_support::run(&tv, &config, &store, body).await;

        assert_eq!(exit, JobExit::Done, "the later restore: {}", captured.text);
        let (plan, _report) = the_execution(&observed);
        assert!(
            plan.advisories.contains(&Advisory::UnreferencedBackup {
                object_id: OL_OBJECT
            }),
            "the backup is reported unreferenced: {:?}",
            plan.advisories
        );
        assert!(plan.held.is_empty(), "nothing held: {:?}", plan.held);
        assert!(dead_set(&tv).await.is_empty(), "nothing registered");
        assert_eq!(
            store
                .list_layer_checkpoints(OL_OBJECT)
                .await
                .expect("the listing")
                .len(),
            1,
            "the backup is still listed"
        );
    })
    .await;
}

/// **S1, the mixed case** (the Engineer's broadening; fail-on-old). The same
/// layer with an older full backup that does not decode beside the
/// over-limit newest one: the walk steps past both and finds nothing
/// usable, and accept-loss unmaps the layer and keeps both backups,
/// registering nothing, with AL13 naming the over-limit generation only.
/// With the layer judged a lost layer like any other, the object is
/// registered dead.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn s1_accept_loss_keeps_an_over_limit_backup_beside_a_broken_older_one() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        over_limit_fixture(&tv, &store, true).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let run = accept_loss(&tv, &config, &store).await;

        assert_over_limit_layer_unmapped_and_kept(
            &tv,
            &store,
            &run,
            &al13_of_the_over_limit_layer(MIXED_EXHAUSTION),
            2,
        )
        .await;
    })
    .await;
}

/// **PD-4, Job-level** (test review 7 (ii); fail-on-old). A lost layer
/// whose newest full backup is over the per-delta entry limit and whose
/// older one is valid: accept-loss steps down and restores the older one
/// — the layer stays mapped, its content the older backup's — discards
/// nothing and exits 0. With the planner's lineage walk run under
/// restore's policy, the layer holds on the over-limit base and the run
/// stays held.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pd4_accept_loss_restores_the_older_backup_under_an_over_limit_newest_one() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let older = base(1, 0x44, &[]);
        put_base(store.as_ref(), OL_OBJECT, OL, "l.usda", 20, older.clone()).await;
        put_base(
            store.as_ref(),
            OL_OBJECT,
            OL,
            "l.usda",
            40,
            over_limit_base(0x45),
        )
        .await;
        put_session(
            store.as_ref(),
            &session_backup(OL, &[("l.usda", OL_OBJECT)], &[]),
        )
        .await;
        put_counters(store.as_ref()).await;
        seed_session_record(&tv, OL).await;
        seed_mapping(&tv, OL, "l.usda", OL_OBJECT).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let run = accept_loss(&tv, &config, &store).await;

        assert_eq!(run.exit, JobExit::Done, "{}", run.captured.text);
        let (plan, _report) = run.the_plan();
        let (_head, base_seq, _tails) = restored_lineage(plan, OL_OBJECT);
        assert_eq!(base_seq, 20, "restored from the older backup");
        let header = header(&tv, OL_OBJECT).await;
        let seeded: u64 = header["seq"].parse().expect("a seq");
        assert_restored(
            &tv,
            OL_OBJECT,
            &expected_state(OL_OBJECT, 20, &older, &[], seeded),
        )
        .await;
        assert_eq!(
            mapping(&tv, OL, "l.usda").await,
            Some(OL_OBJECT.to_string())
        );
        let (lost, _report) = run.the_discard();
        assert!(lost.is_empty(), "nothing is lost: {lost:?}");
        assert!(dead_set(&tv).await.is_empty(), "nothing registered");
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// A13: ADD 9; A14: defer-65's un-hold outcomes
// ═════════════════════════════════════════════════════════════════════════════

/// **A13, ADD 9 (a)** (fail-on-old). A write unit run between the recount
/// and the un-hold (`with_write_after_recount_for_test`), after the
/// discards: the finish refuses the un-hold — one ERROR (text 13), exit 1 —
/// and the marker stays, the counter untouched. With the check removed, the
/// marker is deleted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a13_add9_a_accept_loss_with_a_write_after_its_recount_exits_1_and_keeps_the_marker() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        lost_layer_fixture(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let run = accept_loss_with(&tv, &config, Arc::clone(&store) as _, false, |b| {
            b.with_write_after_recount_for_test()
        })
        .await;

        assert!(
            marker_held(&tv).await,
            "the marker stays: {}",
            run.captured.text
        );
        assert_eq!(run.exit, JobExit::Failed);
        assert_eq!(run.line(WRITE_AFTER_RECOUNT_LINE)["level"], "ERROR");
        assert_eq!(run.the_finish().unhold, None);
        assert_eq!(counter(&tv).await, None, "the counter is untouched");
        assert_eq!(
            dead_set(&tv).await,
            BTreeSet::from([LS_LOST]),
            "fixture: the discards ran before the recount"
        );
    })
    .await;
}

/// How many values A13 (b)'s layer references: more than one recount group
/// (64), so the mid-recount kill point fires.
const MANY: u8 = 70;

/// **A13, ADD 9 (b).** A recount stopped after its first group
/// (`KillPoint::RecountAfterFirstChunk`), after the discards: exit 1, no
/// receipt, the marker kept; the re-run converges, exit 0 and the marker
/// gone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a13_add9_b_accept_loss_killed_mid_recount_keeps_the_marker_and_converges() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        lost_layer_fixture(&tv, &store).await;
        let hashes: Vec<[u8; 32]> = (0..MANY)
            .map(|i| {
                let mut hash = [0xE4; 32];
                hash[0] = i;
                hash
            })
            .collect();
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
            put_value(&store, hash, b"backup-only").await;
        }
        let config = writer_config();
        hold(&tv, &config).await;

        let run = accept_loss_with(&tv, &config, Arc::clone(&store) as _, false, |b| {
            b.with_kill_point_for_test(KillPoint::RecountAfterFirstChunk)
        })
        .await;

        assert_eq!(run.exit, JobExit::Failed, "{}", run.captured.text);
        assert_eq!(run.lines(AL_KILL_LINE).len(), 1);
        assert_eq!(run.the_finish().recount, None, "no receipt");
        assert!(marker_held(&tv).await);
        assert_eq!(
            dead_set(&tv).await,
            BTreeSet::from([LS_LOST]),
            "discarded first"
        );

        let rerun = accept_loss(&tv, &config, &store).await;
        assert_eq!(
            rerun.exit,
            JobExit::Done,
            "the re-run: {}",
            rerun.captured.text
        );
        assert!(!marker_held(&tv).await);
    })
    .await;
}

/// **A14, defer-65 `LeaseLost`** (fail-on-old). The un-hold script handed a
/// lease value that is not the Job's (`with_unhold_lease_value_for_test`)
/// answers `lease_lost`: the Job logs text 12 and exits 1, the marker kept
/// and the counter untouched. With `LeaseLost` mapped to exit 0, the exit is
/// 0.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a14_accept_loss_whose_unhold_finds_the_lease_lost_exits_1_and_keeps_the_marker() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        lost_layer_fixture(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let run = accept_loss_with(&tv, &config, Arc::clone(&store) as _, false, |b| {
            b.with_unhold_lease_value_for_test("not-this-jobs-lease")
        })
        .await;

        assert_eq!(run.exit, JobExit::Failed, "{}", run.captured.text);
        assert_eq!(run.line(LEASE_LOST_LINE)["level"], "ERROR");
        assert!(
            matches!(
                run.the_finish().unhold,
                Some(UnholdOutcome::LeaseLost { .. })
            ),
            "{:?}",
            run.the_finish().unhold
        );
        assert!(marker_held(&tv).await, "the marker stays");
        assert_eq!(counter(&tv).await, None, "the counter is untouched");
        assert!(run.lines(UNHELD_LINE).is_empty());
    })
    .await;
}

/// **A14, defer-65 `AlreadyUnheld`** (fail-on-old). The un-hold's first
/// attempt runs its script and then loses its reply
/// (`with_unhold_reply_lost_for_test`), so its write step retries: the
/// in-doubt path, whose second attempt answers `already_unheld`. The Job
/// logs text 10 and exits 0, the marker gone. With `AlreadyUnheld` mapped to
/// exit 1, the exit is 1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a14_accept_loss_whose_unhold_reply_is_lost_finds_it_already_unheld_and_exits_0() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        lost_layer_fixture(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let run = accept_loss_with(&tv, &config, Arc::clone(&store) as _, false, |b| {
            b.with_unhold_reply_lost_for_test()
        })
        .await;

        assert_eq!(run.exit, JobExit::Done, "{}", run.captured.text);
        let line = run.line(&format!(
            "the hold for incident {INCIDENT} was already cleared"
        ));
        assert_eq!(line["level"], "INFO");
        assert!(
            matches!(
                run.the_finish().unhold,
                Some(UnholdOutcome::AlreadyUnheld { .. })
            ),
            "{:?}",
            run.the_finish().unhold
        );
        assert!(!marker_held(&tv).await, "the hold is cleared");
        assert!(
            run.lines(UNHELD_LINE).is_empty(),
            "the first answer was lost"
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// A15, A17: kills converge; the lines are the report
// ═════════════════════════════════════════════════════════════════════════════

/// The convergence fixture's lost-layer session and object, and its lost
/// session and object.
const CONV: &str = "s-conv";
const CONV_LOST: u64 = 1_501;
const GONE: &str = "s-gone";
const GONE_OBJECT: u64 = 1_502;
/// The batch seam every convergence run uses: 3 commands, so each removed
/// row (2 commands) is a batch of its own and a kill can land between a
/// value removal's row batches.
const SMALL_BATCH: u64 = 3;

/// What the convergence fixture made through the service.
struct Converging {
    /// The value layer's session (a generated id).
    live: String,
    /// The value layer's object.
    value_object: u64,
    /// The value three rows reference: fields `a` and `b`, and a time
    /// sample at 1.0.
    three_rows: [u8; 32],
    /// The value one row references: field `c`.
    one_row: [u8; 32],
}

/// The convergence fixture, every kind of discard at once:
///
/// - a lost layer: `lost.usda` → [`CONV_LOST`] of session [`CONV`] (record
///   surviving), no header, no backup; [`CONV`]'s backup is listed, so it is
///   rewritten;
/// - a lost session: [`GONE`], no record, no marker, its listed backup
///   undecodable, its complete layer `gone.usda` → [`GONE_OBJECT`] (a header,
///   a section, a stream entry);
/// - a value removal: a service-made layer whose two values' bytes are in no
///   store, one referenced by three rows and one by one.
async fn converging_fixture(tv: &TestValkey, store: &WriterStore) -> Converging {
    let svc = build_service(tv, Arc::new(AllowAllPermissions));
    let (live, value_object) = create_session_and_layer(tv, &svc, "conv", VALUE_LAYER).await;
    let (three_rows, three_bytes) = value_of(0xC1, 64);
    let (one_row, one_bytes) = value_of(0xC2, 64);
    let delta = OwnedDelta {
        set_fields: vec![
            ext_field("a", &three_rows, three_bytes.len(), 1, 1),
            ext_field("b", &three_rows, three_bytes.len(), 1, 2),
            ext_field("c", &one_row, one_bytes.len(), 2, 3),
        ],
        time_samples: vec![TimeSample {
            section_id: 1,
            time: 1.0,
            value_or_ext_hash: three_rows.to_vec(),
            ext_value_size: three_bytes.len() as u64,
            ext_value_index: 1,
        }],
        ..OwnedDelta::default()
    };
    update(
        &svc,
        &live,
        value_object,
        &delta,
        &[&three_bytes, &one_bytes],
    )
    .await;
    drop(svc);
    lose_bytes(tv, &three_rows).await;
    lose_bytes(tv, &one_row).await;

    seed_session_record(tv, CONV).await;
    seed_mapping(tv, CONV, "lost.usda", CONV_LOST).await;
    put_session(
        store,
        &session_backup(CONV, &[("lost.usda", CONV_LOST)], &[]),
    )
    .await;

    seed_complete(tv, GONE_OBJECT, GONE, "gone.usda", 10, &[]).await;
    seed_residue(tv, GONE_OBJECT).await;
    xadd(tv, GONE_OBJECT, "10-0").await;
    seed_mapping(tv, GONE, "gone.usda", GONE_OBJECT).await;
    store
        .put_session_checkpoint(
            GONE,
            Bytes::from_static(b"not a session backup"),
            backup_meta(GONE, "", 0),
        )
        .await
        .expect("the undecodable session backup put");
    put_counters(store).await;
    Converging {
        live,
        value_object,
        three_rows,
        one_row,
    }
}

/// The hash fields two stores of one fixture tell apart by when or with
/// which nonce a write ran, never by what it wrote: a write's
/// `commit_chain` (fresh per write) and the times the fixture's own service
/// writes stamped on its session record and its layer's header. Compared by
/// their presence alone.
const VOLATILE_FIELDS: [&str; 4] = [
    "commit_chain",
    "created_at",
    "modified_at",
    "last_server_ts",
];

/// Every key under `tv`'s prefix and its content, normalized so two
/// fixtures' stores compare: the prefix dropped, each `(from, to)` of
/// `renames` replaced in names and contents (a generated session id), a
/// session's backup build stamp (`…:ckpt`, minted once per build) by its
/// presence alone, each of the [`VOLATILE_FIELDS`] by its presence alone, a
/// layer's commit-token record (`…:obj:wtok`, a nonce and a time per
/// client) by its clients alone, a sorted set by its members alone, and a
/// stream by its length.
async fn normalized_keyspace(
    tv: &TestValkey,
    renames: &[(&str, &str)],
) -> BTreeMap<String, String> {
    let rename = |s: &str| {
        renames
            .iter()
            .fold(s.to_owned(), |acc, (from, to)| acc.replace(from, to))
    };
    let pool = &tv.client().pool;
    let mut out = BTreeMap::new();
    for key in crate::common::scan_keys(tv.client(), &format!("{}*", tv.prefix())).await {
        let name = key.as_str().expect("keys are UTF-8").to_owned();
        let short = name.strip_prefix(tv.prefix()).expect("under the prefix");
        let kind: String = pool.r#type(&name).await.expect("TYPE");
        let content = match kind.as_str() {
            _ if short.ends_with(":ckpt") => "a session backup's build stamp".to_owned(),
            "string" => {
                let value: Option<Bytes> = pool.get(&name).await.expect("GET");
                format!("{value:?}")
            }
            "hash" if short.ends_with(":obj:wtok") => {
                let fields: HashMap<String, Bytes> = pool.hgetall(&name).await.expect("HGETALL");
                let clients: BTreeSet<String> = fields.into_keys().collect();
                format!("the commit tokens of {clients:?}")
            }
            "hash" => {
                let fields: HashMap<String, Bytes> = pool.hgetall(&name).await.expect("HGETALL");
                let fields: BTreeMap<String, Bytes> = fields
                    .into_iter()
                    .map(|(field, value)| {
                        if VOLATILE_FIELDS.contains(&field.as_str()) {
                            (field, Bytes::from_static(b"<present>"))
                        } else {
                            (field, value)
                        }
                    })
                    .collect();
                format!("{fields:?}")
            }
            "set" => {
                let members: BTreeSet<Bytes> = pool.smembers(&name).await.expect("SMEMBERS");
                format!("{members:?}")
            }
            "zset" => {
                let mut members: Vec<Bytes> = pool
                    .zrange(&name, 0, -1, None, false, None, false)
                    .await
                    .expect("ZRANGE");
                members.sort();
                format!("{members:?}")
            }
            "list" => {
                let items: Vec<Bytes> = pool.lrange(&name, 0, -1).await.expect("LRANGE");
                format!("{items:?}")
            }
            "stream" => {
                let len: u64 = pool.xlen(&name).await.expect("XLEN");
                format!("a stream of {len} entries")
            }
            other => format!("a {other}"),
        };
        out.insert(rename(short), rename(&content));
    }
    out
}

/// The keys whose content differs between `left` and `right`, or which one
/// of them lacks.
fn differences(left: &BTreeMap<String, String>, right: &BTreeMap<String, String>) -> Vec<String> {
    let keys: BTreeSet<&String> = left.keys().chain(right.keys()).collect();
    keys.into_iter()
        .filter(|key| left.get(*key) != right.get(*key))
        .map(|key| {
            format!(
                "{key}:\n  reference: {:?}\n  converged: {:?}",
                left.get(key),
                right.get(key)
            )
        })
        .collect()
}

/// What the backup store holds of the convergence fixture's sessions:
/// [`CONV`]'s backup's mappings, and whether [`GONE`]'s backup is there.
async fn backup_sessions(store: &WriterStore) -> (Option<BTreeMap<String, u64>>, bool) {
    let conv = session_backup_of(store, CONV)
        .await
        .map(|backup| backup.layer_mappings);
    let gone = store
        .get_session_checkpoint(GONE)
        .await
        .expect("the in-memory store answers")
        .is_some();
    (conv, gone)
}

/// An accept-loss run of the convergence fixture with the small batch seam
/// and `seams`.
async fn converging_run(
    tv: &TestValkey,
    config: &Config,
    store: &Arc<WriterStore>,
    seams: impl FnOnce(AcceptLossBody) -> AcceptLossBody,
) -> AcceptLossRun {
    accept_loss_with(tv, config, Arc::clone(store) as _, false, |b| {
        seams(b.with_batch_commands_for_test(SMALL_BATCH))
    })
    .await
}

/// The convergence fixture run uninterrupted, its store kept.
struct Uninterrupted {
    tv: TestValkey,
    fixture: Converging,
    /// Its normalized store.
    keyspace: BTreeMap<String, String>,
    /// What its backup store holds of its sessions.
    backups: (Option<BTreeMap<String, u64>>, bool),
    run: AcceptLossRun,
}

/// The convergence fixture run uninterrupted (exit 0).
async fn uninterrupted() -> Uninterrupted {
    let tv = TestValkey::setup().await;
    let store = WriterStore::new();
    let fixture = converging_fixture(&tv, &store).await;
    let config = writer_config();
    hold(&tv, &config).await;
    let run = converging_run(&tv, &config, &store, |b| b).await;
    assert_eq!(
        run.exit,
        JobExit::Done,
        "the reference: {}",
        run.captured.text
    );
    Uninterrupted {
        keyspace: normalized_keyspace(&tv, &[(fixture.live.as_str(), "<live>")]).await,
        backups: backup_sessions(&store).await,
        tv,
        fixture,
        run,
    }
}

/// The convergence fixture's run killed at a point, with an uninterrupted
/// run's store to compare its re-run against.
struct Killed {
    reference: BTreeMap<String, String>,
    reference_backups: (Option<BTreeMap<String, u64>>, bool),
    tv: TestValkey,
    store: Arc<WriterStore>,
    fixture: Converging,
    config: Config,
    point: KillPoint,
}

/// **A15**'s first half for one kill point: an uninterrupted run's store,
/// then the convergence fixture's run killed at the point `point` names
/// (exit 1, its kill line, the marker kept).
async fn killed_at(point: impl FnOnce(&Converging) -> KillPoint) -> Killed {
    let reference = uninterrupted().await;
    let tv = TestValkey::setup().await;
    let store = WriterStore::new();
    let fixture = converging_fixture(&tv, &store).await;
    let config = writer_config();
    hold(&tv, &config).await;
    let point = point(&fixture);

    let killed = converging_run(&tv, &config, &store, |b| {
        b.with_kill_point_for_test(point.clone())
    })
    .await;
    assert_eq!(
        killed.exit,
        JobExit::Failed,
        "{point:?}: {}",
        killed.captured.text
    );
    assert_eq!(
        killed.lines(AL_KILL_LINE).len(),
        1,
        "{point:?}: the run reached its kill point; capture:\n{}",
        killed.captured.text
    );
    assert!(marker_held(&tv).await, "{point:?}: the marker stays");
    Killed {
        reference: reference.keyspace,
        reference_backups: reference.backups,
        tv,
        store,
        fixture,
        config,
        point,
    }
}

impl Killed {
    /// **A15**'s second half: the killed run run again (exit 0) leaves the
    /// store, normalized, and the backup store's sessions as the
    /// uninterrupted run left them.
    async fn rerun_converges(self) {
        let Killed {
            reference,
            reference_backups,
            tv,
            store,
            fixture,
            config,
            point,
        } = self;
        let rerun = converging_run(&tv, &config, &store, |b| b).await;
        assert_eq!(
            rerun.exit,
            JobExit::Done,
            "{point:?}: the re-run: {}",
            rerun.captured.text
        );
        let converged = normalized_keyspace(&tv, &[(fixture.live.as_str(), "<live>")]).await;
        let differences = differences(&reference, &converged);
        assert!(
            differences.is_empty(),
            "{point:?}: the store differs from an uninterrupted run's:\n{}",
            differences.join("\n")
        );
        assert_eq!(
            backup_sessions(&store).await,
            reference_backups,
            "{point:?}: the backup store's sessions"
        );
    }
}

/// **A15** for one kill point: [`killed_at`], then
/// [`Killed::rerun_converges`].
async fn converges_after_a_kill(point: impl FnOnce(&Converging) -> KillPoint) {
    killed_at(point).await.rerun_converges().await;
}

/// **A15**, killed after the lost layers' registration.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a15_accept_loss_killed_after_registering_the_lost_layers_converges() {
    within_test_budget(converges_after_a_kill(|_| {
        KillPoint::DiscardAfterRegister(None)
    }))
    .await;
}

/// **A15**, killed after the lost layers' wipe.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a15_accept_loss_killed_after_wiping_the_lost_layers_converges() {
    within_test_budget(converges_after_a_kill(|_| {
        KillPoint::DiscardAfterWipe(None)
    }))
    .await;
}

/// **A15**, killed after the lost layers' unmap (fail-on-old). The re-run
/// finds the lost object registered, so the session backup's stale mapping
/// of it is dropped and the backup rewritten. With the registration moved
/// after the unmap, the killed run's lost object is never registered: the
/// re-run's session restore writes the stale mapping back and the session
/// record from that backup, and the store differs from an uninterrupted
/// run's.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a15_accept_loss_killed_after_unmapping_the_lost_layers_converges() {
    within_test_budget(converges_after_a_kill(|_| {
        KillPoint::DiscardAfterUnmap(None)
    }))
    .await;
}

/// **A15**, killed after the lost session's registration (PM N1): its
/// object is on the dead-object set and its header still stands when the
/// run stops.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a15_accept_loss_killed_after_registering_a_lost_session_converges() {
    within_test_budget(async {
        let killed = killed_at(|_| KillPoint::DiscardAfterRegister(Some(GONE.to_owned()))).await;
        assert!(
            dead_set(&killed.tv).await.contains(&GONE_OBJECT),
            "the lost session's object is registered"
        );
        assert!(
            !header(&killed.tv, GONE_OBJECT).await.is_empty(),
            "and not yet wiped"
        );
        killed.rerun_converges().await;
    })
    .await;
}

/// **A15**, killed after the lost session's wipe (PM N1): its object's keys
/// are gone and its mapping still stands when the run stops.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a15_accept_loss_killed_after_wiping_a_lost_session_converges() {
    within_test_budget(async {
        let killed = killed_at(|_| KillPoint::DiscardAfterWipe(Some(GONE.to_owned()))).await;
        assert_eq!(
            object_keys(&killed.tv, GONE_OBJECT).await,
            Vec::<String>::new(),
            "the lost session's object is wiped"
        );
        assert_eq!(
            mapping(&killed.tv, GONE, "gone.usda").await,
            Some(GONE_OBJECT.to_string()),
            "and not yet unmapped"
        );
        killed.rerun_converges().await;
    })
    .await;
}

/// **A15**, killed after the lost session's unmap.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a15_accept_loss_killed_after_unmapping_a_lost_session_converges() {
    within_test_budget(converges_after_a_kill(|_| {
        KillPoint::DiscardAfterUnmap(Some(GONE.to_owned()))
    }))
    .await;
}

/// **A15**, killed after the lost session's backup was deleted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a15_accept_loss_killed_after_deleting_a_lost_sessions_backup_converges() {
    within_test_budget(converges_after_a_kill(|_| {
        KillPoint::LostSessionBackupDeleted(None)
    }))
    .await;
}

/// **A15**, killed inside a value removal's row batches
/// (`KillPoint::ValueRowsAfterBatch`): after the first of its four row
/// batches (one row each under [`SMALL_BATCH`]), with three rows still to
/// delete. Every `HDEL obj:ext` follows the last row batch, so the killed
/// run leaves both values in `obj:ext`; the re-run finds them referenced
/// still, walks the indices from the start and removes the rest.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a15_accept_loss_killed_inside_a_values_row_batches_converges() {
    within_test_budget(async {
        let killed = killed_at(|fixture| KillPoint::ValueRowsAfterBatch {
            object: Some(fixture.value_object),
            batch: 0,
        })
        .await;
        for (what, hash) in [
            ("the value three rows name", &killed.fixture.three_rows),
            ("the value one row names", &killed.fixture.one_row),
        ] {
            assert!(
                ext_indexed(&killed.tv, killed.fixture.value_object, hash).await,
                "the kill came before the obj:ext batches: {what} is still in obj:ext"
            );
        }
        killed.rerun_converges().await;
    })
    .await;
}

/// **A15**, killed after a value removal's row and `obj:ext` batches and
/// before its banded header (`KillPoint::ValueRemovalBeforeHeader`;
/// fail-on-old). The last hash's `HDEL obj:ext` rides the header
/// transaction, so the killed run leaves that hash in `obj:ext` while the
/// other one is gone and the header is not yet banded; the re-run finds the
/// removal unfinished, removes it and stamps the banded header, as an
/// uninterrupted run does. With every `HDEL` in the batches before the
/// header and none in the header transaction, the killed run removes both
/// hashes, so no re-run would find the removal unfinished: the first
/// assertion fails.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a15_accept_loss_killed_before_a_value_removals_banded_header_converges() {
    within_test_budget(async {
        let killed =
            killed_at(|fixture| KillPoint::ValueRemovalBeforeHeader(Some(fixture.value_object)))
                .await;
        let fixture = &killed.fixture;
        let (first, last) = if fixture.three_rows < fixture.one_row {
            (fixture.three_rows, fixture.one_row)
        } else {
            (fixture.one_row, fixture.three_rows)
        };
        assert!(
            ext_indexed(&killed.tv, fixture.value_object, &last).await,
            "the last hash stays in obj:ext until the header lands"
        );
        assert!(
            !ext_indexed(&killed.tv, fixture.value_object, &first).await,
            "the other hash's HDEL, in the obj:ext batch after the rows, landed before the kill"
        );
        assert!(
            !header(&killed.tv, fixture.value_object)
                .await
                .contains_key("restore_seq_bump"),
            "the header is not banded yet"
        );
        killed.rerun_converges().await;
    })
    .await;
}

/// **A15**, killed before a session backup's rewrite.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a15_accept_loss_killed_before_rewriting_a_session_backup_converges() {
    within_test_budget(converges_after_a_kill(|_| {
        KillPoint::SessionRewriteBefore(None)
    }))
    .await;
}

/// The mixed over-limit fixture ([`over_limit_fixture`] with its broken
/// older backup) run uninterrupted (exit 0): its store, normalized.
async fn over_limit_uninterrupted() -> BTreeMap<String, String> {
    let tv = TestValkey::setup().await;
    let store = WriterStore::new();
    over_limit_fixture(&tv, &store, true).await;
    let config = writer_config();
    hold(&tv, &config).await;
    let run = accept_loss(&tv, &config, &store).await;
    assert_eq!(
        run.exit,
        JobExit::Done,
        "the reference: {}",
        run.captured.text
    );
    normalized_keyspace(&tv, &[]).await
}

/// The mixed over-limit fixture's run killed at `point`, with an
/// uninterrupted run's store to compare its re-run against.
struct OverLimitKilled {
    reference: BTreeMap<String, String>,
    tv: TestValkey,
    store: Arc<WriterStore>,
    config: Config,
    point: KillPoint,
}

/// What [`OL`]'s session backup maps when the mixed over-limit fixture's
/// run is killed at `point`: the layer still at D1's point, which comes
/// before D3's early rewrite of the session's backup; nothing from that
/// rewrite's own point on, the unmap's and D6's included, so no kill leaves
/// a backup naming an unmapped over-limit layer.
fn over_limit_backup_at(point: &KillPoint) -> BTreeMap<String, u64> {
    match point {
        KillPoint::DiscardAfterRegister(_) => BTreeMap::from([("l.usda".to_owned(), OL_OBJECT)]),
        KillPoint::DiscardAfterOverLimitRewrite
        | KillPoint::DiscardAfterUnmap(_)
        | KillPoint::SessionRewriteBefore(_) => BTreeMap::new(),
        other => panic!("no over-limit kill test stops at {other:?}"),
    }
}

/// **A15 on an over-limit layer**, first half: an uninterrupted run's
/// store, then the mixed over-limit fixture's run killed at `point` (exit 1,
/// its kill line, the marker kept). The killed run registers nothing, and
/// leaves [`OL`]'s session backup as [`over_limit_backup_at`] says.
async fn over_limit_killed_at(point: KillPoint) -> OverLimitKilled {
    let reference = over_limit_uninterrupted().await;
    let tv = TestValkey::setup().await;
    let store = WriterStore::new();
    over_limit_fixture(&tv, &store, true).await;
    let config = writer_config();
    hold(&tv, &config).await;

    let killed = accept_loss_with(&tv, &config, Arc::clone(&store) as _, false, |b| {
        b.with_kill_point_for_test(point.clone())
    })
    .await;

    assert_eq!(
        killed.exit,
        JobExit::Failed,
        "{point:?}: {}",
        killed.captured.text
    );
    assert_eq!(
        killed.lines(AL_KILL_LINE).len(),
        1,
        "{point:?}: the run reached its kill point; capture:\n{}",
        killed.captured.text
    );
    assert!(marker_held(&tv).await, "{point:?}: the marker stays");
    assert!(
        dead_set(&tv).await.is_empty(),
        "{point:?}: the killed run registered nothing"
    );
    assert_eq!(
        session_backup_of(&store, OL)
            .await
            .expect("the session backup")
            .layer_mappings,
        over_limit_backup_at(&point),
        "{point:?}: the session backup the killed run left"
    );
    OverLimitKilled {
        reference,
        tv,
        store,
        config,
        point,
    }
}

impl OverLimitKilled {
    /// **A15 on an over-limit layer**, second half: the killed run run again
    /// (exit 0) plans no Step B for [`OL`], so its record keeps the name
    /// the store gave it (not its stale backup's, [`name_of`]); it leaves
    /// the store, normalized, as the uninterrupted run left it; [`OL`]'s
    /// session backup no longer names the layer; both of its object's
    /// backups are still listed; and the dead-object set is empty.
    async fn rerun_converges(self) {
        let OverLimitKilled {
            reference,
            tv,
            store,
            config,
            point,
        } = self;
        let rerun = accept_loss(&tv, &config, &store).await;
        assert_eq!(
            rerun.exit,
            JobExit::Done,
            "{point:?}: the re-run: {}",
            rerun.captured.text
        );
        let (plan, _report) = rerun.the_plan();
        let step_b: Vec<&str> = plan
            .sessions
            .iter()
            .filter(|s| matches!(s.action, SessionAction::Restore { step_b: true, .. }))
            .map(|s| s.session_id.as_str())
            .collect();
        assert!(
            step_b.is_empty(),
            "{point:?}: the re-run restores no session's record from its backup: {step_b:?}"
        );
        let name: Option<String> = tv
            .client()
            .pool
            .hget(tv.key_schema().session_meta(OL), "name")
            .await
            .expect("HGET name");
        assert_eq!(
            name.as_deref(),
            Some("session"),
            "{point:?}: the session record keeps its name"
        );
        let rewritten = session_backup_of(&store, OL)
            .await
            .expect("the session backup");
        assert!(
            rewritten.layer_mappings.is_empty(),
            "{point:?}: the session backup no longer names the layer: {rewritten:?}"
        );
        assert_eq!(
            store
                .list_layer_checkpoints(OL_OBJECT)
                .await
                .expect("the listing")
                .len(),
            2,
            "{point:?}: both backups are still listed"
        );
        assert!(
            dead_set(&tv).await.is_empty(),
            "{point:?}: the object is not registered"
        );
        let converged = normalized_keyspace(&tv, &[]).await;
        let differences = differences(&reference, &converged);
        assert!(
            differences.is_empty(),
            "{point:?}: the store differs from an uninterrupted run's:\n{}\nthe session \
             backup: {rewritten:?}",
            differences.join("\n")
        );
    }
}

/// **A15 on an over-limit layer**, killed after D1's registration (test
/// review r2, 2), which registers nothing for it: the layer is still mapped
/// and its session backup still names it when the run stops, and the re-run
/// converges.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a15_accept_loss_killed_after_registering_beside_an_over_limit_layer_converges() {
    within_test_budget(async {
        let killed = over_limit_killed_at(KillPoint::DiscardAfterRegister(None)).await;
        assert_eq!(
            mapping(&killed.tv, OL, "l.usda").await,
            Some(OL_OBJECT.to_string()),
            "the kill landed before the unmap"
        );
        killed.rerun_converges().await;
    })
    .await;
}

/// **A15 on an over-limit layer**, killed after D3's early rewrite of its
/// session backup and before its unmap
/// (`KillPoint::DiscardAfterOverLimitRewrite`; fail-on-old): the layer is
/// still mapped while its session backup no longer names it, and the
/// re-run finds it lost again, takes no Step B and converges. With the
/// early rewrite a no-op, the killed run leaves the backup naming the
/// layer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a15_accept_loss_killed_after_rewriting_an_over_limit_layers_session_early_converges() {
    within_test_budget(async {
        let killed = over_limit_killed_at(KillPoint::DiscardAfterOverLimitRewrite).await;
        assert_eq!(
            mapping(&killed.tv, OL, "l.usda").await,
            Some(OL_OBJECT.to_string()),
            "the kill landed before the unmap"
        );
        assert!(
            indexed(&killed.tv, OL, "l.usda").await,
            "and the layers index still names it"
        );
        killed.rerun_converges().await;
    })
    .await;
}

/// **A15 on an over-limit layer**, killed after D3's unmap of it and before
/// D6 (fail-on-old): the mapping is gone and the session backup, rewritten
/// ahead of the unmap, no longer names the layer, so the re-run takes no
/// Step B for the session and converges. With D3's early rewrite a no-op,
/// the killed run leaves the backup naming the unmapped layer, and a re-run
/// from there takes Step B for the session, writing its record back from
/// that backup.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a15_accept_loss_killed_after_unmapping_an_over_limit_layer_converges() {
    within_test_budget(async {
        let killed = over_limit_killed_at(KillPoint::DiscardAfterUnmap(None)).await;
        assert_eq!(
            mapping(&killed.tv, OL, "l.usda").await,
            None,
            "the kill landed after the unmap"
        );
        killed.rerun_converges().await;
    })
    .await;
}

/// **A15 on an over-limit layer**, killed before D6's rewrite of its
/// session backup (fail-on-old): the mapping is gone and the backup D3
/// rewrote ahead of the unmap no longer names the layer, and the re-run
/// converges. With D3's early rewrite a no-op, the killed run leaves the
/// backup naming the unmapped layer, and a re-run from there takes Step B
/// for the session, writing its record back from that backup.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a15_accept_loss_killed_before_rewriting_an_over_limit_layers_session_converges() {
    within_test_budget(async {
        let killed = over_limit_killed_at(KillPoint::SessionRewriteBefore(None)).await;
        assert_eq!(
            mapping(&killed.tv, OL, "l.usda").await,
            None,
            "the kill landed after the unmap"
        );
        killed.rerun_converges().await;
    })
    .await;
}

/// **A15**, a restore after a partial accept-loss. Accept-loss killed after
/// registering the lost layer's object: the restore Job run next holds
/// (exit 3) and writes no mapping that names an object on the dead-object
/// set — the lost layer's mapping still names its registered object, as
/// before the restore.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a15_restore_after_accept_loss_registered_its_lost_layers_holds_and_maps_nothing_dead() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        converging_fixture(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;
        let killed = converging_run(&tv, &config, &store, |b| {
            b.with_kill_point_for_test(KillPoint::DiscardAfterRegister(None))
        })
        .await;
        assert_eq!(killed.exit, JobExit::Failed, "{}", killed.captured.text);
        let dead = dead_set(&tv).await;
        assert_eq!(dead, BTreeSet::from([CONV_LOST]), "fixture: registered");
        let mappings_before = mappings(&tv).await;

        let (exit, captured) = run(&tv, &config, &store, RestoreBody::new()).await;

        assert_eq!(exit, JobExit::StaysHeld, "{}", captured.text);
        let mappings_after = mappings(&tv).await;
        for (key, object) in &mappings_after {
            if dead.contains(object) {
                assert_eq!(
                    mappings_before.get(key),
                    Some(object),
                    "{key}: the restore wrote no mapping to a registered object"
                );
            }
        }
        assert_eq!(
            mapping(&tv, CONV, "lost.usda").await,
            Some(CONV_LOST.to_string())
        );
    })
    .await;
}

/// Every layer mapping in the store: its key and the object it names.
async fn mappings(tv: &TestValkey) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    for key in
        crate::common::scan_keys(tv.client(), &tv.key_schema().layer_mapping_scan_pattern()).await
    {
        let name = key.as_str().expect("keys are UTF-8").to_owned();
        let object = get(tv, name.clone())
            .await
            .expect("the mapping")
            .parse()
            .expect("an object id");
        out.insert(name, object);
    }
    out
}

/// **A17.** On the convergence fixture run uninterrupted, the accept-loss
/// lines carry what the lost set and the discard report hold: AL4's counts
/// are the lost set's, and its words, without the second confirm, carry no
/// unconfirmed clause; one AL5 per discarded layer, one AL6 per discarded
/// session and one AL7 per value-removed layer, each with its report's
/// fields; no AL8; and AL11's totals are the report's, in its fields and
/// its words. The report itself is the fixture's: the lost layer discarded,
/// the lost session discarded and its backup deleted, two values removed
/// from one layer, both objects registered, [`CONV`]'s backup rewritten.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a17_accept_loss_lines_carry_the_lost_set_and_the_discard_report() {
    within_test_budget(async {
        let Uninterrupted { run, .. } = uninterrupted().await;
        let (lost, report) = run.the_discard();

        assert_eq!(
            report.layers_discarded,
            vec![(CONV.to_owned(), "lost.usda".to_owned(), CONV_LOST)]
        );
        assert_eq!(report.sessions_discarded, vec![GONE.to_owned()]);
        assert_eq!(report.values.len(), 1);
        assert_eq!(report.value_count(), 2);
        assert_eq!(report.registered, 2);
        assert_eq!(report.session_backups_deleted, 1);
        assert_eq!(report.rewritten, vec![CONV.to_owned()]);
        assert!(report.left.is_empty());

        let counts = DiscardPlanCounts::of(lost);
        let plan = &run.line(AL4_PLAN)["fields"];
        assert_eq!(
            (
                plan["layers"].as_u64(),
                plan["sessions"].as_u64(),
                plan["values"].as_u64(),
                plan["value_layers"].as_u64(),
                plan["rewrites"].as_u64(),
                plan["registered"].as_u64(),
                plan["unconfirmed"].as_u64(),
            ),
            (
                Some(counts.layers as u64),
                Some(counts.sessions as u64),
                Some(counts.values as u64),
                Some(counts.value_layers as u64),
                Some(counts.rewrites as u64),
                Some(counts.registered as u64),
                Some(counts.unconfirmed as u64),
            ),
            "AL4 is the lost set's counts"
        );
        assert_eq!(
            run.line(AL4_PLAN)["fields"]["message"],
            "the accept-loss Job discards 1 layers, 1 sessions and 2 large values from 1 \
             layers, and rewrites 1 session backups; 2 layers are registered as deleted",
            "AL4's words, with no unconfirmed clause without the second confirm"
        );

        let layers = run.lines(AL5_LAYER);
        assert_eq!(layers.len(), report.layers_discarded.len());
        for ((session, layer, object), line) in report.layers_discarded.iter().zip(&layers) {
            assert_eq!(line["level"], "WARN");
            assert_eq!(line["fields"]["session_id"], session.as_str());
            assert_eq!(line["fields"]["layer_name"], layer.as_str());
            assert_eq!(line["fields"]["object_id"], *object);
        }
        let sessions = run.lines(AL6_SESSION);
        assert_eq!(sessions.len(), report.sessions_discarded.len());
        assert_eq!(sessions[0]["fields"]["session_id"], GONE);
        assert_eq!(sessions[0]["fields"]["backup_deleted"], true);
        let removals = run.lines(AL7_VALUES);
        assert_eq!(removals.len(), report.values.len());
        for line in removals {
            let object = line["fields"]["object_id"].as_u64().expect("object_id");
            let record = &report.values[&object];
            assert_eq!(
                (
                    line["fields"]["values"].as_u64(),
                    line["fields"]["seeded_seq"].as_u64(),
                    line["fields"]["band"].as_u64()
                ),
                (
                    Some(record.hashes.len() as u64),
                    Some(record.seeded_seq),
                    Some(record.band)
                ),
                "AL7 of object {object} is its record"
            );
        }
        assert!(run.lines(AL8_LEFT).is_empty());

        let summary = &run.line(AL11_SUMMARY)["fields"];
        assert_eq!(
            (
                summary["layers"].as_u64(),
                summary["sessions"].as_u64(),
                summary["values"].as_u64(),
                summary["value_layers"].as_u64(),
                summary["left"].as_u64(),
                summary["registered"].as_u64(),
                summary["session_backups"].as_u64(),
                summary["rewrites"].as_u64(),
                summary["unconfirmed"].as_u64(),
            ),
            (
                Some(report.layers_discarded.len() as u64),
                Some(report.sessions_discarded.len() as u64),
                Some(report.value_count() as u64),
                Some(report.values.len() as u64),
                Some(report.left.len() as u64),
                Some(report.registered as u64),
                Some(report.session_backups_deleted as u64),
                Some(report.rewritten.len() as u64),
                Some(report.unconfirmed as u64),
            ),
            "AL11 is the discard report"
        );
        assert_eq!(
            run.line(AL11_SUMMARY)["fields"]["message"],
            "the accept-loss Job's summary: discarded 1 layers and 1 sessions, removed 2 large \
             values from 1 layers, left 0 mappings as found, registered 2 layers as deleted, \
             deleted 1 session backups and rewrote 1",
            "AL11's words, with no unconfirmed clause without the second confirm"
        );
    })
    .await;
}

/// **A17, D5's result in the store** (test review 5). On the convergence
/// fixture run uninterrupted, the value removal left the layer with no row
/// naming a removed value: the field rows `a`, `b` and `c` and the time
/// sample at 1.0 are gone, and so are their `obj:flds` and `obj:tss`
/// members; its record counts those 4 rows; its header transaction, the
/// last of its batches, queued the constant 3 commands (the stream
/// `UNLINK`, the header `HSET` and the last hash's `HDEL obj:ext`); and the
/// layer's catch-up from nothing succeeds, carrying no value.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a17_accept_loss_value_removal_leaves_no_row_naming_a_removed_value() {
    within_test_budget(async {
        let Uninterrupted {
            tv, fixture, run, ..
        } = uninterrupted().await;
        let object_id = fixture.value_object;
        let oid = object_id.to_string();
        let keys = tv.key_schema();

        let (_lost, report) = run.the_discard();
        let record = &report.values[&object_id];
        assert_eq!(record.rows, 4, "fields a, b and c and the time sample");
        assert_eq!(
            record.batches.last().map(|b| b.commands),
            Some(3),
            "the header transaction: UNLINK, HSET and one HDEL"
        );
        for field in ["a", "b", "c"] {
            assert!(
                !exists(&tv, field_key(&tv, object_id, "1", field)).await,
                "field row {field} is gone"
            );
            let member: Option<f64> = tv
                .client()
                .pool
                .zscore(
                    keys.obj_fields_index(&oid),
                    KeySchema::field_member("1", field),
                )
                .await
                .expect("ZSCORE obj:flds");
            assert_eq!(member, None, "field {field}'s obj:flds member is gone");
        }
        assert!(
            !exists(&tv, keys.obj_time_sample(&oid, "1", 1.0)).await,
            "the time sample at 1.0 is gone"
        );
        let member: Option<f64> = tv
            .client()
            .pool
            .zscore(
                keys.obj_time_samples_index(&oid),
                KeySchema::time_sample_member("1", 1.0),
            )
            .await
            .expect("ZSCORE obj:tss");
        assert_eq!(member, None, "its obj:tss member is gone");
        let values = catch_up_values(
            &build_service(&tv, Arc::new(AllowAllPermissions)),
            &fixture.live,
        )
        .await
        .expect("the layer's catch-up from nothing succeeds");
        assert!(values.is_empty(), "and carries no value: {values:?}");
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// A16: PD-6, a lost session with a complete layer
// ═════════════════════════════════════════════════════════════════════════════

/// The lost session of A16 and its complete layer.
const LOST_SESSION: &str = "s-16";
const LOST_SESSION_OBJECT: u64 = 1_601;

/// [`LOST_SESSION`]: no record, no deletion marker, no session backup, and
/// a complete layer `g.usda` → [`LOST_SESSION_OBJECT`] (a header naming it, a
/// section, a stream entry, its mapping and layers-index member).
async fn lost_session_fixture(tv: &TestValkey) {
    seed_complete(tv, LOST_SESSION_OBJECT, LOST_SESSION, "g.usda", 10, &[]).await;
    seed_residue(tv, LOST_SESSION_OBJECT).await;
    xadd(tv, LOST_SESSION_OBJECT, "10-0").await;
    seed_mapping(tv, LOST_SESSION, "g.usda", LOST_SESSION_OBJECT).await;
}

/// What A16 asserts of a run that discarded [`LOST_SESSION`]: exit 0, its
/// object registered and wiped (every key of it gone), its mapping and
/// layers index gone, one AL6 WARN, the hold cleared.
async fn assert_lost_session_discarded(tv: &TestValkey, run: &AcceptLossRun, case: &str) {
    assert_eq!(run.exit, JobExit::Done, "{case}: {}", run.captured.text);
    assert!(
        dead_set(tv).await.contains(&LOST_SESSION_OBJECT),
        "{case}: registered"
    );
    assert_eq!(
        object_keys(tv, LOST_SESSION_OBJECT).await,
        Vec::<String>::new(),
        "{case}: wiped, its stream included"
    );
    assert_eq!(mapping(tv, LOST_SESSION, "g.usda").await, None, "{case}");
    assert!(
        !exists(tv, tv.key_schema().layers_index(LOST_SESSION)).await,
        "{case}: its layers index is gone"
    );
    let warn = run.line(AL6_SESSION);
    assert_eq!(warn["level"], "WARN", "{case}");
    assert_eq!(warn["fields"]["session_id"], LOST_SESSION, "{case}");
    let (_lost, report) = run.the_discard();
    assert_eq!(
        report.sessions_discarded,
        vec![LOST_SESSION.to_owned()],
        "{case}"
    );
    assert!(!marker_held(tv).await, "{case}");
}

/// The backup store A16's lost session is discarded under.
#[derive(Clone, Copy, Debug)]
enum A16Store {
    /// A non-empty session listing: another session's backup, and its
    /// layer's.
    Listing,
    /// The `Disabled` adapter: no backup store.
    Disabled,
}

/// A16's fixture ([`lost_session_fixture`]) under `case`, held: its backup
/// store and configuration.
async fn a16_fixture(tv: &TestValkey, case: A16Store) -> (Arc<WriterStore>, Config) {
    let store = WriterStore::new();
    lost_session_fixture(tv).await;
    let config = match case {
        A16Store::Listing => {
            put_session(
                store.as_ref(),
                &session_backup("s-other", &[("o.usda", 1_611)], &[]),
            )
            .await;
            put_base(
                store.as_ref(),
                1_611,
                "s-other",
                "o.usda",
                5,
                base(1, 0x16, &[]),
            )
            .await;
            put_counters(store.as_ref()).await;
            writer_config()
        }
        A16Store::Disabled => {
            let mut config = crate::common::test_config();
            config.checkpoint_adapter = CheckpointAdapter::Disabled;
            config
        }
    };
    hold(tv, &config).await;
    (store, config)
}

/// **A16** (PD-6), a non-empty session listing. A session named by its
/// complete layer, with no record, no marker and no listed backup (the
/// backup store lists another session's) is lost: its complete layer is
/// registered and wiped and its mapping removed; the run exits 0.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a16_accept_loss_discards_a_lost_session_with_a_complete_layer_under_a_listing() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let (store, config) = a16_fixture(&tv, A16Store::Listing).await;

        let run = accept_loss(&tv, &config, &store).await;

        assert_lost_session_discarded(&tv, &run, "a non-empty listing").await;
        assert_eq!(
            mapping(&tv, "s-other", "o.usda").await,
            Some("1611".to_owned()),
            "the listed session is restored"
        );
    })
    .await;
}

/// **A16** (PD-6), no backup store. The same lost session under the
/// `Disabled` adapter: discarded the same way, its object registered (PD-5),
/// with no call of the backup store.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a16_accept_loss_discards_a_lost_session_with_a_complete_layer_with_no_backup_store() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let (store, config) = a16_fixture(&tv, A16Store::Disabled).await;

        let run = accept_loss(&tv, &config, &store).await;

        assert_lost_session_discarded(&tv, &run, "Disabled").await;
        assert_eq!(store.calls(), 0, "no call of the backup store");
    })
    .await;
}

/// **A15 on A16's fixture** (PM N1) for one kill point of the lost
/// session's discard, under `case`: an uninterrupted run's store; then the
/// run killed at `point` (exit 1, its kill line, the marker kept, the lost
/// session's object already registered), then run again (exit 0): the
/// store, normalized, is the uninterrupted run's.
async fn a16_converges_after_a_kill(case: A16Store, point: KillPoint) {
    let reference_tv = TestValkey::setup().await;
    let (reference_store, reference_config) = a16_fixture(&reference_tv, case).await;
    let reference = accept_loss(&reference_tv, &reference_config, &reference_store).await;
    assert_eq!(
        reference.exit,
        JobExit::Done,
        "{case:?}: the reference: {}",
        reference.captured.text
    );
    let reference = normalized_keyspace(&reference_tv, &[]).await;

    let tv = TestValkey::setup().await;
    let (store, config) = a16_fixture(&tv, case).await;
    let killed = accept_loss_with(&tv, &config, Arc::clone(&store) as _, false, |b| {
        b.with_kill_point_for_test(point.clone())
    })
    .await;
    assert_eq!(
        killed.exit,
        JobExit::Failed,
        "{case:?} {point:?}: {}",
        killed.captured.text
    );
    assert_eq!(
        killed.lines(AL_KILL_LINE).len(),
        1,
        "{case:?} {point:?}: the run reached its kill point; capture:\n{}",
        killed.captured.text
    );
    assert!(
        marker_held(&tv).await,
        "{case:?} {point:?}: the marker stays"
    );
    assert!(
        dead_set(&tv).await.contains(&LOST_SESSION_OBJECT),
        "{case:?} {point:?}: the lost session's object is registered"
    );

    let rerun = accept_loss(&tv, &config, &store).await;
    assert_eq!(
        rerun.exit,
        JobExit::Done,
        "{case:?} {point:?}: the re-run: {}",
        rerun.captured.text
    );
    let differences = differences(&reference, &normalized_keyspace(&tv, &[]).await);
    assert!(
        differences.is_empty(),
        "{case:?} {point:?}: the store differs from an uninterrupted run's:\n{}",
        differences.join("\n")
    );
}

/// **A15 on A16's fixture**, a non-empty listing, killed after the lost
/// session's registration.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a16_accept_loss_killed_after_registering_a_lost_session_under_a_listing_converges() {
    within_test_budget(a16_converges_after_a_kill(
        A16Store::Listing,
        KillPoint::DiscardAfterRegister(Some(LOST_SESSION.to_owned())),
    ))
    .await;
}

/// **A15 on A16's fixture**, a non-empty listing, killed after the lost
/// session's wipe.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a16_accept_loss_killed_after_wiping_a_lost_session_under_a_listing_converges() {
    within_test_budget(a16_converges_after_a_kill(
        A16Store::Listing,
        KillPoint::DiscardAfterWipe(Some(LOST_SESSION.to_owned())),
    ))
    .await;
}

/// **A15 on A16's fixture**, no backup store, killed after the lost
/// session's registration.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a16_accept_loss_killed_after_registering_a_lost_session_with_no_backup_store_converges() {
    within_test_budget(a16_converges_after_a_kill(
        A16Store::Disabled,
        KillPoint::DiscardAfterRegister(Some(LOST_SESSION.to_owned())),
    ))
    .await;
}

/// **A15 on A16's fixture**, no backup store, killed after the lost
/// session's wipe.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a16_accept_loss_killed_after_wiping_a_lost_session_with_no_backup_store_converges() {
    within_test_budget(a16_converges_after_a_kill(
        A16Store::Disabled,
        KillPoint::DiscardAfterWipe(Some(LOST_SESSION.to_owned())),
    ))
    .await;
}
