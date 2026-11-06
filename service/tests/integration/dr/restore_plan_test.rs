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

//! The restore planner end to end (`live_edit::dr::job::plan`; SDD SR-9.2 to
//! SR-9.6 steps 1–2, the Jobs section of SDD §22), over a real store and a
//! backup store the test scripts.
//!
//! Most tests call [`RestorePlanner::plan`] on a hand-built [`JobRun`], whose
//! deadlines put the quiet period's end in the past and `T_safe` either past
//! or [`T_SAFE_AHEAD`] ahead, so an absence before `T_safe` is real and its
//! re-observation at `T_safe` is a real wait. The tests whose subject is the
//! frame around the plan — the quiet period (defer-39), an exit 3 with nothing
//! written, the body's finish, the canary — run the Job through `run_job`
//! with [`RestoreBody::with_plan_observer`], stopped just before its un-hold.
//!
//! The backup store is an in-memory one wrapped in [`PlanStore`], which counts
//! every call and answers what a test scripts: a generation absent or failing,
//! or a session backup absent, until a chosen time on the store's clock, a value
//! absent until such a time, failing or never answering, a denied or failing
//! listing, a generation landing before a chosen listing, extra listed
//! artifacts, extra ext-refs sidecar hashes, probes that park until a number
//! of them are in flight, and probes that answer one at a time in arrival
//! order. Store time, not wall time, decides every time switch, so no switch
//! races the planner's own reading of the clock.
//!
//! Every Valkey-backed run uses a per-test key prefix (`TestValkey`). The
//! fixtures these tests write the same way as the restore writer's tests —
//! a backup generation's meta and put, the counters record, a surviving
//! header, a deletion marker, a dead object — and the Job's context, the
//! stays-held ERROR's opening and the log reader are `writer_support`'s.
//!
//! # Time
//!
//! No assertion bounds how long anything took. [`RUN_BUDGET`] (30 s) bounds a
//! hang; a healthy run takes at most about 4 s. The longest healthy tests wait for `T_safe` ([`T_SAFE_AHEAD`], 2 s)
//! or for the quiet period's end ([`QUIET_LEFT`], 4 s), I6's late probes each
//! wait out [`PARK_LIMIT`] (2 s) in parallel, and the probe-timeout test waits
//! out one call ceiling of about 1.2 s.
//!
//! Worst cases, against nextest's 120 s kill timeout:
//!
//! - a test that plans or runs the Job once: one [`RUN_BUDGET`], 30 s;
//! - a test that runs several plans or Jobs (I8 (i)–(iii), the
//!   `Permanent`-then-`Transient` identity test, the `identify_listed_objects`
//!   test, and b.f3t's `restore_plan_exits_1_when_its_two_walks_of_a_pattern_disagree`
//!   (five plans) and `restore_plan_runs_on_a_store_whose_run_id_is_unusable_or_changes`
//!   (three)): one [`RUN_BUDGET`] over all of them, 30 s;
//! - I14 and I15: the Job and the wait on the store's clock run at once, each
//!   bounded by one [`RUN_BUDGET`], 30 s.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface};
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

use live_edit::checkpoint::adapters::in_memory::InMemoryCheckpointStore;
use live_edit::checkpoint::store::{LayerArtifactKind, MissingMetaCompanion};
use live_edit::checkpoint::{
    encode_tail_envelope, CheckpointAdapter, CheckpointBlob, CheckpointError, CheckpointHandle,
    CheckpointKind, CheckpointMeta, CheckpointStore, LayerArtifact, LiveExtSet, SessionCheckpoint,
    TailHeader, TailRecord,
};
use live_edit::config::Config;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section};
use live_edit::dr::job::absence::{Contradiction, OwnPuts, StoreClock, LIST_ROUTE_CAUSE};
use live_edit::dr::job::execute::KillPoint;
use live_edit::dr::job::permit::WritePermit;
use live_edit::dr::job::plan::decide::{Listing, StoreFacts, UnidentifiedObject};
use live_edit::dr::job::plan::model::{
    AbandonedGeneration, Advisory, CoverageReport, HashVerdict, HeldItem, HeldReason, ObjectAction,
    RestorePlan, RestoreReason, SessionAction, ValueSite,
};
use live_edit::dr::job::plan::{report, snapshot, PlanOutcome};
use live_edit::dr::job::quiet::{QuietPeriod, StoreDeadlines};
use live_edit::dr::job::{
    run_job, BackupAccess, JobContext, JobExit, JobMode, JobRun, RestoreBody, RestorePlanner,
};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::dr_job::{store_time_ms, CheckpointStamp};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::keys::{sha256_to_hex, KeySchema};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::{HoldMarkerSeed, TestValkey};
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand};

use super::support::{keyspace, keyspace_writes, lines_starting, WriteRules};
use super::writer_support::{
    backup_meta, context, hset, lines, metrics, put_base, put_counters, seed_complete, seed_dead,
    seed_deletion_marker, HOUR_MS, INCIDENT, RUN_BUDGET, SPEC_ATTRIBUTE, SPEC_PRIM,
    STAYS_HELD_LINE,
};
use crate::checkpoint::adapter_storage_api_test::{
    build_fake_backed_adapter, mount_fake_storage_routes, FakeStorageState,
};
use crate::grpc::log_capture::{capture, Captured};

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

/// The session most tests plan in.
const S: &str = "sess-plan";
/// The layer most tests plan.
const L: &str = "plan.usda";
/// How far ahead `T_safe` is in the tests that watch an absence seen before
/// it be re-observed at it.
const T_SAFE_AHEAD: Duration = Duration::from_secs(2);
/// How much of the quiet period is left in the defer-39 tests.
const QUIET_LEFT: Duration = Duration::from_secs(4);
/// How long before the quiet period's end the defer-39 tests land their late
/// write: after a planner reading at Q's start, before one reading at Q's end.
const LATE_WRITE_BEFORE_QUIET_END_MS: u64 = 2_000;
/// How long a parked probe waits for its peers before answering (I6).
const PARK_LIMIT: Duration = Duration::from_secs(2);

// ═════════════════════════════════════════════════════════════════════════════
// The backup store
// ═════════════════════════════════════════════════════════════════════════════

/// What a scripted generation's fetch answers instead of the stored one.
#[derive(Clone, Copy, Debug)]
enum GetFault {
    /// `Ok(None)`.
    Absent,
    /// The typed missing-meta `Permanent`.
    MissingMeta,
    /// A `Transient` (a 503 after the adapter's retries).
    Transient,
    /// A `Permanent` that is not the missing-meta one (a 403).
    Permanent,
}

/// One generation's address: `(object, seq, kind)`.
type Generation = (u64, u64, CheckpointKind);

/// A base generation scripted to land before a numbered listing:
/// `(listing, object, seq, bytes, meta)`.
type ScriptedPut = (usize, u64, u64, Bytes, CheckpointMeta);

/// A store-clock time no test reaches: "always".
const ALWAYS: u64 = u64::MAX;

/// A backup store over an in-memory one whose answers the test scripts, each
/// switched off when the store's clock reaches a chosen time, and which
/// counts every call.
struct PlanStore {
    inner: InMemoryCheckpointStore,
    client: ValkeyClient,
    keys: KeySchema,
    /// `(object, seq, kind)` → the fault and the store time it lasts until.
    get_faults: Mutex<HashMap<Generation, (GetFault, u64)>>,
    /// Values `ext_blob_exists` answers `false` for until the store time.
    ext_absent_until: Mutex<HashMap<String, u64>>,
    /// Values `ext_blob_exists` fails `Transient` for.
    ext_failing: Mutex<HashSet<String>>,
    /// Probes park until this many are in flight, or [`PARK_LIMIT`] passes.
    park_until_in_flight: Mutex<Option<usize>>,
    /// The session listing answers empty, as a denied first page reads.
    sessions_denied: Mutex<bool>,
    gets: Mutex<Vec<(u64, u64, CheckpointKind)>>,
    probes: Mutex<Vec<String>>,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
    /// The store time of the first `list_layer_artifacts`.
    first_listing_at_ms: Mutex<Option<u64>>,
    /// How many times `list_layer_artifacts` was called.
    listings: AtomicUsize,
    /// Base generations put into the store just before the listing with this
    /// 1-based number answers.
    put_at_listing: Mutex<Vec<ScriptedPut>>,
    /// Artifacts every listing answers on top of the stored ones (an orphan
    /// companion, say).
    extra_artifacts: Mutex<Vec<LayerArtifact>>,
    /// `list_layer_artifacts` fails `Transient`.
    listing_fails: Mutex<bool>,
    /// Session backups `get_session_checkpoint` answers `Ok(None)` for until
    /// the store time.
    session_absent_until: Mutex<HashMap<String, u64>>,
    /// Every `get_session_checkpoint`, by session.
    session_gets: Mutex<Vec<String>>,
    /// Hashes the ext-refs sidecar of `(object, seq)` names on top of the
    /// stored sidecar's.
    sidecar_extra: Mutex<HashMap<(u64, u64), Vec<String>>>,
    /// Values whose probe never answers.
    ext_hanging: Mutex<HashSet<String>>,
    /// Probes answer one at a time in arrival order, every one failing
    /// `Transient` but the one with this 0-based arrival number, which
    /// answers present.
    sequenced: Mutex<Option<usize>>,
    tickets: AtomicUsize,
    answered: AtomicUsize,
    /// The store time of the first `ext_blob_exists`.
    first_probe_at_ms: Mutex<Option<u64>>,
    /// Writes and deletes of any kind.
    writes: AtomicUsize,
    /// Every call of any method.
    calls: AtomicUsize,
}

impl PlanStore {
    fn new(tv: &TestValkey) -> Arc<Self> {
        Arc::new(Self {
            inner: InMemoryCheckpointStore::new(),
            client: tv.client().clone(),
            keys: tv.key_schema().clone(),
            get_faults: Mutex::new(HashMap::new()),
            ext_absent_until: Mutex::new(HashMap::new()),
            ext_failing: Mutex::new(HashSet::new()),
            park_until_in_flight: Mutex::new(None),
            sessions_denied: Mutex::new(false),
            gets: Mutex::new(Vec::new()),
            probes: Mutex::new(Vec::new()),
            in_flight: AtomicUsize::new(0),
            max_in_flight: AtomicUsize::new(0),
            first_listing_at_ms: Mutex::new(None),
            listings: AtomicUsize::new(0),
            put_at_listing: Mutex::new(Vec::new()),
            extra_artifacts: Mutex::new(Vec::new()),
            listing_fails: Mutex::new(false),
            session_absent_until: Mutex::new(HashMap::new()),
            session_gets: Mutex::new(Vec::new()),
            sidecar_extra: Mutex::new(HashMap::new()),
            ext_hanging: Mutex::new(HashSet::new()),
            sequenced: Mutex::new(None),
            tickets: AtomicUsize::new(0),
            answered: AtomicUsize::new(0),
            first_probe_at_ms: Mutex::new(None),
            writes: AtomicUsize::new(0),
            calls: AtomicUsize::new(0),
        })
    }

    /// `object`'s base at `seq` answers `fault` until the store's clock
    /// reaches `until_ms`.
    fn fault_base(&self, object: u64, seq: u64, fault: GetFault, until_ms: u64) {
        self.get_faults
            .lock()
            .unwrap()
            .insert((object, seq, CheckpointKind::Base), (fault, until_ms));
    }

    /// The session backup of `session` reads `Ok(None)` until the store's
    /// clock reaches `until_ms`.
    fn fault_session(&self, session: &str, until_ms: u64) {
        self.session_absent_until
            .lock()
            .unwrap()
            .insert(session.to_owned(), until_ms);
    }

    fn session_gets_of(&self, session: &str) -> usize {
        self.session_gets
            .lock()
            .unwrap()
            .iter()
            .filter(|s| *s == session)
            .count()
    }

    /// A base of `layer` in `session` at `seq` lands just before the listing
    /// numbered `listing` (1-based) answers.
    fn put_base_at_listing(
        &self,
        listing: usize,
        object_id: u64,
        session: &str,
        layer: &str,
        seq: u64,
        bytes: Bytes,
    ) {
        self.put_at_listing.lock().unwrap().push((
            listing,
            object_id,
            seq,
            bytes,
            backup_meta(session, layer, seq),
        ));
    }

    fn absent_until(&self, hash: &[u8; 32], until_ms: u64) {
        self.ext_absent_until
            .lock()
            .unwrap()
            .insert(sha256_to_hex(hash), until_ms);
    }

    fn failing(&self, hash: &[u8; 32]) {
        self.ext_failing.lock().unwrap().insert(sha256_to_hex(hash));
    }

    fn gets_of(&self, object: u64, seq: u64) -> usize {
        self.gets
            .lock()
            .unwrap()
            .iter()
            .filter(|g| (g.0, g.1) == (object, seq))
            .count()
    }

    fn probes_of(&self, hash: &[u8; 32]) -> usize {
        let hex = sha256_to_hex(hash);
        self.probes
            .lock()
            .unwrap()
            .iter()
            .filter(|p| **p == hex)
            .count()
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    async fn now_ms(&self) -> u64 {
        store_time_ms(&self.client, &self.keys).await.expect("TIME")
    }

    fn call(&self) {
        self.calls.fetch_add(1, Ordering::SeqCst);
    }

    fn write(&self) {
        self.call();
        self.writes.fetch_add(1, Ordering::SeqCst);
    }
}

fn fault_error(
    fault: GetFault,
    handle: &CheckpointHandle,
) -> Result<Option<CheckpointBlob>, CheckpointError> {
    match fault {
        GetFault::Absent => Ok(None),
        GetFault::MissingMeta => Err(CheckpointError::missing_meta_companion(
            MissingMetaCompanion {
                artifact: format!("{}.{}.ompf", handle.object_id, handle.seq()),
                companion: format!("{}.{}.meta.json", handle.object_id, handle.seq()),
                object_id: handle.object_id,
                sequence: handle.seq(),
                kind: handle.kind,
            },
        )),
        GetFault::Transient => Err(CheckpointError::Transient {
            source: "the backup store answered 503".into(),
        }),
        GetFault::Permanent => Err(CheckpointError::Permanent {
            source: "the backup store answered 403".into(),
        }),
    }
}

#[async_trait]
impl CheckpointStore for PlanStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.write();
        self.inner
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }
    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.call();
        self.inner.get_latest_layer_checkpoint(object_id).await
    }
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.call();
        let key = (handle.object_id, handle.seq(), handle.kind);
        self.gets.lock().unwrap().push(key);
        let scripted = self.get_faults.lock().unwrap().get(&key).copied();
        if let Some((fault, until_ms)) = scripted {
            if until_ms == ALWAYS || self.now_ms().await < until_ms {
                return fault_error(fault, handle);
            }
        }
        self.inner.get_layer_checkpoint(handle).await
    }
    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.call();
        self.inner.list_layer_checkpoints(object_id).await
    }
    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.write();
        self.inner.delete_layer_checkpoint(handle).await
    }
    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.write();
        self.inner
            .put_session_checkpoint(session_id, body, meta)
            .await
    }
    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.call();
        self.session_gets
            .lock()
            .unwrap()
            .push(session_id.to_owned());
        let absent_until = self
            .session_absent_until
            .lock()
            .unwrap()
            .get(session_id)
            .copied();
        if let Some(until_ms) = absent_until {
            if self.now_ms().await < until_ms {
                return Ok(None);
            }
        }
        self.inner.get_session_checkpoint(session_id).await
    }
    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.write();
        self.inner.delete_session_checkpoint(session_id).await
    }
    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.call();
        if *self.sessions_denied.lock().unwrap() {
            return Ok(Vec::new());
        }
        self.inner.list_session_checkpoint_ids().await
    }
    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.write();
        self.inner.put_ext_blob(hash_hex, bytes).await
    }
    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.call();
        self.inner.get_ext_blob(hash_hex).await
    }
    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.call();
        self.probes.lock().unwrap().push(hash_hex.to_owned());
        let now_in_flight = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight
            .fetch_max(now_in_flight, Ordering::SeqCst);
        if self.first_probe_at_ms.lock().unwrap().is_none() {
            let now = self.now_ms().await;
            self.first_probe_at_ms.lock().unwrap().get_or_insert(now);
        }
        if self.ext_hanging.lock().unwrap().contains(hash_hex) {
            std::future::pending::<()>().await;
        }
        let sequenced = *self.sequenced.lock().unwrap();
        if let Some(present_at) = sequenced {
            // Answer strictly in arrival order: this probe completes only
            // once every earlier one has, so the planner sees the answers in
            // that order.
            let ticket = self.tickets.fetch_add(1, Ordering::SeqCst);
            while self.answered.load(Ordering::SeqCst) != ticket {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            let answer = if ticket == present_at {
                Ok(true)
            } else {
                Err(CheckpointError::Transient {
                    source: "the backup store answered 503".into(),
                })
            };
            self.answered.fetch_add(1, Ordering::SeqCst);
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            return answer;
        }
        let park = *self.park_until_in_flight.lock().unwrap();
        if let Some(peers) = park {
            let deadline = tokio::time::Instant::now() + PARK_LIMIT;
            while self.in_flight.load(Ordering::SeqCst) < peers
                && tokio::time::Instant::now() < deadline
            {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }
        let failing = self.ext_failing.lock().unwrap().contains(hash_hex);
        let absent_until = self.ext_absent_until.lock().unwrap().get(hash_hex).copied();
        let answer = if failing {
            Err(CheckpointError::Transient {
                source: "the backup store answered 503".into(),
            })
        } else if absent_until.is_some_and(|until| until == ALWAYS) {
            Ok(false)
        } else {
            match absent_until {
                Some(until) if self.now_ms().await < until => Ok(false),
                _ => self.inner.ext_blob_exists(hash_hex).await,
            }
        };
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        answer
    }
    async fn verify_list_route(&self) -> Result<(), CheckpointError> {
        self.call();
        self.inner.verify_list_route().await
    }
    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.write();
        self.inner.delete_ext_blob(hash_hex).await
    }
    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.write();
        self.inner.put_counters_sidecar(body).await
    }
    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.call();
        self.inner.get_counters_sidecar().await
    }
    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.call();
        self.inner.list_all_object_ids().await
    }
    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        self.call();
        self.inner.list_referenced_ext_hashes().await
    }
    async fn list_live_ext_set(&self) -> Result<LiveExtSet, CheckpointError> {
        self.call();
        self.inner.list_live_ext_set().await
    }
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<HashSet<String>, CheckpointError> {
        self.call();
        let mut listed = self.inner.read_ext_refs_sidecar(handle).await?;
        if let Some(extra) = self
            .sidecar_extra
            .lock()
            .unwrap()
            .get(&(handle.object_id, handle.seq()))
        {
            listed.extend(extra.iter().cloned());
        }
        Ok(listed)
    }
    async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
        self.call();
        let listing = self.listings.fetch_add(1, Ordering::SeqCst) + 1;
        let first = self.first_listing_at_ms.lock().unwrap().is_none();
        if first {
            let now = self.now_ms().await;
            self.first_listing_at_ms.lock().unwrap().get_or_insert(now);
        }
        let due: Vec<_> = {
            let mut pending = self.put_at_listing.lock().unwrap();
            let (due, later): (Vec<_>, Vec<_>) = pending.drain(..).partition(|p| p.0 == listing);
            *pending = later;
            due
        };
        for (_, object_id, seq, bytes, meta) in due {
            self.inner
                .put_layer_checkpoint(object_id, seq, bytes, meta)
                .await
                .expect("the scripted put");
        }
        if *self.listing_fails.lock().unwrap() {
            return Err(CheckpointError::Transient {
                source: "the backup store answered 503".into(),
            });
        }
        let mut artifacts = self.inner.list_layer_artifacts().await?;
        artifacts.extend(self.extra_artifacts.lock().unwrap().iter().copied());
        Ok(artifacts)
    }
    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.write();
        self.inner
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }
    fn publish_worst_case(&self) -> Duration {
        self.inner.publish_worst_case()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Backup fixtures
// ═════════════════════════════════════════════════════════════════════════════

/// A base OMPF of a layer with the prim `/World` and, per `(name, hash)`, an
/// attribute of `/World` whose `default` is the external value `hash`. A base
/// carries hash references with no value parts, as the assembler writes it.
fn base_ompf(attributes: &[(&str, [u8; 32])]) -> Bytes {
    let mut struct_commands = vec![StructCommand::CreateSection(CommandCreateSection {
        parent_id: 0,
        section_id: 1,
        section_name: "World".to_owned(),
        section_type: SPEC_PRIM,
    })];
    let mut set_fields = Vec::new();
    for (i, (name, hash)) in attributes.iter().enumerate() {
        let section_id = 2 + i as u64;
        struct_commands.push(StructCommand::CreateSection(CommandCreateSection {
            parent_id: 1,
            section_id,
            section_name: (*name).to_owned(),
            section_type: SPEC_ATTRIBUTE,
        }));
        set_fields.push(SetField {
            section_id,
            key_name: "default".to_owned(),
            value_or_ext_hash: hash.to_vec(),
            ext_value_size: 4_096,
            ext_value_index: 1,
            set_order: i as u64,
        });
    }
    let delta = OwnedDelta {
        is_diff: true,
        base_version: 0,
        struct_commands,
        set_fields,
        time_samples: Vec::new(),
        source_format: "usda".to_owned(),
    };
    let delta_bytes = oodf_ffi::build_delta(&delta).expect("the base delta builds");
    Bytes::from(OmpfContainer::serialize(&delta_bytes, &[]).expect("the base serializes"))
}

/// A session backup of `session` mapping each `(layer, object)`.
fn session_backup(session: &str, mappings: &[(&str, u64)]) -> SessionCheckpoint {
    SessionCheckpoint {
        schema_version: 1,
        session_id: session.to_owned(),
        name: format!("{session}-name"),
        base_path: format!("stage:/{session}.usd"),
        created_by: "user".to_owned(),
        created_at: 1,
        layer_mappings: mappings
            .iter()
            .map(|(l, o)| ((*l).to_owned(), *o))
            .collect(),
        layers_index: mappings.iter().map(|(l, _)| (*l).to_owned()).collect(),
        path_name_members: BTreeSet::new(),
        path_index_session_id_member: session.to_owned(),
    }
}

async fn put_session(store: &dyn CheckpointStore, session: &str, mappings: &[(&str, u64)]) {
    store
        .put_session_checkpoint(
            session,
            session_backup(session, mappings)
                .to_bytes()
                .expect("the session backup encodes"),
            backup_meta(session, "", 0),
        )
        .await
        .expect("the session put succeeds");
}

/// A distinct external value.
fn value(byte: u8) -> [u8; 32] {
    [byte; 32]
}

// ═════════════════════════════════════════════════════════════════════════════
// Store fixtures
// ═════════════════════════════════════════════════════════════════════════════

async fn now_ms(tv: &TestValkey) -> u64 {
    store_time_ms(tv.client(), tv.key_schema())
        .await
        .expect("TIME")
}

/// A surviving object header with `fields`.
async fn seed_header(tv: &TestValkey, object_id: u64, fields: &[(&str, &str)]) {
    hset(
        tv,
        tv.key_schema().obj_header(&object_id.to_string()),
        fields,
    )
    .await;
}

/// A layer mapping alone, with no layers-index member (the writer tests'
/// `writer_support::seed_mapping` adds the member).
async fn seed_mapping(tv: &TestValkey, session: &str, layer: &str, object_id: u64) {
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().layer_mapping(session, layer),
            object_id.to_string(),
            None,
            None,
            false,
        )
        .await
        .expect("SET mapping");
}

/// A session whose record survives.
async fn seed_session(tv: &TestValkey, session: &str) {
    hset(
        tv,
        tv.key_schema().session_meta(session),
        &[("name", "session"), ("base_path", "stage:/s.usd")],
    )
    .await;
}

/// A surviving object's `obj:ext` index naming `hashes`.
async fn seed_obj_ext(tv: &TestValkey, object_id: u64, hashes: &[[u8; 32]]) {
    let fields: Vec<(String, String)> = hashes
        .iter()
        .map(|h| (sha256_to_hex(h), "1".to_owned()))
        .collect();
    let _: i64 = tv
        .client()
        .pool
        .hset(
            tv.key_schema().obj_ext_index(&object_id.to_string()),
            fields,
        )
        .await
        .expect("HSET obj:ext");
}

/// Valkey holds the external value `hash`'s blob.
async fn seed_ext_blob(tv: &TestValkey, hash: &[u8; 32]) {
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().ext_data(&sha256_to_hex(hash)),
            "blob",
            None,
            None,
            false,
        )
        .await
        .expect("SET ext data");
}

// ═════════════════════════════════════════════════════════════════════════════
// Running the planner
// ═════════════════════════════════════════════════════════════════════════════

/// Deadlines whose quiet period has ended, with `T_safe` `t_safe_in` from
/// now on the store's clock (`Duration::ZERO`: already past).
async fn deadlines(tv: &TestValkey, t_safe_in: Duration) -> StoreDeadlines {
    let now = now_ms(tv).await;
    let ahead = u64::try_from(t_safe_in.as_millis()).expect("fits");
    StoreDeadlines {
        detected_at_ms: now - HOUR_MS,
        quiet_ends_at_ms: now - 1,
        t_safe_ms: if ahead == 0 { now - 1 } else { now + ahead },
    }
}

/// A restore run on `tv` reading backups from `backup`, with `deadlines`.
fn job_run(
    tv: &TestValkey,
    backup: BackupAccess,
    deadlines: StoreDeadlines,
    metrics: &Arc<Metrics>,
) -> JobRun {
    let config = crate::common::test_config();
    JobRun {
        mode: JobMode::Restore,
        incident: INCIDENT.to_owned(),
        quiet: QuietPeriod::from_config(&config),
        not_found_cache: T_SAFE_AHEAD,
        config: Arc::new(config),
        valkey: tv.client().clone(),
        keys: tv.key_schema().clone(),
        backup,
        clock: StoreClock::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            deadlines,
            Duration::from_millis(250),
        ),
        permit: WritePermit::new(),
        own_puts: OwnPuts::default(),
        metrics: Arc::clone(metrics),
        lease_holder: String::new(),
        counters_sidecar_max: std::sync::OnceLock::new(),
        allocator_purge: Some(live_edit::storage::allocator_purge::AllocatorPurge::new(
            || true,
        )),
    }
}

/// A run reading `store`, `T_safe` `t_safe_in` from now.
async fn run_on(
    tv: &TestValkey,
    store: &Arc<PlanStore>,
    t_safe_in: Duration,
) -> (JobRun, Arc<Metrics>) {
    let metrics = metrics();
    let backup = BackupAccess::Store(Arc::clone(store) as Arc<dyn CheckpointStore>);
    let run = job_run(tv, backup, deadlines(tv, t_safe_in).await, &metrics);
    (run, metrics)
}

/// Run `future` within [`RUN_BUDGET`], capturing what it logs.
async fn captured<F: Future>(future: F) -> (F::Output, Captured) {
    capture(async {
        tokio::time::timeout(RUN_BUDGET, future)
            .await
            .expect("the run ends")
    })
    .await
}

/// Bound a test's several plans or Job runs by one [`RUN_BUDGET`] together,
/// so the test's worst case stays one budget however many runs it makes.
async fn within_budget<F: Future>(future: F) -> F::Output {
    tokio::time::timeout(RUN_BUDGET, future)
        .await
        .expect("the test's runs end within one budget")
}

/// Plan, then finalise the coverage check, as the body does after its writes.
async fn plan_and_finalize(run: &JobRun) -> Result<(RestorePlan, CoverageReport), JobExit> {
    let plan = RestorePlanner::plan(run).await?;
    let report = plan.coverage.finalize(run).await?;
    Ok((plan, report))
}

fn object_action(plan: &RestorePlan, object_id: u64) -> &ObjectAction {
    let found: Vec<_> = plan
        .objects
        .iter()
        .filter(|o| o.object_id == object_id)
        .collect();
    assert_eq!(
        found.len(),
        1,
        "object {object_id} is planned once: {:?}",
        plan.objects
    );
    &found[0].action
}

fn planned(plan: &RestorePlan) -> BTreeSet<u64> {
    plan.objects.iter().map(|o| o.object_id).collect()
}

/// The head of `object_id`'s planned restore.
fn restored_head(plan: &RestorePlan, object_id: u64) -> u64 {
    match object_action(plan, object_id) {
        ObjectAction::Restore { lineage, .. } => lineage.head_seq,
        other => panic!("object {object_id} must be restored, got {other:?}"),
    }
}

fn session_action<'p>(plan: &'p RestorePlan, session: &str) -> &'p SessionAction {
    let found: Vec<_> = plan
        .sessions
        .iter()
        .filter(|p| p.session_id == session)
        .collect();
    assert_eq!(found.len(), 1, "session {session} is planned once");
    &found[0].action
}

/// The one line `message` at `level`.
fn one<'c>(captured: &'c Captured, level: &str, message: &str) -> &'c serde_json::Value {
    let found = lines(captured, message);
    assert_eq!(
        found.len(),
        1,
        "one {level} line {message:?}; capture:\n{}",
        captured.text
    );
    assert_eq!(found[0]["level"], level, "{message}");
    found[0]
}

/// Fixture premise of a test about an absence seen before `T_safe`: the
/// plan's first listing was taken before `T_safe`. Asserted first, so a run
/// too slow to see `T_safe` ahead reads as a fixture miss, not a planner bug.
fn assert_listed_before_t_safe(store: &PlanStore, run: &JobRun) {
    let listed_at = store
        .first_listing_at_ms
        .lock()
        .unwrap()
        .expect("the backup store was listed");
    let t_safe_ms = run.deadlines().t_safe_ms;
    assert!(
        listed_at < t_safe_ms,
        "fixture: the first listing, at {listed_at} ms, must be before T_safe at {t_safe_ms} ms"
    );
}

/// [`assert_listed_before_t_safe`] for a coverage probe: the first probe was
/// taken before `T_safe`.
fn assert_probed_before_t_safe(store: &PlanStore, run: &JobRun) {
    let probed_at = store
        .first_probe_at_ms
        .lock()
        .unwrap()
        .expect("the backup store was probed");
    let t_safe_ms = run.deadlines().t_safe_ms;
    assert!(
        probed_at < t_safe_ms,
        "fixture: the first probe, at {probed_at} ms, must be before T_safe at {t_safe_ms} ms"
    );
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

fn probe_cell(metrics: &Metrics, result: &str) -> f64 {
    sample_value(
        &render(metrics),
        "live_edit_dr_restore_coverage_probes_total",
        &[("result", result)],
    )
    .unwrap_or_else(|| panic!("the result={result} cell renders"))
}

fn uncovered_gauge(metrics: &Metrics) -> f64 {
    sample_value(
        &render(metrics),
        "live_edit_dr_restore_values_uncovered",
        &[],
    )
    .expect("the gauge renders")
}

// ═════════════════════════════════════════════════════════════════════════════
// Running the Job
// ═════════════════════════════════════════════════════════════════════════════

/// A StorageApi configuration whose not-found cache lifetime is `l_secs`.
fn storage_api_config(l_secs: u64) -> Config {
    let mut config = crate::common::test_config();
    config.checkpoint_adapter = CheckpointAdapter::StorageApi;
    config.checkpoint_storage_api_not_found_cache_secs = l_secs;
    config
}

/// Hold the store under [`INCIDENT`] with its quiet period under `config`
/// ending `left` from now on the store's clock. Returns Q's end.
async fn hold_with_quiet_left(tv: &TestValkey, config: &Config, left: Duration) -> u64 {
    let q = u64::try_from(QuietPeriod::from_config(config).total().as_millis()).expect("Q fits");
    let left = u64::try_from(left.as_millis()).expect("fits");
    let now = now_ms(tv).await;
    tv.write_hold_marker(
        &HoldMarkerSeed::default()
            .with_incident(INCIDENT)
            .detected_at(now + left - q),
    )
    .await;
    now + left
}

/// What [`observing_body`] saw: each `(plan, report)` it was handed.
type Observed = Arc<Mutex<Vec<(RestorePlan, CoverageReport)>>>;

/// The kill line of a run stopped just before its un-hold
/// ([`observing_body`]).
const BEFORE_UNHOLD_KILL: &str = "the restore Job stopped at its test kill point BeforeUnhold, \
                                  as a kill there would; it writes nothing more";

/// The restore body, recording the plan and report it finishes with, and
/// stopping just before its un-hold (`KillPoint::BeforeUnhold`), so a
/// writable plan's run exits 1 with the marker left, as these planner tests
/// read it; a plan that stays held exits 3 after its recount.
fn observing_body() -> (RestoreBody, Observed) {
    let observed: Observed = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&observed);
    let body = RestoreBody::new()
        .with_kill_point_for_test(KillPoint::BeforeUnhold)
        .with_plan_observer(Arc::new(move |plan, report| {
            sink.lock().unwrap().push((plan.clone(), report.clone()));
        }));
    (body, observed)
}

/// Run the Job to its exit within [`RUN_BUDGET`], capturing what it logs.
async fn run(ctx: JobContext, body: RestoreBody) -> (JobExit, Captured) {
    captured(run_job(ctx, body)).await
}

/// The one plan the body was handed.
fn the_plan(observed: &Observed) -> (RestorePlan, CoverageReport) {
    let observed = observed.lock().unwrap();
    assert_eq!(observed.len(), 1, "the body finishes one plan");
    observed[0].clone()
}

// ═════════════════════════════════════════════════════════════════════════════
// I1, I19: a full wipe, through the Job, stopped before its un-hold
// ═════════════════════════════════════════════════════════════════════════════

/// I1 and I19. After a full wipe, with every session and layer backed up,
/// the plan restores every session (Step B, every mapping) and every mapped
/// layer, and is writable. The body writes the plan, then finalises the
/// coverage check, logs the summary, hands the plan to the observer,
/// recounts, and stops at its test kill point before the un-hold (exit 1,
/// [`observing_body`]). Its base-only lineages carry no value, so nothing is
/// written to the backup store. (What the writer wrote to the store is
/// W1's.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_after_a_full_wipe_plans_every_session_and_layer_then_stops_before_its_unhold()
{
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    put_session(store.as_ref(), "s-a", &[("a1.usda", 101), ("a2.usda", 102)]).await;
    put_session(store.as_ref(), "s-b", &[("b1.usda", 201)]).await;
    put_base(store.as_ref(), 101, "s-a", "a1.usda", 5, base_ompf(&[])).await;
    put_base(store.as_ref(), 102, "s-a", "a2.usda", 7, base_ompf(&[])).await;
    put_base(store.as_ref(), 201, "s-b", "b1.usda", 9, base_ompf(&[])).await;
    put_counters(store.as_ref()).await;
    let config = storage_api_config(0);
    hold_with_quiet_left(&tv, &config, Duration::ZERO).await;
    let writes_before = store.writes.load(Ordering::SeqCst);
    let metrics = metrics();
    let (body, observed) = observing_body();

    let (exit, captured) = run(
        context(
            &tv,
            JobMode::Restore,
            config,
            Arc::clone(&store) as _,
            &metrics,
        ),
        body,
    )
    .await;

    assert_eq!(exit, JobExit::Failed);
    one(&captured, "INFO", BEFORE_UNHOLD_KILL);
    one(
        &captured,
        "INFO",
        "the restore plan: 3 layers to restore, 0 kept, 0 moved to their successor, 0 retirement \
         flags to clear, 0 keeping the service held, 2 sessions to restore, 0 unfinished deletes \
         to finish; outcome: writable; 0 layers restored from an older backup because the newest \
         was unreadable; 0 layers restored from an older backup because the newest needs a \
         backup file absent from the backup store at T_safe; 0 sessions named by surviving data \
         or a backup have no record and no session backup and are not restored",
    );
    let (plan, report) = the_plan(&observed);
    assert_eq!(plan.outcome(&report), PlanOutcome::Writable);
    for (session, mappings) in [
        ("s-a", vec![("a1.usda", 101), ("a2.usda", 102)]),
        ("s-b", vec![("b1.usda", 201)]),
    ] {
        match session_action(&plan, session) {
            SessionAction::Restore { checkpoint, step_b } => {
                assert!(*step_b, "{session}: the record is gone");
                let want: BTreeMap<String, u64> = mappings
                    .iter()
                    .map(|(l, o)| ((*l).to_owned(), *o))
                    .collect();
                assert_eq!(checkpoint.layer_mappings, want, "{session}");
            }
            other => panic!("{session}: expected a restore, got {other:?}"),
        }
    }
    for (object_id, head) in [(101, 5), (102, 7), (201, 9)] {
        assert_eq!(restored_head(&plan, object_id), head);
        assert!(matches!(
            object_action(&plan, object_id),
            ObjectAction::Restore {
                reason: RestoreReason::HeaderAbsent,
                ..
            }
        ));
    }
    assert!(plan.held.is_empty(), "{:?}", plan.held);
    assert_eq!(plan.listed_max_object_id, Some(201));
    assert_eq!(plan.surviving_max_object_id, None);
    assert_eq!(
        store.writes.load(Ordering::SeqCst),
        writes_before,
        "nothing is written to the backup store"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// I2, I3: survivors
// ═════════════════════════════════════════════════════════════════════════════

/// I2. A partial wipe: only the lost object is restored; a survivor newer
/// than its backup is kept; the session backup's mappings match the
/// surviving ones, so Step B does not run.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_after_a_partial_wipe_restores_only_the_lost_object() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    put_session(store.as_ref(), S, &[("kept.usda", 101), ("lost.usda", 102)]).await;
    put_base(store.as_ref(), 101, S, "kept.usda", 40, base_ompf(&[])).await;
    put_base(store.as_ref(), 102, S, "lost.usda", 30, base_ompf(&[])).await;
    seed_session(&tv, S).await;
    seed_complete(&tv, 101, S, "kept.usda", 50, &[]).await;
    seed_mapping(&tv, S, "kept.usda", 101).await;
    seed_mapping(&tv, S, "lost.usda", 102).await;
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (result, _captured) = captured(plan_and_finalize(&run)).await;
    let (plan, report) = result.expect("the plan");

    assert!(matches!(object_action(&plan, 101), ObjectAction::Keep));
    assert_eq!(restored_head(&plan, 102), 30);
    match session_action(&plan, S) {
        SessionAction::Restore { checkpoint, step_b } => {
            assert!(
                checkpoint.layer_mappings.is_empty(),
                "every mapping survives"
            );
            assert!(!step_b);
        }
        other => panic!("expected a session restore, got {other:?}"),
    }
    assert_eq!(plan.outcome(&report), PlanOutcome::Writable);
    assert_eq!(plan.surviving_max_object_id, Some(101));
}

/// I3. A retiring survivor with a complete successor of the same layer is
/// remapped to it, and the successor restored when its backup is newer; one
/// whose successor is of another layer has its flags cleared when no newer
/// backup is listed, and is restored from a newer one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_remaps_to_a_matching_successor_and_otherwise_clears_or_restores_by_backup_head(
) {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    put_session(
        store.as_ref(),
        S,
        &[("a.usda", 301), ("b.usda", 303), ("c.usda", 305)],
    )
    .await;
    seed_session(&tv, S).await;
    // a: 301 → 302, the same layer; 302's backup is newer than it.
    seed_complete(
        &tv,
        301,
        S,
        "a.usda",
        10,
        &[("superseded", "1"), ("superseded_by", "302")],
    )
    .await;
    seed_complete(&tv, 302, S, "a.usda", 20, &[]).await;
    seed_mapping(&tv, S, "a.usda", 301).await;
    put_base(store.as_ref(), 302, S, "a.usda", 25, base_ompf(&[])).await;
    // b: 303's successor 304 is of another layer; no newer backup.
    seed_complete(
        &tv,
        303,
        S,
        "b.usda",
        30,
        &[("superseded", "1"), ("superseded_by", "304")],
    )
    .await;
    seed_complete(&tv, 304, S, "other-b.usda", 31, &[]).await;
    seed_mapping(&tv, S, "b.usda", 303).await;
    // c: 305's successor 306 is of another layer; a newer backup of 305.
    seed_complete(
        &tv,
        305,
        S,
        "c.usda",
        50,
        &[("superseded", "1"), ("superseded_by", "306")],
    )
    .await;
    seed_complete(&tv, 306, S, "other-c.usda", 51, &[]).await;
    seed_mapping(&tv, S, "c.usda", 305).await;
    put_base(store.as_ref(), 305, S, "c.usda", 60, base_ompf(&[])).await;
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (plan, _captured) = captured(RestorePlanner::plan(&run)).await;
    let plan = plan.expect("the plan");

    assert!(matches!(
        object_action(&plan, 301),
        ObjectAction::Remap { from: 301, to: 302 }
    ));
    assert!(matches!(
        object_action(&plan, 302),
        ObjectAction::Restore {
            reason: RestoreReason::NewerThanSurvivor { survivor_seq: 20 },
            ..
        }
    ));
    assert!(matches!(
        object_action(&plan, 303),
        ObjectAction::ClearFlags { band } if band.floor_seq == 30
    ));
    assert!(matches!(
        object_action(&plan, 305),
        ObjectAction::Restore {
            reason: RestoreReason::NewerThanRetiringSurvivor { survivor_seq: 50 },
            ..
        }
    ));
    assert_eq!(restored_head(&plan, 305), 60);
    assert_eq!(planned(&plan), BTreeSet::from([301, 302, 303, 305]));
    assert!(plan.held.is_empty(), "{:?}", plan.held);
}

// ═════════════════════════════════════════════════════════════════════════════
// I4, I5, I20: coverage verdicts, the report and the metrics
// ═════════════════════════════════════════════════════════════════════════════

/// A tail record: an ordinary commit adding the attribute `normals` under
/// `/World` (section 9) whose `default` is the external value `hash`, the
/// value's bytes carried in the record.
fn tail_record_adding(hash: [u8; 32]) -> Bytes {
    let delta = OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 1,
            section_id: 9,
            section_name: "normals".to_owned(),
            section_type: SPEC_ATTRIBUTE,
        })],
        set_fields: vec![SetField {
            section_id: 9,
            key_name: "default".to_owned(),
            value_or_ext_hash: hash.to_vec(),
            ext_value_size: 16,
            ext_value_index: 1,
            set_order: 0,
        }],
        time_samples: Vec::new(),
        source_format: String::new(),
    };
    let delta_bytes = oodf_ffi::build_delta(&delta).expect("the record delta builds");
    Bytes::from(
        OmpfContainer::serialize(&delta_bytes, &[b"tail-carried val".as_slice()])
            .expect("the record serializes"),
    )
}

/// I4. A value only a restored lineage's tails carry is covered by them: it
/// is never probed and never reported. A value only the backup store holds
/// is probed once and covered.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_coverage_counts_a_value_only_the_restored_tails_carry_as_covered() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    let in_base = value(0x41);
    let in_tail = value(0x42);
    put_session(store.as_ref(), S, &[(L, 401)]).await;
    put_base(
        store.as_ref(),
        401,
        S,
        L,
        5,
        base_ompf(&[("points", in_base)]),
    )
    .await;
    store
        .put_ext_blob(&sha256_to_hex(&in_base), Bytes::from_static(b"base value"))
        .await
        .expect("the value put");
    let envelope = encode_tail_envelope(
        &TailHeader {
            object_id: 401,
            base_snap_seq: 5,
            from_seq: 5,
            to_seq: 6,
        },
        vec![TailRecord::Verbatim {
            sequence_num: 6,
            server_ts: 1,
            ompf: tail_record_adding(in_tail),
        }],
    )
    .expect("the tail encodes");
    store
        .put_layer_checkpoint(
            401,
            6,
            envelope,
            CheckpointMeta {
                kind: CheckpointKind::Tail,
                base_snap_seq: Some(5),
                from_seq: Some(5),
                ..backup_meta(S, L, 6)
            },
        )
        .await
        .expect("the tail put");
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (result, _captured) = captured(plan_and_finalize(&run)).await;
    let (plan, report) = result.expect("the plan");

    match object_action(&plan, 401) {
        ObjectAction::Restore { lineage, .. } => {
            assert_eq!(lineage.head_seq, 6);
            assert_eq!(lineage.tails.len(), 1, "the composed lineage is restored");
        }
        other => panic!("expected a restore, got {other:?}"),
    }
    assert_eq!(report.verdicts.get(&in_tail), Some(&HashVerdict::Tails));
    assert_eq!(report.verdicts.get(&in_base), Some(&HashVerdict::Backup));
    assert_eq!(
        store.probes_of(&in_tail),
        0,
        "a tail-carried value is never probed"
    );
    assert_eq!(store.probes_of(&in_base), 1);
    assert!(report.layers.is_empty(), "{:?}", report.layers);
    assert_eq!(plan.outcome(&report), PlanOutcome::Writable);
}

/// I5 and I20. A restored layer references four values: one in Valkey (not
/// probed), one in the backup store, one in no store, one the backup store
/// cannot answer for. The report names the two that keep the service held —
/// session, layer, prim path, attribute, count, hash prefix — the run stays
/// held, the gauge reads 1, and the probe counter has one cell per answer.
/// A server process's metrics render neither family.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_coverage_reports_an_uncovered_and_an_unknown_value_by_layer_and_attribute() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    let in_valkey = value(0x51);
    let in_backup = value(0x52);
    let nowhere = value(0x53);
    let unknown = value(0x54);
    put_session(store.as_ref(), S, &[(L, 501)]).await;
    put_base(
        store.as_ref(),
        501,
        S,
        L,
        5,
        base_ompf(&[
            ("colors", in_valkey),
            ("uvs", in_backup),
            ("points", nowhere),
            ("normals", unknown),
        ]),
    )
    .await;
    store
        .put_ext_blob(
            &sha256_to_hex(&in_backup),
            Bytes::from_static(b"in the backup"),
        )
        .await
        .expect("the value put");
    store.failing(&unknown);
    seed_ext_blob(&tv, &in_valkey).await;
    let (run, metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (result, captured) = captured(plan_and_finalize(&run)).await;
    let (plan, report) = result.expect("the plan");

    assert_eq!(report.verdicts.get(&in_valkey), Some(&HashVerdict::Valkey));
    assert_eq!(report.verdicts.get(&in_backup), Some(&HashVerdict::Backup));
    assert_eq!(report.verdicts.get(&nowhere), Some(&HashVerdict::Uncovered));
    assert!(matches!(
        report.verdicts.get(&unknown),
        Some(HashVerdict::Unknown(e)) if e.contains("503")
    ));
    assert_eq!(
        store.probes_of(&in_valkey),
        0,
        "Valkey's copy is never probed"
    );
    for hash in [in_backup, nowhere, unknown] {
        assert_eq!(store.probes_of(&hash), 1, "one probe per value");
    }
    assert_eq!(report.layers.len(), 1, "{:?}", report.layers);
    let layer = &report.layers[0];
    assert_eq!(
        (
            layer.object_id,
            layer.session_id.as_str(),
            layer.layer_name.as_str()
        ),
        (501, S, L)
    );
    assert_eq!(layer.uncovered, BTreeSet::from([nowhere]));
    assert_eq!(layer.unknown, BTreeSet::from([unknown]));
    let site = |attribute: &str, hash| ValueSite {
        prim_path: "/World".to_owned(),
        attribute: attribute.to_owned(),
        hash,
        count: 1,
    };
    assert_eq!(layer.uncovered_sites, vec![site("points", nowhere)]);
    assert_eq!(layer.unknown_sites, vec![site("normals", unknown)]);
    assert_eq!(layer.naming_error, None);
    assert_eq!(plan.outcome(&report), PlanOutcome::StaysHeld);

    let uncovered_line = one(
        &captured,
        "WARN",
        "layer plan.usda of session sess-plan references 1 values that are in no store",
    );
    assert_eq!(
        uncovered_line["fields"]["values"],
        format!("/World points ×1 {}", &sha256_to_hex(&nowhere)[..16])
    );
    let unknown_line = one(
        &captured,
        "WARN",
        "layer plan.usda of session sess-plan references 1 values whose presence the backup \
         store could not confirm",
    );
    assert_eq!(
        unknown_line["fields"]["values"],
        format!("/World normals ×1 {}", &sha256_to_hex(&unknown)[..16])
    );

    assert_eq!(probe_cell(&metrics, "present"), 1.0);
    assert_eq!(probe_cell(&metrics, "absent"), 1.0);
    assert_eq!(probe_cell(&metrics, "error"), 1.0);
    assert_eq!(uncovered_gauge(&metrics), 1.0);
    let server = render(&crate::common::metrics_scrape_only("restore-plan-server"));
    for family in [
        "live_edit_dr_restore_coverage_probes_total",
        "live_edit_dr_restore_values_uncovered",
    ] {
        assert!(
            !server.contains(family),
            "a server's metrics never render {family}"
        );
    }
}

/// I6. The probe stage keeps at most 32 checks in flight, one adapter call
/// per value, with no retry: 40 values the backup store holds, each probe
/// parked until 32 are in flight.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn restore_coverage_probes_at_most_32_values_at_once_one_call_each() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    let hashes: Vec<[u8; 32]> = (0..40u8).map(|i| value(0x80 + i)).collect();
    for hash in &hashes {
        store
            .put_ext_blob(&sha256_to_hex(hash), Bytes::from_static(b"held"))
            .await
            .expect("the value put");
    }
    *store.park_until_in_flight.lock().unwrap() = Some(32);
    seed_session(&tv, S).await;
    seed_complete(&tv, 601, S, L, 10, &[]).await;
    seed_mapping(&tv, S, L, 601).await;
    seed_obj_ext(&tv, 601, &hashes).await;
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (plan, _captured) = captured(RestorePlanner::plan(&run)).await;
    let plan = plan.expect("the plan");

    assert_eq!(store.max_in_flight.load(Ordering::SeqCst), 32);
    let probes = store.probes.lock().unwrap().clone();
    assert_eq!(probes.len(), 40, "one adapter call per value");
    assert_eq!(
        probes.iter().collect::<HashSet<_>>().len(),
        40,
        "no value is probed twice"
    );
    assert!(plan
        .coverage
        .verdicts
        .values()
        .all(|v| *v == HashVerdict::Backup));
}

// ═════════════════════════════════════════════════════════════════════════════
// I7: no backup store
// ═════════════════════════════════════════════════════════════════════════════

/// I7. With the `Disabled` adapter the Job never calls the backup store it
/// was handed: a lost object holds `NoBackupStore`, and a surviving value
/// missing from Valkey is uncovered with no probe and no wait.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_with_the_disabled_adapter_never_calls_the_backup_store() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    let missing = value(0x71);
    seed_session(&tv, S).await;
    seed_mapping(&tv, S, "lost.usda", 701).await;
    seed_complete(&tv, 702, S, "kept.usda", 10, &[]).await;
    seed_mapping(&tv, S, "kept.usda", 702).await;
    seed_obj_ext(&tv, 702, &[missing]).await;
    let mut config = crate::common::test_config();
    config.checkpoint_adapter = CheckpointAdapter::Disabled;
    hold_with_quiet_left(&tv, &config, Duration::ZERO).await;
    let metrics = metrics();
    let (body, observed) = observing_body();

    let (exit, captured) = run(
        context(
            &tv,
            JobMode::Restore,
            config,
            Arc::clone(&store) as _,
            &metrics,
        ),
        body,
    )
    .await;

    assert_eq!(exit, JobExit::StaysHeld, "the plan stays held");
    let held = lines_starting(&captured, STAYS_HELD_LINE);
    assert_eq!(held.len(), 1, "capture:\n{}", captured.text);
    assert_eq!(held[0]["level"], "ERROR");
    assert_eq!(store.calls(), 0, "no call of the backup store");
    let (plan, report) = the_plan(&observed);
    assert_eq!(
        plan.held,
        vec![HeldItem {
            session_id: S.to_owned(),
            layer_name: "lost.usda".to_owned(),
            object_id: Some(701),
            reason: HeldReason::NoBackupStore,
        }]
    );
    assert_eq!(report.verdicts.get(&missing), Some(&HashVerdict::Uncovered));
    assert_eq!(plan.outcome(&report), PlanOutcome::StaysHeld);
    assert_eq!(plan.epoch.listing_taken_at_ms, None, "nothing is listed");
}

// ═════════════════════════════════════════════════════════════════════════════
// I8: a StorageApi backup store answering 404 for an empty prefix (defer-6(c))
// ═════════════════════════════════════════════════════════════════════════════

/// A StorageApi adapter over the stateful fake, which answers an Enumerate
/// of a prefix holding nothing with 404 as storage-api does, plus the fake's
/// state and server.
async fn storage_api_store(
    name: &str,
) -> (Arc<dyn CheckpointStore>, MockServer, tempfile::TempDir) {
    let server = MockServer::start().await;
    let state: FakeStorageState = Arc::new(Mutex::new(HashMap::new()));
    mount_fake_storage_routes(&server, &state).await;
    let (adapter, token_dir) = build_fake_backed_adapter(&server, name);
    let store: Arc<dyn CheckpointStore> = Arc::new(adapter);
    (store, server, token_dir)
}

/// I8 (i)–(iii). On a StorageApi backup store whose empty `layers/` or
/// `sessions/` prefix lists as 404, a store holding session backups with no
/// layer backups, one holding layer backups with no session backups (their
/// session records surviving), and one holding only the counters record,
/// each plan writable, and the Job passes the frame's checks to the body.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_on_a_storage_api_store_with_an_empty_prefix_plans_writable() {
    within_budget(cases_on_a_storage_api_store_with_an_empty_prefix()).await;
}

/// I8's three cases, one Job run each.
async fn cases_on_a_storage_api_store_with_an_empty_prefix() {
    for case in [
        "sessions without layers",
        "layers without sessions",
        "the counters record only",
    ] {
        let tv = TestValkey::setup().await;
        let (store, _server, _token_dir) = storage_api_store("restore-plan-defer-6c").await;
        put_counters(store.as_ref()).await;
        match case {
            "sessions without layers" => put_session(store.as_ref(), "s-i", &[]).await,
            "layers without sessions" => {
                put_base(store.as_ref(), 801, "s-ii", L, 5, base_ompf(&[])).await;
                seed_session(&tv, "s-ii").await;
                seed_mapping(&tv, "s-ii", L, 801).await;
            }
            _ => {}
        }
        let config = storage_api_config(0);
        hold_with_quiet_left(&tv, &config, Duration::ZERO).await;
        let metrics = metrics();
        let (body, observed) = observing_body();

        let (exit, captured) = run(
            context(&tv, JobMode::Restore, config, store, &metrics),
            body,
        )
        .await;

        assert_eq!(exit, JobExit::Failed, "{case}");
        one(&captured, "INFO", BEFORE_UNHOLD_KILL);
        let (plan, report) = the_plan(&observed);
        assert_eq!(
            plan.outcome(&report),
            PlanOutcome::Writable,
            "{case}: {:?}",
            plan.held
        );
        match case {
            "sessions without layers" => {
                assert!(matches!(
                    session_action(&plan, "s-i"),
                    SessionAction::Restore { step_b: true, .. }
                ));
                assert!(plan.objects.is_empty());
            }
            "layers without sessions" => {
                assert_eq!(restored_head(&plan, 801), 5);
                assert!(matches!(
                    session_action(&plan, "s-ii"),
                    SessionAction::Nothing
                ));
            }
            _ => {
                assert!(plan.objects.is_empty() && plan.sessions.is_empty());
                assert!(plan.epoch.at_or_after_t_safe);
            }
        }
    }
}

/// I8, the denied route. A StorageApi backup store whose Enumerate route
/// answers 404 while its by-address routes work lists empty: before `T_safe`
/// the planner waits for it and lists again (trigger (a)), trusts the empty
/// listing at `T_safe`, and the frame's list-route canary then keeps the
/// service held with exit 3, naming the route.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_whose_listing_route_answers_404_relists_at_t_safe_then_the_canary_holds() {
    let tv = TestValkey::setup().await;
    let (store, server, _token_dir) = storage_api_store("restore-plan-404").await;
    Mock::given(method("GET"))
        .and(path_regex(r"^/v1alpha/fileobject/data-objects/"))
        .respond_with(ResponseTemplate::new(404))
        .with_priority(1)
        .mount(&server)
        .await;
    put_counters(store.as_ref()).await;
    put_base(store.as_ref(), 811, S, L, 5, base_ompf(&[])).await;
    let config = storage_api_config(2);
    hold_with_quiet_left(&tv, &config, Duration::ZERO).await;
    let metrics = metrics();
    let (body, observed) = observing_body();

    let (exit, captured) = run(
        context(&tv, JobMode::Restore, config, store, &metrics),
        body,
    )
    .await;

    assert_eq!(exit, JobExit::StaysHeld);
    one(
        &captured,
        "ERROR",
        &format!("the restore Job exits 3 and the service stays held: {LIST_ROUTE_CAUSE}"),
    );
    assert!(
        waits_for(
            &captured,
            "the backup store listed no layer backups and no session backups before T_safe; \
             listing it again then"
        ) >= 1,
        "the empty listing is re-taken at T_safe:\n{}",
        captured.text
    );
    assert!(observed.lock().unwrap().is_empty(), "the body never runs");
}

// ═════════════════════════════════════════════════════════════════════════════
// I9, I10, I12 and PD-6: the absence rule and the step-down
// ═════════════════════════════════════════════════════════════════════════════

/// The layer 901 of `S`, mapped and lost, with a valid backup at 10 and its
/// newest at 20 of `newest` bytes; `S`'s record survives and its backup is
/// listed, so no identity read reaches the backup store.
async fn lost_layer_with_two_backups(tv: &TestValkey, store: &PlanStore, newest: Bytes) {
    put_session(store, S, &[(L, 901)]).await;
    put_base(store, 901, S, L, 10, base_ompf(&[])).await;
    put_base(store, 901, S, L, 20, newest).await;
    seed_session(tv, S).await;
    seed_mapping(tv, S, L, 901).await;
}

/// I9, the decider condition (fail-on-old). The newest generation reads
/// absent before `T_safe` and is present at it: the planner defers the
/// object, waits for `T_safe` (I12: logging the wait and why), selects again
/// and restores the newest generation, not the older one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_restores_a_generation_absent_before_t_safe_and_present_at_it() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    lost_layer_with_two_backups(&tv, &store, base_ompf(&[])).await;
    let (run, _metrics) = run_on(&tv, &store, T_SAFE_AHEAD).await;
    store.fault_base(901, 20, GetFault::Absent, run.deadlines().t_safe_ms);

    let (plan, captured) = captured(RestorePlanner::plan(&run)).await;
    assert_listed_before_t_safe(&store, &run);
    let plan = plan.expect("the plan");

    assert_eq!(
        restored_head(&plan, 901),
        20,
        "the generation present at T_safe is restored"
    );
    assert!(plan.epoch.relisted_at_t_safe);
    assert!(plan.epoch.at_or_after_t_safe);
    assert!(
        !plan
            .advisories
            .iter()
            .any(|a| matches!(a, Advisory::RestoredFromOlderBackup { .. })),
        "{:?}",
        plan.advisories
    );
    assert_eq!(store.gets_of(901, 20), 2, "read before T_safe, then at it");
    assert!(
        waits_for(
            &captured,
            "the backup of layer plan.usda in session sess-plan (object 901, generation 20) read \
             as absent before T_safe, when the backup store's not-found answers can still be \
             stale; reading it again then"
        ) >= 1,
        "the wait for T_safe is logged with why:\n{}",
        captured.text
    );
    one(
        &captured,
        "INFO",
        "the restore plan saw 1 write-shaping absences before T_safe; it waits for T_safe, lists \
         the backup store again and plans again",
    );
}

/// I10 and PD-6 condition 1 (absence). The newest generation still reads
/// absent at `T_safe`: the step-down stands, the older generation is
/// restored, the WARN names both generations and says the newest needs a
/// file absent from the backup store at `T_safe`, and the summary counts it
/// in the second count.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_steps_down_past_a_generation_absent_at_t_safe_and_says_so() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    lost_layer_with_two_backups(&tv, &store, base_ompf(&[])).await;
    store.fault_base(901, 20, GetFault::Absent, ALWAYS);
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (result, captured) = captured(async {
        let (plan, report) = plan_and_finalize(&run).await?;
        report::log_summary(&plan, &report);
        Ok::<_, JobExit>((plan, report))
    })
    .await;
    let (plan, report) = result.expect("the plan");

    assert_eq!(restored_head(&plan, 901), 10);
    assert!(plan
        .advisories
        .contains(&Advisory::RestoredFromOlderBackup {
            object_id: 901,
            session_id: S.to_owned(),
            layer_name: L.to_owned(),
            restored_seq: 10,
            abandoned: vec![AbandonedGeneration {
                seq: 20,
                reason_label: "stepped_down_link_broken",
                absent: true,
            }],
        }));
    let warn = one(
        &captured,
        "WARN",
        "layer plan.usda of session sess-plan (object 901) is restored from an older backup, \
         generation 10, because its newest backup, generation 20, needs a backup file that is \
         absent from the backup store at T_safe (stepped_down_link_broken)",
    );
    assert_eq!(warn["fields"]["restored_generation"], 10);
    assert_eq!(warn["fields"]["newest_generation"], 20);
    assert_eq!(warn["fields"]["abandoned"], "20 (stepped_down_link_broken)");
    let counts = plan.counts();
    assert_eq!(
        (
            counts.restored_from_older,
            counts.restored_from_older_absent
        ),
        (0, 1)
    );
    let summary = one(
        &captured,
        "INFO",
        &format!(
            "the restore plan: 1 layers to restore, 0 kept, 0 moved to their successor, 0 \
             retirement flags to clear, 0 keeping the service held, 1 sessions to restore, 0 \
             unfinished deletes to finish; outcome: {}; 0 layers restored from an older backup \
             because the newest was unreadable; 1 layers restored from an older backup because \
             the newest needs a backup file absent from the backup store at T_safe; 0 sessions \
             named by surviving data or a backup have no record and no session backup and are \
             not restored",
            plan.outcome(&report).as_str()
        ),
    );
    assert_eq!(summary["fields"]["restored_from_older"], 0);
    assert_eq!(summary["fields"]["restored_from_older_absent"], 1);
}

/// PD-6 condition 1 (bytes). A corrupt newest generation is a bytes verdict:
/// a lost layer is restored from the older generation and a survivor older
/// than the corrupt one is kept, each with a WARN naming both generations
/// and saying the newest was unreadable; the summary counts the restore in
/// the first count.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_steps_down_past_an_unreadable_generation_and_says_so() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    put_session(store.as_ref(), S, &[(L, 911), ("kept.usda", 912)]).await;
    for object_id in [911, 912] {
        put_base(store.as_ref(), object_id, S, L, 10, base_ompf(&[])).await;
        put_base(
            store.as_ref(),
            object_id,
            S,
            L,
            20,
            Bytes::from_static(b"not an OMPF"),
        )
        .await;
    }
    seed_session(&tv, S).await;
    seed_mapping(&tv, S, L, 911).await;
    seed_complete(&tv, 912, S, "kept.usda", 15, &[]).await;
    seed_mapping(&tv, S, "kept.usda", 912).await;
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (plan, captured) = captured(RestorePlanner::plan(&run)).await;
    let plan = plan.expect("the plan");

    assert_eq!(restored_head(&plan, 911), 10);
    assert!(matches!(object_action(&plan, 912), ObjectAction::Keep));
    assert!(plan.held.is_empty(), "{:?}", plan.held);
    let restored = one(
        &captured,
        "WARN",
        "layer plan.usda of session sess-plan (object 911) is restored from an older backup, \
         generation 10, because its newest backup, generation 20, was unreadable \
         (stepped_down_compose)",
    );
    assert_eq!(restored["fields"]["newest_generation"], 20);
    let kept = one(
        &captured,
        "WARN",
        "layer kept.usda of session sess-plan (object 912) keeps its surviving copy at seq 15, \
         because its newer backup, generation 20, was unreadable (stepped_down_compose)",
    );
    assert_eq!(kept["fields"]["kept_seq"], 15);
    let counts = plan.counts();
    assert_eq!(
        (
            counts.restored_from_older,
            counts.restored_from_older_absent
        ),
        (1, 0)
    );
}

/// PD-6 condition 3 (decider pin), end to end. A fetch of the newest
/// generation that fails `Transient` after the adapter's retries is never a
/// step-down: the lost layer is held as unknown and nothing is restored,
/// though an older generation is valid.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_holds_a_layer_whose_newest_fetch_fails_rather_than_stepping_down() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    lost_layer_with_two_backups(&tv, &store, base_ompf(&[])).await;
    store.fault_base(901, 20, GetFault::Transient, ALWAYS);
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (plan, _captured) = captured(RestorePlanner::plan(&run)).await;
    let plan = plan.expect("the plan");

    assert!(matches!(object_action(&plan, 901), ObjectAction::Nothing));
    assert!(matches!(
        plan.held.as_slice(),
        [HeldItem { object_id: Some(901), reason: HeldReason::Unknown { error }, .. }]
            if error.contains("503")
    ));
    assert_eq!(
        store.gets_of(901, 10),
        0,
        "the older generation is never fetched"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// I11, I12: the coverage re-probe at T_safe
// ═════════════════════════════════════════════════════════════════════════════

/// I11, I12 and the .t9 AC's probe rule. With no write-shaping absence the
/// plan probes before `T_safe`. Each value is probed once, plus one re-probe
/// at or after `T_safe` only when its first answer was `false` before it:
/// absent before and present at `T_safe` is covered; absent at `T_safe` too
/// is uncovered; present at once is not probed again; one this run stored
/// itself meanwhile is covered by that PUT with no re-probe; Valkey's is
/// never probed. `finalize` logs the wait.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_coverage_reprobes_a_value_absent_before_t_safe_once_at_it() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    let late = value(0xa1);
    let never = value(0xa2);
    let present = value(0xa3);
    let own = value(0xa4);
    let in_valkey = value(0xa5);
    for hash in [late, present] {
        store
            .put_ext_blob(&sha256_to_hex(&hash), Bytes::from_static(b"held"))
            .await
            .expect("the value put");
    }
    put_base(store.as_ref(), 1001, S, L, 5, base_ompf(&[])).await;
    seed_session(&tv, S).await;
    seed_complete(&tv, 1001, S, L, 10, &[]).await;
    seed_mapping(&tv, S, L, 1001).await;
    seed_obj_ext(&tv, 1001, &[late, never, present, own, in_valkey]).await;
    seed_ext_blob(&tv, &in_valkey).await;
    let (run, metrics) = run_on(&tv, &store, T_SAFE_AHEAD).await;
    let t_safe = run.deadlines().t_safe_ms;
    for hash in [late, never, own] {
        store.absent_until(&hash, t_safe);
    }
    store.absent_until(&never, ALWAYS);

    let (result, captured) = captured(async {
        let plan = RestorePlanner::plan(&run).await?;
        let planned = plan.coverage.verdicts.clone();
        run.own_puts.record(&sha256_to_hex(&own));
        let report = plan.coverage.finalize(&run).await?;
        Ok::<_, JobExit>((plan, planned, report))
    })
    .await;
    assert_probed_before_t_safe(&store, &run);
    let (plan, planned_verdicts, report) = result.expect("the plan");

    assert!(!plan.epoch.relisted_at_t_safe, "no write-shaping absence");
    for hash in [late, never, own] {
        assert_eq!(
            planned_verdicts.get(&hash),
            Some(&HashVerdict::PendingReprobe)
        );
    }
    assert_eq!(planned_verdicts.get(&present), Some(&HashVerdict::Backup));
    assert_eq!(report.verdicts.get(&late), Some(&HashVerdict::Backup));
    assert_eq!(report.verdicts.get(&never), Some(&HashVerdict::Uncovered));
    assert_eq!(report.verdicts.get(&present), Some(&HashVerdict::Backup));
    assert_eq!(report.verdicts.get(&own), Some(&HashVerdict::OwnPut));
    assert_eq!(report.verdicts.get(&in_valkey), Some(&HashVerdict::Valkey));
    for (name, hash, calls) in [
        ("absent, then present", late, 2),
        ("absent twice", never, 2),
        ("present at once", present, 1),
        ("stored by this run meanwhile", own, 1),
        ("in Valkey", in_valkey, 0),
    ] {
        assert_eq!(store.probes_of(&hash), calls, "{name}");
    }
    assert!(
        waits_for(
            &captured,
            "3 referenced values read as absent from the backup store before T_safe; checking \
             them again then"
        ) >= 1,
        "the wait is logged:\n{}",
        captured.text
    );
    assert_eq!(probe_cell(&metrics, "present"), 2.0);
    assert_eq!(probe_cell(&metrics, "absent"), 4.0);
    assert_eq!(probe_cell(&metrics, "error"), 0.0);
    assert_eq!(uncovered_gauge(&metrics), 1.0);
    assert_eq!(plan.outcome(&report), PlanOutcome::StaysHeld);
}

// ═════════════════════════════════════════════════════════════════════════════
// I13 and PM r2 S7: the session-listing cross-check (PD-1)
// ═════════════════════════════════════════════════════════════════════════════

/// The S7 ERROR naming `evidence` and `session`.
fn s7_error(evidence: &str, session: &str) -> String {
    format!(
        "the restore Job wrote nothing and exits 3; the service stays held: the backup store \
         lists no session backups, but {evidence} names session {session}, which has no record \
         in the store and no deletion marker; the backup store's identity may lack permission to \
         list .live-edit/sessions/ under the backup prefix"
    )
}

/// I13. A layer backup names session `s-gone`, which has no record and no
/// deletion marker, and the session listing answers empty though a session
/// backup exists (a denied listing reads empty). Before `T_safe` that is a
/// trigger, re-checked at `T_safe`; there the Job writes nothing, exits 3
/// and names the listing permission.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_holds_when_a_backup_names_a_session_the_denied_session_listing_omits() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    put_base(store.as_ref(), 1101, "s-gone", L, 5, base_ompf(&[])).await;
    put_session(store.as_ref(), "s-other", &[]).await;
    put_counters(store.as_ref()).await;
    *store.sessions_denied.lock().unwrap() = true;
    let config = storage_api_config(2);
    hold_with_quiet_left(&tv, &config, Duration::ZERO).await;
    let store_before = keyspace(&tv).await;
    let writes_before = store.writes.load(Ordering::SeqCst);
    let metrics = metrics();
    let (body, observed) = observing_body();

    let (exit, captured) = run(
        context(
            &tv,
            JobMode::Restore,
            config,
            Arc::clone(&store) as _,
            &metrics,
        ),
        body,
    )
    .await;

    assert_eq!(exit, JobExit::StaysHeld);
    assert_eq!(exit.code(), 3);
    let evidence = "the listed backup of object 1101 (layer plan.usda)";
    one(&captured, "ERROR", &s7_error(evidence, "s-gone"));
    assert!(
        waits_for(
            &captured,
            &format!(
                "session s-gone, named by {evidence}, has no session backup listed before \
                 T_safe; listing again then"
            )
        ) >= 1,
        "the cross-check is re-taken at T_safe:\n{}",
        captured.text
    );
    assert!(
        observed.lock().unwrap().is_empty(),
        "nothing is planned to write"
    );
    let writes = keyspace_writes(&store_before, &keyspace(&tv).await, WriteRules::STRICT);
    assert!(
        writes.is_empty(),
        "nothing is written to the store: {writes:?}"
    );
    assert_eq!(
        store.writes.load(Ordering::SeqCst),
        writes_before,
        "nothing is written to the backup store"
    );
}

/// I13's contrast. The same backups with the session listing working: it
/// lists `s-other`, which proves the route and the permission, so the plan
/// completes, reporting the backup no mapping names rather than holding.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_under_a_non_empty_session_listing_reports_rather_than_holds() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    put_base(store.as_ref(), 1101, "s-gone", L, 5, base_ompf(&[])).await;
    put_session(store.as_ref(), "s-other", &[]).await;
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (plan, captured) = captured(RestorePlanner::plan(&run)).await;
    let plan = plan.expect("the plan completes");

    assert!(plan
        .advisories
        .contains(&Advisory::UnreferencedBackup { object_id: 1101 }));
    one(
        &captured,
        "INFO",
        "1 listed layer backups are named by no layer mapping after the plan and are not restored",
    );
    assert!(lines_starting(&captured, "the restore Job wrote nothing and exits 3").is_empty());
    assert_eq!(
        store.gets_of(1101, 5),
        0,
        "no identity read under a non-empty listing"
    );
}

/// PM r2 S7, through a StorageApi backup store: a layer backup names a
/// session, and the session listing is denied with a 404 though a session
/// backup is stored. The run stays held, names the cause and writes nothing,
/// rather than completing as if no session existed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_holds_when_the_storage_api_session_listing_is_denied_with_a_404() {
    let tv = TestValkey::setup().await;
    let (store, server, _token_dir) = storage_api_store("restore-plan-s7").await;
    Mock::given(method("GET"))
        .and(path_regex(r"^/v1alpha/fileobject/data-objects/.*sessions"))
        .respond_with(ResponseTemplate::new(404))
        .with_priority(1)
        .mount(&server)
        .await;
    put_counters(store.as_ref()).await;
    put_base(store.as_ref(), 1201, "s-gone", L, 5, base_ompf(&[])).await;
    put_session(store.as_ref(), "s-other", &[]).await;
    assert!(
        store
            .get_session_checkpoint("s-other")
            .await
            .expect("by-address reads work")
            .is_some(),
        "fixture: a session backup is stored"
    );
    let config = storage_api_config(0);
    hold_with_quiet_left(&tv, &config, Duration::ZERO).await;
    let store_before = keyspace(&tv).await;
    let metrics = metrics();
    let (body, observed) = observing_body();

    let (exit, captured) = run(
        context(&tv, JobMode::Restore, config, store, &metrics),
        body,
    )
    .await;

    assert_eq!(exit, JobExit::StaysHeld);
    one(
        &captured,
        "ERROR",
        &s7_error(
            "the listed backup of object 1201 (layer plan.usda)",
            "s-gone",
        ),
    );
    assert!(observed.lock().unwrap().is_empty());
    let writes = keyspace_writes(&store_before, &keyspace(&tv).await, WriteRules::STRICT);
    assert!(
        writes.is_empty(),
        "nothing is written to the store: {writes:?}"
    );
}

/// PD-1 loudness. Under a non-empty session listing, a session a surviving
/// header names with a listed backup is reported with its layers, and one a
/// surviving mapping names with no listed backup is reported without them
/// (no `layers` field).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_reports_a_session_without_a_record_with_and_without_its_layers() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    put_session(store.as_ref(), "s-other", &[]).await;
    put_base(store.as_ref(), 1121, "s-pd1", L, 5, base_ompf(&[])).await;
    seed_complete(&tv, 1121, "s-pd1", L, 10, &[]).await;
    seed_mapping(&tv, "s-pd2", "m.usda", 1122).await;
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (plan, captured) = captured(RestorePlanner::plan(&run)).await;
    let plan = plan.expect("the plan");

    let with_layers = one(
        &captured,
        "WARN",
        "session s-pd1, named by the surviving header of object 1121 (layer plan.usda), has no \
         record in the store, no deletion marker and no session backup; it is not restored, and \
         the backups of its layers (plan.usda) are left unreferenced",
    );
    assert_eq!(with_layers["fields"]["layers"], "plan.usda");
    let without = one(
        &captured,
        "WARN",
        "session s-pd2, named by the surviving mapping of layer m.usda (object 1122), has no \
         record in the store, no deletion marker and no session backup; it is not restored",
    );
    assert!(without["fields"].get("layers").is_none(), "{without}");
    assert!(plan.held.is_empty(), "{:?}", plan.held);
    assert_eq!(plan.counts().sessions_without_backup, 2);
}

// ═════════════════════════════════════════════════════════════════════════════
// I14, I15: defer-39 — every plan read is taken at or after Q's end
// ═════════════════════════════════════════════════════════════════════════════

/// Wait until the store's clock reaches `target_ms`.
async fn until_store_time(tv: &TestValkey, target_ms: u64) {
    let deadline = tokio::time::Instant::now() + RUN_BUDGET;
    while now_ms(tv).await < target_ms {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the store's clock never reached {target_ms}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// I14, defer-39 on the backup store (fail-on-old). A newer generation of a
/// lost layer lands in the backup store during the quiet period, after a
/// planner reading at Q's start would have listed: the plan, which reads
/// nothing before Q's end, restores the newer generation, and its listing is
/// taken at or after Q's end on the store's clock.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_sees_a_backup_written_during_the_quiet_period() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    put_session(store.as_ref(), S, &[(L, 1301)]).await;
    put_base(store.as_ref(), 1301, S, L, 10, base_ompf(&[])).await;
    put_counters(store.as_ref()).await;
    seed_session(&tv, S).await;
    seed_mapping(&tv, S, L, 1301).await;
    let config = storage_api_config(0);
    let quiet_ends = hold_with_quiet_left(&tv, &config, QUIET_LEFT).await;
    let metrics = metrics();
    let (body, observed) = observing_body();
    let job = tokio::spawn(run(
        context(
            &tv,
            JobMode::Restore,
            config,
            Arc::clone(&store) as _,
            &metrics,
        ),
        body,
    ));

    until_store_time(&tv, quiet_ends - LATE_WRITE_BEFORE_QUIET_END_MS).await;
    put_base(store.as_ref(), 1301, S, L, 20, base_ompf(&[])).await;
    let (exit, captured) = job.await.expect("the Job task");

    assert_eq!(exit, JobExit::Failed, "stopped before its un-hold");
    let (plan, _report) = the_plan(&observed);
    assert_eq!(
        restored_head(&plan, 1301),
        20,
        "the generation written during Q is restored"
    );
    let frame_quiet_ends = one(
        &captured,
        "INFO",
        "the store is held under this incident; the restore Job writes nothing until the store's \
         clock passes the quiet period",
    )["fields"]["quiet_ends_at_ms"]
        .as_u64()
        .expect("the frame names Q's end");
    let listed_at = store
        .first_listing_at_ms
        .lock()
        .unwrap()
        .expect("the backup store was listed");
    assert!(
        listed_at >= frame_quiet_ends,
        "the first listing was taken at {listed_at} ms, before Q ended at {frame_quiet_ends} ms"
    );
}

/// I15, defer-39 on Valkey (fail-on-old). A surviving layer's `seq` rises
/// above its backup's head during the quiet period: the plan, which reads
/// the store only after Q's end, keeps the survivor rather than restoring
/// the older backup over it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_sees_a_survivor_advanced_during_the_quiet_period() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    put_session(store.as_ref(), S, &[(L, 1302)]).await;
    put_base(store.as_ref(), 1302, S, L, 10, base_ompf(&[])).await;
    put_counters(store.as_ref()).await;
    seed_session(&tv, S).await;
    seed_complete(&tv, 1302, S, L, 5, &[]).await;
    seed_mapping(&tv, S, L, 1302).await;
    let config = storage_api_config(0);
    let quiet_ends = hold_with_quiet_left(&tv, &config, QUIET_LEFT).await;
    let metrics = metrics();
    let (body, observed) = observing_body();
    let job = tokio::spawn(run(
        context(
            &tv,
            JobMode::Restore,
            config,
            Arc::clone(&store) as _,
            &metrics,
        ),
        body,
    ));

    until_store_time(&tv, quiet_ends - LATE_WRITE_BEFORE_QUIET_END_MS).await;
    seed_header(&tv, 1302, &[("seq", "20")]).await;
    let (exit, _captured) = job.await.expect("the Job task");

    assert_eq!(exit, JobExit::Failed, "stopped before its un-hold");
    let (plan, _report) = the_plan(&observed);
    assert!(
        matches!(object_action(&plan, 1302), ObjectAction::Keep),
        "the survivor advanced during Q is kept: {:?}",
        object_action(&plan, 1302)
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// I16, I17: plan-level failures
// ═════════════════════════════════════════════════════════════════════════════

/// I16. A stamped survivor the listing does not show contradicts it: at
/// `T_safe` the Job writes nothing and exits 3, naming the contradiction and
/// what to check.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_holds_on_a_listing_that_contradicts_the_store() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    put_counters(store.as_ref()).await;
    put_base(store.as_ref(), 1402, S, "other.usda", 5, base_ompf(&[])).await;
    seed_session(&tv, S).await;
    seed_complete(&tv, 1401, S, L, 10, &[("last_checkpoint_seq", "7")]).await;
    seed_mapping(&tv, S, L, 1401).await;
    let config = storage_api_config(0);
    hold_with_quiet_left(&tv, &config, Duration::ZERO).await;
    let store_before = keyspace(&tv).await;
    let metrics = metrics();
    let (body, observed) = observing_body();

    let (exit, captured) = run(
        context(
            &tv,
            JobMode::Restore,
            config,
            Arc::clone(&store) as _,
            &metrics,
        ),
        body,
    )
    .await;

    assert_eq!(exit, JobExit::StaysHeld);
    let contradiction = Contradiction::Layer(CheckpointStamp {
        object_id: 1401,
        session_id: S.to_owned(),
        layer_name: L.to_owned(),
        last_checkpoint_seq: Some(7),
        last_checkpoint_ms: None,
    });
    one(
        &captured,
        "ERROR",
        &format!(
            "the restore Job wrote nothing and exits 3; the service stays held: the backup \
             store's listing contradicts the store: {contradiction}; check the backup store's \
             address, credentials and listing permissions"
        ),
    );
    assert!(observed.lock().unwrap().is_empty());
    let writes = keyspace_writes(&store_before, &keyspace(&tv).await, WriteRules::STRICT);
    assert!(
        writes.is_empty(),
        "nothing is written to the store: {writes:?}"
    );
}

/// I17, the breaker (PD-5). 40 values whose every probe fails: the stage
/// stops after the 32nd consecutive error and the plan exits 1 with its
/// ERROR, the error cell counting exactly those 32.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn restore_coverage_stops_after_32_probe_errors_in_a_row_and_exits_1() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    let hashes: Vec<[u8; 32]> = (0..40u8).map(|i| value(0xc0 + i)).collect();
    for hash in &hashes {
        store.failing(hash);
    }
    seed_session(&tv, S).await;
    seed_complete(&tv, 1501, S, L, 10, &[]).await;
    seed_mapping(&tv, S, L, 1501).await;
    seed_obj_ext(&tv, 1501, &hashes).await;
    let (run, metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (plan, captured) = captured(RestorePlanner::plan(&run)).await;

    assert_eq!(plan.err(), Some(JobExit::Failed));
    let breaker = one(
        &captured,
        "ERROR",
        "the backup store failed 32 existence checks in a row; the restore Job stops checking, \
         writes nothing, and exits 1 so Kubernetes retries it",
    );
    assert_eq!(breaker["fields"]["consecutive_errors"], 32);
    assert_eq!(probe_cell(&metrics, "error"), 32.0);
}

/// I17's other side: one failing value among values the backup store holds
/// is unknown, and the plan completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_coverage_holds_one_failing_value_as_unknown_without_tripping_the_breaker() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    let (held, failing, also_held) = (value(0xd1), value(0xd2), value(0xd3));
    for hash in [held, also_held] {
        store
            .put_ext_blob(&sha256_to_hex(&hash), Bytes::from_static(b"held"))
            .await
            .expect("the value put");
    }
    store.failing(&failing);
    seed_session(&tv, S).await;
    seed_complete(&tv, 1502, S, L, 10, &[]).await;
    seed_mapping(&tv, S, L, 1502).await;
    seed_obj_ext(&tv, 1502, &[held, failing, also_held]).await;
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (plan, _captured) = captured(RestorePlanner::plan(&run)).await;
    let plan = plan.expect("the plan completes");

    assert!(matches!(
        plan.coverage.verdicts.get(&failing),
        Some(HashVerdict::Unknown(_))
    ));
    assert_eq!(
        plan.coverage.verdicts.get(&held),
        Some(&HashVerdict::Backup)
    );
    assert_eq!(
        plan.coverage.verdicts.get(&also_held),
        Some(&HashVerdict::Backup)
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// PD-1 identity reads: gate w8h2 (`saw_absent`) and the code-review pins
// ═════════════════════════════════════════════════════════════════════════════

/// Gate w8h2 (fail-on-old). Under an empty session listing, object 1701's
/// newest generation reads absent before `T_safe` and is present at it,
/// naming session `s9`; its older generation fails `Permanent`. A
/// pre-`T_safe` trigger elsewhere, (b), makes the planner re-take at
/// `T_safe` (1701's own absence is trigger (g) too; the test with (g) alone
/// is `restore_plan_retakes_at_t_safe_for_an_identity_absence_alone_and_identifies_the_object`),
/// and the re-take reads 1701 again because its earlier read saw an absence: it
/// is identified, its session judged (a deletion marker: an abandoned
/// delete), and it is not held as unknown. Object 1702, whose only
/// generation fails `Permanent`, is read once across both passes and logged
/// once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_reads_an_unidentified_object_again_at_t_safe_after_an_absence() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    put_base(store.as_ref(), 1701, "s9", L, 10, base_ompf(&[])).await;
    put_base(store.as_ref(), 1701, "s9", L, 20, base_ompf(&[])).await;
    put_base(store.as_ref(), 1702, "s10", L, 30, base_ompf(&[])).await;
    seed_deletion_marker(&tv, "s9").await;
    // Trigger (b): a mapped layer with no header and no listed backup.
    seed_session(&tv, "s-other").await;
    seed_mapping(&tv, "s-other", "x.usda", 1750).await;
    let (run, _metrics) = run_on(&tv, &store, T_SAFE_AHEAD).await;
    store.fault_base(1701, 20, GetFault::Absent, run.deadlines().t_safe_ms);
    store.fault_base(1701, 10, GetFault::Permanent, ALWAYS);
    store.fault_base(1702, 30, GetFault::Permanent, ALWAYS);

    let (plan, captured) = captured(RestorePlanner::plan(&run)).await;
    assert_listed_before_t_safe(&store, &run);
    let plan = plan.expect("the plan");

    assert!(plan.epoch.relisted_at_t_safe, "the T_safe re-take ran");
    assert!(
        !plan.held.iter().any(|h| h.object_id == Some(1701)),
        "1701 is identified at T_safe, not held unknown: {:?}",
        plan.held
    );
    assert!(matches!(
        session_action(&plan, "s9"),
        SessionAction::CompleteAbandonedDelete { .. }
    ));
    assert!(plan
        .advisories
        .contains(&Advisory::UnreferencedBackup { object_id: 1701 }));
    assert_eq!(store.gets_of(1701, 20), 2, "read before T_safe, then at it");
    assert!(matches!(
        plan.held.iter().find(|h| h.object_id == Some(1702)),
        Some(HeldItem { reason: HeldReason::Unknown { error }, .. }) if error.contains("generation 30")
    ));
    assert_eq!(
        store.gets_of(1702, 30),
        1,
        "an all-Permanent object is read once"
    );
    assert_eq!(
        lines_starting(
            &captured,
            "the restore Job could not read backup generation 30 of object 1702 to learn its \
             session"
        )
        .len(),
        1,
        "and logged once"
    );
}

/// The `identify_listed_objects` read itself (code review r4): an object
/// whose generations read absent (or lack their meta) and then fail
/// `Permanent` is recorded unidentified with `saw_absent` set, and returned
/// among the absences (trigger (g)'s input) with its newest absent
/// generation; one that only fails `Permanent` is recorded without it and not
/// returned. A second read over the same listing reads the first two again,
/// returns their absences again, and skips the third. A `Transient` read
/// ends the plan with exit 1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn identify_listed_objects_records_each_unidentified_object_and_whether_it_saw_an_absence() {
    within_budget(identify_listed_objects_three_times()).await;
}

/// The test above's three identity reads.
async fn identify_listed_objects_three_times() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    for (object_id, seqs) in [(1801u64, [10u64, 20]), (1802, [25, 30]), (1803, [35, 40])] {
        for seq in seqs {
            put_base(store.as_ref(), object_id, "s18", L, seq, base_ompf(&[])).await;
        }
    }
    store.fault_base(1801, 20, GetFault::Absent, ALWAYS);
    store.fault_base(1801, 10, GetFault::Permanent, ALWAYS);
    store.fault_base(1802, 30, GetFault::MissingMeta, ALWAYS);
    store.fault_base(1802, 25, GetFault::Permanent, ALWAYS);
    store.fault_base(1803, 40, GetFault::Permanent, ALWAYS);
    store.fault_base(1803, 35, GetFault::Permanent, ALWAYS);
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;
    let listing = Listing {
        generations: live_edit::checkpoint::lineage::group_layer_generations(
            &store
                .inner
                .list_layer_artifacts()
                .await
                .expect("the listing"),
        ),
        at_or_after_t_safe: false,
        ..Listing::default()
    };
    let mut facts = StoreFacts::default();
    let mut known = BTreeMap::new();
    let mut unidentified: BTreeMap<u64, UnidentifiedObject> = BTreeMap::new();

    let absences = snapshot::identify_listed_objects(
        &run,
        store.as_ref(),
        &listing,
        &mut facts,
        &mut known,
        &mut unidentified,
    )
    .await
    .expect("a Permanent identity read is no plan failure");

    assert_eq!(
        absences,
        BTreeMap::from([(1801, 20), (1802, 30)]),
        "each object that saw an absence, with its newest absent generation; not the \
         all-Permanent one"
    );
    assert!(known.is_empty());
    let recorded = |object_id: u64| unidentified.get(&object_id).cloned().expect("recorded");
    for (object_id, listed, seq, saw_absent) in [
        (1801, vec![20, 10], 10, true),
        (1802, vec![30, 25], 25, true),
        (1803, vec![40, 35], 40, false),
    ] {
        let u = recorded(object_id);
        assert_eq!(
            (u.listed.clone(), u.seq, u.saw_absent),
            (listed, seq, saw_absent),
            "object {object_id}"
        );
        assert!(u.error.contains("403"), "object {object_id}: {}", u.error);
    }
    let gets_before: Vec<usize> = [1801, 1802, 1803]
        .iter()
        .map(|o| {
            store
                .gets
                .lock()
                .unwrap()
                .iter()
                .filter(|g| g.0 == *o)
                .count()
        })
        .collect();

    let absences = snapshot::identify_listed_objects(
        &run,
        store.as_ref(),
        &listing,
        &mut facts,
        &mut known,
        &mut unidentified,
    )
    .await
    .expect("the second read");

    assert_eq!(
        absences,
        BTreeMap::from([(1801, 20), (1802, 30)]),
        "the second read records the absences it saw itself"
    );
    let gets_after: Vec<usize> = [1801, 1802, 1803]
        .iter()
        .map(|o| {
            store
                .gets
                .lock()
                .unwrap()
                .iter()
                .filter(|g| g.0 == *o)
                .count()
        })
        .collect();
    assert_eq!(
        gets_after,
        vec![gets_before[0] + 2, gets_before[1] + 2, gets_before[2]],
        "after an absence the object is read again; an all-Permanent one is not"
    );

    store.fault_base(1803, 40, GetFault::Transient, ALWAYS);
    let fresh = Listing {
        generations: listing.generations.clone(),
        ..Listing::default()
    };
    let result = snapshot::identify_listed_objects(
        &run,
        store.as_ref(),
        &fresh,
        &mut StoreFacts::default(),
        &mut BTreeMap::new(),
        &mut BTreeMap::new(),
    )
    .await;
    assert_eq!(
        result,
        Err(JobExit::Failed),
        "a Transient identity read exits 1"
    );
}

/// Code-review pin: after a wipe, under an empty session listing, a listed
/// object whose every generation fails `Permanent` (a 403) holds the service
/// as unknown, naming the object, and is not reported unreferenced.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_holds_a_listed_object_whose_identity_cannot_be_read() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    put_base(store.as_ref(), 1911, "s19", L, 30, base_ompf(&[])).await;
    store.fault_base(1911, 30, GetFault::Permanent, ALWAYS);
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (result, captured) = captured(plan_and_finalize(&run)).await;
    let (plan, report) = result.expect("the plan");

    let reason = HeldReason::Unknown {
        error: "no listed backup of it could be read to learn its session and layer (generation \
                30: permanent checkpoint-store error: the backup store answered 403)"
            .to_owned(),
    };
    assert_eq!(
        plan.held,
        vec![HeldItem {
            session_id: String::new(),
            layer_name: String::new(),
            object_id: Some(1911),
            reason: reason.clone(),
        }]
    );
    assert!(!plan
        .advisories
        .contains(&Advisory::UnreferencedBackup { object_id: 1911 }));
    assert_eq!(plan.outcome(&report), PlanOutcome::StaysHeld);
    one(
        &captured,
        "WARN",
        &format!(
            "object 1911, listed in the backup store, keeps the service held: {}",
            reason.describe()
        ),
    );
}

/// Code-review pin: a newest generation failing `Permanent` above an older
/// one that reads identifies the object from the older one, with no hold; a
/// `Transient` instead ends the plan with exit 1, naming the identity read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_identifies_past_a_permanent_newest_generation_and_exits_1_on_a_transient() {
    within_budget(identifies_past_a_permanent_newest_generation_then_a_transient()).await;
}

/// The test above's two plans.
async fn identifies_past_a_permanent_newest_generation_then_a_transient() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    put_base(store.as_ref(), 1921, "s21", L, 20, base_ompf(&[])).await;
    put_base(store.as_ref(), 1921, "s21", L, 30, base_ompf(&[])).await;
    seed_session(&tv, "s21").await;
    store.fault_base(1921, 30, GetFault::Permanent, ALWAYS);
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (plan, _captured) = captured(RestorePlanner::plan(&run)).await;
    let plan = plan.expect("a Permanent newest generation is no plan failure");

    assert!(
        plan.held.is_empty(),
        "identified, not held: {:?}",
        plan.held
    );
    assert!(matches!(
        session_action(&plan, "s21"),
        SessionAction::Nothing
    ));
    assert!(plan
        .advisories
        .contains(&Advisory::UnreferencedBackup { object_id: 1921 }));

    store.fault_base(1921, 30, GetFault::Transient, ALWAYS);
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;
    let (plan, captured) = captured(RestorePlanner::plan(&run)).await;
    assert_eq!(plan.err(), Some(JobExit::Failed));
    one(
        &captured,
        "ERROR",
        "the restore Job could not read the backup store to plan the restore (a listed backup's \
         identity), wrote nothing and exits 1",
    );
}

/// Code-review pin (rules 2 and 4): a surviving mapping to a deleted object
/// keeps an older session-backup mapping of the same layer out, so the older
/// object is never brought back; the deleted survivor holds.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_keeps_an_older_backup_mapping_out_behind_a_deleted_survivor() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    put_session(store.as_ref(), S, &[(L, 1903)]).await;
    put_base(store.as_ref(), 1903, S, L, 5, base_ompf(&[])).await;
    seed_session(&tv, S).await;
    seed_mapping(&tv, S, L, 1905).await;
    seed_dead(&tv, 1905).await;
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (plan, _captured) = captured(RestorePlanner::plan(&run)).await;
    let plan = plan.expect("the plan");

    match session_action(&plan, S) {
        SessionAction::Restore { checkpoint, step_b } => {
            assert!(
                checkpoint.layer_mappings.is_empty(),
                "{:?}",
                checkpoint.layer_mappings
            );
            assert!(!step_b);
        }
        other => panic!("expected a session restore, got {other:?}"),
    }
    assert!(planned(&plan).is_empty(), "{:?}", plan.objects);
    assert!(plan
        .advisories
        .contains(&Advisory::UnreferencedBackup { object_id: 1903 }));
    assert_eq!(
        plan.held,
        vec![HeldItem {
            session_id: S.to_owned(),
            layer_name: L.to_owned(),
            object_id: Some(1905),
            reason: HeldReason::MappedToDeletedObject {
                session_id: S.to_owned(),
                layer_name: L.to_owned(),
            },
        }]
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// defer-52: trigger (g), an identity read's absence before T_safe (INV-1)
// ═════════════════════════════════════════════════════════════════════════════

/// Trigger (g)'s `why`: the identity read of `object_id` saw generation `seq`
/// as absent before `T_safe`.
fn identity_absence_why(object_id: u64, seq: u64) -> String {
    format!(
        "the backup of object {object_id} (generation {seq}) read as absent before T_safe while \
         the restore was learning which session it belongs to, when the backup store's \
         not-found answers can still be stale; reading it again then"
    )
}

/// The INFO the planner logs when one trigger makes it re-take at `T_safe`.
const ONE_TRIGGER: &str = "the restore plan saw 1 write-shaping absences before T_safe; it waits \
                           for T_safe, lists the backup store again and plans again";

/// Object `object_id` listed after a wipe in `session` at generations 10 and
/// 20; nothing in the store names it.
async fn listed_only(store: &PlanStore, object_id: u64, session: &str) {
    put_base(store, object_id, session, L, 10, base_ompf(&[])).await;
    put_base(store, object_id, session, L, 20, base_ompf(&[])).await;
}

/// defer-52, decider-required (fail-on-old). After a wipe (no header, no
/// mapping), under an empty session listing, the only listed object, 2001,
/// reads absent before `T_safe` on both its generations (the newest
/// `Ok(None)`, the older one missing its meta) and is present at `T_safe`,
/// naming session `s-hidden`, which has no record and no deletion marker. No
/// other trigger fires. The identity read's absence is trigger (g): the
/// planner waits for `T_safe`, logging (g)'s why, lists again, reads 2001's
/// newest generation again, learns `s-hidden` and ends S7 (exit 3), naming
/// the session and the backup and writing nothing. Without (g), 2001 is an
/// unreferenced backup and the plan completes, leaving `s-hidden` silently
/// unrestored.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_retakes_at_t_safe_for_an_identity_absence_and_holds_on_the_session_it_hid() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    listed_only(&store, 2001, "s-hidden").await;
    let (run, _metrics) = run_on(&tv, &store, T_SAFE_AHEAD).await;
    let t_safe_ms = run.deadlines().t_safe_ms;
    store.fault_base(2001, 20, GetFault::Absent, t_safe_ms);
    store.fault_base(2001, 10, GetFault::MissingMeta, t_safe_ms);
    let store_before = keyspace(&tv).await;
    let writes_before = store.writes.load(Ordering::SeqCst);

    let (result, captured) = captured(RestorePlanner::plan(&run)).await;
    assert_listed_before_t_safe(&store, &run);

    let exit = match result {
        Ok(plan) => panic!(
            "S7 ends the plan held; it completed with advisories {:?} and held {:?}",
            plan.advisories, plan.held
        ),
        Err(exit) => exit,
    };
    assert_eq!(exit, JobExit::StaysHeld);
    assert_eq!(exit.code(), 3);
    one(
        &captured,
        "ERROR",
        &s7_error(
            "the listed backup of object 2001 (layer plan.usda)",
            "s-hidden",
        ),
    );
    one(&captured, "INFO", ONE_TRIGGER);
    assert!(
        waits_for(&captured, &identity_absence_why(2001, 20)) >= 1,
        "the wait for T_safe is logged with (g)'s why:\n{}",
        captured.text
    );
    assert_eq!(
        store.listings.load(Ordering::SeqCst),
        2,
        "listed before T_safe, then again at it"
    );
    assert_eq!(store.gets_of(2001, 20), 2, "read before T_safe, then at it");
    assert_eq!(
        store.gets_of(2001, 10),
        1,
        "at T_safe the newest generation identifies 2001"
    );
    let writes = keyspace_writes(&store_before, &keyspace(&tv).await, WriteRules::STRICT);
    assert!(
        writes.is_empty(),
        "nothing is written to the store: {writes:?}"
    );
    assert_eq!(
        store.writes.load(Ordering::SeqCst),
        writes_before,
        "nothing is written to the backup store"
    );
}

/// defer-52 at `T_safe`. The same wipe and empty session listing, with 2011's
/// generations absent at `T_safe` too and the first listing taken after
/// `T_safe`: an absence at `T_safe` stands, so there is no trigger (g), no
/// wait and no re-take, and with no session learnt there is no S7. 2011 is
/// reported as an unreferenced backup and the plan completes writable,
/// reading each generation once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_takes_an_identity_absence_at_t_safe_as_final_and_completes() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    listed_only(&store, 2011, "s-hidden").await;
    store.fault_base(2011, 20, GetFault::Absent, ALWAYS);
    store.fault_base(2011, 10, GetFault::MissingMeta, ALWAYS);
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (result, captured) = captured(plan_and_finalize(&run)).await;
    let (plan, report) = result.expect("the plan completes");

    assert!(plan.epoch.at_or_after_t_safe);
    assert!(!plan.epoch.relisted_at_t_safe);
    assert_eq!(store.listings.load(Ordering::SeqCst), 1);
    assert!(
        lines_starting(&captured, "the restore plan saw ").is_empty(),
        "no trigger:\n{}",
        captured.text
    );
    assert_eq!(waits_for(&captured, &identity_absence_why(2011, 20)), 0);
    assert!(lines_starting(&captured, "the restore Job wrote nothing and exits 3").is_empty());
    assert!(plan
        .advisories
        .contains(&Advisory::UnreferencedBackup { object_id: 2011 }));
    assert!(plan.held.is_empty(), "{:?}", plan.held);
    assert!(
        plan.sessions.iter().all(|p| p.session_id != "s-hidden"),
        "{:?}",
        plan.sessions
    );
    assert_eq!(plan.outcome(&report), PlanOutcome::Writable);
    assert_eq!(
        (store.gets_of(2011, 20), store.gets_of(2011, 10)),
        (1, 1),
        "each generation is read once"
    );
}

/// INV-1's "once". 2021's generations read absent before `T_safe` and still
/// at it: (g) makes the planner wait for `T_safe` and re-take once; at
/// `T_safe` the absence stands, so 2021 is an unreferenced backup and the
/// plan completes writable. Each generation is read once per pass.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_reobserves_an_identity_absence_once_at_t_safe_then_takes_it_as_final() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    listed_only(&store, 2021, "s-hidden").await;
    store.fault_base(2021, 20, GetFault::Absent, ALWAYS);
    store.fault_base(2021, 10, GetFault::MissingMeta, ALWAYS);
    let (run, _metrics) = run_on(&tv, &store, T_SAFE_AHEAD).await;

    let (result, captured) = captured(plan_and_finalize(&run)).await;
    assert_listed_before_t_safe(&store, &run);
    let (plan, report) = result.expect("the plan completes");

    assert!(plan.epoch.relisted_at_t_safe);
    assert!(plan.epoch.at_or_after_t_safe);
    assert_eq!(store.listings.load(Ordering::SeqCst), 2);
    one(&captured, "INFO", ONE_TRIGGER);
    assert!(
        waits_for(&captured, &identity_absence_why(2021, 20)) >= 1,
        "{}",
        captured.text
    );
    assert!(lines_starting(&captured, "the restore Job wrote nothing and exits 3").is_empty());
    assert!(plan
        .advisories
        .contains(&Advisory::UnreferencedBackup { object_id: 2021 }));
    assert!(plan.held.is_empty(), "{:?}", plan.held);
    assert_eq!(plan.outcome(&report), PlanOutcome::Writable);
    assert_eq!(
        (store.gets_of(2021, 20), store.gets_of(2021, 10)),
        (2, 2),
        "each generation is read before T_safe, then once at it"
    );
}

/// Gate w8h2 with trigger (g) alone (defer-52). Under an empty session
/// listing, with no other trigger, 2031's newest generation reads absent
/// before `T_safe` and is present at it, naming session `s31` (which has a
/// deletion marker); its older generation fails `Permanent`. (g) makes the
/// planner re-take at `T_safe`, the re-take reads 2031 again because its
/// earlier read saw an absence, and 2031 is identified: its session is
/// judged an abandoned delete, and 2031 is not held as unknown.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_retakes_at_t_safe_for_an_identity_absence_alone_and_identifies_the_object() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    listed_only(&store, 2031, "s31").await;
    seed_deletion_marker(&tv, "s31").await;
    let (run, _metrics) = run_on(&tv, &store, T_SAFE_AHEAD).await;
    store.fault_base(2031, 20, GetFault::Absent, run.deadlines().t_safe_ms);
    store.fault_base(2031, 10, GetFault::Permanent, ALWAYS);

    let (plan, captured) = captured(RestorePlanner::plan(&run)).await;
    assert_listed_before_t_safe(&store, &run);
    let plan = plan.expect("the plan");

    assert!(plan.epoch.relisted_at_t_safe, "the T_safe re-take ran");
    one(&captured, "INFO", ONE_TRIGGER);
    assert!(
        waits_for(&captured, &identity_absence_why(2031, 20)) >= 1,
        "(g) is the trigger:\n{}",
        captured.text
    );
    assert!(
        plan.held.is_empty(),
        "2031 is identified at T_safe, not held unknown: {:?}",
        plan.held
    );
    assert!(matches!(
        session_action(&plan, "s31"),
        SessionAction::CompleteAbandonedDelete { .. }
    ));
    assert!(plan
        .advisories
        .contains(&Advisory::UnreferencedBackup { object_id: 2031 }));
    assert_eq!(store.gets_of(2031, 20), 2, "read before T_safe, then at it");
    assert_eq!(
        store.gets_of(2031, 10),
        1,
        "at T_safe the newest generation identifies 2031"
    );
    assert_eq!(
        lines_starting(
            &captured,
            "the restore Job could not read backup generation 10 of object 2031 to learn its \
             session"
        )
        .len(),
        1
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The planner's own read and coverage sites, end to end
// ═════════════════════════════════════════════════════════════════════════════

/// Trigger (d) end to end (`snapshot::take_listing`). After a wipe, `S`'s
/// listed session backup reads absent before `T_safe` and present at it. The
/// re-take reads it again instead of reusing the absent read, so `S` is
/// restored (Step B, with its mapping of 911) and nothing is held. A re-take
/// that reused the absent read would hold `S` as an unreadable session backup.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_reads_a_session_backup_absent_before_t_safe_again_at_it() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    put_session(store.as_ref(), S, &[(L, 911)]).await;
    put_base(store.as_ref(), 911, S, L, 10, base_ompf(&[])).await;
    let (run, _metrics) = run_on(&tv, &store, T_SAFE_AHEAD).await;
    store.fault_session(S, run.deadlines().t_safe_ms);

    let (plan, captured) = captured(RestorePlanner::plan(&run)).await;
    assert_listed_before_t_safe(&store, &run);
    let plan = plan.expect("the plan");

    assert!(plan.epoch.relisted_at_t_safe);
    one(&captured, "INFO", ONE_TRIGGER);
    assert!(
        waits_for(
            &captured,
            "the session backup of session sess-plan read as absent before T_safe, when the \
             backup store's not-found answers can still be stale; reading it again then"
        ) >= 1,
        "{}",
        captured.text
    );
    assert!(
        plan.held.is_empty(),
        "the session backup present at T_safe is restored, not held: {:?}",
        plan.held
    );
    match session_action(&plan, S) {
        SessionAction::Restore { checkpoint, step_b } => {
            assert!(step_b);
            assert_eq!(
                checkpoint.layer_mappings,
                BTreeMap::from([(L.to_owned(), 911)])
            );
        }
        other => panic!("expected a session restore, got {other:?}"),
    }
    assert_eq!(restored_head(&plan, 911), 10);
    assert_eq!(
        store.session_gets_of(S),
        2,
        "read before T_safe, then at it"
    );
}

/// The lineage memo across the `T_safe` re-take (`run_pass`). 901's newest
/// generation reads absent before `T_safe` (trigger (c)). 902's listing is
/// the same at the re-list, so its first verdict stands and its backup is
/// fetched once. 903 gains a newer generation, 30, between the two listings,
/// so it is selected again and 30 is restored. A memo never consulted would
/// fetch 902 twice; one keyed on the object alone would restore 903 from 10.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_keeps_an_unchanged_lineage_verdict_across_the_retake_and_selects_a_changed_one_again(
) {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    let layers = [(L, 901), ("b.usda", 902), ("c.usda", 903)];
    put_session(store.as_ref(), S, &layers).await;
    put_base(store.as_ref(), 901, S, L, 10, base_ompf(&[])).await;
    put_base(store.as_ref(), 901, S, L, 20, base_ompf(&[])).await;
    put_base(store.as_ref(), 902, S, "b.usda", 10, base_ompf(&[])).await;
    put_base(store.as_ref(), 903, S, "c.usda", 10, base_ompf(&[])).await;
    store.put_base_at_listing(2, 903, S, "c.usda", 30, base_ompf(&[]));
    seed_session(&tv, S).await;
    for (layer, object_id) in layers {
        seed_mapping(&tv, S, layer, object_id).await;
    }
    let (run, _metrics) = run_on(&tv, &store, T_SAFE_AHEAD).await;
    store.fault_base(901, 20, GetFault::Absent, run.deadlines().t_safe_ms);

    let (plan, _captured) = captured(RestorePlanner::plan(&run)).await;
    assert_listed_before_t_safe(&store, &run);
    let plan = plan.expect("the plan");

    assert!(plan.epoch.relisted_at_t_safe);
    assert_eq!(store.listings.load(Ordering::SeqCst), 2);
    assert_eq!(restored_head(&plan, 901), 20);
    assert_eq!(restored_head(&plan, 902), 10);
    assert_eq!(
        store.gets_of(902, 10),
        1,
        "an unchanged listing keeps its first verdict: no second fetch"
    );
    assert_eq!(
        restored_head(&plan, 903),
        30,
        "a listing that changed at the re-list is selected again"
    );
    assert_eq!(store.gets_of(903, 30), 1);
}

/// A surviving layer whose `/World` prim has a `points` attribute holding
/// the external value `hash`, as `create_layer` writes it.
fn survivor_state(hash: [u8; 32]) -> ObjectState {
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        next_set_order: 10,
        ..Default::default()
    };
    for (section_id, parent_id, name, section_type) in
        [(1, 0, "World", SPEC_PRIM), (2, 1, "points", SPEC_ATTRIBUTE)]
    {
        state.sections.insert(
            section_id,
            Section {
                section_id,
                parent_id,
                name: name.to_owned(),
                section_type,
                section_order: 0,
                first_version: 1,
                last_version: 1,
            },
        );
    }
    state.children.insert(0, vec![1]);
    state.children.insert(1, vec![2]);
    state.fields.insert(
        (2, "default".to_owned()),
        Field {
            section_id: 2,
            key_name: "default".to_owned(),
            value: FieldValue::External {
                hash,
                size: 4_096,
                ext_value_index: 0,
            },
            set_order: 4,
            authoring_set_order: 104,
            version: 1,
        },
    );
    state
}

/// `finalize`'s survivor arm (`name_sites`, the `dr_restore` slim read). A
/// layer that survives in the store, created through the service's own write
/// path, references a value no store holds: the report names its row — prim
/// path, attribute, count — read from the surviving state, and logs it. A
/// survivor arm that named nothing, or failed its read, would leave the site
/// list empty.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_coverage_names_a_surviving_layers_uncovered_value_from_its_state() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    let nowhere = value(0x61);
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("survivor", "/survivor", "user")
        .await
        .expect("the session");
    let object_id = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .create_layer(
            &session_id,
            L,
            "user",
            &survivor_state(nowhere),
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("the layer");
    seed_obj_ext(&tv, object_id, &[nowhere]).await;
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (result, captured) = captured(plan_and_finalize(&run)).await;
    let (plan, report) = result.expect("the plan");

    assert!(
        matches!(object_action(&plan, object_id), ObjectAction::Keep),
        "fixture: the layer survives"
    );
    assert_eq!(report.verdicts.get(&nowhere), Some(&HashVerdict::Uncovered));
    assert_eq!(report.layers.len(), 1, "{:?}", report.layers);
    let layer = &report.layers[0];
    assert_eq!(
        (
            layer.object_id,
            layer.session_id.as_str(),
            layer.layer_name.as_str()
        ),
        (object_id, session_id.as_str(), L)
    );
    assert_eq!(layer.naming_error, None);
    assert_eq!(
        layer.uncovered_sites,
        vec![ValueSite {
            prim_path: "/World".to_owned(),
            attribute: "points".to_owned(),
            hash: nowhere,
            count: 1,
        }]
    );
    let line = one(
        &captured,
        "WARN",
        &format!(
            "layer plan.usda of session {session_id} references 1 values that are in no store"
        ),
    );
    assert_eq!(
        line["fields"]["values"],
        format!("/World points ×1 {}", &sha256_to_hex(&nowhere)[..16])
    );
}

/// The probe's call ceiling (`probe_all`). A value whose probe never answers
/// is unknown once one ceiling passes, naming the ceiling, and the plan
/// completes, with the value held beside it covered. The ceiling is the
/// StorageApi adapter's at a 200 ms request timeout, about 1.2 s. Without the
/// ceiling the plan would hang until [`RUN_BUDGET`].
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_coverage_holds_a_value_whose_probe_never_answers_as_unknown_after_one_ceiling() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    let (wedged, held) = (value(0xe1), value(0xe2));
    store
        .put_ext_blob(&sha256_to_hex(&held), Bytes::from_static(b"held"))
        .await
        .expect("the value put");
    store
        .ext_hanging
        .lock()
        .unwrap()
        .insert(sha256_to_hex(&wedged));
    seed_session(&tv, S).await;
    seed_complete(&tv, 1961, S, L, 10, &[]).await;
    seed_mapping(&tv, S, L, 1961).await;
    seed_obj_ext(&tv, 1961, &[wedged, held]).await;
    let (mut run, metrics) = run_on(&tv, &store, Duration::ZERO).await;
    let mut config = crate::common::test_config();
    config.checkpoint_adapter = CheckpointAdapter::StorageApi;
    config.checkpoint_request_timeout_ms = 200;
    let ceiling = config.checkpoint_adapter_call_ceiling();
    assert!(
        (Duration::from_secs(1)..=Duration::from_secs(2)).contains(&ceiling),
        "fixture: a short ceiling of whole seconds, got {ceiling:?}"
    );
    run.config = Arc::new(config);

    let (plan, _captured) = captured(RestorePlanner::plan(&run)).await;
    let plan = plan.expect("the plan completes");

    assert_eq!(
        plan.coverage.verdicts.get(&wedged),
        Some(&HashVerdict::Unknown(format!(
            "no answer within the backup store's call ceiling of {}s",
            ceiling.as_secs()
        )))
    );
    assert_eq!(
        plan.coverage.verdicts.get(&held),
        Some(&HashVerdict::Backup)
    );
    assert_eq!(store.probes_of(&wedged), 1, "one call, never retried");
    assert_eq!(probe_cell(&metrics, "error"), 1.0);
}

/// The breaker counts errors in a row, not in total (`probe_all`, PD-5). 63
/// values answer one at a time: 31 errors, one present, then 31 errors. That
/// is 62 errors, but never 32 in a row, so the breaker does not trip: the plan
/// completes with 62 values unknown and one covered. A breaker that never
/// reset would stop at the 32nd error.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn restore_coverage_breaker_counts_only_errors_in_a_row() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    let hashes: Vec<[u8; 32]> = (0..63u8).map(|i| value(0x10 + i)).collect();
    *store.sequenced.lock().unwrap() = Some(31);
    seed_session(&tv, S).await;
    seed_complete(&tv, 1971, S, L, 10, &[]).await;
    seed_mapping(&tv, S, L, 1971).await;
    seed_obj_ext(&tv, 1971, &hashes).await;
    let (run, metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (plan, captured) = captured(RestorePlanner::plan(&run)).await;
    let plan = plan.expect("62 errors, never 32 in a row, do not trip the breaker");

    let verdicts: Vec<&HashVerdict> = plan.coverage.verdicts.values().collect();
    assert_eq!(verdicts.len(), 63);
    assert_eq!(
        verdicts
            .iter()
            .filter(|v| matches!(v, HashVerdict::Unknown(_)))
            .count(),
        62
    );
    assert_eq!(
        verdicts
            .iter()
            .filter(|v| ***v == HashVerdict::Backup)
            .count(),
        1
    );
    assert_eq!(probe_cell(&metrics, "error"), 62.0);
    assert_eq!(probe_cell(&metrics, "present"), 1.0);
    assert!(lines_starting(&captured, "the backup store failed").is_empty());
}

/// Contradiction (e) on a session, through `read_session_facts`' `pub_kind`
/// read. Session `s-pub`'s record survives and its checkpoint state says its
/// last backup-store operation stored a session backup (`pub_kind=write`),
/// but the session listing, which works (it lists `s-other`), does not show
/// one. Before `T_safe` that is a trigger; at `T_safe` the plan ends held
/// (exit 3), naming the session, and nothing is written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_holds_on_a_session_whose_last_backup_operation_wrote_a_backup_the_listing_lacks(
) {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    put_session(store.as_ref(), "s-other", &[]).await;
    seed_session(&tv, "s-pub").await;
    hset(
        &tv,
        tv.key_schema().session_checkpoint_state("s-pub"),
        &[("pub_kind", "write")],
    )
    .await;
    seed_complete(&tv, 1981, "s-pub", L, 10, &[]).await;
    seed_mapping(&tv, "s-pub", L, 1981).await;
    let (run, _metrics) = run_on(&tv, &store, T_SAFE_AHEAD).await;
    let store_before = keyspace(&tv).await;
    let writes_before = store.writes.load(Ordering::SeqCst);

    let (result, captured) = captured(RestorePlanner::plan(&run)).await;
    assert_listed_before_t_safe(&store, &run);

    let exit = match result {
        Ok(plan) => panic!(
            "the contradiction ends the plan held; it completed with {:?}",
            plan.sessions
        ),
        Err(exit) => exit,
    };
    assert_eq!((exit, exit.code()), (JobExit::StaysHeld, 3));
    let contradiction = Contradiction::Session {
        session_id: "s-pub".to_owned(),
    };
    one(
        &captured,
        "ERROR",
        &format!(
            "the restore Job wrote nothing and exits 3; the service stays held: the backup \
             store's listing contradicts the store: {contradiction}; check the backup store's \
             address, credentials and listing permissions"
        ),
    );
    assert!(
        waits_for(
            &captured,
            &format!(
                "the backup store's listing contradicts the store before T_safe: \
                 {contradiction}; listing again then"
            )
        ) >= 1,
        "{}",
        captured.text
    );
    let writes = keyspace_writes(&store_before, &keyspace(&tv).await, WriteRules::STRICT);
    assert!(
        writes.is_empty(),
        "nothing is written to the store: {writes:?}"
    );
    assert_eq!(store.writes.load(Ordering::SeqCst), writes_before);
}

/// The plan's largest listed object id counts every listed artifact,
/// companions included (the counter term the restore writer seeds from). An
/// orphan meta companion of object 99 921, above every listed generation, is
/// the largest listed id, though it is no generation and plans nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_counts_an_orphan_companion_in_its_largest_listed_object_id() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    put_session(store.as_ref(), S, &[(L, 921)]).await;
    put_base(store.as_ref(), 921, S, L, 5, base_ompf(&[])).await;
    store.extra_artifacts.lock().unwrap().push(LayerArtifact {
        object_id: 99_921,
        snap_seq: 3,
        kind: LayerArtifactKind::Meta,
    });
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (plan, _captured) = captured(RestorePlanner::plan(&run)).await;
    let plan = plan.expect("the plan");

    assert_eq!(plan.listed_max_object_id, Some(99_921));
    assert_eq!(restored_head(&plan, 921), 5);
    assert_eq!(planned(&plan), BTreeSet::from([921]));
    assert!(plan.held.is_empty(), "{:?}", plan.held);
    assert!(!plan
        .advisories
        .contains(&Advisory::UnreferencedBackup { object_id: 99_921 }));
}

/// PD-2 end to end. The restored base of 931 has an ext-refs sidecar naming
/// one value no row references and no store holds: the plan is writable,
/// reports the advisory and its WARN, and never checks or probes that value.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_reports_a_sidecar_only_value_without_checking_or_holding_on_it() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    let (in_backup, sidecar_only) = (value(0x91), value(0x92));
    put_session(store.as_ref(), S, &[(L, 931)]).await;
    put_base(
        store.as_ref(),
        931,
        S,
        L,
        5,
        base_ompf(&[("points", in_backup)]),
    )
    .await;
    store
        .put_ext_blob(&sha256_to_hex(&in_backup), Bytes::from_static(b"held"))
        .await
        .expect("the value put");
    store
        .sidecar_extra
        .lock()
        .unwrap()
        .insert((931, 5), vec![sha256_to_hex(&sidecar_only)]);
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (result, captured) = captured(plan_and_finalize(&run)).await;
    let (plan, report) = result.expect("the plan");

    assert_eq!(restored_head(&plan, 931), 5);
    assert!(plan
        .advisories
        .contains(&Advisory::SidecarHashesNotInState {
            object_id: 931,
            count: 1,
            base_only: true,
        }));
    one(
        &captured,
        "WARN",
        "the backup restored for object 931 lists 1 external values in its reference list that \
         none of its rows references; they are not checked",
    );
    assert_eq!(report.verdicts.get(&sidecar_only), None);
    assert_eq!(store.probes_of(&sidecar_only), 0);
    assert_eq!(report.verdicts.get(&in_backup), Some(&HashVerdict::Backup));
    assert_eq!(plan.outcome(&report), PlanOutcome::Writable);
}

/// A failed backup-store listing ends the plan with exit 1, naming the read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_exits_1_when_the_backup_listing_fails() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    put_session(store.as_ref(), S, &[(L, 941)]).await;
    put_base(store.as_ref(), 941, S, L, 5, base_ompf(&[])).await;
    *store.listing_fails.lock().unwrap() = true;
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (plan, captured) = captured(RestorePlanner::plan(&run)).await;

    assert_eq!(plan.err(), Some(JobExit::Failed));
    one(
        &captured,
        "ERROR",
        "the restore Job could not read the backup store to plan the restore (the layer backups' \
         listing), wrote nothing and exits 1",
    );
}

/// A failed store read ends the plan with exit 1, naming the read, before
/// the backup store is called: here an object header that is not a hash, so
/// its read answers WRONGTYPE.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_plan_exits_1_when_a_store_read_fails() {
    let tv = TestValkey::setup().await;
    let store = PlanStore::new(&tv);
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().obj_header("1991"),
            "not-a-hash",
            None,
            None,
            false,
        )
        .await
        .expect("SET");
    let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

    let (plan, captured) = captured(RestorePlanner::plan(&run)).await;

    assert_eq!(plan.err(), Some(JobExit::Failed));
    one(
        &captured,
        "ERROR",
        "the restore Job could not read the store to plan the restore (the object headers), \
         wrote nothing and exits 1",
    );
    assert_eq!(store.calls(), 0, "the backup store is never called");
}

// ═════════════════════════════════════════════════════════════════════════════
// b.f3t: the planner's keyspace walks, each an agreeing pair (b.gns before)
// ═════════════════════════════════════════════════════════════════════════════

/// The planner's two keyspace walks (`storage::dr_job`'s walks of the object
/// headers and the layer mappings), each walked twice and used only when the
/// two walks return the same keys (`storage::scan_guard::collect_keyspace_agreed`),
/// through the walk seam (`storage::scan_guard::arm_walk_fault_for_test`,
/// keyed by the walk's pattern, which carries the test's prefix): a key one
/// walk skips (the seam's `DropMember`) ends the plan with exit 1, writing
/// nothing and reading no backup (T1); a store whose `INFO server` reports
/// no usable run_id, or one that changes, no longer holds the plan (T6); and,
/// on a cluster, a pinned node whose `CLUSTER NODES` has no myself line
/// still ends it with exit 3 (test 19's planner part, cluster-gated).
/// Compiled only with `test-fault-injection`.
///
/// Fail-on-old (in-place perturbations, red then green once restored): with
/// `scan_guard::keyspace_walks_agree` answering `Ok` whatever the two sets,
/// every forged plan completes and the Job run exits 1 only at its kill
/// point before the un-hold, having planned; with `KeyspaceWalk::start_job`
/// beginning a guarded walk again (its `INFO server` witness put back), the
/// placeholder run_ids exit 3 and the change exits 1. The cluster test's
/// perturbation (an unusable myself line mapped to a `WalkError::Read`,
/// exit 1) is not run in this sandbox, which has no cluster.
#[cfg(feature = "test-fault-injection")]
mod serving_process_guard {
    use super::*;

    use fred::interfaces::{ClientLike, ClusterInterface};
    use live_edit::storage::scan_guard::{
        arm_walk_fault_for_test, forge_process_change_for_test, ArmedWalkFault, WalkFault,
    };

    /// Drop `key` from walk number `walk` of `pattern`, whichever step
    /// returns it: a keyspace walk of a shared store answers the test's key
    /// at a step no test can know in advance, so the seam's `DropMember`
    /// names the key on every step of the walk (`step: None`). A drop that
    /// never fires leaves the walks agreeing, and the test that expects them
    /// to disagree fails.
    fn drop_key(pattern: &str, walk: usize, key: &str) -> ArmedWalkFault {
        arm_walk_fault_for_test(
            pattern,
            WalkFault::DropMember {
                walk,
                step: None,
                member: Some(key.to_owned()),
            },
        )
    }

    /// The one ERROR whose message contains `needle`; it must name every
    /// one of `parts`.
    fn the_error(captured: &Captured, needle: &str, parts: &[&str]) {
        let found: Vec<&str> = captured
            .events
            .iter()
            .filter(|e| e["level"] == "ERROR")
            .filter_map(|e| e["fields"]["message"].as_str())
            .filter(|m| m.contains(needle))
            .collect();
        assert_eq!(
            found.len(),
            1,
            "one ERROR containing {needle:?}; capture:\n{}",
            captured.text
        );
        for part in parts {
            assert!(
                found[0].contains(part),
                "the ERROR names {part:?}: {}",
                found[0]
            );
        }
    }

    /// I2's partial wipe: a kept survivor and a lost layer, both mapped.
    async fn partial_wipe(tv: &TestValkey, store: &PlanStore) {
        put_session(store, S, &[("kept.usda", 101), ("lost.usda", 102)]).await;
        put_base(store, 101, S, "kept.usda", 40, base_ompf(&[])).await;
        put_base(store, 102, S, "lost.usda", 30, base_ompf(&[])).await;
        seed_session(tv, S).await;
        seed_complete(tv, 101, S, "kept.usda", 50, &[]).await;
        seed_mapping(tv, S, "kept.usda", 101).await;
        seed_mapping(tv, S, "lost.usda", 102).await;
    }

    /// The planner's ERROR naming two walks of `pattern` that disagreed, the
    /// first returning `first` distinct keys and the second `second`.
    fn walks_disagree(pattern: &str, first: u64, second: u64) -> String {
        format!(
            "two SCAN walks of {pattern} returned different keys: {first} and {second} distinct \
             keys, {} found only by the first walk and {} only by the second",
            first.saturating_sub(second),
            second.saturating_sub(first)
        )
    }

    /// T1 (b.f3t). One key skipped by one of the planner's two walks of a
    /// pattern — the kept survivor's header from the first walk of the
    /// object headers, then from the second, then its mapping from each walk
    /// of the layer mappings — ends the plan with exit 1 and one ERROR
    /// saying the backing store changed while the Job read it, naming the
    /// pattern and each walk's distinct keys, and "wrote nothing and exits 1
    /// so Kubernetes retries it"; the backup store is never called. The
    /// control, unarmed, plans.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn restore_plan_exits_1_when_its_two_walks_of_a_pattern_disagree() {
        let tv = TestValkey::setup().await;
        within_budget(two_walks_of_a_pattern_disagree(&tv)).await;
    }

    /// [`restore_plan_exits_1_when_its_two_walks_of_a_pattern_disagree`]'s
    /// five plans on `tv`, which its caller bounds by one budget together.
    async fn two_walks_of_a_pattern_disagree(tv: &TestValkey) {
        let store = PlanStore::new(tv);
        partial_wipe(tv, store.as_ref()).await;
        let (run, _metrics) = run_on(tv, &store, Duration::ZERO).await;
        let (control, _captured) = captured(plan_and_finalize(&run)).await;
        assert!(control.is_ok(), "control: the unarmed plan completes");
        let calls = store.calls();
        let keys = tv.key_schema();

        // (what, the pattern, the key skipped, how many keys the pattern
        // matches)
        let walks = [
            (
                "the object headers",
                keys.obj_header_scan_pattern(),
                keys.obj_header("101"),
                1,
            ),
            (
                "the layer mappings",
                keys.layer_mapping_scan_pattern(),
                keys.layer_mapping(S, "kept.usda"),
                2,
            ),
        ];
        for (what, pattern, key, matched) in walks {
            for walk in [0, 1] {
                let case = format!("{what}, walk {walk}");
                let armed = drop_key(&pattern, walk, &key);
                let (plan, captured) = captured(RestorePlanner::plan(&run)).await;
                drop(armed);

                assert_eq!(plan.err(), Some(JobExit::Failed), "{case}: exit 1");
                let (first, second) = if walk == 0 {
                    (matched - 1, matched)
                } else {
                    (matched, matched - 1)
                };
                the_error(
                    &captured,
                    &format!(
                        "the restore Job could not plan the restore ({what}): the backing store \
                         changed while the Job read it"
                    ),
                    &[
                        &walks_disagree(&pattern, first, second),
                        "wrote nothing and exits 1 so Kubernetes retries it",
                    ],
                );
            }
        }
        assert_eq!(store.calls(), calls, "the backup store is never called");
    }

    /// T1 at Job level (b.f3t). A restore Job whose second walk of the object
    /// headers skips the kept survivor's header exits 1 having planned
    /// nothing and written nothing to the store but its lease, nor anything
    /// to the backup store.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn restore_job_whose_header_walks_disagree_exits_1_and_writes_nothing() {
        let tv = TestValkey::setup().await;
        let store = PlanStore::new(&tv);
        partial_wipe(&tv, store.as_ref()).await;
        put_counters(store.as_ref()).await;
        let config = storage_api_config(2);
        hold_with_quiet_left(&tv, &config, Duration::ZERO).await;
        let store_before = keyspace(&tv).await;
        let writes_before = store.writes.load(Ordering::SeqCst);
        let metrics = metrics();
        let (body, observed) = observing_body();
        let pattern = tv.key_schema().obj_header_scan_pattern();

        let armed = drop_key(&pattern, 1, &tv.key_schema().obj_header("101"));
        let (exit, captured) = run(
            context(
                &tv,
                JobMode::Restore,
                config,
                Arc::clone(&store) as _,
                &metrics,
            ),
            body,
        )
        .await;
        drop(armed);

        assert_eq!(exit, JobExit::Failed, "{}", captured.text);
        assert_eq!(exit.code(), 1);
        the_error(
            &captured,
            "the restore Job could not plan the restore (the object headers): the backing store \
             changed while the Job read it",
            &[&walks_disagree(&pattern, 1, 0)],
        );
        assert!(
            observed.lock().unwrap().is_empty(),
            "nothing is planned to write"
        );
        let writes = keyspace_writes(&store_before, &keyspace(&tv).await, WriteRules::STRICT);
        assert!(
            writes.is_empty(),
            "nothing is written to the store but the lease: {writes:?}"
        );
        assert_eq!(
            store.writes.load(Ordering::SeqCst),
            writes_before,
            "nothing is written to the backup store"
        );
    }

    /// T6 for the planner (b.f3t; test 9's planner consumer before). The
    /// planner's walks read no serving-process witness, so a store whose
    /// `INFO server` reports a placeholder run_id (`"0"`) or no `run_id`
    /// line, or whose run_id changes between two reads, no longer holds the
    /// plan: with each forged through the walk seam on both its patterns (a
    /// `RunId` fault on the first witness read, or a change of serving
    /// process at the second), the plan completes and logs no ERROR.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn restore_plan_runs_on_a_store_whose_run_id_is_unusable_or_changes() {
        type Forge = fn(&str) -> ArmedWalkFault;
        let cases: [(&str, Forge); 3] = [
            ("run_id \"0\"", |pattern| {
                arm_walk_fault_for_test(
                    pattern,
                    WalkFault::RunId {
                        read: 0,
                        run_id: Some("0".to_owned()),
                    },
                )
            }),
            ("no run_id line", |pattern| {
                arm_walk_fault_for_test(
                    pattern,
                    WalkFault::RunId {
                        read: 0,
                        run_id: None,
                    },
                )
            }),
            ("a change of run_id", |pattern| {
                forge_process_change_for_test(pattern, 1)
            }),
        ];
        let tv = TestValkey::setup().await;
        let store = PlanStore::new(&tv);
        partial_wipe(&tv, store.as_ref()).await;
        let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;
        let patterns = [
            tv.key_schema().obj_header_scan_pattern(),
            tv.key_schema().layer_mapping_scan_pattern(),
        ];

        // Every case is run before the test fails, so one run names every
        // case a regression reaches; the three plans share one budget.
        let failures = within_budget(async {
            let mut failures: Vec<String> = Vec::new();
            for (case, forge) in cases {
                let armed = patterns.each_ref().map(|pattern| forge(pattern.as_str()));
                let (plan, captured) = captured(plan_and_finalize(&run)).await;
                drop(armed);

                if let Err(exit) = plan {
                    failures.push(format!(
                        "{case}: the plan completes, got exit {}",
                        exit.code()
                    ));
                    continue;
                }
                let errors = captured
                    .events
                    .iter()
                    .filter(|e| e["level"] == "ERROR")
                    .count();
                if errors != 0 {
                    failures.push(format!(
                        "{case}: no ERROR, got {errors}; capture:\n{}",
                        captured.text
                    ));
                }
            }
            failures
        })
        .await;
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// Test 19's planner part (b.gns, policy 6; cluster-gated, CI's
    /// `test-cluster:` job). On a three-shard cluster, the planner's
    /// object-header walk whose first pinned node answers `CLUSTER NODES`
    /// with no myself line ends the plan with exit 3 and one ERROR naming
    /// `CLUSTER NODES`, the node and "no myself line".
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn restore_plan_on_a_cluster_exits_3_when_a_pinned_node_answers_no_myself_line() {
        let Some(url) = live_edit::storage::test_support::test_cluster_url() else {
            eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set");
            return;
        };
        let tv = TestValkey::setup_with_url(url, 2).await;
        let client = tv.client().maintenance_pool.next();
        assert!(
            client.is_clustered(),
            "LIVE_EDIT_TEST_CLUSTER_URL names a cluster"
        );
        let node = client
            .cached_cluster_state()
            .expect("a routing table")
            .get_server(0)
            .cloned()
            .expect("the table names slot 0's owner");
        let store = PlanStore::new(&tv);
        partial_wipe(&tv, store.as_ref()).await;
        let (run, _metrics) = run_on(&tv, &store, Duration::ZERO).await;

        let armed = arm_walk_fault_for_test(
            &tv.key_schema().obj_header_scan_pattern(),
            WalkFault::ClusterNodesReply {
                read: 0,
                reply: "0000000000000000000000000000000000000001 127.0.0.1:7000@17000 master - \
                        0 0 1 connected 0-16383\n"
                    .to_owned(),
            },
        );
        let (plan, captured) = captured(RestorePlanner::plan(&run)).await;
        drop(armed);

        assert_eq!(plan.err(), Some(JobExit::StaysHeld), "exit 3");
        the_error(
            &captured,
            "cannot plan the restore (the object headers)",
            &[
                "`CLUSTER NODES`",
                &format!("node {node}"),
                "no myself line",
                "exits 3",
            ],
        );
        println!(
            "CLUSTER-GATED RAN: restore_plan_on_a_cluster_exits_3_when_a_pinned_node_answers_no_myself_line"
        );
    }
}
