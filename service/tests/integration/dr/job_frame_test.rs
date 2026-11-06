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

//! The DR Job frame in process (`live_edit::dr::job::run_job`; SDD SR-8.1,
//! SR-9.1, SR-9.3, SR-10), over a real store, with shortened lease timings
//! and a body of the test's own that records what the frame hands it. Every
//! exit of the frame's run order, each in a test:
//!
//! - **The precondition.** An un-held store exits 0 and takes no lease;
//!   another incident exits 4; a marker with no usable `incident` or
//!   `detected_at_ms` exits 3 naming the field; none of them takes the lease,
//!   calls the backup store or writes a key. Read again under the lease: a
//!   marker removed, changed to another incident or made unusable while the
//!   Job waited for its lease ends the run with the same exits, and the lease
//!   is released.
//! - **The adapter.** `Disabled` and `InMemory` never call the backup store;
//!   a restore on `InMemory` exits 3 before the lease, and an accept-loss
//!   reaches the body with no backup store.
//! - **The lease.** A busy lease is waited on for the acquire wait, then the
//!   Job exits 1 naming its holder; a lease lost during the quiet period, or
//!   while the body writes, exits 1; SIGTERM, during the quiet period or while
//!   the body writes, exits 1 and releases the lease, so another Job takes it
//!   at once.
//! - **The quiet period** (T-R1c). The body plans once, while the permit is
//!   still paused, and its first write unit starts only once the store's
//!   clock has passed Q; the Job holds its lease while it writes and releases
//!   it at the exit.
//! - **The absence rule** with a persistent backup store. A counters record
//!   absent before `T_safe` is read again there: absent again, the Job exits
//!   3 with the named cause and the body never runs; present, the run goes
//!   on. The list-route canary the same way: failing at `T_safe` exits 3,
//!   failing before it and passing there goes on, and a backend fault exits 1.
//!   A storage-api backup store whose listing route answers 404 while its
//!   by-address routes work exits 3, with nothing written. A contradiction
//!   (a stamped surviving object the listing does not show) seen before
//!   `T_safe` is looked at again there through the same primitive, and the
//!   second look decides.
//!
//! The binary's own dispatch, refusal and exit are
//! `tests/misc/dr_job_binary.rs`'s; Q's arithmetic is
//! `tests/misc/dr_job_rules.rs`'s; the lease's refresh and loss timer are
//! `job_lease_test.rs`'s.
//!
//! # Time
//!
//! The store's clock decides every wait here, as it does for the Job: a hold
//! "an hour ago" has its quiet period and `T_safe` long past; one "ending in
//! a few seconds" is written with `detected_at_ms` = store `TIME` − Q + that
//! much. The absence-rule tests use the `StorageApi` adapter's not-found
//! cache lifetime L of [`L_SECS`] with a quiet period that ends at once, so
//! `T_safe` is a few seconds ahead and a wait for it is real. No assertion
//! bounds how long a wait took; [`RUN_BUDGET`] only bounds a hang.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

use live_edit::checkpoint::adapters::in_memory::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    CheckpointAdapter, CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointMeta,
    CheckpointStore, LayerArtifact, LiveExtSet,
};
use live_edit::config::Config;
use live_edit::dr::job::absence::{LIST_ROUTE_CAUSE, SIDECAR_ABSENT_CAUSE};
use live_edit::dr::job::lease::{JobLease, LeaseTimings};
use live_edit::dr::job::permit::{PermitState, RevokeCause, WritePermit};
use live_edit::dr::job::quiet::QuietPeriod;
use live_edit::dr::job::{
    run_job, BackupAccess, JobArgs, JobBody, JobContext, JobExit, JobMode, JobRun, JobTimings,
    NoBackupStore,
};
use live_edit::storage::dr_hold::{MARKER_FIELD_DETECTED_AT_MS, MARKER_FIELD_INCIDENT};
use live_edit::storage::dr_job::store_time_ms;
use live_edit::storage::external_values::{
    try_acquire_leader_lock, LEADER_ACQUIRE_OR_REFRESH_SCRIPT,
};
use live_edit::storage::leader_lease::LeaderIdentity;
use live_edit::storage::test_support::{HoldMarkerSeed, TestValkey};

use super::support::{keyspace, keyspace_writes, WriteRules};
use crate::checkpoint::adapter_storage_api_test::{
    build_fake_backed_adapter, mount_fake_storage_routes, FakeStorageState, BASE_PATH,
};
use crate::grpc::log_capture::{capture, Captured};
use crate::storage::snapshot_read_support::{
    client_through, test_valkey_addr, ProxyAction, ScriptReplyProxy,
};

/// An hour, in milliseconds: how far before the store's clock a hold whose
/// quiet period and `T_safe` are long over was detected.
const HOUR_MS: u64 = 3_600_000;

/// L for the absence-rule tests: `T_safe` this many seconds after Q's end.
const L_SECS: u64 = 3;

/// How much of the quiet period is left in the tests that watch it end.
const QUIET_LEFT: Duration = Duration::from_secs(3);

/// Hang bound on one Job run. A healthy run ends within a few seconds.
const RUN_BUDGET: Duration = Duration::from_secs(30);

/// Hang bound on a wait for something the Job does mid-run.
const SETTLE: Duration = Duration::from_secs(10);

/// The incident every Job here is started for.
const INCIDENT: &str = "20261016t0800-frame";

/// The ERROR of a Job held under another incident, for [`INCIDENT`].
fn different_incident_error(held: &str) -> String {
    format!(
        "held under a different incident {held}: the restore Job was started for incident \
         {INCIDENT}, wrote nothing and exits 4"
    )
}

/// The ERROR of a marker missing `field`.
fn marker_incomplete_error(field: &str) -> String {
    format!(
        "the restore Job wrote nothing and exits 3; the service stays held: the hold marker's \
         {field} field is missing or unreadable, and the Job never guesses it"
    )
}

// ═════════════════════════════════════════════════════════════════════════════
// The body
// ═════════════════════════════════════════════════════════════════════════════

/// What [`RecordingBody`] does once its first write unit has started.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Writes {
    /// One unit, then `Done`.
    One,
    /// One unit, then wait for ever: the frame must end the run.
    UntilStopped,
}

/// What a [`RecordingBody`] saw.
#[derive(Default)]
struct BodyLog {
    plans: AtomicUsize,
    executions: AtomicUsize,
    /// The permit's state each time `plan` was called.
    plan_permit_states: Mutex<Vec<PermitState>>,
    /// The backup store each `execute` was handed.
    backups: Mutex<Vec<&'static str>>,
    /// The store's clock just after the first write unit was let in, and the
    /// quiet period's end on it.
    first_unit_at_ms: Mutex<Option<(u64, u64)>>,
    /// How many write units found the lease key holding this Job's own
    /// lock value ([`job_identity`]), not merely some lease.
    lease_held_while_writing: AtomicUsize,
    /// The run's write permit (a clone shares its state), kept from the
    /// first `execute`, so a test can read how the frame ended it after the
    /// body was dropped.
    permit: Mutex<Option<WritePermit>>,
    /// Notified once the first write unit has started.
    writing: Notify,
}

impl BodyLog {
    /// The permit's state now, as the frame left it.
    fn permit_state(&self) -> PermitState {
        self.permit
            .lock()
            .unwrap()
            .as_ref()
            .expect("the body was handed a permit")
            .state()
    }
}

/// A body that records what the frame hands it, and writes per [`Writes`].
struct RecordingBody {
    log: Arc<BodyLog>,
    writes: Writes,
    /// The lock value of the Job this body runs in: [`job_identity`]'s, the
    /// identity [`context_with`] gives the Job on the same fixture.
    holder: String,
}

impl RecordingBody {
    fn new(writes: Writes, tv: &TestValkey) -> (Self, Arc<BodyLog>) {
        let log = Arc::new(BodyLog::default());
        (
            Self {
                log: Arc::clone(&log),
                writes,
                holder: job_identity(tv).lock_value(),
            },
            log,
        )
    }
}

fn backup_name(backup: &BackupAccess) -> &'static str {
    match backup {
        BackupAccess::None(NoBackupStore::Disabled) => "none: disabled",
        BackupAccess::None(NoBackupStore::InMemory) => "none: in-memory",
        BackupAccess::Store(_) => "store",
    }
}

#[async_trait]
impl JobBody for RecordingBody {
    type Plan = ();

    async fn plan(&self, run: &JobRun) -> Result<(), JobExit> {
        self.log.plans.fetch_add(1, Ordering::SeqCst);
        self.log
            .plan_permit_states
            .lock()
            .unwrap()
            .push(run.permit.state());
        Ok(())
    }

    async fn execute(&self, run: &JobRun, _plan: ()) -> JobExit {
        self.log.executions.fetch_add(1, Ordering::SeqCst);
        self.log
            .backups
            .lock()
            .unwrap()
            .push(backup_name(&run.backup));
        *self.log.permit.lock().unwrap() = Some(run.permit.clone());
        if run.permit.begin_unit().await.is_err() {
            return JobExit::Failed;
        }
        let now = store_time_ms(&run.valkey, &run.keys).await.expect("TIME");
        *self.log.first_unit_at_ms.lock().unwrap() = Some((now, run.deadlines().quiet_ends_at_ms));
        let holder: Option<String> = run
            .valkey
            .pool
            .get(run.keys.dr_job_lease())
            .await
            .expect("GET");
        if holder.as_deref() == Some(self.holder.as_str()) {
            self.log
                .lease_held_while_writing
                .fetch_add(1, Ordering::SeqCst);
        }
        self.log.writing.notify_one();
        match self.writes {
            Writes::One => JobExit::Done,
            Writes::UntilStopped => std::future::pending().await,
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// A backup store whose answers the test scripts
// ═════════════════════════════════════════════════════════════════════════════

/// One answer of [`ScriptedStore::verify_list_route`].
#[derive(Clone, Copy, Debug)]
enum Canary {
    Passes,
    /// The listing route does not list the canary back: `Permanent`.
    NotListed,
    /// A backend fault: `Transient`.
    Fault,
}

/// A backup store over an empty in-memory one whose counters-record reads and
/// list-route canary answer what the test scripts, in order (the last answer
/// repeats), and which counts every call.
struct ScriptedStore {
    inner: InMemoryCheckpointStore,
    /// Whether each counters-record read finds one.
    sidecar: Mutex<VecDeque<bool>>,
    canary: Mutex<VecDeque<Canary>>,
    sidecar_reads: AtomicUsize,
    canary_runs: AtomicUsize,
    other_calls: AtomicUsize,
}

impl ScriptedStore {
    fn new(sidecar: &[bool], canary: &[Canary]) -> Arc<Self> {
        Arc::new(Self {
            inner: InMemoryCheckpointStore::new(),
            sidecar: Mutex::new(sidecar.iter().copied().collect()),
            canary: Mutex::new(canary.iter().copied().collect()),
            sidecar_reads: AtomicUsize::new(0),
            canary_runs: AtomicUsize::new(0),
            other_calls: AtomicUsize::new(0),
        })
    }

    /// A counters record present, and a canary that passes.
    fn healthy() -> Arc<Self> {
        Self::new(&[true], &[Canary::Passes])
    }

    fn sidecar_reads(&self) -> usize {
        self.sidecar_reads.load(Ordering::SeqCst)
    }

    fn canary_runs(&self) -> usize {
        self.canary_runs.load(Ordering::SeqCst)
    }

    /// Every call the store answered, of any method.
    fn calls(&self) -> usize {
        self.sidecar_reads() + self.canary_runs() + self.other_calls.load(Ordering::SeqCst)
    }

    fn other(&self) {
        self.other_calls.fetch_add(1, Ordering::SeqCst);
    }

    fn next<T: Copy>(script: &Mutex<VecDeque<T>>) -> T {
        let mut script = script.lock().unwrap();
        if script.len() > 1 {
            script.pop_front().expect("non-empty")
        } else {
            *script.front().expect("a scripted answer")
        }
    }
}

fn canary_error(canary: Canary) -> Result<(), CheckpointError> {
    match canary {
        Canary::Passes => Ok(()),
        Canary::NotListed => Err(CheckpointError::Permanent {
            source: "the listing did not list the canary back".into(),
        }),
        Canary::Fault => Err(CheckpointError::Transient {
            source: "the backup store answered 503".into(),
        }),
    }
}

#[async_trait]
impl CheckpointStore for ScriptedStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.other();
        self.inner
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }
    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.other();
        self.inner.get_latest_layer_checkpoint(object_id).await
    }
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.other();
        self.inner.get_layer_checkpoint(handle).await
    }
    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.other();
        self.inner.list_layer_checkpoints(object_id).await
    }
    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.other();
        self.inner.delete_layer_checkpoint(handle).await
    }
    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.other();
        self.inner
            .put_session_checkpoint(session_id, body, meta)
            .await
    }
    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.other();
        self.inner.get_session_checkpoint(session_id).await
    }
    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.other();
        self.inner.delete_session_checkpoint(session_id).await
    }
    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.other();
        self.inner.list_session_checkpoint_ids().await
    }
    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.other();
        self.inner.put_ext_blob(hash_hex, bytes).await
    }
    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.other();
        self.inner.get_ext_blob(hash_hex).await
    }
    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.other();
        self.inner.ext_blob_exists(hash_hex).await
    }
    async fn verify_list_route(&self) -> Result<(), CheckpointError> {
        self.canary_runs.fetch_add(1, Ordering::SeqCst);
        canary_error(Self::next(&self.canary))
    }
    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.other();
        self.inner.delete_ext_blob(hash_hex).await
    }
    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.other();
        self.inner.put_counters_sidecar(body).await
    }
    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.sidecar_reads.fetch_add(1, Ordering::SeqCst);
        // A record of the shape the counters tick writes: the Job parses it
        // (PD-7).
        Ok(Self::next(&self.sidecar).then(|| {
            Bytes::from_static(
                b"{\"max_object_id_observed\":0,\"updated_at\":\"2026-10-20T12:00:00Z\"}",
            )
        }))
    }
    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.other();
        self.inner.list_all_object_ids().await
    }
    async fn list_referenced_ext_hashes(
        &self,
    ) -> Result<std::collections::HashSet<String>, CheckpointError> {
        self.other();
        self.inner.list_referenced_ext_hashes().await
    }
    async fn list_live_ext_set(&self) -> Result<LiveExtSet, CheckpointError> {
        self.other();
        self.inner.list_live_ext_set().await
    }
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<std::collections::HashSet<String>, CheckpointError> {
        self.other();
        self.inner.read_ext_refs_sidecar(handle).await
    }
    async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
        self.other();
        self.inner.list_layer_artifacts().await
    }
    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.other();
        self.inner
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }
    fn publish_worst_case(&self) -> Duration {
        self.inner.publish_worst_case()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// The harness
// ═════════════════════════════════════════════════════════════════════════════

/// The frame's timings here. The lease's TTL (30 s) is long and its refresh
/// interval (1 s) short: a lease deleted under the Job is found gone at the
/// next refresh, within about a second, while a refresh slowed by a loaded
/// runner still leaves 29 s before the loss deadline. The lease's loss timer
/// is not under test here (`job_lease_test.rs` pins it), so it stays off the
/// path of the quiet-period and absence-rule tests.
fn timings() -> JobTimings {
    JobTimings {
        lease: LeaseTimings {
            ttl: Duration::from_secs(30),
            refresh_interval: Duration::from_secs(1),
            acquire_wait: Duration::from_secs(3),
            acquire_poll: Duration::from_millis(100),
        },
        wait_log_interval: Duration::from_millis(500),
    }
}

/// The fixture's configuration with `adapter`.
fn config(adapter: CheckpointAdapter) -> Config {
    let mut config = crate::common::test_config();
    config.checkpoint_adapter = adapter;
    config
}

/// The `StorageApi` configuration of the absence-rule tests: L = [`L_SECS`].
fn storage_api_config() -> Config {
    let mut config = config(CheckpointAdapter::StorageApi);
    config.checkpoint_storage_api_not_found_cache_secs = L_SECS;
    config
}

/// The identity every Job on `tv` holds its lease with: one per fixture, so
/// a [`RecordingBody`] can tell its own Job's lease from another's.
fn job_identity(tv: &TestValkey) -> LeaderIdentity {
    LeaderIdentity::new("dr-job-frame-test", tv.prefix())
}

/// A Job context on `tv` for `mode` with `config` and `store`, never
/// signalled, holding its lease as [`job_identity`].
fn context_with(
    tv: &TestValkey,
    mode: JobMode,
    config: Config,
    store: Arc<dyn CheckpointStore>,
) -> JobContext {
    JobContext {
        args: JobArgs {
            mode,
            incident: INCIDENT.to_owned(),
            discard_unconfirmed: false,
        },
        config: Arc::new(config),
        valkey: tv.client().clone(),
        keys: tv.key_schema().clone(),
        checkpoint_store: store,
        metrics: Arc::new(crate::common::metrics_scrape_only("dr-job-frame")),
        metrics_listener: None,
        timings: timings(),
        shutdown: Box::pin(std::future::pending()),
        identity: job_identity(tv),
        allocator_purge: Some(live_edit::storage::allocator_purge::AllocatorPurge::new(
            || true,
        )),
    }
}

/// A restore Job context on `tv` with the `Disabled` adapter.
fn context(tv: &TestValkey) -> JobContext {
    context_with(
        tv,
        JobMode::Restore,
        config(CheckpointAdapter::Disabled),
        Arc::new(InMemoryCheckpointStore::new()),
    )
}

/// A shutdown future that resolves when the returned token is cancelled.
fn shutdown_on(token: &CancellationToken) -> Pin<Box<dyn Future<Output = ()> + Send>> {
    let token = token.clone();
    Box::pin(async move { token.cancelled().await })
}

/// Run `ctx` with `body` to its exit within [`RUN_BUDGET`], capturing what the
/// frame logs.
async fn run<B: JobBody>(ctx: JobContext, body: B) -> (JobExit, Captured) {
    capture(async {
        tokio::time::timeout(RUN_BUDGET, run_job(ctx, body))
            .await
            .expect("the Job ends")
    })
    .await
}

async fn lease_exists(tv: &TestValkey) -> bool {
    let n: i64 = tv
        .client()
        .pool
        .exists(tv.key_schema().dr_job_lease())
        .await
        .expect("EXISTS");
    n == 1
}

/// Wait until the Job's lease key exists: the Job has taken its lease.
async fn until_lease_taken(tv: &TestValkey) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    while !lease_exists(tv).await {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the Job did not take its lease within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn now_ms(tv: &TestValkey) -> u64 {
    store_time_ms(tv.client(), tv.key_schema())
        .await
        .expect("TIME")
}

/// Hold the store under [`INCIDENT`], detected `ago_ms` before the store's
/// clock.
async fn hold_detected_ago(tv: &TestValkey, ago_ms: u64) {
    let now = now_ms(tv).await;
    tv.write_hold_marker(
        &HoldMarkerSeed::default()
            .with_incident(INCIDENT)
            .detected_at(now - ago_ms),
    )
    .await;
}

/// A hold whose quiet period and `T_safe` are over.
async fn hold_an_hour_ago(tv: &TestValkey) {
    hold_detected_ago(tv, HOUR_MS).await;
}

/// A hold whose quiet period under `config` ends `left` from now on the
/// store's clock (at once for `Duration::ZERO`). A `left` longer than Q
/// writes a `detected_at_ms` ahead of the store's clock.
async fn hold_with_quiet_left(tv: &TestValkey, config: &Config, left: Duration) {
    let q = u64::try_from(QuietPeriod::from_config(config).total().as_millis()).expect("Q fits");
    let left = u64::try_from(left.as_millis()).expect("fits");
    let now = now_ms(tv).await;
    tv.write_hold_marker(
        &HoldMarkerSeed::default()
            .with_incident(INCIDENT)
            .detected_at(now + left - q),
    )
    .await;
}

/// The captured lines whose message is exactly `message`.
fn lines<'c>(captured: &'c Captured, message: &str) -> Vec<&'c serde_json::Value> {
    captured
        .events
        .iter()
        .filter(|e| e["fields"]["message"].as_str() == Some(message))
        .collect()
}

/// Assert exactly one line `message` at `level` was captured.
fn assert_one(captured: &Captured, level: &str, message: &str) {
    let found = lines(captured, message);
    assert_eq!(
        found.len(),
        1,
        "one {level} line {message:?}; capture:\n{}",
        captured.text
    );
    assert_eq!(found[0]["level"], level, "{message}");
}

/// Assert no line of the capture starts with `prefix`.
fn assert_none_starting(captured: &Captured, prefix: &str) {
    assert!(
        !captured.events.iter().any(|e| e["fields"]["message"]
            .as_str()
            .is_some_and(|m| m.starts_with(prefix))),
        "no line starts with {prefix:?}; capture:\n{}",
        captured.text
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The precondition
// ═════════════════════════════════════════════════════════════════════════════

/// An un-held store: exit 0, "nothing to do", with no lease taken, no call of
/// the backup store (a persistent adapter is configured) and no key written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_job_on_an_unheld_store_exits_0_without_taking_the_lease() {
    let tv = TestValkey::setup().await;
    let store = ScriptedStore::healthy();
    let before = keyspace(&tv).await;
    let (body, log) = RecordingBody::new(Writes::One, &tv);

    let ctx = context_with(
        &tv,
        JobMode::Restore,
        config(CheckpointAdapter::LocalDisk),
        Arc::clone(&store) as _,
    );
    let (exit, captured) = run(ctx, body).await;

    assert_eq!(exit, JobExit::NothingToDo);
    assert_eq!(exit.code(), 0);
    assert_one(
        &captured,
        "INFO",
        "the store is not held: nothing to do; the restore Job wrote nothing and exits 0",
    );
    assert!(!lease_exists(&tv).await, "no lease on the not-held path");
    assert_none_starting(&captured, "took the DR Job lease");
    assert_eq!(store.calls(), 0, "no call of the backup store");
    assert_eq!(log.plans.load(Ordering::SeqCst), 0);
    let writes = keyspace_writes(&before, &keyspace(&tv).await, WriteRules::STRICT);
    assert!(writes.is_empty(), "nothing written: {writes:?}");
}

/// A store held under another incident: exit 4 naming the held incident, with
/// no lease taken and nothing written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_store_held_under_another_incident_exits_4_naming_it_and_takes_no_lease() {
    let tv = TestValkey::setup().await;
    let held = "20261016t0700-other";
    tv.write_hold_marker(&HoldMarkerSeed::default().with_incident(held))
        .await;
    let before = keyspace(&tv).await;
    let (body, log) = RecordingBody::new(Writes::One, &tv);

    let (exit, captured) = run(context(&tv), body).await;

    assert_eq!(exit, JobExit::DifferentIncident);
    assert_eq!(exit.code(), 4);
    assert_one(&captured, "ERROR", &different_incident_error(held));
    assert_none_starting(&captured, "took the DR Job lease");
    assert!(!lease_exists(&tv).await);
    assert_eq!(log.plans.load(Ordering::SeqCst), 0);
    let writes = keyspace_writes(&before, &keyspace(&tv).await, WriteRules::STRICT);
    assert!(writes.is_empty(), "nothing written: {writes:?}");
}

/// A hold marker with no usable `incident`, or with a `detected_at_ms` that
/// is missing or not a whole number: exit 3 naming the field (the Job never
/// guesses either), with no lease taken and nothing written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_marker_without_a_usable_incident_or_detection_time_exits_3_naming_the_field() {
    for (case, field, value) in [
        ("no incident", MARKER_FIELD_INCIDENT, None),
        ("an empty incident", MARKER_FIELD_INCIDENT, Some("")),
        ("no detection time", MARKER_FIELD_DETECTED_AT_MS, None),
        (
            "an unreadable detection time",
            MARKER_FIELD_DETECTED_AT_MS,
            Some("soon"),
        ),
    ] {
        let tv = TestValkey::setup().await;
        hold_an_hour_ago(&tv).await;
        let marker = tv.key_schema().dr_hold_marker();
        match value {
            None => {
                let _: i64 = tv.client().pool.hdel(&marker, field).await.expect("HDEL");
            }
            Some(value) => {
                let _: i64 = tv
                    .client()
                    .pool
                    .hset(&marker, vec![(field, value)])
                    .await
                    .expect("HSET");
            }
        }
        let before = keyspace(&tv).await;
        let (body, log) = RecordingBody::new(Writes::One, &tv);

        let (exit, captured) = run(context(&tv), body).await;

        assert_eq!(exit, JobExit::StaysHeld, "{case}");
        assert_eq!(exit.code(), 3, "{case}");
        assert_one(&captured, "ERROR", &marker_incomplete_error(field));
        assert_eq!(
            lines(&captured, &marker_incomplete_error(field))[0]["fields"]["field"],
            field,
            "{case}"
        );
        assert_none_starting(&captured, "took the DR Job lease");
        assert!(!lease_exists(&tv).await, "{case}");
        assert_eq!(log.plans.load(Ordering::SeqCst), 0, "{case}");
        let writes = keyspace_writes(&before, &keyspace(&tv).await, WriteRules::STRICT);
        assert!(writes.is_empty(), "{case}: nothing written: {writes:?}");
    }
}

/// The precondition is read again under the lease. While the Job waits for a
/// lease another Job holds, the marker is removed, moved to another incident
/// or made unusable; once the lease is free the Job takes it, reads the
/// marker again, exits 0, 4 or 3 with the second read's line, never plans or
/// writes, and releases the lease.
///
/// The Job reads the store through a proxy that counts its lease attempts, so
/// the marker changes only after its first read and while it waits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_marker_changed_while_the_job_waits_for_its_lease_is_read_again_under_the_lease() {
    let other_incident = "20261016t0815-moved";
    for case in ["removed", "another incident", "no detection time"] {
        let tv = TestValkey::setup().await;
        hold_an_hour_ago(&tv).await;
        let lease_key = tv.key_schema().dr_job_lease();
        let other = LeaderIdentity::new("another-dr-job", uuid::Uuid::new_v4().to_string());
        assert!(
            try_acquire_leader_lock(tv.client(), &lease_key, &other.lock_value(), 60_000)
                .await
                .expect("the other Job's lease"),
            "{case}: fixture: another Job holds the lease"
        );
        let acquire_sha = LEADER_ACQUIRE_OR_REFRESH_SCRIPT.sha1().to_string();
        let watched_key = lease_key.clone();
        let proxy = ScriptReplyProxy::start(
            test_valkey_addr(),
            Arc::new(move |args: &[Vec<u8>]| {
                let is_acquire = args.first()?.eq_ignore_ascii_case(b"EVALSHA")
                    && args
                        .get(1)
                        .is_some_and(|a| a.as_slice() == acquire_sha.as_bytes())
                    && args.iter().any(|a| a.as_slice() == watched_key.as_bytes());
                is_acquire.then_some(ProxyAction::Delay(Duration::ZERO))
            }),
        )
        .await;
        let mut ctx = context(&tv);
        ctx.valkey = client_through(&proxy).await;
        ctx.timings.lease.acquire_wait = SETTLE;
        let (body, log) = RecordingBody::new(Writes::One, &tv);
        let job = tokio::spawn(run(ctx, body));

        let deadline = tokio::time::Instant::now() + SETTLE;
        while proxy.hits() == 0 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "{case}: the Job never tried the lease"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let marker = tv.key_schema().dr_hold_marker();
        let (want_exit, want_line, want_level) = match case {
            "removed" => {
                assert!(tv.remove_hold_marker().await);
                (
                    JobExit::NothingToDo,
                    "the store is not held (read again under the lease): nothing to do; the \
                     restore Job wrote nothing and exits 0"
                        .to_owned(),
                    "INFO",
                )
            }
            "another incident" => {
                let _: i64 = tv
                    .client()
                    .pool
                    .hset(&marker, vec![(MARKER_FIELD_INCIDENT, other_incident)])
                    .await
                    .expect("HSET");
                (
                    JobExit::DifferentIncident,
                    different_incident_error(other_incident),
                    "ERROR",
                )
            }
            _ => {
                let _: i64 = tv
                    .client()
                    .pool
                    .hdel(&marker, MARKER_FIELD_DETECTED_AT_MS)
                    .await
                    .expect("HDEL");
                (
                    JobExit::StaysHeld,
                    marker_incomplete_error(MARKER_FIELD_DETECTED_AT_MS),
                    "ERROR",
                )
            }
        };
        let _: i64 = tv.client().pool.del(&lease_key).await.expect("DEL");

        let (exit, captured) = job.await.expect("the Job task");

        assert_eq!(exit, want_exit, "{case}");
        assert_one(&captured, want_level, &want_line);
        assert_one(&captured, "INFO", "took the DR Job lease");
        assert_one(&captured, "INFO", "released the DR Job lease");
        assert!(!lease_exists(&tv).await, "{case}: the lease is released");
        assert_eq!(log.plans.load(Ordering::SeqCst), 0, "{case}: never plans");
        assert_eq!(log.executions.load(Ordering::SeqCst), 0, "{case}");
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// The adapter
// ═════════════════════════════════════════════════════════════════════════════

/// A store held an hour ago, with the `Disabled` adapter: the body plans
/// once, writes through the permit while the Job holds its lease, is handed
/// no backup store, and the backup store is never called; the lease is
/// released at the end.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_held_store_past_its_quiet_period_runs_the_body_under_the_lease() {
    let tv = TestValkey::setup().await;
    hold_an_hour_ago(&tv).await;
    let store = ScriptedStore::healthy();
    let (body, log) = RecordingBody::new(Writes::One, &tv);

    let ctx = context_with(
        &tv,
        JobMode::Restore,
        config(CheckpointAdapter::Disabled),
        Arc::clone(&store) as _,
    );
    let (exit, captured) = run(ctx, body).await;

    assert_eq!(exit, JobExit::Done);
    assert_eq!(log.plans.load(Ordering::SeqCst), 1);
    assert_eq!(log.executions.load(Ordering::SeqCst), 1);
    assert_eq!(log.lease_held_while_writing.load(Ordering::SeqCst), 1);
    assert_eq!(*log.backups.lock().unwrap(), vec!["none: disabled"]);
    assert_eq!(store.calls(), 0, "Disabled: no call of the backup store");
    assert!(!lease_exists(&tv).await, "the lease is released at exit");
    assert_one(
        &captured,
        "INFO",
        "the quiet period is over: the restore Job's write permit is open",
    );
    assert_one(&captured, "INFO", "the restore Job exits 0");
}

/// The `InMemory` adapter's store lives inside one server process, so a Job
/// can read nothing from it: a restore exits 3 naming that, before it takes
/// the lease; an accept-loss reaches its body with no backup store, past no
/// counters-record read and no canary. Neither calls the store.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_in_memory_adapter_keeps_a_restore_held_and_hands_accept_loss_no_backup_store() {
    let tv = TestValkey::setup().await;
    hold_an_hour_ago(&tv).await;

    let store = ScriptedStore::healthy();
    let (body, log) = RecordingBody::new(Writes::One, &tv);
    let ctx = context_with(
        &tv,
        JobMode::Restore,
        config(CheckpointAdapter::InMemory),
        Arc::clone(&store) as _,
    );
    let (exit, captured) = run(ctx, body).await;
    assert_eq!(exit, JobExit::StaysHeld);
    assert_eq!(exit.code(), 3);
    assert_one(
        &captured,
        "ERROR",
        "the restore Job wrote nothing and exits 3; the service stays held: the in-memory \
         backup store cannot be read by a restore Job; use accept-loss",
    );
    assert_none_starting(&captured, "took the DR Job lease");
    assert!(!lease_exists(&tv).await);
    assert_eq!(log.plans.load(Ordering::SeqCst), 0);
    assert_eq!(store.calls(), 0);

    let store = ScriptedStore::healthy();
    let (body, log) = RecordingBody::new(Writes::One, &tv);
    let ctx = context_with(
        &tv,
        JobMode::AcceptLoss,
        config(CheckpointAdapter::InMemory),
        Arc::clone(&store) as _,
    );
    let (exit, _captured) = run(ctx, body).await;
    assert_eq!(exit, JobExit::Done);
    assert_eq!(*log.backups.lock().unwrap(), vec!["none: in-memory"]);
    assert_eq!(
        store.calls(),
        0,
        "no counters-record read and no canary on InMemory"
    );
    assert!(!lease_exists(&tv).await);
}

// ═════════════════════════════════════════════════════════════════════════════
// The quiet period (T-R1c)
// ═════════════════════════════════════════════════════════════════════════════

/// A hold whose quiet period has [`QUIET_LEFT`] to run on the store's clock:
/// the Job takes its lease, the body plans once with the permit still paused,
/// the Job logs the wait left, and the body's first write unit starts only
/// once the store's clock has passed Q. The Job holds its lease while it
/// writes and releases it at the exit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_body_plans_during_the_quiet_period_and_writes_only_after_the_store_clock_passes_it() {
    let tv = TestValkey::setup().await;
    let config = config(CheckpointAdapter::Disabled);
    let q = QuietPeriod::from_config(&config);
    hold_with_quiet_left(&tv, &config, QUIET_LEFT).await;
    let (body, log) = RecordingBody::new(Writes::One, &tv);

    let ctx = context_with(
        &tv,
        JobMode::Restore,
        config,
        Arc::new(InMemoryCheckpointStore::new()),
    );
    let (exit, captured) = run(ctx, body).await;

    assert_eq!(exit, JobExit::Done);
    assert_eq!(log.plans.load(Ordering::SeqCst), 1, "the body plans once");
    assert_eq!(
        *log.plan_permit_states.lock().unwrap(),
        vec![PermitState::Paused],
        "it plans during the quiet period, before the permit opens"
    );
    let (unit_at, quiet_ends) = log
        .first_unit_at_ms
        .lock()
        .unwrap()
        .expect("a write unit started");
    assert!(
        unit_at >= quiet_ends,
        "the first write unit started at {unit_at} ms on the store's clock, before Q ended at \
         {quiet_ends} ms"
    );
    assert_eq!(log.lease_held_while_writing.load(Ordering::SeqCst), 1);
    assert!(!lease_exists(&tv).await, "the lease is released at exit");
    let waits: Vec<_> = captured
        .events
        .iter()
        .filter(|e| {
            e["fields"]["message"].as_str().is_some_and(|m| {
                m.starts_with("the restore Job is waiting out the quiet period: ")
                    && m.ends_with("s left on the store's clock before its first write")
            })
        })
        .collect();
    assert!(
        !waits.is_empty(),
        "the Job logs the wait left; capture:\n{}",
        captured.text
    );
    let start = lines(
        &captured,
        "the store is held under this incident; the restore Job writes nothing until the \
         store's clock passes the quiet period",
    );
    assert_eq!(start.len(), 1, "one quiet-period INFO:\n{}", captured.text);
    assert_eq!(
        start[0]["fields"]["shutdown_bound_secs"].as_f64(),
        Some(q.shutdown_bound.as_secs_f64()),
        "the line carries G"
    );
    assert_eq!(
        start[0]["fields"]["quiet_period_secs"].as_f64(),
        Some(q.total().as_secs_f64())
    );
    assert_eq!(
        start[0]["fields"]["quiet_ends_at_ms"].as_u64(),
        Some(quiet_ends)
    );
    assert_one(
        &captured,
        "INFO",
        "the quiet period is over: the restore Job's write permit is open",
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The lease
// ═════════════════════════════════════════════════════════════════════════════

/// A lease another Job holds for the whole acquire wait: the Job waits it
/// out, then exits 1 naming the holder, having written nothing; the other
/// Job's lease is left as it was.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_busy_lease_is_waited_on_then_the_job_exits_1_naming_its_holder() {
    let tv = TestValkey::setup().await;
    hold_an_hour_ago(&tv).await;
    let lease_key = tv.key_schema().dr_job_lease();
    let other = LeaderIdentity::new("another-dr-job", uuid::Uuid::new_v4().to_string());
    assert!(
        try_acquire_leader_lock(tv.client(), &lease_key, &other.lock_value(), 60_000)
            .await
            .expect("the other Job's lease")
    );
    let (body, log) = RecordingBody::new(Writes::One, &tv);
    let ctx = context(&tv);
    let acquire_wait = ctx.timings.lease.acquire_wait;

    let started = tokio::time::Instant::now();
    let (exit, captured) = run(ctx, body).await;

    assert_eq!(exit, JobExit::Failed);
    assert_eq!(exit.code(), 1);
    assert!(
        started.elapsed() >= acquire_wait,
        "the Job waited out its acquire wait before giving up"
    );
    assert_one(
        &captured,
        "INFO",
        "another DR Job holds the lease; waiting for it to be released or to lapse",
    );
    let busy = "another DR Job holds the lease: the restore Job wrote nothing and exits 1; \
                Kubernetes retries it";
    assert_one(&captured, "ERROR", busy);
    assert_eq!(
        lines(&captured, busy)[0]["fields"]["holder"],
        other.lock_value(),
        "the ERROR names the holder"
    );
    assert_eq!(log.plans.load(Ordering::SeqCst), 0);
    let holder: Option<String> = tv.client().pool.get(&lease_key).await.expect("GET");
    assert_eq!(
        holder,
        Some(other.lock_value()),
        "the other Job's lease is untouched"
    );
}

/// The lease lost during the quiet period (deleted under the Job): the Job
/// exits 1 having written nothing; the body never writes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lease_lost_during_the_quiet_period_exits_1_having_written_nothing() {
    let tv = TestValkey::setup().await;
    let config = config(CheckpointAdapter::Disabled);
    hold_with_quiet_left(&tv, &config, Duration::from_secs(600)).await;
    let (body, log) = RecordingBody::new(Writes::One, &tv);
    let ctx = context_with(
        &tv,
        JobMode::Restore,
        config,
        Arc::new(InMemoryCheckpointStore::new()),
    );
    let job = tokio::spawn(run(ctx, body));

    until_lease_taken(&tv).await;
    let _: i64 = tv
        .client()
        .pool
        .del(tv.key_schema().dr_job_lease())
        .await
        .expect("DEL");
    let (exit, captured) = job.await.expect("the Job task");

    assert_eq!(exit, JobExit::Failed);
    assert_one(
        &captured,
        "ERROR",
        "the restore Job lost its lease before writing; it wrote nothing and exits 1",
    );
    assert_eq!(log.executions.load(Ordering::SeqCst), 0, "no write");
    assert!(!lease_exists(&tv).await);
}

/// The lease lost while the body writes: the frame revokes the permit, drops
/// the body, and exits 1; the service stays held.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lease_lost_while_the_body_writes_revokes_the_permit_and_exits_1() {
    let tv = TestValkey::setup().await;
    hold_an_hour_ago(&tv).await;
    let (body, log) = RecordingBody::new(Writes::UntilStopped, &tv);
    let job = tokio::spawn(run(context(&tv), body));

    tokio::time::timeout(SETTLE, log.writing.notified())
        .await
        .expect("the body starts writing");
    let _: i64 = tv
        .client()
        .pool
        .del(tv.key_schema().dr_job_lease())
        .await
        .expect("DEL");
    let (exit, captured) = job.await.expect("the Job task");

    assert_eq!(exit, JobExit::Failed);
    assert_one(
        &captured,
        "ERROR",
        "the restore Job lost its lease, stopped writing and exits 1; the service stays held, \
         and a re-run continues from what it wrote",
    );
    assert_eq!(
        log.permit_state(),
        PermitState::Revoked(RevokeCause::LeaseLost(format!(
            "the lease {} no longer holds this Job's value",
            tv.key_schema().dr_job_lease()
        ))),
        "the permit the body writes under is revoked for the lost lease"
    );
    assert!(
        tv.client()
            .pool
            .exists::<i64, _>(tv.key_schema().dr_hold_marker())
            .await
            .expect("EXISTS")
            == 1,
        "the service stays held"
    );
}

/// SIGTERM during the quiet period: the Job exits 1 having written nothing
/// and releases its lease, so another Job takes it at once rather than after
/// its TTL.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sigterm_during_the_quiet_period_releases_the_lease_so_another_job_takes_it_at_once() {
    let tv = TestValkey::setup().await;
    let config = config(CheckpointAdapter::Disabled);
    hold_with_quiet_left(&tv, &config, Duration::from_secs(600)).await;
    let sigterm = CancellationToken::new();
    let (body, log) = RecordingBody::new(Writes::One, &tv);
    let mut ctx = context_with(
        &tv,
        JobMode::Restore,
        config,
        Arc::new(InMemoryCheckpointStore::new()),
    );
    ctx.shutdown = shutdown_on(&sigterm);
    let job = tokio::spawn(run(ctx, body));

    until_lease_taken(&tv).await;
    sigterm.cancel();
    let (exit, captured) = job.await.expect("the Job task");

    assert_eq!(exit, JobExit::Failed);
    assert_one(
        &captured,
        "INFO",
        "the restore Job received SIGTERM before writing; it wrote nothing, releases its lease \
         and exits 1",
    );
    assert_one(&captured, "INFO", "released the DR Job lease");
    assert_eq!(log.executions.load(Ordering::SeqCst), 0);
    assert!(!lease_exists(&tv).await, "released, not left to lapse");
    let next = LeaseTimings {
        acquire_wait: Duration::ZERO,
        ..timings().lease
    };
    let lease = JobLease::acquire(
        tv.client(),
        tv.key_schema(),
        LeaderIdentity::new("the-next-dr-job", uuid::Uuid::new_v4().to_string()),
        next,
        &CancellationToken::new(),
    )
    .await
    .expect("another Job takes the lease at its first attempt");
    lease.release().await;
}

/// SIGTERM while the body writes: the frame revokes the permit, drops the
/// body, releases the lease and exits 1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sigterm_while_the_body_writes_revokes_the_permit_and_releases_the_lease() {
    let tv = TestValkey::setup().await;
    hold_an_hour_ago(&tv).await;
    let sigterm = CancellationToken::new();
    let (body, log) = RecordingBody::new(Writes::UntilStopped, &tv);
    let mut ctx = context(&tv);
    ctx.shutdown = shutdown_on(&sigterm);
    let job = tokio::spawn(run(ctx, body));

    tokio::time::timeout(SETTLE, log.writing.notified())
        .await
        .expect("the body starts writing");
    sigterm.cancel();
    let (exit, captured) = job.await.expect("the Job task");

    assert_eq!(exit, JobExit::Failed);
    assert_one(
        &captured,
        "INFO",
        "the restore Job received SIGTERM: it stopped writing, releases its lease and exits 1; \
         the service stays held, and a re-run continues from what it wrote",
    );
    assert_eq!(
        log.permit_state(),
        PermitState::Revoked(RevokeCause::Shutdown),
        "the permit the body writes under is revoked for the shutdown"
    );
    assert_one(&captured, "INFO", "released the DR Job lease");
    assert!(!lease_exists(&tv).await);
}

// ═════════════════════════════════════════════════════════════════════════════
// The absence rule (persistent backup store)
// ═════════════════════════════════════════════════════════════════════════════

/// The wait line a Job logs while it waits for `T_safe` to read `what` again.
fn absence_wait_suffix(what: &str) -> String {
    format!(
        "{what} read as absent before T_safe, when the backup store's not-found answers can \
         still be stale; reading it again then"
    )
}

fn waits_for(captured: &Captured, what: &str) -> usize {
    let suffix = absence_wait_suffix(what);
    captured
        .events
        .iter()
        .filter(|e| {
            e["fields"]["message"].as_str().is_some_and(|m| {
                m.starts_with("waiting ")
                    && m.contains("s on the store's clock: ")
                    && m.ends_with(&suffix)
            })
        })
        .count()
}

const SIDECAR_WHAT: &str = "the backup store's counters record";
const CANARY_WHAT: &str = "the backup store's listing route check (the list-route canary)";

/// A counters record absent before `T_safe` is read again at `T_safe`;
/// absent there too, the Job exits 3 with the named cause. Nothing past it
/// runs: no canary, no body write.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_counters_record_absent_at_t_safe_keeps_the_service_held_with_exit_3() {
    let tv = TestValkey::setup().await;
    let config = storage_api_config();
    hold_with_quiet_left(&tv, &config, Duration::ZERO).await;
    let store = ScriptedStore::new(&[false], &[Canary::Passes]);
    let (body, log) = RecordingBody::new(Writes::One, &tv);

    let ctx = context_with(&tv, JobMode::Restore, config, Arc::clone(&store) as _);
    let (exit, captured) = run(ctx, body).await;

    assert_eq!(exit, JobExit::StaysHeld);
    assert_eq!(exit.code(), 3);
    assert_one(
        &captured,
        "ERROR",
        &format!(
            "the restore Job wrote nothing and exits 3; the service stays held: \
             {SIDECAR_ABSENT_CAUSE}"
        ),
    );
    assert!(
        waits_for(&captured, SIDECAR_WHAT) >= 1,
        "the first absence before T_safe waits for T_safe; capture:\n{}",
        captured.text
    );
    assert_eq!(store.sidecar_reads(), 2, "read before T_safe, then at it");
    assert_eq!(store.canary_runs(), 0, "no canary after an absent record");
    assert_eq!(log.executions.load(Ordering::SeqCst), 0, "no body write");
    assert!(!lease_exists(&tv).await);
}

/// A counters record absent before `T_safe` and present at it: the Job waits
/// for `T_safe`, logging why, reads it again, and goes on to the canary and
/// the body.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_counters_record_absent_before_t_safe_and_present_at_it_lets_the_run_go_on() {
    let tv = TestValkey::setup().await;
    let config = storage_api_config();
    hold_with_quiet_left(&tv, &config, Duration::ZERO).await;
    let store = ScriptedStore::new(&[false, true], &[Canary::Passes]);
    let (body, log) = RecordingBody::new(Writes::One, &tv);

    let ctx = context_with(&tv, JobMode::Restore, config, Arc::clone(&store) as _);
    let (exit, captured) = run(ctx, body).await;

    assert_eq!(exit, JobExit::Done);
    assert!(
        waits_for(&captured, SIDECAR_WHAT) >= 1,
        "the wait for T_safe is logged; capture:\n{}",
        captured.text
    );
    assert_eq!(store.sidecar_reads(), 2);
    assert_eq!(store.canary_runs(), 1);
    assert_eq!(log.executions.load(Ordering::SeqCst), 1);
    assert_eq!(*log.backups.lock().unwrap(), vec!["store"]);
}

/// A list-route canary that fails before `T_safe` is run again at `T_safe`;
/// failing there too, the Job exits 3 with the named cause, and the body
/// never runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_list_route_canary_failing_at_t_safe_keeps_the_service_held_with_exit_3() {
    let tv = TestValkey::setup().await;
    let config = storage_api_config();
    hold_with_quiet_left(&tv, &config, Duration::ZERO).await;
    let store = ScriptedStore::new(&[true], &[Canary::NotListed]);
    let (body, log) = RecordingBody::new(Writes::One, &tv);

    let ctx = context_with(&tv, JobMode::Restore, config, Arc::clone(&store) as _);
    let (exit, captured) = run(ctx, body).await;

    assert_eq!(exit, JobExit::StaysHeld);
    assert_eq!(exit.code(), 3);
    assert_one(
        &captured,
        "ERROR",
        &format!("the restore Job exits 3 and the service stays held: {LIST_ROUTE_CAUSE}"),
    );
    assert!(waits_for(&captured, CANARY_WHAT) >= 1, "{}", captured.text);
    assert_eq!(store.canary_runs(), 2, "run before T_safe, then at it");
    assert_eq!(log.executions.load(Ordering::SeqCst), 0, "no body write");
}

/// A list-route canary that fails before `T_safe` and passes at it: the run
/// goes on to the body.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_list_route_canary_failing_before_t_safe_and_passing_at_it_lets_the_run_go_on() {
    let tv = TestValkey::setup().await;
    let config = storage_api_config();
    hold_with_quiet_left(&tv, &config, Duration::ZERO).await;
    let store = ScriptedStore::new(&[true], &[Canary::NotListed, Canary::Passes]);
    let (body, log) = RecordingBody::new(Writes::One, &tv);

    let ctx = context_with(&tv, JobMode::Restore, config, Arc::clone(&store) as _);
    let (exit, captured) = run(ctx, body).await;

    assert_eq!(exit, JobExit::Done);
    assert!(waits_for(&captured, CANARY_WHAT) >= 1, "{}", captured.text);
    assert_eq!(store.canary_runs(), 2);
    assert_eq!(log.executions.load(Ordering::SeqCst), 1);
    assert_one(
        &captured,
        "INFO",
        "the backup store's listing route listed back the list-route canary",
    );
}

/// A canary that meets a backend fault is no failed canary: the Job exits 1
/// (Kubernetes retries it) without waiting for `T_safe`, and the body never
/// runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_list_route_canary_backend_fault_exits_1() {
    let tv = TestValkey::setup().await;
    let config = storage_api_config();
    hold_with_quiet_left(&tv, &config, Duration::ZERO).await;
    let store = ScriptedStore::new(&[true], &[Canary::Fault]);
    let (body, log) = RecordingBody::new(Writes::One, &tv);

    let ctx = context_with(&tv, JobMode::Restore, config, Arc::clone(&store) as _);
    let (exit, captured) = run(ctx, body).await;

    assert_eq!(exit, JobExit::Failed);
    assert_one(
        &captured,
        "ERROR",
        "the restore Job's list-route canary failed with a backup-store fault; it exits 1 and \
         the service stays held",
    );
    assert_eq!(store.canary_runs(), 1);
    assert_eq!(waits_for(&captured, CANARY_WHAT), 0);
    assert_eq!(log.executions.load(Ordering::SeqCst), 0);
}

/// The contradiction check through the absence rule's primitive
/// (`observe_under_absence_rule`, which Task 7 calls for it): a hit — a
/// stamped surviving object the listing does not show — observed before
/// `T_safe` makes the caller wait for `T_safe` and observe again, and the
/// second observation decides: gone there, the run goes on; still there, it
/// is acted on, naming the object. A hit observed at or after `T_safe` is
/// acted on with one observation, and no hit before `T_safe` needs no
/// second one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_contradiction_seen_before_t_safe_is_checked_again_at_t_safe_and_the_second_look_decides()
{
    use std::collections::HashSet;

    use live_edit::dr::job::absence::{
        find_contradictions, observe_under_absence_rule, Contradiction, StepError, StoreClock,
    };
    use live_edit::dr::job::quiet::StoreDeadlines;
    use live_edit::storage::dr_job::CheckpointStamp;

    let tv = TestValkey::setup().await;
    let stamp = CheckpointStamp {
        object_id: 7,
        session_id: "s".to_owned(),
        layer_name: "l".to_owned(),
        last_checkpoint_seq: Some(3),
        last_checkpoint_ms: None,
    };
    for (case, t_safe_ahead_ms, listed_on, want_calls, want_hit, want_after) in [
        ("hit, then gone at T_safe", 2_000, Some(2), 2, false, true),
        ("hit, still there at T_safe", 2_000, None, 2, true, true),
        ("hit at or after T_safe", 0, None, 1, true, true),
        ("no hit before T_safe", 2_000, Some(1), 1, false, false),
    ] {
        let now = now_ms(&tv).await;
        let t_safe_ms = now + t_safe_ahead_ms;
        let clock = StoreClock::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            StoreDeadlines {
                detected_at_ms: now - HOUR_MS,
                quiet_ends_at_ms: now,
                t_safe_ms,
            },
            Duration::from_millis(500),
        );
        let calls = AtomicUsize::new(0);
        let (observed, captured) = capture(observe_under_absence_rule::<_, StepError, _, _>(
            &clock,
            "the listing's coverage of the surviving objects",
            || {
                // The listing shows the object from the `listed_on`th
                // observation on.
                let call = calls.fetch_add(1, Ordering::SeqCst) + 1;
                let listed: HashSet<u64> = match listed_on {
                    Some(from) if call >= from => HashSet::from([7]),
                    _ => HashSet::new(),
                };
                let found = find_contradictions(
                    std::slice::from_ref(&stamp),
                    &[],
                    &HashSet::new(),
                    &listed,
                    &HashSet::new(),
                );
                async move { Ok::<_, StepError>(found) }
            },
            |found: &Vec<Contradiction>| !found.is_empty(),
        ))
        .await;
        let observed = observed.expect("the store's clock answers");

        assert_eq!(
            calls.load(Ordering::SeqCst),
            want_calls,
            "{case}: observations"
        );
        assert_eq!(
            !observed.value.is_empty(),
            want_hit,
            "{case}: the deciding look"
        );
        assert_eq!(observed.at_or_after_t_safe, want_after, "{case}");
        if want_hit {
            assert!(
                observed.value[0].to_string().starts_with("object 7 "),
                "{case}: the contradiction names the object: {}",
                observed.value[0]
            );
        }
        if want_calls == 2 {
            assert!(
                now_ms(&tv).await >= t_safe_ms,
                "{case}: the second look is at or after T_safe"
            );
            assert!(
                waits_for(&captured, "the listing's coverage of the surviving objects") >= 1,
                "{case}: the wait is logged:\n{}",
                captured.text
            );
        }
    }
}

/// A storage-api backup store whose listing route answers 404 while its
/// by-address routes work (the counters record reads present): the canary it
/// writes is not listed back, so the Job exits 3 with the named cause. The
/// canary is deleted again, so the backup store holds what it held before,
/// and the store's keyspace is unchanged: nothing is written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_storage_api_listing_route_answering_404_keeps_the_service_held_with_nothing_written() {
    let tv = TestValkey::setup().await;
    let mut config = config(CheckpointAdapter::StorageApi);
    config.checkpoint_storage_api_not_found_cache_secs = 0;
    hold_an_hour_ago(&tv).await;

    let server = MockServer::start().await;
    let state: FakeStorageState = Arc::new(Mutex::new(std::collections::HashMap::new()));
    mount_fake_storage_routes(&server, &state).await;
    Mock::given(method("GET"))
        .and(path_regex(r"^/v1alpha/fileobject/data-objects/"))
        .respond_with(ResponseTemplate::new(404))
        .with_priority(1)
        .mount(&server)
        .await;
    let (adapter, _token_dir) = build_fake_backed_adapter(&server, "dr-job-frame-canary");
    adapter
        .put_counters_sidecar(Bytes::from_static(
            b"{\"max_object_id_observed\":0,\"updated_at\":\"2026-10-20T12:00:00Z\"}",
        ))
        .await
        .expect("fixture: the counters record");
    let backup_before = state.lock().unwrap().clone();
    let store_before = keyspace(&tv).await;
    let (body, log) = RecordingBody::new(Writes::One, &tv);

    let ctx = context_with(&tv, JobMode::Restore, config, Arc::new(adapter));
    let (exit, captured) = run(ctx, body).await;

    assert_eq!(exit, JobExit::StaysHeld);
    assert_one(
        &captured,
        "ERROR",
        &format!("the restore Job exits 3 and the service stays held: {LIST_ROUTE_CAUSE}"),
    );
    assert_eq!(log.executions.load(Ordering::SeqCst), 0, "no body write");
    let canary_puts = server
        .received_requests()
        .await
        .expect("wiremock records requests")
        .into_iter()
        .filter(|r| r.method.as_str() == "PUT" && r.url.path().contains("dr-canary"))
        .count();
    assert_eq!(canary_puts, 1, "the canary ran, and its listing failed it");
    assert_eq!(
        *state.lock().unwrap(),
        backup_before,
        "the backup store holds what it held before: the canary is gone again"
    );
    assert!(
        backup_before
            .keys()
            .all(|k| k.starts_with(&format!("{BASE_PATH}/.live-edit/"))),
        "fixture: only the counters record"
    );
    let writes = keyspace_writes(&store_before, &keyspace(&tv).await, WriteRules::STRICT);
    assert!(
        writes.is_empty(),
        "nothing written to the store: {writes:?}"
    );
}
