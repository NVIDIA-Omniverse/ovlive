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

//! The restore and accept-loss Jobs' tests' shared scaffolding. The restore
//! writer's tests (`restore_write_test`, `restore_sessions_test`,
//! `restore_memory_test`), the finish's (`restore_finish_test`) and the
//! accept-loss Job's (`accept_loss_test`, which also uses the backup store's
//! time-switched faults) use all of it: a backup store the tests script and
//! count, backup and store fixtures, the Job run to its exit with the
//! writer's test seams, and a restored object read back. The planner's tests (`restore_plan_test`)
//! share the fixtures both write the same way — a backup generation's meta
//! and put, the counters record, a surviving header, a deletion marker, a
//! dead object — and the Job's context, the kill line and the log reader.
//! The fixtures that differ stay with their tests: a writer test's
//! mapping carries its layers-index member and its session backup the
//! path-index members ListSessions reads, as the service writes them, where
//! the planner's tests seed only what each planning rule reads.
//!
//! # Time
//!
//! No assertion bounds how long anything took; every bound here is a hang
//! bound. [`RUN_BUDGET`] (30 s) bounds one Job run or one plan, and gives a
//! hung run a message naming it; a healthy run of these fixtures takes a few
//! seconds (the lease, a quiet period already over, the plan, the writes).
//! Each writer test's whole body — its fixture, every run or plan it makes,
//! its read-backs and its RPCs — runs under one
//! [`TEST_BUDGET`] (90 s) through [`within_test_budget`], so its worst case
//! is 90 s plus the `TestValkey` clean-up, inside nextest's 120 s kill
//! whatever the run count. W2 has its own budget and nextest override
//! (`restore_write_test`). The planner's tests bound their runs themselves
//! (`restore_plan_test`, *Time*).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::{
    HashesInterface, KeysInterface, SetsInterface, SortedSetsInterface, StreamsInterface,
};

use live_edit::checkpoint::adapters::in_memory::InMemoryCheckpointStore;
use live_edit::checkpoint::seq_band::flatten_to_seeded_seq;
use live_edit::checkpoint::{
    encode_tail_envelope, CheckpointAdapter, CheckpointBlob, CheckpointError, CheckpointHandle,
    CheckpointKind, CheckpointMeta, CheckpointStore, LayerArtifact, LiveExtSet, SessionCheckpoint,
    TailHeader, TailRecord,
};
use live_edit::config::Config;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::ObjectState;
use live_edit::dr::job::absence::{OwnPuts, StoreClock};
use live_edit::dr::job::execute::{ExecutionReport, KillPoint};
use live_edit::dr::job::lease::LeaseTimings;
use live_edit::dr::job::permit::WritePermit;
use live_edit::dr::job::plan::lineage::{decode_planned, DecodedLineage};
use live_edit::dr::job::plan::model::{ObjectAction, RestorePlan};
use live_edit::dr::job::quiet::{QuietPeriod, StoreDeadlines};
use live_edit::dr::job::{
    run_job, BackupAccess, JobArgs, JobContext, JobExit, JobMode, JobRun, JobTimings, RestoreBody,
};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::allocator_purge::AllocatorPurge;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::diff_read::{read_object_state_for_test, snapshot_to_object_state};
use live_edit::storage::dr_job::store_time_ms;
use live_edit::storage::external_values::ext_content_hash;
use live_edit::storage::keys::{hex_encode, sha256_to_hex, KeySchema};
use live_edit::storage::leader_lease::LeaderIdentity;
use live_edit::storage::test_support::{HoldMarkerSeed, TestValkey};
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand, TimeSample};

use crate::grpc::log_capture::{capture, Captured};

/// The incident every hold of the restore Job's tests is under.
pub(crate) const INCIDENT: &str = "20261020t1200-restore";
/// Hang bound on one Job run or one plan (module docs, *Time*);
/// `restore_plan_test` also bounds a test's several runs together by it.
pub(crate) const RUN_BUDGET: Duration = Duration::from_secs(30);
/// Hang bound on one writer test's whole body (module docs, *Time*).
pub(crate) const TEST_BUDGET: Duration = Duration::from_secs(90);
/// The band every fixture here gets: the test configuration's
/// `checkpoint_seq_safety_margin`, since no backup records a commit span.
pub(crate) const BAND: u64 = 1_000;
/// How the INFO of a test kill point starts.
pub(crate) const KILL_LINE: &str = "the restore Job stopped at its test kill point";
/// How the ERROR of a run that stays held after its recount starts.
pub(crate) const STAYS_HELD_LINE: &str =
    "the restore Job restored everything it could and recounted the references of the large values";
/// The `SdfSpecType` values clients write: a prim, an attribute.
pub(crate) const SPEC_PRIM: u8 = 6;
pub(crate) const SPEC_ATTRIBUTE: u8 = 1;
/// An hour, in milliseconds.
pub(crate) const HOUR_MS: u64 = 3_600_000;

/// Run one writer test's `body` within [`TEST_BUDGET`].
pub(crate) async fn within_test_budget<F: Future>(body: F) -> F::Output {
    tokio::time::timeout(TEST_BUDGET, body)
        .await
        .expect("the test's body ends within its budget")
}

// ═════════════════════════════════════════════════════════════════════════════
// The backup store
// ═════════════════════════════════════════════════════════════════════════════

/// How [`WriterStore`] holds each generation fetch before it answers.
#[derive(Clone, Copy, Debug)]
enum Pacing {
    /// Until `target` fetches are in flight at once, or `wait` passed
    /// ([`WriterStore::pace_fetches`]).
    UntilInFlight { target: usize, wait: Duration },
    /// Until the fetch's group of `size` consecutive arrivals is complete,
    /// or `wait` passed ([`WriterStore::pace_fetches_in_groups`]).
    Groups { size: usize, wait: Duration },
}

/// A backup store over an in-memory one: a generation's fetch can be scripted
/// to read absent after a number of fetches, or to fail `Permanent` always;
/// every external-value PUT and probe is recorded, and so is the most
/// generation fetches ever in flight at once and, for each fetch from the
/// pacing on, how many were in flight when it arrived. Fetches can be paced
/// so that concurrent fetches overlap ([`Self::pace_fetches`]) or answer
/// together ([`Self::pace_fetches_in_groups`]). Every call of any method is
/// counted ([`Self::calls`]).
///
/// With the store's clock switched on ([`Self::use_the_store_clock`]), the
/// accept-loss tests' time-switched faults, lifted from `restore_plan_test`'s
/// `PlanStore`: the layer listing answers empty until a time on the store's
/// clock ([`Self::list_nothing_until`]), a value's probe answers "absent"
/// until such a time ([`Self::value_absent_until`]), and a session's backup
/// is neither listed nor read until such a time
/// ([`Self::hide_session_until`]); each switch panics when the clock is off,
/// so a test cannot arm one that never fires. The store time of the first
/// call of every switched method is recorded ([`Self::first_calls_at_ms`]).
/// Store time, not wall time, decides each switch, as it decides the Job's
/// `T_safe`. A value's probe can also fail `Transient` until the fault is
/// cleared ([`Self::fail_probes_of`], [`Self::clear_probe_faults`]), and so
/// can a session backup's read ([`Self::fail_session_reads_of`]).
pub(crate) struct WriterStore {
    inner: InMemoryCheckpointStore,
    /// `(object, seq)` → how many fetches still read it before it reads
    /// absent.
    absent_after: Mutex<HashMap<(u64, u64), usize>>,
    /// `(object, seq)` whose fetch fails `Permanent`.
    permanent: Mutex<HashSet<(u64, u64)>>,
    /// Every `put_ext_blob`, by hash.
    pub(crate) ext_puts: Mutex<Vec<String>>,
    /// Every `ext_blob_exists`, by hash.
    pub(crate) probes: Mutex<Vec<String>>,
    /// Every generation fetch, `(object, seq)`.
    pub(crate) gets: Mutex<Vec<(u64, u64)>>,
    /// Every session backup deleted.
    pub(crate) session_deletes: AtomicUsize,
    /// Generation fetches in flight now, and the most at once.
    fetches_in_flight: AtomicUsize,
    peak_fetches: AtomicUsize,
    /// Per generation fetch since [`Self::pace_fetches`], in arrival order:
    /// the fetches in flight as it arrived, itself included.
    fetch_arrivals: Mutex<Vec<usize>>,
    /// How each fetch is held before it answers, if at all.
    pacing: Mutex<Option<Pacing>>,
    /// Woken on each fetch's arrival.
    fetch_arrived: tokio::sync::Notify,
    /// Every call of any method.
    calls: AtomicUsize,
    /// The store whose clock the time-switched faults read.
    clock: Mutex<Option<(ValkeyClient, KeySchema)>>,
    /// The layer listing answers empty until this store time.
    listing_empty_until: Mutex<Option<u64>>,
    /// Values whose probe answers `false` until the store time.
    ext_absent_until: Mutex<HashMap<String, u64>>,
    /// Values whose probe fails `Transient`.
    probe_failing: Mutex<HashSet<String>>,
    /// The store time of the first layer listing, with the clock on.
    first_listing_at_ms: Mutex<Option<u64>>,
    /// The store time of the first probe, with the clock on.
    first_probe_at_ms: Mutex<Option<u64>>,
    /// Sessions whose backup is neither listed nor read until the store
    /// time.
    sessions_hidden_until: Mutex<HashMap<String, u64>>,
    /// Sessions whose backup's read fails `Transient`.
    session_reads_failing: Mutex<HashSet<String>>,
    /// The store time of the first call of each switched method, by name,
    /// with the clock on.
    first_calls_at_ms: Mutex<BTreeMap<&'static str, u64>>,
}

impl WriterStore {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: InMemoryCheckpointStore::new(),
            absent_after: Mutex::new(HashMap::new()),
            permanent: Mutex::new(HashSet::new()),
            ext_puts: Mutex::new(Vec::new()),
            probes: Mutex::new(Vec::new()),
            gets: Mutex::new(Vec::new()),
            session_deletes: AtomicUsize::new(0),
            fetches_in_flight: AtomicUsize::new(0),
            peak_fetches: AtomicUsize::new(0),
            fetch_arrivals: Mutex::new(Vec::new()),
            pacing: Mutex::new(None),
            fetch_arrived: tokio::sync::Notify::new(),
            calls: AtomicUsize::new(0),
            clock: Mutex::new(None),
            listing_empty_until: Mutex::new(None),
            ext_absent_until: Mutex::new(HashMap::new()),
            probe_failing: Mutex::new(HashSet::new()),
            first_listing_at_ms: Mutex::new(None),
            first_probe_at_ms: Mutex::new(None),
            sessions_hidden_until: Mutex::new(HashMap::new()),
            session_reads_failing: Mutex::new(HashSet::new()),
            first_calls_at_ms: Mutex::new(BTreeMap::new()),
        })
    }

    /// How many calls, of any method, the store has had.
    pub(crate) fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn call(&self) {
        self.calls.fetch_add(1, Ordering::SeqCst);
    }

    /// Read `tv`'s store clock for the time-switched faults and the first
    /// listing's and first probe's times.
    pub(crate) fn use_the_store_clock(&self, tv: &TestValkey) {
        *self.clock.lock().unwrap() = Some((tv.client().clone(), tv.key_schema().clone()));
    }

    /// The store's clock, when switched on.
    async fn store_now(&self) -> Option<u64> {
        let clock = self.clock.lock().unwrap().clone();
        match clock {
            Some((client, keys)) => Some(store_time_ms(&client, &keys).await.expect("TIME")),
            None => None,
        }
    }

    /// The store's clock for switched method `method`, recording the time
    /// of its first call; `None` with the clock off.
    async fn switched_call(&self, method: &'static str) -> Option<u64> {
        let now = self.store_now().await?;
        self.first_calls_at_ms
            .lock()
            .unwrap()
            .entry(method)
            .or_insert(now);
        Some(now)
    }

    /// Panic unless the store's clock is on: a time switch armed without it
    /// would never fire, and a test relying on it would pass vacuously.
    fn require_the_clock(&self, switch: &str) {
        assert!(
            self.clock.lock().unwrap().is_some(),
            "{switch} needs the store's clock: call use_the_store_clock first"
        );
    }

    /// The layer listing answers empty until the store's clock reaches
    /// `until_ms`. Panics with the clock off.
    pub(crate) fn list_nothing_until(&self, until_ms: u64) {
        self.require_the_clock("list_nothing_until");
        *self.listing_empty_until.lock().unwrap() = Some(until_ms);
    }

    /// `hash`'s probe answers `false` until the store's clock reaches
    /// `until_ms`. Panics with the clock off.
    pub(crate) fn value_absent_until(&self, hash: &[u8; 32], until_ms: u64) {
        self.require_the_clock("value_absent_until");
        self.ext_absent_until
            .lock()
            .unwrap()
            .insert(sha256_to_hex(hash), until_ms);
    }

    /// `session`'s backup is neither listed nor read (it reads absent) until
    /// the store's clock reaches `until_ms`, as a listing that does not show
    /// it yet answers. Panics with the clock off.
    pub(crate) fn hide_session_until(&self, session: &str, until_ms: u64) {
        self.require_the_clock("hide_session_until");
        self.sessions_hidden_until
            .lock()
            .unwrap()
            .insert(session.to_owned(), until_ms);
    }

    /// Whether `session` is hidden at store time `now`.
    fn session_hidden(&self, session: &str, now: Option<u64>) -> bool {
        let until = self
            .sessions_hidden_until
            .lock()
            .unwrap()
            .get(session)
            .copied();
        matches!((until, now), (Some(until), Some(now)) if now < until)
    }

    /// Every read of `session`'s backup fails `Transient`, as a backup-store
    /// brownout answers; its listing is unchanged.
    pub(crate) fn fail_session_reads_of(&self, session: &str) {
        self.session_reads_failing
            .lock()
            .unwrap()
            .insert(session.to_owned());
    }

    /// The store time of the first call of every switched method that was
    /// called, with the clock on: `list_layer_artifacts`,
    /// `list_session_checkpoint_ids`, `get_session_checkpoint` and
    /// `ext_blob_exists`.
    pub(crate) fn first_calls_at_ms(&self) -> BTreeMap<&'static str, u64> {
        self.first_calls_at_ms.lock().unwrap().clone()
    }

    /// Every probe of `hash` fails `Transient`, as a backup-store brownout
    /// answers, until [`Self::clear_probe_faults`].
    pub(crate) fn fail_probes_of(&self, hash: &[u8; 32]) {
        self.probe_failing
            .lock()
            .unwrap()
            .insert(sha256_to_hex(hash));
    }

    /// End every [`Self::fail_probes_of`] fault.
    pub(crate) fn clear_probe_faults(&self) {
        self.probe_failing.lock().unwrap().clear();
    }

    /// The store time of the first layer listing, with the clock on.
    pub(crate) fn first_listing_at_ms(&self) -> Option<u64> {
        *self.first_listing_at_ms.lock().unwrap()
    }

    /// The store time of the first probe, with the clock on.
    pub(crate) fn first_probe_at_ms(&self) -> Option<u64> {
        *self.first_probe_at_ms.lock().unwrap()
    }

    /// From now on, hold each generation fetch until `target` fetches are in
    /// flight at once or `wait` passed, whichever is first, and count the
    /// most in flight at once, and the arrivals, from zero. A caller fetching
    /// at most `target - 1` at once therefore has that many overlap (the
    /// later ones arrive within `wait`), and one fetching `target` or more
    /// reaches `target`.
    pub(crate) fn pace_fetches(&self, target: usize, wait: Duration) {
        self.start_pacing(Pacing::UntilInFlight { target, wait });
    }

    /// From now on, release generation fetches in groups of `size`
    /// consecutive arrivals: each is held until its group is complete or
    /// `wait` passed, and the group's fetches answer together. Counts the
    /// most in flight at once, and the arrivals, from zero, as
    /// [`Self::pace_fetches`] does. A caller running `size` fetches at once
    /// therefore has them answer in one moment, so what it does next with
    /// each overlaps.
    pub(crate) fn pace_fetches_in_groups(&self, size: usize, wait: Duration) {
        self.start_pacing(Pacing::Groups { size, wait });
    }

    fn start_pacing(&self, pacing: Pacing) {
        *self.pacing.lock().unwrap() = Some(pacing);
        self.peak_fetches.store(0, Ordering::SeqCst);
        self.fetch_arrivals.lock().unwrap().clear();
    }

    /// The most generation fetches in flight at once.
    pub(crate) fn peak_fetches(&self) -> usize {
        self.peak_fetches.load(Ordering::SeqCst)
    }

    /// For each generation fetch since [`Self::pace_fetches`], in arrival
    /// order, the fetches in flight as it arrived (itself included). The
    /// most over a run of consecutive fetches is the most of them in flight
    /// at once.
    pub(crate) fn fetch_arrivals(&self) -> Vec<usize> {
        self.fetch_arrivals.lock().unwrap().clone()
    }

    /// Hold the fetch that arrived `index`-th since the pacing began as the
    /// pacing says; it is already counted in flight.
    async fn paced(&self, index: usize) {
        let Some(pacing) = *self.pacing.lock().unwrap() else {
            return;
        };
        let wait = match pacing {
            Pacing::UntilInFlight { wait, .. } | Pacing::Groups { wait, .. } => wait,
        };
        let deadline = tokio::time::Instant::now() + wait;
        loop {
            let arrival = self.fetch_arrived.notified();
            tokio::pin!(arrival);
            arrival.as_mut().enable();
            let released = match pacing {
                Pacing::UntilInFlight { target, .. } => {
                    self.fetches_in_flight.load(Ordering::SeqCst) >= target
                }
                Pacing::Groups { size, .. } => {
                    self.fetch_arrivals.lock().unwrap().len() >= (index / size + 1) * size
                }
            };
            if released {
                return;
            }
            if tokio::time::timeout_at(deadline, arrival).await.is_err() {
                return;
            }
        }
    }

    /// One generation fetch, its answer as scripted.
    async fn fetch(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        let key = (handle.object_id, handle.seq());
        self.gets.lock().unwrap().push(key);
        if self.permanent.lock().unwrap().contains(&key) {
            return Err(CheckpointError::Permanent {
                source: "the backup store answered 403".into(),
            });
        }
        {
            let mut absent = self.absent_after.lock().unwrap();
            if let Some(left) = absent.get_mut(&key) {
                if *left == 0 {
                    return Ok(None);
                }
                *left -= 1;
            }
        }
        self.inner.get_layer_checkpoint(handle).await
    }

    /// Generation `seq` of `object` reads as it is `fetches` times, then
    /// absent.
    pub(crate) fn absent_after(&self, object: u64, seq: u64, fetches: usize) {
        self.absent_after
            .lock()
            .unwrap()
            .insert((object, seq), fetches);
    }

    /// Every fetch of generation `seq` of `object` fails `Permanent`.
    pub(crate) fn permanent(&self, object: u64, seq: u64) {
        self.permanent.lock().unwrap().insert((object, seq));
    }

    pub(crate) fn puts_of(&self, hash: &[u8; 32]) -> usize {
        let hex = sha256_to_hex(hash);
        self.ext_puts
            .lock()
            .unwrap()
            .iter()
            .filter(|h| **h == hex)
            .count()
    }

    pub(crate) fn probes_of(&self, hash: &[u8; 32]) -> usize {
        let hex = sha256_to_hex(hash);
        self.probes
            .lock()
            .unwrap()
            .iter()
            .filter(|h| **h == hex)
            .count()
    }

    pub(crate) fn gets_of(&self, object: u64, seq: u64) -> usize {
        self.gets
            .lock()
            .unwrap()
            .iter()
            .filter(|g| **g == (object, seq))
            .count()
    }

    /// Whether the backup store holds `hash`'s blob, read past the probe
    /// record.
    pub(crate) async fn holds_value(&self, hash: &[u8; 32]) -> bool {
        self.inner
            .ext_blob_exists(&sha256_to_hex(hash))
            .await
            .expect("the in-memory store answers")
    }

    /// Whether the backup store holds `session`'s backup.
    pub(crate) async fn holds_session(&self, session: &str) -> bool {
        self.inner
            .get_session_checkpoint(session)
            .await
            .expect("the in-memory store answers")
            .is_some()
    }
}

#[async_trait]
impl CheckpointStore for WriterStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.call();
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
        let now = self.fetches_in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak_fetches.fetch_max(now, Ordering::SeqCst);
        let index = {
            let mut arrivals = self.fetch_arrivals.lock().unwrap();
            arrivals.push(now);
            arrivals.len() - 1
        };
        self.fetch_arrived.notify_waiters();
        self.paced(index).await;
        let answer = self.fetch(handle).await;
        self.fetches_in_flight.fetch_sub(1, Ordering::SeqCst);
        answer
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
        self.call();
        self.inner.delete_layer_checkpoint(handle).await
    }
    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.call();
        self.inner
            .put_session_checkpoint(session_id, body, meta)
            .await
    }
    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.call();
        let now = self.switched_call("get_session_checkpoint").await;
        if self
            .session_reads_failing
            .lock()
            .unwrap()
            .contains(session_id)
        {
            return Err(CheckpointError::Transient {
                source: "the backup store answered 503".into(),
            });
        }
        if self.session_hidden(session_id, now) {
            return Ok(None);
        }
        self.inner.get_session_checkpoint(session_id).await
    }
    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.call();
        self.session_deletes.fetch_add(1, Ordering::SeqCst);
        self.inner.delete_session_checkpoint(session_id).await
    }
    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.call();
        let now = self.switched_call("list_session_checkpoint_ids").await;
        let mut ids = self.inner.list_session_checkpoint_ids().await?;
        ids.retain(|session| !self.session_hidden(session, now));
        Ok(ids)
    }
    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.call();
        self.ext_puts.lock().unwrap().push(hash_hex.to_owned());
        self.inner.put_ext_blob(hash_hex, bytes).await
    }
    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.call();
        self.inner.get_ext_blob(hash_hex).await
    }
    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.call();
        self.probes.lock().unwrap().push(hash_hex.to_owned());
        let now = self.switched_call("ext_blob_exists").await;
        if let Some(now) = now {
            self.first_probe_at_ms.lock().unwrap().get_or_insert(now);
        }
        if self.probe_failing.lock().unwrap().contains(hash_hex) {
            return Err(CheckpointError::Transient {
                source: "the backup store answered 503".into(),
            });
        }
        let absent_until = self.ext_absent_until.lock().unwrap().get(hash_hex).copied();
        if let (Some(until), Some(now)) = (absent_until, now) {
            if now < until {
                return Ok(false);
            }
        }
        self.inner.ext_blob_exists(hash_hex).await
    }
    async fn verify_list_route(&self) -> Result<(), CheckpointError> {
        self.call();
        self.inner.verify_list_route().await
    }
    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.call();
        self.inner.delete_ext_blob(hash_hex).await
    }
    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.call();
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
        self.inner.read_ext_refs_sidecar(handle).await
    }
    async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
        self.call();
        let now = self.switched_call("list_layer_artifacts").await;
        if let Some(now) = now {
            self.first_listing_at_ms.lock().unwrap().get_or_insert(now);
        }
        let empty_until = *self.listing_empty_until.lock().unwrap();
        if let (Some(until), Some(now)) = (empty_until, now) {
            if now < until {
                return Ok(Vec::new());
            }
        }
        self.inner.list_layer_artifacts().await
    }
    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.call();
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

/// A layer's base: the prim `/World` and `attributes` attributes under it,
/// each with an inline `default` derived from `tag` and its index and a
/// `typeName`, the first with a time sample at 1.0, and per hash in
/// `external` one more attribute whose `default` names that external value
/// (a base carries hash references with no value parts, as the assembler
/// writes it).
pub(crate) fn base(attributes: u64, tag: u8, external: &[[u8; 32]]) -> Bytes {
    let delta = base_delta(attributes, tag, external);
    let delta_bytes = oodf_ffi::build_delta(&delta).expect("the base delta builds");
    Bytes::from(OmpfContainer::serialize(&delta_bytes, &[]).expect("the base serializes"))
}

/// [`base`]'s reset delta.
pub(crate) fn base_delta(attributes: u64, tag: u8, external: &[[u8; 32]]) -> OwnedDelta {
    let mut struct_commands = vec![StructCommand::CreateSection(CommandCreateSection {
        parent_id: 0,
        section_id: 1,
        section_name: "World".to_owned(),
        section_type: SPEC_PRIM,
    })];
    let mut set_fields = Vec::new();
    let mut time_samples = Vec::new();
    let mut set_order = 0;
    let mut next_field = |section_id: u64, key: &str, value: Vec<u8>, ext: Option<u64>| {
        set_order += 1;
        SetField {
            section_id,
            key_name: key.to_owned(),
            ext_value_size: ext.unwrap_or(0),
            ext_value_index: u32::from(ext.is_some()),
            value_or_ext_hash: value,
            set_order,
        }
    };
    for i in 0..attributes {
        let section_id = 2 + i;
        struct_commands.push(StructCommand::CreateSection(CommandCreateSection {
            parent_id: 1,
            section_id,
            section_name: format!("attr{i}"),
            section_type: SPEC_ATTRIBUTE,
        }));
        let index = u8::try_from(i % 256).expect("fits");
        set_fields.push(next_field(
            section_id,
            "default",
            vec![tag, index, 1, 2],
            None,
        ));
        set_fields.push(next_field(section_id, "typeName", b"float".to_vec(), None));
        if i == 0 {
            time_samples.push(TimeSample {
                section_id,
                time: 1.0,
                value_or_ext_hash: vec![tag, 0xAA],
                ext_value_size: 0,
                ext_value_index: 0,
            });
        }
    }
    for (j, hash) in external.iter().enumerate() {
        let section_id = 2 + attributes + j as u64;
        struct_commands.push(StructCommand::CreateSection(CommandCreateSection {
            parent_id: 1,
            section_id,
            section_name: format!("ext{j}"),
            section_type: SPEC_ATTRIBUTE,
        }));
        set_fields.push(next_field(
            section_id,
            "default",
            hash.to_vec(),
            Some(4_096),
        ));
    }
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        struct_commands,
        set_fields,
        time_samples,
        source_format: "usda".to_owned(),
    }
}

/// A value carried by a tail, and its content hash.
pub(crate) fn carried(bytes: &'static [u8]) -> ([u8; 32], &'static [u8]) {
    (ext_content_hash(bytes), bytes)
}

/// A tail record: an ordinary commit adding the attribute `section_name`
/// (section `section_id`) under `/World` whose `default` is the external
/// value `hash`, the value's bytes carried in the record.
fn tail_record(section_id: u64, section_name: &str, hash: [u8; 32], value: &[u8]) -> Bytes {
    let delta = OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 1,
            section_id,
            section_name: section_name.to_owned(),
            section_type: SPEC_ATTRIBUTE,
        })],
        set_fields: vec![SetField {
            section_id,
            key_name: "default".to_owned(),
            value_or_ext_hash: hash.to_vec(),
            ext_value_size: value.len() as u64,
            ext_value_index: 1,
            set_order: 0,
        }],
        time_samples: Vec::new(),
        source_format: String::new(),
    };
    let delta_bytes = oodf_ffi::build_delta(&delta).expect("the record delta builds");
    Bytes::from(OmpfContainer::serialize(&delta_bytes, &[value]).expect("the record serializes"))
}

/// A tail of `object` on its base at `base_seq`: one record per carried
/// value, at `base_seq + 1`, `base_seq + 2`, …, each adding an attribute
/// whose default is that value. Returns the envelope and the head it
/// reaches.
pub(crate) fn tail(object: u64, base_seq: u64, values: &[([u8; 32], &[u8])]) -> (Bytes, u64) {
    let to_seq = base_seq + values.len() as u64;
    let records = values.iter().enumerate().map(|(i, (hash, value))| {
        let seq = base_seq + 1 + i as u64;
        TailRecord::Verbatim {
            sequence_num: seq,
            server_ts: 1,
            ompf: tail_record(900 + i as u64, &format!("carried{i}"), *hash, value),
        }
    });
    let envelope = encode_tail_envelope(
        &TailHeader {
            object_id: object,
            base_snap_seq: base_seq,
            from_seq: base_seq,
            to_seq,
        },
        records,
    )
    .expect("the tail encodes");
    (envelope, to_seq)
}

/// The meta of a backup generation of `layer` in `session`.
pub(crate) fn backup_meta(session: &str, layer: &str, seq: u64) -> CheckpointMeta {
    CheckpointMeta {
        creator_replica_id: "restore-job-test".to_owned(),
        written_at_unix_ms: 1_700_000_000_000 + seq,
        snap_seq: Some(seq),
        session_id: session.to_owned(),
        layer_name: layer.to_owned(),
        created_by: "user".to_owned(),
        server_ts: 1,
        observed_commit_span: 0,
        kind: CheckpointKind::Base,
        base_snap_seq: None,
        from_seq: None,
        sidecar_bytes: None,
    }
}

pub(crate) async fn put_base(
    store: &dyn CheckpointStore,
    object_id: u64,
    session: &str,
    layer: &str,
    seq: u64,
    bytes: Bytes,
) {
    store
        .put_layer_checkpoint(object_id, seq, bytes, backup_meta(session, layer, seq))
        .await
        .expect("the base put succeeds");
}

/// Put `envelope`, a tail of `object` on its base at `base_seq`, ending at
/// `to_seq`.
pub(crate) async fn put_tail(
    store: &dyn CheckpointStore,
    object_id: u64,
    session: &str,
    layer: &str,
    base_seq: u64,
    to_seq: u64,
    envelope: Bytes,
) {
    store
        .put_layer_checkpoint(
            object_id,
            to_seq,
            envelope,
            CheckpointMeta {
                kind: CheckpointKind::Tail,
                base_snap_seq: Some(base_seq),
                from_seq: Some(base_seq),
                ..backup_meta(session, layer, to_seq)
            },
        )
        .await
        .expect("the tail put succeeds");
}

/// The base path of `session`'s fixtures.
pub(crate) fn base_path_of(session: &str) -> String {
    format!("omniverse://example.com/projects/{session}.usd")
}

/// The name of `session`'s fixtures.
pub(crate) fn name_of(session: &str) -> String {
    format!("{session}-name")
}

/// A session backup of `session`, named [`name_of`], under
/// [`base_path_of`], mapping each `(layer, object)`. Its path-names members
/// are the session's own (`name\0session`, as ListSessions reads it) and
/// `extra_name_members`.
pub(crate) fn session_backup(
    session: &str,
    mappings: &[(&str, u64)],
    extra_name_members: &[&str],
) -> SessionCheckpoint {
    let mut path_name_members: BTreeSet<String> =
        extra_name_members.iter().map(|m| (*m).to_owned()).collect();
    path_name_members.insert(KeySchema::path_name_member(&name_of(session), session));
    SessionCheckpoint {
        schema_version: 1,
        session_id: session.to_owned(),
        name: name_of(session),
        base_path: base_path_of(session),
        created_by: "user".to_owned(),
        created_at: 1,
        layer_mappings: mappings
            .iter()
            .map(|(l, o)| ((*l).to_owned(), *o))
            .collect(),
        layers_index: mappings.iter().map(|(l, _)| (*l).to_owned()).collect(),
        path_name_members,
        path_index_session_id_member: session.to_owned(),
    }
}

pub(crate) async fn put_session(store: &dyn CheckpointStore, checkpoint: &SessionCheckpoint) {
    store
        .put_session_checkpoint(
            &checkpoint.session_id,
            checkpoint.to_bytes().expect("the session backup encodes"),
            backup_meta(&checkpoint.session_id, "", 0),
        )
        .await
        .expect("the session put succeeds");
}

/// The counters record a fixture puts: the shape the counters tick writes
/// (`checkpoint::CountersSidecar`), its counter reading 0, so the counter
/// floor's sidecar term is the margin alone.
pub(crate) const COUNTERS_RECORD: &[u8] =
    b"{\"max_object_id_observed\":0,\"updated_at\":\"2026-10-20T12:00:00Z\"}";

pub(crate) async fn put_counters(store: &dyn CheckpointStore) {
    store
        .put_counters_sidecar(Bytes::from_static(COUNTERS_RECORD))
        .await
        .expect("the counters record put succeeds");
}

// ═════════════════════════════════════════════════════════════════════════════
// Store fixtures and reads
// ═════════════════════════════════════════════════════════════════════════════

pub(crate) async fn hset(tv: &TestValkey, key: String, fields: &[(&str, &str)]) {
    let fields: Vec<(String, String)> = fields
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    let _: i64 = tv.client().pool.hset(key, fields).await.expect("HSET");
}

pub(crate) async fn sadd(tv: &TestValkey, key: String, member: &str) {
    let _: i64 = tv.client().pool.sadd(key, member).await.expect("SADD");
}

pub(crate) async fn smembers(tv: &TestValkey, key: String) -> BTreeSet<String> {
    tv.client().pool.smembers(key).await.expect("SMEMBERS")
}

pub(crate) async fn exists(tv: &TestValkey, key: String) -> bool {
    let n: i64 = tv.client().pool.exists(key).await.expect("EXISTS");
    n > 0
}

pub(crate) async fn get(tv: &TestValkey, key: String) -> Option<String> {
    tv.client().pool.get(key).await.expect("GET")
}

/// A structurally complete surviving header of `layer` in `session` at
/// `seq`, plus `extra` fields.
pub(crate) async fn seed_complete(
    tv: &TestValkey,
    object_id: u64,
    session: &str,
    layer: &str,
    seq: u64,
    extra: &[(&str, &str)],
) {
    let seq = seq.to_string();
    let mut fields = vec![
        ("seq", seq.as_str()),
        ("created_by", "user"),
        ("session_id", session),
        ("layer_name", layer),
    ];
    fields.extend_from_slice(extra);
    hset(
        tv,
        tv.key_schema().obj_header(&object_id.to_string()),
        &fields,
    )
    .await;
}

/// A layer mapping and its layers-index member.
pub(crate) async fn seed_mapping(tv: &TestValkey, session: &str, layer: &str, object_id: u64) {
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
    sadd(
        tv,
        tv.key_schema().layers_index(session),
        &hex_encode(layer),
    )
    .await;
}

/// A session whose record survives.
pub(crate) async fn seed_session_record(tv: &TestValkey, session: &str) {
    hset(
        tv,
        tv.key_schema().session_meta(session),
        &[
            ("name", "session"),
            ("base_path", base_path_of(session).as_str()),
            ("created_by", "user"),
        ],
    )
    .await;
}

pub(crate) async fn seed_deletion_marker(tv: &TestValkey, session: &str) {
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().session_deleted_marker(session),
            "1",
            None,
            None,
            false,
        )
        .await
        .expect("SET marker");
}

pub(crate) async fn seed_dead(tv: &TestValkey, object_id: u64) {
    let _: i64 = tv
        .client()
        .pool
        .zadd(
            tv.key_schema().checkpoint_reconcile_dead(),
            None,
            None,
            false,
            false,
            (0.0, object_id.to_string()),
        )
        .await
        .expect("ZADD dead");
}

/// The objects on the dead-object set.
pub(crate) async fn dead_set(tv: &TestValkey) -> BTreeSet<u64> {
    let members: Vec<String> = tv
        .client()
        .pool
        .zrange(
            tv.key_schema().checkpoint_reconcile_dead(),
            0,
            -1,
            None,
            false,
            None,
            false,
        )
        .await
        .expect("ZRANGE dead");
    members
        .iter()
        .map(|m| m.parse().expect("an object id"))
        .collect()
}

/// Append an entry to `object_id`'s stream at `id` (`*`: an auto-ID, as a
/// `deleted` tombstone's); the id the server gave it.
pub(crate) async fn xadd(tv: &TestValkey, object_id: u64, id: &str) -> String {
    tv.client()
        .pool
        .xadd(
            tv.key_schema().layer_stream(object_id),
            false,
            None,
            id,
            vec![("payload", "x")],
        )
        .await
        .unwrap_or_else(|e| panic!("XADD {id} must succeed in test setup: {e}"))
}

/// Residue of an earlier object at `object_id`: a section row and its
/// index member that no backup names, which a restore must wipe.
pub(crate) async fn seed_residue(tv: &TestValkey, object_id: u64) {
    let oid = object_id.to_string();
    hset(
        tv,
        tv.key_schema().obj_section(&oid, "777"),
        &[("parent_id", "1"), ("name", "stale"), ("type", "6")],
    )
    .await;
    sadd(tv, tv.key_schema().obj_sections_index(&oid), "777").await;
}

/// Every key of `object_id` (its `{oid}` hash tag), stream included.
pub(crate) async fn object_keys(tv: &TestValkey, object_id: u64) -> Vec<String> {
    let pattern = format!("{}{{{object_id}}}:*", tv.prefix());
    crate::common::scan_keys(tv.client(), &pattern)
        .await
        .iter()
        .map(|k| k.as_str().expect("keys are UTF-8").to_owned())
        .collect()
}

/// `object_id`'s header.
pub(crate) async fn header(tv: &TestValkey, object_id: u64) -> HashMap<String, String> {
    tv.client()
        .pool
        .hgetall(tv.key_schema().obj_header(&object_id.to_string()))
        .await
        .expect("HGETALL")
}

/// `object_id`'s state as stored, read back whole.
pub(crate) async fn stored_state(tv: &TestValkey, object_id: u64) -> ObjectState {
    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
        .await
        .unwrap_or_else(|e| panic!("object {object_id} reads back: {e}"));
    snapshot_to_object_state(&snapshot)
}

/// The state a restore of `object_id` from `base` (at `base_seq`) and
/// `tails` must write, seeded at `seeded_seq`: the planner's decode, with
/// `first_version` 0 and every version flattened onto `seeded_seq`.
pub(crate) fn expected_state(
    object_id: u64,
    base_seq: u64,
    base: &[u8],
    tails: &[Bytes],
    seeded_seq: u64,
) -> ObjectState {
    let DecodedLineage::Ok { mut state, .. } = decode_planned(object_id, base_seq, base, tails)
    else {
        panic!("fixture: the backup of object {object_id} decodes");
    };
    state.first_version = 0;
    flatten_to_seeded_seq(&mut state, seeded_seq);
    *state
}

/// `state` as a store holds it: with its empty children lists dropped (a
/// store keeps no key for one) and every external value's
/// `ext_value_index` at 0 (the index is a position in one delta's
/// external-value parts, not part of the value; a stored row does not keep
/// it).
fn as_stored(mut state: ObjectState) -> ObjectState {
    use live_edit::domain::types::FieldValue;
    let unindex = |value: &mut FieldValue| {
        if let FieldValue::External {
            ext_value_index, ..
        } = value
        {
            *ext_value_index = 0;
        }
    };
    state.children.retain(|_, ids| !ids.is_empty());
    for field in state.fields.values_mut() {
        unindex(&mut field.value);
    }
    for sample in state.time_samples.values_mut() {
        unindex(&mut sample.value);
    }
    state
}

/// Assert `object_id`'s stored content is `expected` ([`as_stored`]): every
/// section, children list, field and time sample, with their versions, the
/// sequence number and the header's state fields.
pub(crate) async fn assert_restored(tv: &TestValkey, object_id: u64, expected: &ObjectState) {
    let stored = as_stored(stored_state(tv, object_id).await);
    let expected = as_stored(expected.clone());
    assert_eq!(
        stored.sections, expected.sections,
        "object {object_id}: sections"
    );
    assert_eq!(
        stored.children, expected.children,
        "object {object_id}: children"
    );
    assert_eq!(stored.fields, expected.fields, "object {object_id}: fields");
    assert_eq!(
        stored.time_samples, expected.time_samples,
        "object {object_id}: time samples"
    );
    assert_eq!(
        (
            stored.sequence_num,
            stored.first_version,
            &stored.source_format,
            stored.next_set_order
        ),
        (
            expected.sequence_num,
            expected.first_version,
            &expected.source_format,
            expected.next_set_order
        ),
        "object {object_id}: the header's state fields"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Running the Job
// ═════════════════════════════════════════════════════════════════════════════

pub(crate) fn metrics() -> Arc<Metrics> {
    Arc::new(crate::common::metrics_scrape_only("restore-job"))
}

/// The test configuration with a `StorageApi` backup store whose not-found
/// cache lifetime is 0, so `T_safe` is the quiet period's end.
pub(crate) fn writer_config() -> Config {
    let mut config = crate::common::test_config();
    config.checkpoint_adapter = CheckpointAdapter::StorageApi;
    config.checkpoint_storage_api_not_found_cache_secs = 0;
    config
}

/// The `mode` Job's context on `tv` with `config`, reading backups from
/// `store`, under [`INCIDENT`], with second-scale lease timings and a
/// 250 ms wait-log interval; no shutdown signal, and no second accept-loss
/// confirm. Its allocator purge accepts every call and purges nothing (this
/// binary's allocator is the system's), so the value removal takes the
/// binary's purged path.
pub(crate) fn context(
    tv: &TestValkey,
    mode: JobMode,
    config: Config,
    store: Arc<dyn CheckpointStore>,
    metrics: &Arc<Metrics>,
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
        metrics: Arc::clone(metrics),
        metrics_listener: None,
        timings: JobTimings {
            lease: LeaseTimings {
                ttl: Duration::from_secs(30),
                refresh_interval: Duration::from_secs(1),
                acquire_wait: Duration::from_secs(3),
                acquire_poll: Duration::from_millis(100),
            },
            wait_log_interval: Duration::from_millis(250),
        },
        shutdown: Box::pin(std::future::pending()),
        identity: LeaderIdentity::new("restore-job-test", tv.prefix()),
        allocator_purge: Some(AllocatorPurge::new(|| true)),
    }
}

/// Hold the store under [`INCIDENT`] with its quiet period under `config`
/// already over.
pub(crate) async fn hold(tv: &TestValkey, config: &Config) {
    let q = u64::try_from(QuietPeriod::from_config(config).total().as_millis()).expect("Q fits");
    let now = store_time_ms(tv.client(), tv.key_schema())
        .await
        .expect("TIME");
    tv.write_hold_marker(
        &HoldMarkerSeed::default()
            .with_incident(INCIDENT)
            .detected_at(now - q),
    )
    .await;
}

/// What a run's body handed its execution observer: `(plan, report)`.
pub(crate) type Observed = Arc<Mutex<Vec<(RestorePlan, ExecutionReport)>>>;

/// The restore body with `seams` applied, recording what the writer did.
pub(crate) fn observed_body(
    seams: impl FnOnce(RestoreBody) -> RestoreBody,
) -> (RestoreBody, Observed) {
    let observed: Observed = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&observed);
    let body = seams(RestoreBody::new()).with_execution_observer(Arc::new(move |plan, report| {
        sink.lock().unwrap().push((plan.clone(), report.clone()));
    }));
    (body, observed)
}

/// The one `(plan, report)` a run's body recorded.
pub(crate) fn the_execution(observed: &Observed) -> (RestorePlan, ExecutionReport) {
    let observed = observed.lock().unwrap();
    assert_eq!(observed.len(), 1, "the writer finished once");
    observed[0].clone()
}

/// Run `future` within `budget`, capturing what it logs.
pub(crate) async fn captured_within<F: Future>(
    budget: Duration,
    future: F,
) -> (F::Output, Captured) {
    capture(async {
        tokio::time::timeout(budget, future)
            .await
            .expect("the run ends within its budget")
    })
    .await
}

/// Run the restore Job with `body` to its exit within `budget`, capturing
/// what it logs.
pub(crate) async fn run_within(
    budget: Duration,
    tv: &TestValkey,
    config: &Config,
    store: &Arc<WriterStore>,
    body: RestoreBody,
) -> (JobExit, Captured) {
    let metrics = metrics();
    let context = context(
        tv,
        JobMode::Restore,
        config.clone(),
        Arc::clone(store) as Arc<dyn CheckpointStore>,
        &metrics,
    );
    captured_within(budget, run_job(context, body)).await
}

/// [`run_within`] [`RUN_BUDGET`].
pub(crate) async fn run(
    tv: &TestValkey,
    config: &Config,
    store: &Arc<WriterStore>,
    body: RestoreBody,
) -> (JobExit, Captured) {
    run_within(RUN_BUDGET, tv, config, store, body).await
}

/// Run the Job with no other seam to just before its un-hold
/// (`KillPoint::BeforeUnhold`): the writer must finish, the recount and the
/// summary run, and the body exit 1 with the kill line, so the store stays
/// held and a re-run converges on what this run wrote. Returns what it
/// wrote. A fixture that stays held runs [`run_to_the_end_held`] instead.
pub(crate) async fn run_to_the_end(
    tv: &TestValkey,
    config: &Config,
    store: &Arc<WriterStore>,
) -> (RestorePlan, ExecutionReport, Captured) {
    run_to_the_end_with(RUN_BUDGET, tv, config, store, |b| b).await
}

/// [`run_to_the_end`] with `seams` applied to the body, within `budget`:
/// a writable run, which must stop at the kill point (exit 1, the kill
/// line).
pub(crate) async fn run_to_the_end_with(
    budget: Duration,
    tv: &TestValkey,
    config: &Config,
    store: &Arc<WriterStore>,
    seams: impl FnOnce(RestoreBody) -> RestoreBody,
) -> (RestorePlan, ExecutionReport, Captured) {
    run_to_the_end_expecting(budget, tv, config, store, seams, JobExit::Failed, KILL_LINE).await
}

/// The Job run to its end on a fixture that stays held: the writer must
/// finish, and the run, which never reaches the un-hold's kill point, exit 3
/// after its recount with the stays-held ERROR, the marker kept. Returns
/// what it wrote.
pub(crate) async fn run_to_the_end_held(
    tv: &TestValkey,
    config: &Config,
    store: &Arc<WriterStore>,
) -> (RestorePlan, ExecutionReport, Captured) {
    run_to_the_end_expecting(
        RUN_BUDGET,
        tv,
        config,
        store,
        |b| b,
        JobExit::StaysHeld,
        STAYS_HELD_LINE,
    )
    .await
}

/// The Job run with `seams` and the kill point before its un-hold, which
/// must exit `expected` with one line starting `line` and record the
/// writer's finish.
async fn run_to_the_end_expecting(
    budget: Duration,
    tv: &TestValkey,
    config: &Config,
    store: &Arc<WriterStore>,
    seams: impl FnOnce(RestoreBody) -> RestoreBody,
    expected: JobExit,
    line: &str,
) -> (RestorePlan, ExecutionReport, Captured) {
    let (body, observed) =
        observed_body(|b| seams(b).with_kill_point_for_test(KillPoint::BeforeUnhold));
    let (exit, captured) = run_within(budget, tv, config, store, body).await;
    assert_eq!(exit, expected, "{}", captured.text);
    assert_eq!(
        lines_starting(&captured, line),
        1,
        "the writer finished and the run stopped before its un-hold; capture:\n{}",
        captured.text
    );
    let (plan, report) = the_execution(&observed);
    (plan, report, captured)
}

/// Run the Job with `body`, which must stop at its kill point: exit 1, one
/// kill INFO, and no execution recorded.
pub(crate) async fn run_to_the_kill(
    tv: &TestValkey,
    config: &Config,
    store: &Arc<WriterStore>,
    seams: impl FnOnce(RestoreBody) -> RestoreBody,
) {
    let (body, observed) = observed_body(seams);
    let (exit, captured) = run(tv, config, store, body).await;
    assert_eq!(exit, JobExit::Failed, "{}", captured.text);
    assert_eq!(
        lines_starting(&captured, KILL_LINE),
        1,
        "the writer reached its kill point; capture:\n{}",
        captured.text
    );
    assert!(
        observed.lock().unwrap().is_empty(),
        "a killed writer finishes nothing"
    );
}

/// The captured lines whose message is exactly `message`.
pub(crate) fn lines<'c>(captured: &'c Captured, message: &str) -> Vec<&'c serde_json::Value> {
    captured
        .events
        .iter()
        .filter(|e| e["fields"]["message"].as_str() == Some(message))
        .collect()
}

/// How many captured lines' messages start with `prefix`.
pub(crate) fn lines_starting(captured: &Captured, prefix: &str) -> usize {
    captured
        .events
        .iter()
        .filter(|e| {
            e["fields"]["message"]
                .as_str()
                .is_some_and(|m| m.starts_with(prefix))
        })
        .count()
}

/// The restore of `object_id` in `plan`: `(head_seq, base_seq, tails)`.
pub(crate) fn restored_lineage(plan: &RestorePlan, object_id: u64) -> (u64, u64, usize) {
    let found: Vec<_> = plan
        .objects
        .iter()
        .filter(|o| o.object_id == object_id)
        .collect();
    match found.as_slice() {
        [object] => match &object.action {
            ObjectAction::Restore { lineage, .. } => {
                (lineage.head_seq, lineage.base.seq(), lineage.tails.len())
            }
            other => panic!("object {object_id} must be restored, got {other:?}"),
        },
        other => panic!("object {object_id} is planned once: {other:?}"),
    }
}

/// A hand-built run reading `store`, its quiet period and `T_safe` past, for
/// planning without the frame (the planner writes nothing).
pub(crate) async fn planning_run(
    tv: &TestValkey,
    config: &Config,
    store: &Arc<WriterStore>,
) -> JobRun {
    let now = store_time_ms(tv.client(), tv.key_schema())
        .await
        .expect("TIME");
    JobRun {
        mode: JobMode::Restore,
        incident: INCIDENT.to_owned(),
        quiet: QuietPeriod::from_config(config),
        not_found_cache: Duration::ZERO,
        config: Arc::new(config.clone()),
        valkey: tv.client().clone(),
        keys: tv.key_schema().clone(),
        backup: BackupAccess::Store(Arc::clone(store) as Arc<dyn CheckpointStore>),
        clock: StoreClock::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            StoreDeadlines {
                detected_at_ms: now - HOUR_MS,
                quiet_ends_at_ms: now - 1,
                t_safe_ms: now - 1,
            },
            Duration::from_millis(250),
        ),
        permit: WritePermit::new(),
        own_puts: OwnPuts::default(),
        metrics: metrics(),
        lease_holder: String::new(),
        counters_sidecar_max: std::sync::OnceLock::new(),
        allocator_purge: Some(AllocatorPurge::new(|| true)),
    }
}
