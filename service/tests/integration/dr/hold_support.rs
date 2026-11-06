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

//! Fixtures the hold tests share: a checkpoint store that forwards to a real
//! one while the test parks or slows the calls it is watching, an
//! external-value store over a real one that counts or parks, the settings of
//! a one-shot hold read the test expects to answer ([`answering`]), and held
//! mode run in process on real time ([`HeldModeRun`]: `held_mode_test`, and
//! the held replicas the restore Job's un-hold releases in
//! `restore_finish_test`).

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::KeysInterface;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot, watch, Notify};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tonic::transport::server::TcpIncoming;
use tonic::transport::Channel;
use tonic_health::server::HealthReporter;

use live_edit::dr::hold::{evaluate_hold_at_startup, HoldObservation, HoldPollSettings};
use live_edit::error::ServiceError;
use live_edit::grpc::hold::{
    run_held_mode, HeldMode, HeldModeError, HoldExit, HoldReactorSettings,
};
use live_edit::grpc::metrics_layer::MetricsLayer;
use live_edit::observability::health::create_health_service;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::observability::metrics_server::serve_metrics_with_scaling;
use live_edit::observability::scaling::ScalingEndpoint;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::test_support::TestValkey;
use live_edit_scaler::snapshot::ScalingSnapshot;

use super::support::{RecordingExit, READINESS};
use live_edit::storage::ext_release_gate::ExtReleaseGate;
use live_edit::storage::external_values::{
    ExtRelease, ExtValueLengths, ExtValueProbe, ExtValueVerify, ExternalValueStore, GcBatchResult,
    GcPassState, GcPolicy, ReleaseAttemptOutcome, ValkeyExternalValueStore,
};

use live_edit::checkpoint::{
    CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointMeta, CheckpointStore,
    LayerArtifact, LiveExtSet,
};

/// Bound on waiting for a parked call to start. Unreachable on the healthy
/// path.
const PARK_ENTRY_BUDGET: Duration = Duration::from_secs(10);

/// One parked call of a [`ParkedStore`]: it signals when the call starts and
/// returns only once released. Released, it parks nothing more.
#[derive(Default)]
pub(crate) struct Park {
    entered: Notify,
    released: watch::Sender<bool>,
}

impl Park {
    /// Signal that a call reached this park, then wait for its release.
    pub(crate) async fn hold(&self) {
        self.entered.notify_one();
        let mut released = self.released.subscribe();
        let _ = released.wait_for(|r| *r).await;
    }

    /// Wait until a call has reached this park, within [`PARK_ENTRY_BUDGET`].
    pub(crate) async fn entered(&self, what: &str) {
        tokio::time::timeout(PARK_ENTRY_BUDGET, self.entered.notified())
            .await
            .unwrap_or_else(|_| {
                panic!("{what} did not reach the store within {PARK_ENTRY_BUDGET:?}")
            });
    }

    /// Let every parked call, and every later one, through.
    pub(crate) fn release(&self) {
        self.released.send_replace(true);
    }
}

/// A checkpoint store forwarding every call to `inner` — every method of the
/// trait, the ones with defaults included, so nothing behind it changes
/// behaviour — except that:
///
/// - the counters-sidecar read, the session-checkpoint read, the
///   layer-checkpoint listing and the layer-checkpoint `PUT` each wait on
///   their [`Park`] first;
/// - an external-value `PUT` (the mirror workers') waits `ext_put_delay`
///   first, and is counted once it has landed ([`Self::ext_puts`]);
/// - a call named by [`Self::close_on`] cancels its token just as it answers,
///   inside the same poll: the caller gets the answer with the token already
///   cancelled, past any race it ran against the token during the call.
pub(crate) struct ParkedStore {
    inner: Arc<dyn CheckpointStore>,
    pub(crate) sidecar: Park,
    pub(crate) session: Park,
    pub(crate) listing: Park,
    pub(crate) layer_put: Park,
    ext_put_delay: Duration,
    ext_puts: AtomicUsize,
    close_on: Mutex<Option<(StoreCall, CancellationToken)>>,
}

/// A read [`ParkedStore::close_on`] can cancel a token at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StoreCall {
    /// `get_counters_sidecar`.
    Sidecar,
    /// `get_session_checkpoint`.
    Session,
    /// `get_layer_checkpoint`.
    LayerGet,
    /// `get_ext_blob`: the external-value fetch-on-miss's backup fetch.
    ExtGet,
}

impl ParkedStore {
    fn build(inner: Arc<dyn CheckpointStore>, ext_put_delay: Duration) -> Self {
        Self {
            inner,
            sidecar: Park::default(),
            session: Park::default(),
            listing: Park::default(),
            layer_put: Park::default(),
            ext_put_delay,
            ext_puts: AtomicUsize::new(0),
            close_on: Mutex::new(None),
        }
    }

    /// Cancel `token` as the next `call` answers.
    pub(crate) fn close_on(&self, call: StoreCall, token: CancellationToken) {
        *self.close_on.lock().unwrap() = Some((call, token));
    }

    fn answered(&self, call: StoreCall) {
        let mut slot = self.close_on.lock().unwrap();
        if slot.as_ref().is_some_and(|(c, _)| *c == call) {
            if let Some((_, token)) = slot.take() {
                token.cancel();
            }
        }
    }

    /// Over `inner`, with every park closed but the layer-checkpoint `PUT`'s,
    /// and no external-value `PUT` delay.
    pub(crate) fn over(inner: Arc<dyn CheckpointStore>) -> Arc<Self> {
        let store = Self::build(inner, Duration::ZERO);
        store.layer_put.release();
        Arc::new(store)
    }

    /// Over `inner`, with nothing parked.
    pub(crate) fn passing(inner: Arc<dyn CheckpointStore>) -> Arc<Self> {
        Self::with_ext_put_delay(inner, Duration::ZERO)
    }

    /// Over `inner`, with only the layer-checkpoint `PUT` parked.
    pub(crate) fn parking_layer_puts(inner: Arc<dyn CheckpointStore>) -> Arc<Self> {
        let store = Self::build(inner, Duration::ZERO);
        store.sidecar.release();
        store.session.release();
        store.listing.release();
        Arc::new(store)
    }

    /// Over `inner`, each external-value `PUT` delayed by `delay`. Nothing is
    /// parked.
    pub(crate) fn with_ext_put_delay(
        inner: Arc<dyn CheckpointStore>,
        delay: Duration,
    ) -> Arc<Self> {
        let store = Self::build(inner, delay);
        store.sidecar.release();
        store.session.release();
        store.listing.release();
        store.layer_put.release();
        Arc::new(store)
    }

    /// The external-value `PUT`s that have landed in `inner`.
    pub(crate) fn ext_puts(&self) -> usize {
        self.ext_puts.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl CheckpointStore for ParkedStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.layer_put.hold().await;
        self.inner
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }
    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_latest_layer_checkpoint(object_id).await
    }
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        let answer = self.inner.get_layer_checkpoint(handle).await;
        self.answered(StoreCall::LayerGet);
        answer
    }
    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.listing.hold().await;
        self.inner.list_layer_checkpoints(object_id).await
    }
    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.inner.delete_layer_checkpoint(handle).await
    }
    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.inner
            .put_session_checkpoint(session_id, body, meta)
            .await
    }
    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.session.hold().await;
        let answer = self.inner.get_session_checkpoint(session_id).await;
        self.answered(StoreCall::Session);
        answer
    }
    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.inner.delete_session_checkpoint(session_id).await
    }
    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.inner.list_session_checkpoint_ids().await
    }
    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        if !self.ext_put_delay.is_zero() {
            tokio::time::sleep(self.ext_put_delay).await;
        }
        let put = self.inner.put_ext_blob(hash_hex, bytes).await;
        if put.is_ok() {
            self.ext_puts.fetch_add(1, Ordering::SeqCst);
        }
        put
    }
    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        let answer = self.inner.get_ext_blob(hash_hex).await;
        self.answered(StoreCall::ExtGet);
        answer
    }
    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.inner.ext_blob_exists(hash_hex).await
    }

    async fn verify_list_route(&self) -> Result<(), CheckpointError> {
        self.inner.verify_list_route().await
    }
    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.inner.delete_ext_blob(hash_hex).await
    }
    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.inner.put_counters_sidecar(body).await
    }
    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.sidecar.hold().await;
        let answer = self.inner.get_counters_sidecar().await;
        self.answered(StoreCall::Sidecar);
        answer
    }
    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.inner.list_all_object_ids().await
    }
    async fn list_referenced_ext_hashes(
        &self,
    ) -> Result<std::collections::HashSet<String>, CheckpointError> {
        self.inner.list_referenced_ext_hashes().await
    }
    async fn list_live_ext_set(&self) -> Result<LiveExtSet, CheckpointError> {
        self.inner.list_live_ext_set().await
    }
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<std::collections::HashSet<String>, CheckpointError> {
        self.inner.read_ext_refs_sidecar(handle).await
    }
    async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
        self.inner.list_layer_artifacts().await
    }
    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.inner
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }
    fn publish_worst_case(&self) -> Duration {
        self.inner.publish_worst_case()
    }
}

/// An external-value store over a real [`ValkeyExternalValueStore`] that
/// counts its `store_values` calls and can park its first GC pass. It does
/// **not** override `store_values_unless_held`, so the trait's default — which
/// ignores the hold write fence, right for a store that sends nothing to
/// Valkey — is what a fenced pre-store reaches through it: a test of a fence
/// check made upstream of the store.
pub(crate) struct ShimExtStore {
    inner: ValkeyExternalValueStore,
    store_values_calls: AtomicUsize,
    /// When set, every `process_pending`, the first one parked until released,
    /// raises `Park::entered`, waits for the park's release, then `SET`s
    /// `pass_finished_key` (a write the pass makes only if it runs to its end)
    /// before delegating. Once released, `Park::hold` returns at once, so a
    /// later pass does not wait but still arms the drop flag and sets the key.
    /// `pass_dropped` turns true if any pass's future is dropped before that
    /// `SET` completes.
    gc_park: Option<GcPark>,
}

/// The parked GC pass of a [`ShimExtStore`].
pub(crate) struct GcPark {
    pub(crate) park: Park,
    pub(crate) pass_finished_key: String,
    pub(crate) pass_dropped: Arc<AtomicBool>,
    client: ValkeyClient,
}

impl GcPark {
    pub(crate) fn new(client: ValkeyClient, pass_finished_key: String) -> Self {
        Self {
            park: Park::default(),
            pass_finished_key,
            pass_dropped: Arc::new(AtomicBool::new(false)),
            client,
        }
    }
}

/// Sets its flag when dropped while armed.
struct DropFlag(Option<Arc<AtomicBool>>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        if let Some(flag) = self.0.take() {
            flag.store(true, Ordering::SeqCst);
        }
    }
}

impl ShimExtStore {
    pub(crate) fn new(inner: ValkeyExternalValueStore) -> Self {
        Self {
            inner,
            store_values_calls: AtomicUsize::new(0),
            gc_park: None,
        }
    }

    pub(crate) fn with_gc_park(mut self, park: GcPark) -> Self {
        self.gc_park = Some(park);
        self
    }

    pub(crate) fn gc_park(&self) -> &GcPark {
        self.gc_park.as_ref().expect("built with_gc_park")
    }

    /// `store_values` calls so far. Read only by a
    /// `test-checkpoint-fault-injection` test.
    #[cfg_attr(not(feature = "test-checkpoint-fault-injection"), allow(dead_code))]
    pub(crate) fn store_values_calls(&self) -> usize {
        self.store_values_calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl ExternalValueStore for ShimExtStore {
    async fn store_values(&self, values: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError> {
        self.store_values_calls.fetch_add(1, Ordering::SeqCst);
        self.inner.store_values(values).await
    }

    async fn store_verified_values(
        &self,
        values: &[(&[u8; 32], &[u8])],
    ) -> Result<(), ServiceError> {
        self.inner.store_verified_values(values).await
    }

    fn ext_value_verify(&self) -> ExtValueVerify<'_> {
        self.inner.ext_value_verify()
    }

    async fn load_values(
        &self,
        hashes: &[&[u8; 32]],
    ) -> Result<Vec<([u8; 32], Vec<u8>)>, ServiceError> {
        self.inner.load_values(hashes).await
    }

    async fn read_batch(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<HashMap<[u8; 32], Vec<u8>>, ServiceError> {
        self.inner.read_batch(hashes).await
    }

    async fn probe_values(&self, hashes: &[[u8; 32]]) -> Result<ExtValueProbe, ServiceError> {
        self.inner.probe_values(hashes).await
    }

    // The catch-up payload builder's reads, delegated so a catch-up through
    // this shim takes the production `STRLEN` / `GETRANGE` / `recover_value`
    // path rather than the trait's whole-value defaults.
    async fn probe_value_lengths(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<ExtValueLengths, ServiceError> {
        self.inner.probe_value_lengths(hashes).await
    }

    async fn read_value_range(
        &self,
        hash: &[u8; 32],
        start: u64,
        len: usize,
    ) -> Result<Bytes, ServiceError> {
        self.inner.read_value_range(hash, start, len).await
    }

    async fn recover_value(
        &self,
        hash: &[u8; 32],
        on_fetched: live_edit::storage::external_values::OnFetched<'_>,
    ) -> Result<bytes::Bytes, ServiceError> {
        self.inner.recover_value(hash, on_fetched).await
    }

    async fn enqueue_retirement_attempt(
        &self,
        object_id: &str,
        attempt: &str,
        releases: &[ExtRelease],
        send_by: std::time::Instant,
    ) -> Result<ReleaseAttemptOutcome, ServiceError> {
        self.inner
            .enqueue_retirement_attempt(object_id, attempt, releases, send_by)
            .await
    }

    async fn process_pending(
        &self,
        policy: &GcPolicy,
        state: &mut GcPassState,
    ) -> Result<GcBatchResult, ServiceError> {
        if let Some(gc) = self.gc_park.as_ref() {
            let mut armed = DropFlag(Some(Arc::clone(&gc.pass_dropped)));
            gc.park.hold().await;
            let _: () = gc
                .client
                .pool
                .set(&gc.pass_finished_key, "1", None, None, false)
                .await
                .expect("the parked pass's closing SET must succeed");
            armed.0 = None;
        }
        self.inner.process_pending(policy, state).await
    }

    fn ext_release_gate(&self) -> Option<&Arc<ExtReleaseGate>> {
        self.inner.ext_release_gate()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// One-shot hold reads
// ═════════════════════════════════════════════════════════════════════════════

/// The settings of a single hold read the test expects to answer, on a client
/// whose Valkey command timeout is `command_timeout_ms`: the largest accepted
/// interval, so the read's bound is that command timeout and a loaded runner
/// cannot turn an answer into a timeout. A running poll keeps its own short
/// interval, and the per-read bound that comes with it, because its timing is
/// what its tests exercise.
pub(crate) fn answering(command_timeout_ms: u64) -> HoldPollSettings {
    assert!(
        (1..=live_edit::config::DR_HOLD_POLL_INTERVAL_MS_MAX).contains(&command_timeout_ms),
        "fixture: a command timeout of {command_timeout_ms} ms does not bound the read"
    );
    HoldPollSettings::new(
        Duration::from_millis(live_edit::config::DR_HOLD_POLL_INTERVAL_MS_MAX),
        command_timeout_ms,
    )
}

// ═════════════════════════════════════════════════════════════════════════════
// Held mode, in process, on real time
// ═════════════════════════════════════════════════════════════════════════════

/// Upper bound on any single awaited event of the real-time tests.
pub(crate) const SETTLE: Duration = Duration::from_secs(10);

/// The un-hold exit's deadline: one poll interval, the 5 s jitter bound, and
/// slack for a loaded runner.
pub(crate) const EXIT_DEADLINE: Duration = Duration::from_secs(20);

/// The hold poll's interval in the held-server tests: short, so a removed
/// marker is read within a fraction of a second.
pub(crate) const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// The fixture clients' Valkey command timeout (`common::test_config`).
pub(crate) const COMMAND_TIMEOUT_MS: u64 = 5_000;

/// The held-mode health supervisor's probe interval.
pub(crate) const HEALTH_PROBE_INTERVAL: Duration = Duration::from_millis(200);

/// The metrics layer's service label.
pub(crate) const SERVICE_LABEL: &str = "held-mode";

/// `run_held_mode` running on a spawned task over `valkey`, as `main.rs` runs
/// it: on a TCP socket of its own, beside the production `/metrics` and
/// `/scaling/v1` server over the same registry and scaling slot, with a test
/// exit seam and a shutdown signal the test fires.
pub(crate) struct HeldModeRun {
    pub(crate) grpc: SocketAddr,
    pub(crate) http: SocketAddr,
    pub(crate) metrics: Arc<Metrics>,
    pub(crate) reporter: HealthReporter,
    pub(crate) exits: mpsc::UnboundedReceiver<(i32, Instant)>,
    pub(crate) signal: Option<oneshot::Sender<()>>,
    pub(crate) run: JoinHandle<Result<(), HeldModeError>>,
    pub(crate) metrics_server: JoinHandle<()>,
}

impl HeldModeRun {
    /// Held mode over `valkey`, seeded with `startup`, with a recording exit
    /// seam.
    pub(crate) async fn start(
        valkey: ValkeyClient,
        keys: &KeySchema,
        startup: HoldObservation,
    ) -> Self {
        let (calls, exits) = mpsc::unbounded_channel();
        Self::start_with_exit(
            valkey,
            keys,
            startup,
            Arc::new(RecordingExit { calls }),
            exits,
        )
        .await
    }

    pub(crate) async fn start_with_exit(
        valkey: ValkeyClient,
        keys: &KeySchema,
        startup: HoldObservation,
        exit: Arc<dyn HoldExit>,
        exits: mpsc::UnboundedReceiver<(i32, Instant)>,
    ) -> Self {
        Self::start_on_endpoint(
            valkey,
            keys,
            startup,
            exit,
            exits,
            Arc::new(ScalingEndpoint::new()),
        )
        .await
    }

    /// [`Self::start_with_exit`] with `scaling` as the `/scaling/v1` slot
    /// held mode registers its source on, and the metrics server serves.
    pub(crate) async fn start_on_endpoint(
        valkey: ValkeyClient,
        keys: &KeySchema,
        startup: HoldObservation,
        exit: Arc<dyn HoldExit>,
        exits: mpsc::UnboundedReceiver<(i32, Instant)>,
        scaling: Arc<ScalingEndpoint>,
    ) -> Self {
        Self::start_full(
            valkey,
            keys,
            startup,
            exit,
            exits,
            scaling,
            live_edit::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT,
        )
        .await
    }

    /// [`Self::start`] with `upload_admission_max_waiters` as
    /// `HeldMode::upload_admission_max_waiters` in place of the default.
    pub(crate) async fn start_with_upload_waiters(
        valkey: ValkeyClient,
        keys: &KeySchema,
        startup: HoldObservation,
        upload_admission_max_waiters: usize,
    ) -> Self {
        let (calls, exits) = mpsc::unbounded_channel();
        Self::start_full(
            valkey,
            keys,
            startup,
            Arc::new(RecordingExit { calls }),
            exits,
            Arc::new(ScalingEndpoint::new()),
            upload_admission_max_waiters,
        )
        .await
    }

    async fn start_full(
        valkey: ValkeyClient,
        keys: &KeySchema,
        startup: HoldObservation,
        exit: Arc<dyn HoldExit>,
        exits: mpsc::UnboundedReceiver<(i32, Instant)>,
        scaling: Arc<ScalingEndpoint>,
        upload_admission_max_waiters: usize,
    ) -> Self {
        assert!(startup.verdict.is_held(), "fixture: held mode starts held");
        let metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
        let http_listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the metrics port");
        let http = http_listener.local_addr().expect("metrics addr");
        let metrics_server = tokio::spawn(serve_metrics_with_scaling(
            http_listener,
            metrics.prometheus_exporter(),
            Arc::clone(&scaling),
        ));
        let (reporter, _bootstrap_health) = create_health_service(READINESS).await;
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the gRPC port");
        let grpc = listener.local_addr().expect("gRPC addr");
        let (signal_tx, signal_rx) = oneshot::channel::<()>();
        let mode = HeldMode {
            valkey,
            keys: keys.clone(),
            poll: HoldPollSettings::new(POLL_INTERVAL, COMMAND_TIMEOUT_MS),
            startup,
            metrics: Arc::clone(&metrics),
            metrics_layer: MetricsLayer::new(Arc::clone(&metrics), SERVICE_LABEL, 30_000, 2_000),
            health_reporter: reporter.clone(),
            readiness_service_name: READINESS.to_owned(),
            health_probe_interval: HEALTH_PROBE_INTERVAL,
            health_probe_timeout: Duration::from_secs(1),
            service_token_provider: None,
            scaling_endpoint: scaling,
            process_started: std::time::Instant::now(),
            catchup_read_concurrency: 3,
            upload_admission_max_waiters,
            reactor: HoldReactorSettings {
                heartbeat_interval: Duration::from_secs(300),
                exit,
                verdict_stale: watch::Sender::new(false),
                held: watch::Sender::new(false),
            },
        };
        let run = tokio::spawn(run_held_mode(
            mode,
            TcpIncoming::from(listener),
            async move {
                let _ = signal_rx.await;
            },
        ));
        Self {
            grpc,
            http,
            metrics,
            reporter,
            exits,
            signal: Some(signal_tx),
            run,
            metrics_server,
        }
    }

    pub(crate) async fn channel(&self) -> Channel {
        tokio::time::timeout(
            SETTLE,
            Channel::from_shared(format!("http://{}", self.grpc))
                .expect("valid endpoint")
                .connect(),
        )
        .await
        .expect("connect within the deadline")
        .expect("connect to the held server")
    }

    pub(crate) async fn http_get(&self, path: &str) -> reqwest::Response {
        tokio::time::timeout(
            SETTLE,
            reqwest::Client::new()
                .get(format!("http://{}{path}", self.http))
                .send(),
        )
        .await
        .unwrap_or_else(|_| panic!("GET {path} within the deadline"))
        .unwrap_or_else(|e| panic!("GET {path} must not fail at the transport level: {e}"))
    }

    /// `GET /metrics`, as Prometheus scrapes it.
    pub(crate) async fn scrape(&self) -> String {
        let resp = self.http_get("/metrics").await;
        assert_eq!(resp.status().as_u16(), 200);
        resp.text().await.expect("a /metrics body")
    }

    /// `GET /scaling/v1`, as the scaler polls it.
    pub(crate) async fn scaling(&self) -> ScalingSnapshot {
        let resp = self.http_get("/scaling/v1").await;
        assert_eq!(
            resp.status().as_u16(),
            200,
            "held mode registers its scaling source"
        );
        resp.json::<ScalingSnapshot>()
            .await
            .expect("the body is a scaling snapshot")
    }

    /// `live_endpoint_calls_total{endpoint, grpc_code}` on held mode's
    /// metrics layer.
    pub(crate) fn calls(&self, endpoint: &str, code: &str) -> u64 {
        sample_value(
            &render(&self.metrics),
            "live_endpoint_calls_total",
            &[
                ("endpoint", endpoint),
                ("service", SERVICE_LABEL),
                ("grpc_code", code),
            ],
        )
        .unwrap_or(0.0) as u64
    }

    /// Wait, under one deadline, until every one of `endpoints` has recorded
    /// one `UNAVAILABLE`: each completion line is emitted before its call is
    /// recorded.
    pub(crate) async fn await_refused(&self, endpoints: &[&str]) {
        let deadline = Instant::now() + SETTLE;
        for endpoint in endpoints {
            while self.calls(endpoint, "UNAVAILABLE") < 1 {
                assert!(
                    Instant::now() < deadline,
                    "{endpoint}: no UNAVAILABLE recorded within {SETTLE:?}"
                );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }

    /// Fire the shutdown signal (in production, the process shutdown request
    /// held mode claimed at its entry, cancelled on the first SIGTERM or
    /// SIGINT).
    pub(crate) fn signal(&mut self) {
        let _ = self.signal.take().expect("signalled once").send(());
    }

    /// Wait for `run_held_mode` to return.
    pub(crate) async fn returned(
        &mut self,
        within: Duration,
        what: &str,
    ) -> Result<(), HeldModeError> {
        tokio::time::timeout(within, &mut self.run)
            .await
            .unwrap_or_else(|_| panic!("{what}: run_held_mode did not return within {within:?}"))
            .unwrap_or_else(|e| panic!("{what}: the run_held_mode task failed: {e}"))
    }

    /// The next exit-seam call's code, within `within`.
    pub(crate) async fn next_exit(&mut self, within: Duration, what: &str) -> i32 {
        tokio::time::timeout(within, self.exits.recv())
            .await
            .unwrap_or_else(|_| panic!("{what}: the exit seam was not called within {within:?}"))
            .unwrap_or_else(|| panic!("{what}: the reactor dropped its exit seam uncalled"))
            .0
    }

    pub(crate) fn assert_no_exit_yet(&mut self, what: &str) {
        if let Ok(call) = self.exits.try_recv() {
            panic!("{what}: held mode must not exit; got {call:?}");
        }
    }
}

impl Drop for HeldModeRun {
    fn drop(&mut self) {
        self.run.abort();
        self.metrics_server.abort();
    }
}

/// The startup evaluation against the fixture's store: `Held`.
pub(crate) async fn startup_held(tv: &TestValkey) -> HoldObservation {
    let observation = tokio::time::timeout(
        SETTLE,
        evaluate_hold_at_startup(tv.client(), tv.key_schema(), answering(COMMAND_TIMEOUT_MS)),
    )
    .await
    .expect("the startup evaluation returns")
    .expect("the startup evaluation answers");
    assert!(
        observation.verdict.is_held(),
        "fixture: the marker holds the fleet"
    );
    observation
}
