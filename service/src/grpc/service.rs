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

use std::pin::Pin;
use std::sync::Arc;

use dashmap::DashMap;
use futures::Stream;
use tonic::{Request, Response, Status, Streaming};

use super::graceful_shutdown::GracefulShutdown;
use super::ping::handle_ping;
use super::proto::{
    live_service_server::LiveService, CreateLayerRequest, CreateLayerResponse,
    CreateSessionRequest, CreateSessionResponse, DeleteLayerRequest, DeleteLayerResponse,
    DeleteSessionRequest, DeleteSessionResponse, JoinSessionRequest, JoinSessionResponse,
    ListSessionsRequest, ListSessionsResponse, PingRequest, PingResponse, StatLayerRequest,
    StatLayerResponse, SubscribeReadLayerRequest, SubscribeReadLayerResponse, UpdateLayerRequest,
    UpdateLayerResponse,
};
use super::session_cache::SessionBasePathCache;
use super::upload_admission::UploadAdmissionGate;
use super::{layer, presence, session, subscribe};
use crate::auth::permissions::PermissionsClient;
use crate::auth::service_token::ServiceTokenProvider;
use crate::auth::userinfo::UserinfoClient;
use crate::checkpoint::{BaseBootstrap, RecoveryHook, SessionRecoveryHook};
use crate::observability::metrics::SharedMetrics;
use crate::presence::LocalPresenceRouter;
use crate::storage::client::ValkeyClient;
use crate::storage::external_values::ExternalValueStore;
use crate::storage::keys::KeySchema;
use crate::storage::layer::LayerStore;
use crate::storage::session::SessionStore;
use crate::storage::snapshot_flights::SnapshotFlights;

/// Boxed server-streaming output type.
type BoxStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

/// The Live Edit gRPC service implementation.
#[derive(Clone)]
pub struct LiveEditService {
    pub session_store: Arc<SessionStore>,
    pub layer_store: Arc<LayerStore>,
    pub ext_value_store: Arc<dyn ExternalValueStore>,
    /// Per-session cancellation tokens, each carrying why it was cancelled
    /// ([`session::SessionCancel`]): an explicit `DeleteSession`, or the
    /// session found missing from the live store. Streams choose their ending
    /// from that cause (SDD §12.6).
    pub session_tokens: Arc<DashMap<String, session::SessionCancel>>,
    pub userinfo_client: Arc<dyn UserinfoClient>,
    pub permissions_client: Arc<dyn PermissionsClient>,
    pub max_layer_size: usize,
    /// The per-replica upload memory budget (SDD §5.4): `CreateLayer` and
    /// `UpdateLayer` reserve from it after authorization and before reading
    /// their content, and hold the reservation until the call ends. Built in
    /// `main.rs` from the budget derived from the pod memory limit,
    /// `LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS` and
    /// `LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS`.
    pub upload_admission: Arc<UploadAdmissionGate>,
    pub valkey: ValkeyClient,
    pub key_schema: KeySchema,
    pub presence_router: Arc<LocalPresenceRouter>,
    /// Presence lease TTL in seconds. Resolved from
    /// `LIVE_EDIT_PRESENCE_LEASE_TTL_SECS` (b.w29). JoinSession passes it to
    /// the presence register and refresh scripts as the lease's `EX`, to the
    /// unregister script as the teardown marker's `EX`, and adds it to the
    /// store's clock to form the register's deadline.
    pub presence_lease_ttl_secs: u64,
    /// Presence lease refresh interval in seconds. Resolved from
    /// `LIVE_EDIT_PRESENCE_REFRESH_INTERVAL_SECS` (b.w29). The tick interval
    /// of each JoinSession connection's lease keeper.
    pub presence_refresh_interval_secs: u64,
    /// JoinSession client keepalive interval in milliseconds; `0` is off.
    /// Resolved from `LIVE_EDIT_PRESENCE_KEEPALIVE_INTERVAL_MS`. A join whose
    /// header set `client_keepalive` is told it in `SessionConnected` and is
    /// ended after three intervals with no data from the client (SDD §13.2).
    pub presence_keepalive_interval_ms: u64,
    /// Tracker for every JoinSession task and every connection's lease keeper
    /// (b.w29 item 4). Shutdown awaits the tracker so each join's teardown —
    /// whose unregister script publishes `left` to peers — completes before
    /// the post-server sequence calls `ValkeyClient::quit()` and tears down
    /// the pools. Without this, the bounded shutdown sequence dropped
    /// outstanding `run_join_session` futures mid-cleanup and peers saw
    /// phantom members until the lease TTL expired.
    pub presence_cleanup_tracker: tokio_util::task::TaskTracker,
    /// Tracker for every `SubscribeReadLayer` subscription task. The
    /// post-server shutdown sequence closes it and waits for it
    /// ([`drain_stream_tasks`](super::transport::drain_stream_tasks)), bounded
    /// by the per-task shutdown budget and alongside the presence drain,
    /// before `ValkeyClient::quit()`, so every subscription ends with its own
    /// shutdown status while the pools are still up.
    ///
    /// The subscription task is what is tracked: it owns the RPC's admission
    /// and every consumer it opens, including across re-bootstraps, and drops
    /// each one — unregistering it from the stream multiplexer — on its way
    /// out.
    pub subscription_tracker: tokio_util::task::TaskTracker,
    /// The graceful-shutdown handle (SDD §5.6, SR-4.7): the process-wide
    /// service shutdown token, the `begun` token and the stream enrollment
    /// registry ([`GracefulShutdown`]). Created with the service in `main.rs`
    /// (and in test builders). Streams keep serving through the readiness
    /// drain; just behind the server's GOAWAY (t0)
    /// [`end_streams_after_drain`](super::transport::end_streams_after_drain)
    /// cancels `begun` and ends the enrolled streams in paced, shuffled
    /// batches, then cancels the service token, once and for good.
    ///
    /// It is the only signal graceful shutdown sends to the streaming RPCs;
    /// shutdown never cancels a per-session token, which therefore means only
    /// that the session is gone. Both streaming RPCs take their shutdown token
    /// only by enrolling ([`GracefulShutdown::enroll_stream`]), a per-RPC child
    /// so their hot paths never contend on a shared token's locks:
    ///
    /// * `SubscribeReadLayer` enrolls before its admission and reads its token
    ///   at admission, at every send and in its idle wait, and ends
    ///   `UNAVAILABLE "server shutting down"` in every phase.
    /// * `JoinSession`'s handler refuses once `begun` is cancelled; the join
    ///   task enrolls at its start and checks its token at once, then reads it
    ///   (`JoinCtx::shutdown` in `presence.rs`) everywhere later: the check
    ///   after `acquire_session_cancel`, the lease keeper, the bidirectional
    ///   loop and its event send. Each ends the join `UNAVAILABLE "server
    ///   shutting down"`, after its teardown where one is owed.
    ///
    /// `CreateLayer` and `UpdateLayer` race `begun` in their admission and
    /// processing waits and at the external-value release budget
    /// (`ExtReleaseGate::admit`), and a chained `UpdateLayer` in its
    /// chain-gate wait (`storage::chain_gate`), so an upload waiting or
    /// arriving at t0 is refused with nothing applied.
    pub shutdown: GracefulShutdown,
    pub metrics: SharedMetrics,
    pub service_label: String,
    /// Late-join / catch-up snapshot switch, hoisted from
    /// [`Config::catchup_snapshot_enabled`](crate::config::Config::catchup_snapshot_enabled)
    /// (SDD §12.3 catch-up regimes). `true` (the default) routes eligible
    /// catch-up through the corrected late-join snapshot path; `false` selects
    /// byte-identical legacy catch-up behavior. Read by `SubscribeReadLayer`
    /// (`subscribe.rs`) to choose the catch-up entrypoint; wired per-knob rather
    /// than via a whole `Config` reference, matching `max_layer_size` /
    /// `service_label`.
    pub catchup_snapshot_enabled: bool,
    /// Short-TTL positive cache of `session_id → base_path` (b.jr8
    /// mediums). Invalidated on local `DeleteSession` and on
    /// `le:session:deleted` pub/sub receipt, so the staleness ceiling is
    /// the TTL defined in [`SessionBasePathCache`].
    pub session_base_path_cache: SessionBasePathCache,
    /// Lazy on-demand layer-state recovery hook (SDD SR-2.4 / SR-2.6).
    /// Constructed in `main.rs` from the checkpoint-store +
    /// [`LayerStore`] + [`Config`](crate::config::Config) + metrics. The
    /// three Subtask-2 / Subtask-3 gRPC handlers call
    /// [`RecoveryHook::try_restore_layer`](crate::checkpoint::RecoveryHook::try_restore_layer)
    /// when their `obj:hdr` lookup returns `NotFound`. Threaded as
    /// `Arc<RecoveryHook>` so the wiring is zero-cost when checkpointing
    /// is disabled (the hook short-circuits `NotRecoverable` without
    /// touching the admission map or the store).
    pub recovery: Arc<RecoveryHook>,
    /// Lazy on-demand session-state recovery hook (SDD SR-2.7, Epic 2).
    /// Constructed in `main.rs` from the checkpoint-store, Valkey client,
    /// [`Config`](crate::config::Config), and metrics handle. Threaded into
    /// [`SessionBasePathCache`] via
    /// [`SessionBasePathCache::with_recovery_hook`] — the single choke
    /// point that every session-resolving gRPC handler funnels through
    /// (CreateLayer, UpdateLayer, DeleteLayer, StatLayer, DeleteSession,
    /// JoinSession, SubscribeReadLayer). Wiring at the cache layer covers
    /// all seven handlers with one edit; `ListSessions` bypasses the cache
    /// and is wired separately.
    ///
    /// Composes with the per-`object_id` [`Self::recovery`] hook above:
    /// inside `UpdateLayer`, `get_base_path` (and therefore session
    /// recovery) runs BEFORE the layer-mapping lookup, so the ordering
    /// invariant (session-recovery FIRST, layer-recovery SECOND) is
    /// already correct by construction.
    pub session_recovery: Arc<SessionRecoveryHook>,
    /// Creation-time first-base bootstrap (SDD SR-7.1, Epic 3). Constructed
    /// in `main.rs` from the checkpoint-store, Valkey client, key schema, the
    /// shared session-checkpoint `TaskTracker`, and metrics.
    /// [`handle_create_layer`](super::layer::handle_create_layer) calls
    /// [`BaseBootstrap::spawn_creation_base`](crate::checkpoint::BaseBootstrap::spawn_creation_base)
    /// post-commit so every layer holds the lineage root tail capture requires
    /// from its first sequence, before any periodic tick has run for it.
    ///
    /// Constructed unconditionally with the three-way gate
    /// (`checkpoint_enabled` && adapter non-`Disabled` &&
    /// `checkpoint_tail_enabled`) resolved INSIDE the unit — the same posture
    /// as [`Self::recovery`] / [`Self::session_recovery`], so the handler
    /// never branches on configuration. Threaded as `Arc<BaseBootstrap>`
    /// so the wiring is zero-cost when the feature is off (the spawn
    /// short-circuits before any allocation, Valkey command, or metric).
    pub base_bootstrap: Arc<BaseBootstrap>,
    /// Cached service-identity Bearer token provider (Epic `t1.fgj.dp`,
    /// SDD SR-6.1 / SR-6.2). Constructed in `main.rs` when
    /// `LIVE_EDIT_SERVICE_TOKEN_FILE` is set; `None` otherwise. In the
    /// binary, `None` never coexists with the `StorageApi` checkpoint adapter:
    /// `Config::from_env` refuses that adapter without the token file, and
    /// `build_checkpoint_store` refuses it again without a provider. Threaded as
    /// `Option<Arc<ServiceTokenProvider>>` so the field is reachable from
    /// the Epic 5 `StorageApi` adapter call sites without forcing every
    /// configuration that doesn't need outbound identity to allocate one.
    pub service_token_provider: Option<Arc<ServiceTokenProvider>>,
    /// This replica's shared snapshot-read flights (Spec SDD SR-6;
    /// [`SnapshotFlights`]): a `SubscribeReadLayer` catch-up of a layer above
    /// the atomic-read limits attaches to its object's flight, so one fenced
    /// read serves every catch-up of that layer on this replica. Built once
    /// per service, over the service's own [`Self::valkey`] and
    /// [`Self::key_schema`]; clones share it.
    pub snapshot_flights: SnapshotFlights,
    /// Test-only synchronization seam for `run_catch_up` (b.fdy). When
    /// `Some`, the in-window regime of `run_catch_up` pauses between
    /// `read_object_header` and the in-window stream read (snapshot mode's
    /// point read, rollback mode's paged replay read) long
    /// enough for a test to fire a `create_layer` overwrite while the
    /// subscriber is parked at the seam. This is the only way to deterministically
    /// exercise the literal `catch_up_in_window` mid-window `Deleted`
    /// variant (`src/grpc/subscribe.rs:345`) — its sibling steady-state
    /// handler (`subscribe.rs:504`) is already covered by the b.q2f
    /// pool-saturation test, but both funnel through the same
    /// `shutdown_and_release → handle_deleted_and_rebootstrap → catch_up_brand_new`
    /// chain, so the literal path had no direct regression
    /// test until this seam landed.
    ///
    /// **Test-only.** Production call sites never populate this — it is
    /// gated behind `#[cfg(any(test, feature = "test-fault-injection"))]`
    /// so release builds do not carry the seam or the field.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub catch_up_xrange_seam_for_test: Option<Arc<super::subscribe::CatchUpXrangeSeam>>,
    /// Test-only override for
    /// [`OUTER_SEND_DEADLINE`](crate::storage::notifications::OUTER_SEND_DEADLINE),
    /// the per-send bound `SubscribeReadLayer` puts on handing a message to the
    /// gRPC response channel (b.f4f). `None` keeps the production 5 s.
    ///
    /// Mirrors `StreamMuxKnobs::send_watchdog` (the stream multiplexer's
    /// test-overridable delivery watchdog) and exists for the same reason: a slow-reader regression test would
    /// otherwise spend 5 s of wall clock per run. The production constant is
    /// pinned by its own `const` assertion against `CONSUMER_SEND_WATCHDOG`, so
    /// overriding it here cannot mask a drift in the relationship between the
    /// two.
    ///
    /// **Test-only.** Production call sites never populate this — gated behind
    /// `#[cfg(any(test, feature = "test-fault-injection"))]` so release builds
    /// carry neither the field nor the branch that reads it.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub outer_send_deadline_for_test: Option<std::time::Duration>,
    /// Test-only synchronization seam inside `deliver_catch_up_content`'s chunk
    /// loop (b.f4f).
    ///
    /// That loop serves the full-state, in-window-snapshot and out-of-window
    /// regimes. It is NOT the only chunk loop on the subscribe path — the legacy
    /// in-window replay chunks inside `send_per_commit_payload` — so a test running
    /// with `LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false` on an in-window
    /// reconnect never reaches this seam.
    ///
    /// When `Some`, `deliver_catch_up_content` parks immediately after handing
    /// its **first** chunk to `send_or_disconnect`, so a test can cancel the
    /// service shutdown token, or fire a `DeleteSession` (or cancel the session
    /// token directly), at a point where content has been delivered but the
    /// caught-up ack has not. Releasing the seam then lets the next
    /// `send_or_disconnect` — the next chunk, or the ack itself — observe the
    /// cancellation.
    ///
    /// This is the catch-up counterpart of the steady-state mid-send cancel arm,
    /// and it is the case CR1-2 was about: before b.f4f that cancellation was
    /// invisible until a steady-state loop boundary, so a truncated OMPF sequence
    /// was presented to the client as a clean `OK`. A test driving this seam can
    /// assert the stream ends `UNAVAILABLE` (shutdown) or `NOT_FOUND`
    /// (deletion), that no cursor advanced over undelivered content, and that
    /// the admission, the stream registration and the task are released
    /// promptly. A catch-up parked here holds no catch-up slot: it released its
    /// slot before its first send.
    ///
    /// **Test-only.** Production call sites never populate this — gated behind
    /// `#[cfg(any(test, feature = "test-fault-injection"))]` so release builds
    /// carry neither the field nor the branch that reads it.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub catch_up_chunk_seam_for_test: Option<Arc<super::subscribe::CatchUpChunkSeam>>,
    /// Test-only synchronization seam for `run_catch_up_inner` (b.xmk / b.msb).
    /// When `Some`, the subscriber parks after the layer-mapping read (past the
    /// overwrite-detect branch) and before the same-object
    /// `read_object_header(OLD)` call — the "Site B" position — long enough for
    /// a test to fire a `create_layer` overwrite + drain its cleanup body
    /// (`cleanup_old_object_retrying`) while the subscriber is parked at the seam. This is the only way to
    /// deterministically exercise the b.jvx Site B `ServiceError::NotFound →
    /// handle_deleted_and_rebootstrap` classifier. The b.msb two-phase
    /// `parked`/`released` gate (see `RunCatchUpInnerSeam`) removes the race
    /// that made the old single-barrier seam deadlock — the test fires its
    /// overwrite only after the subscriber has parked (and thus already read
    /// `mapping == OLD`), so the park is always reached.
    ///
    /// **Test-only.** Production call sites never populate this — it is
    /// gated behind `#[cfg(any(test, feature = "test-fault-injection"))]`
    /// so release builds do not carry the seam or the field.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub run_catch_up_inner_seam_for_test: Option<Arc<super::subscribe::RunCatchUpInnerSeam>>,
    /// Test-only synchronization seam for `catch_up_out_of_window` (b.xmk).
    /// When `Some`, the subscriber pauses at one of two distinct points
    /// inside `catch_up_out_of_window` — `pre-header-read` (Site C arm 1)
    /// or `pre-compute_diff` (Site C arm 2) — selected by the constructor
    /// used (`CatchUpOutOfWindowSeam::new_before_header_read` /
    /// `new_before_compute_diff`). This is the only way to deterministically
    /// exercise the b.jvx Site C `ServiceError::NotFound →
    /// handle_deleted_and_rebootstrap` classifier arms — without the seam,
    /// the race window between the reader reservation and the state read
    /// closes faster than a test can drive an overwrite. Both points sit
    /// before the catch-up's state read (its read permit and atomic read, or
    /// its attach to the object's shared fenced-read flight), back to back,
    /// since that one read covers the header and the state.
    ///
    /// **Test-only.** Production call sites never populate this — it is
    /// gated behind `#[cfg(any(test, feature = "test-fault-injection"))]`
    /// so release builds do not carry the seam or the field.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub catch_up_out_of_window_seam_for_test: Option<Arc<super::subscribe::CatchUpOutOfWindowSeam>>,
    /// Test-only synchronization seam for the catch-up **state read** (b.vd8).
    /// When `Some`, the subscriber parks between its first header read (P) and
    /// the state read whose result it serves — before the read permit, the
    /// atomic read (or the attach to the object's shared fenced-read flight)
    /// and the stream registration at its cut — at either the Regime 1
    /// (`pre-full-state`) or Regime 2 (`pre-coalesced`) position — or, in
    /// rollback mode's Regime 2 per-commit replay, after its first per-commit
    /// read and before the next (`between-replay-computes`, a two-phase park
    /// the test releases; empty commits take no read and are skipped) —
    /// selected by the constructor used
    /// (`CatchUpStateReadSeam::new_before_full_state` / `new_before_coalesced` /
    /// `new_between_replay_computes`). A fourth point (b.yuo,
    /// `new_before_ext_probe`) parks the first catch-up build of any regime
    /// after its state read and before it reads any external-value blob, so a
    /// test can stage the stale-miss race. Every point holds the catch-up
    /// slot, except the Regime 1 point of a catch-up that found its object's
    /// flight registered, which parks holding none (it attaches without one),
    /// and the pre-ext-probe point of a catch-up building from its flight's
    /// shared snapshot, which holds its build turn. Regime 3's equivalent of
    /// the first two points already exists on `CatchUpOutOfWindowSeam`.
    ///
    /// This is the only way to deterministically land a commit inside the
    /// catch-up/live overlap window: in production it closes in well under an
    /// RTT, so without the seam a test cannot reliably produce the interleaving
    /// the stable cut exists to handle.
    ///
    /// **Test-only.** Production call sites never populate this — it is
    /// gated behind `#[cfg(any(test, feature = "test-fault-injection"))]`
    /// so release builds do not carry the seam or the field.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub catch_up_state_read_seam_for_test: Option<Arc<super::subscribe::CatchUpStateReadSeam>>,
    /// Test-only park point inside a catch-up's **build turn** in its shared
    /// fenced-read flight: after the flight's read, the catch-up's regression
    /// check and its stream registration, once it holds its turn (the
    /// flight's own, or an extra build's) and before it builds, so the turn
    /// guard, the snapshot handle and its place in the flight stay held while
    /// it is parked. Selects the catch-ups it parks by `client_id`, or as the
    /// first N to reach a turn
    /// ([`CatchUpBuildTurnSeam`](super::subscribe::CatchUpBuildTurnSeam)),
    /// so a test can hold a build chain deterministically: a second wave
    /// queueing behind it, a burst's extra builds, a cancellation inside a
    /// turn. The park races the subscription's endings, so a parked catch-up
    /// still ends on shutdown, session cancellation or client disconnect.
    ///
    /// **Test-only.** Production call sites never populate this — gated behind
    /// `#[cfg(any(test, feature = "test-fault-injection"))]` so release builds
    /// carry neither the field nor the branch that reads it.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub catch_up_build_turn_seam_for_test: Option<Arc<super::subscribe::CatchUpBuildTurnSeam>>,
    /// Test-only recorder of every `SubscribeReadLayer` capacity-budget charge:
    /// the admission wait, each stream-reader wait, each catch-up slot wait,
    /// each snapshot read-permit wait and each wait for a shared fenced-read
    /// flight's read (charged the catch-up's share of the flight's capacity
    /// waits), with what the budget held before and after and the deadline
    /// the wait was given. See
    /// [`CapacityBudgetSeam`](super::subscribe::CapacityBudgetSeam). It only
    /// observes: nothing it records feeds back into a wait.
    ///
    /// **Test-only.** Production call sites never populate this — gated behind
    /// `#[cfg(any(test, feature = "test-fault-injection"))]` so release builds
    /// carry neither the field nor the branches that read it.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub capacity_budget_seam_for_test: Option<Arc<super::subscribe::CapacityBudgetSeam>>,
    /// Test-only park points in a JoinSession's registration (SDD SR-4 steps
    /// 2–6): after the first session-index add, between the register's
    /// store-clock read and the register itself, and before
    /// `acquire_session_cancel`. See
    /// [`JoinRegistrationSeam`](super::presence::JoinRegistrationSeam).
    ///
    /// **Test-only.** Production call sites never populate this — gated behind
    /// `#[cfg(any(test, feature = "test-fault-injection"))]` so release builds
    /// carry neither the field nor the branches that read it.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub join_registration_seam_for_test: Option<Arc<super::presence::JoinRegistrationSeam>>,
    /// Test-only park points in, and observations of, a JoinSession's
    /// membership snapshot and replay (SDD SR-4 steps 4 and 10, SR-4.4) and
    /// of the replay-delivered set D the bidi loop uses to suppress duplicate
    /// `PeerJoined`, plus an override of D's time limit. The park points,
    /// observations and override are listed on
    /// [`PresenceReplaySeam`](super::presence::PresenceReplaySeam).
    ///
    /// **Test-only.** Gated as above.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub presence_replay_seam_for_test: Option<Arc<super::presence::PresenceReplaySeam>>,
    /// Test-only park points in a JoinSession connection's lease keeper (SDD
    /// SR-4.2): inside a step before its refresh, and inside a self-rejoin
    /// after its store-clock read, before the rejoin register. See
    /// [`LeaseKeeperSeam`](super::presence::LeaseKeeperSeam).
    ///
    /// **Test-only.** Gated as above.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub lease_keeper_seam_for_test: Option<Arc<super::presence::LeaseKeeperSeam>>,
    /// Test-only park points around a session's Valkey channel `SUBSCRIBE`
    /// (in the subscribing join) and `UNSUBSCRIBE` (in the last local
    /// leaver's teardown), and a one-shot injected `SUBSCRIBE` failure. See
    /// [`PresenceChannelSeam`](super::presence::PresenceChannelSeam).
    ///
    /// **Test-only.** Gated as above.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub presence_channel_seam_for_test: Option<Arc<super::presence::PresenceChannelSeam>>,
}

#[tonic::async_trait]
impl LiveService for LiveEditService {
    // ── Session management ────────────────────────────────────────────────────

    async fn create_session(
        &self,
        request: Request<CreateSessionRequest>,
    ) -> Result<Response<CreateSessionResponse>, Status> {
        session::handle_create_session(self, request).await
    }

    async fn delete_session(
        &self,
        request: Request<DeleteSessionRequest>,
    ) -> Result<Response<DeleteSessionResponse>, Status> {
        session::handle_delete_session(self, request).await
    }

    async fn list_sessions(
        &self,
        request: Request<ListSessionsRequest>,
    ) -> Result<Response<ListSessionsResponse>, Status> {
        session::handle_list_sessions(self, request).await
    }

    // ── Presence ──────────────────────────────────────────────────────────────

    type JoinSessionStream = BoxStream<JoinSessionResponse>;

    async fn join_session(
        &self,
        request: Request<Streaming<JoinSessionRequest>>,
    ) -> Result<Response<Self::JoinSessionStream>, Status> {
        presence::handle_join_session(self, request).await
    }

    // ── Layer operations ──────────────────────────────────────────────────────

    async fn create_layer(
        &self,
        request: Request<Streaming<CreateLayerRequest>>,
    ) -> Result<Response<CreateLayerResponse>, Status> {
        layer::handle_create_layer(self, request).await
    }

    async fn update_layer(
        &self,
        request: Request<Streaming<UpdateLayerRequest>>,
    ) -> Result<Response<UpdateLayerResponse>, Status> {
        layer::handle_update_layer(self, request).await
    }

    async fn delete_layer(
        &self,
        request: Request<DeleteLayerRequest>,
    ) -> Result<Response<DeleteLayerResponse>, Status> {
        layer::handle_delete_layer(self, request).await
    }

    async fn stat_layer(
        &self,
        request: Request<StatLayerRequest>,
    ) -> Result<Response<StatLayerResponse>, Status> {
        layer::handle_stat_layer(self, request).await
    }

    // ── Subscriptions ─────────────────────────────────────────────────────────

    type SubscribeReadLayerStream = BoxStream<SubscribeReadLayerResponse>;

    async fn subscribe_read_layer(
        &self,
        request: Request<SubscribeReadLayerRequest>,
    ) -> Result<Response<Self::SubscribeReadLayerStream>, Status> {
        subscribe::handle_subscribe_read_layer(self, request).await
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    async fn ping(&self, request: Request<PingRequest>) -> Result<Response<PingResponse>, Status> {
        Ok(Response::new(handle_ping(request.into_inner())))
    }
}
