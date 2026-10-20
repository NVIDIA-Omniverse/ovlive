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

//! b.y3m Epics 2–3 — brand-new catch-ups of large layers under concurrent
//! writers, served from one proven read (Spec SDD SR-3.1–SR-3.3, SR-6, Test
//! Fixtures "Structural double-apply regression"; PRD FR-3), driven
//! end-to-end through `handle_subscribe_read_layer`.
//!
//! What is pinned here:
//!
//! * **No structural double-apply.** Brand-new subscribers of a 2,000-section
//!   layer join one after another while a writer alternates `CreateSection` of
//!   a new section and `MoveSection` of an older section under it; each
//!   catch-up is held after its header read P while one more such pair
//!   commits, and a field writer keeps the header moving through its read.
//!   Each subscriber
//!   applies everything it is sent to a local model with the domain
//!   `apply_delta` — the server's own engine — and every model applies cleanly
//!   and ends equal to the server's final state, in both catch-up modes, and
//!   both below the atomic-read ceiling (each joiner's own atomic read) and
//!   above it (each joiner's layer's shared fenced read, [`Ceiling::ForcedLow`]).
//!   A commit delivered twice fails that: a re-applied `CreateSection` is a
//!   `DuplicateSection`.
//! * **Large layers under writers end verified.** Brand-new catch-ups of a
//!   2,150-section layer (the join-burst layer's shape, below the shipped
//!   atomic-read ceiling) joining under a writer that commits back to back all
//!   record `verified`, each from one atomic read, with no retry; each acks the
//!   cut its content carries, and the first commit it is sent live is the one
//!   right after that cut.
//! * **The writer's stall diagnostic measures what it reports.** The
//!   [`WakeToPoll`] adapter, driven by hand, records a poll held off after a
//!   wake as not polled, does not count the wait before a wake, records time
//!   inside a poll as polling, and passes wakes and the output through. The
//!   runtime probe ([`run_tick_probe`]), on a one-thread runtime the test
//!   holds, counts a tick held past its due as late, before and after it is
//!   polled, and does not count its on-time waits.
//!
//! # Fails on the pre-change code
//!
//! At e1220ea0 a brand-new catch-up re-read its header (P′) under its slot,
//! registered its stream at P′, and only then reached the state-read seam's
//! park point and read its state, proving it with the quiescence loop — which
//! loses whenever the header moves during every attempt. Unproven, it acked
//! P′, so a commit landed after P′ and folded into its state was delivered
//! again live.
//!
//! Both tests were run against e1220ea0 on 2026-10-01, from a port of this
//! file in a detached worktree of that commit. The port differs from this file
//! only where e1220ea0 lacks what it names: no forced-low branch in
//! `Fixture::setup` (no `snapshot_atomic_read_max_items`), no
//! `capacity_budget_seam_for_test` field, and no
//! `live_edit_snapshot_read_total` assertion in the large-layer test (no
//! atomic read); nothing else had to change to compile. The logs of these runs
//! were not retained, so the results below are their only record:
//!
//! * the double-apply test failed in both modes, on every run (two each):
//!   parked after P′ and its registration, every joiner's held pair landed
//!   above P′, the field writer defeated its proof, and the pair came back
//!   live — its `CreateSection` failed `DuplicateSection`. (Load alone does
//!   not open that window in a debug build: a structural commit on this layer
//!   is slower than a quiescence attempt, and fast-path writes fast enough to
//!   defeat the proof also defeat the structural writer's `WATCH`, which is
//!   why the window is held open with the seam.)
//! * the large-layer test failed on every run (nine): four of the nine on its
//!   cut assertion, every catch-up `unverified` (0 verified, 4 unverified); the
//!   other five at a [`SETTLE`]-bounded wait for the writer's next commits —
//!   in the one such run that was instrumented, after every joiner had
//!   reported. That stall was not traced at e1220ea0; what it most likely was
//!   is under "The writer stall" below.
//!
//! On this code, two perturbations stood in for e1220ea0 (recorded at Epic 2,
//! while a forced-low ceiling still selected the interim path):
//!
//! * acking and registering at the header read first (P) while serving the
//!   read's state fails the double-apply test with `DuplicateSection`. It is
//!   faithful because the park point now comes after P and before the read:
//!   the held pair lands above P exactly as it landed above e1220ea0's P′.
//! * a forced-low ceiling — which then sent every catch-up down the interim
//!   path, e1220ea0's catch-up unchanged — failed the large-layer test the same
//!   way, 0 verified and 4 unverified. It did NOT reproduce the double-apply:
//!   the park point came before the interim path's re-read P′, so the held pair
//!   was at or below the pin, never delivered again, and the models converged.
//!
//! Since Epic 3 a forced-low ceiling ([`Ceiling::ForcedLow`]) sends every
//! catch-up to its layer's shared fenced-read flight instead, so it no longer
//! reproduces e1220ea0; the double-apply test runs under it as its fenced-path
//! variant.
//!
//! # The writer stall: this test's own followers, not the server
//!
//! The stall's harness is unchanged here, so it was traced on this code: ten
//! runs of the large-layer test under a forced-low ceiling (which then sent
//! every catch-up down e1220ea0's interim path), a debug build, the writer's commits and the
//! followers' applies timed (2026-10-02; record and data in
//! `docs-internal/validation/writer-residual-attribution-20261002/`). No
//! [`SETTLE`] wait timed out; the slowest run's wait "after every joiner
//! reported" took 8.8 s of its 10 s. In every run the writer **slowed but
//! never stopped**: its commits completed throughout every wait and no field
//! commit stayed in flight past 0.7 s, but in the slowest run it made about
//! one every 0.45 s. That trace timed each commit by wall clock only, so it
//! could not say how much of a commit was spent woken but waiting to be
//! polled.
//!
//! What occupied the runtime's four workers then was the test's own
//! followers, not the server's catch-up builds, which end before each
//! follower's content is sent. Once caught up, each follower applies its
//! backlog of live commits back to back with the domain engine, about 20 ms
//! per commit in a debug build, and the wait lasted as long as one follower's
//! applies: 0.69, 2.22 and 8.81 s waits against 0.67, 2.21 and 8.78 s of
//! applies. That backlog is the commits the writer made during the
//! catch-ups, so a somewhat larger one would cross `SETTLE`; e1220ea0's five
//! stalls fit that, by inference, not measurement. Not shown: how the
//! runtime interleaved the writer with the followers, and that the effect
//! needs the forced-low ceiling. So the wait "after every joiner reported" is
//! no longer bounded by `SETTLE` alone: it grows by [`APPLY_ALLOWANCE`] (about
//! 3× the measured apply) per commit the followers may still have to apply —
//! every commit since the first join — up to [`FOLLOW_BUDGET`]
//! ([`backlog_wait`]). Run 10's backlog (473 commits during the catch-ups)
//! would get about 40 s, where its wait took 8.8 s.
//!
//! So the writer now measures how busy the runtime was around it, though
//! still not which task each worker ran. Two measures:
//!
//! * Each commit's future runs inside a [`WakeToPoll`] adapter, which
//!   records, without changing what the commit does, how long the commit
//!   spent woken but not yet polled and how long inside its polls. The rest
//!   of its time it awaited a wake: the store, a lock or a timer — or a wake
//!   the runtime was slow to deliver, since the I/O driver and fred's
//!   connection task, which deliver the store's replies, run on the same four
//!   workers. So time awaiting a wake does not by itself clear the runtime.
//! * A probe task beside the writer on the same runtime ([`run_tick_probe`],
//!   as the trace used) ticks every [`PROBE_TICK`] and sums how late each
//!   tick is polled. A runtime too busy to poll its tasks or to reach its
//!   timer and I/O drivers makes the probe late, whichever task it is
//!   running.
//!
//! A writer wait's timeout names its phase and reports both, with the commits
//! completed and any commit in flight ([`Writer::wait_for`],
//! [`CommitLog::diagnose`]). Its verdict is **idle** when the writer had no
//! write to make for at least half the wait, else **not scheduled** when the
//! writer's task went unpolled, or the probe was late, for at least half the
//! wait — the runtime was slow to poll its tasks; the verdict does not say
//! what occupied it — else **blocked** when one commit spans at least half
//! the wait, else **slow**: commits are completing. Under blocked or slow, the
//! figures say whether the commits' time went to awaiting a wake or to running
//! inside their polls, and the probe's lateness says how far a busy runtime
//! can account for the awaiting; neither says which await. A recurrence reads
//! as a writer stall at a named wait, not as the proof failing.
//!
//! # A second, silent activation condition
//!
//! The double-apply test holds catch-ups with the `CatchUpStateReadSeam`, so it
//! exists only under the **`test-fault-injection`** cargo feature: without it
//! all four of its tests (`..._snapshot_mode` and `..._rollback_mode`, at each
//! ceiling) are compiled out and the run reports four tests fewer, not a
//! failure.
//!
//! # Load, not timing
//!
//! The field writers commit back to back: that is the workload, not
//! synchronisation, and every assertion is on what was delivered and recorded,
//! never on how fast. In the double-apply test the structural writes are made
//! between the joins and the field writer runs from each held catch-up's
//! release until it has caught up, so the two never contend for the header's
//! `WATCH`; in the large-layer test a joiner joins once the field writer has
//! made a further [`FIELD_COMMITS_PER_JOIN`] commits, and the writer keeps
//! writing until every joiner has caught up. Each test's last commit creates a
//! sentinel section, which every follower must reach.
//!
//! # Wait budget
//!
//! [`SETTLE`] (10 s) bounds each wait for writer progress, each park, each
//! release and each catch-up, except the large-layer test's wait after every
//! joiner has reported, which [`backlog_wait`] scales with the followers'
//! backlog up to [`FOLLOW_BUDGET`]; [`FOLLOW_BUDGET`] (60 s) bounds each
//! follower from its subscribe to the sentinel, all followers running at once.
//! A stalled run fails at its first exhausted bound; the heaviest healthy waits
//! sum far below nextest's 120 s hang-killer. The healthy path is a debug
//! build's layer creation and structural commits (each structural commit reads
//! and applies the whole layer): the double-apply test takes about 10 s (about
//! 20 s above the ceiling, where each joiner's fenced read holds the writers
//! off for its epoch), the large-layer test about 20 s.

use std::collections::{BTreeSet, HashMap};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Wake, Waker};
use std::time::{Duration, Instant};

use futures::{Stream, StreamExt};
use tonic::{Request, Status};

use live_edit::domain::delta::apply_delta;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::ObjectState;
use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{
    subscribe_read_layer_response, update_layer_request, SubscribeReadLayerRequest,
    SubscribeReadLayerResponse, UpdateLayerHeader, UpdateLayerRequest,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::diff_read::snapshot_to_object_state;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::read_object_state_for_test;
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand};

use crate::common::owned_delta::delta_to_ompf_bytes;

use super::helpers::{
    make_create_data, make_create_header, make_request, streaming_create, streaming_update,
    AllowAllPermissions,
};

// ── Budgets and constants ───────────────────────────────────────────────────

/// Upper bound on one wait for writer progress, one park or one catch-up.
const SETTLE: Duration = Duration::from_secs(10);

/// Upper bound on one follower, from its subscribe to the sentinel commit.
const FOLLOW_BUDGET: Duration = Duration::from_secs(60);

/// Structural writes between two joins of the double-apply test: one
/// `CreateSection` + `MoveSection` pair.
#[cfg(feature = "test-fault-injection")]
const COMMITS_PER_JOIN: u64 = 2;

/// Brand-new subscribers per test.
const JOINERS: usize = 4;

/// The layer stream's `MAXLEN`: far above every commit a test makes, so no
/// follower's position is trimmed away while it catches up.
const STREAM_MAXLEN: u64 = 1_000_000;

/// `Metrics` / `service` label of the fixture.
const SERVICE: &str = "test";

/// The section each test's last commit creates: a follower that has applied
/// it has been sent everything.
const SENTINEL_SECTION: u64 = 9_999_999;

/// Ids of sections the structural writes create start here, above the seeded
/// ones.
#[cfg(feature = "test-fault-injection")]
const NEW_SECTION_BASE: u64 = 1_000_000;

/// Ids of the sections the held pairs create start here.
#[cfg(feature = "test-fault-injection")]
const HELD_SECTION_BASE: u64 = 2_000_000;

/// The writers' `client_id`, which no subscriber uses (so self-echo never hides
/// a commit).
const WRITER: &str = "proven-read-writer";

// ── Fixture ─────────────────────────────────────────────────────────────────

/// The atomic-read item ceiling the fixture's client runs with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ceiling {
    /// The shipped default: both layers here are below it, and every
    /// catch-up is served from its own atomic read.
    Shipped,
    /// One item: every catch-up's atomic read answers `TOO_LARGE`, and the
    /// catch-up is served from its layer's shared fenced-read flight. Only the
    /// double-apply test runs it, which needs `test-fault-injection`.
    #[cfg(feature = "test-fault-injection")]
    ForcedLow,
}

/// A client with its own key prefix and one `Metrics` handle installed on both
/// the client (snapshot reads, read permits) and the service (stable cuts).
struct Fixture {
    client: ValkeyClient,
    key_schema: KeySchema,
    prefix: String,
    metrics: Arc<Metrics>,
}

impl Fixture {
    async fn setup(ceiling: Ceiling) -> Self {
        live_edit::storage::test_support::raise_fd_limit_for_tests();
        let prefix = format!("test_{}:", uuid::Uuid::new_v4());
        let mut config = crate::common::test_config();
        config.valkey_connect_timeout_secs = 30;
        match ceiling {
            Ceiling::Shipped => {}
            #[cfg(feature = "test-fault-injection")]
            Ceiling::ForcedLow => config.snapshot_atomic_read_max_items = 1,
        }
        let client = live_edit::storage::test_support::connect_test_client(&config)
            .await
            .expect("Fixture: connect must succeed");
        let metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE));
        client
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        Self {
            key_schema: KeySchema::new(&prefix),
            client,
            prefix,
            metrics,
        }
    }

    /// A service over the fixture in the given catch-up mode.
    fn service(&self, catchup_snapshot_enabled: bool) -> LiveEditService {
        let layer_store = Arc::new(live_edit::storage::layer::LayerStore::new(
            self.client.clone(),
            self.key_schema.clone(),
            STREAM_MAXLEN,
        ));
        let metrics = Arc::clone(&self.metrics);
        let recovery = super::helpers::build_disabled_recovery_hook(
            Arc::clone(&layer_store),
            Arc::clone(&metrics),
        );
        let mut session_recovery_config = crate::common::test_config();
        session_recovery_config.checkpoint_enabled = false;
        let session_recovery = Arc::new(live_edit::checkpoint::SessionRecoveryHook::new(
            Arc::new(live_edit::checkpoint::adapters::InMemoryCheckpointStore::new()),
            self.client.clone(),
            self.key_schema.clone(),
            Arc::new(session_recovery_config),
            Arc::clone(&metrics),
        ));
        LiveEditService {
            catchup_snapshot_enabled,
            session_store: Arc::new(SessionStore::new(
                self.client.clone(),
                self.key_schema.clone(),
            )),
            layer_store,
            ext_value_store: Arc::new(
                live_edit::storage::external_values::MockExternalValueStore::new(),
            ),
            session_tokens: Arc::new(dashmap::DashMap::new()),
            userinfo_client: Arc::new(super::helpers::StubUserinfo),
            permissions_client: Arc::new(AllowAllPermissions),
            max_layer_size: 10 * 1024 * 1024,
            upload_admission: std::sync::Arc::new(
                live_edit::grpc::upload_admission::UploadAdmissionGate::new(
                    Default::default(),
                    None,
                ),
            ),
            snapshot_flights: live_edit::storage::snapshot_flights::SnapshotFlights::new(
                self.client.clone(),
                self.key_schema.clone(),
            ),
            valkey: self.client.clone(),
            key_schema: self.key_schema.clone(),
            presence_router: Arc::new(live_edit::presence::LocalPresenceRouter::new()),
            presence_lease_ttl_secs: live_edit::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
            presence_refresh_interval_secs:
                live_edit::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
            presence_keepalive_interval_ms:
                live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
            presence_cleanup_tracker: tokio_util::task::TaskTracker::new(),
            subscription_tracker: tokio_util::task::TaskTracker::new(),
            shutdown: live_edit::grpc::graceful_shutdown::GracefulShutdown::new(),
            base_bootstrap: crate::common::build_disabled_base_bootstrap(
                self.client.clone(),
                self.key_schema.clone(),
                Arc::clone(&metrics),
            ),
            metrics,
            service_label: SERVICE.to_string(),
            session_base_path_cache: live_edit::grpc::session_cache::SessionBasePathCache::new(),
            recovery,
            session_recovery,
            service_token_provider: None,
            #[cfg(feature = "test-fault-injection")]
            catch_up_xrange_seam_for_test: None,
            #[cfg(feature = "test-fault-injection")]
            outer_send_deadline_for_test: None,
            #[cfg(feature = "test-fault-injection")]
            catch_up_chunk_seam_for_test: None,
            #[cfg(feature = "test-fault-injection")]
            run_catch_up_inner_seam_for_test: None,
            #[cfg(feature = "test-fault-injection")]
            catch_up_out_of_window_seam_for_test: None,
            #[cfg(feature = "test-fault-injection")]
            catch_up_state_read_seam_for_test: None,
            #[cfg(feature = "test-fault-injection")]
            catch_up_build_turn_seam_for_test: None,
            #[cfg(feature = "test-fault-injection")]
            capacity_budget_seam_for_test: None,
            #[cfg(feature = "test-fault-injection")]
            join_registration_seam_for_test: None,
            #[cfg(feature = "test-fault-injection")]
            presence_replay_seam_for_test: None,
            #[cfg(feature = "test-fault-injection")]
            lease_keeper_seam_for_test: None,
            #[cfg(feature = "test-fault-injection")]
            presence_channel_seam_for_test: None,
        }
    }
}

/// Teardown: unlink every key under the fixture's prefix, in `SCAN` order.
///
/// This is unfenced, unlike production deletion, which removes the layer
/// mapping and then unlinks the object header before the object's other keys
/// (SDD §7.11 steps 2 and 4), so no later writer can load a half-deleted
/// object.
/// When a test fails while its writers are still running, a write racing this
/// teardown can read a half-unlinked object and trip the engine's debug
/// cross-map assertion — a secondary panic printed after the real one. Read
/// the first failure; the cross-map panic from a teardown race is noise.
impl Drop for Fixture {
    fn drop(&mut self) {
        use fred::interfaces::KeysInterface;
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        if handle.runtime_flavor() != tokio::runtime::RuntimeFlavor::MultiThread {
            return;
        }
        tokio::task::block_in_place(|| {
            handle.block_on(async {
                let keys =
                    crate::common::scan_keys(&self.client, &format!("{}*", self.prefix)).await;
                for key in keys {
                    let _ = self.client.pool.unlink::<(), _>(key).await;
                }
            })
        });
    }
}

// ── Layers and commits ──────────────────────────────────────────────────────

/// A session and a layer created with `sections` sections under the root (ids
/// `1..=sections`, named `s{id}`), each with `fields` inline 16-byte fields.
/// Returns `(session_id, object_id)`.
async fn create_large_layer(
    fx: &Fixture,
    svc: &LiveEditService,
    layer: &str,
    sections: u64,
    fields: u64,
) -> (String, u64) {
    // Boxed, as `commit`'s handler call is.
    let session_id = Box::pin(
        SessionStore::new(fx.client.clone(), fx.key_schema.clone()).create(
            &format!("{layer}-session"),
            "/stage",
            "test-user",
        ),
    )
    .await
    .expect("session creation must succeed");
    let delta = OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: (1..=sections)
            .map(|id| {
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 0,
                    section_id: id,
                    section_name: format!("s{id}"),
                    section_type: 0,
                })
            })
            .collect(),
        set_fields: (1..=sections)
            .flat_map(|id| {
                (0..fields).map(move |f| SetField {
                    section_id: id,
                    key_name: format!("f{f}"),
                    value_or_ext_hash: format!("{id:08}-{f:07}").into_bytes(),
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 0,
                })
            })
            .collect(),
        ..Default::default()
    };
    // Boxed, as `commit`'s handler call is.
    let object_id = Box::pin(handle_create_layer(
        svc,
        make_request(streaming_create(vec![
            make_create_header(&session_id, layer),
            make_create_data(delta_to_ompf_bytes(&delta, &[])),
        ])),
    ))
    .await
    .expect("CreateLayer of the large layer must succeed")
    .into_inner()
    .object_id;
    (session_id, object_id)
}

/// One commit of `delta` from a writer.
async fn commit(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    object_id: u64,
    delta: &OwnedDelta,
) {
    let header = UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Header(
            UpdateLayerHeader {
                session_id: session_id.to_owned(),
                layer_name: layer.to_owned(),
                object_id,
                timestamp_microseconds_since_epoch: 1,
                client_id: WRITER.to_owned(),
                operation_id: String::new(),
            },
        )),
    };
    let data = super::helpers::make_update_data(delta_to_ompf_bytes(delta, &[]));
    // Boxed, as tonic polls a handler from the heap. Inline, the handler's
    // future is held in every helper future above it, and in a debug build
    // each of their poll frames on the test thread's stack holds copies of it;
    // the fenced-layer tests then came within 256 KiB of std's default 2 MiB
    // test-thread stack.
    Box::pin(handle_update_layer(
        svc,
        make_request(streaming_update(vec![header, data])),
    ))
    .await
    .unwrap_or_else(|e| panic!("the writer's UpdateLayer must succeed: {e}"));
}

fn create_section(parent_id: u64, section_id: u64, name: String) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id,
            section_id,
            section_name: name,
            section_type: 0,
        })],
        ..Default::default()
    }
}

/// Move seeded section `section_id` (`s{id}`, under the root) under
/// `new_parent_id`.
#[cfg(feature = "test-fault-injection")]
fn move_seeded_section(section_id: u64, new_parent_id: u64) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::MoveSection(oodf_ffi::CommandMoveSection {
            old_parent_id: 0,
            new_parent_id,
            section_id,
            new_name: format!("s{section_id}"),
        })],
        ..Default::default()
    }
}

/// The `k`th structural write between the double-apply test's joins: even
/// `k` creates section `NEW_SECTION_BASE + k/2` under the root; odd `k` moves
/// the older seeded section `k/2 + 1` under it.
#[cfg(feature = "test-fault-injection")]
fn structural_write(k: u64) -> OwnedDelta {
    let n = k / 2;
    if k.is_multiple_of(2) {
        create_section(0, NEW_SECTION_BASE + n, format!("new{n}"))
    } else {
        move_seeded_section(n + 1, NEW_SECTION_BASE + n)
    }
}

/// The field writer's `k`th commit: a new 16-byte value on a seeded section's
/// field `f0`, cycling through the sections. Field-only, so it takes the Lua
/// fast commit path and moves the header far faster than a structural commit.
fn field_write(k: u64) -> Option<OwnedDelta> {
    Some(OwnedDelta {
        set_fields: vec![SetField {
            section_id: k % DOUBLE_APPLY_SECTIONS + 1,
            key_name: "f0".to_owned(),
            value_or_ext_hash: format!("w{k:015}").into_bytes(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    })
}

/// One of a [`Writer`]'s commits: its `k`th write, or the sentinel.
#[derive(Clone, Copy, Debug)]
enum CommitName {
    Write(u64),
    Sentinel,
}

impl std::fmt::Display for CommitName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Write(k) => write!(f, "write {k}"),
            Self::Sentinel => f.write_str("the sentinel commit"),
        }
    }
}

/// How a future's time divides, as its [`WakeToPoll`] adapter measured it.
#[derive(Clone, Copy, Debug, Default)]
struct TimeSplit {
    /// From its first poll to its completion, or to now.
    total: Duration,
    /// Woken but not yet polled again: runnable, waiting for the runtime to
    /// poll its task.
    not_polled: Duration,
    /// Inside its own polls.
    polling: Duration,
}

impl TimeSplit {
    /// The rest of its time: pending and not yet woken. That is awaiting the
    /// store, a lock or a timer — or a wake the runtime was slow to deliver,
    /// since the I/O driver and fred's connection task, which deliver the
    /// store's replies, run on the same workers. So time awaiting a wake does
    /// not clear a busy runtime; [`TickLog`]'s probe is the check on that.
    fn awaiting(&self) -> Duration {
        self.total.saturating_sub(self.not_polled + self.polling)
    }

    fn plus(self, other: Self) -> Self {
        Self {
            total: self.total + other.total,
            not_polled: self.not_polled + other.not_polled,
            polling: self.polling + other.polling,
        }
    }

    fn minus(self, other: Self) -> Self {
        Self {
            total: self.total.saturating_sub(other.total),
            not_polled: self.not_polled.saturating_sub(other.not_polled),
            polling: self.polling.saturating_sub(other.polling),
        }
    }
}

/// The timestamps a [`WakeToPoll`] adapter and its wakers record.
#[derive(Default)]
struct WakeClock {
    first_poll: Option<Instant>,
    finished: Option<Instant>,
    /// The earliest wake not yet followed by a poll.
    woken: Option<Instant>,
    /// The poll in progress, if any.
    polling_since: Option<Instant>,
    not_polled: Duration,
    polling: Duration,
}

impl WakeClock {
    /// The split so far, counting a wake not yet polled and a poll in progress
    /// up to `now`.
    fn split(&self, now: Instant) -> TimeSplit {
        let end = self.finished.unwrap_or(now);
        let open = |since: Option<Instant>| since.map_or(Duration::ZERO, |t| end - t);
        TimeSplit {
            total: self.first_poll.map_or(Duration::ZERO, |t| end - t),
            not_polled: self.not_polled + open(self.woken),
            polling: self.polling + open(self.polling_since),
        }
    }
}

/// A [`WakeClock`], locked. It is never held across an await or a poll of the
/// wrapped future, and nothing panics while holding it.
fn lock_clock(clock: &Mutex<WakeClock>) -> MutexGuard<'_, WakeClock> {
    clock
        .lock()
        .expect("the wake clock's lock is never poisoned")
}

/// A test-only wrapper that times a future without changing it: how long it
/// waits from each wake to its next poll, and how long it spends inside its
/// polls ([`TimeSplit`]). Each poll passes the task's waker to the wrapped
/// future inside a [`RecordingWaker`], which notes the wake's time and then
/// wakes the task exactly as the original would; the wrapped future is polled
/// once per poll of this one, and its output is returned unchanged. Dropping
/// the adapter drops the wrapped future.
struct WakeToPoll<F> {
    inner: Pin<Box<F>>,
    clock: Arc<Mutex<WakeClock>>,
    /// The task waker last passed in and the recording waker around it,
    /// reused while the task's waker stays the same.
    waker: Option<(Waker, Waker)>,
}

impl<F: Future> WakeToPoll<F> {
    fn new(inner: F) -> Self {
        Self {
            inner: Box::pin(inner),
            clock: Arc::default(),
            waker: None,
        }
    }

    fn clock(&self) -> Arc<Mutex<WakeClock>> {
        Arc::clone(&self.clock)
    }
}

impl<F: Future> Future for WakeToPoll<F> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        let this = &mut *self;
        let start = Instant::now();
        {
            let mut clock = lock_clock(&this.clock);
            clock.first_poll.get_or_insert(start);
            if let Some(woken) = clock.woken.take() {
                clock.not_polled += start.saturating_duration_since(woken);
            }
            clock.polling_since = Some(start);
        }
        let reuse = matches!(&this.waker, Some((task, _)) if task.will_wake(cx.waker()));
        if !reuse {
            let recording = Waker::from(Arc::new(RecordingWaker {
                task: cx.waker().clone(),
                clock: Arc::clone(&this.clock),
            }));
            this.waker = Some((cx.waker().clone(), recording));
        }
        let (_, recording) = this.waker.as_ref().expect("set above");
        let out = this
            .inner
            .as_mut()
            .poll(&mut Context::from_waker(recording));
        let end = Instant::now();
        let mut clock = lock_clock(&this.clock);
        clock.polling_since = None;
        clock.polling += end - start;
        if out.is_ready() {
            clock.finished = Some(end);
            clock.woken = None;
        } else if clock.woken.is_some() {
            // Woken during this poll: the task cannot be polled again before
            // this poll returns, so its wait to be polled starts now.
            clock.woken = Some(end);
        }
        out
    }
}

/// The waker a [`WakeToPoll`] adapter hands its future: a wake records its
/// time, once per wait to be polled, then wakes the task.
struct RecordingWaker {
    task: Waker,
    clock: Arc<Mutex<WakeClock>>,
}

impl Wake for RecordingWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        {
            let mut clock = lock_clock(&self.clock);
            if clock.finished.is_none() {
                clock.woken.get_or_insert_with(Instant::now);
            }
        }
        self.task.wake_by_ref();
    }
}

/// The tick of a [`Writer`]'s runtime probe ([`run_tick_probe`]).
const PROBE_TICK: Duration = Duration::from_millis(10);

/// What a [`run_tick_probe`] task has seen of its runtime: each tick is due a
/// fixed time after the probe was last polled, and its lateness is how long
/// after that due time the probe was polled again. A busy runtime — workers
/// that neither run the probe nor reach the timer driver — makes every task on
/// it late the same way, whichever task the workers are running.
#[derive(Default)]
struct TickLog {
    /// The armed tick's due time, if one is armed.
    pending_due: Option<Instant>,
    /// The completed ticks' lateness, summed.
    late: Duration,
    ticks: u64,
}

impl TickLog {
    fn arm(&mut self, due: Instant) {
        self.pending_due = Some(due);
    }

    /// The armed tick was polled at `polled`.
    fn tick(&mut self, polled: Instant) {
        if let Some(due) = self.pending_due.take() {
            self.late += polled.saturating_duration_since(due);
            self.ticks += 1;
        }
    }

    /// Lateness so far, counting an armed tick that is overdue at `now`, and
    /// the ticks completed.
    fn late_by(&self, now: Instant) -> (Duration, u64) {
        let overdue = self
            .pending_due
            .map_or(Duration::ZERO, |due| now.saturating_duration_since(due));
        (self.late + overdue, self.ticks)
    }
}

/// A [`TickLog`], locked. It is never held across an await, and nothing
/// panics while holding it.
fn lock_ticks(ticks: &Mutex<TickLog>) -> MutexGuard<'_, TickLog> {
    ticks.lock().expect("the tick log's lock is never poisoned")
}

/// A test-only probe of its runtime's occupancy, as the 2026-10-02 trace
/// used: tick every `tick` and record in `log` how late each tick is polled.
/// It runs until its task is aborted.
async fn run_tick_probe(log: Arc<Mutex<TickLog>>, tick: Duration) {
    loop {
        let due = Instant::now() + tick;
        lock_ticks(&log).arm(due);
        tokio::time::sleep_until(due.into()).await;
        lock_ticks(&log).tick(Instant::now());
    }
}

/// A task aborted when this handle is dropped.
struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// The start of one wait on a [`Writer`]: when it began, how many commits had
/// completed by then, the split of the commit then in flight, if any, so far —
/// subtracted, so that commit counts only from the wait's start — and the
/// runtime probe's lateness and ticks so far, subtracted the same way.
#[derive(Clone, Copy)]
struct WaitWindow {
    from: Instant,
    done_before: usize,
    carried: TimeSplit,
    late_before: Duration,
    ticks_before: u64,
}

/// A [`Writer`]'s commit in flight: its name and its adapter's clock.
struct InFlight {
    name: CommitName,
    clock: Arc<Mutex<WakeClock>>,
}

/// A [`Writer`]'s commit bookkeeping, read only by its stall diagnostics
/// ([`CommitLog::diagnose`]): the commit in flight, if any, each completed
/// commit's [`TimeSplit`], in completion order, when the writer sat idle with
/// no write to make, and the writer's runtime probe ([`TickLog`]).
struct CommitLog {
    in_flight: Option<InFlight>,
    done: Vec<TimeSplit>,
    /// The writer's idle stretch, if it began one: from, and until if it
    /// ended. A writer idles once, from its first `None` write until stopped.
    idle: Option<(Instant, Option<Instant>)>,
    probe: Arc<Mutex<TickLog>>,
}

impl CommitLog {
    fn new(probe: Arc<Mutex<TickLog>>) -> Self {
        Self {
            in_flight: None,
            done: Vec::new(),
            idle: None,
            probe,
        }
    }

    fn begin_idle(&mut self) {
        self.idle = Some((Instant::now(), None));
    }

    fn end_idle(&mut self) {
        if let Some((_, until)) = &mut self.idle {
            until.get_or_insert_with(Instant::now);
        }
    }

    fn begin(&mut self, name: CommitName, clock: Arc<Mutex<WakeClock>>) {
        self.in_flight = Some(InFlight { name, clock });
    }

    fn end(&mut self) {
        if let Some(f) = self.in_flight.take() {
            self.done.push(lock_clock(&f.clock).split(Instant::now()));
        }
    }

    /// A wait beginning now.
    fn window(&self) -> WaitWindow {
        let from = Instant::now();
        let (late_before, ticks_before) = lock_ticks(&self.probe).late_by(from);
        WaitWindow {
            from,
            done_before: self.done.len(),
            carried: self
                .in_flight
                .as_ref()
                .map_or_else(TimeSplit::default, |f| lock_clock(&f.clock).split(from)),
            late_before,
            ticks_before,
        }
    }

    /// What the writer did from `window`'s start until now, for a timed-out
    /// wait's message: a verdict, then the figures it rests on — the commits
    /// completed in the wait with their max and mean duration, the commit in
    /// flight with its age, the writer's time in the wait split into time
    /// woken but not polled, inside polls, awaiting a wake ([`TimeSplit`]),
    /// idle with no write to make, and outside any commit (the loop's only
    /// await between commits is the idle one, so the rest is time before its
    /// task's first poll), and the runtime probe's lateness in the wait
    /// ([`TickLog`]). Two measures of a busy runtime: the writer's own time
    /// not polled, and the probe's lateness, which also catches a runtime too
    /// busy to deliver the writer's wakes (that time reads as the writer
    /// awaiting a wake). The verdict:
    ///
    /// * **idle** — the writer had no write to make for at least half the
    ///   wait: it was not committing, whatever the runtime did.
    /// * **not scheduled** — otherwise, the writer was not polled (woken but
    ///   not polled, plus outside any commit) for at least half the wait, or
    ///   the probe was late by at least half the wait: the runtime was slow to
    ///   poll its tasks. It does not say what occupied the runtime.
    /// * **blocked** — otherwise, one commit in flight for at least half the
    ///   wait; the split says whether it was awaiting a wake or running.
    /// * **slow** — otherwise: commits are completing; the split says whether
    ///   their time went to awaiting or to running.
    ///
    /// Under blocked or slow, time awaiting a wake is the store, a lock or a
    /// timer only as far as the probe's lateness rules out a busy runtime.
    fn diagnose(&self, window: WaitWindow) -> String {
        let now = Instant::now();
        let waited = now.saturating_duration_since(window.from);
        let completed = &self.done[window.done_before..];
        let n = completed.len();
        let max = completed.iter().map(|s| s.total).max().unwrap_or_default();
        let mean = if n == 0 {
            Duration::ZERO
        } else {
            completed.iter().map(|s| s.total).sum::<Duration>() / n as u32
        };
        let current = self
            .in_flight
            .as_ref()
            .map(|f| (f.name, lock_clock(&f.clock).split(now)));
        // The wait's commits, the first counted from the wait's start:
        // commits run one at a time, so the one in flight then is the first
        // to complete after it, or still the one in flight.
        let mut in_wait: Vec<TimeSplit> = completed
            .iter()
            .copied()
            .chain(current.map(|(_, s)| s))
            .collect();
        if let Some(first) = in_wait.first_mut() {
            *first = first.minus(window.carried);
        }
        let current_in_wait = current.and(in_wait.last().copied());
        let split = in_wait
            .into_iter()
            .fold(TimeSplit::default(), TimeSplit::plus);
        let idle = self.idle.map_or(Duration::ZERO, |(from, until)| {
            until
                .unwrap_or(now)
                .saturating_duration_since(from.max(window.from))
        });
        let outside = waited.saturating_sub(split.total + idle);
        let not_polled = split.not_polled + outside;
        let (late_now, ticks_now) = lock_ticks(&self.probe).late_by(now);
        let probe_late = late_now.saturating_sub(window.late_before);
        let probe_ticks = ticks_now.saturating_sub(window.ticks_before);
        let verdict = match (current, current_in_wait) {
            _ if idle >= waited / 2 => format!(
                "IDLE: the writer had no write to make for {idle:.1?} of the {waited:.1?} wait"
            ),
            _ if not_polled >= waited / 2 || probe_late >= waited / 2 => format!(
                "NOT SCHEDULED: the runtime was slow to poll its tasks — the writer's task went \
                 unpolled for {not_polled:.1?} and a {PROBE_TICK:?} probe tick on the same \
                 runtime was late by {probe_late:.1?} in all, of the {waited:.1?} wait"
            ),
            (Some((name, whole)), Some(part)) if part.total >= waited / 2 => format!(
                "BLOCKED: {name} has been in flight for {:.1?}, in the wait {:.1?} of it \
                 awaiting a wake and {:.1?} inside its polls",
                whole.total,
                part.awaiting(),
                part.polling
            ),
            _ => format!(
                "SLOW: commits are completing, in the wait {:.1?} of their time awaiting a \
                 wake and {:.1?} inside their polls",
                split.awaiting(),
                split.polling
            ),
        };
        let in_flight = match current {
            Some((name, whole)) => format!("{name} in flight for {:.1?}", whole.total),
            None => "no commit in flight".to_owned(),
        };
        format!(
            "{verdict} [over the {waited:.1?} wait: {n} commits completed, max {max:.1?}, mean \
             {mean:.1?}; {in_flight}; the writer's time: woken but not polled {:.1?}, inside \
             polls {:.1?}, awaiting a wake {:.1?} (the store, a lock, a timer, or a wake a \
             busy runtime was slow to deliver), idle {idle:.1?}, outside any commit \
             {outside:.1?}; runtime probe: late {probe_late:.1?} in all over {probe_ticks} ticks \
             of {PROBE_TICK:?}]",
            split.not_polled,
            split.polling,
            split.awaiting()
        )
    }
}

/// The commit log, locked. It is never held across an await, and nothing
/// panics while holding it.
fn lock(log: &Mutex<CommitLog>) -> MutexGuard<'_, CommitLog> {
    log.lock().expect("the commit log's lock is never poisoned")
}

/// A writer committing `write(k)` for k = 0, 1, … back to back until stopped
/// (or, once `write` answers `None`, idle until stopped), then, if it is the
/// `sentinel` writer, the sentinel commit. Its progress is the number of
/// commits made; each commit runs inside a [`WakeToPoll`] adapter, a runtime
/// probe ([`run_tick_probe`]) runs beside it on the same runtime until the
/// writer is dropped, and its [`CommitLog`] keeps both's timings, so a
/// timed-out wait on it says how the writer stalled.
struct Writer {
    progress: tokio::sync::watch::Receiver<u64>,
    stop: tokio_util::sync::CancellationToken,
    task: tokio::task::JoinHandle<()>,
    log: Arc<Mutex<CommitLog>>,
    _probe: AbortOnDrop,
}

impl Writer {
    fn start(
        svc: Arc<LiveEditService>,
        session_id: String,
        layer: String,
        object_id: u64,
        write: fn(u64) -> Option<OwnedDelta>,
        sentinel: bool,
    ) -> Self {
        let (tx, progress) = tokio::sync::watch::channel(0u64);
        let stop = tokio_util::sync::CancellationToken::new();
        let ticks = Arc::new(Mutex::new(TickLog::default()));
        let probe = AbortOnDrop(tokio::spawn(run_tick_probe(Arc::clone(&ticks), PROBE_TICK)));
        let log = Arc::new(Mutex::new(CommitLog::new(ticks)));
        let task = {
            let stop = stop.clone();
            let log = Arc::clone(&log);
            tokio::spawn(async move {
                let mut k = 0u64;
                while !stop.is_cancelled() {
                    let Some(delta) = write(k) else {
                        lock(&log).begin_idle();
                        stop.cancelled().await;
                        lock(&log).end_idle();
                        break;
                    };
                    let timed =
                        WakeToPoll::new(commit(&svc, &session_id, &layer, object_id, &delta));
                    lock(&log).begin(CommitName::Write(k), timed.clock());
                    timed.await;
                    lock(&log).end();
                    k += 1;
                    tx.send_replace(k);
                }
                if sentinel {
                    let sentinel_section =
                        create_section(0, SENTINEL_SECTION, "sentinel".to_owned());
                    let timed = WakeToPoll::new(commit(
                        &svc,
                        &session_id,
                        &layer,
                        object_id,
                        &sentinel_section,
                    ));
                    lock(&log).begin(CommitName::Sentinel, timed.clock());
                    timed.await;
                    lock(&log).end();
                }
            })
        };
        Self {
            progress,
            stop,
            task,
            log,
            _probe: probe,
        }
    }

    /// Commits made so far.
    fn made(&self) -> u64 {
        *self.progress.borrow()
    }

    /// Wait, bounded by [`SETTLE`], until the writer has made at least
    /// `commits` commits ([`Writer::wait_for_within`]).
    async fn wait_for(&mut self, commits: u64, phase: &str) {
        self.wait_for_within(commits, phase, SETTLE).await;
    }

    /// Wait, bounded by `within`, until the writer has made at least `commits`
    /// commits. `phase` says which wait this is (e.g. before which joiner); a
    /// timeout names it and reports a writer stall — the writer's own commits
    /// did not land in time, which is not a catch-up's proof or cut failing
    /// (module header) — with whether the writer was blocked, slow, not
    /// scheduled or idle ([`CommitLog::diagnose`]).
    async fn wait_for_within(&mut self, commits: u64, phase: &str, within: Duration) {
        let made = self.made();
        let window = lock(&self.log).window();
        let reached = tokio::time::timeout(within, self.progress.wait_for(|&k| k >= commits))
            .await
            .map(|r| r.map(|_| ()));
        match reached {
            Ok(running) => running.expect("the writer is running"),
            Err(_) => {
                let diagnosis = lock(&self.log).diagnose(window);
                panic!(
                    "writer stall {phase}: the writer did not reach {commits} commits within \
                     {within:?} (it had made {made} when the wait began, {} now), which is not \
                     a catch-up's proof failing. {diagnosis}",
                    self.made()
                )
            }
        }
    }

    /// Stop the writer, after its sentinel commit if it makes one. A timeout
    /// reports a writer stall as [`Writer::wait_for`] does.
    async fn finish(self) {
        let window = lock(&self.log).window();
        self.stop.cancel();
        match tokio::time::timeout(SETTLE, self.task).await {
            Ok(joined) => joined.expect("the writer must not panic"),
            Err(_) => {
                let diagnosis = lock(&self.log).diagnose(window);
                panic!(
                    "writer stall at finish: the writer did not stop within {SETTLE:?} (its \
                     last commit, then any sentinel commit, did not land). {diagnosis}"
                )
            }
        }
    }
}

// ── WakeToPoll checks ───────────────────────────────────────────────────────
//
// The adapter driven by hand, with no runtime and no Valkey: a test polls it
// with its own `Context`, so it decides exactly when a poll happens. The
// delays are the quantities measured, not synchronisation; every bound is one
// the clock guarantees (a recorded wait at least the delay, or at most the
// span the test measured around it), never a speed.

/// The delay these checks hold off a poll for, or spend inside one.
const WAKE_CHECK_DELAY: Duration = Duration::from_millis(30);

/// A future that stays pending until [`Gate::open`], then completes with
/// [`Gate::OUTPUT`]; opening wakes the waker of its last poll.
#[derive(Clone, Default)]
struct Gate(Arc<Mutex<(bool, Option<Waker>)>>);

impl Gate {
    const OUTPUT: u32 = 7;

    fn open(&self) {
        let waker = {
            let mut g = self.0.lock().expect("the gate's lock is never poisoned");
            g.0 = true;
            g.1.take()
        };
        waker.expect("the gate was polled before it opened").wake();
    }
}

impl Future for Gate {
    type Output = u32;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u32> {
        let mut g = self.0.lock().expect("the gate's lock is never poisoned");
        if g.0 {
            Poll::Ready(Self::OUTPUT)
        } else {
            g.1 = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// A task waker that counts its wakes.
#[derive(Default)]
struct CountingWaker(std::sync::atomic::AtomicUsize);

impl CountingWaker {
    fn wakes(&self) -> usize {
        self.0.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl Wake for CountingWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Poll `fut` once with `waker`.
fn poll_once<F: Future>(fut: &mut WakeToPoll<F>, waker: &Waker) -> Poll<F::Output> {
    Pin::new(fut).poll(&mut Context::from_waker(waker))
}

fn split_of<F>(fut: &WakeToPoll<F>) -> TimeSplit {
    lock_clock(&fut.clock).split(Instant::now())
}

#[test]
fn wake_to_poll_counts_a_poll_delayed_after_a_wake_as_not_polled() {
    let task = Waker::from(Arc::new(CountingWaker::default()));
    let gate = Gate::default();
    let mut timed = WakeToPoll::new(gate.clone());
    assert!(poll_once(&mut timed, &task).is_pending());

    gate.open();
    std::thread::sleep(WAKE_CHECK_DELAY);
    assert!(poll_once(&mut timed, &task).is_ready());

    let split = split_of(&timed);
    assert!(
        split.not_polled >= WAKE_CHECK_DELAY,
        "a poll held off {WAKE_CHECK_DELAY:?} after its wake must record at least that as not \
         polled: {split:?}"
    );
}

#[test]
fn wake_to_poll_does_not_count_the_wait_before_a_wake_as_not_polled() {
    let task = Waker::from(Arc::new(CountingWaker::default()));
    let gate = Gate::default();
    let mut timed = WakeToPoll::new(gate.clone());
    assert!(poll_once(&mut timed, &task).is_pending());
    // Pending and not woken: awaiting, not waiting to be polled.
    std::thread::sleep(WAKE_CHECK_DELAY);

    let woken = Instant::now();
    gate.open();
    assert!(poll_once(&mut timed, &task).is_ready());
    let span = woken.elapsed();

    let split = split_of(&timed);
    assert!(
        split.not_polled <= span,
        "a poll made at once after its wake records at most the {span:?} from the wake to the \
         poll's end as not polled, none of the {WAKE_CHECK_DELAY:?} before the wake: {split:?}"
    );
}

#[test]
fn wake_to_poll_counts_time_inside_polls_as_polling() {
    let task = Waker::from(Arc::new(CountingWaker::default()));
    let mut timed = WakeToPoll::new(std::future::poll_fn(|_| {
        std::thread::sleep(WAKE_CHECK_DELAY);
        Poll::Ready(())
    }));
    assert!(poll_once(&mut timed, &task).is_ready());

    let split = split_of(&timed);
    assert!(
        split.polling >= WAKE_CHECK_DELAY,
        "a poll that runs {WAKE_CHECK_DELAY:?} must record at least that inside polls: {split:?}"
    );
}

#[test]
fn wake_to_poll_passes_each_wake_through_and_returns_the_inner_output() {
    let counter = Arc::new(CountingWaker::default());
    let task = Waker::from(Arc::clone(&counter));
    let gate = Gate::default();
    let mut timed = WakeToPoll::new(gate.clone());
    assert!(poll_once(&mut timed, &task).is_pending());
    assert_eq!(counter.wakes(), 0, "no wake before the gate opens");

    gate.open();
    assert_eq!(
        counter.wakes(),
        1,
        "the inner future's wake reaches the task's waker"
    );
    assert_eq!(
        poll_once(&mut timed, &task),
        Poll::Ready(Gate::OUTPUT),
        "the inner future's output comes back unchanged"
    );
}

// ── Runtime probe checks ────────────────────────────────────────────────────
//
// The probe on a one-thread runtime the test builds: while the test's own
// future blocks that thread, nothing else on the runtime runs, so the test
// decides when the probe can be polled. Bounds rest on the clock only: a held
// tick is late by at least the time from its due to the end of the hold, and
// on-time ticks leave at most the span less a tick per tick completed.

fn one_thread_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("a one-thread runtime builds")
}

/// Wait (bounded by [`SETTLE`]) until `ready` holds of the probe's log,
/// letting the runtime run between looks.
async fn until_probe(log: &Mutex<TickLog>, ready: impl Fn(&TickLog) -> bool) {
    tokio::time::timeout(SETTLE, async {
        while !ready(&lock_ticks(log)) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("the probe makes progress on a runtime the test does not hold");
}

#[test]
fn tick_probe_counts_a_tick_held_past_its_due_as_late() {
    let log = Arc::new(Mutex::new(TickLog::default()));
    one_thread_runtime().block_on(async {
        let _probe = AbortOnDrop(tokio::spawn(run_tick_probe(Arc::clone(&log), PROBE_TICK)));
        until_probe(&log, |t| t.pending_due.is_some()).await;
        let (due, ticks_before) = {
            let t = lock_ticks(&log);
            (t.pending_due.expect("armed"), t.ticks)
        };

        // Hold the runtime's only thread past the armed tick's due.
        std::thread::sleep(PROBE_TICK + WAKE_CHECK_DELAY);
        let held_until = Instant::now();
        let held_late = held_until.saturating_duration_since(due);

        let (late_while_held, _) = lock_ticks(&log).late_by(held_until);
        assert!(
            late_while_held >= held_late,
            "a tick still overdue counts as late before it is polled: {late_while_held:?} < \
             {held_late:?}"
        );
        until_probe(&log, |t| t.ticks > ticks_before).await;
        let late = lock_ticks(&log).late;
        assert!(
            late >= held_late,
            "a tick held {held_late:?} past its due counts that as late once polled: {late:?}"
        );
    });
}

#[test]
fn tick_probe_does_not_count_on_time_waiting_as_late() {
    let log = Arc::new(Mutex::new(TickLog::default()));
    one_thread_runtime().block_on(async {
        let start = Instant::now();
        let _probe = AbortOnDrop(tokio::spawn(run_tick_probe(Arc::clone(&log), PROBE_TICK)));
        until_probe(&log, |t| t.ticks >= 3).await;

        let now = Instant::now();
        let (late, ticks) = lock_ticks(&log).late_by(now);
        // Each tick is armed no earlier than the last was polled, so the span
        // holds a full tick per tick completed plus every tick's lateness.
        let on_time = PROBE_TICK * u32::try_from(ticks).expect("a few ticks");
        let span = now.saturating_duration_since(start);
        assert!(
            late <= span.saturating_sub(on_time),
            "{ticks} ticks of {PROBE_TICK:?} in {span:?} leave at most {:?} late, not {late:?}",
            span.saturating_sub(on_time)
        );
    });
}

// ── Followers ───────────────────────────────────────────────────────────────

type SubStream = Pin<Box<dyn Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Send>>;

async fn subscribe(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    client_id: &str,
) -> SubStream {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer.to_owned(),
        object_id: 0,
        sequence_num: 0,
        client_id: client_id.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    // Boxed, as `commit`'s handler call is.
    Box::pin(handle_subscribe_read_layer(svc, request))
        .await
        .unwrap_or_else(|s| panic!("the subscribe must be admitted: {s:?}"))
        .into_inner()
}

/// What one brand-new subscriber was sent, applied to its own model.
struct Followed {
    client_id: String,
    /// The sequence its catch-up content carried.
    content_seq: u64,
    /// The sequence its caught-up ack carried.
    ack_seq: u64,
    /// The first live message's sequence.
    first_live: Option<u64>,
    /// Its model after applying everything it was sent.
    model: ObjectState,
}

/// One message's header and reassembled body; `Err` on a status or the
/// stream's end.
async fn next_message(
    stream: &mut SubStream,
) -> Result<(live_edit::grpc::proto::SubscribeReadLayerHeader, Vec<u8>), String> {
    let header = match stream.next().await {
        Some(Ok(SubscribeReadLayerResponse {
            response_type: Some(subscribe_read_layer_response::ResponseType::Header(h)),
        })) => h,
        Some(Ok(other)) => return Err(format!("expected a header, got {other:?}")),
        Some(Err(status)) => return Err(format!("the stream ended {status:?}")),
        None => return Err("the stream ended with no status".to_owned()),
    };
    let mut body = Vec::new();
    while (body.len() as u64) < header.total_size {
        match stream.next().await {
            Some(Ok(SubscribeReadLayerResponse {
                response_type: Some(subscribe_read_layer_response::ResponseType::Chunk(c)),
            })) => body.extend_from_slice(&c.data),
            other => return Err(format!("expected a body chunk, got {other:?}")),
        }
    }
    Ok((header, body))
}

/// Apply one delivered OMPF body to `model` with the domain engine; whether it
/// created the sentinel section.
fn apply(model: &mut ObjectState, body: &[u8]) -> Result<bool, String> {
    let container = OmpfContainer::parse(body).map_err(|e| format!("parse OMPF: {e:?}"))?;
    let delta = if container.delta_bytes.is_empty() {
        None
    } else {
        Some(
            oodf_ffi::parse_delta(container.delta_bytes)
                .map_err(|e| format!("parse delta: {e:?}"))?,
        )
    };
    let creates_sentinel = delta.as_ref().is_some_and(|d| {
        d.struct_commands.iter().any(
            |c| matches!(c, StructCommand::CreateSection(s) if s.section_id == SENTINEL_SECTION),
        )
    });
    apply_delta(model, delta).map_err(|e| format!("{e:?}"))?;
    Ok(creates_sentinel)
}

/// A follower's one report on its `caught_up` channel: sent at its caught-up
/// ack, or — if it never gets that far — when it stops, whether it failed,
/// ran out of [`FOLLOW_BUDGET`] or panicked, so the test's
/// [`await_caught_up`] never waits on a follower that has already stopped.
struct CaughtUpReport(Option<tokio::sync::mpsc::Sender<()>>);

impl CaughtUpReport {
    /// Report once; later calls and the drop are no-ops. `try_send` never
    /// finds the channel full: each follower reports at most once and the
    /// channel holds one report per follower.
    fn send(&mut self) {
        if let Some(tx) = self.0.take() {
            let _ = tx.try_send(());
        }
    }
}

impl Drop for CaughtUpReport {
    fn drop(&mut self) {
        self.send();
    }
}

/// Follow one brand-new subscription to the writer's sentinel commit: its
/// catch-up content, its caught-up ack, then every live commit in order — each
/// exactly the one after the last — applied to a local model. It reports on
/// `caught_up` at the ack, or when it stops without one ([`CaughtUpReport`]);
/// its error, if any, is its return value.
async fn follow(
    mut stream: SubStream,
    client_id: String,
    caught_up: tokio::sync::mpsc::Sender<()>,
) -> Result<Followed, String> {
    let mut caught_up = CaughtUpReport(Some(caught_up));
    let run = async {
        let mut model = ObjectState::default();
        let (content, body) = next_message(&mut stream).await?;
        if content.total_size == 0 {
            return Err(format!("the first message was not content: {content:?}"));
        }
        apply(&mut model, &body).map_err(|e| {
            format!(
                "the catch-up content (seq {}) did not apply: {e}",
                content.sequence_num
            )
        })?;
        let (ack, _) = next_message(&mut stream).await?;
        if ack.total_size != 0 {
            return Err(format!(
                "the catch-up sent more than one content message: {ack:?}"
            ));
        }
        caught_up.send();
        let mut last = ack.sequence_num;
        let mut first_live = None;
        loop {
            let (live, body) = next_message(&mut stream).await?;
            if live.sequence_num != last + 1 {
                return Err(format!(
                    "live commit {} arrived after {last}: every commit after the ack must arrive \
                     exactly once, in order",
                    live.sequence_num
                ));
            }
            first_live.get_or_insert(live.sequence_num);
            last = live.sequence_num;
            let sentinel = apply(&mut model, &body).map_err(|e| {
                format!(
                    "live commit {} did not apply: {e} — a commit the catch-up already folded in \
                     was delivered again (double-apply)",
                    live.sequence_num
                )
            })?;
            if sentinel {
                return Ok(Followed {
                    client_id: client_id.clone(),
                    content_seq: content.sequence_num,
                    ack_seq: ack.sequence_num,
                    first_live,
                    model,
                });
            }
        }
    };
    tokio::time::timeout(FOLLOW_BUDGET, run)
        .await
        .unwrap_or_else(|_| {
            Err(format!(
                "did not reach the sentinel within {FOLLOW_BUDGET:?}"
            ))
        })
        .map_err(|e| format!("[{client_id}] {e}"))
}

/// Wait (bounded by [`SETTLE`]) for one follower's report. A follower that
/// stops before its ack reports as it stops ([`CaughtUpReport`]), so this
/// returns and the test goes on to [`followed`], which reports that
/// follower's own error; the timeout fires only on a follower still waiting
/// for its catch-up.
async fn await_caught_up(caught_up: &mut tokio::sync::mpsc::Receiver<()>) {
    let _ = tokio::time::timeout(SETTLE, caught_up.recv())
        .await
        .unwrap_or_else(|_| panic!("a joiner did not catch up within {SETTLE:?}"));
}

/// Await every follower's report.
async fn followed(
    followers: Vec<tokio::task::JoinHandle<Result<Followed, String>>>,
) -> Vec<Result<Followed, String>> {
    let mut out = Vec::with_capacity(followers.len());
    for f in followers {
        out.push(f.await.expect("a follower must not panic"));
    }
    out
}

// ── Assertions ──────────────────────────────────────────────────────────────

/// The server's state of `object_id` now.
async fn server_state(fx: &Fixture, object_id: u64) -> ObjectState {
    snapshot_to_object_state(
        &read_object_state_for_test(&fx.client, &fx.key_schema, &object_id.to_string())
            .await
            .expect("the final state must load"),
    )
}

/// Assert `model` has the server's sections — ids, parents and names — and
/// each parent's children in the server's order.
///
/// A model places its catch-up content by the order each row carries (sibling
/// indexes in snapshot mode, stored keys in rollback mode) and then applies
/// every live commit, where a created section and a section moved under a new
/// parent append. The server stores them the same way (b.xje), so the model
/// must end with the server's list, in order, under every parent: a commit
/// applied twice or missed changes membership or a parent, and a create or
/// move placed anywhere but last changes the order. `section_order` itself is
/// not compared — a snapshot-mode model holds sibling indexes where the server
/// holds its own keys, and only the order the keys produce is the contract.
fn assert_model_matches(who: &str, model: &ObjectState, server: &ObjectState) {
    let shape = |s: &ObjectState| -> HashMap<u64, (u64, String)> {
        s.sections
            .iter()
            .map(|(id, sec)| (*id, (sec.parent_id, sec.name.clone())))
            .collect()
    };
    let (mine, theirs) = (shape(model), shape(server));
    if mine != theirs {
        let only_mine: BTreeSet<u64> = mine
            .keys()
            .filter(|k| !theirs.contains_key(k))
            .copied()
            .collect();
        let only_theirs: BTreeSet<u64> = theirs
            .keys()
            .filter(|k| !mine.contains_key(k))
            .copied()
            .collect();
        let differing: BTreeSet<u64> = mine
            .iter()
            .filter(|(k, v)| theirs.get(k).is_some_and(|t| t != *v))
            .map(|(k, _)| *k)
            .collect();
        panic!(
            "[{who}] the model's sections differ from the server's: only in the model \
             {only_mine:?}, only on the server {only_theirs:?}, different parent or name \
             {differing:?}"
        );
    }
    let children = |s: &ObjectState| -> HashMap<u64, Vec<u64>> {
        s.children
            .iter()
            .filter(|(_, c)| !c.is_empty())
            .map(|(p, c)| (*p, c.clone()))
            .collect()
    };
    let (mine, theirs) = (children(model), children(server));
    let parents: BTreeSet<u64> = mine.keys().chain(theirs.keys()).copied().collect();
    for parent in parents {
        let empty = Vec::new();
        let m = mine.get(&parent).unwrap_or(&empty);
        let t = theirs.get(&parent).unwrap_or(&empty);
        if m != t {
            let first_difference = m
                .iter()
                .zip(t)
                .position(|(a, b)| a != b)
                .unwrap_or(m.len().min(t.len()));
            panic!(
                "[{who}] parent {parent}'s children differ from the server's, first at index \
                 {first_difference}: the model holds {} children, the server {}; from there the \
                 model has {:?} and the server {:?}",
                m.len(),
                t.len(),
                &m[first_difference..m.len().min(first_difference + 8)],
                &t[first_difference..t.len().min(first_difference + 8)],
            );
        }
    }
}

/// Assert every follower applied everything it was sent and acked its
/// content's cut, and that its model is the server's final state.
fn assert_every_follower_converged(
    case: &str,
    followed: &[Result<Followed, String>],
    server: &ObjectState,
) {
    let failures: Vec<&String> = followed.iter().filter_map(|f| f.as_ref().err()).collect();
    assert!(
        failures.is_empty(),
        "[{case}] every brand-new subscriber must apply every commit it is sent, exactly once: \
         {failures:#?}"
    );
    for f in followed.iter().flatten() {
        assert_eq!(
            f.content_seq, f.ack_seq,
            "[{case} / {}] the ack carries the cut the content serves",
            f.client_id
        );
        assert_model_matches(&f.client_id, &f.model, server);
    }
}

/// `live_subscribe_stable_cut_total{outcome, regime="brand_new"}`.
fn brand_new_cuts(metrics: &Metrics, outcome: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_subscribe_stable_cut_total",
        &[
            ("outcome", outcome),
            ("regime", "brand_new"),
            ("service", SERVICE),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// `live_edit_snapshot_read_total{consumer="catchup", path, outcome}`.
fn catchup_reads(metrics: &Metrics, path: &str, outcome: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_snapshot_read_total",
        &[
            ("consumer", "catchup"),
            ("path", path),
            ("outcome", outcome),
        ],
    )
    .unwrap_or(0.0) as u64
}

// ═══════════════════════════════════════════════════════════════════════════
// Structural double-apply regression
// ═══════════════════════════════════════════════════════════════════════════

/// The section count of the double-apply layer.
const DOUBLE_APPLY_SECTIONS: u64 = 2_000;

/// Brand-new subscribers of a 2,000-section layer join one after another,
/// with a pair of structural writes — `CreateSection` of a new section and
/// `MoveSection` of an older section under it — before each join and after
/// the last. Each joiner is held by the state-read seam after
/// its header read P and before its read; one more such pair commits — a new
/// section `HELD_SECTION_BASE + i` and a seeded section from the top moved
/// under it — and a field writer then commits back to back until the joiner
/// has caught up, so the header moves through its read. Every follower applies
/// its catch-up content and every live commit after its ack — each exactly the
/// one after the last — with the domain `apply_delta`, and its model ends
/// equal to the server's final state: the same sections, parents and names,
/// and the same children, in the same order, under each parent
/// ([`assert_model_matches`]).
///
/// Run at the shipped ceiling (each joiner's own atomic read) and above it
/// ([`Ceiling::ForcedLow`]: each joiner's flight's fenced read, which holds the
/// writers off for its epoch rather than racing them), in both modes.
///
/// Fails at e1220ea0 in both modes (module header): each joiner, registered
/// and parked after its re-read P′, is sent the held pair again, and its
/// re-applied `CreateSection` fails `DuplicateSection`. On this code, a
/// catch-up that acks and registers at P while serving its read's state fails
/// it the same way, at either ceiling.
#[cfg(feature = "test-fault-injection")]
async fn structural_writes_are_applied_exactly_once_by_every_brand_new_subscriber(
    catchup_snapshot_enabled: bool,
    ceiling: Ceiling,
) {
    let case = format!("snapshot={catchup_snapshot_enabled} / {ceiling:?}");
    let fx = Fixture::setup(ceiling).await;
    let mut svc = fx.service(catchup_snapshot_enabled);
    let seam = Arc::new(live_edit::grpc::subscribe::CatchUpStateReadSeam::new_before_full_state());
    svc.catch_up_state_read_seam_for_test = Some(Arc::clone(&seam));
    let svc = Arc::new(svc);
    let layer = format!("double-apply-{catchup_snapshot_enabled}-{ceiling:?}");
    let (session_id, object_id) =
        create_large_layer(&fx, &svc, &layer, DOUBLE_APPLY_SECTIONS, 0).await;

    // Structural writes between the joins, the held pairs and the field
    // bursts are all commits of this one task or its burst writer, in order.
    let mut structural = 0u64;
    let (caught_up_tx, mut caught_up) = tokio::sync::mpsc::channel(JOINERS);
    let mut followers = Vec::with_capacity(JOINERS);
    for i in 0..JOINERS as u64 {
        for _ in 0..COMMITS_PER_JOIN {
            commit(
                &svc,
                &session_id,
                &layer,
                object_id,
                &structural_write(structural),
            )
            .await;
            structural += 1;
        }
        let client_id = format!("double-apply-joiner-{i}");
        let stream = subscribe(&svc, &session_id, &layer, &client_id).await;
        followers.push(tokio::spawn(follow(
            stream,
            client_id,
            caught_up_tx.clone(),
        )));
        tokio::time::timeout(SETTLE, seam.wait_until_parked_pre_full_state())
            .await
            .expect("the joiner parks after its header read, before its read");

        // After P, before the read: a pair the read must fold in and the
        // stream must never deliver again.
        let held = HELD_SECTION_BASE + i;
        commit(
            &svc,
            &session_id,
            &layer,
            object_id,
            &create_section(0, held, format!("held{i}")),
        )
        .await;
        commit(
            &svc,
            &session_id,
            &layer,
            object_id,
            &move_seeded_section(DOUBLE_APPLY_SECTIONS - i, held),
        )
        .await;
        // The header moves through the read: field commits back to back until
        // the joiner has caught up.
        let fields = Writer::start(
            Arc::clone(&svc),
            session_id.clone(),
            layer.clone(),
            object_id,
            field_write,
            false,
        );
        tokio::time::timeout(SETTLE, seam.arrive_at_pre_full_state())
            .await
            .expect("the parked joiner accepts its release");
        await_caught_up(&mut caught_up).await;
        fields.finish().await;
    }
    drop(caught_up_tx);
    for _ in 0..COMMITS_PER_JOIN {
        commit(
            &svc,
            &session_id,
            &layer,
            object_id,
            &structural_write(structural),
        )
        .await;
        structural += 1;
    }
    commit(
        &svc,
        &session_id,
        &layer,
        object_id,
        &create_section(0, SENTINEL_SECTION, "sentinel".to_owned()),
    )
    .await;
    let followed = followed(followers).await;

    let server = server_state(&fx, object_id).await;
    assert!(
        server.sections.contains_key(&SENTINEL_SECTION),
        "fixture self-check: the sentinel is the server's last commit"
    );
    assert_every_follower_converged(&case, &followed, &server);
    assert_eq!(
        (
            brand_new_cuts(&fx.metrics, "verified"),
            brand_new_cuts(&fx.metrics, "unverified"),
        ),
        (JOINERS as u64, 0),
        "[{case}] every catch-up served a proven cut"
    );
    let (served_path, atomic_too_large) = match ceiling {
        Ceiling::Shipped => ("atomic", 0),
        Ceiling::ForcedLow => ("fenced", JOINERS as u64),
    };
    assert_eq!(
        (
            catchup_reads(&fx.metrics, served_path, "ok"),
            catchup_reads(&fx.metrics, "atomic", "too_large"),
        ),
        (JOINERS as u64, atomic_too_large),
        "[{case}] each joiner was served from one {served_path} read"
    );
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn structural_writes_are_applied_exactly_once_by_every_brand_new_subscriber_snapshot_mode() {
    structural_writes_are_applied_exactly_once_by_every_brand_new_subscriber(
        true,
        Ceiling::Shipped,
    )
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn structural_writes_are_applied_exactly_once_by_every_brand_new_subscriber_rollback_mode() {
    structural_writes_are_applied_exactly_once_by_every_brand_new_subscriber(
        false,
        Ceiling::Shipped,
    )
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn structural_writes_are_applied_exactly_once_by_every_brand_new_subscriber_of_a_fenced_layer_snapshot_mode(
) {
    structural_writes_are_applied_exactly_once_by_every_brand_new_subscriber(
        true,
        Ceiling::ForcedLow,
    )
    .await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn structural_writes_are_applied_exactly_once_by_every_brand_new_subscriber_of_a_fenced_layer_rollback_mode(
) {
    structural_writes_are_applied_exactly_once_by_every_brand_new_subscriber(
        false,
        Ceiling::ForcedLow,
    )
    .await;
}

// ═══════════════════════════════════════════════════════════════════════════
// A large layer under writers: every brand-new catch-up ends verified
// ═══════════════════════════════════════════════════════════════════════════

/// The join-burst layer's section count.
const LARGE_SECTIONS: u64 = 2_150;

/// Field writes between two joins on the large layer.
const FIELD_COMMITS_PER_JOIN: u64 = 20;

/// A follower's allowance per live commit in its backlog, for the large-layer
/// test's wait after every joiner has reported: about 3× the 20.4 ms one live
/// apply took in a debug build in the 2026-10-02 trace (module header).
const APPLY_ALLOWANCE: Duration = Duration::from_millis(60);

/// The bound on the large-layer test's wait after every joiner has reported,
/// for followers with at most `backlog` live commits still to apply: the
/// writer's commits slow while the followers apply their backlogs on the
/// runtime it shares (module header), so the wait is [`SETTLE`] plus an
/// [`APPLY_ALLOWANCE`] per backlog commit. It never exceeds
/// [`FOLLOW_BUDGET`]: no follower can reach the sentinel during this wait
/// (the writer commits it only at [`Writer::finish`], afterwards), and each
/// follower's own budget started at its subscribe, before this wait, so a
/// longer wait could only end with every follower already failed.
fn backlog_wait(backlog: u64) -> Duration {
    let allowance = APPLY_ALLOWANCE.saturating_mul(u32::try_from(backlog).unwrap_or(u32::MAX));
    SETTLE.saturating_add(allowance).min(FOLLOW_BUDGET)
}

/// Brand-new catch-ups of a 2,150-section layer with one inline field per
/// section — below the shipped atomic-read ceiling — joining under a field
/// writer that commits back to back (and keeps committing until every joiner
/// has caught up) all record `verified`, each served from one atomic read
/// (`live_edit_snapshot_read_total{path="atomic", outcome="ok"}`, no
/// `too_large`) with no retry. Each acks the cut its content carries, the
/// first commit sent to it live is the one right after that cut, and its model
/// ends equal to the server's.
///
/// Fails at e1220ea0 and under the forced-low ceiling (module header): a state
/// read of this layer outlasts the gap between the writer's commits, so the
/// quiescence proof loses and the catch-ups record `unverified` (b.xtb stage 7
/// M2 measured 0 % verified on this layer at 275 commits/s).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn brand_new_catch_ups_of_a_large_layer_under_writers_all_end_verified_at_their_atomic_cut() {
    let fx = Fixture::setup(Ceiling::Shipped).await;
    let svc = Arc::new(fx.service(true));
    let layer = "large-under-writers";
    let (session_id, object_id) = create_large_layer(&fx, &svc, layer, LARGE_SECTIONS, 1).await;

    let mut writer = Writer::start(
        Arc::clone(&svc),
        session_id.clone(),
        layer.to_owned(),
        object_id,
        field_write,
        true,
    );
    let (caught_up_tx, mut caught_up) = tokio::sync::mpsc::channel(JOINERS);
    let mut followers = Vec::with_capacity(JOINERS);
    // The writer's commits when the first joiner subscribes: no follower's
    // cut is below it, so no follower's live backlog holds a commit before it.
    let mut made_at_first_join = None;
    for i in 0..JOINERS {
        writer
            .wait_for(
                writer.made() + FIELD_COMMITS_PER_JOIN,
                &format!("before joiner {i} subscribes"),
            )
            .await;
        made_at_first_join.get_or_insert(writer.made());
        let client_id = format!("large-joiner-{i}");
        let stream = subscribe(&svc, &session_id, layer, &client_id).await;
        followers.push(tokio::spawn(follow(
            stream,
            client_id,
            caught_up_tx.clone(),
        )));
    }
    drop(caught_up_tx);
    // Every catch-up reads under writes: the writer stops only once each has
    // caught up.
    for _ in 0..JOINERS {
        await_caught_up(&mut caught_up).await;
    }
    // Each follower now applies its live backlog — at most every commit since
    // the first join, and the ones this wait asks for — on the runtime the
    // writer shares, so this wait is bounded by that backlog, not SETTLE.
    let made = writer.made();
    let backlog = made - made_at_first_join.expect("a joiner subscribed") + FIELD_COMMITS_PER_JOIN;
    writer
        .wait_for_within(
            made + FIELD_COMMITS_PER_JOIN,
            "after every joiner reported",
            backlog_wait(backlog),
        )
        .await;
    writer.finish().await;
    let followed = followed(followers).await;

    assert_eq!(
        (
            brand_new_cuts(&fx.metrics, "verified"),
            brand_new_cuts(&fx.metrics, "unverified"),
        ),
        (JOINERS as u64, 0),
        "every brand-new catch-up of the large layer ends verified"
    );
    assert_eq!(
        (
            catchup_reads(&fx.metrics, "atomic", "ok"),
            catchup_reads(&fx.metrics, "atomic", "too_large"),
        ),
        (JOINERS as u64, 0),
        "each catch-up served its one atomic read: the layer is below the shipped ceiling"
    );
    let server = server_state(&fx, object_id).await;
    assert_every_follower_converged("large", &followed, &server);
    for f in followed.iter().flatten() {
        assert_eq!(
            f.first_live,
            Some(f.ack_seq + 1),
            "[{}] the stream starts right after the acked cut",
            f.client_id
        );
    }
}
