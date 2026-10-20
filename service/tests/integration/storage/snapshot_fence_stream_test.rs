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

//! The snapshot fence under streams of fenced reads (b.y3m Epic 3 Task 10,
//! Subtask t3.y3m.3a.td.f2; Spec SDD Test Fixtures: "structural writers under
//! a stream", "readers under a stream from several replicas", the crashed
//! reader's gap; SR-5.2–SR-5.5, SR-7.5): the scale scenarios of the Task 10
//! report, as correctness assertions with no wall-clock threshold.
//!
//! Every layer here is far above the fixture clients' forced-low atomic-read
//! ceiling ([`LOW_ITEMS`] / [`LOW_BYTES`]; a [`LAYER_SECTIONS`]-section layer
//! is well over 4× it), so every brand-new catch-up reads through the fence.
//! The coefficients are the recorded set at that ceiling (the only knob the
//! cost model takes, `FenceCoefficients::recorded`): `poll` ≈ 1.0 s, a lease
//! ≈ 2.0 s, `reader_wait_max(L)` ≈ 27 s.
//!
//! | Test | Pins |
//! |---|---|
//! | structural writers under a stream | closed-loop structural writers keep committing while closed-loop joiners keep reading through the fence: every commit lands (no retry-budget exhaustion, no fence-wait timeout), the fence-caused conflicts are fence waits, and the readers keep starting epochs while the writers run; on the WATCH path (`lua-watch-structural`) intents are registered, and on the cached route (`lua`) the cache loads the layer through its fenced read and registers none |
//! | two replicas, a join stream | two `LiveEditService` instances on one store alternate brand-new joins of one layer at a stated rate under continuous writers: every join is served verified, none reaches `reader_wait_max(L)`, each join is its own epoch, and every epoch's cut is past the one before it — a commit landed in every writers' gap |
//! | missed join window | one replica's read stage waits for catch-up memory (the whole budget held, as a build chain holds it) while the other replica's epoch starts and ends: it misses that join window, then starts its own epoch at a later cut, served, never capped; two epochs counted |
//! | crashed reader | a lease nobody renews or releases lapses; the next reader is still held off by the writers' gap after it, a commit lands in that gap, and the next epoch's cut includes it |
//!
//! # Counts, not clocks
//!
//! Nothing here asserts how long anything took. What would show starvation
//! is a count: commits that never land, epochs that stop starting, a cut
//! that does not advance between two epochs, a stream ending at
//! `reader_wait_max(L)`. The two-replica stream paces its joins
//! ([`JOIN_PERIOD`], 2.5 s) only so that each join is its own epoch, with a
//! gap between epochs longer than a fenced writer's step (`poll`); the
//! writers must then use every gap, and once each join is served the fence
//! must hold no reader lease, which an epoch that did not end at its release
//! (writers fenced out until the lease lapses, 2.0 s, plus the crash gap after
//! it) would break.
//!
//! # Wait budget
//!
//! Every test body runs under [`TEST_BUDGET`] (100 s), inside nextest's 120 s
//! hang-killer, so a stuck test fails with its own message. Inside it: one
//! awaited message or polled condition [`SETTLE`] (10 s), one catch-up drain
//! [`DRAIN`] (20 s), a lease lapsing [`LAPSE_BUDGET`] (15 s). Paid on purpose:
//! the two-replica stream's [`STREAM_JOINS`] × [`JOIN_PERIOD`] (20 s); the
//! crashed reader's lease (~2.0 s) and the gap after it (~2.0 s); the
//! structural writers' fence sleeps (~1.0 s each). The longest test, the
//! two-replica stream, takes about 25 s.

use std::collections::BTreeSet;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{HashesInterface, KeysInterface};
use fred::types::scripts::Script;
use futures::{Stream, StreamExt};
use tonic::{Request, Status};

use live_edit::config::UpdateLayerBackend;
use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{
    subscribe_read_layer_response, SubscribeReadLayerHeader, SubscribeReadLayerRequest,
    SubscribeReadLayerResponse,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::render;
use live_edit::observability::metrics::{
    Metrics, STREAM_EXIT_CATCHUP_CAPACITY, STREAM_EXIT_FENCED_READ_WAIT_CAP,
};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::snapshot_fence::{
    self as fence, AcquireVerdict, GrantKind, LayerSize, ReleaseVerdict, RetryReason,
};
use live_edit::storage::snapshot_flights::SnapshotFlights;
use live_edit::storage::test_support::{connect_test_client, TestValkey};
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, SetField, StructCommand};

use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::grpc::helpers::{
    build_disabled_recovery_hook, build_service, make_create_data, make_create_header,
    make_request, make_update_data, make_update_header, streaming_create, streaming_update,
    AllowAllPermissions,
};

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

/// The fixture clients' atomic-read item ceiling: far below every layer here.
const LOW_ITEMS: u64 = 16;

/// The fixture clients' atomic-read byte ceiling.
const LOW_BYTES: u64 = 256;

/// Sections each layer is created with (about eight items each): some 20× the
/// item ceiling, so the "≥ 4× the ceiling" of the scenarios holds.
const LAYER_SECTIONS: u64 = 40;

/// The first section id a structural writer adds (above every created one).
const FIRST_NEW_SECTION: u64 = 1_000;

/// Catch-up slots per replica.
const SLOTS: usize = 8;

/// Each subscription's capacity budget: no wait here runs out of it, so a
/// refusal would be a defect, not the budget.
const ROOMY_BUDGET_MS: u64 = 60_000;

/// Bound on one awaited message or polled condition.
const SETTLE: Duration = Duration::from_secs(10);

/// One budget for a whole catch-up drain.
const DRAIN: Duration = Duration::from_secs(20);

/// Bound on a lease nobody renews lapsing: one lease (~2.0 s) plus slack.
const LAPSE_BUDGET: Duration = Duration::from_secs(15);

/// The one deadline over a whole test; see the module's wait budget.
const TEST_BUDGET: Duration = Duration::from_secs(100);

/// Structural writers in the structural-stream test, each a closed loop.
const STRUCTURAL_WRITERS: u64 = 3;

/// Closed-loop joiners in the structural-stream test.
const STREAM_JOINERS: usize = 2;

/// The structural-stream test runs until at least this many joins were served
/// while the writers ran ...
const JOINS_UNDER_WRITERS: u64 = 4;

/// ... and at least this many commits landed.
const COMMITS_UNDER_READERS: u64 = 12;

/// Joins in the two-replica stream, alternating between the replicas.
const STREAM_JOINS: usize = 8;

/// The spacing of the two-replica stream's joins: one join per replica every
/// 2 × this (0.2 joins/s per replica, far below one replica's same-layer
/// serving rate for a layer this size). Longer than a fenced writer's step
/// (`poll` ≈ 1.0 s at [`LOW_ITEMS`]) by a margin, so each join is its own epoch
/// and every gap between two epochs outlasts a writer's fence sleep.
const JOIN_PERIOD: Duration = Duration::from_millis(2_500);

/// Fast-path writers in the two-replica stream, each a closed loop.
const FIELD_WRITERS: u64 = 2;

/// The fence scripts' clock: `TIME` in whole ms, as they read it.
static CLOCK_SCRIPT: std::sync::LazyLock<Script> = std::sync::LazyLock::new(|| {
    Script::from_lua(
        "local t = redis.call('TIME') \
         return tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)",
    )
});

/// Run a test's body under [`TEST_BUDGET`], boxed so the body's future lives
/// on the heap rather than in the test thread's stack frame.
fn within_test_budget<'a>(
    body: impl std::future::Future<Output = ()> + Send + 'a,
) -> Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>> {
    let body = Box::pin(body);
    Box::pin(async move {
        tokio::time::timeout(TEST_BUDGET, body)
            .await
            .unwrap_or_else(|_| panic!("the test did not finish within {TEST_BUDGET:?}"));
    })
}

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

type SubStream = Pin<Box<dyn Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Send>>;

/// One replica: its own client (catch-up slots, fenced read gate, memory
/// budget, stream multiplexer) and metrics, and a service over them, on the
/// shared store and key prefix.
struct Replica {
    name: &'static str,
    client: ValkeyClient,
    metrics: Arc<Metrics>,
    svc: LiveEditService,
}

impl Replica {
    async fn new(tv: &TestValkey, name: &'static str, backend: UpdateLayerBackend) -> Replica {
        let mut config = crate::common::test_config();
        config.snapshot_atomic_read_max_items = LOW_ITEMS;
        config.snapshot_atomic_read_max_bytes = LOW_BYTES;
        config.subscriber_max_concurrent_catchups = SLOTS;
        config.subscriber_admission_wait_ms = ROOMY_BUDGET_MS;
        config.subscriber_admission_max_waiters = None;
        config.valkey_connect_timeout_secs = 30;
        let client = connect_test_client(&config)
            .await
            .unwrap_or_else(|e| panic!("replica {name}: the client must connect: {e:?}"));
        let metrics = Arc::new(crate::common::metrics_scrape_only(name));
        client
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let keys = tv.key_schema().clone();
        // The writer counters (fence waits, fence-wait timeouts, intents) are
        // the layer store's own: without its metrics they would read zero.
        let layer_store = Arc::new(
            LayerStore::new(client.clone(), keys.clone(), 4096)
                .with_metrics(Arc::clone(&metrics))
                .with_update_layer_backend(backend),
        );
        let recovery = build_disabled_recovery_hook(Arc::clone(&layer_store), Arc::clone(&metrics));
        let svc = LiveEditService {
            catchup_snapshot_enabled: true,
            session_store: Arc::new(SessionStore::new(client.clone(), keys.clone())),
            layer_store,
            snapshot_flights: SnapshotFlights::new(client.clone(), keys),
            valkey: client.clone(),
            metrics: Arc::clone(&metrics),
            recovery,
            ..build_service(tv, Arc::new(AllowAllPermissions))
        };
        Replica {
            name,
            client,
            metrics,
            svc,
        }
    }

    /// The sum of every sample of `name` whose labels include `labels`.
    fn metric(&self, name: &str, labels: &[(&str, &str)]) -> f64 {
        let text = render(&self.metrics);
        text.lines()
            .filter(|l| !l.starts_with('#'))
            .filter_map(|l| {
                let (series, value) = l.trim().rsplit_once(char::is_whitespace)?;
                let (series_name, body) = series.split_once('{').unwrap_or((series, ""));
                (series_name == name
                    && labels
                        .iter()
                        .all(|(k, v)| body.contains(&format!("{k}=\"{v}\""))))
                .then(|| value.parse::<f64>().ok())?
            })
            .sum()
    }

    /// `live_edit_snapshot_fence_acquire_total{result}`.
    fn acquires(&self, result: &str) -> u64 {
        self.metric(
            "live_edit_snapshot_fence_acquire_total",
            &[("result", result)],
        ) as u64
    }

    /// `live_edit_snapshot_read_total{consumer="catchup", path="fenced", outcome}`.
    fn fenced_reads(&self, outcome: &str) -> u64 {
        self.metric(
            "live_edit_snapshot_read_total",
            &[
                ("consumer", "catchup"),
                ("path", "fenced"),
                ("outcome", outcome),
            ],
        ) as u64
    }

    /// `live_edit_subscriber_stream_exits_total{reason}`.
    fn exits(&self, reason: &str) -> u64 {
        self.metric(
            "live_edit_subscriber_stream_exits_total",
            &[("reason", reason)],
        ) as u64
    }

    /// `update_layer` calls that met a fence, committed or not
    /// (`live_edit_update_layer_fence_wait_seconds_count`).
    fn fence_waits(&self) -> u64 {
        self.metric("live_edit_update_layer_fence_wait_seconds_count", &[]) as u64
    }

    /// Commits whose fence wait hit its cap (`live_edit_update_layer_fence_timeout_total`).
    fn fence_timeouts(&self) -> u64 {
        self.metric("live_edit_update_layer_fence_timeout_total", &[]) as u64
    }

    /// `live_edit_structural_cold_start_reads_total{route}`.
    fn cold_starts(&self, route: &str) -> u64 {
        self.metric(
            "live_edit_structural_cold_start_reads_total",
            &[("route", route)],
        ) as u64
    }

    /// `live_edit_update_layer_fence_intent_total{result}`.
    fn intents(&self, result: &str) -> u64 {
        self.metric(
            "live_edit_update_layer_fence_intent_total",
            &[("result", result)],
        ) as u64
    }

    /// No stream of this replica ended at `reader_wait_max(L)` or on a
    /// capacity refusal, and no fenced read was broken.
    fn assert_never_capped(&self) {
        assert_eq!(
            (
                self.exits(STREAM_EXIT_FENCED_READ_WAIT_CAP),
                self.exits(STREAM_EXIT_CATCHUP_CAPACITY),
                self.fenced_reads("fence_broken"),
            ),
            (0, 0, 0),
            "replica {}: no reader reached reader_wait_max(L), none was refused for capacity, \
             no fenced read was broken",
            self.name
        );
    }
}

/// One above-ceiling layer on a fresh key prefix, served by `replicas`.
struct Fx {
    tv: TestValkey,
    replicas: Vec<Replica>,
    layer: String,
    session_id: String,
    oid: u64,
}

impl Fx {
    async fn new(layer: &str, names: &[&'static str]) -> Fx {
        Fx::with_backend(layer, names, UpdateLayerBackend::Lua).await
    }

    /// [`Fx::new`] with every replica's layer store on `backend`.
    async fn with_backend(layer: &str, names: &[&'static str], backend: UpdateLayerBackend) -> Fx {
        live_edit::storage::test_support::raise_fd_limit_for_tests();
        let tv = TestValkey::setup().await;
        let mut replicas = Vec::new();
        for name in names {
            replicas.push(Replica::new(&tv, name, backend).await);
        }
        let svc = &replicas[0].svc;
        let session_id = SessionStore::new(replicas[0].client.clone(), tv.key_schema().clone())
            .create(&format!("{layer}-session"), "/stage", "test-user")
            .await
            .expect("session creation must succeed");
        let sections: Vec<u64> = (1..=LAYER_SECTIONS).collect();
        // Boxed, as `commit`'s handler call is.
        let oid = Box::pin(handle_create_layer(
            svc,
            make_request(streaming_create(vec![
                make_create_header(&session_id, layer),
                make_create_data(delta_to_ompf_bytes(&sections_delta(&sections), &[])),
            ])),
        ))
        .await
        .expect("CreateLayer must succeed")
        .into_inner()
        .object_id;
        Fx {
            tv,
            replicas,
            layer: layer.to_owned(),
            session_id,
            oid,
        }
    }

    fn keys(&self) -> &KeySchema {
        self.tv.key_schema()
    }

    /// A brand-new subscriber on replica `r`.
    async fn join(&self, r: usize, client_id: &str) -> SubStream {
        let mut request = Request::new(SubscribeReadLayerRequest {
            session_id: self.session_id.clone(),
            layer_name: self.layer.clone(),
            object_id: 0,
            sequence_num: 0,
            client_id: client_id.to_owned(),
        });
        request
            .extensions_mut()
            .insert(BearerToken::new("test-token"));
        handle_subscribe_read_layer(&self.replicas[r].svc, request)
            .await
            .unwrap_or_else(|s| panic!("{client_id}: the subscribe must be admitted: {s:?}"))
            .into_inner()
    }

    /// A brand-new join on replica `r`, drained to its caught-up ack. Returns
    /// the cut it was served at.
    async fn join_served(&self, r: usize, client_id: &str) -> u64 {
        let mut stream = self.join(r, client_id).await;
        recv_served(&mut stream, self.oid, client_id).await
    }

    /// One `UpdateLayer` of `delta` through replica `r`. The handler call is
    /// boxed: on the WATCH path (`LuaWatchStructural`) its future is too large
    /// to hold inline in this helper's frame on a debug build's test-thread
    /// stack.
    async fn commit(&self, r: usize, delta: &OwnedDelta) -> Result<(), Status> {
        Box::pin(handle_update_layer(
            &self.replicas[r].svc,
            make_request(streaming_update(vec![
                make_update_header(&self.session_id, &self.layer, self.oid),
                make_update_data(delta_to_ompf_bytes(delta, &[])),
            ])),
        ))
        .await
        .map(|_| ())
    }

    /// The layer's header sequence now.
    async fn seq(&self) -> u64 {
        self.replicas[0]
            .svc
            .layer_store
            .read_object_header(self.oid)
            .await
            .expect("the header reads")
            .sequence_num
    }

    fn fence_key(&self) -> String {
        self.keys().obj_snapshot_fence(&self.oid.to_string())
    }

    /// The store's clock, as the fence scripts read it.
    async fn now(&self) -> i64 {
        CLOCK_SCRIPT
            .evalsha_with_reload::<i64, _, _>(
                self.tv.client().pool.next(),
                vec![self.fence_key()],
                Vec::<String>::new(),
            )
            .await
            .expect("TIME must answer")
    }

    /// The fence key's reader leases' deadlines (`r:<token>` → deadline).
    async fn lease_deadlines(&self) -> Vec<i64> {
        let kind: String = self
            .tv
            .client()
            .pool
            .r#type(self.fence_key())
            .await
            .expect("TYPE fence must succeed");
        if kind != "hash" {
            return Vec::new();
        }
        let fields: std::collections::HashMap<String, String> = self
            .tv
            .client()
            .pool
            .hgetall(self.fence_key())
            .await
            .expect("HGETALL fence must succeed");
        fields
            .into_iter()
            .filter(|(k, _)| k.starts_with("r:"))
            .map(|(_, v)| v.parse().expect("a deadline is an integer"))
            .collect()
    }

    /// Wait until no reader lease on the fence key is live.
    async fn until_no_live_lease(&self, what: &str) {
        let wait = async {
            loop {
                let now = self.now().await;
                if self.lease_deadlines().await.iter().all(|d| *d <= now) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        };
        tokio::time::timeout(LAPSE_BUDGET, wait)
            .await
            .unwrap_or_else(|_| panic!("{what}: a lease was still live after {LAPSE_BUDGET:?}"));
    }
}

/// A delta adding each of `section_ids` under the root: a structural commit
/// (the slow path).
fn sections_delta(section_ids: &[u64]) -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: section_ids
            .iter()
            .map(|&section_id| {
                StructCommand::DiffChangeSection(CommandDiffChangeSection {
                    parent_id: 0,
                    section_id,
                    section_name: format!("sec-{section_id}"),
                    section_type: 0,
                    section_order: 0,
                })
            })
            .collect(),
        ..Default::default()
    }
}

/// A delta setting one field of an existing section: a fast-path commit.
fn field_delta(section_id: u64, value: u64) -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        base_version: 0,
        set_fields: vec![SetField {
            section_id,
            key_name: "stream:value".to_owned(),
            value_or_ext_hash: value.to_le_bytes().to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    }
}

// ── Stream helpers ──────────────────────────────────────────────────────────

/// The next item, which must arrive within [`SETTLE`] and not be end of
/// stream.
async fn next_item(
    stream: &mut SubStream,
    what: &str,
) -> Result<SubscribeReadLayerResponse, Status> {
    tokio::time::timeout(SETTLE, stream.next())
        .await
        .unwrap_or_else(|_| panic!("{what} did not arrive within {SETTLE:?}"))
        .unwrap_or_else(|| panic!("the stream ended cleanly before {what}"))
}

/// One header with its reassembled body.
async fn recv_message(stream: &mut SubStream, what: &str) -> (SubscribeReadLayerHeader, Vec<u8>) {
    let header = match next_item(stream, what).await {
        Ok(SubscribeReadLayerResponse {
            response_type: Some(subscribe_read_layer_response::ResponseType::Header(h)),
        }) => h,
        other => panic!("{what}: expected a header, got {other:?}"),
    };
    let mut body = Vec::new();
    while (body.len() as u64) < header.total_size {
        match next_item(stream, what).await {
            Ok(SubscribeReadLayerResponse {
                response_type: Some(subscribe_read_layer_response::ResponseType::Chunk(c)),
            }) => body.extend_from_slice(&c.data),
            other => panic!("{what}: expected a body chunk, got {other:?}"),
        }
    }
    (header, body)
}

/// A brand-new catch-up's content and its caught-up ack, under one [`DRAIN`]
/// deadline: the content at `(object_id, cut)` and the ack at the same cut —
/// a catch-up the service served only after its fenced read was proven
/// (`VERIFIED`) at that cut. Returns the cut.
async fn recv_served(stream: &mut SubStream, object_id: u64, who: &str) -> u64 {
    tokio::time::timeout(DRAIN, async {
        let (content, body) = recv_message(stream, &format!("{who}'s content")).await;
        assert!(
            content.total_size > 0 && !body.is_empty(),
            "{who}: the first message must be content, not the ack: {content:?}"
        );
        let (ack, _) = recv_message(stream, &format!("{who}'s caught-up ack")).await;
        assert_eq!(
            ack.total_size, 0,
            "{who}: expected the caught-up ack: {ack:?}"
        );
        assert_eq!(
            (content.object_id, ack.object_id, ack.sequence_num),
            (object_id, object_id, content.sequence_num),
            "{who}: the ack carries the cut the content serves"
        );
        content.sequence_num
    })
    .await
    .unwrap_or_else(|_| panic!("{who}: the catch-up did not finish within {DRAIN:?}"))
}

/// Poll `condition` until it holds or [`SETTLE`] elapses. The sleep is the
/// probe interval, not a coordination primitive.
async fn await_condition<F: FnMut() -> bool>(mut condition: F, what: &str) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    while !condition() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what} did not hold within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Structural writers under a stream of fenced reads
// ═════════════════════════════════════════════════════════════════════════════

/// Spec SDD Test Fixtures, "structural writers under a stream": on a layer
/// far above the ceiling, closed-loop structural writers (`CreateSection`)
/// keep committing while closed-loop joiners keep reading it through the
/// fence. Every commit lands — no retry-budget exhaustion, no fence-wait
/// timeout — and the conflicts the fence causes are fence waits, uncounted.
///
/// Runs [`STRUCTURAL_WRITERS`] closed-loop structural writers and
/// [`STREAM_JOINERS`] closed-loop joiners on `fx`'s above-ceiling layer, both
/// on replica 0, and asserts what holds on every structural route.
///
/// The writers start after the first fenced read, so the fence key exists
/// and every structural attempt meets it. The run is bounded by counts, not
/// time: it ends once [`JOINS_UNDER_WRITERS`] joins were served AND
/// [`COMMITS_UNDER_READERS`] commits landed while both ran. A reader starved
/// by the writers, or writers starved by the epochs, never reaches its count
/// and fails the test at [`TEST_BUDGET`].
async fn structural_stream_commits_every_write(fx: Arc<Fx>) {
    let first = fx.join_served(0, "warm-up").await;
    let seq0 = fx.seq().await;
    assert_eq!(
        first, seq0,
        "fixture: the first join reads the layer as created"
    );
    let epochs0 = fx.replicas[0].acquires("started_epoch");

    let stop = Arc::new(AtomicBool::new(false));
    let writers_running = Arc::new(AtomicBool::new(false));
    let joins = Arc::new(AtomicU64::new(0));
    let commits = Arc::new(AtomicU64::new(0));

    let mut joiners = Vec::new();
    for j in 0..STREAM_JOINERS {
        let (fx, stop, joins, running) = (
            Arc::clone(&fx),
            Arc::clone(&stop),
            Arc::clone(&joins),
            Arc::clone(&writers_running),
        );
        joiners.push(tokio::spawn(async move {
            let mut n = 0u64;
            while !stop.load(Ordering::SeqCst) {
                let counted = running.load(Ordering::SeqCst);
                fx.join_served(0, &format!("joiner-{j}-{n}")).await;
                if counted {
                    joins.fetch_add(1, Ordering::SeqCst);
                }
                n += 1;
            }
        }));
    }
    let mut writers = Vec::new();
    for w in 0..STRUCTURAL_WRITERS {
        let (fx, stop, commits) = (Arc::clone(&fx), Arc::clone(&stop), Arc::clone(&commits));
        writers.push(tokio::spawn(async move {
            let mut errors = Vec::new();
            let mut n = 0u64;
            while !stop.load(Ordering::SeqCst) {
                let section = FIRST_NEW_SECTION + w * 10_000 + n;
                match fx.commit(0, &sections_delta(&[section])).await {
                    Ok(()) => {
                        commits.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(status) => errors.push(format!("writer {w} commit {n}: {status:?}")),
                }
                n += 1;
            }
            errors
        }));
    }
    writers_running.store(true, Ordering::SeqCst);

    // The probe interval is not a synchronisation: the loop ends on the
    // two counts, or as soon as a task ended early (a panic, surfaced by
    // its join below).
    while joins.load(Ordering::SeqCst) < JOINS_UNDER_WRITERS
        || commits.load(Ordering::SeqCst) < COMMITS_UNDER_READERS
    {
        if writers.iter().any(|t| t.is_finished()) || joiners.iter().any(|t| t.is_finished()) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    stop.store(true, Ordering::SeqCst);
    let mut errors = Vec::new();
    for w in writers {
        errors.extend(w.await.expect("a writer task must not panic"));
    }
    for j in joiners {
        j.await.expect("a joiner task must not panic");
    }

    let r = &fx.replicas[0];
    assert!(
        errors.is_empty(),
        "every structural commit under the stream lands (no retry-budget exhaustion, no \
             fence-wait cap): {errors:?}"
    );
    let landed = commits.load(Ordering::SeqCst);
    assert_eq!(
        fx.seq().await - seq0,
        landed,
        "each acknowledged structural commit advanced the layer exactly once"
    );
    assert_eq!(r.fence_timeouts(), 0, "no commit hit the fence-wait cap");
    assert!(
        r.fence_waits() >= 1,
        "the writers met the readers' fence (fence-caused conflicts are fence waits); else \
             the stream never overlapped a commit and the test proves nothing"
    );
    assert!(
        r.acquires("started_epoch") > epochs0,
        "the readers kept starting epochs while the writers committed"
    );
    r.assert_never_capped();
}

/// The WATCH path (`lua-watch-structural`): every structural attempt on the
/// fenced layer finds the fence key and registers a writer intent (SR-5.4),
/// and every commit still lands.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_fence_stream_structural_writers_under_fenced_reads_all_commit_without_retry_exhaustion(
) {
    within_test_budget(async {
        let fx = Arc::new(
            Fx::with_backend(
                "stream-structural",
                &["S"],
                UpdateLayerBackend::LuaWatchStructural,
            )
            .await,
        );
        structural_stream_commits_every_write(Arc::clone(&fx)).await;
        assert!(
            fx.replicas[0].intents("registered") >= 1,
            "the structural attempts on the fenced layer registered writer intents"
        );
    })
    .await;
}

/// The cached structural route (`lua`): the cache cold-starts the
/// above-ceiling layer through its own fenced read, once, and every commit
/// stays on the cached route (no fenced read failed for its call, no call took
/// the WATCH path); its commits wait out the readers' fences inside the commit
/// script and register no writer intent (structural-edits SR-8.2), and every
/// commit still lands while the joins keep starting epochs.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_fence_stream_cached_structural_writers_under_fenced_reads_all_commit_without_intents(
) {
    within_test_budget(async {
        let fx = Arc::new(Fx::new("stream-structural-cached", &["S"]).await);
        structural_stream_commits_every_write(Arc::clone(&fx)).await;
        let r = &fx.replicas[0];
        assert_eq!(
            r.cold_starts("fenced"),
            1,
            "the cache loaded the above-ceiling layer through its fenced read, once"
        );
        assert_eq!(
            r.metric(
                "live_edit_structural_cache_uncacheable_total",
                &[("reason", "fenced_read_failed")],
            ),
            0.0,
            "no fenced cold start failed for its call (which would send that call to WATCH)"
        );
        assert_eq!(
            r.metric(
                "live_edit_update_layer_attempts_histogram_count",
                &[("path", "watch")],
            ),
            0.0,
            "no call took the WATCH path: every commit stayed on the cached route"
        );
        assert!(
            r.metric(
                "live_edit_update_layer_attempts_histogram_count",
                &[("path", "structural")],
            ) + r.metric(
                "live_edit_update_layer_attempts_histogram_count",
                &[("path", "structural_full")],
            ) >= 1.0,
            "vacuity guard: the calls were counted, on the cached route's paths"
        );
        assert_eq!(
            r.intents("registered"),
            0,
            "the cached structural route registers no writer intent"
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Readers under a stream from two replicas
// ═════════════════════════════════════════════════════════════════════════════

/// Spec SDD Test Fixtures, "readers under a stream from several replicas":
/// two `LiveEditService` instances, each with its own client (slots, fenced
/// read gate, memory budget, multiplexer), share one store. Brand-new joins of
/// one layer alternate between them, one every [`JOIN_PERIOD`] (0.2 joins/s
/// per replica, below one replica's serving rate), while [`FIELD_WRITERS`]
/// closed-loop fast-path writers commit through both.
///
/// Every join is served verified (content and ack at one cut; every fenced
/// read counted `ok`, none broken), none reaches `reader_wait_max(L)` or a
/// capacity refusal, each join is its own epoch, and the cuts strictly
/// increase from join to join: a commit landed in every writers' gap between
/// two epochs (counted, not timed; SR-5.5).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_fence_stream_two_replicas_serve_every_join_verified_and_writers_commit_in_every_gap(
) {
    within_test_budget(async {
        let fx = Arc::new(Fx::new("stream-two-replicas", &["S", "W"]).await);
        let stop = Arc::new(AtomicBool::new(false));
        let mut writers = Vec::new();
        for w in 0..FIELD_WRITERS {
            let (fx, stop) = (Arc::clone(&fx), Arc::clone(&stop));
            writers.push(tokio::spawn(async move {
                let (mut ok, mut errors, mut n) = (0u64, Vec::new(), 0u64);
                while !stop.load(Ordering::SeqCst) {
                    let via = (w as usize + n as usize) % 2;
                    match fx.commit(via, &field_delta(1 + w, n)).await {
                        Ok(()) => ok += 1,
                        Err(status) => errors.push(format!("writer {w} commit {n}: {status:?}")),
                    }
                    n += 1;
                }
                (ok, errors)
            }));
        }

        let mut pace = tokio::time::interval(JOIN_PERIOD);
        pace.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut cuts = Vec::new();
        for i in 0..STREAM_JOINS {
            pace.tick().await;
            let r = i % 2;
            cuts.push(fx.join_served(r, &format!("joiner-{i}")).await);
            // The flight's read stage releases its lease (the release script
            // removes the reader's own field) before the catch-up is built and
            // served, and no other reader is live at this spacing: so once a
            // join is served the fence holds no reader lease. An epoch that did
            // not end at its release would still hold one here, until it lapsed.
            let leases = fx.lease_deadlines().await;
            assert!(
                leases.is_empty(),
                "joiner-{i}: once served, its epoch has ended at its release — no reader lease \
                 may remain on the fence: {leases:?}"
            );
        }
        stop.store(true, Ordering::SeqCst);
        let (mut landed, mut errors) = (0u64, Vec::new());
        for w in writers {
            let (ok, errs) = w.await.expect("a writer task must not panic");
            landed += ok;
            errors.extend(errs);
        }

        assert!(errors.is_empty(), "every commit lands: {errors:?}");
        assert!(landed > 0);
        let epochs: u64 = fx
            .replicas
            .iter()
            .map(|r| r.acquires("started_epoch"))
            .sum();
        let verified: u64 = fx.replicas.iter().map(|r| r.fenced_reads("ok")).sum();
        assert_eq!(
            (epochs, verified),
            (STREAM_JOINS as u64, STREAM_JOINS as u64),
            "each paced join is its own epoch, and each epoch's read was proven"
        );
        assert!(
            cuts.windows(2).all(|w| w[0] < w[1]),
            "every epoch's cut is past the one before it: the writers committed in every gap \
             between two epochs: {cuts:?}"
        );
        assert_eq!(
            cuts.iter().collect::<BTreeSet<_>>().len(),
            STREAM_JOINS,
            "one distinct cut per epoch"
        );
        for r in &fx.replicas {
            r.assert_never_capped();
            assert_eq!(
                r.fence_timeouts(),
                0,
                "replica {}: no fence-wait cap",
                r.name
            );
        }
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// A missed join window
// ═════════════════════════════════════════════════════════════════════════════

/// Spec SDD SR-5.6 / SR-7.5 and the Task 10 scenario "one replica's budget is
/// held by a build chain at the gap end": replica S's flight cannot take its
/// combined memory reservation (the test holds S's whole catch-up memory
/// budget, as a long build chain would), so its read stage waits — charged,
/// within its roomy capacity budget — while replica W's join starts an epoch
/// and ends it. S misses that epoch's join window. Once the memory is free S
/// acquires after the gap, starts its own epoch and is served, at a later cut
/// (a commit landed between the two epochs), never refused and never at
/// `reader_wait_max(L)`. Two epochs are counted, one per replica.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_fence_stream_replica_holding_its_memory_misses_the_join_window_and_is_still_served(
) {
    within_test_budget(async {
        let fx = Arc::new(Fx::new("stream-missed-window", &["S", "W"]).await);
        let (s, w) = (&fx.replicas[0], &fx.replicas[1]);
        let budget = &s.client.catchup_build_budget;
        let held = budget
            .reserve(budget.total_bytes(), None)
            .await
            .unwrap_or_else(|r| panic!("the whole budget must be free: {r:?}"));

        let s_join = {
            let fx = Arc::clone(&fx);
            tokio::spawn(async move { fx.join_served(0, "s-joiner").await })
        };
        await_condition(
            || s.client.catchup_gate.available() == SLOTS - 1,
            "S's flight holds its slot and waits in its read stage",
        )
        .await;
        assert_eq!(s.acquires("started_epoch") + s.acquires("joined_epoch"), 0);

        let w_cut = fx.join_served(1, "w-joiner").await;
        assert_eq!(
            (s.acquires("started_epoch"), s.acquires("joined_epoch")),
            (0, 0),
            "S has not reached the fence while its memory is held"
        );
        fx.commit(1, &field_delta(1, 1))
            .await
            .expect("a commit in the gap after W's epoch lands");
        drop(held);

        let s_cut = tokio::time::timeout(DRAIN, s_join)
            .await
            .expect("S's join completes once its memory is free")
            .expect("S's join task must not panic");
        assert!(
            s_cut > w_cut,
            "S read in its own, later epoch, after the commit in the gap: {s_cut} > {w_cut}"
        );
        assert_eq!(
            (
                s.acquires("started_epoch"),
                s.acquires("joined_epoch"),
                w.acquires("started_epoch"),
            ),
            (1, 0, 1),
            "two epochs: S missed W's join window and started its own"
        );
        for r in &fx.replicas {
            r.assert_never_capped();
        }
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// A crashed reader's epoch is still followed by the writers' gap
// ═════════════════════════════════════════════════════════════════════════════

/// Spec SDD SR-5.2 (the crash gap): a reader that takes the fence and then
/// neither renews nor releases it (a crashed process: its lease guard is
/// leaked, so not even its drop's best-effort release runs) holds the
/// writers off only until its lease lapses. The next reader is then still
/// refused `RETRY_AFTER gap` — the epoch ended by expiry is followed by the
/// writers' gap like any other — a commit lands in that gap, and the next
/// epoch starts after it, at a cut that includes it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_fence_stream_crashed_readers_epoch_is_followed_by_the_writers_gap() {
    within_test_budget(async {
        let fx = Fx::new("stream-crashed-reader", &["S"]).await;
        let client = &fx.replicas[0].client;
        let estimate = LayerSize {
            items: LAYER_SECTIONS * 8,
            bytes: LAYER_SECTIONS * 100,
        };
        let crashed = match fence::acquire_fence(client, fx.keys(), fx.oid, estimate)
            .await
            .expect("acquire must answer")
        {
            AcquireVerdict::Granted(lease) => lease,
            other => panic!("the first reader must be granted: {other:?}"),
        };
        assert_eq!(crashed.kind(), GrantKind::StartedEpoch);
        let crashed_cut = crashed.cut();
        // The crash: no renewal, no release, not even the guard's drop.
        std::mem::forget(crashed);

        fx.until_no_live_lease("the crashed reader's lease").await;
        match fence::acquire_fence(client, fx.keys(), fx.oid, estimate)
            .await
            .expect("acquire must answer")
        {
            AcquireVerdict::RetryAfter { reason, .. } => assert_eq!(
                reason,
                RetryReason::Gap,
                "after the lapse the writers' gap still holds readers off"
            ),
            other => panic!("the next reader must wait for the writers' gap: {other:?}"),
        }

        fx.commit(0, &field_delta(1, 7))
            .await
            .expect("a commit in the gap after the crashed epoch lands");
        let committed = fx.seq().await;
        assert_eq!(committed, crashed_cut + 1, "exactly the one commit landed");

        let next = tokio::time::timeout(LAPSE_BUDGET, async {
            loop {
                match fence::acquire_fence(client, fx.keys(), fx.oid, estimate)
                    .await
                    .expect("acquire must answer")
                {
                    AcquireVerdict::Granted(lease) => return lease,
                    AcquireVerdict::RetryAfter { hint, .. } => tokio::time::sleep(hint).await,
                    AcquireVerdict::NotFound => panic!("the layer must still exist"),
                }
            }
        })
        .await
        .unwrap_or_else(|_| panic!("the next reader was not granted within {LAPSE_BUDGET:?}"));
        assert_eq!(next.kind(), GrantKind::StartedEpoch);
        assert_eq!(
            next.cut(),
            committed,
            "the next epoch's cut includes the commit made in the crash gap"
        );
        assert_eq!(
            (*next).release(None).await.expect("release must answer"),
            ReleaseVerdict::Verified
        );
    })
    .await;
}
