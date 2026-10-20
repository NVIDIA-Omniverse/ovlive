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

//! The guard's fixtures: the measured service, the layer content, the gRPC
//! calls the cases make, and the metric reads their route checks rest on.
//!
//! Setup data is written on the test's own runtime through a separate setup
//! service (its own `LayerStore`, so its structural cache is not the measured
//! one), never through the measured server.

use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface};
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand};
use tonic::codec::Streaming;
use tonic::metadata::MetadataValue;
use tonic::transport::Channel;
use tonic::{Request, Status};

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    CheckpointMeta, CheckpointStore, RecoveryHook, SessionCheckpoint, SessionRecoveryHook,
};
use live_edit::config::{UpdateLayerBackend, MAX_LAYER_SIZE_DEFAULT, RPC_DEADLINE_MS_DEFAULT};
use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::live_service_client::LiveServiceClient;
use live_edit::grpc::proto::{
    subscribe_read_layer_response, CreateLayerResponse, SubscribeReadLayerRequest,
    SubscribeReadLayerResponse, UpdateLayerResponse,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::session_cache::SessionBasePathCache;
use live_edit::grpc::upload_admission::{UploadAdmissionConfig, UploadAdmissionGate};
use live_edit::observability::metrics::test_util::{render, sample_sum};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::external_values::ValkeyExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::snapshot_flights::SnapshotFlights;
use live_edit::storage::test_support::{connect_test_client, TestValkey};

use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::harness::{SERVICE_LABEL, SETTLE};
use crate::helpers::{
    build_disabled_recovery_hook, build_service, make_create_data, make_create_header,
    make_request, make_update_data, make_update_header, streaming_create, streaming_update,
    AllowAllPermissions,
};

pub type Client = LiveServiceClient<Channel>;

/// The bearer every request carries; the mock permission client allows it.
const BEARER: &str = "stack-guard-bearer";

/// Base path of every session here.
pub const BASE_PATH: &str = "/stage";

/// The layer every case works on.
pub const LAYER: &str = "stack-guard-layer";

/// The subscriber's `client_id`: no writer here uses it, so self-echo
/// suppression never hides a commit from it.
const SUBSCRIBER: &str = "stack-guard-subscriber";

/// Content per upload message, as clients chunk it.
const CHUNK: usize = 1024 * 1024;

/// Sections in [`base_content`], each a child of the root.
pub const BASE_SECTIONS: u64 = 8;

/// The section [`field_update`] writes to.
pub const FIELD_SECTION: u64 = 1;

/// The client ceilings of [`low_ceiling_client`]: far below
/// [`base_content`]'s size, so every atomic read of it answers `TOO_LARGE`.
const LOW_ITEMS: u64 = 16;
const LOW_BYTES: u64 = 256;

/// `request` with the bearer the interceptor requires.
pub fn authed<T>(message: T) -> Request<T> {
    let mut request = Request::new(message);
    request.metadata_mut().insert(
        "authorization",
        MetadataValue::try_from(format!("Bearer {BEARER}")).expect("a valid header"),
    );
    request
}

// ── Content ─────────────────────────────────────────────────────────────────

fn create_section(parent_id: u64, section_id: u64) -> StructCommand {
    StructCommand::CreateSection(CommandCreateSection {
        parent_id,
        section_id,
        section_name: format!("s{section_id}"),
        section_type: 0,
    })
}

fn inline_field(section_id: u64, key: String, len: usize) -> SetField {
    SetField {
        section_id,
        key_name: key,
        value_or_ext_hash: vec![0x5a; len],
        ext_value_size: 0,
        ext_value_index: 0,
        set_order: 0,
    }
}

/// An ordinary layer: [`BASE_SECTIONS`] sections under the root, four 64-byte
/// fields on each.
pub fn base_content() -> OwnedDelta {
    OwnedDelta {
        struct_commands: (1..=BASE_SECTIONS)
            .map(|id| create_section(0, id))
            .collect(),
        set_fields: (1..=BASE_SECTIONS)
            .flat_map(|id| (0..4).map(move |f| inline_field(id, format!("f{f}"), 64)))
            .collect(),
        ..Default::default()
    }
}

/// A linear chain of `len` sections, ids `1..=len`, each the child of the
/// one before: a section tree `len` deep.
pub fn chain(len: usize) -> OwnedDelta {
    OwnedDelta {
        struct_commands: (1..=len as u64)
            .map(|id| create_section(id - 1, id))
            .collect(),
        ..Default::default()
    }
}

/// A field-only update of [`FIELD_SECTION`]: the Lua fast path.
pub fn field_update(salt: u64) -> OwnedDelta {
    OwnedDelta {
        set_fields: (0..4)
            .map(|f| inline_field(FIELD_SECTION, format!("u{salt}-{f}"), 64))
            .collect(),
        ..Default::default()
    }
}

/// A structural update: one new section `section_id` under `parent_id`.
pub fn structural_update(parent_id: u64, section_id: u64) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![create_section(parent_id, section_id)],
        ..Default::default()
    }
}

/// A reset (`is_diff`, `base_version` 0) replacing the layer with
/// [`base_content`].
pub fn reset_update() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        ..base_content()
    }
}

pub fn ompf(delta: &OwnedDelta) -> Vec<u8> {
    delta_to_ompf_bytes(delta, &[])
}

// ── The fixture ─────────────────────────────────────────────────────────────

/// How a case's measured service is configured.
pub struct Opts {
    /// `LIVE_EDIT_UPDATE_LAYER_BACKEND`.
    pub backend: UpdateLayerBackend,
    /// `LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED` (production default: on).
    pub snapshot: bool,
    /// An enabled layer recovery hook over this store, as with checkpointing
    /// on; disabled when `None`.
    pub layer_checkpoints: Option<Arc<dyn CheckpointStore>>,
    /// An enabled session recovery hook over this store, wired into the
    /// base-path cache as `main.rs` wires it; disabled when `None`.
    pub session_checkpoints: Option<Arc<dyn CheckpointStore>>,
    /// The Valkey client to serve over; the fixture's own when `None`.
    pub client: Option<ValkeyClient>,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            backend: UpdateLayerBackend::Lua,
            snapshot: true,
            layer_checkpoints: None,
            session_checkpoints: None,
            client: None,
        }
    }
}

/// A Valkey namespace and a setup service over it.
pub struct Fx {
    pub tv: TestValkey,
    setup: LiveEditService,
}

impl Fx {
    pub async fn new() -> Self {
        let tv = TestValkey::setup().await;
        let setup = build_service(&tv, Arc::new(AllowAllPermissions));
        Fx { tv, setup }
    }

    pub async fn session(&self) -> String {
        SessionStore::new(self.tv.client().clone(), self.tv.key_schema().clone())
            .create("stack-guard-session", BASE_PATH, "test-user")
            .await
            .expect("fixture: session creation")
    }

    /// Create [`LAYER`] in `session_id` with `content`, through the setup
    /// service; its object id.
    pub async fn layer(&self, session_id: &str, content: &OwnedDelta) -> u64 {
        let messages = std::iter::once(make_create_header(session_id, LAYER))
            .chain(
                ompf(content)
                    .chunks(CHUNK)
                    .map(|c| make_create_data(c.to_vec())),
            )
            .collect();
        handle_create_layer(&self.setup, make_request(streaming_create(messages)))
            .await
            .expect("fixture: CreateLayer")
            .into_inner()
            .object_id
    }

    /// A session holding [`LAYER`] with `content`.
    pub async fn session_with_layer(&self, content: &OwnedDelta) -> (String, u64) {
        let session_id = self.session().await;
        let object_id = self.layer(&session_id, content).await;
        (session_id, object_id)
    }

    /// Commit `delta` to [`LAYER`] through the setup service; the new
    /// sequence number.
    pub async fn update(&self, session_id: &str, object_id: u64, delta: &OwnedDelta) -> u64 {
        let messages = vec![
            make_update_header(session_id, LAYER, object_id),
            make_update_data(ompf(delta)),
        ];
        handle_update_layer(&self.setup, make_request(streaming_update(messages)))
            .await
            .expect("fixture: UpdateLayer")
            .into_inner()
            .sequence_num
    }

    /// Commit `delta` through `store`, on this runtime, as another replica's
    /// writer would; the new sequence number.
    pub async fn commit_through(
        store: &LayerStore,
        session_id: &str,
        object_id: u64,
        delta: OwnedDelta,
    ) -> u64 {
        let relay = bytes::Bytes::from(ompf(&delta));
        store
            .update_layer(
                session_id,
                LAYER,
                object_id,
                Some(delta),
                Some(relay),
                "stack-guard-writer",
                0,
            )
            .await
            .expect("fixture: commit through the store")
            .0
    }

    /// The service to measure, configured as production configures it where
    /// that reaches the measured paths: the real external-value store, the
    /// production layer-size cap and upload gate, the fence-wait deadline,
    /// metrics on the store and the client.
    pub async fn service(&self, opts: Opts) -> LiveEditService {
        let base = build_service(&self.tv, Arc::new(AllowAllPermissions));
        let metrics = Arc::clone(&base.metrics);
        let client = opts.client.unwrap_or_else(|| self.tv.client().clone());
        client
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let keys = self.tv.key_schema().clone();
        let layer_store = Arc::new(
            LayerStore::new(client.clone(), keys.clone(), 4096)
                .with_update_layer_backend(opts.backend)
                .with_fence_wait_deadline(
                    Duration::from_millis(RPC_DEADLINE_MS_DEFAULT),
                    Arc::new(live_edit::grpc::deadline::remaining),
                )
                .with_metrics(Arc::clone(&metrics)),
        );
        let recovery = match opts.layer_checkpoints {
            Some(store) => Arc::new(RecoveryHook::new(
                store,
                Arc::clone(&layer_store),
                Arc::new(checkpointing_config()),
                Arc::clone(&metrics),
            )),
            None => build_disabled_recovery_hook(Arc::clone(&layer_store), Arc::clone(&metrics)),
        };
        let (session_recovery, session_base_path_cache) = match opts.session_checkpoints {
            Some(store) => {
                let hook = Arc::new(SessionRecoveryHook::new(
                    store,
                    client.clone(),
                    keys.clone(),
                    Arc::new(checkpointing_config()),
                    Arc::clone(&metrics),
                ));
                let cache = SessionBasePathCache::new().with_recovery_hook(Arc::clone(&hook));
                (hook, cache)
            }
            None => (base.session_recovery.clone(), SessionBasePathCache::new()),
        };
        LiveEditService {
            catchup_snapshot_enabled: opts.snapshot,
            session_store: Arc::new(SessionStore::new(client.clone(), keys.clone())),
            layer_store,
            ext_value_store: Arc::new(ValkeyExternalValueStore::new(client.clone(), keys.clone())),
            max_layer_size: MAX_LAYER_SIZE_DEFAULT,
            upload_admission: Arc::new(UploadAdmissionGate::new(
                UploadAdmissionConfig::default(),
                Some(Arc::clone(&metrics)),
            )),
            snapshot_flights: SnapshotFlights::new(client.clone(), keys.clone()),
            valkey: client,
            key_schema: keys,
            recovery,
            session_recovery,
            session_base_path_cache,
            metrics,
            ..base
        }
    }

    /// `UNLINK` every key of `object_id`, leaving the session and its layer
    /// mapping: the shape a lost object leaves.
    pub async fn wipe_object(&self, object_id: u64) {
        crate::common::wipe_object_keys(&self.tv, object_id).await;
    }

    /// `UNLINK` the layer mapping of `session_id`'s [`LAYER`], leaving the
    /// session and the object.
    pub async fn wipe_mapping(&self, session_id: &str) {
        let removed: i64 = self
            .tv
            .client()
            .pool
            .unlink(self.tv.key_schema().layer_mapping(session_id, LAYER))
            .await
            .expect("fixture: UNLINK the layer mapping");
        assert_eq!(removed, 1, "fixture: the layer mapping existed");
    }

    /// `XTRIM MINID` the layer stream of `object_id` to `min_seq`: every
    /// entry below it is gone, as retention trims it.
    pub async fn trim_stream_below(&self, object_id: u64, min_seq: u64) {
        use fred::interfaces::StreamsInterface;
        let cap = fred::types::streams::XCap::try_from(("MINID", "=", format!("{min_seq}-0")))
            .expect("a MINID cap");
        let _: () = self
            .tv
            .client()
            .pool
            .xtrim(self.tv.key_schema().layer_stream(object_id), cap)
            .await
            .expect("fixture: XTRIM");
    }

    /// Whether `object_id`'s header still exists.
    pub async fn header_exists(&self, object_id: u64) -> bool {
        let found: bool = self
            .tv
            .client()
            .pool
            .exists(self.tv.key_schema().obj_header(&object_id.to_string()))
            .await
            .expect("fixture: EXISTS");
        found
    }

    /// The session checkpoint of the live session `session_id` holding
    /// [`LAYER`] at `object_id`: what the session checkpoint writer would
    /// put at the adapter.
    pub async fn session_checkpoint(&self, session_id: &str, object_id: u64) -> SessionCheckpoint {
        let meta: std::collections::HashMap<String, String> = self
            .tv
            .client()
            .pool
            .hgetall(self.tv.key_schema().session_meta(session_id))
            .await
            .expect("fixture: HGETALL session meta");
        let field = |name: &str| {
            meta.get(name)
                .unwrap_or_else(|| panic!("fixture: session meta has no `{name}`: {meta:?}"))
                .clone()
        };
        let mut checkpoint = SessionCheckpoint {
            schema_version:
                live_edit::checkpoint::session_payload::SESSION_CHECKPOINT_SCHEMA_VERSION,
            session_id: session_id.to_owned(),
            name: field("name"),
            base_path: field("base_path"),
            created_by: field("created_by"),
            created_at: field("created_at")
                .parse()
                .expect("fixture: created_at is an integer"),
            layer_mappings: std::collections::BTreeMap::from([(LAYER.to_owned(), object_id)]),
            layers_index: std::collections::BTreeSet::from([LAYER.to_owned()]),
            path_name_members: std::collections::BTreeSet::new(),
            path_index_session_id_member: session_id.to_owned(),
        };
        let name = checkpoint.name.clone();
        checkpoint.add_path_name_member(&name, session_id);
        checkpoint
    }

    /// Remove `session_id`'s session-scoped state the way a keyspace loss
    /// does — metadata, layers index, layer mapping and its path-index
    /// members — leaving no deletion marker and the layer's object intact.
    pub async fn flush_session(&self, session_id: &str) {
        let keys = self.tv.key_schema();
        let pool = &self.tv.client().pool;
        let removed: i64 = pool
            .del(vec![
                keys.session_meta(session_id),
                keys.layers_index(session_id),
                keys.layer_mapping(session_id, LAYER),
            ])
            .await
            .expect("fixture: DEL the session-scoped keys");
        assert!(removed >= 1, "fixture: the session existed");
        let _: i64 = pool
            .srem(keys.path_index(BASE_PATH), session_id)
            .await
            .expect("fixture: SREM path index");
        let names: Vec<String> = pool
            .smembers(keys.path_names_index(BASE_PATH))
            .await
            .expect("fixture: SMEMBERS names index");
        for member in names.iter().filter(|m| m.ends_with(session_id)) {
            let _: i64 = pool
                .srem(keys.path_names_index(BASE_PATH), member.as_str())
                .await
                .expect("fixture: SREM names member");
        }
    }
}

/// A test `Config` with checkpointing on, for an enabled recovery hook. Its
/// `service_label` is the label the route checks read.
fn checkpointing_config() -> live_edit::config::Config {
    let mut config = crate::common::test_config();
    config.checkpoint_enabled = true;
    config.service_label = SERVICE_LABEL.to_owned();
    config
}

/// A checkpoint store holding one layer checkpoint of `object_id` at
/// `snap_seq` (an empty reset), mapped to `session_id`'s [`LAYER`].
pub async fn layer_checkpoints(
    session_id: &str,
    object_id: u64,
    snap_seq: u64,
) -> Arc<dyn CheckpointStore> {
    let store = InMemoryCheckpointStore::new();
    crate::common::seed_layer_checkpoint(&store, object_id, session_id, LAYER, snap_seq).await;
    Arc::new(store)
}

/// A checkpoint store holding `checkpoint`.
pub async fn session_checkpoints(checkpoint: &SessionCheckpoint) -> Arc<dyn CheckpointStore> {
    let store = InMemoryCheckpointStore::new();
    store
        .put_session_checkpoint(
            &checkpoint.session_id,
            checkpoint
                .to_bytes()
                .expect("fixture: session checkpoint bytes"),
            CheckpointMeta {
                session_id: checkpoint.session_id.clone(),
                ..CheckpointMeta::default()
            },
        )
        .await
        .expect("fixture: put_session_checkpoint");
    Arc::new(store)
}

/// A client whose atomic-read ceilings are far below [`base_content`], so
/// every catch-up of it takes its object's shared fenced-read flight.
pub async fn low_ceiling_client() -> ValkeyClient {
    let mut config = crate::common::test_config();
    config.snapshot_atomic_read_max_items = LOW_ITEMS;
    config.snapshot_atomic_read_max_bytes = LOW_BYTES;
    config.valkey_connect_timeout_secs = 30;
    connect_test_client(&config)
        .await
        .expect("fixture: the low-ceiling client must connect")
}

// ── The measured calls ──────────────────────────────────────────────────────

/// `CreateLayer` of [`LAYER`] with `content`, chunked as clients chunk it.
pub async fn create_layer(
    client: &mut Client,
    session_id: &str,
    content: &OwnedDelta,
) -> Result<CreateLayerResponse, Status> {
    let messages: Vec<_> = std::iter::once(make_create_header(session_id, LAYER))
        .chain(
            ompf(content)
                .chunks(CHUNK)
                .map(|c| make_create_data(c.to_vec())),
        )
        .collect();
    within(client.create_layer(authed(tokio_stream::iter(messages))))
        .await
        .map(tonic::Response::into_inner)
}

/// `UpdateLayer` of [`LAYER`] at `object_id`: the header, then `delta`'s
/// content, or nothing for a header-only upload.
pub async fn update_layer(
    client: &mut Client,
    session_id: &str,
    object_id: u64,
    delta: Option<&OwnedDelta>,
) -> Result<UpdateLayerResponse, Status> {
    let mut messages = vec![make_update_header(session_id, LAYER, object_id)];
    if let Some(delta) = delta {
        messages.extend(
            ompf(delta)
                .chunks(CHUNK)
                .map(|c| make_update_data(c.to_vec())),
        );
    }
    within(client.update_layer(authed(tokio_stream::iter(messages))))
        .await
        .map(tonic::Response::into_inner)
}

/// Subscribe to [`LAYER`] as a client holding `(have_object_id,
/// have_version)`.
pub async fn subscribe(
    client: &mut Client,
    session_id: &str,
    have_object_id: u64,
    have_version: u64,
) -> Streaming<SubscribeReadLayerResponse> {
    within(
        client.subscribe_read_layer(authed(SubscribeReadLayerRequest {
            session_id: session_id.to_owned(),
            layer_name: LAYER.to_owned(),
            object_id: have_object_id,
            sequence_num: have_version,
            client_id: SUBSCRIBER.to_owned(),
        })),
    )
    .await
    .unwrap_or_else(|s| panic!("the subscription must be admitted: {s:?}"))
    .into_inner()
}

/// What a catch-up delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatchUp {
    /// Content deliveries: headers with a body, each read to its end.
    pub contents: usize,
    /// The caught-up ack's sequence.
    pub ack_seq: u64,
}

/// Read a subscription's catch-up through its caught-up ack (the one
/// `total_size == 0` header), under one deadline for the whole drain.
pub async fn drain_catch_up(stream: &mut Streaming<SubscribeReadLayerResponse>) -> CatchUp {
    tokio::time::timeout(SETTLE, async {
        let mut contents = 0;
        loop {
            match next(stream).await {
                subscribe_read_layer_response::ResponseType::Header(h) if h.total_size == 0 => {
                    return CatchUp {
                        contents,
                        ack_seq: h.sequence_num,
                    };
                }
                subscribe_read_layer_response::ResponseType::Header(h) => {
                    read_body(stream, h.total_size).await;
                    contents += 1;
                }
                subscribe_read_layer_response::ResponseType::Chunk(_) => {
                    panic!("a chunk arrived with no header before it")
                }
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the catch-up did not reach its ack within {SETTLE:?}"))
}

/// Read one relayed commit — its header at `sequence_num`, then its body —
/// under one deadline.
pub async fn read_relay(stream: &mut Streaming<SubscribeReadLayerResponse>, sequence_num: u64) {
    tokio::time::timeout(SETTLE, async {
        match next(stream).await {
            subscribe_read_layer_response::ResponseType::Header(h) => {
                assert_eq!(
                    h.sequence_num, sequence_num,
                    "the relayed header must carry the committed sequence"
                );
                assert!(h.total_size > 0, "a relayed commit carries a body");
                read_body(stream, h.total_size).await;
            }
            other => panic!("expected the relayed commit's header, got {other:?}"),
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the commit was not relayed within {SETTLE:?}"))
}

async fn next(
    stream: &mut Streaming<SubscribeReadLayerResponse>,
) -> subscribe_read_layer_response::ResponseType {
    stream
        .message()
        .await
        .unwrap_or_else(|s| panic!("the subscription failed: {s:?}"))
        .expect("the subscription ended")
        .response_type
        .expect("a response carries a header or a chunk")
}

async fn read_body(stream: &mut Streaming<SubscribeReadLayerResponse>, total_size: u64) {
    let mut read = 0u64;
    while read < total_size {
        match next(stream).await {
            subscribe_read_layer_response::ResponseType::Chunk(c) => read += c.data.len() as u64,
            other => panic!("expected a chunk of a {total_size}-byte body, got {other:?}"),
        }
    }
}

async fn within<F: std::future::Future>(call: F) -> F::Output {
    tokio::time::timeout(SETTLE, call)
        .await
        .unwrap_or_else(|_| panic!("the call did not answer within {SETTLE:?}"))
}

// ── Route checks ────────────────────────────────────────────────────────────

/// The sum of every `name` series whose labels include `labels`.
fn metric(svc: &LiveEditService, name: &str, labels: &[(&str, &str)]) -> f64 {
    sample_sum(&render(&svc.metrics), name, labels).unwrap_or(0.0)
}

/// `UpdateLayer` commits on `path`
/// (`live_edit_update_layer_attempts_histogram`).
pub fn commits_on(svc: &LiveEditService, path: &str) -> f64 {
    metric(
        svc,
        "live_edit_update_layer_attempts_histogram_count",
        &[("path", path)],
    )
}

/// Structural cache lookups that answered `outcome`
/// (`live_edit_structural_cache_lookups_total`): `hit` only on the cached
/// warm route, which plans in place on the entry it took.
pub fn cache_lookups(svc: &LiveEditService, outcome: &str) -> f64 {
    metric(
        svc,
        "live_edit_structural_cache_lookups_total",
        &[("outcome", outcome)],
    )
}

/// Whether `object_id`'s cached entry holds plan indexes whose depth flag is
/// set — the flag under which a plan in place checks its depth incrementally
/// from the ids it listed instead of walking the whole tree
/// (`domain::plan_index`); `None` when the object has no entry. Read by a
/// checkout whose guard puts the entry straight back, unchanged.
pub fn cached_depth_ok(svc: &LiveEditService, object_id: u64) -> Option<bool> {
    let held = svc.layer_store.structural_cache().checkout(object_id)?;
    Some(
        held.entry()
            .plan
            .as_ref()
            .is_some_and(|indexes| indexes.depth_ok()),
    )
}

/// Catch-ups in `regime` delivered as `catchup_mode`
/// (`live_subscribe_regime_total`).
pub fn catch_ups(svc: &LiveEditService, regime: &str, catchup_mode: &str) -> f64 {
    metric(
        svc,
        "live_subscribe_regime_total",
        &[
            ("regime", regime),
            ("catchup_mode", catchup_mode),
            ("service", SERVICE_LABEL),
        ],
    )
}

/// Diff builds that record their case (`live_diff_case_total`): the
/// snapshot mode's faithful builds do, the rollback mode's legacy builds
/// and per-commit replays do not.
pub fn faithful_builds(svc: &LiveEditService) -> f64 {
    metric(svc, "live_diff_case_total", &[("service", SERVICE_LABEL)])
}

/// Checkpoint recoveries of `scope` that ended `result`
/// (`live_edit_checkpoint_recovery_attempts_total`).
pub fn recoveries(svc: &LiveEditService, result: &str, scope: &str) -> f64 {
    metric(
        svc,
        "live_edit_checkpoint_recovery_attempts_total",
        &[
            ("result", result),
            ("scope", scope),
            ("service", SERVICE_LABEL),
        ],
    )
}

/// Restore-source selections whose lineage ended `outcome`
/// (`live_edit_checkpoint_restore_lineage_total`): `gathered_base` for a
/// selection that found a base checkpoint and no tails.
pub fn restore_lineages(svc: &LiveEditService, outcome: &str) -> f64 {
    metric(
        svc,
        "live_edit_checkpoint_restore_lineage_total",
        &[("outcome", outcome), ("service", SERVICE_LABEL)],
    )
}

/// Checkpoint recoveries of `scope` that restored.
pub fn restores(svc: &LiveEditService, scope: &str) -> f64 {
    recoveries(svc, "restored", scope)
}

/// `UpdateLayer` uploads the upload gate admitted
/// (`live_edit_upload_admission_total{rpc="update_layer",result="admitted"}`):
/// the handler admits an upload after it resolves the layer mapping and
/// before it reads the content.
pub fn update_uploads_admitted(svc: &LiveEditService) -> f64 {
    metric(
        svc,
        "live_edit_upload_admission_total",
        &[("rpc", "update_layer"), ("result", "admitted")],
    )
}

/// Fenced-read flights a catch-up created
/// (`live_edit_catchup_snapshot_flight_total{role="leader"}`).
pub fn flights_led(svc: &LiveEditService) -> f64 {
    metric(
        svc,
        "live_edit_catchup_snapshot_flight_total",
        &[("role", "leader")],
    )
}
