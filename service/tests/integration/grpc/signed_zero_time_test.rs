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

//! b.8jc — a time sample at `-0.0` through the real handlers, with bytes that
//! physically carry `-0.0`, as ovlive's encoder sends them (it forces
//! FlatBuffers defaults to keep signed-zero time bits). `oodf_ffi::build_delta`
//! cannot write `-0.0` — FlatBuffers drops a field equal to its default — so
//! every delta here is built at a sentinel time and patched
//! (`common::owned_delta::build_delta_with_negative_zero_times`), and the
//! patch asserts the `-0.0` pattern is in the bytes the handler receives.
//!
//! **Ingest canonicalization (test 4).** Each case commits through
//! `UpdateLayer` on three paths — the `lua` fast path (a time-sample-only
//! delta), the `lua` cached structural route and the `txn` WATCH path (each a
//! delta that also creates a section) — and asserts on the section's time 0:
//! no `…:8000000000000000` key or member anywhere in the object, exactly one
//! row and one `obj:tss` member, and the readers' atomic snapshot read
//! serving exactly that stored row. The cases:
//!
//! | Case | Delta | Survives |
//! |---|---|---|
//! | one `-0.0` write | set `-0.0` | that set |
//! | both signs, `-0.0` first | set `-0.0`, set `0.0` | the last (`0.0`'s) |
//! | both signs, `0.0` first | set `0.0`, set `-0.0` | the last (`-0.0`'s) |
//! | whole-map replace, `0.0` key to `-0.0` | erase `0.0`, set `-0.0` over a stored `0.0` row | the set |
//! | whole-map replace, `-0.0` key to `0.0` | erase `-0.0`, set `0.0` over a stored `0.0` row | the set |
//! | a set at `-0.0` on a stored `0.0` row | set `-0.0` | the set |
//! | an erase at `0.0` of a legacy `-0.0` row | erase `0.0` over a planted older-build row | the erase, served at once; after one merge, the store's one empty `0.0` row |
//!
//! `CreateLayer` takes the first three. Every case whose delta carries `-0.0`
//! fails without the parse boundary's canonicalization
//! (`oodf-ffi/src/convert.rs`): the patched `-0.0` reaches validation, which
//! refuses it (`INVALID_ARGUMENT`). The erase of a legacy row sends no
//! `-0.0`; it pins rule B and the merge on the handler path instead.
//!
//! **The relay (test 5, pins PD1).** A live subscriber receives the writer's
//! bytes unchanged — the `-0.0` included — while a late joiner's catch-up,
//! built from the stored state, carries the time as `0.0`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{HashesInterface, SortedSetsInterface};
use futures::{Stream, StreamExt};
use tonic::{Request, Status};

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::{FieldValue, ObjectState};
use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{
    subscribe_read_layer_response, SubscribeReadLayerHeader, SubscribeReadLayerRequest,
    SubscribeReadLayerResponse,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::storage::diff_read::{
    read_object_snapshot, read_object_state_for_test, snapshot_to_object_state, SnapshotRead,
    SnapshotReadConsumer,
};
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::structural_cache::StructuralCache;
use live_edit::storage::test_support::TestValkey;
use live_edit::storage::time_sample_merge::{
    merge_signed_zero_rows, negative_zero_sections, NEGATIVE_ZERO_TIME_HEX,
};
use oodf_ffi::{CommandCreateSection, OwnedDelta, StructCommand, TimeSample as OodfTimeSample};

use crate::common::owned_delta::{
    carries_negative_zero, delta_to_ompf_bytes_with_negative_zero_times, NEGATIVE_ZERO_SENTINEL,
    NEGATIVE_ZERO_SENTINEL_2,
};

use super::helpers::*;

/// The section every case writes time 0 of.
const SECTION: u64 = 5;
/// The first section id the structural paths' deltas create, one per commit.
const FIRST_EXTRA_SECTION: u64 = 100;
/// Deadline on one subscription read (a message, or a whole catch-up).
const SETTLE: Duration = Duration::from_secs(10);
/// A cache budget no fixture entry comes near.
const BUDGET_BYTES: u64 = 64 * 1024 * 1024;

/// The three `UpdateLayer` paths a case runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Path {
    /// `lua`, a time-sample-only delta: the fast path.
    Fast,
    /// `lua` with a structural cache, a delta that also creates a section:
    /// the cached structural route.
    Cached,
    /// `txn`, a delta that also creates a section: the WATCH path.
    Watch,
}

const UPDATE_PATHS: [Path; 3] = [Path::Fast, Path::Cached, Path::Watch];

/// A service whose store commits on `path`'s backend.
fn service(tv: &TestValkey, path: Path) -> LiveEditService {
    let mut svc = build_service(tv, Arc::new(AllowAllPermissions));
    let store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    svc.layer_store = Arc::new(match path {
        Path::Fast => store.with_update_layer_backend(UpdateLayerBackend::Lua),
        Path::Cached => store
            .with_update_layer_backend(UpdateLayerBackend::Lua)
            .with_structural_cache(StructuralCache::new(BUDGET_BYTES)),
        Path::Watch => store.with_update_layer_backend(UpdateLayerBackend::Txn),
    });
    svc
}

/// One time sample of a case's delta: its time and its inline value (empty:
/// an erase).
#[derive(Debug, Clone, Copy)]
struct Sample {
    negative_zero: bool,
    value: &'static [u8],
}

const fn at_zero(value: &'static [u8]) -> Sample {
    Sample {
        negative_zero: false,
        value,
    }
}

const fn at_negative_zero(value: &'static [u8]) -> Sample {
    Sample {
        negative_zero: true,
        value,
    }
}

/// The OMPF of a delta setting `samples` on [`SECTION`], in order, each
/// `-0.0` one physically carrying `-0.0`; with `create`, it also creates
/// that section under the root (a structural delta).
fn ompf_of(samples: &[Sample], create: Option<u64>) -> Vec<u8> {
    let sentinels = [NEGATIVE_ZERO_SENTINEL, NEGATIVE_ZERO_SENTINEL_2];
    let mut patched = Vec::new();
    let time_samples = samples
        .iter()
        .map(|sample| {
            let time = if sample.negative_zero {
                let sentinel = sentinels[patched.len()];
                patched.push(sentinel);
                sentinel
            } else {
                0.0
            };
            OodfTimeSample {
                section_id: SECTION,
                time,
                value_or_ext_hash: sample.value.to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
            }
        })
        .collect();
    let delta = OwnedDelta {
        struct_commands: create
            .map(|section_id| {
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 0,
                    section_id,
                    section_name: format!("Extra{section_id}"),
                    section_type: 1,
                })
            })
            .into_iter()
            .collect(),
        time_samples,
        ..Default::default()
    };
    let ompf = delta_to_ompf_bytes_with_negative_zero_times(&delta, &patched, &[]);
    assert_eq!(
        carries_negative_zero(&ompf),
        !patched.is_empty(),
        "fixture: the OMPF carries -0.0 exactly when a sample is at -0.0"
    );
    ompf
}

/// One layer on one path, created with [`SECTION`] under the root and,
/// optionally, a stored `0.0` sample of it.
struct Layer {
    tv: TestValkey,
    svc: LiveEditService,
    path: Path,
    session_id: String,
    name: String,
    oid: u64,
    next_extra: std::cell::Cell<u64>,
}

impl Layer {
    async fn new(path: Path, name: &str, stored_zero: Option<&'static [u8]>) -> Self {
        let tv = TestValkey::setup().await;
        let svc = service(&tv, path);
        let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .create(name, "/stage", "test-user")
            .await
            .expect("session creation must succeed");
        let mut content = OwnedDelta {
            struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id: SECTION,
                section_name: "Cube".to_owned(),
                section_type: 1,
            })],
            ..Default::default()
        };
        if let Some(value) = stored_zero {
            content.time_samples.push(OodfTimeSample {
                section_id: SECTION,
                time: 0.0,
                value_or_ext_hash: value.to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
            });
        }
        let oid = create(
            &svc,
            &session_id,
            name,
            delta_to_ompf_bytes_with_negative_zero_times(&content, &[], &[]),
        )
        .await
        .expect("the fixture layer is created");
        Self {
            tv,
            svc,
            path,
            session_id,
            name: name.to_owned(),
            oid,
            next_extra: std::cell::Cell::new(FIRST_EXTRA_SECTION),
        }
    }

    /// Commit `samples` through the `UpdateLayer` handler on this layer's
    /// path; returns the OMPF sent.
    async fn update(&self, samples: &[Sample]) -> Vec<u8> {
        let create = (self.path != Path::Fast).then(|| {
            let id = self.next_extra.get();
            self.next_extra.set(id + 1);
            id
        });
        let ompf = ompf_of(samples, create);
        Box::pin(handle_update_layer(
            &self.svc,
            make_request(streaming_update(vec![
                make_update_header(&self.session_id, &self.name, self.oid),
                make_update_data(ompf.clone()),
            ])),
        ))
        .await
        .unwrap_or_else(|e| panic!("{:?}: UpdateLayer must commit; got {e:?}", self.path));
        ompf
    }

    fn context(&self) -> String {
        format!("{:?} {}", self.path, self.name)
    }
}

/// CreateLayer of `ompf`, returning the object id.
async fn create(
    svc: &LiveEditService,
    session_id: &str,
    name: &str,
    ompf: Vec<u8>,
) -> Result<u64, Status> {
    Box::pin(handle_create_layer(
        svc,
        make_request(streaming_create(vec![
            make_create_header(session_id, name),
            make_create_data(ompf),
        ])),
    ))
    .await
    .map(|r| r.into_inner().object_id)
}

/// Every key of `oid` and every `obj:tss` member, checked for the `-0.0`
/// spelling.
async fn assert_no_negative_zero_key_or_member(tv: &TestValkey, oid: u64, context: &str) {
    let suffix = format!(":{NEGATIVE_ZERO_TIME_HEX}");
    let keys = crate::common::scan_object_keys(tv.client(), tv.prefix(), oid).await;
    let negative: Vec<String> = keys
        .iter()
        .map(|k| k.as_str_lossy().into_owned())
        .filter(|k| k.ends_with(&suffix))
        .collect();
    assert!(
        negative.is_empty(),
        "{context}: no -0.0 row key; got {negative:?}"
    );
    let members = tss_members(tv, oid).await;
    assert!(
        negative_zero_sections(&members).is_empty(),
        "{context}: no -0.0 obj:tss member; got {members:?}"
    );
}

async fn tss_members(tv: &TestValkey, oid: u64) -> Vec<String> {
    tv.client()
        .pool
        .zrange(
            tv.key_schema().obj_time_samples_index(&oid.to_string()),
            0,
            -1,
            None,
            false,
            None,
            false,
        )
        .await
        .expect("ZRANGE obj:tss must succeed")
}

/// What every full reader serves: the atomic snapshot read, decoded.
async fn served(tv: &TestValkey, oid: u64) -> ObjectState {
    match read_object_snapshot(
        tv.client(),
        tv.key_schema(),
        &oid.to_string(),
        SnapshotReadConsumer::Catchup,
    )
    .await
    {
        Ok(SnapshotRead::Snapshot(snapshot)) => snapshot_to_object_state(&snapshot.state),
        Ok(SnapshotRead::TooLarge(t)) => panic!("the fixture is within every ceiling: {t:?}"),
        Err(e) => panic!("the snapshot read must succeed: {e:?}"),
    }
}

/// `SECTION`'s time 0 is exactly one stored row — at `+0.0` — with one
/// `obj:tss` member, holding inline `value`, and the readers serve exactly
/// that row; no `-0.0` key or member is left in the object.
async fn assert_one_canonical_row(tv: &TestValkey, oid: u64, value: &[u8], context: &str) {
    assert_no_negative_zero_key_or_member(tv, oid, context).await;
    let stored = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("the reference read must succeed");
    let rows: Vec<_> = stored
        .time_samples
        .iter()
        .filter(|t| t.section_id == SECTION && t.time.into_inner() == 0.0)
        .collect();
    assert_eq!(rows.len(), 1, "{context}: exactly one stored row at time 0");
    let row = rows[0];
    assert_eq!(
        row.time.into_inner().to_bits(),
        0,
        "{context}: stored at +0.0"
    );
    assert_eq!(
        row.value,
        FieldValue::Inline(value.to_vec()),
        "{context}: the surviving value"
    );
    let zero_member = KeySchema::time_sample_member(&SECTION.to_string(), 0.0);
    assert_eq!(
        tss_members(tv, oid)
            .await
            .iter()
            .filter(|m| m.starts_with(&format!("{SECTION}:")) && **m == zero_member)
            .count(),
        1,
        "{context}: exactly one obj:tss member at time 0"
    );
    let served = served(tv, oid).await;
    let served_row = served
        .time_samples
        .get(&(SECTION, ordered_float::OrderedFloat(0.0)))
        .unwrap_or_else(|| panic!("{context}: the readers serve the row"));
    assert_eq!(
        (
            served_row.time.into_inner().to_bits(),
            &served_row.value,
            served_row.version
        ),
        (0, &row.value, row.version),
        "{context}: the readers serve exactly the stored row"
    );
}

/// Run `samples` on every update path over a fresh layer (with a stored
/// `0.0` row of `stored_zero` when given), then assert `survivor` is the one
/// canonical row.
async fn update_case(
    name: &str,
    stored_zero: Option<&'static [u8]>,
    samples: &[Sample],
    survivor: &[u8],
) {
    for path in UPDATE_PATHS {
        let layer = Layer::new(path, &format!("{name}-{path:?}"), stored_zero).await;
        layer.update(samples).await;
        assert_one_canonical_row(&layer.tv, layer.oid, survivor, &layer.context()).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_one_negative_zero_write_stores_one_row_at_positive_zero() {
    update_case("one-neg", None, &[at_negative_zero(b"a")], b"a").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_both_signs_negative_first_keeps_the_last_in_one_row() {
    update_case(
        "both-neg-first",
        None,
        &[at_negative_zero(b"a"), at_zero(b"b")],
        b"b",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_both_signs_positive_first_keeps_the_last_in_one_row() {
    update_case(
        "both-pos-first",
        None,
        &[at_zero(b"a"), at_negative_zero(b"b")],
        b"b",
    )
    .await;
}

/// ovlive's whole-map `timeSamples` replace that changes the time-0 key's
/// sign: an erase at the old bits, then a set at the new bits, in one
/// update, over the stored row. The set wins in both sign orders: before
/// canonicalization the erase row and the set row were two stored rows of
/// one version, and the erase could win the readers' tie.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_whole_map_replace_keeps_the_set_in_both_sign_orders() {
    update_case(
        "replace-to-neg",
        Some(b"old"),
        &[at_zero(b""), at_negative_zero(b"new")],
        b"new",
    )
    .await;
    update_case(
        "replace-to-pos",
        Some(b"old"),
        &[at_negative_zero(b""), at_zero(b"new")],
        b"new",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_negative_zero_set_on_a_stored_zero_row_rewrites_that_row() {
    update_case(
        "set-neg-on-pos",
        Some(b"old"),
        &[at_negative_zero(b"new")],
        b"new",
    )
    .await;
}

/// An erase at `0.0` of a sample a build older than b.8jc stored at `-0.0`
/// (planted: its row and member, at version 1). The store then holds the
/// legacy row and the newer erase row; the readers serve the erase at once
/// (rule B: the higher version wins). One reconciler merge later the store
/// holds one row, at `+0.0`, empty, and the readers serve the same.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_erase_at_zero_of_a_legacy_negative_zero_row_is_served_at_once_and_merged() {
    for path in UPDATE_PATHS {
        let layer = Layer::new(path, &format!("erase-legacy-{path:?}"), None).await;
        let (tv, oid) = (&layer.tv, layer.oid);
        let context = layer.context();
        let (keys, oid_str, sid) = (tv.key_schema(), oid.to_string(), SECTION.to_string());
        let _: () = tv
            .client()
            .pool
            .hset(
                keys.obj_time_sample(&oid_str, &sid, -0.0),
                [("value", "b2xk"), ("version", "1")],
            )
            .await
            .expect("plant the legacy row");
        let _: i64 = tv
            .client()
            .pool
            .zadd(
                keys.obj_time_samples_index(&oid_str),
                None,
                None,
                false,
                false,
                (1.0, KeySchema::time_sample_member(&sid, -0.0)),
            )
            .await
            .expect("plant its member");

        layer.update(&[at_zero(b"")]).await;

        let served_now = served(tv, oid).await;
        assert_eq!(
            served_now
                .time_samples
                .get(&(SECTION, ordered_float::OrderedFloat(0.0)))
                .map(|t| (&t.value, t.time.into_inner().to_bits())),
            Some((&FieldValue::Inline(Vec::new()), 0)),
            "{context}: the erase is served at once, at +0.0"
        );
        let merged = merge_signed_zero_rows(
            tv.client(),
            keys,
            &oid_str,
            &negative_zero_sections(&tss_members(tv, oid).await),
            None,
        )
        .await;
        assert_eq!(merged.merged, 1, "{context}: the merge kept the erase");
        assert_one_canonical_row(tv, oid, b"", &context).await;
        assert_eq!(
            served(tv, oid).await,
            served_now,
            "{context}: the merge changed nothing the readers serve"
        );
    }
}

/// `CreateLayer` canonicalizes too: a layer created with a `-0.0` sample, or
/// both signs of one, stores one row at `+0.0`, the last one in the content.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_with_negative_zero_samples_stores_one_row_at_positive_zero() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("create-neg", "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    let section = Some(SECTION);
    for (name, samples, survivor) in [
        ("create-one-neg", vec![at_negative_zero(b"a")], &b"a"[..]),
        (
            "create-neg-first",
            vec![at_negative_zero(b"a"), at_zero(b"b")],
            &b"b"[..],
        ),
        (
            "create-pos-first",
            vec![at_zero(b"a"), at_negative_zero(b"b")],
            &b"b"[..],
        ),
    ] {
        let oid = create(&svc, &session_id, name, ompf_of(&samples, section))
            .await
            .unwrap_or_else(|e| panic!("{name}: CreateLayer must succeed; got {e:?}"));
        assert_one_canonical_row(&tv, oid, survivor, name).await;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// The relay (PD1)
// ═════════════════════════════════════════════════════════════════════════════

fn subscribe_request(
    session_id: &str,
    layer_name: &str,
    client_id: &str,
) -> Request<SubscribeReadLayerRequest> {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
        object_id: 0,
        sequence_num: 0,
        client_id: client_id.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

/// The next header and its whole body.
async fn read_message<S>(stream: &mut S, what: &str) -> (SubscribeReadLayerHeader, Vec<u8>)
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    let first = stream
        .next()
        .await
        .unwrap_or_else(|| panic!("{what}: the stream ended"))
        .unwrap_or_else(|s| panic!("{what}: the stream failed: {s:?}"));
    let header = match first.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => h,
        other => panic!("{what}: expected a header, got {other:?}"),
    };
    let mut body = Vec::new();
    while (body.len() as u64) < header.total_size {
        let next = stream
            .next()
            .await
            .unwrap_or_else(|| panic!("{what}: the stream ended mid-body"))
            .unwrap_or_else(|s| panic!("{what}: the stream failed mid-body: {s:?}"));
        match next.response_type {
            Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                body.extend_from_slice(&c.data)
            }
            other => panic!("{what}: expected a body chunk, got {other:?}"),
        }
    }
    (header, body)
}

/// The catch-up's content bodies, read through its caught-up ack (the one
/// `total_size == 0` header), all under one [`SETTLE`] deadline.
async fn read_catch_up<S>(stream: &mut S, what: &str) -> Vec<Vec<u8>>
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    tokio::time::timeout(SETTLE, async {
        let mut bodies = Vec::new();
        loop {
            let (header, body) = read_message(stream, what).await;
            if header.total_size == 0 {
                return bodies;
            }
            bodies.push(body);
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{what}: the catch-up did not end within {SETTLE:?}"))
}

/// A live subscriber receives a writer's update byte for byte — the `-0.0`
/// time included — because the relay forwards the writer's OMPF verbatim;
/// a late joiner's catch-up, built from the stored state, carries no `-0.0`
/// and decodes the sample at time `0.0` with the writer's value.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn relay_forwards_the_writers_negative_zero_while_a_late_joiner_gets_zero() {
    let layer = Layer::new(Path::Fast, "relay-neg", None).await;
    let mut live = handle_subscribe_read_layer(
        &layer.svc,
        subscribe_request(&layer.session_id, &layer.name, "live-subscriber"),
    )
    .await
    .expect("the live subscription opens")
    .into_inner();
    read_catch_up(&mut live, "live subscriber's catch-up").await;

    let sent = layer.update(&[at_negative_zero(b"relayed")]).await;
    let (header, relayed) = tokio::time::timeout(SETTLE, read_message(&mut live, "the relay"))
        .await
        .expect("the relayed update arrives");

    assert_eq!(header.object_id, layer.oid);
    assert_eq!(
        relayed, sent,
        "the live subscriber receives the writer's OMPF byte for byte"
    );
    assert!(
        carries_negative_zero(&relayed),
        "the relayed bytes keep the writer's -0.0"
    );

    let mut late = handle_subscribe_read_layer(
        &layer.svc,
        subscribe_request(&layer.session_id, &layer.name, "late-joiner"),
    )
    .await
    .expect("the late subscription opens")
    .into_inner();
    let bodies = read_catch_up(&mut late, "late joiner's catch-up").await;
    let content: Vec<u8> = bodies.concat();
    let container = OmpfContainer::parse(&content).expect("the catch-up is one OMPF");
    assert!(
        !carries_negative_zero(container.delta_bytes),
        "the late joiner's delta carries no -0.0"
    );
    let delta = oodf_ffi::parse_delta(container.delta_bytes).expect("the catch-up delta parses");
    let samples: HashMap<u64, (u64, Vec<u8>)> = delta
        .time_samples
        .iter()
        .map(|t| {
            (
                t.section_id,
                (t.time.to_bits(), t.value_or_ext_hash.clone()),
            )
        })
        .collect();
    assert_eq!(
        samples.get(&SECTION),
        Some(&(0, b"relayed".to_vec())),
        "the late joiner gets the sample at time 0.0"
    );
}
