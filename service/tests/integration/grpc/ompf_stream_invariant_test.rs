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

// SR-1.2 — the ompf-on-every-non-empty-commit classification invariant.
//
// **The invariant: every non-empty update commit XADDs the `ompf` field.**
//
// Checkpoint tail capture classifies an `update`-typed `{seq}-0` stream entry
// with NO `ompf` field as a §8.4 *empty* commit — by invariant, not by byte
// shape. The two shapes are byte-identical on the wire (see `RELAY_OMPF_FIELD`
// in `live_edit::storage::notifications`), so the absence of the field is the
// entire signal. If a state-changing commit ever reached the stream without its
// relay OMPF, tail capture would record it as an empty delta and the restore
// would replay a no-op over a real state change, minting a generation whose
// claimed coverage exceeds its actual content — the b.pw6 defect class this
// subsystem exists to forbid. The relay consumer reads the absence the same
// way: a `SubscribeReadLayer` subscriber sends an ompf-less entry above
// sequence 1 as the empty delta, with no read, so such a commit would also
// reach every live subscriber without its changes. That is why the property is
// load-bearing and gets its own regression module rather than riding along on
// the b.gze relay tests: the relay consumer and the tail-capture classifier are
// two independent reasons the field must be present, and a change made for
// either one can break the other.
//
// Everything here drives the WRITE side through `handle_update_layer` — the gRPC
// surface — rather than through `LayerStore::update_layer` directly. That is
// deliberate: the internal storage API can represent the forbidden
// `Some(state-changing delta)` / `None` relay pair (dozens of existing test
// call sites do exactly that), but production cannot, because
// `handle_update_layer` derives both values from one `ompf_bytes.is_empty()`
// test. Testing the storage API would be testing a shape production cannot
// produce. The enforcement that guards the derivation staying coupled is
// `src/grpc/layer.rs::debug_assert_relay_invariant`, whose two tiers are:
//
//   * the forged-violation unit in that file's `#[cfg(test)] mod tests`
//     (proves the predicate is armed and correctly oriented), and
//   * `grpc_layer_source_wires_relay_invariant_assert_at_derivation_point`
//     below (proves the call is still wired in — the release-build protection,
//     since `debug_assert!` compiles out of release).
//
// Backend coverage. The commit paths compose the XADD independently: the slow
// path queues it inside its MULTI (`src/storage/layer.rs`, "SR-1.2
// CLASSIFICATION INVARIANT (slow path)"), and the fast and structural routes
// pass the bytes as ARGV[7] to `update_layer_fast.lua`, which branches on
// `relay_ompf ~= ""` at its XADD sites. Every positive case therefore runs under
// both `UpdateLayerBackend::Lua` and `::Txn`, selected via the `LayerStore`
// builder — never via `LIVE_EDIT_UPDATE_LAYER_BACKEND`, which is process-global
// env state and would contaminate sibling tests. The structural-command case
// runs under all three backends, because each reaches a different composition
// site: under `lua` a `CreateSection` (not fast-pathable,
// `storage::lua::is_fast_path`) takes the cached structural route, whose commit
// is the script's structural mode; under `lua-watch-structural` and `txn` it
// takes the slow path's MULTI.
//
// Readback decodes stream field VALUES as `Bytes`, never `String`: the binary
// `ompf` field is not valid UTF-8, and a `String` decode fails the WHOLE row —
// dropping the entry, payload and all, so an assertion written against a
// `String` readback would report "no entries" rather than the real shape.
// Precedent: `grpc/subscribe_regimes_test.rs`; the trap is documented at
// `storage/stream_consumer_test.rs::consumer_delivers_an_entry_whose_ompf_field_is_not_valid_utf8`.

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use fred::interfaces::StreamsInterface;
use tonic::Request;

use live_edit::config::UpdateLayerBackend;
use live_edit::grpc::layer::{handle_create_layer, handle_delete_layer, handle_update_layer};
use live_edit::grpc::proto::DeleteLayerRequest;
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::notifications::{
    DELETED_STREAM_PAYLOAD, NOTIFICATION_PAYLOAD_FIELD, RELAY_OMPF_FIELD,
};
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand, TimeSample};

use crate::common::owned_delta::delta_to_ompf_bytes;

use super::helpers::*;

/// Stream `MAXLEN ~` bound for the per-test `LayerStore`. Matches
/// `helpers::build_service`; every test here writes a handful of entries, well
/// under the bound, so no trim can hide an entry from the sweep.
const STREAM_MAXLEN: u64 = 4096;

/// The CreateLayer seed entry's fixed stream ID. It is the ONE `update`-typed
/// entry that carries content while having no `ompf` field (SR-1.2 seed
/// carve-out, `src/storage/layer.rs`), so every sweep over a whole stream has to
/// exclude it explicitly — and prove it is there while doing so, otherwise the
/// exclusion could silently swallow the entire stream.
const SEED_ENTRY_ID: &str = "1-0";

// ── Fixtures ─────────────────────────────────────────────────────────────────

/// One stream entry as read back: `(entry_id, field -> raw bytes)`.
type StreamEntry = (String, HashMap<String, Bytes>);

/// Build a service whose `LayerStore` commits through `backend`.
///
/// The backend is selected on the store builder rather than through
/// `LIVE_EDIT_UPDATE_LAYER_BACKEND` so parallel tests in this binary cannot
/// contaminate each other's process-global env.
///
/// The `RecoveryHook` that `build_service` wired still points at the original
/// store, which is harmless: `build_disabled_recovery_hook` sets
/// `checkpoint_enabled = false`, so `try_restore_layer` short-circuits to
/// `NotRecoverable` without touching any store at all.
fn service_with_backend(tv: &TestValkey, backend: UpdateLayerBackend) -> LiveEditService {
    let mut svc = build_service(tv, Arc::new(AllowAllPermissions));
    svc.layer_store = Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), STREAM_MAXLEN)
            .with_update_layer_backend(backend),
    );
    svc
}

/// Short label for assertion messages so a failure names the backend arm.
fn backend_label(backend: UpdateLayerBackend) -> &'static str {
    match backend {
        UpdateLayerBackend::Lua => "lua",
        UpdateLayerBackend::LuaWatchStructural => "lua-watch-structural",
        UpdateLayerBackend::Txn => "txn",
    }
}

fn delete_request(session_id: &str, layer_name: &str) -> Request<DeleteLayerRequest> {
    let mut request = Request::new(DeleteLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

/// Commit `delta` through the real `UpdateLayer` handler, returning the
/// `(sequence_num, inbound_ompf)` pair so the caller can assert byte-identity
/// against what landed on the stream.
async fn commit_delta(
    svc: &LiveEditService,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    delta: &OwnedDelta,
) -> (u64, Vec<u8>) {
    let ompf = delta_to_ompf_bytes(delta, &[]);
    let stream = streaming_update(vec![
        make_update_header(session_id, layer_name, object_id),
        make_update_data(ompf.clone()),
    ]);
    let response = handle_update_layer(svc, make_request(stream))
        .await
        .expect("UpdateLayer commit must succeed");
    (response.into_inner().sequence_num, ompf)
}

/// Commit the §8.4 *empty* shape: a header with no data chunks at all, so
/// `ompf_bytes` is empty at the handler and both `owned` and `relay_ompf`
/// derive to `None`.
async fn commit_empty(
    svc: &LiveEditService,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
) -> u64 {
    let stream = streaming_update(vec![make_update_header(session_id, layer_name, object_id)]);
    let response = handle_update_layer(svc, make_request(stream))
        .await
        .expect("empty UpdateLayer commit (header only) must succeed");
    response.into_inner().sequence_num
}

/// XRANGE the whole per-object stream, decoding field VALUES as `Bytes`.
async fn read_stream(tv: &TestValkey, object_id: u64) -> Vec<StreamEntry> {
    tv.client()
        .pool
        .xrange_values(tv.key_schema().layer_stream(object_id), "-", "+", None)
        .await
        .expect("XRANGE over the per-object stream must succeed")
}

/// Locate the entry at `{seq}-0`.
fn entry_at_seq(entries: &[StreamEntry], seq: u64) -> &StreamEntry {
    let id = format!("{seq}-0");
    entries
        .iter()
        .find(|(entry_id, _)| *entry_id == id)
        .unwrap_or_else(|| {
            panic!(
                "no stream entry at the committed sequence's explicit id `{id}`; \
                 got ids {:?}",
                entries.iter().map(|(i, _)| i).collect::<Vec<_>>()
            )
        })
}

/// The UTF-8 `payload` field, lossily decoded for assertion messages and for
/// the `update` / `deleted` type test.
fn payload_text(fields: &HashMap<String, Bytes>) -> String {
    let raw = fields
        .get(NOTIFICATION_PAYLOAD_FIELD)
        .unwrap_or_else(|| panic!("every stream entry carries `{NOTIFICATION_PAYLOAD_FIELD}`"));
    String::from_utf8_lossy(raw).into_owned()
}

/// The core positive assertion: this entry carries `ompf`, and those bytes are
/// the writer's inbound OMPF **byte for byte** — not merely present, since a
/// truncated or re-encoded payload would satisfy presence while still failing
/// both the relay consumer and any byte-level capture.
fn assert_carries_ompf_verbatim(entry: &StreamEntry, expected_ompf: &[u8], context: &str) {
    let (entry_id, fields) = entry;
    let stored = fields.get(RELAY_OMPF_FIELD).unwrap_or_else(|| {
        panic!(
            "SR-1.2 VIOLATION [{context}]: the non-empty update commit at `{entry_id}` \
             XADDed no `{RELAY_OMPF_FIELD}` field. Tail capture classifies an ompf-less \
             `update` entry as a §8.4 empty commit, so this commit would restore as a \
             no-op over a real state change (the b.pw6 defect class). Entry fields: {:?}",
            fields.keys().collect::<Vec<_>>()
        )
    });
    assert_eq!(
        stored.as_ref(),
        expected_ompf,
        "[{context}]: the `{RELAY_OMPF_FIELD}` field on `{entry_id}` must be the writer's \
         inbound OMPF byte-for-byte (stored {} bytes, inbound {} bytes)",
        stored.len(),
        expected_ompf.len()
    );
}

fn assert_is_ompf_less(entry: &StreamEntry, context: &str) {
    let (entry_id, fields) = entry;
    assert!(
        !fields.contains_key(RELAY_OMPF_FIELD),
        "[{context}]: entry `{entry_id}` must be `{RELAY_OMPF_FIELD}`-less — the classifier \
         reads this absence as the §8.4 empty-commit signal, and attaching relay bytes here \
         would change how tail capture types the entry. Fields: {:?}",
        fields.keys().collect::<Vec<_>>()
    );
}

// ── Delta shapes ─────────────────────────────────────────────────────────────

/// A field-set commit. `is_fast_path` = true → the Lua backend serves this from
/// `update_layer_fast.lua`'s ARGV[7] composition site.
fn field_set_delta(section_id: u64, key: &str, set_order: u64) -> OwnedDelta {
    OwnedDelta {
        set_fields: vec![SetField {
            section_id,
            key_name: key.to_owned(),
            value_or_ext_hash: vec![0xDE, 0xAD, 0xBE, 0xEF],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order,
        }],
        ..Default::default()
    }
}

/// A time-sample commit. Also `is_fast_path` = true, but a distinct mutation
/// encoding from a field set, so it exercises a different ARGV record shape
/// reaching the same XADD.
fn time_sample_delta(section_id: u64, time: f64) -> OwnedDelta {
    OwnedDelta {
        time_samples: vec![TimeSample {
            section_id,
            time,
            value_or_ext_hash: vec![0xAA, 0xBB, 0xCC],
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        ..Default::default()
    }
}

/// A structural commit: `CreateSection` plus one attribute on the new section.
/// `storage::lua::is_fast_path` returns false for any non-`DiffDeleteSection`
/// struct command, so this delta takes the structural route under `lua` and the
/// WATCH/MULTI/EXEC slow path under `lua-watch-structural` and `txn` — the case
/// that proves both of those XADD composition sites.
fn structural_delta(section_id: u64, section_name: &str) -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id,
            section_name: section_name.to_owned(),
            section_type: 1,
        })],
        set_fields: vec![SetField {
            section_id,
            key_name: "radius".to_owned(),
            value_or_ext_hash: vec![1, 2, 3, 4],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 4242,
        }],
        ..Default::default()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. Field-set commit → `{seq}-0` carries the inbound OMPF byte-identically
// ═════════════════════════════════════════════════════════════════════════════

async fn field_set_commit_case(backend: UpdateLayerBackend) {
    let label = backend_label(backend);
    let tv = TestValkey::setup().await;
    let svc = service_with_backend(&tv, backend);

    let session = format!("sr12-field-{label}-session");
    let layer = format!("sr12-field-{label}-layer");
    let (session_id, object_id) = create_session_and_layer(&tv, &svc, &session, &layer).await;

    let (seq, inbound) = commit_delta(
        &svc,
        &session_id,
        &layer,
        object_id,
        &field_set_delta(3, "displayColor", 7),
    )
    .await;

    let entries = read_stream(&tv, object_id).await;
    assert_carries_ompf_verbatim(
        entry_at_seq(&entries, seq),
        &inbound,
        &format!("field-set commit, {label} backend"),
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_field_set_commit_lua_backend_xadds_verbatim_ompf() {
    field_set_commit_case(UpdateLayerBackend::Lua).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_field_set_commit_txn_backend_xadds_verbatim_ompf() {
    field_set_commit_case(UpdateLayerBackend::Txn).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. Time-sample commit → same invariant
// ═════════════════════════════════════════════════════════════════════════════

async fn time_sample_commit_case(backend: UpdateLayerBackend) {
    let label = backend_label(backend);
    let tv = TestValkey::setup().await;
    let svc = service_with_backend(&tv, backend);

    let session = format!("sr12-ts-{label}-session");
    let layer = format!("sr12-ts-{label}-layer");
    let (session_id, object_id) = create_session_and_layer(&tv, &svc, &session, &layer).await;

    let (seq, inbound) = commit_delta(
        &svc,
        &session_id,
        &layer,
        object_id,
        &time_sample_delta(2, 12.5),
    )
    .await;

    let entries = read_stream(&tv, object_id).await;
    assert_carries_ompf_verbatim(
        entry_at_seq(&entries, seq),
        &inbound,
        &format!("time-sample commit, {label} backend"),
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_time_sample_commit_lua_backend_xadds_verbatim_ompf() {
    time_sample_commit_case(UpdateLayerBackend::Lua).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_time_sample_commit_txn_backend_xadds_verbatim_ompf() {
    time_sample_commit_case(UpdateLayerBackend::Txn).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. Structural commit → the structural route's and the SLOW path's sites
// ═════════════════════════════════════════════════════════════════════════════

async fn structural_commit_case(backend: UpdateLayerBackend) {
    let label = backend_label(backend);
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("sr12-struct"));
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    svc.layer_store = Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), STREAM_MAXLEN)
            .with_update_layer_backend(backend)
            .with_metrics(Arc::clone(&metrics)),
    );
    // The composition site this arm claims: the structural route (a cold
    // start, `structural_full`) under `lua`, the WATCH path otherwise.
    let path = match backend {
        UpdateLayerBackend::Lua => "structural_full",
        UpdateLayerBackend::LuaWatchStructural | UpdateLayerBackend::Txn => "watch",
    };

    let session = format!("sr12-struct-{label}-session");
    let layer = format!("sr12-struct-{label}-layer");
    let (session_id, object_id) = create_session_and_layer(&tv, &svc, &session, &layer).await;

    let (seq, inbound) = commit_delta(
        &svc,
        &session_id,
        &layer,
        object_id,
        &structural_delta(77, "MySphere"),
    )
    .await;

    let entries = read_stream(&tv, object_id).await;
    assert_carries_ompf_verbatim(
        entry_at_seq(&entries, seq),
        &inbound,
        &format!("structural CreateSection commit, {label} backend"),
    );
    assert_eq!(
        sample_value(
            &render(&metrics),
            "live_edit_update_layer_attempts_histogram_count",
            &[("path", path)],
        ),
        Some(1.0),
        "fixture: under the {label} backend the commit took the {path} path"
    );
}

/// The load-bearing arm: the configured backend is `lua` and a `CreateSection`
/// delta is not fast-pathable, so this commit is served by the cached
/// structural route — the script's structural mode composes the XADD in a
/// default-configured production replica.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_structural_commit_under_lua_backend_takes_structural_route_and_xadds_ompf() {
    structural_commit_case(UpdateLayerBackend::Lua).await;
}

/// The rollback knob `lua-watch-structural`: the same delta is served by the
/// WATCH/MULTI/EXEC site, so the slow path's XADD composition is reachable
/// beside a fast path that stays on.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_structural_commit_lua_watch_structural_backend_takes_slow_path_and_xadds_ompf(
) {
    structural_commit_case(UpdateLayerBackend::LuaWatchStructural).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_structural_commit_txn_backend_xadds_verbatim_ompf() {
    structural_commit_case(UpdateLayerBackend::Txn).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. Multi-commit sweep — the property is about a RANGE, not one entry
// ═════════════════════════════════════════════════════════════════════════════

/// Tail capture reads a *slice* of the stream, so the property that matters is
/// "every non-empty update entry in the range carries `ompf`", not "the last one
/// did". A single-entry assertion would stay green against a regression that
/// dropped the field on, say, only the structural path or only after the first
/// commit on an object.
///
/// Four commits of mixed shape (field set → time sample → structural → field
/// set), then ONE XRANGE over the whole stream, checked entry by entry against
/// the exact inbound bytes recorded per sequence.
async fn multi_commit_sweep_case(backend: UpdateLayerBackend) {
    let label = backend_label(backend);
    let tv = TestValkey::setup().await;
    let svc = service_with_backend(&tv, backend);

    let session = format!("sr12-sweep-{label}-session");
    let layer = format!("sr12-sweep-{label}-layer");
    let (session_id, object_id) = create_session_and_layer(&tv, &svc, &session, &layer).await;

    let shapes: Vec<OwnedDelta> = vec![
        field_set_delta(3, "displayColor", 1),
        time_sample_delta(3, 4.5),
        structural_delta(88, "SweptSphere"),
        field_set_delta(88, "radius", 9),
    ];
    assert!(
        shapes.len() >= 4,
        "the sweep needs N >= 4 commits to be a range assertion"
    );

    let mut expected: HashMap<u64, Vec<u8>> = HashMap::new();
    for delta in &shapes {
        let (seq, inbound) = commit_delta(&svc, &session_id, &layer, object_id, delta).await;
        assert!(
            expected.insert(seq, inbound).is_none(),
            "each commit must land on its own sequence number; seq {seq} was reused"
        );
    }

    let entries = read_stream(&tv, object_id).await;

    // The seed carve-out must actually be present, so the skip below cannot
    // vacuously swallow a stream that lost its update entries.
    let seed = entries
        .iter()
        .find(|(id, _)| id == SEED_ENTRY_ID)
        .unwrap_or_else(|| panic!("the CreateLayer seed entry `{SEED_ENTRY_ID}` must be present"));
    assert_is_ompf_less(seed, "CreateLayer seed entry inside the sweep range");

    let mut checked = 0usize;
    for entry in &entries {
        let (entry_id, fields) = entry;
        if entry_id == SEED_ENTRY_ID {
            continue;
        }
        let payload = payload_text(fields);
        assert!(
            payload.starts_with("update|"),
            "unexpected non-update entry `{entry_id}` on the sweep stream: {payload}"
        );
        let seq: u64 = entry_id
            .split('-')
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("stream id `{entry_id}` must start with a numeric sequence"));
        assert!(
            entry_id.ends_with("-0"),
            "every update commit XADDs at the explicit `{{seq}}-0` id; got `{entry_id}`"
        );
        let inbound = expected
            .get(&seq)
            .unwrap_or_else(|| panic!("stream carries an entry at seq {seq} nobody committed"));
        assert_carries_ompf_verbatim(
            entry,
            inbound,
            &format!("multi-commit sweep, {label} backend"),
        );
        checked += 1;
    }

    assert_eq!(
        checked,
        expected.len(),
        "the sweep must have inspected every committed sequence — {} committed, {checked} found \
         on the stream (a missing entry means the sweep proved nothing about it)",
        expected.len()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_multi_commit_sweep_lua_backend_every_update_entry_carries_ompf() {
    multi_commit_sweep_case(UpdateLayerBackend::Lua).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_multi_commit_sweep_txn_backend_every_update_entry_carries_ompf() {
    multi_commit_sweep_case(UpdateLayerBackend::Txn).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. Empty commit → the ompf-LESS shape the classifier depends on
// ═════════════════════════════════════════════════════════════════════════════

/// The invariant is one-directional, and this is its other half: an *empty*
/// commit must produce an ompf-LESS `{seq}-0` entry. Without this case the
/// invariant could be "satisfied" by attaching `ompf` unconditionally, which
/// would destroy the classifier's only signal for §8.4 empty commits.
async fn empty_commit_case(backend: UpdateLayerBackend) {
    let label = backend_label(backend);
    let tv = TestValkey::setup().await;
    let svc = service_with_backend(&tv, backend);

    let session = format!("sr12-empty-{label}-session");
    let layer = format!("sr12-empty-{label}-layer");
    let (session_id, object_id) = create_session_and_layer(&tv, &svc, &session, &layer).await;

    let seq = commit_empty(&svc, &session_id, &layer, object_id).await;

    let entries = read_stream(&tv, object_id).await;
    let entry = entry_at_seq(&entries, seq);
    assert!(
        payload_text(&entry.1).starts_with("update|"),
        "the empty commit still XADDs an `update`-typed entry — the classifier's whole job is \
         telling it apart from a non-empty one by the `{RELAY_OMPF_FIELD}` field alone"
    );
    assert_is_ompf_less(entry, &format!("empty commit, {label} backend"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_empty_commit_lua_backend_xadds_ompf_less_entry() {
    empty_commit_case(UpdateLayerBackend::Lua).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_empty_commit_txn_backend_xadds_ompf_less_entry() {
    empty_commit_case(UpdateLayerBackend::Txn).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 6. Negative control — CreateLayer's seed entry at `1-0` is ompf-less
// ═════════════════════════════════════════════════════════════════════════════

/// The invariant is SCOPED to update commits, and this pins the boundary: the
/// enforcement must never be over-tightened into firing on layer creation.
/// CreateLayer materializes its initial content directly into the object keys
/// and relays nothing, so there is nothing to attach; the entry is safe only
/// because tail capture's window starts at `{S0+1}-0` with `S0 >= 1`
/// (SR-7.1 / SR-7.3), which puts `1-0` structurally out of range.
///
/// The checkpoint-side sibling
/// (`checkpoint/first_base_bootstrap_test.rs`) asserts the same shape from the
/// first-base bootstrap's fence. This copy exists because the carve-out is a
/// premise of the SR-1.2 sweep directly above it: if `1-0` ever gained an `ompf`
/// field, that sweep's skip would be silently wrong here, not there.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_seed_entry_is_ompf_less() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (_session_id, object_id) =
        create_session_and_layer(&tv, &svc, "sr12-seed-session", "sr12-seed-layer").await;

    let entries = read_stream(&tv, object_id).await;
    assert_eq!(
        entries.len(),
        1,
        "a freshly created layer's stream holds exactly the seed entry; got {:?}",
        entries.iter().map(|(i, _)| i).collect::<Vec<_>>()
    );
    let entry = &entries[0];
    assert_eq!(
        entry.0, SEED_ENTRY_ID,
        "the seed entry is written at the fixed id `{SEED_ENTRY_ID}` inside CreateLayer's \
         MULTI/EXEC"
    );
    assert_is_ompf_less(entry, "CreateLayer seed entry (SR-1.2 seed carve-out)");
}

// ═════════════════════════════════════════════════════════════════════════════
// 7. Negative control — the `deleted` sentinel is ompf-less
// ═════════════════════════════════════════════════════════════════════════════

/// The other scoping boundary. The retirement sentinel is written with an
/// auto-ID (`*`), so it is not one of the versioned `{seq}-0` commit entries at
/// all: it carries only the `payload` field, and nothing about it should ever
/// acquire relay bytes. Read inside the post-delete PEXPIRE grace window, which
/// is why the stream is still readable here.
///
/// Note the sentinel's ID is NOT distinguishable by shape — Valkey mints `*` as
/// `<ms>-0`, which also ends in `-0`. What separates it from a commit entry is
/// that its sequence part is a millisecond timestamp nobody committed, so the
/// assertion below is written as "not equal to any id we produced", not as a
/// pattern match on the id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_deleted_sentinel_is_ompf_less() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "sr12-sentinel-session", "sr12-sentinel-layer").await;

    // One real commit first, so the stream holds an ompf-carrying entry and the
    // sentinel assertion cannot pass merely because nothing ever attaches ompf
    // on this object.
    let (seq, inbound) = commit_delta(
        &svc,
        &session_id,
        "sr12-sentinel-layer",
        object_id,
        &field_set_delta(3, "displayColor", 1),
    )
    .await;

    handle_delete_layer(&svc, delete_request(&session_id, "sr12-sentinel-layer"))
        .await
        .expect("DeleteLayer must succeed");

    let entries = read_stream(&tv, object_id).await;
    assert_carries_ompf_verbatim(
        entry_at_seq(&entries, seq),
        &inbound,
        "pre-delete commit (control for the sentinel assertion)",
    );

    let sentinels: Vec<&StreamEntry> = entries
        .iter()
        .filter(|(_, fields)| payload_text(fields) == DELETED_STREAM_PAYLOAD)
        .collect();
    assert!(
        !sentinels.is_empty(),
        "DeleteLayer must append a `{DELETED_STREAM_PAYLOAD}` sentinel within the grace window; \
         got ids {:?}",
        entries.iter().map(|(i, _)| i).collect::<Vec<_>>()
    );
    let commit_id = format!("{seq}-0");
    for sentinel in sentinels {
        assert_ne!(
            sentinel.0, commit_id,
            "the retirement sentinel must be its own auto-ID entry, never an overwrite of the \
             commit entry at `{commit_id}`"
        );
        assert_ne!(
            sentinel.0, SEED_ENTRY_ID,
            "the retirement sentinel must not land on the CreateLayer seed id"
        );
        assert_eq!(
            sentinel.1.len(),
            1,
            "the retirement sentinel carries ONLY the `{NOTIFICATION_PAYLOAD_FIELD}` field; \
             got {:?}",
            sentinel.1.keys().collect::<Vec<_>>()
        );
        assert_is_ompf_less(sentinel, "DeleteLayer `deleted` sentinel");
    }
}

/// The sentinel's SECOND producer. `DeleteLayer` is not the only site that
/// retires an object: a `CreateLayer` against an existing layer name overwrites
/// it, appending the same `deleted` sentinel to the OLD object's stream and
/// seeding a fresh `1-0` on the NEW object's. Both are ompf-less carve-outs, and
/// both are written by a different code path than the one the test above
/// covers — so a change that started attaching relay bytes to retirement or seed
/// entries could land here while the DeleteLayer case stayed green.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_overwrite_sentinel_and_reseed_are_both_ompf_less() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let layer = "sr12-overwrite-layer";
    let (session_id, old_object_id) =
        create_session_and_layer(&tv, &svc, "sr12-overwrite-session", layer).await;

    // A real commit on the old object, so its stream carries an ompf-bearing
    // entry and the carve-out assertions below are not vacuous.
    let (seq, inbound) = commit_delta(
        &svc,
        &session_id,
        layer,
        old_object_id,
        &field_set_delta(3, "displayColor", 1),
    )
    .await;

    // Overwrite: CreateLayer against the same session + layer name.
    let overwrite = streaming_create(vec![
        make_create_header(&session_id, layer),
        make_create_data(delta_to_ompf_bytes(&structural_delta(5, "Reseeded"), &[])),
    ]);
    let new_object_id = handle_create_layer(&svc, make_request(overwrite))
        .await
        .expect("CreateLayer overwrite must succeed")
        .into_inner()
        .object_id;
    assert_ne!(
        new_object_id, old_object_id,
        "an overwrite mints a new generation (object_id), which is what makes the old \
         object's stream a retired one"
    );

    let old_entries = read_stream(&tv, old_object_id).await;
    assert_carries_ompf_verbatim(
        entry_at_seq(&old_entries, seq),
        &inbound,
        "pre-overwrite commit (control for the overwrite carve-outs)",
    );
    let old_sentinels: Vec<&StreamEntry> = old_entries
        .iter()
        .filter(|(_, fields)| payload_text(fields) == DELETED_STREAM_PAYLOAD)
        .collect();
    assert!(
        !old_sentinels.is_empty(),
        "the overwrite must retire the old object with a `{DELETED_STREAM_PAYLOAD}` sentinel so \
         subscribers re-bootstrap on the new object_id; got ids {:?}",
        old_entries.iter().map(|(i, _)| i).collect::<Vec<_>>()
    );
    for sentinel in old_sentinels {
        assert_is_ompf_less(sentinel, "CreateLayer-overwrite `deleted` sentinel");
    }

    let new_entries = read_stream(&tv, new_object_id).await;
    assert_eq!(
        new_entries.len(),
        1,
        "the overwritten-into object's stream holds exactly its seed entry; got {:?}",
        new_entries.iter().map(|(i, _)| i).collect::<Vec<_>>()
    );
    assert_eq!(
        new_entries[0].0, SEED_ENTRY_ID,
        "the reseed is written at the fixed id `{SEED_ENTRY_ID}`, same as a fresh CreateLayer"
    );
    assert_is_ompf_less(
        &new_entries[0],
        "CreateLayer-overwrite reseed entry (SR-1.2 seed carve-out)",
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 9. Wiring guard — the derivation-point assert must still be called
// ═════════════════════════════════════════════════════════════════════════════
//
// (Case 8, the forged violation, is an in-crate unit in
// `src/grpc/layer.rs`'s `#[cfg(test)] mod tests` — it has to call the helper
// directly, because the forbidden pair is unreachable through the handler.)

/// `src/grpc/layer.rs`, baked into this binary at compile time so the scan has
/// no runtime I/O and cannot read a stale working copy.
const GRPC_LAYER_SOURCE: &str = include_str!("../../../src/grpc/layer.rs");

/// The derivation of the relay payload — the first of the two values the
/// invariant relates.
const DERIVATION_ANCHOR: &str = "let relay_ompf: Option<Bytes> = if ompf_bytes.is_empty() {";

/// The production call. Spelled with its exact argument list so the in-crate
/// forged-violation unit's calls (different arguments) cannot satisfy the scan.
const ASSERT_CALL: &str = "debug_assert_relay_invariant(&owned, &relay_ompf);";

/// The commit call the two values are moved into. The handler commits through
/// `LayerStore::update_layer_with_summary`, which `update_layer_with_operation`
/// (b.v6d) and its no-operation wrapper `update_layer` wrap for storage-level
/// callers; neither wrapper is called here.
const COMMIT_CALL: &str = ".update_layer_with_summary(";

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// Release builds compile `debug_assert!` out, so the forged-violation unit in
/// `src/grpc/layer.rs` proves nothing there. This scan is the protection that
/// survives: it fails if the assert call is deleted from between the relay
/// derivation and the first `update_layer_with_summary` call — the refactor
/// shape that would let the two derivations drift apart unnoticed. Precedent
/// for the `include_str!` source-scan idiom: `storage/mapper_routing_test.rs`.
#[test]
fn grpc_layer_source_wires_relay_invariant_assert_at_derivation_point() {
    let derivation = GRPC_LAYER_SOURCE
        .find(DERIVATION_ANCHOR)
        .unwrap_or_else(|| {
            panic!(
                "could not locate the relay-OMPF derivation in src/grpc/layer.rs. Anchor: \
             `{DERIVATION_ANCHOR}`. If the derivation was legitimately reworded, update this \
             anchor AND re-verify that the SR-1.2 assert still sits between the derivation and \
             the commit call — do not delete the guard."
            )
        });

    let assert_idx = GRPC_LAYER_SOURCE.find(ASSERT_CALL).unwrap_or_else(|| {
        panic!(
            "SR-1.2 WIRING REGRESSION: `{ASSERT_CALL}` is gone from src/grpc/layer.rs. The \
             ompf-on-every-non-empty-commit invariant is enforced at exactly one production \
             site — the gRPC derivation point — because `handle_update_layer` is the only \
             production caller of `LayerStore::update_layer_with_summary` (which \
             `LayerStore::update_layer_with_operation` and `LayerStore::update_layer` \
             wrap). Removing the call leaves the \
             invariant with no write-side enforcement at all; an ompf-less state-changing \
             commit would then be captured as an empty delta and replayed as a no-op (the \
             b.pw6 defect class)."
        )
    });

    assert_eq!(
        GRPC_LAYER_SOURCE.matches(ASSERT_CALL).count(),
        1,
        "the SR-1.2 assert call must appear exactly once in src/grpc/layer.rs — a second \
         occurrence (e.g. quoted inside a comment) would let the ordering checks below be \
         satisfied by text that enforces nothing"
    );

    let commit_idx = GRPC_LAYER_SOURCE[derivation..]
        .find(COMMIT_CALL)
        .map(|offset| derivation + offset)
        .unwrap_or_else(|| {
            panic!(
                "could not locate the first `{COMMIT_CALL}` call after the relay derivation in \
                 src/grpc/layer.rs — the handler's commit call must have moved; re-anchor this \
                 guard rather than deleting it."
            )
        });

    assert!(
        derivation < assert_idx,
        "the SR-1.2 assert must sit AFTER the relay-OMPF derivation (it reads `relay_ompf`); \
         found the call at byte {assert_idx}, derivation at byte {derivation}"
    );
    assert!(
        assert_idx < commit_idx,
        "the SR-1.2 assert must sit BEFORE the first `{COMMIT_CALL}` call — after it, both \
         values have been moved and a decoupled derivation would already have written the \
         ompf-less entry the assert exists to forbid. Found the assert at byte {assert_idx}, \
         the commit call at byte {commit_idx}"
    );

    // The helper the call resolves to must still hold an armed `debug_assert!`
    // labelled "relay-OMPF invariant" — a call left in place against a
    // hollowed-out helper would satisfy the ordering checks above while
    // enforcing nothing.
    let helper_idx = GRPC_LAYER_SOURCE
        .find("fn debug_assert_relay_invariant(")
        .expect("the SR-1.2 helper fn must still be defined in src/grpc/layer.rs");
    let helper_body = &GRPC_LAYER_SOURCE[helper_idx..];
    let helper_end = helper_body
        .find("\n}\n")
        .expect("the SR-1.2 helper fn must have a closing brace");
    let helper_body = &helper_body[..helper_end];
    assert!(
        helper_body.contains("debug_assert!("),
        "the SR-1.2 helper must still contain a `debug_assert!` — got body:\n{helper_body}"
    );
    assert!(
        helper_body.contains("relay-OMPF invariant"),
        "the SR-1.2 helper's panic message must still name the \"relay-OMPF invariant\"; the \
         forged-violation unit in src/grpc/layer.rs matches on that string via \
         `#[should_panic(expected = \"relay-OMPF invariant\")]`, so a reworded message \
         silently turns that test into a no-op"
    );
}
