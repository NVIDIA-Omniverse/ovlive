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

// Wire-compatibility regression test for Bee `b.duy`.
//
// Drives the full gRPC pipeline (CreateLayer → StatLayer → UpdateLayer →
// SubscribeReadLayer) with the exact OMPF chunk captured from the legacy
// C++ live-edit service's `SubscribeReadLayer` response. Before Bee `b.duy`
// this chunk failed to parse with the Rust FlatBuffers verifier — the whole
// Epic exists to make this end-to-end test pass.
//
// Source fixture: `src/domain/ompf.rs::B64_CHUNK` (296-byte OMPF wrapping a
// 280-byte OODF payload carrying 2 `CommandDiffChangeSection` entries, plus
// a `SetField` for the second section). Decoded at test-start time via the
// base64 crate. Coordinated with the 280-byte OODF-only fixture at
// `oodf-ffi/tests/fixtures/postman_chunk.bin`.

use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use futures::StreamExt;
use oodf_ffi::StructCommand;
use tonic::Request;

use live_edit::domain::ompf::OmpfContainer;
use live_edit::grpc::layer::{handle_create_layer, handle_stat_layer, handle_update_layer};
use live_edit::grpc::proto::subscribe_read_layer_response;
use live_edit::grpc::proto::{StatLayerRequest, SubscribeReadLayerRequest};
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta};

use crate::common::owned_delta::delta_to_ompf_bytes;

use super::helpers::*;

// ── Captured Postman OMPF chunk ──────────────────────────────────────────────

/// The base64-encoded 296-byte OMPF chunk captured from the C++ live-edit
/// service. Pinned here so the integration test and the lower-level ompf.rs
/// unit test share the same canonical value.
const POSTMAN_OMPF_B64: &str = "T01QRgEAAAAQAAAAAAAAABwAAABPT0RGAAAAABAAGAAHAAAACAAMABAAFAAQAAAAAAAAASgAAAAYAAAADAAAAAQAAAAAAAAAAQAAACAAAAACAAAAvAAAAIQAAAACAAAABQUKABQADAAEAAgACgAAABAAAAAYAAAABQAAAAAAAAAGAAAAZmllbGQwAABAAAAARUlYUURSS0tNRFBWUFlPRFpLRE9HTU1ZREZJUURRUVVHTUtGQVhMRFlLRFhDRUJKR0RVTEpMVllRRlJYR1dNTgwAIAAMABQACAAHAAwAAAAAAAABGAAAAAEAAAAAAAAABQAAAAAAAAAAAAAABQAAAFByaW01AAoAEAAAAAgABAAKAAAADAAAAAEAAAAAAAAAAAAAAAAAAAA=";

fn postman_ompf_bytes() -> Vec<u8> {
    B64.decode(POSTMAN_OMPF_B64).expect("valid base64")
}

// ── Subscribe helper (duplicated from subscribe_test to keep this file standalone) ──

fn make_subscribe_request(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    sequence_num: u64,
    client_id: &str,
) -> Request<SubscribeReadLayerRequest> {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
        object_id,
        sequence_num,
        client_id: client_id.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

fn stat_request(session_id: &str, layer_name: &str) -> Request<StatLayerRequest> {
    let mut request = Request::new(StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

// ── 1. Raw fixture parses as OMPF with expected shape ───────────────────────
//
// Guards the fixture itself: if the B64 constant drifts, this test fails
// with a clear message before any gRPC pipeline runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn postman_fixture_parses_as_ompf_with_oodf_payload() {
    let bytes = postman_ompf_bytes();
    assert_eq!(
        bytes.len(),
        296,
        "captured Postman OMPF chunk must be exactly 296 bytes \
         (16-byte OMPF header + 280-byte OODF payload)"
    );

    let container = OmpfContainer::parse(&bytes).expect("Postman chunk must parse as OMPF");
    // 1 part (the Delta), no external values.
    assert!(
        container.external_values.is_empty(),
        "Postman chunk has no external values"
    );
    // OODF file_identifier at offset 4 of the delta slice (FlatBuffers convention).
    assert_eq!(
        &container.delta_bytes[4..8],
        b"OODF",
        "delta payload must carry the OODF file identifier"
    );

    // Decode the OODF payload and confirm the 2 DiffChangeSection structure.
    let delta =
        oodf_ffi::parse_delta(container.delta_bytes).expect("OODF payload must verify + decode");
    assert_eq!(
        delta.struct_commands.len(),
        2,
        "Postman fixture carries exactly 2 struct commands"
    );
    for (i, cmd) in delta.struct_commands.iter().enumerate() {
        assert!(
            matches!(cmd, StructCommand::DiffChangeSection(_)),
            "Postman struct_commands[{i}] must be DiffChangeSection, got {cmd:?}"
        );
    }
}

// ── 2. THE regression test: CreateLayer accepts the captured OMPF chunk ────
//
// Pre-Bee `b.duy` this failed with InvalidArgument ("unaligned i32") because
// the Rust parser read the C++-emitted union-vector wire format with the
// wrong layout. Post-Bee the OODF FFI handles it correctly.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_accepts_captured_postman_chunk() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("postman-create", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let ompf_bytes = postman_ompf_bytes();

    let stream = streaming_create(vec![
        make_create_header(&session_id, "postman-layer"),
        make_create_data(ompf_bytes),
    ]);
    let request = make_request(stream);

    let response = handle_create_layer(&svc, request)
        .await
        .expect("captured Postman chunk MUST parse (this is the Bee b.duy regression)");

    let resp = response.into_inner();
    assert!(
        resp.object_id > 0,
        "object_id must be positive after successful CreateLayer"
    );
}

// ── 3. StatLayer after CreateLayer reports the layer exists ────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_after_create_with_postman_chunk_reports_metadata() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("postman-stat", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // CreateLayer with the captured chunk.
    let stream = streaming_create(vec![
        make_create_header(&session_id, "postman-layer"),
        make_create_data(postman_ompf_bytes()),
    ]);
    handle_create_layer(&svc, make_request(stream))
        .await
        .expect("CreateLayer with Postman chunk must succeed");

    // StatLayer returns metadata.
    let response = handle_stat_layer(&svc, stat_request(&session_id, "postman-layer"))
        .await
        .expect("StatLayer on the newly created layer must succeed");
    let stat = response.into_inner();
    assert!(
        !stat.created_by_user_name.is_empty(),
        "created_by_user_name must be populated"
    );
    assert!(stat.created_at > 0, "created_at timestamp must be non-zero");
    assert!(
        stat.last_modified_at > 0,
        "last_modified_at timestamp must be non-zero"
    );
    assert!(
        stat.last_modified_at >= stat.created_at,
        "last_modified_at must be >= created_at"
    );
}

// ── 4. UpdateLayer with a constructed OwnedDelta increments sequence_num ───
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_after_postman_create_increments_sequence_num() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("postman-update", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // CreateLayer with the captured chunk.
    let create_stream = streaming_create(vec![
        make_create_header(&session_id, "postman-layer"),
        make_create_data(postman_ompf_bytes()),
    ]);
    let create_resp = handle_create_layer(&svc, make_request(create_stream))
        .await
        .expect("CreateLayer must succeed")
        .into_inner();
    let object_id = create_resp.object_id;
    assert!(object_id > 0);

    // UpdateLayer with a non-reset diff carrying one DiffChangeSection.
    // (is_diff=false here — the helper `delta_with_diff_change_section` sets
    // is_diff=true + base_version=0 which would RESET state and erase the
    // sections the Postman chunk just materialized. We want additive semantics.)
    let update_delta = OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![oodf_ffi::StructCommand::DiffChangeSection(
            CommandDiffChangeSection {
                parent_id: 0,
                section_id: 99,
                section_name: "post-update".to_string(),
                section_type: 1,
                section_order: 0,
            },
        )],
        ..Default::default()
    };
    let update_ompf = delta_to_ompf_bytes(&update_delta, &[]);

    use live_edit::grpc::proto::{update_layer_request, UpdateLayerHeader, UpdateLayerRequest};

    let messages = vec![
        UpdateLayerRequest {
            message_type: Some(update_layer_request::MessageType::Header(
                UpdateLayerHeader {
                    session_id: session_id.clone(),
                    layer_name: "postman-layer".to_owned(),
                    object_id,
                    timestamp_microseconds_since_epoch: 2_000,
                    client_id: "postman-client".to_owned(),
                    operation_id: String::new(),
                },
            )),
        },
        UpdateLayerRequest {
            message_type: Some(update_layer_request::MessageType::Chunk(
                live_edit::grpc::proto::UpdateLayerChunk { data: update_ompf },
            )),
        },
    ];

    let request = make_request(streaming_update(messages));

    let update_resp = handle_update_layer(&svc, request)
        .await
        .expect("UpdateLayer after CreateLayer must succeed")
        .into_inner();

    // CreateLayer leaves the freshly created object at sequence_num=1; the
    // UpdateLayer bumps it to 2 (one increment per apply_delta — SDD §9.1).
    assert_eq!(
        update_resp.sequence_num, 2,
        "UpdateLayer must advance sequence_num from 1 to 2, got {}",
        update_resp.sequence_num
    );
    assert!(
        update_resp.server_timestamp_microseconds_since_epoch > 0,
        "server_timestamp must be non-zero"
    );
}

// ── 5. SubscribeReadLayer emits the Postman commands + the update ──────────
//
// Full round-trip through the service: subscriber pulls catch-up content,
// the returned OMPF decodes via `oodf_ffi::parse_delta`, and the resulting
// OwnedDelta carries the DiffChangeSection entries from the original
// Postman chunk (materialized as CreateSection during catch-up) plus the
// update we layered on top.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_after_postman_create_and_update_yields_expected_sections() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("postman-subscribe", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // CreateLayer with the captured chunk.
    let create_stream = streaming_create(vec![
        make_create_header(&session_id, "postman-layer"),
        make_create_data(postman_ompf_bytes()),
    ]);
    let object_id = handle_create_layer(&svc, make_request(create_stream))
        .await
        .expect("CreateLayer must succeed")
        .into_inner()
        .object_id;
    assert!(object_id > 0);

    // UpdateLayer with a non-reset diff adding a fresh section. Non-reset
    // matters: using `is_diff=true, base_version=0` would trigger SDD §9.1.3
    // reset semantics and erase the two sections the Postman chunk just
    // materialized.
    let update_section_id: u64 = 99;
    let update_delta = OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![oodf_ffi::StructCommand::DiffChangeSection(
            CommandDiffChangeSection {
                parent_id: 0,
                section_id: update_section_id,
                section_name: "subscriber-can-see-me".to_string(),
                section_type: 1,
                section_order: 0,
            },
        )],
        ..Default::default()
    };
    let update_ompf = delta_to_ompf_bytes(&update_delta, &[]);

    use live_edit::grpc::proto::{update_layer_request, UpdateLayerHeader, UpdateLayerRequest};

    let messages = vec![
        UpdateLayerRequest {
            message_type: Some(update_layer_request::MessageType::Header(
                UpdateLayerHeader {
                    session_id: session_id.clone(),
                    layer_name: "postman-layer".to_owned(),
                    object_id,
                    timestamp_microseconds_since_epoch: 3_000,
                    client_id: "postman-writer".to_owned(),
                    operation_id: String::new(),
                },
            )),
        },
        UpdateLayerRequest {
            message_type: Some(update_layer_request::MessageType::Chunk(
                live_edit::grpc::proto::UpdateLayerChunk { data: update_ompf },
            )),
        },
    ];
    handle_update_layer(&svc, make_request(streaming_update(messages)))
        .await
        .expect("UpdateLayer must succeed");

    // Subscribe fresh (object_id=0, sequence_num=0) → Case 1: full content catch-up.
    let request = make_subscribe_request(&session_id, "postman-layer", 0, 0, "postman-reader");
    let response = handle_subscribe_read_layer(&svc, request)
        .await
        .expect("SubscribeReadLayer must succeed");
    let mut stream = response.into_inner();

    // First message is the header; collect total_size.
    let first = stream
        .next()
        .await
        .expect("stream must yield at least one message")
        .expect("first message must be Ok");
    let total_size = match first.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => {
            assert!(
                h.total_size > 0,
                "full-content subscribe must have total_size > 0"
            );
            assert_eq!(h.object_id, object_id, "header object_id mismatch");
            assert_eq!(
                h.sequence_num, 2,
                "post-update sequence_num must be 2, got {}",
                h.sequence_num
            );
            h.total_size
        }
        other => panic!("expected Header message, got {other:?}"),
    };

    // Accumulate chunk payloads until we've read total_size bytes.
    let mut ompf_accum: Vec<u8> = Vec::with_capacity(total_size as usize);
    while (ompf_accum.len() as u64) < total_size {
        let next = stream
            .next()
            .await
            .expect("stream must yield data chunks for total_size > 0")
            .expect("data chunk must be Ok");
        match next.response_type {
            Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                ompf_accum.extend_from_slice(&c.data);
            }
            other => panic!("expected Chunk after header, got {other:?}"),
        }
    }
    assert_eq!(
        ompf_accum.len() as u64,
        total_size,
        "accumulated chunk bytes must match total_size"
    );

    // Decode the OMPF + OODF payload and confirm the subscriber sees:
    //   - the 2 sections originally carried by the Postman chunk, and
    //   - the section added via UpdateLayer.
    let container = OmpfContainer::parse(&ompf_accum).expect("subscribe OMPF must parse");
    let delta = oodf_ffi::parse_delta(container.delta_bytes)
        .expect("subscribe OODF payload must verify + decode");

    // diff_read emits CreateSection for existing sections during full-content
    // catch-up (SDD §12.2), not DiffChangeSection. Count both variants to be
    // robust to either representation.
    let section_cmds: Vec<&StructCommand> = delta
        .struct_commands
        .iter()
        .filter(|c| {
            matches!(
                c,
                StructCommand::CreateSection(_) | StructCommand::DiffChangeSection(_)
            )
        })
        .collect();
    assert!(
        section_cmds.len() >= 3,
        "subscriber must see at least 3 section commands (2 from Postman chunk + 1 update), got {}: {:?}",
        section_cmds.len(),
        delta.struct_commands,
    );

    // The update's section must be present by section_id somewhere in the
    // emitted commands.
    let section_ids: Vec<u64> = delta
        .struct_commands
        .iter()
        .filter_map(|c| match c {
            StructCommand::CreateSection(s) => Some(s.section_id),
            StructCommand::DiffChangeSection(s) => Some(s.section_id),
            _ => None,
        })
        .collect();
    assert!(
        section_ids.contains(&update_section_id),
        "subscriber must see the update's section_id={update_section_id}, saw {section_ids:?}"
    );
}
