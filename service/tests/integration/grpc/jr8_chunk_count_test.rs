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

// Bug b.jr8 H-grpc-4: the upload content reader caps total chunk count.
//
// The byte-budget cap (`max_layer_size`) on its own does not bound the number
// of frames that pass through prost's decoder: an attacker sending N × 1-byte
// chunks can burn CPU proportional to N while staying under the byte cap.
// The fix adds `chunk_count_cap(max_size) = max(1, max_size / 1024)` enforced
// inside `read_upload_body`'s chunk loop, raising `InvalidArgument` on
// overflow.
//
// These tests exercise the fix end-to-end through both CreateLayer and
// UpdateLayer so the cap's wire-up on both client-streaming RPCs is pinned.
// The service's `max_layer_size` is tuned so the ticket's "100 × 1-byte
// chunks" shape trips the cap (`chunk_count_cap(max_size) < 100`).

use std::sync::Arc;

use dashmap::DashMap;

use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{
    create_layer_request, update_layer_request, CreateLayerChunk, CreateLayerHeader,
    CreateLayerRequest, UpdateLayerChunk, UpdateLayerHeader, UpdateLayerRequest,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::presence::LocalPresenceRouter;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;

// ── Small-budget service builder ─────────────────────────────────────────────
//
// `helpers::build_service` wires `max_layer_size = 10 * 1024 * 1024` which
// puts `chunk_count_cap = 10240` — far above a realistic per-test count. For
// H-grpc-4 we want the cap itself on trial, so we build a service with a
// small budget so sub-KiB chunks trip the cap fast.
//
// 80 KiB budget → cap = 80. Sending 100 × 1-byte chunks exceeds it.
const SMALL_MAX_LAYER_SIZE: usize = 80 * 1024;

fn build_service_with_max_layer_size(tv: &TestValkey, max_layer_size: usize) -> LiveEditService {
    let session_store = Arc::new(SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let layer_store = Arc::new(LayerStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
        4096,
    ));
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    let recovery = super::helpers::build_disabled_recovery_hook(
        Arc::clone(&layer_store),
        Arc::clone(&metrics),
    );
    let session_recovery =
        super::helpers::build_disabled_session_recovery_hook(tv, Arc::clone(&metrics));
    LiveEditService {
        catchup_snapshot_enabled: false,
        session_store,
        layer_store,
        ext_value_store: Arc::new(MockExternalValueStore::new()),
        session_tokens: Arc::new(DashMap::new()),
        userinfo_client: Arc::new(StubUserinfo),
        permissions_client: Arc::new(AllowAllPermissions),
        max_layer_size,
        upload_admission: std::sync::Arc::new(
            live_edit::grpc::upload_admission::UploadAdmissionGate::new(Default::default(), None),
        ),
        snapshot_flights: live_edit::storage::snapshot_flights::SnapshotFlights::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ),
        valkey: tv.client().clone(),
        key_schema: tv.key_schema().clone(),
        presence_router: Arc::new(LocalPresenceRouter::new()),
        presence_lease_ttl_secs: live_edit::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
        presence_refresh_interval_secs: live_edit::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
        presence_keepalive_interval_ms: live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
        presence_cleanup_tracker: tokio_util::task::TaskTracker::new(),
        subscription_tracker: tokio_util::task::TaskTracker::new(),
        shutdown: live_edit::grpc::graceful_shutdown::GracefulShutdown::new(),
        base_bootstrap: crate::common::build_disabled_base_bootstrap(
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::clone(&metrics),
        ),
        metrics,
        service_label: "test".to_string(),
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

// ── Message builders ─────────────────────────────────────────────────────────
//
// The stream framers (`streaming_create` / `streaming_update`) are the shared
// ones in `helpers`, which carry each upload's end-of-stream record.

fn make_update_header(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    client_id: &str,
) -> UpdateLayerRequest {
    UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Header(
            UpdateLayerHeader {
                session_id: session_id.to_owned(),
                layer_name: layer_name.to_owned(),
                object_id,
                client_id: client_id.to_owned(),
                operation_id: String::new(),
                timestamp_microseconds_since_epoch: 0,
            },
        )),
    }
}

fn make_update_data(payload: Vec<u8>) -> UpdateLayerRequest {
    UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Chunk(UpdateLayerChunk {
            data: payload,
        })),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// H-grpc-4: CreateLayer — dribble attack rejected with InvalidArgument
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_dribble_attack_100x1byte_chunks_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service_with_max_layer_size(&tv, SMALL_MAX_LAYER_SIZE);

    // Create the session so auth / get_base_path succeed — we want the chunk-
    // count cap to be the rejection reason, not a precondition.
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("jr8-chunk-cap", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // 100 × 1-byte chunks. With SMALL_MAX_LAYER_SIZE = 80 KiB the cap is 80,
    // so chunk #81 is the first to trip the cap with InvalidArgument.
    let mut messages: Vec<CreateLayerRequest> = Vec::with_capacity(101);
    messages.push(CreateLayerRequest {
        message_type: Some(create_layer_request::MessageType::Header(
            CreateLayerHeader {
                session_id: session_id.clone(),
                layer_name: "dribble-layer".to_owned(),
                operation_id: String::new(),
            },
        )),
    });
    for _ in 0..100 {
        messages.push(CreateLayerRequest {
            message_type: Some(create_layer_request::MessageType::Chunk(CreateLayerChunk {
                data: vec![0u8; 1],
            })),
        });
    }

    let stream = streaming_create(messages);
    let request = make_request(stream);

    let status = handle_create_layer(&svc, request)
        .await
        .expect_err("dribble attack must be rejected");

    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "chunk-count overflow must map to INVALID_ARGUMENT, got {} — {}",
        status.code(),
        status.message()
    );
    assert!(
        status.message().contains("chunk count"),
        "error message must identify the chunk-count cap, got: {}",
        status.message()
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// H-grpc-4: UpdateLayer — dribble attack rejected with InvalidArgument
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_dribble_attack_100x1byte_chunks_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service_with_max_layer_size(&tv, SMALL_MAX_LAYER_SIZE);

    // Create a real layer so header-validation (object_id, layer_mapping) would
    // pass — the cap must fire during the chunk loop, not before.
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "jr8-upd-cap", "layer-upd").await;

    let mut messages: Vec<UpdateLayerRequest> = Vec::with_capacity(101);
    messages.push(make_update_header(
        &session_id,
        "layer-upd",
        object_id,
        "jr8-client",
    ));
    for _ in 0..100 {
        messages.push(make_update_data(vec![0u8; 1]));
    }

    let stream = streaming_update(messages);
    let request = make_request(stream);

    let status = handle_update_layer(&svc, request)
        .await
        .expect_err("dribble attack on UpdateLayer must be rejected");

    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "chunk-count overflow must map to INVALID_ARGUMENT, got {} — {}",
        status.code(),
        status.message()
    );
    assert!(
        status.message().contains("chunk count"),
        "error message must identify the chunk-count cap, got: {}",
        status.message()
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// H-grpc-4: boundary — chunks at the cap succeed, one over fails
// ═══════════════════════════════════════════════════════════════════════════════
//
// Pins the exact inclusive/exclusive boundary. With `max_layer_size = 2048`,
// `chunk_count_cap` = 2. Two 1-byte chunks (total 2) must be accepted; a
// third chunk trips the cap.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_chunk_count_at_cap_accepted_over_cap_rejected() {
    // 2048 = min budget that produces a cap ≥ 2 via
    // `chunk_count_cap(max) = max(1, max / 1024)`.
    const TUNED_MAX: usize = 2048;
    // Sanity-check our derivation of the cap: kept as a runtime assert (not
    // `const`) so the test remains readable even if the cap formula ever
    // changes — it'd still yield a clear error message instead of silently
    // no-op'ing.
    assert_eq!(
        live_edit::grpc::streaming::chunk_count_cap(TUNED_MAX),
        2,
        "test relies on cap = 2 at max_layer_size = {TUNED_MAX}"
    );

    let tv = TestValkey::setup().await;
    let svc = build_service_with_max_layer_size(&tv, TUNED_MAX);

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let _session = session_store
        .create("jr8-boundary", "/stage", "test-user")
        .await
        .expect("session create must succeed");

    // At-cap: 2 tiny chunks. Expected to fail OMPF parsing (not the cap) —
    // what we care about is that the *chunk count* rejection does NOT fire.
    // The failure mode therefore is anything other than the chunk-count
    // InvalidArgument ("chunk count" in the message).
    let session_id_at_cap = session_store
        .create("jr8-at-cap-s", "/stage", "test-user")
        .await
        .expect("session create must succeed");
    let messages_at_cap: Vec<CreateLayerRequest> = vec![
        CreateLayerRequest {
            message_type: Some(create_layer_request::MessageType::Header(
                CreateLayerHeader {
                    session_id: session_id_at_cap.clone(),
                    layer_name: "at-cap".to_owned(),
                    operation_id: String::new(),
                },
            )),
        },
        CreateLayerRequest {
            message_type: Some(create_layer_request::MessageType::Chunk(CreateLayerChunk {
                data: vec![0u8; 1],
            })),
        },
        CreateLayerRequest {
            message_type: Some(create_layer_request::MessageType::Chunk(CreateLayerChunk {
                data: vec![0u8; 1],
            })),
        },
    ];
    let stream = streaming_create(messages_at_cap);
    let at_cap_result = handle_create_layer(&svc, make_request(stream)).await;
    match at_cap_result {
        Ok(_) => {
            // Possible but unlikely (would need valid OMPF in 2 bytes) — still
            // satisfies the assertion: the cap did not reject at exactly cap.
        }
        Err(status) => {
            assert!(
                !status.message().contains("chunk count"),
                "chunk-count cap must NOT fire at exactly `chunk_count_cap(max_size)` chunks; \
                 got: {}",
                status.message()
            );
        }
    }

    // One over cap: 3 tiny chunks. The third one must trip the chunk-count
    // cap with InvalidArgument.
    let session_id_over = session_store
        .create("jr8-over-cap", "/stage", "test-user")
        .await
        .expect("session create must succeed");
    let messages_over_cap: Vec<CreateLayerRequest> = vec![
        CreateLayerRequest {
            message_type: Some(create_layer_request::MessageType::Header(
                CreateLayerHeader {
                    session_id: session_id_over.clone(),
                    layer_name: "over-cap".to_owned(),
                    operation_id: String::new(),
                },
            )),
        },
        CreateLayerRequest {
            message_type: Some(create_layer_request::MessageType::Chunk(CreateLayerChunk {
                data: vec![0u8; 1],
            })),
        },
        CreateLayerRequest {
            message_type: Some(create_layer_request::MessageType::Chunk(CreateLayerChunk {
                data: vec![0u8; 1],
            })),
        },
        CreateLayerRequest {
            message_type: Some(create_layer_request::MessageType::Chunk(CreateLayerChunk {
                data: vec![0u8; 1],
            })),
        },
    ];
    let stream = streaming_create(messages_over_cap);
    let over_cap_status = handle_create_layer(&svc, make_request(stream))
        .await
        .expect_err("one-over-cap must be rejected");
    assert_eq!(
        over_cap_status.code(),
        tonic::Code::InvalidArgument,
        "one-over-cap must map to INVALID_ARGUMENT, got {}",
        over_cap_status.code()
    );
    assert!(
        over_cap_status.message().contains("chunk count"),
        "error must identify the chunk-count cap, got: {}",
        over_cap_status.message()
    );
}
