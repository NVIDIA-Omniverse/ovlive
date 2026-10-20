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

// Integration tests for the DeleteSession cascade (bug b.f3b).
//
// Verifies that `handle_delete_session` stops when layer cleanup fails,
// preserving session metadata so that retries can finish the cascade
// idempotently, and that the failure code follows the failing layer's class:
// a transient failure answers retriable `UNAVAILABLE` carrying the progress
// counts, a permanent one passes through (e.g. redacted `INTERNAL`). No path
// answers `ABORTED`, which the Kit client does not retry.

use std::collections::HashMap;
use std::sync::Arc;
#[cfg(feature = "test-fault-injection")]
use std::sync::Mutex;

use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface};
use tonic::Request;

#[cfg(feature = "test-fault-injection")]
use live_edit::error::ServiceError;
use live_edit::grpc::proto::DeleteSessionRequest;
use live_edit::grpc::session::handle_delete_session;
use live_edit::grpc::BearerToken;
use live_edit::storage::keys;
#[cfg(feature = "test-fault-injection")]
use live_edit::storage::layer::LayerStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Build a `tonic::Request<DeleteSessionRequest>` with a bearer token.
fn delete_request(session_id: &str) -> Request<DeleteSessionRequest> {
    let mut request = Request::new(DeleteSessionRequest {
        session_id: session_id.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

// ── 1. Happy path: all layers delete → session metadata removed ─────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_cascade_all_layers_succeed_removes_session() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // Create a session with two layers.
    let (session_id, _oid1) = create_session_and_layer(&tv, &svc, "cascade-happy", "layer-a").await;

    // Add a second layer to the same session.
    let ompf_bytes = crate::common::owned_delta::delta_to_ompf_bytes(
        &crate::common::owned_delta::empty_delta(),
        &[],
    );
    let stream = streaming_create(vec![
        make_create_header(&session_id, "layer-b"),
        make_create_data(ompf_bytes),
    ]);
    let request = make_request(stream);
    live_edit::grpc::layer::handle_create_layer(&svc, request)
        .await
        .expect("second layer creation must succeed");

    // Verify we have two layers in the index.
    let layer_names = svc
        .layer_store
        .list_layer_names(&session_id)
        .await
        .expect("list_layer_names must succeed");
    assert_eq!(layer_names.len(), 2, "session must have exactly 2 layers");

    // Delete the session — both layers should be cleaned up.
    let response = handle_delete_session(&svc, delete_request(&session_id))
        .await
        .expect("delete_session must succeed when all layers delete");

    // Verify response is empty (no fields in DeleteSessionResponse).
    let _ = response.into_inner();

    // Verify session metadata is gone.
    let meta: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().session_meta(&session_id))
        .await
        .expect("HGETALL must succeed");
    assert!(
        meta.is_empty(),
        "session metadata must be removed after successful cascade, got: {meta:?}"
    );

    // Verify layers index is gone.
    let remaining: Vec<String> = tv
        .client()
        .pool
        .smembers(tv.key_schema().layers_index(&session_id))
        .await
        .expect("SMEMBERS must succeed");
    assert!(
        remaining.is_empty(),
        "layers index must be empty after cascade, got: {remaining:?}"
    );
}

// ── 2. Permanent layer failure → Internal, session metadata preserved ───────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_cascade_permanent_layer_failure_returns_internal_preserves_metadata() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // Create a session with one real layer.
    let (session_id, _oid) =
        create_session_and_layer(&tv, &svc, "cascade-fail", "good-layer").await;

    // Inject a corrupt layer: add a fake entry to the layers index with a
    // non-numeric mapping value. When delete_layer reads the mapping it will
    // fail with ServiceError::Internal (corrupt parse), which is not NotFound
    // and is permanent — no retry can repair the stored value.
    let hex_bad = keys::hex_encode("corrupt-layer");
    let layers_index_key = tv.key_schema().layers_index(&session_id);
    tv.client()
        .pool
        .sadd::<(), _, _>(&layers_index_key, &hex_bad)
        .await
        .expect("SADD must succeed");

    let mapping_key = tv.key_schema().layer_mapping(&session_id, "corrupt-layer");
    tv.client()
        .pool
        .set::<(), _, _>(&mapping_key, "not-a-number", None, None, false)
        .await
        .expect("SET corrupt mapping must succeed");

    // Attempt cascade — the permanent class passes through unchanged.
    let status = handle_delete_session(&svc, delete_request(&session_id))
        .await
        .expect_err("delete_session must fail when a layer deletion fails");

    // A permanent layer failure must NOT be dressed up as retriable: it
    // answers INTERNAL, whose wire message is redacted like every other
    // INTERNAL. The layer name and progress counts therefore cannot be
    // asserted here; the transient variant
    // (`delete_session_cascade_transient_layer_failure_returns_unavailable_with_counts`)
    // pins them.
    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "a permanent layer failure must surface as INTERNAL, got: {} — {}",
        status.code(),
        status.message()
    );
    assert_eq!(
        status.message(),
        "internal error",
        "INTERNAL must stay redacted on the wire"
    );

    // Verify session metadata is preserved (the bug fix).
    let meta: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().session_meta(&session_id))
        .await
        .expect("HGETALL must succeed");
    assert!(
        !meta.is_empty(),
        "session metadata must be preserved when cascade fails"
    );

    // Verify the layers index still has entries (so retry can find remaining layers).
    let remaining: Vec<String> = tv
        .client()
        .pool
        .smembers(tv.key_schema().layers_index(&session_id))
        .await
        .expect("SMEMBERS must succeed");
    assert!(
        !remaining.is_empty(),
        "layers index must be preserved when cascade fails"
    );
}

// ── 3. Retry after partial failure completes successfully ───────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_cascade_retry_after_partial_failure_succeeds() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // Create a session with one real layer.
    let (session_id, _oid) =
        create_session_and_layer(&tv, &svc, "cascade-retry", "real-layer").await;

    // Inject a corrupt layer (same as test 2).
    let hex_bad = keys::hex_encode("corrupt-layer");
    let layers_index_key = tv.key_schema().layers_index(&session_id);
    tv.client()
        .pool
        .sadd::<(), _, _>(&layers_index_key, &hex_bad)
        .await
        .expect("SADD must succeed");

    let mapping_key = tv.key_schema().layer_mapping(&session_id, "corrupt-layer");
    tv.client()
        .pool
        .set::<(), _, _>(&mapping_key, "not-a-number", None, None, false)
        .await
        .expect("SET corrupt mapping must succeed");

    // First attempt — fails with the corrupt layer's permanent class.
    let status = handle_delete_session(&svc, delete_request(&session_id))
        .await
        .expect_err("first delete must fail");
    assert_eq!(status.code(), tonic::Code::Internal);

    // After b.91j H-grpc-7 the cascade bails on the first non-NotFound
    // failure, so whether the real layer was deleted on the first attempt
    // depends on `SMEMBERS` iteration order (real vs. corrupt first). The
    // retry below is tolerant of either outcome — the corrupt layer is
    // healed by clearing its mapping, and the real layer either deletes
    // cleanly or returns NotFound if it was already removed in the first
    // attempt.
    tv.client()
        .pool
        .del::<(), _>(&mapping_key)
        .await
        .expect("DEL corrupt mapping must succeed");

    // Retry — the corrupt layer now returns NotFound (mapping gone), and
    // the real layer either deletes cleanly or returns NotFound depending
    // on whether the first attempt reached it. Both paths are tolerated
    // by the cascade's NotFound idempotency marker.
    let response = handle_delete_session(&svc, delete_request(&session_id))
        .await
        .expect("retry must succeed after fixing corrupt layer");

    let _ = response.into_inner();

    // Verify session metadata is now removed.
    let meta: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().session_meta(&session_id))
        .await
        .expect("HGETALL must succeed");
    assert!(
        meta.is_empty(),
        "session metadata must be removed after successful retry, got: {meta:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// b.91j H-grpc-7: cascade bails on first non-NotFound failure
// ═══════════════════════════════════════════════════════════════════════════════
//
// Pre-fix the cascade iterated every layer, collecting failures into a
// `has_failures` bool, and returned `Aborted` only after touching all N
// layers. Under a flapping Valkey a 100-layer session burned `100 ×
// (timeout × retry_max)` work before surfacing — amplifying one outage
// into DEADLINE_EXCEEDED.
//
// The fix returns on the first non-NotFound error. These tests pin the
// two sides of the contract:
//
//   1. `delete_session_cascade_bails_on_first_non_not_found` — 5 layers,
//      inject a transient failure on the 2nd-encountered layer name
//      (discovered via `list_layer_names` pre-check so iteration order is
//      deterministic). Assert: Unavailable, message includes "1 of 5",
//      layers 2..=5 still in layers_index.
//   2. `delete_session_cascade_notfound_not_a_failure` — NotFound on a
//      layer is the idempotency marker; the cascade must continue past
//      it, not bail. (Already covered by the retry test above; this one
//      pins it as an explicit invariant with a fault injector rather
//      than relying on a corrupt-mapping recovery dance.)
//
// The failure-code tests below them pin which gRPC code the bail answers:
// the failing layer's class decides it (see
// `delete_session_cascade_failure_code_follows_the_layer_failure_class`).
//
// `build_service_with_layer_store` and `layer_store_with_name_faults` live in
// `super::helpers`, shared with `rpc_failure_log_test.rs`.

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_cascade_bails_on_first_non_not_found() {
    // Setup: build the service with a LayerStore that carries a per-name
    // fault-injection map.
    use live_edit::grpc::layer::handle_create_layer;

    let tv = TestValkey::setup().await;

    // Shared fault injector — populated after we learn the iteration
    // order from `list_layer_names`.
    let fault_map: Arc<Mutex<HashMap<String, ServiceError>>> = Arc::new(Mutex::new(HashMap::new()));
    let layer_store = layer_store_with_name_faults(&tv, fault_map.clone());
    let svc = build_service_with_layer_store(&tv, layer_store.clone());

    // Create a session with 5 layers.
    let (session_id, _oid0) = create_session_and_layer(&tv, &svc, "h-grpc-7", "layer-0").await;
    for i in 1..5 {
        let layer_name = format!("layer-{i}");
        let ompf_bytes = crate::common::owned_delta::delta_to_ompf_bytes(
            &crate::common::owned_delta::empty_delta(),
            &[],
        );
        let stream = streaming_create(vec![
            make_create_header(&session_id, &layer_name),
            make_create_data(ompf_bytes),
        ]);
        let request = make_request(stream);
        handle_create_layer(&svc, request)
            .await
            .unwrap_or_else(|e| panic!("create {layer_name} must succeed: {e}"));
    }

    // Learn the iteration order the cascade will see. `list_layer_names`
    // is SMEMBERS-backed (unordered in Valkey), so we query it now and
    // inject on the 2nd element in that observed order — the cascade
    // iterates the same returned `Vec<String>`, so the 2nd element is
    // exactly the 2nd encountered layer.
    let observed_order = layer_store
        .list_layer_names(&session_id)
        .await
        .expect("list_layer_names must succeed");
    assert_eq!(
        observed_order.len(),
        5,
        "sanity: 5 layers must be in the index before cascade"
    );
    let target_layer = observed_order[1].clone();
    let untouched_layers: Vec<String> = observed_order[2..].to_vec();

    // A transient class, so the counts reach the wire (an INTERNAL-class
    // failure bails identically but its wire message is redacted).
    fault_map.lock().unwrap().insert(
        target_layer.clone(),
        ServiceError::Aborted("transient I/O: simulated Valkey blip".to_owned()),
    );

    // Run the cascade.
    let status = handle_delete_session(&svc, delete_request(&session_id))
        .await
        .expect_err("cascade must fail when the 2nd layer delete errors");

    // Contract: Unavailable (retriable) with the failing layer name and
    // "1 of 5" counts.
    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "expected Unavailable (retriable), got: {} — {}",
        status.code(),
        status.message()
    );
    assert!(
        status.message().contains(&target_layer),
        "message must name the failing layer '{target_layer}': {}",
        status.message()
    );
    assert!(
        status.message().contains("1 of 5"),
        "message must indicate 1 deleted of 5 total: {}",
        status.message()
    );
    assert!(
        status.message().contains("3 remaining"),
        "message must indicate 3 layers remaining: {}",
        status.message()
    );

    // Storage invariant: layers 3..=5 (in SMEMBERS order) must still be
    // in the layers_index — the cascade bailed before touching them. The
    // 1st layer was deleted; the target (2nd) was never deleted because
    // the injection fires before `get_layer_mapping`, so its mapping and
    // index entry are intact.
    let remaining: Vec<String> = tv
        .client()
        .pool
        .smembers(tv.key_schema().layers_index(&session_id))
        .await
        .expect("SMEMBERS layers_index must succeed");
    assert!(
        !remaining.is_empty(),
        "layers_index must not be empty — the untouched layers must remain"
    );
    let remaining_decoded: Vec<String> = remaining.iter().map(|h| keys::hex_decode(h)).collect();
    // Untouched layers (observed_order[2..]) must all still be present.
    for layer in &untouched_layers {
        assert!(
            remaining_decoded.contains(layer),
            "untouched layer {layer} must still be in layers_index; remaining: {remaining_decoded:?}"
        );
    }
    // The target layer itself was never actually deleted (fault fired
    // before step 1), so its mapping + index entry are intact. This
    // matches the contract: the cascade did not partially destroy state
    // on the failing layer.
    assert!(
        remaining_decoded.contains(&target_layer),
        "target layer {target_layer} must still be in layers_index \
         (fault fires before any step of delete_layer): remaining: {remaining_decoded:?}"
    );

    // Session metadata must still be present — cascade incomplete.
    let meta: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().session_meta(&session_id))
        .await
        .expect("HGETALL must succeed");
    assert!(
        !meta.is_empty(),
        "session metadata must be preserved when cascade bails early"
    );
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_cascade_notfound_is_idempotent_marker_not_bail() {
    // Companion invariant: NotFound on an individual layer is the
    // idempotency marker per SDD §7.8 (partial retry already deleted
    // it). The cascade must continue past NotFound, NOT bail. Pre-fix
    // this also did not bail (the correctness of NotFound was never
    // broken); post-fix we pin it explicitly so a future "bail on any
    // error" refactor doesn't silently break retry idempotency.
    use live_edit::grpc::layer::handle_create_layer;

    let tv = TestValkey::setup().await;

    let fault_map: Arc<Mutex<HashMap<String, ServiceError>>> = Arc::new(Mutex::new(HashMap::new()));
    let layer_store = layer_store_with_name_faults(&tv, fault_map.clone());
    let svc = build_service_with_layer_store(&tv, layer_store.clone());

    // Create a session with 3 layers.
    let (session_id, _oid0) = create_session_and_layer(&tv, &svc, "nf-marker", "layer-0").await;
    for i in 1..3 {
        let layer_name = format!("layer-{i}");
        let ompf_bytes = crate::common::owned_delta::delta_to_ompf_bytes(
            &crate::common::owned_delta::empty_delta(),
            &[],
        );
        let stream = streaming_create(vec![
            make_create_header(&session_id, &layer_name),
            make_create_data(ompf_bytes),
        ]);
        let request = make_request(stream);
        handle_create_layer(&svc, request)
            .await
            .expect("create must succeed");
    }

    // Inject NotFound on every layer — simulates a re-run of a cascade
    // where every layer has already been cleaned up. The cascade must
    // continue through all 3, then delete the session.
    let observed_order = layer_store
        .list_layer_names(&session_id)
        .await
        .expect("list_layer_names must succeed");
    for layer in &observed_order {
        fault_map.lock().unwrap().insert(
            layer.clone(),
            ServiceError::NotFound(format!("layer '{layer}' already gone")),
        );
    }

    // Cascade must succeed — every NotFound is tolerated, then session
    // metadata + layers_index are deleted by `SessionStore::delete`.
    let response = handle_delete_session(&svc, delete_request(&session_id))
        .await
        .expect("cascade must tolerate NotFound on every layer");
    let _ = response.into_inner();

    let meta: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().session_meta(&session_id))
        .await
        .expect("HGETALL must succeed");
    assert!(
        meta.is_empty(),
        "session metadata must be gone after all-NotFound cascade"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cascade failure code: the failing layer's class decides it
// ═══════════════════════════════════════════════════════════════════════════════
//
// A transient layer failure (`Aborted` — the in-process retry signal, or a
// retry loop's exhaustion — or `Unavailable`) answers `UNAVAILABLE`, which the
// Kit client retries, with the progress counts so the retry resumes through
// the NotFound-skip. Every other class passes through unchanged, so a permanent
// failure is not dressed up as retriable. No class answers `ABORTED`.

/// The realistic transient shape end to end: `delete_layer`'s own retry loop
/// (budget 1) exhausts on a transient `DEL layer mapping` blip, the cascade
/// answers `UNAVAILABLE` carrying the counts, and the client's retry — the
/// behaviour `UNAVAILABLE` exists to trigger — completes the delete.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_cascade_transient_layer_failure_returns_unavailable_with_counts() {
    use std::collections::VecDeque;

    const LAYER: &str = "transient-layer";

    let tv = TestValkey::setup().await;
    let queue: Arc<Mutex<VecDeque<ServiceError>>> = Arc::new(Mutex::new(VecDeque::new()));
    // Budget 1: the single attempt's transient failure exhausts the loop, so
    // the `Aborted` escapes `delete_layer` into the cascade.
    let layer_store = Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_retry_max_attempts(1)
            .with_delete_layer_retry_body_fault_injector_for_test(queue.clone()),
    );
    let svc = build_service_with_layer_store(&tv, layer_store);
    let (session_id, _oid) = create_session_and_layer(&tv, &svc, "cascade-transient", LAYER).await;

    queue.lock().unwrap().push_back(ServiceError::Aborted(
        "transient I/O: DEL layer mapping failed (Timeout): synthetic blip".to_owned(),
    ));

    let status = handle_delete_session(&svc, delete_request(&session_id))
        .await
        .expect_err("a transient layer failure must fail the cascade");

    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "a transient layer failure must answer retriable UNAVAILABLE (the Kit client does \
         not retry ABORTED), got: {} — {}",
        status.code(),
        status.message()
    );
    let msg = status.message();
    for needle in [
        "cascade failed on layer 'transient-layer'",
        "after 0 of 1 layers deleted (0 remaining)",
        "retry to continue",
        "attempts exhausted",
    ] {
        assert!(
            msg.contains(needle),
            "the UNAVAILABLE message must carry `{needle}` — the per-layer progress is only \
             visible on the wire for the transient class; got: {msg}"
        );
    }
    assert!(
        queue.lock().unwrap().is_empty(),
        "the fault must have been consumed by the cascade's single delete attempt"
    );

    // The client's retry resumes the cascade and completes it.
    handle_delete_session(&svc, delete_request(&session_id))
        .await
        .expect("the client's retry after UNAVAILABLE must complete the cascade");
    let meta: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().session_meta(&session_id))
        .await
        .expect("HGETALL must succeed");
    assert!(
        meta.is_empty(),
        "session metadata must be gone after the retry completes, got: {meta:?}"
    );
}

/// One row per failure class the cascade can meet, each on its own
/// single-layer session. Only the two transient classes are relabelled (to
/// `UNAVAILABLE`, with the counts); every other class passes through with its
/// own code, so a permanent failure is never answered as retriable.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_cascade_failure_code_follows_the_layer_failure_class() {
    use live_edit::error::KeyspaceRefusal;

    struct Row {
        label: &'static str,
        injected: fn() -> ServiceError,
        want: tonic::Code,
    }
    let rows = [
        Row {
            label: "aborted",
            injected: || ServiceError::Aborted("transient I/O: synthetic blip".to_owned()),
            want: tonic::Code::Unavailable,
        },
        Row {
            label: "unavailable",
            injected: || ServiceError::Unavailable("synthetic single-attempt blip".to_owned()),
            want: tonic::Code::Unavailable,
        },
        Row {
            label: "internal",
            injected: || ServiceError::Internal("synthetic corrupt value".to_owned()),
            want: tonic::Code::Internal,
        },
        Row {
            label: "permanent-keyspace-refusal",
            injected: || ServiceError::PermanentKeyspaceRefusal {
                refusal: KeyspaceRefusal::WrongType,
                detail: "synthetic wrong-type key".to_owned(),
            },
            want: tonic::Code::Internal,
        },
        Row {
            label: "failed-precondition",
            injected: || ServiceError::FailedPrecondition("synthetic precondition".to_owned()),
            want: tonic::Code::FailedPrecondition,
        },
    ];

    let tv = TestValkey::setup().await;
    let fault_map: Arc<Mutex<HashMap<String, ServiceError>>> = Arc::new(Mutex::new(HashMap::new()));
    let svc =
        build_service_with_layer_store(&tv, layer_store_with_name_faults(&tv, fault_map.clone()));

    for row in &rows {
        let layer = format!("layer-{}", row.label);
        let (session_id, _oid) =
            create_session_and_layer(&tv, &svc, &format!("class-{}", row.label), &layer).await;
        fault_map
            .lock()
            .unwrap()
            .insert(layer.clone(), (row.injected)());

        let status = handle_delete_session(&svc, delete_request(&session_id))
            .await
            .expect_err("an injected layer failure must fail the cascade");

        assert_eq!(
            status.code(),
            row.want,
            "[{}] cascade failure code must follow the layer's class, got: {} — {}",
            row.label,
            status.code(),
            status.message()
        );
        assert_ne!(
            status.code(),
            tonic::Code::Aborted,
            "[{}] no DeleteSession outcome may answer ABORTED",
            row.label
        );
        match row.want {
            tonic::Code::Unavailable => {
                assert!(
                    status.message().contains(&format!(
                        "cascade failed on layer '{layer}' after 0 of 1 layers deleted"
                    )),
                    "[{}] a relabelled transient failure must carry the progress counts, got: {}",
                    row.label,
                    status.message()
                );
                // A transient is not a nothing-applied refusal: no retry hint
                // is invented for it.
                crate::common::refusal_detail::assert_no_refusal_detail(&status);
            }
            tonic::Code::Internal => assert_eq!(
                status.message(),
                "internal error",
                "[{}] a permanent failure passes through and stays redacted",
                row.label
            ),
            _ => assert_eq!(
                status.message(),
                "synthetic precondition",
                "[{}] a pass-through class keeps its own message, not the cascade wrapper",
                row.label
            ),
        }
    }
}

/// **The cascade keeps a layer's retry hint (b.kgb).** A layer delete the
/// backing store refused — the store's own refusal, as
/// `StoreWriteRefusal::into_service_error` builds it, injected on the layer's
/// name — fails the cascade `UNAVAILABLE` with the progress counts, as any
/// transient does, and the refusal detail the layer's refusal carried
/// survives the wrapping: `STORE_WRITE_REFUSED`, `condition` naming the
/// refusal, a `retry_delay` in `[2 s, 6 s)`. The layer's own message follows
/// the counts. Once the store accepts writes again, the client's retry
/// resumes the cascade and completes it. Each of the four refusals, on its
/// own single-layer session.
///
/// The store's refusals do not reach the cascade's own writes through a real
/// store path today (the cascade's delete writes are not `refuse_store_write`
/// sites), so the name-keyed fault seam stands in for the refused layer.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_cascade_over_a_store_write_refusal_keeps_the_retry_hint_with_the_progress()
{
    use live_edit::error::{refusal_metadata, refusal_reason, STORE_WRITE_REFUSED_RETRY_DELAY};
    use live_edit::storage::client::StoreWriteRefusal;

    let tv = TestValkey::setup().await;
    let fault_map: Arc<Mutex<HashMap<String, ServiceError>>> = Arc::new(Mutex::new(HashMap::new()));
    let svc =
        build_service_with_layer_store(&tv, layer_store_with_name_faults(&tv, fault_map.clone()));

    for refusal in [
        StoreWriteRefusal::OutOfMemory,
        StoreWriteRefusal::Misconf,
        StoreWriteRefusal::ReadOnly,
        StoreWriteRefusal::NoReplicas,
    ] {
        let label = refusal.reason_label();
        let layer = format!("layer-{label}");
        let (session_id, _oid) =
            create_session_and_layer(&tv, &svc, &format!("refused-{label}"), &layer).await;
        let injected = refusal.into_service_error();
        let layer_message = injected.to_string();
        fault_map.lock().unwrap().insert(layer.clone(), injected);

        let status = handle_delete_session(&svc, delete_request(&session_id))
            .await
            .expect_err("a refused layer delete must fail the cascade");
        assert_eq!(
            status.code(),
            tonic::Code::Unavailable,
            "[{label}] {status:?}"
        );
        assert_eq!(
            status.message(),
            format!(
                "DeleteSession cascade failed on layer '{layer}' after 0 of 1 layers deleted (0 \
                 remaining), retry to continue: {layer_message}"
            ),
            "[{label}] the cascade wraps the refusal with its progress, as it wraps a transient"
        );
        crate::common::refusal_detail::assert_refusal_detail(
            &status,
            refusal_reason::STORE_WRITE_REFUSED,
            &[(refusal_metadata::CONDITION, label)],
            STORE_WRITE_REFUSED_RETRY_DELAY,
        );

        fault_map.lock().unwrap().remove(&layer);
        handle_delete_session(&svc, delete_request(&session_id))
            .await
            .unwrap_or_else(|s| {
                panic!("[{label}] the client's retry must complete the cascade: {s:?}")
            });
    }
}
