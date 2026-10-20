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

// Integration tests for the UpdateLayer handler's validation, auth, and layer
// resolution pipeline.
//
// Tests call `handle_update_layer` directly with a LiveEditService backed by
// a real Valkey (TestValkey fixture), mock auth clients, and gRPC-framed
// streaming requests.

use std::sync::Arc;

use live_edit::config::UpdateLayerBackend;
use live_edit::grpc::layer::handle_update_layer;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::{delta_to_ompf_bytes, delta_with_set_field_external, empty_delta};

// `streaming_update` / `make_update_header` / `make_update_data` used to be
// private fns here; they now live in `helpers` (promoted for the ompf-invariant
// module) and arrive through this glob import unchanged.
use super::helpers::*;

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

// ── 1. object_id=0 → INVALID_ARGUMENT ───────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_zero_object_id_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("zero-oid-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let stream = streaming_update(vec![make_update_header(&session_id, "some-layer", 0)]);
    let request = make_request(stream);

    let status = handle_update_layer(&svc, request)
        .await
        .expect_err("object_id=0 must fail");

    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "object_id=0 must return INVALID_ARGUMENT, got: {}",
        status.code()
    );
}

// ── 2. Non-existent layer → NOT_FOUND ───────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_nonexistent_layer_returns_not_found() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("nonexistent-layer-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // Use object_id=999 for a layer that was never created.
    let stream = streaming_update(vec![make_update_header(&session_id, "no-such-layer", 999)]);
    let request = make_request(stream);

    let status = handle_update_layer(&svc, request)
        .await
        .expect_err("non-existent layer must fail");

    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "non-existent layer must return NOT_FOUND, got: {}",
        status.code()
    );
}

// ── 3. Mismatched object_id → FAILED_PRECONDITION ───────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_mismatched_object_id_returns_failed_precondition() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "mismatch-session", "mismatch-layer").await;

    // Send UpdateLayer with object_id + 1 (wrong).
    let stream = streaming_update(vec![make_update_header(
        &session_id,
        "mismatch-layer",
        object_id + 1,
    )]);
    let request = make_request(stream);

    let status = handle_update_layer(&svc, request)
        .await
        .expect_err("mismatched object_id must fail");

    assert_eq!(
        status.code(),
        tonic::Code::FailedPrecondition,
        "mismatched object_id must return FAILED_PRECONDITION, got: {}",
        status.code()
    );
}

// ── 4. Auth denied → PERMISSION_DENIED ──────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_auth_denied_returns_permission_denied() {
    let tv = TestValkey::setup().await;
    // Create the layer with AllowAll, then try to update with DenyAll.
    let allow_svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, object_id) =
        create_session_and_layer(&tv, &allow_svc, "auth-denied-session", "auth-layer").await;

    // Build service with DenyAll for the update attempt.
    let deny_svc = build_service(&tv, Arc::new(DenyAllPermissions));

    let stream = streaming_update(vec![make_update_header(
        &session_id,
        "auth-layer",
        object_id,
    )]);
    let request = make_request(stream);

    let status = handle_update_layer(&deny_svc, request)
        .await
        .expect_err("denied auth must fail");

    assert_eq!(
        status.code(),
        tonic::Code::PermissionDenied,
        "denied auth must return PERMISSION_DENIED, got: {}",
        status.code()
    );
}

// ── 5. Valid update succeeds ────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_valid_update_succeeds() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "valid-update-session", "valid-layer").await;

    let ompf_bytes = delta_to_ompf_bytes(&empty_delta(), &[]);

    let stream = streaming_update(vec![
        make_update_header(&session_id, "valid-layer", object_id),
        make_update_data(ompf_bytes),
    ]);
    let request = make_request(stream);

    let response = handle_update_layer(&svc, request)
        .await
        .expect("valid update must succeed");

    // Handler currently returns placeholder values — just verify it returns Ok.
    let _resp = response.into_inner();
}

// ── 6. Empty update (no chunks) succeeds ────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_empty_content_succeeds() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "empty-update-session", "empty-update-layer").await;

    // Header only, no data chunks.
    let stream = streaming_update(vec![make_update_header(
        &session_id,
        "empty-update-layer",
        object_id,
    )]);
    let request = make_request(stream);

    let response = handle_update_layer(&svc, request)
        .await
        .expect("empty update (header only) must succeed");

    let _resp = response.into_inner();
}

// ── 7. Invalid layer name → INVALID_ARGUMENT ────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_invalid_layer_name_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("invalid-name-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // Path traversal name — prohibited.
    let stream = streaming_update(vec![make_update_header(&session_id, "../../etc", 1)]);
    let request = make_request(stream);

    let status = handle_update_layer(&svc, request)
        .await
        .expect_err("invalid layer name must fail");

    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "invalid layer name must return INVALID_ARGUMENT, got: {}",
        status.code()
    );
}

// ── 8. b.y2t — invalid header rejected BEFORE buffering body ────────────────
//
// DoS amplification regression for UpdateLayer. See the parallel tests in
// create_layer_test.rs for background, including why the service's cap is
// small. We send a body one byte over the test service's `max_layer_size`
// after an invalid header. A correct implementation refuses the header before
// buffering any chunks and therefore never reaches the size cap; a
// buffer-first implementation would buffer until the cap trips and refuse on
// that. Both refusals are INVALID_ARGUMENT (b.75i), so
// `helpers::assert_header_refused_before_buffering` tells them apart by
// message: the header validator's must be present, the size cap's absent.

/// Body one byte over the test service's `max_layer_size`, and well under the
/// decoder's 4 MiB limit.
const OVER_LIMIT_CHUNK_SIZE: usize = SMALL_SIZE_CAP + 1;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_invalid_session_id_rejected_before_body_buffering() {
    let tv = TestValkey::setup().await;
    let svc = build_service_with_size_cap(&tv, Arc::new(AllowAllPermissions), SMALL_SIZE_CAP);

    // Invalid header: empty session_id. Body strictly larger than
    // max_layer_size — an ingest that buffered before validating would
    // refuse on the size cap instead of the header.
    let stream = streaming_update(vec![
        make_update_header("", "some-layer", 1),
        make_update_data(vec![0xAB; OVER_LIMIT_CHUNK_SIZE]),
    ]);
    let request = make_request(stream);

    let status = handle_update_layer(&svc, request)
        .await
        .expect_err("invalid session_id must fail");

    assert_header_refused_before_buffering(&status, "session_id must not be empty");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_zero_object_id_rejected_before_body_buffering() {
    let tv = TestValkey::setup().await;
    let svc = build_service_with_size_cap(&tv, Arc::new(AllowAllPermissions), SMALL_SIZE_CAP);

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("zero-oid-early-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // Invalid header: object_id = 0. Body > max_layer_size.
    let stream = streaming_update(vec![
        make_update_header(&session_id, "some-layer", 0),
        make_update_data(vec![0xAB; OVER_LIMIT_CHUNK_SIZE]),
    ]);
    let request = make_request(stream);

    let status = handle_update_layer(&svc, request)
        .await
        .expect_err("object_id=0 must fail");

    assert_header_refused_before_buffering(&status, "object_id must be non-zero");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_invalid_layer_name_rejected_before_body_buffering() {
    let tv = TestValkey::setup().await;
    let svc = build_service_with_size_cap(&tv, Arc::new(AllowAllPermissions), SMALL_SIZE_CAP);

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("early-reject-update-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // Invalid header: layer_name with path-traversal. Body > max_layer_size.
    let stream = streaming_update(vec![
        make_update_header(&session_id, "../../etc/passwd", 1),
        make_update_data(vec![0xAB; OVER_LIMIT_CHUNK_SIZE]),
    ]);
    let request = make_request(stream);

    let status = handle_update_layer(&svc, request)
        .await
        .expect_err("invalid layer_name must fail");

    assert_header_refused_before_buffering(&status, "..");
}

// ── 11. Stale per-object stream top → INTERNAL, nothing committed (b.ph9) ────

/// The client-visible half of the b.ph9 guard.
///
/// The storage-level contract (commit refused whole, keyspace untouched) is
/// pinned in `storage/update_layer_lua_test.rs`; what this adds is the status
/// code the client actually receives, which is the operationally load-bearing
/// part: the condition cannot be cleared by anything the client does, so a
/// retry-inviting code (`UNAVAILABLE`, `ABORTED`) would turn one wedged layer
/// into a client-driven retry storm, and `FAILED_PRECONDITION` would send the
/// client down the re-resolve-the-layer-mapping recovery path, which cannot
/// help either. `INTERNAL` is the honest answer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_stale_stream_top_returns_internal_and_commits_nothing() {
    use fred::interfaces::{HashesInterface, StreamsInterface};

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "ph9-wedged-session", "ph9-wedged-layer").await;

    let header_key = tv.key_schema().obj_header(&object_id.to_string());
    let seq_before: String = tv
        .client()
        .pool
        .hget(&header_key, "seq")
        .await
        .expect("HGET obj:hdr seq must succeed");
    let seq_before_num: u64 = seq_before.parse().expect("seq must be numeric");

    // Plant an entry above the header — the observable shape of a DR restore
    // seeded below a surviving stream's top, or of an out-of-band writer.
    let _: String = tv
        .client()
        .pool
        .xadd(
            tv.key_schema().layer_stream(object_id),
            false,
            None,
            format!("{}-0", seq_before_num + 5),
            vec![("payload", "out-of-band|planted-by-test")],
        )
        .await
        .expect("planting the out-of-band stream entry must succeed");

    let stream = streaming_update(vec![
        make_update_header(&session_id, "ph9-wedged-layer", object_id),
        make_update_data(delta_to_ompf_bytes(&empty_delta(), &[])),
    ]);
    let status = handle_update_layer(&svc, make_request(stream))
        .await
        .expect_err("a commit whose {new_seq}-0 XADD is already occupied must not report success");

    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "a stream-ID collision must reach the client as INTERNAL — a retriable code would \
         invite a retry storm against a condition only an operator can clear; got: {} — msg: {}",
        status.code(),
        status.message()
    );

    let seq_after: String = tv
        .client()
        .pool
        .hget(&header_key, "seq")
        .await
        .expect("HGET obj:hdr seq must succeed");
    assert_eq!(
        seq_after, seq_before,
        "the refused commit must not have advanced obj:hdr seq — an advanced seq with no \
         matching stream entry is the silent subscriber divergence b.ph9 closes"
    );
}

// ── 12. G3: wrong-length external hash → INVALID_ARGUMENT, no blob stored ────
//
// External blobs are stored by the handler (`store_external_values`), before
// `LayerStore::update_layer` runs and on neither backend's behalf, so "nothing
// stored" for a refused delta can only be observed here. The collector used to
// zero-pad a 31-byte hash and truncate a 33-byte one, storing the blob under a
// hash the client never sent before validation refused the delta; it now
// refuses the hash itself. The storage-level half (no row, no `obj:ext`, no
// stream entry, `seq` unchanged) is `storage/g3_external_hash_tests.rs`.

/// Wrong external-hash lengths: the pre-fix zero-pad and truncate shapes.
const G3_WRONG_HASH_LENS: [usize; 2] = [31, 33];

/// A service on `backend` whose external-value store the test can inspect.
fn service_with_backend_and_ext_store(
    tv: &TestValkey,
    backend: UpdateLayerBackend,
) -> (
    live_edit::grpc::service::LiveEditService,
    Arc<MockExternalValueStore>,
) {
    let base = build_service(tv, Arc::new(AllowAllPermissions));
    let layer_store = Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(backend),
    );
    let ext_store = Arc::new(MockExternalValueStore::new());
    let recovery =
        build_disabled_recovery_hook(Arc::clone(&layer_store), Arc::clone(&base.metrics));
    let svc = live_edit::grpc::service::LiveEditService {
        layer_store,
        recovery,
        ext_value_store: ext_store.clone(),
        ..base
    };
    (svc, ext_store)
}

/// An UpdateLayer body: one external `SetField` naming `hash`, plus its blob.
/// The blob every G3 `UpdateLayer` body carries.
const G3_BLOB: &[u8] = b"g3-external-blob";

/// [`G3_BLOB`]'s content hash (SDD §11.1), so the control commit passes the
/// upload content check.
static G3_BLOB_CONTENT_HASH: std::sync::LazyLock<[u8; 32]> =
    std::sync::LazyLock::new(|| live_edit::storage::external_values::ext_content_hash(G3_BLOB));

fn external_set_field_ompf(hash: Vec<u8>) -> Vec<u8> {
    let blob = G3_BLOB;
    let delta = delta_with_set_field_external(1, "tex", hash, blob.len() as u64, 1, 0);
    delta_to_ompf_bytes(&delta, &[blob])
}

async fn update_layer_wrong_length_hash_stores_no_blob(backend: UpdateLayerBackend) {
    use fred::interfaces::HashesInterface;

    let tv = TestValkey::setup().await;
    let (svc, ext_store) = service_with_backend_and_ext_store(&tv, backend);
    let layer = "g3-update-layer";
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "g3-update-session", layer).await;
    let header_key = tv.key_schema().obj_header(&object_id.to_string());
    let seq_before: String = tv
        .client()
        .pool
        .hget(&header_key, "seq")
        .await
        .expect("HGET obj:hdr seq must succeed");

    for len in G3_WRONG_HASH_LENS {
        let stream = streaming_update(vec![
            make_update_header(&session_id, layer, object_id),
            make_update_data(external_set_field_ompf(vec![0xAB; len])),
        ]);
        let status = handle_update_layer(&svc, make_request(stream))
            .await
            .expect_err("a wrong-length external hash must be refused");

        assert_eq!(
            status.code(),
            tonic::Code::InvalidArgument,
            "{backend:?}, {len}-byte hash: expected INVALID_ARGUMENT, got {:?}: {}",
            status.code(),
            status.message()
        );
        assert!(
            status.message().contains(&format!(
                "invalid external-value hash length: got {len} bytes, expected 32"
            )),
            "{backend:?}, {len}-byte hash: the refusal must name the hash length, got: {}",
            status.message()
        );
        assert_eq!(
            ext_store.entry_count(),
            0,
            "{backend:?}, {len}-byte hash: no external blob may be stored for a refused delta — \
             pre-fix it was stored under a padded / truncated hash"
        );
    }

    let seq_after: String = tv
        .client()
        .pool
        .hget(&header_key, "seq")
        .await
        .expect("HGET obj:hdr seq must succeed");
    assert_eq!(
        seq_after, seq_before,
        "{backend:?}: a refused delta must not commit"
    );

    // Control: the same body with a 32-byte hash commits and stores its blob,
    // so the store is observably wired to this handler and the zero above is
    // not vacuous.
    let stream = streaming_update(vec![
        make_update_header(&session_id, layer, object_id),
        make_update_data(external_set_field_ompf(G3_BLOB_CONTENT_HASH.to_vec())),
    ]);
    handle_update_layer(&svc, make_request(stream))
        .await
        .expect("a 32-byte external hash must commit");
    assert_eq!(
        ext_store.ref_count(&G3_BLOB_CONTENT_HASH),
        Some(1),
        "{backend:?}: the control commit must store its blob under the hash it named"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_wrong_length_external_hash_returns_invalid_argument_and_stores_no_blob() {
    update_layer_wrong_length_hash_stores_no_blob(UpdateLayerBackend::Lua).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_wrong_length_external_hash_returns_invalid_argument_and_stores_no_blob() {
    update_layer_wrong_length_hash_stores_no_blob(UpdateLayerBackend::Txn).await;
}
