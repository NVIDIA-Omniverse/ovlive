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

//! Epic AC #6 — Orphan sweep reconciles ext-value ref counts after
//! recovery without spurious GCs (Subtask `t3.fgj.zc.p6.3k`).
//!
//! Background. SR-3.2 introduced the fetch-on-miss recovery path:
//! `compute_diff_at_cut` / `resolve_external_values` calls
//! `ValkeyExternalValueStore::read_batch`, and on `MetaNotFound` /
//! `DataNotFound` falls back to
//! `CheckpointStore::get_ext_blob(hash) → store_recovered_values(...)`
//! (the store's `store_values` body, sending no mirror copy). The
//! refilled blob's `ref_count` is whatever the per-hash MULTI/EXEC
//! `HINCRBY ext_meta{hash} ref_count 1` wrote — namely `1` per
//! post-recovery rewrite, NOT the true number of live object references
//! before the wipe (SDD `t1.w5b.7g §SR-3.3` deliberately keeps
//! ext-value ref counts OUT of the OMPF snapshot). This drift is
//! reconciled by the orphan sweep: when `cleanup_orphaned_object`
//! retires an orphaned `obj:hdr`, it invokes
//! `schedule_old_object_ext_gc` BEFORE the structural UNLINK so every
//! still-referenced ext-value blob is correctly ref-decremented and
//! orphan-only blobs are released into the GC pending set.
//!
//! Epic AC #6 demands the sweep do this WITHOUT spuriously decrementing
//! `ref_count` for blobs still referenced by a live object. Without
//! this regression test, the per-Subtask test fleet pins the fetch-on-
//! miss path and the singleflight, but no test exercises the orphan
//! sweep tick AFTER recovery. A future regression in
//! `cleanup_orphaned_object`'s ext-GC scheduling, or a refactor that
//! causes the orphan sweep to release every blob it encounters (not
//! just the orphan's), would land silently. This test pins both
//! invariants on a single sweep tick.
//!
//! Test outline:
//! 1. Real `ValkeyExternalValueStore` wired with
//!    `with_checkpoint_store(..)` — built by the harness itself through
//!    [`CheckpointHarnessOptions::ext_value_store_factory`], so the service,
//!    the recovery hook and this test hold one store over the harness's own
//!    `TestValkey`. The real store runs the production per-hash `MULTI { SET
//!    ext_data; HINCRBY ext_meta ref_count 1 } EXEC` writes and the
//!    `schedule_release → process_pending → GC` cycle against real
//!    `le:ext:*` HASH keys. The wiring claim is asserted in
//!    `harness_parity_test.rs`, not re-argued here.
//! 2. Two payloads: `payload_referenced` (live-object reference, the
//!    spurious-GC canary) and `payload_unreferenced` (orphan-only).
//! 3. CreateLayer through the gRPC handler with a `SetField` referring
//!    to `hash_ref`. After the handler returns, `le:ext:meta:{hash_ref}`
//!    has `ref_count=1` and `le:ext:data:{hash_ref}` holds the bytes.
//! 4. Pre-seed `hash_unref` directly into Valkey via the real store's
//!    `store_values` so its `ref_count == 1` too (orphan obj:hdr will
//!    reference it in step 6).
//! 5. Pre-seed BOTH payloads into the LocalDisk adapter via
//!    `harness.store.put_ext_blob(...)`. This is the fetch-on-miss
//!    payload source.
//! 6. `wait_for_checkpoint` confirms the periodic checkpoint loop has
//!    landed a generation for the live object (not strictly required
//!    for the orphan-sweep arm, but matches the broader recovery story
//!    the test guards).
//! 7. Wipe ONLY `le:ext:data:{hash_ref}` + `le:ext:meta:{hash_ref}` —
//!    leave `hash_unref`'s ext-keys and the live object's `le:{oid}:obj:*`
//!    keys intact. This is the partial Valkey-namespace-loss surface
//!    SR-3.2 is designed to defend against.
//! 8. SubscribeReadLayer with `have_object_id=oid, sequence_num=0`
//!    drives `compute_diff_at_cut → resolve_external_values → read_batch` for
//!    `hash_ref`. The miss arms the singleflight, fetches the blob from
//!    the adapter, rewrites it into Valkey via `store_recovered_values`
//!    (`store_values`' script, no mirror copy). Drain one
//!    diff frame and verify the bytes are byte-identical.
//! 9. Assert post-recovery `ref_count == 1` (SR-3.3 drift the orphan
//!    sweep reconciles).
//! 10. Seed an orphaned `obj:hdr` referencing `hash_unref` — a partial
//!     ghost-orphan shape the orphan sweep is designed to clean up.
//!     Mirror the seeding pattern from
//!     `tests/integration/storage/orphan_sweep_test.rs::sweep_cleans_all_sub_keys_of_orphaned_object`.
//! 11. Spawn the orphan sweep against the harness's real
//!     `ValkeyExternalValueStore` and drive one tick via the
//!     `run_one_sweep_iteration` pattern.
//! 12. After the tick + `ext_store.process_pending(&live_edit::storage::external_values::GcPolicy::IMMEDIATE, &mut live_edit::storage::external_values::GcPassState::new())`, verify:
//!     - Orphaned obj:hdr keys are gone.
//!     - Live object's obj:hdr is UNTOUCHED.
//!     - `ref_count` for `hash_ref` is STILL `1` (no spurious decrement).
//!     - `le:ext:data:{hash_ref}` still exists.
//!     - `le:ext:data:{hash_unref}` is gone (orphan-only blob GC'd).
//!     - Cross-canary: `le:ext:data:{hash_ref}` still present.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface, SortedSetsInterface};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use live_edit::grpc::layer::handle_create_layer;
use live_edit::grpc::proto::{subscribe_read_layer_response, SubscribeReadLayerRequest};
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::storage::external_values::{
    ext_content_hash, ExternalValueStore, ValkeyExternalValueStore,
};
use live_edit::storage::keys::{sha256_to_hex, KeySchema};
use live_edit::storage::orphan_sweep::spawn_orphan_sweep;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use tonic::Request;

use super::helpers::*;
use crate::common::owned_delta::{delta_to_ompf_bytes, delta_with_set_field_external};
use crate::grpc::helpers::{make_create_data, make_create_header, make_request, streaming_create};

const SESSION_NAME: &str = "orphan-sweep-reconcile-session";
const LAYER_NAME: &str = "orphan-sweep-reconcile-layer";
const SUBSCRIBER_CLIENT_ID: &str = "orphan-sweep-reconcile-subscriber";

/// Drive `spawn_orphan_sweep` through exactly one sweep tick. Mirror of
/// `tests/integration/storage/orphan_sweep_test.rs::run_one_sweep_iteration` —
/// kept local rather than re-exported because the orphan_sweep_test file
/// is a sibling under `tests/integration/storage/`, not a parent module,
/// and the orphan-sweep contract surface is small enough that an inline
/// helper keeps the test self-contained without expanding the shared
/// helper surface.
async fn run_one_sweep_iteration<F, Fut>(
    handle: JoinHandle<()>,
    cancel: CancellationToken,
    mut done: F,
) where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    // Iteration 1 needs to acquire the leader lock + run the actual sweep.
    // 5 s budget against a local Valkey is generous; CI jitter on the
    // leader-lock SET + the obj:hdr SCAN is well below this ceiling.
    for _ in 0..50 {
        if done().await {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("orphan sweep must exit within 5 s of cancellation")
        .expect("orphan sweep must not panic");
}

/// Construct a SubscribeReadLayer request with the test's stable bearer
/// extension. `have_object_id=object_id` + `have_version=0` lands in the
/// non-brand-new + out-of-window branch (Regime 3): the subscriber's
/// recorded version is below `current_seq` and the path falls through to
/// `compute_diff_at_cut` → `resolve_external_values` → `read_batch` for every
/// external-hash field on the live object — which is the fetch-on-miss
/// surface SR-3.2 instruments.
fn subscribe_request(session_id: &str, object_id: u64) -> Request<SubscribeReadLayerRequest> {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: LAYER_NAME.to_owned(),
        object_id,
        sequence_num: 0,
        client_id: SUBSCRIBER_CLIENT_ID.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

/// Drain the SubscribeReadLayer stream's bootstrap header + chunks until
/// `total_size` bytes are collected, then return both. Mirrors the
/// `recv_message` helper in
/// `tests/integration/checkpoint/e2e_subscribe_read_layer_recovery_test.rs`
/// — kept local for the same self-contained reason as `run_one_sweep_iteration`.
async fn recv_bootstrap_message<S>(
    stream: &mut S,
) -> (live_edit::grpc::proto::SubscribeReadLayerHeader, Vec<u8>)
where
    S: futures::Stream<
            Item = Result<live_edit::grpc::proto::SubscribeReadLayerResponse, tonic::Status>,
        > + Unpin,
{
    use futures::StreamExt;

    let first = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("subscribe stream timed out waiting for header frame")
        .expect("subscribe stream closed unexpectedly before header frame")
        .expect("subscribe stream yielded an error on header frame");

    let header = match first.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => h,
        other => panic!("expected Header frame from subscribe stream, got: {other:?}"),
    };

    let mut buf = Vec::with_capacity(header.total_size as usize);
    while (buf.len() as u64) < header.total_size {
        let next = tokio::time::timeout(Duration::from_secs(10), stream.next())
            .await
            .expect("subscribe stream timed out waiting for chunk frame")
            .expect("subscribe stream closed before total_size reached")
            .expect("subscribe stream yielded an error on chunk frame");
        match next.response_type {
            Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                buf.extend_from_slice(&c.data);
            }
            other => panic!("expected Chunk frame while filling subscribe body, got: {other:?}"),
        }
    }

    (header, buf)
}

/// Seed an orphaned `obj:hdr` (plus the field-index ZSET and the per-field
/// HASH that carries the external-reference annotation) under `orphan_oid`,
/// referencing `hash_unref`. `schedule_old_object_ext_gc` reads the field
/// HASH via `read_object_snapshot`'s unfenced fan-out → `decode_field_value`,
/// where the
/// presence of a `hash` field (vs. `value`) is what marks the field as
/// external. The hex-encoded content hash is the same shape
/// `LayerStore::create_layer`'s commit path writes (b.k1r).
///
/// The orphan is "orphaned" because no `le:{session_id}:layer:{hex(name)}`
/// mapping points at `orphan_oid` — the sweep's
/// `sweep_orphaned_objects` walk reads `header.layer_name`, looks up the
/// mapping, finds it missing, and routes through `cleanup_orphaned_object`.
async fn seed_orphan_with_ext_ref(
    tv: &TestValkey,
    orphan_oid: &str,
    hash_unref_hex: &str,
    section_id: &str,
    field_key_name: &str,
) {
    let ks = tv.key_schema();
    let pool = &tv.client().pool;

    // 1. obj:hdr with a doomed (no mapping) session/layer pair. The
    //    sweep's orphan check reads layer_name, looks up the mapping
    //    `le:{session_id}:layer:{hex(name)}`, and finds it missing.
    let header_key = ks.obj_header(orphan_oid);
    pool.hset::<(), _, _>(
        &header_key,
        [
            ("seq", "1"),
            ("first_ver", "1"),
            ("src_fmt", "usd"),
            ("session_id", "ghost-session-orphan-reconcile"),
            ("layer_name", "ghost-layer-orphan-reconcile"),
        ],
    )
    .await
    .expect("HSET orphan obj:hdr must succeed");

    // 2. Fields index ZSET + per-field HASH carrying the external-ref
    //    annotation. `decode_field_value` keys on the presence of
    //    `hash`/`size` (vs. `value`) to mark the field as external.
    let flds_index = ks.obj_fields_index(orphan_oid);
    let member = KeySchema::field_member(section_id, field_key_name);
    pool.zadd::<(), _, _>(&flds_index, None, None, false, false, (1.0, &member))
        .await
        .expect("ZADD orphan obj:flds index must succeed");

    let field_key = ks.obj_field_from_member(orphan_oid, &member);
    pool.hset::<(), _, _>(
        &field_key,
        [
            ("hash", hash_unref_hex.to_owned()),
            // `size` is read by the retirement's reference walk
            // (`storage::retirement`) but unused by it (only the `hash`
            // matters for the release). Any value > 0 is fine; matching the
            // unref payload's length keeps the seeded shape honest.
            ("size", "9".to_owned()),
            // version annotations: `last_version=1` is the value the
            // orphan-sweep cleanup will read but does not act on.
            ("last_version", "1".to_owned()),
            ("set_order", "1".to_owned()),
        ],
    )
    .await
    .expect("HSET orphan per-field HASH must succeed");

    // 3. Sections index + section HASH so the orphan holds a
    //    structurally-consistent shape for the sweep's teardown to page. A
    //    section with `name=root`,
    //    `type=1` matches the seeding pattern in
    //    `orphan_sweep_test.rs::sweep_cleans_all_sub_keys_of_orphaned_object`.
    let secs_index = ks.obj_sections_index(orphan_oid);
    pool.sadd::<(), _, _>(&secs_index, section_id)
        .await
        .expect("SADD orphan sections index must succeed");
    let sec_key = ks.obj_section(orphan_oid, section_id);
    pool.hset::<(), _, _>(
        &sec_key,
        [("parent_id", "0"), ("name", "root"), ("type", "1")],
    )
    .await
    .expect("HSET orphan section HASH must succeed");
}

/// Read the live `ref_count` field on `le:ext:meta:{hash_hex}` as a
/// `u64`. Panics with a diagnostic if the field is missing or non-numeric
/// — the contract under test is `ref_count == 1`, so an absent field is a
/// regression in the recovery rewrite (SR-3.2) and a non-numeric value is
/// a corruption that would mask a real assertion failure.
async fn read_ref_count(tv: &TestValkey, hash_hex: &str) -> u64 {
    let ks = tv.key_schema();
    let meta_key = ks.ext_meta(hash_hex);
    let raw: Option<String> = tv
        .client()
        .pool
        .hget(&meta_key, "ref_count")
        .await
        .expect("HGET ref_count must succeed against a live Valkey");
    let raw = raw.unwrap_or_else(|| {
        panic!(
            "le:ext:meta:{{{hash_hex}}}/ref_count must be present after recovery; \
             the SR-3.2 rewrite via store_recovered_values is responsible for setting it to 1"
        )
    });
    raw.parse::<u64>().unwrap_or_else(|e| {
        panic!("le:ext:meta:{{{hash_hex}}}/ref_count must be a u64, got {raw:?}: {e}")
    })
}

// =============================================================================
// Epic AC #6 — Orphan sweep reconciles ext-value ref counts after recovery
//              without spurious GCs for blobs still referenced by live objects.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn orphan_sweep_reconciles_ext_ref_counts_after_recovery_without_spurious_gc() {
    // ── 1. Real ValkeyExternalValueStore, built BY the harness ───────────
    //
    // `ext_value_store_factory` runs inside harness construction, after the
    // harness's `TestValkey` / checkpoint store / metrics exist and before
    // anything is wired, so the store it returns is the ONE the service, the
    // recovery hook and this test all hold — and it is built against the
    // harness's own `TestValkey`, which is what matters here: that
    // `TestValkey` carries the key prefix and owns the `Drop`-time cleanup
    // contract, so a store built against a second one would both miss the
    // harness's keyspace and race its prefix-wipe at tear-down.
    // `with_checkpoint_store(..)` is chained here (it is not a harness
    // default) because this test's subject includes the SR-3.2 fetch-on-miss
    // path. The wiring itself is asserted in `harness_parity_test.rs`.
    //
    // The typed `Arc<ValkeyExternalValueStore>` the sweep drive and
    // `process_pending()` need is stashed out of the closure through a
    // `OnceLock`; the closure returns the erased handle the harness installs.
    let typed_store: Arc<OnceLock<Arc<ValkeyExternalValueStore>>> = Arc::new(OnceLock::new());
    let stash = Arc::clone(&typed_store);
    let mut opts = CheckpointHarnessOptions::new(1);
    opts.ext_value_store_factory = Some(Box::new(move |ctx: ExtStoreCtx<'_>| {
        let store = Arc::new(
            ValkeyExternalValueStore::new(ctx.tv.client().clone(), ctx.tv.key_schema().clone())
                .with_metrics(Arc::clone(ctx.metrics))
                .with_checkpoint_store(Arc::clone(ctx.checkpoint_store)),
        );
        let _ = stash.set(Arc::clone(&store));
        store as Arc<dyn ExternalValueStore>
    }));
    let harness = setup_checkpoint_harness_with_options(opts).await;
    let real_ext_store = Arc::clone(
        typed_store
            .get()
            .expect("the ext-store factory runs during harness construction"),
    );

    // ── 2. Distinct payloads + hashes for the live-ref and orphan-ref ────
    let payload_referenced: &[u8] = b"payload-referenced-ac6";
    let payload_unreferenced: &[u8] = b"payload-orphan-only-ac6";
    // Content hashes: `hash_ref` reaches the CreateLayer upload content check,
    // which refuses any other claim.
    let hash_ref = ext_content_hash(payload_referenced);
    let hash_unref = ext_content_hash(payload_unreferenced);
    let hash_ref_hex = sha256_to_hex(&hash_ref);
    let hash_unref_hex = sha256_to_hex(&hash_unref);

    // ── 3. CreateSession + CreateLayer with a SetField referencing hash_ref ─
    //
    // The session is created via SessionStore directly (matches the
    // pattern in `e2e_subscribe_read_layer_recovery_test.rs`). The
    // CreateLayer gRPC handler runs the production
    // `store_external_values` path which calls
    // `real_ext_store.store_values(...)` — that's how `hash_ref`'s
    // `ref_count=1` lands.
    let session_store =
        SessionStore::new(harness.tv.client().clone(), harness.tv.key_schema().clone());
    let session_id = session_store
        .create(SESSION_NAME, "/stage", "test-user-ac6")
        .await
        .expect("session creation must succeed");

    let delta = delta_with_set_field_external(
        /*section_id=*/ 1,
        /*key=*/ "f",
        /*hash=*/ hash_ref.to_vec(),
        /*ext_value_size=*/ payload_referenced.len() as u64,
        // 1-based: index 1 = the first (and only) external blob in the OMPF.
        /*ext_value_index=*/
        1,
        /*set_order=*/ 1,
    );
    let ompf_bytes = delta_to_ompf_bytes(&delta, &[payload_referenced]);

    let create_stream = streaming_create(vec![
        make_create_header(&session_id, LAYER_NAME),
        make_create_data(ompf_bytes),
    ]);
    let create_response = handle_create_layer(&harness.svc, make_request(create_stream))
        .await
        .expect("CreateLayer with ext-value reference must succeed");
    let live_object_id = create_response.into_inner().object_id;

    // Sanity: ref_count for hash_ref is now 1 (CreateLayer's
    // store_values committed the per-hash MULTI/EXEC).
    let ref_count_pre_wipe = read_ref_count(&harness.tv, &hash_ref_hex).await;
    assert_eq!(
        ref_count_pre_wipe, 1,
        "post-CreateLayer ref_count for the live-referenced hash must be 1 \
         (the store_values MULTI/EXEC seeds ref_count=1); got {ref_count_pre_wipe}"
    );

    // ── 4. Pre-seed payload_unreferenced into Valkey via store_values ─────
    //
    // No live object references this blob — `ref_count=1` is the state a
    // pre-existing live reference would have had before its obj:hdr was
    // orphaned (e.g., a CreateLayer overwrite that landed the new
    // mapping at a different `object_id` but left the old obj:hdr in
    // place for the orphan sweep to clean up). Calling `store_values`
    // here matches the production write path exactly.
    real_ext_store
        .store_values(&[(&hash_unref, payload_unreferenced)])
        .await
        .expect("store_values for the orphan-only payload must succeed");

    let ref_count_unref_pre_sweep = read_ref_count(&harness.tv, &hash_unref_hex).await;
    assert_eq!(
        ref_count_unref_pre_sweep, 1,
        "post-seed ref_count for the orphan-only hash must be 1; got \
         {ref_count_unref_pre_sweep}"
    );

    // ── 5. Pre-seed BOTH payloads into the LocalDisk adapter ──────────────
    //
    // `put_ext_blob` is the fetch-on-miss payload source. Pre-seeding
    // both ensures the SR-3.2 path can recover `hash_ref` after the
    // wipe; pre-seeding `hash_unref` is defensive (it is never fetched
    // via SR-3.2 in this test, but a future regression that broadens
    // recovery should not silently pass because the adapter happened to
    // be empty).
    for (hash_hex, payload) in [
        (&hash_ref_hex, payload_referenced),
        (&hash_unref_hex, payload_unreferenced),
    ] {
        harness
            .store
            .put_ext_blob(hash_hex, Bytes::copy_from_slice(payload))
            .await
            .expect("put_ext_blob into LocalDisk adapter must succeed");
    }

    // ── 6. Wait for the checkpoint loop to land at least one generation ──
    //
    // Not strictly required for the orphan-sweep arm of the test (the
    // sweep operates against Valkey state, not the checkpoint store),
    // but pinning this here means a future regression in the loop
    // surfaces as a clear deadline failure rather than as confusing
    // downstream symptoms. 10 s matches the sibling e2e tests' deadline.
    let _handle = wait_for_checkpoint(&harness.store, live_object_id, 10).await;

    // ── 7. Wipe ONLY hash_ref's ext-keys (the SR-3.2 recovery surface) ────
    //
    // UNLINK both `le:ext:meta:{hash_ref}` and `le:ext:data:{hash_ref}`
    // by name — do NOT use a wildcard that would catch the GC sets, the
    // orphan obj:hdr's keys, or `hash_unref`'s ext-keys. The two keys
    // share the `{hash_hex}` hash tag, so a multi-key UNLINK is slot-safe
    // even in Cluster mode.
    let meta_key_ref = harness.tv.key_schema().ext_meta(&hash_ref_hex);
    let data_key_ref = harness.tv.key_schema().ext_data(&hash_ref_hex);
    let unlinked: u64 = harness
        .tv
        .client()
        .pool
        .unlink(vec![meta_key_ref.clone(), data_key_ref.clone()])
        .await
        .expect("UNLINK of le:ext:meta/le:ext:data for hash_ref must succeed");
    assert_eq!(
        unlinked, 2,
        "UNLINK must report exactly 2 keys removed (meta + data); got {unlinked}"
    );

    // Sanity: the orphan-only blob's ext-keys are intact.
    let unref_meta_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(harness.tv.key_schema().ext_meta(&hash_unref_hex))
        .await
        .expect("EXISTS le:ext:meta:{hash_unref} must succeed");
    assert!(
        unref_meta_exists,
        "pre-sweep wipe must leave the orphan-only hash's ext_meta intact"
    );

    // ── 8. SubscribeReadLayer drives the fetch-on-miss for hash_ref ───────
    //
    // `have_object_id=live_object_id, sequence_num=0` lands in the
    // out-of-window catch-up branch. The coalesced diff emits every
    // section + field (including the ext-ref field), which routes through
    // `resolve_external_values → read_batch`. `read_batch` returns
    // `MetaNotFound` on the freshly-wiped `hash_ref`, which arms the
    // SR-3.2 fallback: `try_recover_via_checkpoint` fetches the blob
    // from the LocalDisk adapter, rewrites it via `store_recovered_values`
    // (`store_values`' script without the mirror copy; re-creating both `le:ext:meta:{hash_ref}` and
    // `le:ext:data:{hash_ref}` with `ref_count=1`), then returns the
    // bytes inline. The subscriber observes them in the OMPF's external-
    // values part.
    let sub_request = subscribe_request(&session_id, live_object_id);
    let response = handle_subscribe_read_layer(&harness.svc, sub_request)
        .await
        .expect("SubscribeReadLayer must succeed after the ext-namespace wipe");
    let mut stream = response.into_inner();

    let (bootstrap_header, bootstrap_body) = recv_bootstrap_message(&mut stream).await;
    assert_eq!(
        bootstrap_header.object_id, live_object_id,
        "bootstrap header must carry the live object_id"
    );
    assert!(
        !bootstrap_body.is_empty(),
        "bootstrap body must be non-empty: the coalesced diff includes the \
         ext-ref field"
    );

    // Decode the OMPF and find the recovered payload as the first
    // external-value part. The OMPF container's `external_values` slice
    // is the same `&[&[u8]]` the engineer wrote during CreateLayer; on
    // recovery the bytes round-trip via `resolve_external_values` →
    // `read_batch` → `try_recover_via_checkpoint` → adapter.
    let container = live_edit::domain::ompf::OmpfContainer::parse(&bootstrap_body)
        .expect("bootstrap OMPF must parse");
    assert!(
        !container.external_values.is_empty(),
        "bootstrap OMPF must carry at least one external-value part \
         (the recovered hash_ref blob); got {} parts",
        container.external_values.len()
    );
    let recovered_bytes = container.external_values[0];
    assert_eq!(
        recovered_bytes, payload_referenced,
        "the recovered ext-value bytes must be byte-identical to the original \
         payload_referenced (SR-3.2 fetch-on-miss must round-trip the adapter \
         payload exactly)"
    );

    // Drop the subscribe stream BEFORE the orphan sweep starts — the
    // bidirectional consumer task holds a reference into Valkey state
    // and we do not want it racing the sweep.
    drop(stream);

    // ── 9. Post-recovery ref_count assertion ──────────────────────────────
    //
    // SR-3.3 drift: the recovery rewrite via `store_recovered_values` sets
    // `ref_count=1` regardless of how many live objects reference the
    // blob. With one live reference, this is "coincidentally" correct;
    // the orphan sweep's reconcile step must NOT decrement it further.
    let ref_count_post_recovery = read_ref_count(&harness.tv, &hash_ref_hex).await;
    assert_eq!(
        ref_count_post_recovery, 1,
        "post-recovery ref_count for the live-referenced hash must be 1 \
         (store_recovered_values' HINCRBY ref_count 1 on the rewrite); got \
         {ref_count_post_recovery}"
    );

    // ── 10. Seed an orphaned obj:hdr referencing hash_unref ───────────────
    let orphan_oid = "987654321";
    seed_orphan_with_ext_ref(
        &harness.tv,
        orphan_oid,
        &hash_unref_hex,
        /*section_id=*/ "10",
        /*field_key_name=*/ "orphan-field",
    )
    .await;

    // Sanity: the orphaned obj:hdr exists and the live object's obj:hdr
    // is also still present.
    let orphan_hdr_key = harness.tv.key_schema().obj_header(orphan_oid);
    let live_hdr_key = harness
        .tv
        .key_schema()
        .obj_header(&live_object_id.to_string());
    assert!(
        harness
            .tv
            .client()
            .pool
            .exists::<bool, _>(&orphan_hdr_key)
            .await
            .expect("EXISTS orphan obj:hdr must succeed"),
        "orphaned obj:hdr must be present pre-sweep"
    );
    assert!(
        harness
            .tv
            .client()
            .pool
            .exists::<bool, _>(&live_hdr_key)
            .await
            .expect("EXISTS live obj:hdr must succeed"),
        "live obj:hdr must be present pre-sweep"
    );

    // ── 11. Spawn the orphan sweep and drive one tick ─────────────────────
    //
    // With a zero grace: the live object is young, and the sweep's grace on
    // young objects would otherwise keep it however it was classified, so
    // "no spurious GC" would hold vacuously.
    harness
        .tv
        .client()
        .set_orphan_sweep_grace_for_test(std::time::Duration::ZERO);
    let sweep_cancel = CancellationToken::new();
    let sweep_handle = spawn_orphan_sweep(
        harness.tv.client().clone(),
        harness.tv.key_schema().clone(),
        Arc::clone(&real_ext_store) as Arc<dyn ExternalValueStore>,
        Arc::new(live_edit::storage::layer::LayerStore::new(
            harness.tv.client().clone(),
            harness.tv.key_schema().clone(),
            0,
        )),
        crate::common::test_leader("test-replica-ac6"),
        sweep_cancel.clone(),
        Some(Arc::clone(&harness.metrics)),
    );

    // Poll until the orphan obj:hdr is gone.
    let orphan_hdr_for_poll = orphan_hdr_key.clone();
    let poll_pool = harness.tv.client().pool.clone();
    run_one_sweep_iteration(sweep_handle, sweep_cancel, move || {
        let key = orphan_hdr_for_poll.clone();
        let pool = poll_pool.clone();
        async move {
            !pool
                .exists::<bool, _>(&key)
                .await
                .expect("EXISTS poll on orphan obj:hdr must succeed")
        }
    })
    .await;

    // ── 12. Assertions after the sweep tick + process_pending() ───────────
    //
    // The sweep schedules `hash_unref` for release (HINCRBY ref_count -1
    // queued into the ext_pending set). `process_pending()` is what
    // actually decrements the counter and deletes the blob when it
    // reaches zero. In production this is driven by the GC sweep loop;
    // here we call it directly so the test is independent of the GC
    // loop's tick cadence.
    let gc_result = real_ext_store
        .process_pending(
            &live_edit::storage::external_values::GcPolicy::IMMEDIATE,
            &mut live_edit::storage::external_values::GcPassState::new(),
        )
        .await
        .expect("process_pending must succeed");
    assert!(
        gc_result.deleted >= 1,
        "process_pending must delete at least one ext blob (the orphan-only \
         hash_unref); got deleted={}, processed={}",
        gc_result.deleted,
        gc_result.processed
    );

    // (a) Orphaned obj:hdr is gone (the sweep's `cleanup_orphaned_object`
    //     UNLINKed all of its structural keys).
    assert!(
        !harness
            .tv
            .client()
            .pool
            .exists::<bool, _>(&orphan_hdr_key)
            .await
            .expect("EXISTS post-sweep orphan obj:hdr must succeed"),
        "orphaned obj:hdr must be UNLINKed by the sweep"
    );

    // (b) Live object's obj:hdr is UNTOUCHED.
    assert!(
        harness
            .tv
            .client()
            .pool
            .exists::<bool, _>(&live_hdr_key)
            .await
            .expect("EXISTS post-sweep live obj:hdr must succeed"),
        "live object's obj:hdr must survive the orphan sweep — the sweep \
         must only retire entries whose layer mapping is gone"
    );

    // (c) ref_count for hash_ref is STILL 1 (no spurious decrement). The
    //     orphan sweep's `schedule_old_object_ext_gc` only collects
    //     hashes from the ORPHAN's own state; `hash_ref` lives on the
    //     LIVE object and must not be touched.
    let ref_count_post_sweep = read_ref_count(&harness.tv, &hash_ref_hex).await;
    assert_eq!(
        ref_count_post_sweep, 1,
        "post-sweep ref_count for the live-referenced hash must STILL be 1 \
         (no spurious decrement); got {ref_count_post_sweep}. A regression \
         here means `cleanup_orphaned_object` started releasing every blob \
         it encounters rather than only the orphan's own hashes."
    );

    // (d) `le:ext:data:{hash_ref}` still exists.
    let live_data_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(harness.tv.key_schema().ext_data(&hash_ref_hex))
        .await
        .expect("EXISTS post-sweep le:ext:data:{hash_ref} must succeed");
    assert!(
        live_data_exists,
        "live-referenced ext_data must survive the orphan sweep + GC tick"
    );

    // (e) After the GC tick, `le:ext:data:{hash_unref}` is GONE
    //     (orphan-only blob correctly GC'd via the
    //     schedule_release → process_pending chain).
    let orphan_data_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(harness.tv.key_schema().ext_data(&hash_unref_hex))
        .await
        .expect("EXISTS post-sweep le:ext:data:{hash_unref} must succeed");
    assert!(
        !orphan_data_exists,
        "orphan-only ext_data must be DELETED after the sweep + process_pending \
         (the SR-3.3 reconcile: orphan release → ref_count 0 → DEL meta + data)"
    );

    // (f) Cross-canary: confirm `le:ext:data:{hash_ref}` is STILL present
    //     after the GC tick. Re-checking after `process_pending` guards
    //     against a regression where the GC pass would walk every meta
    //     key rather than only the entries in `ext_pending`.
    let live_data_still_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(harness.tv.key_schema().ext_data(&hash_ref_hex))
        .await
        .expect("EXISTS final-check le:ext:data:{hash_ref} must succeed");
    assert!(
        live_data_still_exists,
        "live-referenced ext_data must STILL be present after process_pending \
         — a regression here means process_pending broadened its delete scope \
         beyond the ext_pending set"
    );

    // ── 13. Bounded harness drain ────────────────────────────────────────
    harness.cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), harness.loop_handle).await;
}
