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

//! End-to-end coverage for the b.b2j object-id counter repair, driven through
//! the real gRPC handler against the fully-wired [`CheckpointHarness`] and
//! [`StorageApiCheckpointHarness`].
//!
//! ## Why this exists alongside `counter_repair_test`
//!
//! That module drives `LayerStore::create_layer` directly against a
//! hand-assembled store. It proves the *allocation* contract thoroughly and
//! cheaply, but it proves nothing about the **wiring**: every one of its tests
//! constructs the repair hook itself, so all of them would keep passing if
//! `main.rs` stopped installing one.
//!
//! This module closes that gap from the other side. It uses the shared harness
//! — a real `LocalDiskCheckpointStore`, a real `LiveEditService`, the real
//! `handle_create_layer` entry point — and asserts the hook is present by
//! observing behaviour only a hooked `LayerStore` can produce: after a full
//! namespace wipe, the next `CreateLayer` mints an id **above** the pre-wipe
//! line rather than restarting at `1`.
//!
//! The harness mirrors `main.rs` Step 1b-iii.6 exactly (see
//! `object_id_counter_repair_for_harness` in `helpers.rs`): the hook is
//! installed because the harness's adapter is non-`Disabled`, with the same
//! `checkpoint_counter_safety_margin` and the same
//! `checkpoint_recovery_fetch_budget` / `object_id_counter_repair_wait_budget`
//! pair the binary derives.
//!
//! ## The two hops this test composes
//!
//! `full_valkey_wipe` removes EVERY `le:*` key under the fixture prefix, so the
//! post-wipe `CreateLayer` needs both recoveries in series:
//!
//! 1. **Session recovery** — `SessionBasePathCache::get_base_path` funnels the
//!    absent `le:{sid}:sess` into `SessionRecoveryHook`, which re-materializes
//!    session-state from the session-checkpoint blob on disk. Without this the
//!    handler would return `NotFound` before ever reaching an allocation.
//! 2. **Counter repair** — `LayerStore::allocate_object_id` finds
//!    `le:counter:object_id` gone, calls the hook, and the hook re-seeds from
//!    the counters sidecar.
//!
//! Only the second is under test; the first is scaffolding, and its own
//! contract is pinned by `e2e_session_full_wipe_recovery_test`.
//!
//! ## Why the counters sidecar is written by hand
//!
//! The harness pushes the checkpoint loop's `counters_sync` cadence out to an
//! hour so the sidecar timer never fires mid-test. That is deliberate — a timer
//! racing the assertions would make this test's arithmetic nondeterministic —
//! so the test writes the sidecar itself at the point the periodic tick would
//! have. Modelling the tick keeps the test pointed at the id-reuse window
//! rather than at an RPO gap: with a stale sidecar the repair would legitimately
//! seed from an older floor, which is the counters-sync loop's problem, not this
//! ticket's.
//!
//! ## Two harnesses, two failure surfaces
//!
//! 1. [`full_wipe_then_create_layer_mints_above_the_pre_wipe_id_through_the_hook`]
//!    — the LocalDisk [`CheckpointHarness`], full wipe, both recovery hops.
//! 2. [`storage_api_harness_create_layer_repairs_the_counter_through_the_hook`]
//!    — the same wiring claim against [`StorageApiCheckpointHarness`], whose
//!    `LayerStore` had neither `.with_metrics(...)` nor
//!    `.with_object_id_counter_repair(...)` under a comment claiming parity with
//!    the LocalDisk harness. That gap mattered more there than anywhere else:
//!    StorageApi is the adapter whose checkpoint blobs live *outside* the pod,
//!    so an unhooked allocator re-mints ids that survive independently of the
//!    Valkey the wipe destroyed.
//!
//! The storage-api test does not wipe. It does not need to: the counter is
//! absent on a fresh fixture prefix, which is the identical allocator input, and
//! the storage-api harness deliberately wires no session-recovery hook into its
//! base-path cache — so a full wipe there would fail at session resolution
//! before ever reaching an allocation, testing the scaffolding rather than the
//! subject. (The wipe-shaped composition is what test 1 is for; the
//! `full_valkey_wipe` + session-recovery hop is pinned once, on the harness that
//! wires it.)
//!
//! ## Shared-Valkey discipline
//!
//! `full_valkey_wipe` is a SCAN + UNLINK over `tv.prefix()*`. **No `FLUSHALL` /
//! `FLUSHDB`** — the backing instance is shared with concurrently-running
//! agents and CI jobs. From this service's point of view the namespaced wipe is
//! an exact model of the production shape anyway: every key it can name is gone.

use bytes::Bytes;
use fred::interfaces::KeysInterface;

use live_edit::checkpoint::counters::CountersSidecar;
use live_edit::checkpoint::CheckpointStore;
use live_edit::grpc::layer::handle_create_layer;
use live_edit::grpc::proto::CreateSessionRequest;
use live_edit::grpc::session::handle_create_session;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::OwnedDelta;
use std::sync::Arc;

use super::helpers::*;
use crate::common::owned_delta::delta_to_ompf_bytes;

const SESSION_NAME: &str = "e2e-counter-repair-session";
const LAYER_NAME_PRE: &str = "e2e-counter-repair-layer-pre";
const LAYER_NAME_POST: &str = "e2e-counter-repair-layer-post";
const SESSION_BASE_PATH: &str = "/stage/counter-repair";

/// The harness's `checkpoint_counter_safety_margin`, from
/// `common::test_config()`. Named here so the post-wipe id arithmetic below
/// reads as the contract rather than as a magic number.
const SAFETY_MARGIN: u64 = 1000;

/// Smallest delta `CreateLayer` accepts. These assertions are about which
/// `object_id` is minted, not about materialized content, so an empty payload
/// keeps the fixture honest about what is under test.
fn empty_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        base_version: 0,
        ..Default::default()
    }
}

/// `CreateLayer` through the real handler. Returns the minted `object_id`.
///
/// Takes the `LiveEditService` rather than a harness so both the LocalDisk and
/// the storage-api harness drive the identical handler entry point — the whole
/// claim under test is that the two are wired the same, so the call site must
/// not be able to differ.
async fn create_layer(
    svc: &live_edit::grpc::service::LiveEditService,
    session_id: &str,
    layer_name: &str,
) -> u64 {
    let ompf = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(session_id, layer_name),
        make_create_data(ompf),
    ]);
    handle_create_layer(svc, make_streaming_token_request(stream))
        .await
        .expect("CreateLayer must succeed")
        .into_inner()
        .object_id
}

/// Persist a counters sidecar through a harness's real checkpoint store,
/// round-tripping the shared `CountersSidecar` struct rather than hand-rolled
/// JSON — the repair deserializes with the same type, so this pins the actual
/// producer/consumer contract.
async fn put_counters_sidecar(store: &Arc<dyn CheckpointStore>, max_object_id_observed: u64) {
    let sidecar = CountersSidecar {
        max_object_id_observed,
        updated_at: "2026-08-27T12:00:00Z".to_string(),
    };
    let bytes = serde_json::to_vec(&sidecar).expect("CountersSidecar serialization must succeed");
    store
        .put_counters_sidecar(Bytes::from(bytes))
        .await
        .expect("put_counters_sidecar must succeed against the harness adapter");
}

/// Whether `le:counter:object_id` exists in the harness namespace.
async fn counter_key_exists(tv: &TestValkey) -> bool {
    let key = tv.key_schema().counter_object_id();
    tv.client()
        .pool
        .exists(key.clone())
        .await
        .unwrap_or_else(|e| panic!("EXISTS {key} must succeed: {e}"))
}

// =============================================================================
// The wiring test.
// =============================================================================

/// A full namespace wipe under a surviving process, followed by a `CreateLayer`
/// through the real gRPC handler, must mint an `object_id` strictly above every
/// pre-wipe id — with `unhooked_initialized` untouched.
///
/// **`unhooked_initialized == 0` is the load-bearing assertion**, not the id
/// comparison. Both a hooked and an unhooked `LayerStore` return `Ok` here; they
/// differ only in *which* id they hand back and which cell they tick. An
/// unhooked store would establish the counter at `0` and mint `1` — silently
/// re-issuing an id the surviving LocalDisk checkpoint blobs still claim. So
/// the test asserts both: the id (the user-visible consequence) and the cell
/// (the unambiguous witness of which arm ran).
///
/// **Fails without the harness wiring.** Before
/// `.with_object_id_counter_repair(...)` was added to `helpers.rs`, the
/// post-wipe id here is `1` and `unhooked_initialized` is `1` — which is what
/// makes this test, rather than any amount of direct-`LayerStore` coverage, the
/// thing standing between a dropped builder call and a production regression.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_wipe_then_create_layer_mints_above_the_pre_wipe_id_through_the_hook() {
    let harness = setup_checkpoint_harness(1).await;

    // Parity guard up front: the harness must be the shape production wires the
    // hook for. Asserting it before any traffic means a harness regression is
    // reported as itself rather than as a confusing id mismatch downstream.
    assert_harness_carries_object_id_counter_repair(&harness);

    // ── Session, plus the session-checkpoint blob the post-wipe recovery will
    // rebuild from. Without the wait the wipe would race the fire-and-forget
    // writer and the post-wipe CreateLayer would fail at session recovery,
    // never reaching the allocation this test is about.
    let session_id = handle_create_session(
        &harness.svc,
        make_unary_token_request(CreateSessionRequest {
            path: SESSION_BASE_PATH.to_owned(),
            name: SESSION_NAME.to_owned(),
        }),
    )
    .await
    .expect("CreateSession must succeed against an AllowAllPermissions harness")
    .into_inner()
    .session_id;
    wait_for_session_checkpoint(&harness.store, &session_id, 10).await;

    // ── Pre-wipe CreateLayer. This is the id line a post-wipe collision would
    // land on.
    let oid_pre = create_layer(&harness.svc, &session_id, LAYER_NAME_PRE).await;
    assert!(
        oid_pre > 0,
        "CreateLayer must mint a non-zero object_id; got {oid_pre}"
    );
    // CreateLayer mutates the layer-mappings table, so it enqueues its own
    // session-checkpoint write. Drain it before the wipe: a write still in
    // flight would land AFTER the wipe and resurrect a partial blob, which
    // would make the session-recovery hop's behaviour nondeterministic.
    wait_for_session_checkpoint_quiesced(&harness, &session_id, 10).await;
    let _generation = wait_for_checkpoint(&harness.store, oid_pre, 10).await;

    // ── The periodic counters-sync tick, modelled by hand (see module docs).
    put_counters_sidecar(&harness.store, oid_pre).await;

    // ── The keyspace loss. Namespaced wipe — never FLUSHALL.
    full_valkey_wipe(&harness.tv).await;
    assert!(
        !counter_key_exists(&harness.tv).await,
        "fixture precondition: the wipe must have removed le:counter:object_id — that \
         absence is the whole signal the repair path keys off"
    );
    // The pre-wipe calls populated the in-process base-path cache, whose
    // positive TTL outlasts this test. Without invalidation the post-wipe
    // handler would HIT the cache, never consult `session_store.get_base_path`,
    // and session recovery would silently not fire.
    harness.svc.session_base_path_cache.invalidate(&session_id);

    // ── The first post-wipe CreateLayer, through the real handler.
    let oid_post = create_layer(&harness.svc, &session_id, LAYER_NAME_POST).await;

    assert_eq!(
        oid_post,
        oid_pre + SAFETY_MARGIN + 1,
        "the post-wipe allocation must resume above the sidecar-derived floor \
         (max_object_id_observed={oid_pre} + safety_margin={SAFETY_MARGIN}), so the first id \
         off it is {}; got {oid_post}. A value of 1 is the b.b2j bug verbatim — a counter \
         re-established at zero, re-minting ids the surviving LocalDisk checkpoint blobs \
         still claim",
        oid_pre + SAFETY_MARGIN + 1
    );
    assert!(
        oid_post > oid_pre,
        "and it must be strictly above every pre-wipe id ({oid_pre}); a collision lets \
         get_latest_layer_checkpoint hand one layer's blob to another layer's identity on \
         the DR path"
    );

    assert_eq!(
        object_id_counter_repair_metric(&harness, "unhooked_initialized"),
        0,
        "THE assertion of this test: the hooked path must be what ran. This cell increments \
         only on `allocate_object_id`'s no-hook arm, so a tick here means the harness \
         LayerStore lost `.with_object_id_counter_repair(...)` and the id above came from \
         an unhooked establish-at-zero that happened to look right"
    );
    assert_eq!(
        object_id_counter_repair_metric(&harness, "repaired_seeded"),
        1,
        "exactly one repair must have re-seeded from the counters sidecar. Every increment \
         of this cell is evidence the Valkey keyspace was lost under a running pod — it is \
         the operator's primary signal for this failure shape"
    );
    assert_eq!(
        object_id_counter_repair_metric(&harness, "failed")
            + object_id_counter_repair_metric(&harness, "failed_after_repair")
            + object_id_counter_repair_metric(&harness, "wait_timeout"),
        0,
        "nothing may have failed or timed out against a healthy LocalDisk adapter — a tick \
         on any of these three would mean the harness's budgets (derived from the same \
         `Config` methods production uses) are too tight for a local-disk read"
    );

    // Re-run the guard at the end: it is cheap, and the interesting moment for
    // it is AFTER a counter-absent event has actually been served.
    assert_harness_carries_object_id_counter_repair(&harness);

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// =============================================================================
// The storage-api wiring test.
// =============================================================================

const STORAGE_API_SESSION_NAME: &str = "e2e-counter-repair-storage-api-session";
const STORAGE_API_LAYER_NAME: &str = "e2e-counter-repair-storage-api-layer";
const STORAGE_API_SESSION_BASE_PATH: &str = "/stage/counter-repair-storage-api";
const STORAGE_API_CREATED_BY: &str = "b2j-e2e-test-user";

/// The sidecar's `max_object_id_observed` for the storage-api test. Any value
/// works; a large one keeps the expected id (`5000 + 1000 + 1`) unmistakably
/// distinct from the `1` an unhooked allocator hands out.
const STORAGE_API_SIDECAR_MAX_OBJECT_ID: u64 = 5000;

/// A `CreateLayer` against the [`StorageApiCheckpointHarness`], with the counter
/// absent and a counters sidecar present in the storage-api-backed checkpoint
/// store, must mint off the **sidecar-derived floor** — not from `1`.
///
/// This is the storage-api half of the wiring claim, and it is the half that was
/// actually broken: `setup_storage_api_checkpoint_harness` built its
/// `LayerStore` with neither `.with_metrics(...)` nor
/// `.with_object_id_counter_repair(...)` beneath a comment reading "same wiring
/// as the LocalDisk harness". Two consequences, both silent:
///
/// - Every `CreateLayer` in the storage-api e2e took `allocate_object_id`'s
///   UNHOOKED arm, which establishes `le:counter:object_id` at `0` and mints
///   `1` — the b.b2j bug — on the one adapter whose checkpoint blobs live
///   outside the pod and therefore survive the Valkey that lost the counter.
/// - Without `.with_metrics(...)`, the `unhooked_initialized` cell that would
///   have exposed it could not tick at all. The missing metrics wiring is what
///   made the missing hook invisible, which is why
///   [`assert_harness_carries_object_id_counter_repair`] reads a *cell* rather
///   than inspecting the builder: a harness that cannot report is a harness that
///   cannot be trusted to be silent.
///
/// **Why no wipe.** The allocator's input is "the counter key is absent", and a
/// fresh fixture prefix supplies exactly that — the same `None` a wipe produces.
/// The storage-api harness wires no session-recovery hook into its base-path
/// cache (that surface belongs to the LocalDisk harness and its own e2e), so a
/// `full_valkey_wipe` here would fail at session resolution before reaching an
/// allocation and would test the scaffolding instead of the subject. The
/// wipe-shaped composition is pinned once, by the sibling test above.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn storage_api_harness_create_layer_repairs_the_counter_through_the_hook() {
    let harness = setup_storage_api_checkpoint_harness(1).await;

    // Parity guard up front, on the harness that lacked the wiring. Asserting it
    // before any traffic means a harness regression is reported as itself rather
    // than as a confusing id mismatch downstream.
    assert_harness_carries_object_id_counter_repair(&harness);

    // The durable record a surviving checkpoint store holds: "object ids up to
    // 5000 are claimed". Written through the real StorageApiCheckpointStore, so
    // the repair's read is a genuine outbound HTTP round trip against the
    // wiremock fake rather than an in-process map lookup.
    put_counters_sidecar(&harness.store, STORAGE_API_SIDECAR_MAX_OBJECT_ID).await;

    assert!(
        !counter_key_exists(&harness.tv).await,
        "fixture precondition: a fresh prefix must have no le:counter:object_id — that \
         absence is the allocator input the repair keys off, and it is the same input a \
         keyspace loss produces"
    );

    // Session created through the store rather than the RPC: the storage-api
    // harness's base-path cache carries no session-recovery hook, and session
    // creation is scaffolding here in any case — the subject is the allocation
    // that follows.
    let session_store =
        SessionStore::new(harness.tv.client().clone(), harness.tv.key_schema().clone());
    let session_id = session_store
        .create(
            STORAGE_API_SESSION_NAME,
            STORAGE_API_SESSION_BASE_PATH,
            STORAGE_API_CREATED_BY,
        )
        .await
        .expect("session creation must succeed");

    let object_id = create_layer(&harness.svc, &session_id, STORAGE_API_LAYER_NAME).await;

    assert_eq!(
        object_id,
        STORAGE_API_SIDECAR_MAX_OBJECT_ID + SAFETY_MARGIN + 1,
        "the allocation must resume above the sidecar-derived floor \
         (max_object_id_observed={STORAGE_API_SIDECAR_MAX_OBJECT_ID} + \
         safety_margin={SAFETY_MARGIN}), so the first id off it is {}; got {object_id}. A \
         value of 1 is this harness's pre-fix behaviour verbatim — an unhooked allocator \
         establishing the counter at zero and re-minting ids the storage-api checkpoint \
         blobs still claim, from a store that outlives the pod",
        STORAGE_API_SIDECAR_MAX_OBJECT_ID + SAFETY_MARGIN + 1
    );

    assert_eq!(
        object_id_counter_repair_metric(&harness, "unhooked_initialized"),
        0,
        "THE assertion of this test: the hooked path must be what ran. This cell increments \
         only on `allocate_object_id`'s no-hook arm, so a tick here means the storage-api \
         harness LayerStore is missing `.with_object_id_counter_repair(...)` again"
    );
    assert_eq!(
        object_id_counter_repair_metric(&harness, "repaired_seeded"),
        1,
        "exactly one repair must have re-seeded from the counters sidecar the storage-api \
         adapter served — which also proves `.with_metrics(...)` is wired, since an \
         unmetered LayerStore reports nothing at all here"
    );
    assert_eq!(
        object_id_counter_repair_metric(&harness, "repaired_initialized"),
        0,
        "a sidecar IS present over the wire, so the establish-at-zero arm must not have run. \
         A tick here means the adapter answered `no sidecar` — i.e. the outbound GET missed \
         the address the sidecar was PUT to"
    );
    assert_eq!(
        object_id_counter_repair_metric(&harness, "failed")
            + object_id_counter_repair_metric(&harness, "failed_after_repair")
            + object_id_counter_repair_metric(&harness, "wait_timeout"),
        0,
        "nothing may have failed or timed out against the local wiremock adapter — a tick on \
         any of these three would mean the harness's budgets (derived from the same `Config` \
         methods production uses) are too tight for a loopback HTTP round trip"
    );

    // Re-run the guard after a counter-absent event has actually been served:
    // before any traffic the cell is trivially zero, so this is the reading that
    // carries information.
    assert_harness_carries_object_id_counter_repair(&harness);

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}
