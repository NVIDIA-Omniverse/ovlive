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

//! End-to-end integration tests for the SR-3.1 ext-value checkpoint
//! mirror pipeline (Epic `t1.fgj.zc` Task `t2.fgj.zc.nx`, Subtask
//! `t3.fgj.zc.nx.op`).
//!
//! These tests are the load-bearing acceptance criterion for the entire
//! ext-mirror wiring: a `CreateLayer` whose OMPF carries an external-value
//! `SetField` must drive the bytes all the way through to the configured
//! `CheckpointStore` adapter without back-pressuring the hot path. Catching
//! drift at any call-site boundary is the whole point — the unit-test
//! Subtask exercises `ext_mirror.rs` in isolation; here we exercise the
//! full pipeline:
//!
//! 1. The streaming chunk handler decodes the OMPF.
//! 2. `ValkeyExternalValueStore::store_values` runs one `EVALSHA` of
//!    `ext_store_value.lua` per hash (`SET data NX` + `HINCRBY meta
//!    ref_count 1`).
//! 3. The post-commit `ExtMirrorSender::send` charges the value's copy to
//!    the mirror's byte budget and enqueues it, never waiting.
//! 4. A worker dequeues and calls `store.put_ext_blob(...)`.
//! 5. The adapter persists the bytes.
//!
//! Four test variants cover the Task contracts:
//!
//! - **Happy path** — sender wired, workers spawned, mirror lands in
//!   `InMemoryCheckpointStore` inside the 5 s deadline. AC #4.
//! - **No-mirror no-op** — sender absent (the "checkpoint disabled"
//!   wiring shape). Valkey-side write still commits; neither
//!   `dropped_total` nor `errors_total` ticks.
//! - **Drop-on-full** — capacity-1 channel with zero workers. The send of
//!   the second hash sees `Full` and bumps
//!   `checkpoint_ext_mirror_dropped_total{reason="queue_full"}`; the
//!   CreateLayer RPC still returns success because the hot path is
//!   unaffected.
//! - **Byte budget** (T10, b.mhq) — a budget of three values' charges with
//!   no worker running: of six large values the overflow ticks
//!   `reason="budget_full"`, the RPC still succeeds and the store's data and
//!   reference counts are those of a run without a mirror, and workers
//!   started afterwards land the admitted values byte-identical.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use dashmap::DashMap;
use fred::interfaces::KeysInterface;
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    ext_mirror_channel, spawn_ext_mirror_workers, CheckpointStore, ExtMirrorReceiver,
};
use live_edit::grpc::layer::handle_create_layer;
use live_edit::grpc::service::LiveEditService;
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::external_values::{
    ext_content_hash, ExternalValueStore, ValkeyExternalValueStore,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::{delta_to_ompf_bytes, delta_with_set_field_external};
use crate::grpc::helpers::{
    build_disabled_recovery_hook, build_disabled_session_recovery_hook, make_create_data,
    make_create_header, make_request, streaming_create, AllowAllPermissions, StubUserinfo,
};

/// Service label stamped onto the `Metrics` registry and threaded into
/// `LiveEditService::service_label`. Pinned the same way as
/// `tests/integration/checkpoint/helpers.rs::SERVICE_LABEL` so per-service
/// label-cells stay addressable by sibling test helpers.
const SERVICE_LABEL: &str = "test";

/// Bundle returned by [`setup_ext_mirror_harness`]. Holds every handle a
/// per-test body needs to drive CreateLayer + assertion + shutdown.
///
/// The `cancel` token is paired with the `worker_handles`; tests should
/// fire `cancel.cancel()` and `join_all` the handles inside a 5 s timeout
/// to verify clean shutdown.
struct ExtMirrorHarness {
    tv: TestValkey,
    svc: LiveEditService,
    store: Arc<dyn CheckpointStore>,
    metrics: SharedMetrics,
    cancel: CancellationToken,
    worker_handles: Vec<tokio::task::JoinHandle<()>>,
    /// Held only on the `worker_count=0` arm of the harness so the
    /// channel's receiver half stays alive for the test's lifetime. Without
    /// this, `spawn_ext_mirror_workers` would consume `rx` and immediately
    /// drop the `Arc<Mutex<Receiver>>` (no workers cloned it), closing the
    /// channel — every subsequent send would find it closed, which is
    /// logged at DEBUG and does NOT tick
    /// `checkpoint_ext_mirror_dropped_total`. Holding the receiver here
    /// preserves the `Full` semantics the drop-on-full test asserts, and
    /// lets the byte-budget test start its workers only after its upload,
    /// as if they had been held until then.
    held_receiver: Option<ExtMirrorReceiver>,
}

/// A byte budget of 1 GiB, which the small values most of these tests store
/// never exhaust, so only the item count can shed them.
const ROOMY_MIRROR_BUDGET_BYTES: u64 = 1 << 30;

/// Construct a fully-wired harness for the ext-mirror E2E tests.
///
/// - `mirror_capacity` — bounded mpsc channel capacity (drop-on-full
///   contract is exercised via `mirror_capacity=1` + `worker_count=0`).
/// - `mirror_budget_bytes` — the mirror's byte budget, as `run` passes the
///   pod memory plan's `M_ext` to the constructor (`budget_full` is
///   exercised with a budget of a few values' charges).
/// - `worker_count` — number of `spawn_ext_mirror_workers` tasks. Zero is
///   valid (drop-on-full variant — no consumer ever drains the channel);
///   the receiver is then held in `held_receiver`, from which a test can
///   start workers later.
/// - `with_mirror` — when `false`, the `ValkeyExternalValueStore` is built
///   without `with_ext_mirror_sender(...)`, exercising the
///   "checkpoint-disabled" no-op shape that production wires when
///   `checkpoint_enabled=false` or `CheckpointAdapter::Disabled`.
async fn setup_ext_mirror_harness(
    mirror_capacity: usize,
    mirror_budget_bytes: u64,
    worker_count: usize,
    with_mirror: bool,
) -> ExtMirrorHarness {
    let tv = TestValkey::setup().await;

    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());

    // Build the mpsc channel + sender unconditionally so the `with_mirror=false`
    // variant can still spawn (zero) workers against the receiver and assert
    // the sender was never installed — the sender side simply goes out of
    // scope below if not wired.
    let (sender, rx) = ext_mirror_channel(
        mirror_capacity,
        std::num::NonZeroU64::new(mirror_budget_bytes).expect("a test budget is non-zero"),
        Arc::clone(&metrics),
    );

    let cancel = CancellationToken::new();
    // Branch: when workers exist, hand `rx` to `spawn_ext_mirror_workers`
    // (the production wiring shape — workers own the receiver via
    // `Arc<Mutex<>>`). When `worker_count == 0`, keep `rx` alive in the
    // harness so the channel does NOT close — see `held_receiver`
    // docstring for the load-bearing reason.
    let (worker_handles, held_receiver) = if worker_count == 0 {
        (Vec::new(), Some(rx))
    } else {
        let handles = spawn_ext_mirror_workers(
            Arc::clone(&store),
            rx,
            worker_count,
            Arc::clone(&metrics),
            None,
            cancel.clone(),
        );
        (handles, None)
    };

    // ── ValkeyExternalValueStore wiring ─────────────────────────────────────
    //
    // The hot-path `store_values` consults `ext_mirror_sender` via an
    // `if let` — `None` is the production wiring for "checkpoint disabled
    // or adapter Disabled" and must be a complete no-op (no enqueue, no
    // metric, no log). The two arms below model production's two
    // configurations exactly: one calls `.with_ext_mirror_sender(...)`,
    // one omits it. Test 2 is the regression guard for the second arm.
    let ext_value_store: Arc<dyn ExternalValueStore> = {
        let mut builder =
            ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
                .with_metrics(Arc::clone(&metrics));
        if with_mirror {
            builder = builder.with_ext_mirror_sender(sender);
        } else {
            // Drop the sender explicitly so the channel's last sender
            // goes out of scope — the spawned workers (zero in the
            // `with_mirror=false` variant) will observe `recv()` returning
            // `None` and exit cleanly during the post-test cancel.
            drop(sender);
        }
        Arc::new(builder)
    };

    let session_store = Arc::new(SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let layer_store = Arc::new(LayerStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
        4096,
    ));

    let recovery = build_disabled_recovery_hook(Arc::clone(&layer_store), Arc::clone(&metrics));
    let session_recovery = build_disabled_session_recovery_hook(&tv, Arc::clone(&metrics));

    let svc = LiveEditService {
        catchup_snapshot_enabled: false,
        session_store,
        layer_store,
        ext_value_store,
        session_tokens: Arc::new(DashMap::new()),
        userinfo_client: Arc::new(StubUserinfo),
        permissions_client: Arc::new(AllowAllPermissions),
        max_layer_size: 10 * 1024 * 1024,
        upload_admission: std::sync::Arc::new(
            live_edit::grpc::upload_admission::UploadAdmissionGate::new(Default::default(), None),
        ),
        snapshot_flights: live_edit::storage::snapshot_flights::SnapshotFlights::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ),
        valkey: tv.client().clone(),
        key_schema: tv.key_schema().clone(),
        presence_router: Arc::new(live_edit::presence::LocalPresenceRouter::new()),
        presence_lease_ttl_secs: live_edit::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
        presence_refresh_interval_secs: live_edit::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
        presence_keepalive_interval_ms: live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
        presence_cleanup_tracker: tokio_util::task::TaskTracker::new(),
        subscription_tracker: tokio_util::task::TaskTracker::new(),
        shutdown: live_edit::grpc::graceful_shutdown::GracefulShutdown::new(),
        metrics: Arc::clone(&metrics),
        service_label: SERVICE_LABEL.to_string(),
        session_base_path_cache: live_edit::grpc::session_cache::SessionBasePathCache::new(),
        recovery,
        session_recovery,
        base_bootstrap: crate::common::build_disabled_base_bootstrap(
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::clone(&metrics),
        ),
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
    };

    ExtMirrorHarness {
        tv,
        svc,
        store,
        metrics,
        cancel,
        worker_handles,
        held_receiver,
    }
}

/// Cancel the workers and await the join handles inside a 5 s timeout.
/// Asserting clean shutdown verifies the mirror-pipeline's bounded-drain
/// contract (`DRAIN_BUDGET=64` items maximum, then return).
async fn shutdown_workers(
    cancel: CancellationToken,
    worker_handles: Vec<tokio::task::JoinHandle<()>>,
) {
    cancel.cancel();
    let shutdown_deadline = Duration::from_secs(5);
    let join_all = async {
        for handle in worker_handles {
            let _ = handle.await;
        }
    };
    tokio::time::timeout(shutdown_deadline, join_all)
        .await
        .expect("ext-mirror workers must join inside 5 s after cancel");
}

/// Poll `store.get_ext_blob(&hash_hex)` until the blob is present, or
/// panic with a diagnostic on deadline expiry. 5 s is generous against
/// an in-memory channel + in-memory adapter — a healthy pipeline lands
/// within a few hundred microseconds, so a timeout here means the wiring
/// is genuinely broken.
async fn wait_for_ext_blob(
    store: &Arc<dyn CheckpointStore>,
    hash_hex: &str,
    deadline_secs: u64,
) -> Bytes {
    let deadline = Instant::now() + Duration::from_secs(deadline_secs);
    let poll_interval = Duration::from_millis(10);
    loop {
        match store.get_ext_blob(hash_hex).await {
            Ok(Some(bytes)) => return bytes,
            Ok(None) => {
                // Mirror worker has not landed the blob yet — keep polling.
            }
            Err(err) => {
                panic!(
                    "get_ext_blob failed during wait_for_ext_blob \
                     (hash_hex={hash_hex}, deadline_secs={deadline_secs}): {err}"
                );
            }
        }
        if Instant::now() >= deadline {
            panic!(
                "wait_for_ext_blob timed out after {deadline_secs}s waiting for \
                 hash_hex={hash_hex}. Common causes: the test's \
                 ValkeyExternalValueStore was built without \
                 `with_ext_mirror_sender`, the worker pool was spawned with \
                 zero workers, or `store_values` short-circuited before the \
                 post-commit `send` call."
            );
        }
        tokio::time::sleep(poll_interval).await;
    }
}

// =============================================================================
// AC #4 — Happy path: CreateLayer with an ext-value mirrors to the store.
// =============================================================================

/// CreateLayer with an OMPF carrying one external-reference `SetField`
/// drives the bytes into `InMemoryCheckpointStore::get_ext_blob` within a
/// 5 s deadline. This is the load-bearing acceptance criterion for the
/// entire ext-mirror wiring: every call-site boundary (constructor
/// accepting `Option<ExtMirrorSender>`, `main.rs` workers-spawned
/// wiring, `store_values` post-commit `send`, worker `recv()` →
/// `put_ext_blob`) is exercised.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_with_ext_value_mirrors_to_checkpoint_store() {
    // Capacity 16 + 2 workers — comfortably overprovisioned for a single
    // ext-value enqueue; matches the production defaults' ratio (1000 / 2).
    let harness = setup_ext_mirror_harness(16, ROOMY_MIRROR_BUDGET_BYTES, 2, true).await;
    let ExtMirrorHarness {
        tv,
        svc,
        store,
        cancel,
        worker_handles,
        ..
    } = harness;

    // Stand up a session via the SessionStore directly (no auth needed for
    // session creation in the test wiring), then build a CreateLayer
    // streaming request whose OMPF carries one external-reference SetField.
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("ext-mirror-e2e-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let payload: &[u8] = b"my-ext-payload";
    let hash = ext_content_hash(payload);
    let hash_hex = sha256_to_hex(&hash);

    let delta = delta_with_set_field_external(
        /*section_id=*/ 1,
        /*key=*/ "f",
        /*hash=*/ hash.to_vec(),
        /*ext_value_size=*/ payload.len() as u64,
        // 1-based: index 1 = the first (and only) external-value blob.
        /*ext_value_index=*/
        1,
        /*set_order=*/ 1,
    );
    let ompf_bytes = delta_to_ompf_bytes(&delta, &[payload]);

    let stream = streaming_create(vec![
        make_create_header(&session_id, "layer-ext"),
        make_create_data(ompf_bytes),
    ]);
    let request = make_request(stream);

    handle_create_layer(&svc, request)
        .await
        .expect("CreateLayer with ext-value must succeed");

    // Poll the adapter for the mirrored blob. A healthy pipeline lands the
    // blob within sub-millisecond latency (in-memory channel + in-memory
    // adapter), but the 5 s deadline absorbs CI scheduling jitter.
    let landed = wait_for_ext_blob(&store, &hash_hex, 5).await;
    assert_eq!(
        landed.as_ref(),
        payload,
        "mirrored blob bytes must match the CreateLayer payload exactly"
    );

    // Bounded-drain shutdown — workers must join inside 5 s.
    shutdown_workers(cancel, worker_handles).await;
}

// =============================================================================
// AC #4 regression guard #1 — No-mirror-sender path is a complete no-op.
// =============================================================================

/// CreateLayer with an OMPF carrying an external value, against a
/// `ValkeyExternalValueStore` built WITHOUT `with_ext_mirror_sender`,
/// commits the Valkey-side write as usual and ticks neither
/// `dropped_total` nor `errors_total`. This pins the "complete no-op
/// when checkpoint is disabled" contract — the absence of a wired sender
/// must not surface as a spurious metric tick or error log.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_with_ext_value_no_mirror_path_is_noop_when_sender_absent() {
    // `with_mirror=false` builds the store without `with_ext_mirror_sender`.
    // Capacity + worker count don't matter on this arm (channel + workers
    // are constructed but the sender is dropped, so the workers see
    // `recv()` returning `None` and exit during shutdown).
    let harness = setup_ext_mirror_harness(16, ROOMY_MIRROR_BUDGET_BYTES, 0, false).await;
    let ExtMirrorHarness {
        tv,
        svc,
        metrics,
        cancel,
        worker_handles,
        ..
    } = harness;

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("ext-mirror-noop-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let payload: &[u8] = b"my-ext-payload";
    let hash = ext_content_hash(payload);
    let hash_hex = sha256_to_hex(&hash);

    // ext_value_index is 1-based: index 1 = the first (and only) blob.
    let delta = delta_with_set_field_external(1, "f", hash.to_vec(), payload.len() as u64, 1, 1);
    let ompf_bytes = delta_to_ompf_bytes(&delta, &[payload]);

    let stream = streaming_create(vec![
        make_create_header(&session_id, "layer-ext-noop"),
        make_create_data(ompf_bytes),
    ]);
    let request = make_request(stream);

    handle_create_layer(&svc, request)
        .await
        .expect("CreateLayer must still succeed when the mirror sender is absent");

    // Valkey hot-path commit must have landed — proves `store_values` ran
    // to completion even with the mirror sender absent.
    let data_key = tv.key_schema().ext_data(&hash_hex);
    let stored: Bytes = tv
        .client()
        .pool
        .get(&data_key)
        .await
        .expect("GET ext_data:{hash} must succeed (hot path committed)");
    assert_eq!(
        stored.as_ref(),
        payload,
        "the hot-path Valkey write must still commit byte-identically when the mirror sender is absent"
    );

    // Neither mirror metric should have ticked — the `None` sender path is
    // a complete no-op. A spurious tick here means production drift broke
    // the gate inside `store_values`'s `if let (Some(sender), Some(copy)) = ...`.
    assert_eq!(
        live_edit::observability::metrics::test_util::sample_sum(
            &live_edit::observability::metrics::test_util::render(&metrics),
            "live_edit_checkpoint_ext_mirror_dropped_total",
            &[]
        )
        .unwrap_or(0.0) as u64,
        0,
        "dropped_total must stay at 0 for every reason when no sender is wired (no enqueue \
         attempt fires)"
    );
    assert_eq!(
        live_edit::observability::metrics::test_util::sample_value(
            &live_edit::observability::metrics::test_util::render(&metrics),
            "live_edit_checkpoint_ext_mirror_errors_total",
            &[]
        )
        .unwrap_or(0.0) as u64,
        0,
        "errors_total must stay at 0 when no sender is wired (no put_ext_blob fires)"
    );

    shutdown_workers(cancel, worker_handles).await;
}

// =============================================================================
// AC #4 regression guard #2 — Drop-on-full ticks dropped_total only.
// =============================================================================

/// CreateLayer with an OMPF carrying TWO distinct external-reference
/// `SetField`s, against a capacity-1 channel with ZERO workers, lands
/// both bytes in Valkey and increments
/// `checkpoint_ext_mirror_dropped_total` at least once. The CreateLayer
/// RPC itself must still return success — the SR-3.1 contract is that
/// queue saturation surfaces only via the metric, never as a hot-path
/// failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_with_ext_value_drop_on_full_increments_dropped_metric() {
    // Capacity 1 + zero workers: the first send succeeds (channel
    // empty), the second send sees `Full` and bumps the counter.
    // No worker exists to drain, so the saturation is permanent for the
    // test's lifetime — the post-test cancel exits the (empty) handle vec
    // immediately.
    let harness = setup_ext_mirror_harness(1, ROOMY_MIRROR_BUDGET_BYTES, 0, true).await;
    let ExtMirrorHarness {
        tv,
        svc,
        metrics,
        cancel,
        worker_handles,
        ..
    } = harness;

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("ext-mirror-drop-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // Two distinct payloads → two distinct content hashes → two
    // separate per-hash `EVALSHA`s of `ext_store_value.lua` inside
    // `store_values`, each followed by its own send. The second one
    // MUST see `Full`.
    let payload_a: &[u8] = b"payload-a";
    let payload_b: &[u8] = b"payload-b";
    let hash_a = ext_content_hash(payload_a);
    let hash_b = ext_content_hash(payload_b);

    // OwnedDelta carrying two SetFields, each pointing at a distinct
    // ext-value index. `delta_with_set_field_external` only sets ONE
    // SetField, so we build the two-field delta inline using the same
    // shape — keeps the test self-contained without expanding the shared
    // fixture surface. `ext_value_index` is 1-based: index 1 = the first
    // blob in `delta_to_ompf_bytes`'s `external_values` slice, index 2 =
    // the second.
    let delta = oodf_ffi::OwnedDelta {
        set_fields: vec![
            oodf_ffi::SetField {
                section_id: 1,
                key_name: "f0".to_string(),
                value_or_ext_hash: hash_a.to_vec(),
                ext_value_size: payload_a.len() as u64,
                ext_value_index: 1,
                set_order: 1,
            },
            oodf_ffi::SetField {
                section_id: 1,
                key_name: "f1".to_string(),
                value_or_ext_hash: hash_b.to_vec(),
                ext_value_size: payload_b.len() as u64,
                ext_value_index: 2,
                set_order: 2,
            },
        ],
        ..Default::default()
    };
    let ompf_bytes = delta_to_ompf_bytes(&delta, &[payload_a, payload_b]);

    let stream = streaming_create(vec![
        make_create_header(&session_id, "layer-ext-drop"),
        make_create_data(ompf_bytes),
    ]);
    let request = make_request(stream);

    handle_create_layer(&svc, request).await.expect(
        "CreateLayer must succeed even when the mirror channel saturates \
             (the hot-path Valkey commit is independent of the mirror)",
    );

    // Both ext-values must still be in Valkey — the hot-path commit is
    // load-bearingly independent of the mirror's saturation.
    for (hash, expected) in [(&hash_a, payload_a), (&hash_b, payload_b)] {
        let hash_hex = sha256_to_hex(hash);
        let data_key = tv.key_schema().ext_data(&hash_hex);
        let stored: Bytes = tv
            .client()
            .pool
            .get(&data_key)
            .await
            .expect("GET ext_data:{hash} must succeed (hot path committed)");
        assert_eq!(
            stored.as_ref(),
            expected,
            "Valkey hot-path write must commit regardless of mirror channel saturation"
        );
    }

    // The second hash's send saw `Full` and ticked the counter.
    // Asserting `>= 1` (not `== 1`) because the exact ordering of
    // per-hash futures inside `store_values`'s `buffered(N)` stream is
    // not deterministic — both could theoretically see `Full` if the
    // first one hadn't been polled yet at the point the second one's
    // future runs. The contract is "queue-full surfaces only via the
    // metric", which `>= 1` proves.
    let dropped = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_checkpoint_ext_mirror_dropped_total",
        &[("reason", "queue_full")],
    )
    .unwrap_or(0.0) as u64;
    assert!(
        dropped >= 1,
        "checkpoint_ext_mirror_dropped_total must tick when the bounded queue saturates \
         (observed dropped_total={dropped}); the second hash's send should have seen Full"
    );

    // Workers were spawned with count=0, so the cancel-and-join below is
    // a no-op against an empty vec. Still call it through the same helper
    // to keep the shutdown contract uniform across the tests.
    shutdown_workers(cancel, worker_handles).await;
}

// =============================================================================
// T10 (b.mhq) — the byte budget sheds large values, the store is unaffected.
// =============================================================================

/// The large values T10 uploads: six, of 256 KiB each, every byte of one
/// value its index.
const LARGE_VALUES: usize = 6;
const LARGE_VALUE_LEN: usize = 256 * 1024;

/// The budget T10's mirror gets: room for three of its values' charges (each
/// value rounded up to a KiB plus one KiB of overhead, 257 KiB) and not a
/// fourth.
const THREE_LARGE_VALUES_BUDGET_BYTES: u64 = 3 * (257 * 1024) + 100 * 1024;

/// One CreateLayer whose OMPF carries one external-reference `SetField` per
/// value of `values`, on a new session of `tv`'s store; returns each value's
/// content-hash hex, in order.
async fn create_layer_with_external_values(
    tv: &TestValkey,
    svc: &LiveEditService,
    session: &str,
    values: &[Vec<u8>],
) -> Vec<String> {
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(session, "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    let hashes: Vec<[u8; 32]> = values.iter().map(|v| ext_content_hash(v)).collect();
    let delta = oodf_ffi::OwnedDelta {
        set_fields: values
            .iter()
            .zip(&hashes)
            .enumerate()
            .map(|(i, (value, hash))| oodf_ffi::SetField {
                section_id: 1,
                key_name: format!("f{i}"),
                value_or_ext_hash: hash.to_vec(),
                ext_value_size: value.len() as u64,
                // 1-based into the OMPF's external values.
                ext_value_index: i as u32 + 1,
                set_order: i as u64 + 1,
            })
            .collect(),
        ..Default::default()
    };
    let refs: Vec<&[u8]> = values.iter().map(Vec::as_slice).collect();
    let stream = streaming_create(vec![
        make_create_header(&session_id, "layer-ext-budget"),
        make_create_data(delta_to_ompf_bytes(&delta, &refs)),
    ]);
    handle_create_layer(svc, make_request(stream)).await.expect(
        "CreateLayer must succeed whatever the mirror sheds: the store path never waits on it",
    );
    hashes.iter().map(sha256_to_hex).collect()
}

/// Each value's stored data and `ref_count` in `tv`'s store, by hash.
async fn stored_values(tv: &TestValkey, hashes: &[String]) -> Vec<(Option<Bytes>, Option<i64>)> {
    let pool = &tv.client().pool;
    let mut stored = Vec::with_capacity(hashes.len());
    for hash_hex in hashes {
        let data: Option<Bytes> = pool
            .get(tv.key_schema().ext_data(hash_hex))
            .await
            .expect("GET ext data");
        let ref_count: Option<i64> = fred::interfaces::HashesInterface::hget(
            pool,
            tv.key_schema().ext_meta(hash_hex),
            "ref_count",
        )
        .await
        .expect("HGET ext meta ref_count");
        stored.push((data, ref_count));
    }
    stored
}

/// T10: a CreateLayer carrying six 256 KiB external values against a mirror
/// whose byte budget holds three of their charges, its workers not yet
/// running (as if all were held), succeeds: the mirror admits three and
/// drops three `reason="budget_full"` — never `queue_full`, as its 16-item
/// channel has room. The store is untouched by what the mirror shed: every
/// value's data and `ref_count` are those of the same upload against a store
/// with no mirror wired. Once workers start, the three admitted values land
/// in the `InMemoryCheckpointStore` byte-identical, and the three shed never
/// do.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_with_large_ext_values_past_the_mirror_budget_drops_budget_full_and_commits() {
    let values: Vec<Vec<u8>> = (0..LARGE_VALUES)
        .map(|i| vec![i as u8 + 1; LARGE_VALUE_LEN])
        .collect();

    let ExtMirrorHarness {
        tv,
        svc,
        store,
        metrics,
        cancel,
        held_receiver,
        ..
    } = setup_ext_mirror_harness(16, THREE_LARGE_VALUES_BUDGET_BYTES, 0, true).await;
    let hashes = create_layer_with_external_values(&tv, &svc, "ext-mirror-budget", &values).await;

    let text = live_edit::observability::metrics::test_util::render(&metrics);
    let dropped = |reason: &str| {
        live_edit::observability::metrics::test_util::sample_value(
            &text,
            "live_edit_checkpoint_ext_mirror_dropped_total",
            &[("reason", reason)],
        )
        .unwrap_or(0.0) as u64
    };
    assert_eq!(
        dropped("budget_full"),
        3,
        "three of six values do not fit the budget"
    );
    assert_eq!(dropped("queue_full"), 0, "the channel had room");
    assert_eq!(dropped("oversized"), 0, "each value fits the whole budget");

    let unmirrored = setup_ext_mirror_harness(16, ROOMY_MIRROR_BUDGET_BYTES, 0, false).await;
    let unmirrored_hashes = create_layer_with_external_values(
        &unmirrored.tv,
        &unmirrored.svc,
        "ext-mirror-budget-unmirrored",
        &values,
    )
    .await;
    assert_eq!(hashes, unmirrored_hashes);
    let with_mirror = stored_values(&tv, &hashes).await;
    assert_eq!(
        with_mirror,
        stored_values(&unmirrored.tv, &hashes).await,
        "the store's data and reference counts are those of a run without a mirror"
    );
    for ((data, ref_count), value) in with_mirror.iter().zip(&values) {
        assert_eq!(
            data.as_deref(),
            Some(value.as_slice()),
            "every value is stored"
        );
        assert!(ref_count.is_some(), "every value is referenced");
    }
    shutdown_workers(unmirrored.cancel, unmirrored.worker_handles).await;

    // Start the workers now, as if released.
    let worker_handles = spawn_ext_mirror_workers(
        Arc::clone(&store),
        held_receiver.expect("a zero-worker harness holds its receiver"),
        2,
        Arc::clone(&metrics),
        None,
        cancel.clone(),
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    let landed = loop {
        let mut landed = Vec::new();
        for (hash_hex, value) in hashes.iter().zip(&values) {
            if let Some(bytes) = store.get_ext_blob(hash_hex).await.expect("get_ext_blob") {
                assert_eq!(
                    bytes.as_ref(),
                    value.as_slice(),
                    "{hash_hex} lands byte-identical"
                );
                landed.push(hash_hex.clone());
            }
        }
        if landed.len() >= 3 || Instant::now() >= deadline {
            break landed;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    // The queue is drained by the cancel below, so every admitted value has
    // landed once the workers have joined.
    shutdown_workers(cancel, worker_handles).await;
    let mut after = Vec::new();
    for hash_hex in &hashes {
        if store
            .get_ext_blob(hash_hex)
            .await
            .expect("get_ext_blob")
            .is_some()
        {
            after.push(hash_hex.clone());
        }
    }
    assert_eq!(
        (landed.len(), after.len()),
        (3, 3),
        "the three admitted values land, and the three shed never do"
    );
}
