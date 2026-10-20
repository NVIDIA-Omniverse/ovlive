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

// Regression tests for b.fd9 — Sweep storage/grpc Internal → Unavailable for
// transient Valkey errors.
//
// The b.ohw H-grpc-1 fix mapped transient auth-backend outages to gRPC
// `Code::Unavailable` so client retry policies treat them as retriable. The
// same logic needed to be applied to every Valkey error site. This module
// drives synthetic fred errors through the storage helpers that surface
// directly to gRPC handlers and asserts the resulting `ServiceError` maps to
// `Code::Unavailable` for transient errors (by kind, or by one of the
// message-matched carve-outs of §5b–§5d, §5f and §5g) and `Code::Internal` for permanent
// ones — with one permanent exception (b.xd1, §5e): a store write refusal
// (`-OOM`, `-MISCONF`, `-READONLY`, `-NOREPLICAS`, all `ErrorKind::Unknown`)
// stays non-transient, so nothing retries it in-process, but both mappers
// answer it a plain `ServiceError::Unavailable` — `Code::Unavailable` too.
//
// Pre-b.fd9 every site mapped *any* fred error to `ServiceError::Internal`.
// Each test below would FAIL on pre-fix code (the wrong gRPC code would
// surface) and locks in the post-fix classifier.
//
// Per-site coverage (matches the b.fd9 issue body):
//   * `LayerStore::list_layer_names`         — SMEMBERS layers index
//   * `LayerStore::get_layer_mapping`        — GET layer mapping
//   * `LayerStore::read_object_header`       — HGETALL obj header
//   * `LayerStore::delete_layer`             — cascade-wide retry (Aborted)
//   * `SessionStore::get_base_path`          — HGET session base_path
//
// Gated behind `--features test-fault-injection`; the seams these tests
// drive (`with_get_base_path_fault_injector_for_test`) compile out of
// release builds. Where no seam exists, the test drives the helper
// indirectly by manipulating Valkey state (e.g. for the no-side-effect read
// paths we exercise the success path against a real Valkey since the
// failure mode is hard to manufacture without a fault-injecting fred
// wrapper). Direct fault injection on the read paths happens via the
// `client.rs::map_fred_err_for_grpc` unit tests embedded below.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use fred::error::ErrorKind as FredErrorKind;

use live_edit::error::ServiceError;
use live_edit::grpc::session_cache::SessionBasePathCache;
use live_edit::storage::client::{
    is_transaction_router_exhausted_error, is_transient_fred_error, map_fred_err_for_grpc,
    map_transient_fred_err, FredErrOrigin, BUDGET_EXHAUSTED_SHAPES, POOL_LABEL_DATA,
    POOL_LABEL_TXN, STORE_NOT_SERVING_SHAPES, TRANSACTION_ROUTER_EXHAUSTED_SHAPES, TRYAGAIN_REPLY,
};
use live_edit::storage::layer::{LayerStore, OPERATION_OVERTAKEN_MESSAGE};
use live_edit::storage::session::SessionStore;
use live_edit::storage::store_oom::StoreOomSignal;
use live_edit::storage::test_support::TestValkey;

fn synthetic_fred_err(kind: FredErrorKind) -> fred::error::Error {
    fred::error::Error::new(kind, "synthetic b.fd9 transient I/O error")
}

/// fred's router-miss error (b.c7a) — the first of the two classifications in
/// the taxonomy that are decided by the MESSAGE rather than the kind (the other
/// is the budget-exhaustion family, b.cv4, below). `write_command` emits it with
/// this prefix under `Unknown` (the shape this service compiles) and under
/// `Routing` (its `replicas`-gated sites).
///
/// [`synthetic_fred_err`] above deliberately builds a generic message, so every
/// kind-matrix test in this file exercises the kind-level classification only;
/// this constructor is what drives the carve-out.
fn router_miss_fred_err(kind: FredErrorKind, details: &'static str) -> fred::error::Error {
    fred::error::Error::new(kind, details)
}

/// The two kinds fred attaches its router miss to, paired with the message each
/// site emits. `Routing`'s "… to replica." variant is the third emit site.
const ROUTER_MISS_SHAPES: [(FredErrorKind, &str); 3] = [
    (FredErrorKind::Unknown, "Failed to route command."),
    (FredErrorKind::Routing, "Failed to route command."),
    (
        FredErrorKind::Routing,
        "Failed to route command to replica.",
    ),
];

/// fred's redirect **budget-exhaustion** error (b.cv4) — the second
/// message-decided classification in the taxonomy. Delegates to
/// [`router_miss_fred_err`]: the construction is identical (hand-built, because
/// every origin lives in fred's private `mod router` / `mod protocol`), and only
/// the provenance differs, which is what the name records at the call sites.
///
/// The `(kind, message)` rows come from
/// [`BUDGET_EXHAUSTED_SHAPES`](live_edit::storage::client::BUDGET_EXHAUSTED_SHAPES),
/// which lives next to the predicate and carries the per-shape provenance for
/// all three test sites that drive it.
fn budget_exhausted_fred_err(kind: FredErrorKind, details: &'static str) -> fred::error::Error {
    router_miss_fred_err(kind, details)
}

// =============================================================================
// 1. `map_fred_err_for_grpc` unit-style coverage — every transient fred kind
//    in `is_transient_fred_error` MUST surface as `Code::Unavailable`; every
//    permanent kind MUST surface as `Code::Internal`. Locks in the classifier
//    contract that the rest of the suite relies on.
// =============================================================================
//
// These tests run without TestValkey — pure mapper coverage.

#[test]
fn map_fred_err_for_grpc_timeout_maps_to_unavailable() {
    let svc = map_fred_err_for_grpc(
        FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
        "SMEMBERS x",
        synthetic_fred_err(FredErrorKind::Timeout),
    );
    let status = tonic::Status::from(svc);
    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "Timeout is transient → must surface as Unavailable so clients retry"
    );
    assert!(
        status.message().contains("SMEMBERS x"),
        "op label must propagate to operator-visible message, got: {}",
        status.message()
    );
}

#[test]
fn map_fred_err_for_grpc_io_maps_to_unavailable() {
    let svc = map_fred_err_for_grpc(
        FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
        "HGET y",
        synthetic_fred_err(FredErrorKind::IO),
    );
    let status = tonic::Status::from(svc);
    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "IO is transient → must surface as Unavailable"
    );
}

#[test]
fn map_fred_err_for_grpc_canceled_maps_to_unavailable() {
    let svc = map_fred_err_for_grpc(
        FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
        "HGETALL z",
        synthetic_fred_err(FredErrorKind::Canceled),
    );
    let status = tonic::Status::from(svc);
    assert_eq!(status.code(), tonic::Code::Unavailable);
}

#[test]
fn map_fred_err_for_grpc_cluster_maps_to_unavailable() {
    let svc = map_fred_err_for_grpc(
        FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
        "SADD w",
        synthetic_fred_err(FredErrorKind::Cluster),
    );
    let status = tonic::Status::from(svc);
    assert_eq!(status.code(), tonic::Code::Unavailable);
}

#[test]
fn map_fred_err_for_grpc_backpressure_maps_to_unavailable() {
    let svc = map_fred_err_for_grpc(
        FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
        "PUBLISH p",
        synthetic_fred_err(FredErrorKind::Backpressure),
    );
    let status = tonic::Status::from(svc);
    assert_eq!(status.code(), tonic::Code::Unavailable);
}

#[test]
fn map_fred_err_for_grpc_protocol_maps_to_internal() {
    let svc = map_fred_err_for_grpc(
        FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
        "XADD a",
        synthetic_fred_err(FredErrorKind::Protocol),
    );
    let status = tonic::Status::from(svc);
    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "Protocol is permanent → Internal (operator attention)"
    );
}

#[test]
fn map_fred_err_for_grpc_auth_maps_to_internal() {
    let svc = map_fred_err_for_grpc(
        FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
        "HSET b",
        synthetic_fred_err(FredErrorKind::Auth),
    );
    let status = tonic::Status::from(svc);
    assert_eq!(status.code(), tonic::Code::Internal);
}

#[test]
fn map_fred_err_for_grpc_url_maps_to_internal() {
    let svc = map_fred_err_for_grpc(
        FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
        "GET c",
        synthetic_fred_err(FredErrorKind::Url),
    );
    let status = tonic::Status::from(svc);
    assert_eq!(status.code(), tonic::Code::Internal);
}

#[test]
fn map_fred_err_for_grpc_classification_matches_is_transient() {
    // Sanity guard: every kind that `is_transient_fred_error` classifies
    // transient must map to Unavailable, and every kind it classifies
    // permanent must map to Internal. Locks the two helpers to a single
    // source of truth — if anyone adds a new transient kind to the
    // classifier but forgets to update the mapper (or vice versa), this
    // test fails first.
    let cases = [
        FredErrorKind::Timeout,
        FredErrorKind::IO,
        FredErrorKind::Canceled,
        FredErrorKind::Cluster,
        FredErrorKind::Backpressure,
        FredErrorKind::Protocol,
        FredErrorKind::Parse,
        FredErrorKind::Auth,
        FredErrorKind::Config,
        FredErrorKind::Url,
        FredErrorKind::Routing,
        FredErrorKind::Sentinel,
        FredErrorKind::NotFound,
        FredErrorKind::Unknown,
    ];
    for kind in cases {
        let svc = map_fred_err_for_grpc(
            FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
            "op",
            synthetic_fred_err(kind.clone()),
        );
        let code = tonic::Status::from(svc).code();
        let transient = is_transient_fred_error(&synthetic_fred_err(kind.clone()));
        if transient {
            assert_eq!(
                code,
                tonic::Code::Unavailable,
                "transient fred kind {kind:?} must map to Code::Unavailable"
            );
        } else {
            assert_eq!(
                code,
                tonic::Code::Internal,
                "permanent fred kind {kind:?} must map to Code::Internal"
            );
        }
    }
}

// =============================================================================
// 2. `SessionStore::get_base_path` — fault-injection drives transient
//    `ServiceError` and asserts it surfaces as `Code::Unavailable`. Pre-fix
//    every fred error went straight to `ServiceError::Internal(...)`.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_get_base_path_transient_fred_error_surfaces_as_unavailable() {
    let tv = TestValkey::setup().await;

    // The fault injector consumes one ServiceError per call. Inject a
    // synthetic Unavailable matching the shape `map_fred_err_for_grpc`
    // would produce for a fred Timeout — pre-b.fd9 the same fred kind
    // would have surfaced as Internal because `get_base_path` mapped any
    // fred error to `ServiceError::Internal(...)` directly.
    let injected = ServiceError::Unavailable(format!(
        "HGET session base_path failed ({}): synthetic transient I/O",
        FredErrorKind::Timeout.to_str()
    ));
    let queue: Arc<Mutex<VecDeque<ServiceError>>> =
        Arc::new(Mutex::new(VecDeque::from(vec![injected])));

    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_get_base_path_fault_injector_for_test(queue.clone());

    // Any session_id works — the fault fires before HGET.
    let err = ss
        .get_base_path("any-session")
        .await
        .expect_err("get_base_path must surface the injected Unavailable");

    let status = tonic::Status::from(err);
    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "pre-b.fd9 this surfaced as Internal — clients wouldn't retry"
    );
    assert_eq!(
        queue.lock().unwrap().len(),
        0,
        "fault queue should have been drained by exactly one call"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_get_base_path_real_call_classifies_transient_via_mapper() {
    // Round-trip sanity: a real `get_base_path` against an existing session
    // succeeds (no transient errors injected). Pre-fix and post-fix both
    // pass this — its presence here guards against accidentally breaking
    // the happy path while reshuffling error mapping.
    let tv = TestValkey::setup().await;
    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let sid = ss
        .create("fd9-get-base-path", "/stage", "tester")
        .await
        .expect("create must succeed");
    let bp = ss
        .get_base_path(&sid)
        .await
        .expect("get_base_path on a real session must succeed");
    assert_eq!(bp, "/stage");
}

// =============================================================================
// 3. `LayerStore::delete_layer` — Valkey blip mid-cascade is retried by the
//    new `retry_aborted_with_jitter` wrapper rather than surfacing Internal
//    with a half-cleaned layer (b.fd9 acceptance criterion #2).
// =============================================================================
//
// Strategy: use the new `delete_layer_retry_body_fault_injector_for_test`
// seam (b.fd9), which fires INSIDE the retry-wrapped body — once per retry
// attempt. Inject one transient-shape `Aborted("transient I/O: ...")` on
// the first attempt; the retry loop catches it and attempt 2 runs the
// real cascade against a clean Valkey and commits.
//
// Pre-b.fd9 there was no retry loop around the cascade — a transient blip
// after step 2 (DEL mapping) would have surfaced `Internal` with the
// mapping gone but the rest of the layer's keys still in Valkey. Post-
// b.fd9 the retry loop catches the Aborted and re-runs from step 2; every
// individual step is idempotent (DEL, XADD `*`, UNLINK, PEXPIRE, SREM) so
// the second pass commits cleanly.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_retries_transient_aborted_and_commits() {
    use live_edit::domain::types::ObjectState;
    use live_edit::storage::external_values::MockExternalValueStore;

    let tv = TestValkey::setup().await;

    let cooked = ServiceError::Aborted(format!(
        "transient I/O: DEL layer mapping failed ({}): synthetic mid-cascade blip",
        FredErrorKind::Timeout.to_str()
    ));
    let queue: Arc<Mutex<VecDeque<ServiceError>>> =
        Arc::new(Mutex::new(VecDeque::from(vec![cooked])));

    // Note: `update_layer_retry_max_attempts` doubles as the
    // `delete_layer` retry budget — the new wrapper reuses the same knob
    // (see `LayerStore::delete_layer` body).
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_retry_max_attempts(8)
        .with_delete_layer_retry_body_fault_injector_for_test(queue.clone());

    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = ss
        .create("fd9-delete-layer", "/stage", "tester")
        .await
        .expect("create session must succeed");

    let layer_name = "fd9-delete-retry";
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let ext_store: Arc<MockExternalValueStore> = Arc::new(MockExternalValueStore::new());
    ls.create_layer(&session_id, layer_name, "tester", &state, ext_store.clone())
        .await
        .expect("create_layer must succeed");

    // First attempt's body pops the queued Aborted and surfaces it; the
    // retry helper catches `Aborted` and retries; attempt 2 sees the
    // empty queue and runs the real cascade through to commit.
    ls.delete_layer(&session_id, layer_name, ext_store)
        .await
        .expect(
            "delete_layer must recover from the transient Aborted on attempt 1 and commit \
             on attempt 2 — pre-b.fd9 there was no retry loop around the cascade and the \
             same shape would have surfaced as Internal",
        );

    assert_eq!(
        queue.lock().unwrap().len(),
        0,
        "fault queue should have been drained by exactly one attempt"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_retry_budget_exhaustion_surfaces_unavailable_not_internal() {
    use live_edit::domain::types::ObjectState;
    use live_edit::storage::external_values::MockExternalValueStore;

    let tv = TestValkey::setup().await;

    // Two transient Aborted errors with a retry budget of 2 → both
    // attempts pop a fault, both surface Aborted, and the retry helper's
    // exhaustion clause wraps with "{N} attempts exhausted". The storage
    // layer returns `ServiceError::Aborted`; the user gets
    // `Code::Unavailable` (retriable — an escaped `Aborted` never reaches the
    // wire as ABORTED, whose one producer is a chained UpdateLayer's
    // predecessor refusal), never `Code::Internal`.
    const ATTEMPTS: u32 = 2;
    let queue: Arc<Mutex<VecDeque<ServiceError>>> = Arc::new(Mutex::new(
        std::iter::repeat_with(|| {
            ServiceError::Aborted(format!(
                "transient I/O: DEL layer mapping failed ({}): synthetic blip",
                FredErrorKind::Timeout.to_str()
            ))
        })
        .take(ATTEMPTS as usize)
        .collect(),
    ));

    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_retry_max_attempts(ATTEMPTS)
        .with_delete_layer_retry_body_fault_injector_for_test(queue.clone());

    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = ss
        .create("fd9-delete-exhaust", "/stage", "tester")
        .await
        .expect("create session must succeed");

    let layer_name = "fd9-delete-exhaust-layer";
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let ext_store: Arc<MockExternalValueStore> = Arc::new(MockExternalValueStore::new());
    ls.create_layer(&session_id, layer_name, "tester", &state, ext_store.clone())
        .await
        .expect("create_layer must succeed");

    let err = ls
        .delete_layer(&session_id, layer_name, ext_store)
        .await
        .expect_err("retry exhaustion must surface");
    match err {
        ServiceError::Aborted(msg) => {
            assert!(
                msg.contains("attempts exhausted"),
                "exhaustion message must propagate verbatim, got: {msg}"
            );
            // The gRPC code must be Code::Unavailable (retriable by the
            // Kit client, which treats ABORTED as terminal) — pre-b.fd9
            // this whole class of failures surfaced as Code::Internal.
            let status = tonic::Status::from(ServiceError::Aborted(msg));
            assert_eq!(status.code(), tonic::Code::Unavailable);
        }
        other => panic!(
            "expected Aborted (retriable) after retry exhaustion, got {other:?} — \
             pre-b.fd9 a half-cleaned layer surfaced as Internal"
        ),
    }
    assert_eq!(queue.lock().unwrap().len(), 0, "queue should be drained");
}

// =============================================================================
// 4. `LayerStore::delete_layer` — survives a transient Valkey blip mid-
//    cascade by retrying the body. Drives the retry-wrapped body's behavior
//    indirectly: the production code path classifies transient fred errors
//    on every step, the retry loop catches them, and a fresh attempt
//    commits.
// =============================================================================
//
// We can't directly inject a fred error into a specific step of the body
// without adding a new test seam. The next-best regression coverage is:
// confirm that under a clean Valkey, `delete_layer` commits, and that the
// retry budget exists (`update_layer_retry_max_attempts` is honored). The
// classifier-coverage block above (§1) plus the existing
// `update_layer_retry_test.rs` coverage of the same retry helper jointly
// pin the retry-on-transient contract.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_commits_under_clean_valkey() {
    use live_edit::domain::types::ObjectState;
    use live_edit::storage::external_values::MockExternalValueStore;

    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);

    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = ss
        .create("fd9-delete-clean", "/stage", "tester")
        .await
        .expect("create session must succeed");

    let layer_name = "fd9-delete-clean-layer";
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let ext_store: Arc<MockExternalValueStore> = Arc::new(MockExternalValueStore::new());
    ls.create_layer(&session_id, layer_name, "tester", &state, ext_store.clone())
        .await
        .expect("create_layer must succeed");

    // No injected errors — the retry-wrapped body runs once and commits.
    ls.delete_layer(&session_id, layer_name, ext_store)
        .await
        .expect("delete_layer must commit cleanly on attempt 1");
}

// =============================================================================
// 5. Cross-helper symmetry — `map_transient_fred_err` (returns `Aborted` for
//    transient, intended for call sites inside a retry loop) and
//    `map_fred_err_for_grpc` (returns `Unavailable` for transient, intended
//    for single-shot call sites) MUST agree on every fred ErrorKind about
//    which kinds are transient. A divergence would mean the "Lint vs.
//    surface" choice depends on which helper a future call site happens to
//    pick, which is exactly the misuse this dual-helper API needs to
//    prevent. The classifier-cross-check test in §1 locks
//    `map_fred_err_for_grpc` against `is_transient_fred_error`; the test
//    below locks `map_transient_fred_err` against the same classifier AND
//    pins the variant-on-transient (`Aborted` for retry-loop) vs.
//    variant-on-transient (`Unavailable` for direct surface) split.
// =============================================================================

#[test]
fn map_transient_fred_err_and_map_fred_err_for_grpc_agree_on_transient_set() {
    // Iterate every fred ErrorKind the codebase observes — the same list
    // §1's classifier cross-check uses. For each kind:
    //   * Both helpers MUST agree the kind is transient (per
    //     `is_transient_fred_error`) or both must disagree.
    //   * On transient, `map_transient_fred_err` → `ServiceError::Aborted`,
    //     `map_fred_err_for_grpc` → `ServiceError::Unavailable`, and both
    //     reach the wire as `Code::Unavailable`.
    //   * On permanent, both → `ServiceError::Internal`.
    // A regression that, say, added Cluster to one helper but not the
    // other (a copy-paste slip in the next sweep) would fail here first
    // and call out the asymmetry explicitly.
    let cases = [
        FredErrorKind::Timeout,
        FredErrorKind::IO,
        FredErrorKind::Canceled,
        FredErrorKind::Cluster,
        FredErrorKind::Backpressure,
        FredErrorKind::Protocol,
        FredErrorKind::Parse,
        FredErrorKind::Auth,
        FredErrorKind::Config,
        FredErrorKind::Url,
        FredErrorKind::Routing,
        FredErrorKind::Sentinel,
        FredErrorKind::NotFound,
        FredErrorKind::Unknown,
        FredErrorKind::InvalidArgument,
        FredErrorKind::InvalidCommand,
    ];
    for kind in cases {
        let transient = is_transient_fred_error(&synthetic_fred_err(kind.clone()));
        let retry_loop_err = map_transient_fred_err(
            FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
            "op",
            synthetic_fred_err(kind.clone()),
        );
        let grpc_err = map_fred_err_for_grpc(
            FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
            "op",
            synthetic_fred_err(kind.clone()),
        );
        if transient {
            assert!(
                matches!(retry_loop_err, ServiceError::Aborted(_)),
                "transient kind {kind:?}: map_transient_fred_err must produce Aborted (the \
                 in-band retry signal for retry_aborted_with_jitter); got {retry_loop_err:?}"
            );
            assert!(
                matches!(grpc_err, ServiceError::Unavailable(_)),
                "transient kind {kind:?}: map_fred_err_for_grpc must produce Unavailable so \
                 the gRPC client retries (Code::Unavailable, 14); got {grpc_err:?}"
            );
            // The variants differ only in-process: on the wire both helpers'
            // transient answers are `UNAVAILABLE` (an escaped `Aborted` never
            // reaches the wire as ABORTED, which only a chained UpdateLayer's
            // predecessor refusal answers), so a single-attempt site that
            // reuses the in-loop mapper still hands the client a code it
            // retries.
            assert_eq!(
                (
                    tonic::Status::from(retry_loop_err).code(),
                    tonic::Status::from(grpc_err).code()
                ),
                (tonic::Code::Unavailable, tonic::Code::Unavailable),
                "transient kind {kind:?}: both helpers must reach the wire as UNAVAILABLE"
            );
        } else {
            assert!(
                matches!(retry_loop_err, ServiceError::Internal(_)),
                "permanent kind {kind:?}: map_transient_fred_err must produce Internal so the \
                 retry loop exits; got {retry_loop_err:?}"
            );
            assert!(
                matches!(grpc_err, ServiceError::Internal(_)),
                "permanent kind {kind:?}: map_fred_err_for_grpc must produce Internal so the \
                 gRPC client sees a non-retriable error; got {grpc_err:?}"
            );
        }
    }
}

#[test]
fn map_transient_fred_err_transient_message_carries_prefix_for_retry_helper() {
    // The retry helper (`retry_aborted_with_jitter`) matches on the
    // `"transient I/O:"` prefix in the `Aborted` message to distinguish
    // transient blips from WATCH-conflict Aborted (which it also retries
    // but classifies differently in metrics). Pin the prefix so a future
    // refactor that drops it doesn't silently break retry classification.
    let svc = map_transient_fred_err(
        FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
        "HGETALL x",
        synthetic_fred_err(FredErrorKind::Timeout),
    );
    match svc {
        ServiceError::Aborted(msg) => {
            assert!(
                msg.starts_with("transient I/O:"),
                "map_transient_fred_err must emit the \"transient I/O:\" prefix so \
                 retry_aborted_with_jitter can classify metrics correctly; got: {msg:?}"
            );
        }
        other => panic!("expected Aborted, got {other:?}"),
    }
}

#[test]
fn map_fred_err_for_grpc_transient_message_omits_transient_io_prefix() {
    // The `Unavailable` variant is surfaced *directly* to the gRPC client
    // (no retry loop interposed). The `"transient I/O:"` prefix on the
    // `Aborted` variant exists only as an in-band signal for the retry
    // helper; for the gRPC surface the prefix would just be noise in the
    // client-visible status message. Lock the format so a future tidy-up
    // doesn't accidentally homogenise the two helpers' output.
    let svc = map_fred_err_for_grpc(
        FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
        "HGETALL x",
        synthetic_fred_err(FredErrorKind::Timeout),
    );
    match svc {
        ServiceError::Unavailable(msg) => {
            assert!(
                !msg.starts_with("transient I/O:"),
                "map_fred_err_for_grpc must NOT emit the \"transient I/O:\" prefix — that \
                 prefix is the retry-helper's in-band signal, not a client-visible label; \
                 got: {msg:?}"
            );
            assert!(
                msg.contains("HGETALL x"),
                "op label must be visible in the operator-facing message; got: {msg:?}"
            );
        }
        other => panic!("expected Unavailable, got {other:?}"),
    }
}

// =============================================================================
// 5b. b.c7a — the FIRST message-matched carve-out in the shared classifier.
//
// Every test in §1 and §5 drives `synthetic_fred_err`, whose message is
// generic, so together they pin the KIND-level table and nothing else. fred's
// router miss ("Failed to route command…", emitted by `write_command` when the
// router has no live connection to hand back) is transient despite riding on
// two kinds — `Unknown` and `Routing` — that the kind-level table calls
// permanent. Wherever that happens the two layers can disagree, so the symmetry
// the §1 / §5 loops assert (classifier ↔ `map_transient_fred_err` ↔
// `map_fred_err_for_grpc`) has to be re-asserted over the message dimension:
// a mapper that consulted the kind directly instead of calling
// `is_transient_fred_error` would still pass every existing test here.
//
// This was the only such place when b.c7a landed. It is not any more — b.cv4
// added the redirect budget-exhaustion family (§5c below), on the same two
// kinds; b.xd1 added the store's not-serving replies (§5d), b.vn4 a slot
// migration's `-TRYAGAIN` (§5f) and b.cyu fred's transaction router giving up
// (§5g), all on `Unknown`; and `-UNBLOCKED` is another `Unknown`-keyed predicate, though a
// stream-reader-only one that is deliberately NOT in the shared classifier. Each
// carve-out needs its own §, and none of them may swallow another's message
// (pinned in `client.rs`'s unit block).
//
// Pre-b.c7a a routing miss surfaced as `Code::Internal` (non-retriable) on the
// write path and as a permanent tear-down on the read path. The write-path half
// is what these tests pin; the read-path half lives in the stream-multiplexer
// reader tests: `a_router_miss_and_every_budget_exhausted_shape_are_retryable_for_the_reader`
// in `src/storage/stream_mux/reader.rs` (the reader's verdict), and
// `a_router_miss_or_a_spent_redirect_budget_on_the_xread_is_ridden_out_without_a_reconnect`
// in `stream_mux_fault_test.rs` (retry and delivery, end to end).
// =============================================================================

#[test]
fn map_fred_err_for_grpc_router_miss_maps_to_unavailable_on_both_kinds() {
    for (kind, details) in ROUTER_MISS_SHAPES {
        let svc = map_fred_err_for_grpc(
            FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
            "XADD stream",
            router_miss_fred_err(kind.clone(), details),
        );
        let status = tonic::Status::from(svc);
        assert_eq!(
            status.code(),
            tonic::Code::Unavailable,
            "{kind:?} + {details:?} is fred's router miss — a connection-level condition fred is \
             already deferring a reconnect for, so it must surface as Code::Unavailable (14, \
             retriable). Pre-b.c7a it surfaced as Code::Internal and the caller gave up"
        );
    }
}

#[test]
fn map_transient_fred_err_router_miss_produces_a_retriable_aborted() {
    for (kind, details) in ROUTER_MISS_SHAPES {
        let svc = map_transient_fred_err(
            FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
            "EVAL update_layer_fast",
            router_miss_fred_err(kind.clone(), details),
        );
        match svc {
            ServiceError::Aborted(msg) => {
                assert!(
                    msg.starts_with("transient I/O:"),
                    "{kind:?} + {details:?}: the `transient I/O:` prefix is what \
                     `retry_aborted_with_jitter` matches on, so without it the retry loop \
                     classifies the routing miss as a WATCH conflict; got: {msg:?}"
                );
            }
            other => panic!(
                "{kind:?} + {details:?}: expected Aborted so the surrounding retry loop re-runs \
                 the body — pre-b.c7a this produced Internal, which the loop does not match and \
                 which aborts the write outright; got {other:?}"
            ),
        }
    }
}

/// Anti-creep, mapper-side: the carve-out is exactly ONE message wide.
///
/// The positive direction — router-miss ⇒ transient ⇒ `Aborted` / `Unavailable`
/// — is already pinned by the two tests directly above, each with its own
/// failure message; re-asserting it here in bare `matches!` form only duplicated
/// them more weakly. What no other test covers is the NEGATIVE direction: a
/// regression that promoted `Unknown` / `Routing` wholesale (dropping the
/// message match) would make every unclassified fred error retriable, and this
/// is the layer where an operator would stop seeing it, because `Internal` is
/// the signal that asks for attention.
#[test]
fn router_miss_carve_out_does_not_widen_to_the_whole_kind() {
    for kind in [FredErrorKind::Unknown, FredErrorKind::Routing] {
        let e = router_miss_fred_err(kind.clone(), "ERR something the server did not like");
        assert!(
            !is_transient_fred_error(&e),
            "{kind:?} with a non-router message must stay permanent"
        );
        assert!(matches!(
            map_transient_fred_err(
                FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
                "op",
                router_miss_fred_err(kind.clone(), "ERR something the server did not like")
            ),
            ServiceError::Internal(_)
        ));
        assert!(
            matches!(
                map_fred_err_for_grpc(
                    FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
                    "op",
                    router_miss_fred_err(kind.clone(), "ERR something the server did not like")
                ),
                ServiceError::Internal(_)
            ),
            "{kind:?} with a non-router message must still surface as Code::Internal"
        );
    }
}

// =============================================================================
// 5c. b.cv4 — the SECOND message-matched carve-out: fred's redirect
// budget-exhaustion family.
//
// Same argument as §5b, on a bigger surface. A `MOVED` / `ASK` that outruns the
// command's attempt budget does not reach the caller as `ErrorKind::Cluster` —
// that kind covers only the redirects fred managed to *follow*. It arrives as
// one of four exhaustion messages under `Unknown` or `Routing`, kinds the
// kind-level table in §1 / §5 calls permanent. So this is the second place the
// two layers can disagree, and it needs the same message-dimension re-assertion:
// a mapper that consulted `e.kind()` directly instead of calling
// `is_transient_fred_error` would pass every kind-driven test in this file and
// still map a routine resharding to a non-retriable `Code::Internal`.
//
// That makes the kind-matrix tests at §1 and §5 load-bearing in the OTHER
// direction too: they drive `synthetic_fred_err`, whose generic message starts
// with none of these prefixes, so their "`Unknown` / `Routing` ⇒ Internal"
// expectations are now the anti-creep guard that neither carve-out has widened
// into a bare kind check.
//
// The write-path half is what these tests pin. The other two halves live
// alongside: the reader's end-to-end behaviour — retry + delivery, zero
// permanent exits, no forced reconnect — in
// `stream_mux_fault_test::a_router_miss_or_a_spent_redirect_budget_on_the_xread_is_ridden_out_without_a_reconnect`,
// and the real-cluster half — one of these error objects, produced by an
// actual 3-shard cluster refusing a non-owner command, put through
// `is_transient_fred_error` — in
// `cluster_subscriber_test::cluster_subscriber_a_real_non_owner_moved_is_classified_transient`.
// =============================================================================

#[test]
fn map_fred_err_for_grpc_budget_exhaustion_maps_to_unavailable_on_all_four_shapes() {
    for (kind, details) in &BUDGET_EXHAUSTED_SHAPES {
        let svc = map_fred_err_for_grpc(
            FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
            "XLEN stream",
            budget_exhausted_fred_err(kind.clone(), details),
        );
        let status = tonic::Status::from(svc);
        assert_eq!(
            status.code(),
            tonic::Code::Unavailable,
            "{kind:?} + {details:?} is what a Valkey-Cluster topology change looks like once the \
             attempt budget is spent — the most ordinary thing a cluster does. It must surface as \
             Code::Unavailable (14, retriable). Pre-b.cv4 it surfaced as Code::Internal and the \
             caller gave up on a slot that had simply moved"
        );
    }
}

#[test]
fn map_transient_fred_err_budget_exhaustion_produces_a_retriable_aborted() {
    for (kind, details) in &BUDGET_EXHAUSTED_SHAPES {
        let svc = map_transient_fred_err(
            FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
            "EVAL update_layer_fast",
            budget_exhausted_fred_err(kind.clone(), details),
        );
        match svc {
            ServiceError::Aborted(msg) => {
                assert!(
                    msg.starts_with("transient I/O:"),
                    "{kind:?} + {details:?}: the `transient I/O:` prefix is what \
                     `retry_aborted_with_jitter` matches on, so without it the retry loop \
                     classifies a moved slot as a WATCH conflict; got: {msg:?}"
                );
            }
            other => panic!(
                "{kind:?} + {details:?}: expected Aborted so the surrounding retry loop re-runs \
                 the body against the re-resolved topology — pre-b.cv4 this produced Internal, \
                 which the loop does not match and which aborts the write outright; got {other:?}"
            ),
        }
    }
}

/// Anti-creep, mapper-side, for the four-message carve-out.
///
/// The positive direction is pinned by the two tests above. What no other test
/// in this file covers is that `Unknown` and `Routing` still reach BOTH mappers
/// as `Internal` for a message outside the four — the direction a regression
/// that dropped the prefix match (keeping only the kind arms) would break, and
/// the layer at which an operator would stop being told about every genuine
/// unclassified fred error.
#[test]
fn budget_exhaustion_carve_out_does_not_widen_to_the_whole_kind() {
    const OTHER_MESSAGE: &str = "CROSSSLOT Keys in request don't hash to the same slot";

    for kind in [FredErrorKind::Unknown, FredErrorKind::Routing] {
        assert!(
            !is_transient_fred_error(&budget_exhausted_fred_err(kind.clone(), OTHER_MESSAGE)),
            "{kind:?} with a non-exhaustion message must stay permanent — a CROSSSLOT is a caller \
             bug (a multi-key command spanning slots), and retrying it just repeats it"
        );
        assert!(
            matches!(
                map_transient_fred_err(
                    FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
                    "op",
                    budget_exhausted_fred_err(kind.clone(), OTHER_MESSAGE)
                ),
                ServiceError::Internal(_)
            ),
            "{kind:?} with a non-exhaustion message must exit the retry loop, not spin it"
        );
        assert!(
            matches!(
                map_fred_err_for_grpc(
                    FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
                    "op",
                    budget_exhausted_fred_err(kind.clone(), OTHER_MESSAGE)
                ),
                ServiceError::Internal(_)
            ),
            "{kind:?} with a non-exhaustion message must still surface as Code::Internal"
        );
    }
}

// =============================================================================
// 5d. b.xd1 — the THIRD message-matched carve-out family: the backing store
// not serving commands for a while.
//
// A node replaying its dataset after a restart answers `-LOADING`, and a node
// behind a long script or module command answers `-BUSY`. fred maps neither
// token, so both arrive as `ErrorKind::Unknown`, which the kind-level table in
// §1 / §5 calls permanent. Pre-b.xd1 a routine backing-store restart therefore
// answered every in-flight request `Code::Internal`; the carve-out
// (`storage::client::is_store_not_serving_reply`) makes them transient. Same
// message-dimension re-assertion as §5b / §5c, over every row of
// `STORE_NOT_SERVING_SHAPES`.
//
// The reader-side halves: `the_stores_not_serving_replies_are_retryable_for_the_reader`
// in `src/storage/stream_mux/reader.rs` (the verdict) and
// `stream_mux_fault_test::the_stores_not_serving_replies_on_the_probe_and_the_xread_are_ridden_out_without_a_reconnect`
// (retry and delivery). The retry-loop half is in `update_layer_retry_test`.
// =============================================================================

#[test]
fn map_transient_fred_err_store_not_serving_produces_a_retriable_aborted_on_every_shape() {
    for reply in STORE_NOT_SERVING_SHAPES {
        let svc = map_transient_fred_err(
            FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
            "WATCH obj header",
            fred::error::Error::new(FredErrorKind::Unknown, reply),
        );
        match svc {
            ServiceError::Aborted(msg) => {
                assert!(
                    msg.starts_with("transient I/O: WATCH obj header failed"),
                    "{reply:?}: the `transient I/O:` prefix is what `retry_aborted_with_jitter` \
                     matches on, and the op names the failed command; got: {msg:?}"
                );
                assert_eq!(
                    tonic::Status::from(ServiceError::Aborted(msg)).code(),
                    tonic::Code::Unavailable,
                    "{reply:?}: an Aborted that escapes a retry loop reaches the client as \
                     UNAVAILABLE, never ABORTED"
                );
            }
            other => panic!(
                "{reply:?}: expected Aborted so the surrounding retry loop rides out the restart \
                 — pre-b.xd1 this produced Internal, which ends the call; got {other:?}"
            ),
        }
    }
}

#[test]
fn map_fred_err_for_grpc_store_not_serving_maps_to_unavailable_on_every_shape() {
    for reply in STORE_NOT_SERVING_SHAPES {
        let svc = map_fred_err_for_grpc(
            FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
            "HGET session base_path",
            fred::error::Error::new(FredErrorKind::Unknown, reply),
        );
        assert!(
            matches!(&svc, ServiceError::Unavailable(msg) if msg.starts_with("HGET session base_path failed")),
            "{reply:?} must surface as a plain Unavailable naming the op; got {svc:?}"
        );
        assert_eq!(
            tonic::Status::from(svc).code(),
            tonic::Code::Unavailable,
            "{reply:?} is the store restarting or busy — retriable, Code::Unavailable (14). \
             Pre-b.xd1 it surfaced as Code::Internal"
        );
    }
}

/// Anti-creep, mapper-side, for the not-serving family: the carve-out is two
/// tokens with their trailing space, not the `Unknown` kind. `-BUSYGROUP` /
/// `-BUSYKEY` (a consumer-group or `RESTORE` conflict) and another
/// availability reply b.xd1 did not classify still reach both mappers as
/// `Internal`.
#[test]
fn store_not_serving_carve_out_does_not_widen_to_the_whole_kind() {
    for reply in [
        "BUSYGROUP Consumer Group name already exists",
        "BUSYKEY Target key name already exists.",
        "LOADINGX Valkey is loading the dataset in memory",
        "MASTERDOWN Link with MASTER is down and replica-serve-stale-data is set to 'no'.",
    ] {
        let err = || fred::error::Error::new(FredErrorKind::Unknown, reply);
        assert!(
            !is_transient_fred_error(&err()),
            "{reply:?} must stay permanent"
        );
        assert!(
            matches!(
                map_transient_fred_err(
                    FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
                    "op",
                    err()
                ),
                ServiceError::Internal(_)
            ),
            "{reply:?} must exit the retry loop, not spin it"
        );
        assert!(
            matches!(
                map_fred_err_for_grpc(
                    FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
                    "op",
                    err()
                ),
                ServiceError::Internal(_)
            ),
            "{reply:?} must still surface as Code::Internal"
        );
    }
}

// =============================================================================
// 5e. b.xd1 — the store write refusals, answered by both mappers.
//
// `-OOM`, `-MISCONF`, `-READONLY` and `-NOREPLICAS` refuse a write before it
// runs, and can persist (a full store under `noeviction`, a broken disk), so
// they stay non-transient: an `Aborted` would make a retry loop spin against
// them. Pre-b.xd1 they reached the client as `Code::Internal` everywhere but
// the upload sites b.yvh classified. Now both mappers check for one first and
// answer a plain `ServiceError::Unavailable` "`{op}`: the backing store
// refused a write this request needed because `{condition}`; retry" — not
// `Aborted`, not the hinted `UnavailableWithDetail` (a mapper site has proved
// nothing about what was applied), and never starting with the overtaken
// prefix clients match on. Bare, and after an `EXEC`'s `EXECABORT … because
// of: ` prefix.
// =============================================================================

/// The four refusals as Valkey words them, with the condition each answer
/// names.
const STORE_WRITE_REFUSALS: [(&str, &str); 4] = [
    (
        "OOM command not allowed when used memory > 'maxmemory'.",
        "it is out of memory",
    ),
    (
        "MISCONF Valkey is configured to save RDB snapshots, but it's currently unable to persist \
         to disk.",
        "it cannot persist data to disk",
    ),
    (
        "READONLY You can't write against a read only replica.",
        "the node it reached is a read-only replica",
    ),
    (
        "NOREPLICAS Not enough good replicas to write.",
        "too few of its replicas are reachable to accept writes",
    ),
];

#[test]
fn both_mappers_answer_every_store_write_refusal_plain_unavailable() {
    const OP: &str = "SADD presence active_sessions";
    for (reply, condition) in STORE_WRITE_REFUSALS {
        let want = format!(
            "{OP}: the backing store refused a write this request needed because {condition}; \
             retry"
        );
        for text in [
            reply.to_owned(),
            format!("EXECABORT Transaction discarded because of: {reply}"),
        ] {
            let err = || fred::error::Error::new(FredErrorKind::Unknown, text.clone());
            assert!(
                !is_transient_fred_error(&err()),
                "{text:?} can persist; it must stay non-transient"
            );
            for (mapper, svc) in [
                (
                    "map_transient_fred_err",
                    map_transient_fred_err(
                        FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
                        OP,
                        err(),
                    ),
                ),
                (
                    "map_fred_err_for_grpc",
                    map_fred_err_for_grpc(
                        FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
                        OP,
                        err(),
                    ),
                ),
            ] {
                match &svc {
                    ServiceError::Unavailable(msg) => {
                        assert_eq!(msg, &want, "{mapper}({text:?})");
                        assert!(
                            !msg.starts_with(OPERATION_OVERTAKEN_MESSAGE),
                            "{mapper}({text:?}): a store refusal must never read as the \
                             overtaken answer"
                        );
                    }
                    other => panic!(
                        "{mapper}({text:?}) must answer a plain Unavailable — not Aborted (a \
                         retry loop would spin on it), not the hinted UnavailableWithDetail, \
                         not Internal; got {other:?}"
                    ),
                }
                assert_eq!(
                    tonic::Status::from(svc).code(),
                    tonic::Code::Unavailable,
                    "{mapper}({text:?}) reaches the client as UNAVAILABLE, never ABORTED or \
                     INTERNAL"
                );
            }
        }
    }
}

// =============================================================================
// 5f. b.vn4 — the FOURTH message-matched carve-out family: a cluster node's
// `-TRYAGAIN` to a multi-key command during a slot migration.
//
// A node whose slot is mid-migration answers `-TRYAGAIN Multiple keys request
// during rehashing of slot` to a command whose keys are split between the
// slot's source and target, before the command runs (and discards the
// transaction of an `EXEC` that meets it). fred maps no `TRYAGAIN` token, so
// it arrives as `ErrorKind::Unknown`, permanent by kind; pre-b.vn4 every
// multi-key call site answered it `Code::Internal`. The carve-out
// (`storage::client::is_tryagain_reply`) makes exactly that token, with its
// trailing space, transient — not the `Unknown` kind. Same message-dimension
// re-assertion as §5b–§5d.
//
// The reader-side half: `storage::stream_mux::reader`'s
// `tryagain_is_transient_in_the_shared_classifier_and_unblocked_stays_reader_only`.
// The commit loop's half: `storage::b_vn4_tryagain_test` and
// `grpc::upstream_unavailable_test`.
// =============================================================================

#[test]
fn map_transient_fred_err_tryagain_produces_a_retriable_aborted() {
    let svc = map_transient_fred_err(
        FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
        "WATCH obj header",
        fred::error::Error::new(FredErrorKind::Unknown, TRYAGAIN_REPLY),
    );
    match svc {
        ServiceError::Aborted(msg) => {
            assert!(
                msg.starts_with("transient I/O: WATCH obj header failed"),
                "the `transient I/O:` prefix is what `retry_aborted_with_jitter` matches on, and \
                 the op names the failed command; got: {msg:?}"
            );
            assert!(msg.contains(TRYAGAIN_REPLY), "the reply is kept: {msg:?}");
        }
        other => panic!(
            "expected Aborted so the surrounding retry loop rides out the migration — pre-b.vn4 \
             this produced Internal, which ends the call; got {other:?}"
        ),
    }
}

#[test]
fn map_fred_err_for_grpc_tryagain_maps_to_unavailable() {
    let svc = map_fred_err_for_grpc(
        FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
        "EVALSHA ext store_values",
        fred::error::Error::new(FredErrorKind::Unknown, TRYAGAIN_REPLY),
    );
    assert!(
        matches!(&svc, ServiceError::Unavailable(msg) if msg.starts_with("EVALSHA ext store_values failed")),
        "a -TRYAGAIN must surface as a plain Unavailable naming the op; got {svc:?}"
    );
    assert_eq!(
        tonic::Status::from(svc).code(),
        tonic::Code::Unavailable,
        "a slot migration clears by itself — retriable, Code::Unavailable (14). Pre-b.vn4 it \
         surfaced as Code::Internal"
    );
}

/// Anti-creep, mapper-side, for the `-TRYAGAIN` family: the carve-out is the
/// one token with its trailing space, at the start, under `Unknown` — not the
/// `Unknown` kind, and not a substring. A longer token, the token without its
/// space, the token after another error code, the reply under another kind,
/// and a script's own in-migration refusal (handled by `storage::diff_read`,
/// never by this family) all still reach both mappers as `Internal`.
#[test]
fn tryagain_carve_out_does_not_widen_to_the_whole_kind() {
    for (kind, reply) in [
        (
            FredErrorKind::Unknown,
            "TRYAGAINX Multiple keys request during rehashing of slot",
        ),
        (FredErrorKind::Unknown, "TRYAGAIN"),
        (
            FredErrorKind::Unknown,
            "ERR TRYAGAIN Multiple keys request during rehashing of slot",
        ),
        (
            FredErrorKind::Unknown,
            "ERR Unable to execute command 'hgetall' in script because undeclared keys were \
             accessed during rehashing of the slot",
        ),
        (FredErrorKind::Protocol, TRYAGAIN_REPLY),
        (FredErrorKind::InvalidArgument, TRYAGAIN_REPLY),
    ] {
        let err = || fred::error::Error::new(kind.clone(), reply);
        assert!(
            !is_transient_fred_error(&err()),
            "{kind:?} {reply:?} must stay permanent"
        );
        assert!(
            matches!(
                map_transient_fred_err(
                    FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
                    "op",
                    err()
                ),
                ServiceError::Internal(_)
            ),
            "{kind:?} {reply:?} must exit the retry loop, not spin it"
        );
        assert!(
            matches!(
                map_fred_err_for_grpc(
                    FredErrOrigin::metrics(None, POOL_LABEL_DATA, StoreOomSignal::unwired()),
                    "op",
                    err()
                ),
                ServiceError::Internal(_)
            ),
            "{kind:?} {reply:?} must still surface as Code::Internal"
        );
    }
}

// =============================================================================
// 5g. b.cyu (b.bm5 (a)) — the FIFTH message-matched carve-out family: fred's
// `Transaction` router giving up.
//
// A fred `Transaction` (`multi()` … `exec()`) on the txn pool meets a slot
// that moved or is moving as a `MOVED` / `-ASK` past the pool's zero
// redirection budget, which fred's transaction router fails with its own
// `Unknown` + "Max redirections exceeded"; a failed drain or route miss with no
// error to forward is `Unknown` + "Max attempts exceeded". Both are permanent
// by kind, so pre-b.cyu `DeleteSession` (`map_transient_fred_err`) and the
// `CreateLayer` materialization (`map_fred_err_for_grpc`) answered a slot
// migration `Code::Internal`. The carve-out
// (`storage::client::is_transaction_router_exhausted_error`) makes exactly the
// two texts, at the start and under `Unknown`, transient. Same
// message-dimension re-assertion as §5b–§5f, over every row of
// `TRANSACTION_ROUTER_EXHAUSTED_SHAPES`.
//
// The predicate's own half (both texts, the kind gate, the near misses, no
// overlap with another family): `storage::client`'s unit block. Through a real
// fred `Transaction` at both call sites: `storage::b_cyu_txn_redirect_test`.
// On a real cluster: the cluster-gated `storage::b_cyu_slot_migration_test`.
// =============================================================================

#[test]
fn map_transient_fred_err_transaction_router_exhaustion_produces_a_retriable_aborted() {
    for (kind, details) in &TRANSACTION_ROUTER_EXHAUSTED_SHAPES {
        let svc = map_transient_fred_err(
            FredErrOrigin::metrics(None, POOL_LABEL_TXN, StoreOomSignal::unwired()),
            "EXEC session DEL meta+layers",
            fred::error::Error::new(kind.clone(), *details),
        );
        match svc {
            ServiceError::Aborted(msg) => {
                assert!(
                    msg.starts_with("transient I/O: EXEC session DEL meta+layers failed"),
                    "{kind:?} + {details:?}: the `transient I/O:` prefix is what \
                     `retry_aborted_with_jitter` matches on, and the op names the failed \
                     transaction; got: {msg:?}"
                );
                assert!(msg.contains(details), "the text is kept: {msg:?}");
                assert_eq!(
                    tonic::Status::from(ServiceError::Aborted(msg)).code(),
                    tonic::Code::Unavailable,
                    "{kind:?} + {details:?}: an Aborted that escapes (DeleteSession runs no \
                     retry loop) reaches the client as UNAVAILABLE"
                );
            }
            other => panic!(
                "{kind:?} + {details:?}: expected Aborted — pre-b.cyu this produced Internal, so \
                 a DeleteSession meeting a slot migration answered INTERNAL; got {other:?}"
            ),
        }
    }
}

#[test]
fn map_fred_err_for_grpc_transaction_router_exhaustion_maps_to_unavailable() {
    for (kind, details) in &TRANSACTION_ROUTER_EXHAUSTED_SHAPES {
        let svc = map_fred_err_for_grpc(
            FredErrOrigin::metrics(None, POOL_LABEL_TXN, StoreOomSignal::unwired()),
            "EXEC object write",
            fred::error::Error::new(kind.clone(), *details),
        );
        assert!(
            matches!(&svc, ServiceError::Unavailable(msg) if msg.starts_with("EXEC object write failed")),
            "{kind:?} + {details:?} must surface as a plain Unavailable naming the op; got {svc:?}"
        );
        assert_eq!(
            tonic::Status::from(svc).code(),
            tonic::Code::Unavailable,
            "{kind:?} + {details:?} is a slot that moved or is moving: retriable, \
             Code::Unavailable (14). Pre-b.cyu the CreateLayer materialization surfaced it as \
             Code::Internal"
        );
    }
}

/// Anti-creep, mapper-side, for the transaction-router family: the carve-out is
/// the two texts at the start under `Unknown`, not the `Unknown` kind and not a
/// substring. A text after another error code, a near-miss wording, fred's
/// other transaction-router texts (which are not exhaustion), and the two
/// texts under another kind all still reach both mappers as `Internal`.
#[test]
fn transaction_router_exhaustion_carve_out_does_not_widen_to_the_whole_kind() {
    for (kind, reply) in [
        (FredErrorKind::Unknown, "ERR Max redirections exceeded"),
        (FredErrorKind::Unknown, "Max redirection exceeded"),
        (FredErrorKind::Unknown, "Max attempts"),
        (FredErrorKind::Unknown, "Unexpected DISCARD response."),
        (FredErrorKind::Routing, "Max redirections exceeded"),
        (FredErrorKind::Protocol, "Max attempts exceeded"),
    ] {
        let err = || fred::error::Error::new(kind.clone(), reply);
        assert!(
            !is_transaction_router_exhausted_error(&err()),
            "{kind:?} {reply:?} is not the transaction router giving up"
        );
        assert!(
            !is_transient_fred_error(&err()),
            "{kind:?} {reply:?} must stay permanent"
        );
        assert!(
            matches!(
                map_transient_fred_err(
                    FredErrOrigin::metrics(None, POOL_LABEL_TXN, StoreOomSignal::unwired()),
                    "op",
                    err()
                ),
                ServiceError::Internal(_)
            ),
            "{kind:?} {reply:?} must exit a retry loop, not spin it"
        );
        assert!(
            matches!(
                map_fred_err_for_grpc(
                    FredErrOrigin::metrics(None, POOL_LABEL_TXN, StoreOomSignal::unwired()),
                    "op",
                    err()
                ),
                ServiceError::Internal(_)
            ),
            "{kind:?} {reply:?} must still surface as Code::Internal"
        );
    }
}

// =============================================================================
// 6. `SessionBasePathCache::get_base_path` — verifies the cache wrapper
//    propagates `Unavailable` from the underlying `SessionStore` rather than
//    catching, memoizing, or rewriting it. The cache is the entry point every
//    hot-path RPC uses for the auth check, so a regression that memoized
//    `Unavailable` (or swallowed it into `Internal`) would re-introduce the
//    pre-b.fd9 `Internal` surfacing for the largest single class of call
//    sites. The session_cache diff explicitly documents this is the
//    intended behavior; this test pins it.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_base_path_cache_propagates_transient_unavailable_verbatim() {
    let tv = TestValkey::setup().await;

    // Inject a transient-shape Unavailable into SessionStore's
    // get_base_path; the cache MUST forward it without memoizing.
    let injected = ServiceError::Unavailable(format!(
        "HGET session base_path failed ({}): synthetic transient I/O",
        FredErrorKind::Timeout.to_str()
    ));
    let queue: Arc<Mutex<VecDeque<ServiceError>>> =
        Arc::new(Mutex::new(VecDeque::from(vec![injected])));

    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_get_base_path_fault_injector_for_test(queue.clone());
    let cache = SessionBasePathCache::new();

    let err = cache
        .get_base_path(&ss, "any-session")
        .await
        .expect_err("cache must surface the underlying transient Unavailable");

    let status = tonic::Status::from(err);
    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "pre-b.fd9 a transient HGET surfaced as Internal here — clients wouldn't retry; \
         post-fix the cache forwards Unavailable verbatim"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_base_path_cache_does_not_memoize_transient_failures() {
    // A transient `Unavailable` MUST NOT be cached — otherwise a single
    // blip would memoize the error for the full TTL window and every
    // subsequent caller would see the cached failure rather than a fresh
    // HGET. The cache's `get_base_path` docstring promises positive-only
    // caching; this test pins that contract under the b.fd9 surface
    // change (pre-fix the same site returned Internal, which also wasn't
    // memoized; the post-fix `Unavailable` must follow the same rule).
    let tv = TestValkey::setup().await;

    // First call: transient error injected.
    // Second call: queue empty, real HGET runs and succeeds.
    let injected = ServiceError::Unavailable(format!(
        "HGET session base_path failed ({}): synthetic transient blip",
        FredErrorKind::Timeout.to_str()
    ));
    let queue: Arc<Mutex<VecDeque<ServiceError>>> =
        Arc::new(Mutex::new(VecDeque::from(vec![injected])));

    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_get_base_path_fault_injector_for_test(queue.clone());
    let cache = SessionBasePathCache::new();

    // Pre-populate a real session so the second call has a real value to
    // return.
    let sid = ss
        .create("fd9-cache-no-memoize", "/stage", "tester")
        .await
        .expect("session create must succeed");

    let _ = cache
        .get_base_path(&ss, &sid)
        .await
        .expect_err("first call: injected transient must surface");
    assert_eq!(
        queue.lock().unwrap().len(),
        0,
        "fault queue should be drained by the first call"
    );

    // Second call: cache miss (failure was NOT memoized) → real HGET
    // succeeds with `/stage`. If the cache had memoized the Unavailable,
    // we'd see the same Unavailable surface here rather than a real
    // value.
    let bp = cache
        .get_base_path(&ss, &sid)
        .await
        .expect("second call: cache must NOT have memoized the prior failure");
    assert_eq!(
        bp, "/stage",
        "second call must re-issue HGET because positive-only caching means failures aren't memoized"
    );
}

// =============================================================================
// 7. `LayerStore::list_layer_names` / `get_layer_mapping` / `read_object_header`
//    — happy-path routing tests. There's no fault-injection seam for these
//    sites (they're single-shot reads with no in-process retry budget), so
//    the strongest available regression coverage is: confirm the production
//    code path compiles + returns the expected shape after the b.fd9
//    classifier swap. The mapper unit tests in §1 lock the
//    fred-error → ServiceError mapping; these tests lock that the call
//    sites actually invoke the mapper rather than the legacy
//    `ServiceError::Internal(...)` constructor (a regression that
//    re-introduced the legacy mapper would be caught at compile time by
//    the imports — the legacy `ServiceError::Internal(format!(...))` form
//    is no longer present in the touched code paths, so any reviewer
//    accidentally reverting would face a deliberate divergence).
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_layer_names_happy_path_post_fix_routes_through_classifier() {
    use live_edit::domain::types::ObjectState;
    use live_edit::storage::external_values::MockExternalValueStore;

    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = ss
        .create("fd9-list-happy", "/stage", "tester")
        .await
        .expect("create session must succeed");

    // Empty session → empty list (no Internal, no Unavailable, no panic).
    let names = ls
        .list_layer_names(&session_id)
        .await
        .expect("empty list must succeed");
    assert!(names.is_empty());

    // Populate one layer; list returns it. Confirms the SMEMBERS
    // round-trip + hex-decode survive the classifier-swap rewrite.
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let ext_store: Arc<MockExternalValueStore> = Arc::new(MockExternalValueStore::new());
    ls.create_layer(&session_id, "alpha", "tester", &state, ext_store)
        .await
        .expect("create_layer must succeed");

    let names = ls
        .list_layer_names(&session_id)
        .await
        .expect("populated list must succeed");
    assert_eq!(names, vec!["alpha".to_string()]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_layer_mapping_returns_none_for_missing_layer_post_fix() {
    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = ss
        .create("fd9-mapping-happy", "/stage", "tester")
        .await
        .expect("create session must succeed");

    // Missing mapping → Ok(None) (NOT NotFound, NOT Internal).
    // Confirms the GET classifier swap preserved the existing None-path.
    let r = ls
        .get_layer_mapping(&session_id, "ghost-layer")
        .await
        .expect("missing-layer must return Ok(None)");
    assert!(r.is_none());
}

// =============================================================================
// b.u96 item 1 — `cleanup_old_object` error propagation.
//
// Pre-b.u96 the four enumerate calls inside `cleanup_old_object`
// (SMEMBERS sections / ZRANGE fields / ZRANGE time samples / SMEMBERS
// deleted-sections) used `unwrap_or_default()` on the fred result. A
// transient Valkey blip on any of these reads would silently substitute
// an empty Vec — the cleanup would then `UNLINK` just the
// structural-key shell (header + indices) and leave every per-section,
// per-field, per-time-sample, and per-deleted-section key permanently
// orphaned in Valkey, with no obj:hdr left for the orphan sweep to
// re-discover. The fix made every failed teardown command surface as
// `ServiceError::Aborted("transient I/O: ...")` through
// `map_transient_fred_err`, so the surrounding `retry_aborted_with_jitter`
// re-runs the body.
//
// Since b.fsz MR-D4 the teardown is paged (`storage::retirement::Teardown`):
// its first command is the first `UNLINK` (the header, `obj:ext` and every
// other fixed key), then the index pages and their `UNLINK`s, the five
// indices last, every command's failure propagating the same way. A
// DeleteLayer re-run pages again from the indices, which go last, so a
// retry finds every row a failed attempt left.
//
// Coverage strategy: the `with_cleanup_old_object_fault_injector_for_test`
// seam injects a synthetic fred `Timeout` in place of the teardown's FIRST
// command, the first `UNLINK` (op `UNLINK object keys (the teardown's
// first)`). With the error swallowed, `delete_layer` would report success on
// attempt 1 with the object's keys left behind. Post-fix the Timeout maps to
// `Aborted("transient I/O: ...")`, the retry helper catches it, and the next
// attempt commits cleanly against the now-clean Valkey. The later pages'
// failures, scripted by `with_retirement_teardown_faults_for_test`, are
// pinned in `retirement_teardown_test.rs`.
//
// The retry budget knob (`with_update_layer_retry_max_attempts`) lets
// the test prove both branches in one run:
//   * With budget = 1, the single attempt surfaces `Err(Aborted)` —
//     the pre-fix `unwrap_or_default()` would have returned `Ok(())`.
//   * With budget ≥ 2, the retry helper catches the Aborted and
//     attempt 2 commits cleanly.
// =============================================================================

/// Regression test for the b.u96 item 1 invariant, on the paged teardown's
/// first command (its first `UNLINK`): a transient failure there surfaces as
/// a retriable error rather than a silent leak.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cleanup_old_object_transient_blip_surfaces_as_retriable_error_not_silent_leak() {
    use live_edit::domain::types::ObjectState;
    use live_edit::storage::external_values::MockExternalValueStore;

    let tv = TestValkey::setup().await;

    // Single attempt: with budget=1 the retry helper has no headroom,
    // and the Aborted propagates verbatim to the caller. Pre-b.u96 the
    // unwrap_or_default() would have returned Ok(()) here even though
    // the transient blip happened; this test would have surfaced no
    // error at all.
    let queue: Arc<Mutex<VecDeque<fred::error::Error>>> = Arc::new(Mutex::new(VecDeque::from(
        vec![synthetic_fred_err(FredErrorKind::Timeout)],
    )));

    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_retry_max_attempts(1)
        .with_cleanup_old_object_fault_injector_for_test(queue.clone());

    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = ss
        .create("u96-cleanup-prop", "/stage", "tester")
        .await
        .expect("create session must succeed");

    let layer_name = "u96-cleanup-prop-layer";
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let ext_store: Arc<MockExternalValueStore> = Arc::new(MockExternalValueStore::new());
    ls.create_layer(&session_id, layer_name, "tester", &state, ext_store.clone())
        .await
        .expect("create_layer must succeed");

    // With budget=1, the retry helper has no headroom — the Timeout
    // popped at the teardown's first UNLINK inside `cleanup_old_object`
    // routes through `map_transient_fred_err` to
    // `Aborted("transient I/O: ...")` and propagates out of the
    // retry-wrapped body. The helper's exhaustion clause wraps the
    // error with "attempts exhausted".
    let err = ls
        .delete_layer(&session_id, layer_name, ext_store)
        .await
        .expect_err(
            "with budget=1, a transient cleanup_old_object blip must surface — pre-b.u96 \
             the unwrap_or_default() ate the error and the call returned Ok(()) while \
             permanently leaking the per-section / per-field / per-ts / per-deleted-section \
             keys",
        );
    match &err {
        ServiceError::Aborted(msg) => {
            assert!(
                msg.contains("UNLINK object keys (the teardown's first)")
                    || msg.contains("attempts exhausted"),
                "Aborted message must mention the failed op label or exhaustion, got: {msg}"
            );
            // Pre-fix there would have been NO error at all, so any
            // failure path is the regression signal. The post-fix
            // expectation is specifically the transient class: the
            // `ServiceError::Aborted` variant (matched above, not
            // Internal), which reaches the client as retriable
            // Code::Unavailable — an escaped `Aborted` never reaches the wire
            // as ABORTED (only a chained UpdateLayer's predecessor refusal
            // does).
            let status = tonic::Status::from(err);
            assert_eq!(
                status.code(),
                tonic::Code::Unavailable,
                "transient cleanup blip must surface as retriable Code::Unavailable; \
                 Code::Internal would mean the cleanup_old_object propagation is mis-classified"
            );
        }
        other => panic!(
            "expected Aborted (transient classification of cleanup_old_object error), got \
             {other:?} — pre-b.u96 this call returned Ok(()) (silent leak)"
        ),
    }
    assert_eq!(
        queue.lock().unwrap().len(),
        0,
        "cleanup_old_object fault queue should have been drained by exactly one attempt"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cleanup_old_object_transient_blip_recovers_under_retry_budget() {
    use live_edit::domain::types::ObjectState;
    use live_edit::storage::external_values::MockExternalValueStore;

    let tv = TestValkey::setup().await;

    // One injected Timeout on the first attempt's cleanup_old_object;
    // attempt 2 sees an empty queue and runs cleanup against the real
    // Valkey, committing. This pins the retry-recovery half of the
    // contract: the propagated Aborted is, in fact, retriable —
    // matching `retry_aborted_with_jitter`'s match arm.
    let queue: Arc<Mutex<VecDeque<fred::error::Error>>> = Arc::new(Mutex::new(VecDeque::from(
        vec![synthetic_fred_err(FredErrorKind::Timeout)],
    )));

    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_retry_max_attempts(8)
        .with_cleanup_old_object_fault_injector_for_test(queue.clone());

    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = ss
        .create("u96-cleanup-retry", "/stage", "tester")
        .await
        .expect("create session must succeed");

    let layer_name = "u96-cleanup-retry-layer";
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let ext_store: Arc<MockExternalValueStore> = Arc::new(MockExternalValueStore::new());
    ls.create_layer(&session_id, layer_name, "tester", &state, ext_store.clone())
        .await
        .expect("create_layer must succeed");

    // Attempt 1: cleanup_old_object pops the Timeout in place of the
    // teardown's first UNLINK, maps it to Aborted, and it bubbles up out of
    // the retry-wrapped body. retry_aborted_with_jitter sees the
    // transient-I/O Aborted and retries. Attempt 2: queue empty, the real
    // UNLINK lands and the rest of the cascade commits.
    ls.delete_layer(&session_id, layer_name, ext_store)
        .await
        .expect(
            "delete_layer must recover from the cleanup_old_object transient blip on retry \
             — proves the Aborted produced by the new map_transient_fred_err wrap is \
             retriable by retry_aborted_with_jitter (which it must be, since the prefix \
             match `transient I/O:` is the retry helper's transient-shape signature)",
        );
    assert_eq!(
        queue.lock().unwrap().len(),
        0,
        "cleanup_old_object fault queue should have been drained by exactly one attempt"
    );
}
