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

//! SR-1.3 conformance wrapper for the [`StorageApiCheckpointStore`] adapter.
//!
//! Task `t2.fgj.jd.yz` Subtask `t3.fgj.jd.yz.vq`. This file is the per-adapter
//! test-only entry point that drives every enumerated SR-1.3 conformance case
//! (`case_01_*` … `case_42_*`, defined in [`super::conformance`]) against a
//! freshly-built [`StorageApiCheckpointStore`] backed by a per-test
//! `wiremock::MockServer`. The cases #19–#25 landed in sibling Subtasks
//! `t3.fgj.jd.yz.de` and `t3.fgj.jd.yz.kw`.
//!
//! ## Group J — tail generations (#28–#37)
//!
//! The Epic-2 kind-aware surface: a tail put through the shared
//! `put_layer_checkpoint`, get-by-handle addressing off
//! `(object_id, snap_seq, kind)`, and a listing that interleaves both kinds
//! newest-first with the pinned Base-before-Tail tie-break at an equal
//! sequence. On this adapter a tail is the `<oid>.<to_seq>.tail` address
//! published meta-first, with no ext-refs artifact of any kind. The cases run
//! against the regular `setup` fake with no per-case route programming, and
//! carry no cfg gate and no skip gate.
//!
//! ## Per-test isolation contract
//!
//! Every case stands up its own `MockServer` via [`wiremock::MockServer::start`].
//! The server is owned by [`MockServerGuard`] alongside the shared "fake
//! storage server" `Arc<Mutex<HashMap>>` (the in-memory map that PUT/GET/HEAD/
//! LIST/DELETE share); both die together when the guard drops at end-of-test. Case
//! #17 (concurrent puts and deletes on unrelated keys) is the load-bearing
//! consumer of this isolation contract — its assertion that "every delete-side
//! `object_id` is absent after the storm" only holds if no other test's
//! lingering map state leaks into the fake.
//!
//! ## HTTP-status → CheckpointError mapping (cited from
//! `StorageApiCheckpointStore::send_classified` in
//! `src/checkpoint/adapters/storage_api.rs`)
//!
//! Per the doc-comment on `StorageApiCheckpointStore::send_classified` in
//! `src/checkpoint/adapters/storage_api.rs`, the adapter routes every outbound
//! request through `send_classified`, which maps responses to
//! [`CheckpointError`] variants per SDD §SR-4.4:
//!
//! - **5xx (500, 502, 503, 504), 429, and retriable transport errors
//!   (timeout, connect-refused, DNS, non-HTTP transport)** → `Transient`.
//!   Inner retry loop runs up to 3 attempts with full-jitter exponential
//!   backoff (100 ms → 1 s cap).
//! - **Non-401 4xx (400, 403, 405, 409, 410, 413, 415, 422, …) and TLS
//!   handshake failures** → `Permanent`. Non-retried.
//! - **404** → trait-surface `Ok(None)` for GET, `Ok(())` for DELETE
//!   (idempotent-on-missing per SR-1.3 cases #14 / #16), `Ok(false)` for the
//!   `ext_blob_exists` HEAD; 404 on PUT is reclassified as `Permanent`. A 404
//!   on a LIST is decided per page: on the first page it is an empty listing
//!   (`Ok(vec![])`) for the `sessions/` and every `layers/` listing alike; on
//!   a later page it is `Transient`. The in-process fake below answers a
//!   first-page listing that matches nothing with that 404, so the empty-store
//!   cases run the first-page row; the later-page row is pinned by the
//!   adapter's in-crate `dr_restore_read_method_tests`.
//! - **412** → `Conflict`. `put_ext_blob` carves out a content-addressed
//!   idempotency path that translates 412 to `Ok(())` per SR-1.3 case #10
//!   (re-PUT of an existing hash is a no-op); every other method translates
//!   `Conflict` to `Err(CheckpointError::Conflict)`.
//! - **401** → outer auth-attempt loop calls
//!   `ServiceTokenProvider::force_refresh()` and retries once. A second 401
//!   surfaces as `Permanent`.
//!
//! The conformance cases #1–#18 in this file exercise only the happy-path
//! and not-found surfaces (200 → trait-surface OK, 404 → trait-surface
//! Ok(None) / Ok(())); the deeper classification surfaces (Transient,
//! Permanent, Conflict, 401-refresh) land via the per-adapter unit tests in
//! `src/checkpoint/adapters/storage_api.rs::tests::classification_tests`, the
//! SR-1.3 cases #19–#22 in the sibling Subtasks above, and Group J's #35 / #36
//! (the same surface over `get_layer_checkpoint` and a tail put).
//!
//! ## Fault-injection op labels
//!
//! This adapter has no `CheckpointFaultInjector` seam — wiremock is the
//! fault injector, and `setup_with_injector` bridges the two by probing the
//! harness's injector for each known op label and mounting a priority-1
//! override on the route that label's FIRST outbound request takes. The label
//! set is carried in two places that must agree — the probe list inside
//! `setup_with_injector` and the route table in
//! `first_request_route_for_op` — and is currently:
//!
//! | Op label | Overridden route |
//! |----------|------------------|
//! | `put_layer_checkpoint` | `PUT` `by-address/` (kind-blind: covers base ext-refs / meta / OMPF and tail meta / artifact puts alike) |
//! | `get_latest_layer_checkpoint` | `GET` `data-objects/` (the LIST it issues before any GET) |
//! | `get_layer_checkpoint` | `GET` `by-address/` (get-by-handle issues no LIST) |
//! | `get_session_checkpoint` | `GET` `by-address/` (forward-compat; unimplemented, fires no HTTP yet) |
//! | `list_session_checkpoint_ids` | `GET` `data-objects/` (the `sessions/` LIST) |
//! | `ext_blob_exists` | `HEAD` `by-address/` |
//!
//! Labels name the *method*, never the artifact kind, so there is no
//! tail-qualified put or get label. `delete_layer_checkpoint` /
//! `list_layer_checkpoints` are deliberately absent: no case injects on them,
//! and an unexercised arm is dead configuration — a later case adds its arm
//! alongside itself. The overrides are route-level rather than call-ordered;
//! see the constraint that places on a case body, spelled out above the probe
//! list.
//!
//! ## Concurrency cases #17 and #18
//!
//! These two cases use `#[tokio::test(flavor = "multi_thread", worker_threads
//! = 4)]` so the spawned writer / reader / delete tasks make real progress in
//! parallel rather than cooperatively scheduling on a single thread. The
//! shared `Arc<Mutex<HashMap>>` "fake storage server" is `Send + Sync` and
//! serializes its mutations under the standard `Mutex`, which is the moral
//! analog of LocalDisk's filesystem-level atomic-rename — distinct keys do
//! not contend, and same-key writes swap atomically rather than briefly
//! clearing.
//!
//! ## In-process fake storage server
//!
//! The default-route programs (mounted in [`setup`]) implement a minimal
//! in-process emulation of the storage-api `by-address` + `data-objects/`
//! routes:
//!
//! - PUT to `<base>/v1alpha/fileobject/by-address/<percent-encoded-address>`
//!   records the request body in the shared map keyed by the
//!   percent-decoded address. Responds `201` with an opaque-handle JSON
//!   payload.
//! - GET to the same URL family responds `200` with the recorded bytes if
//!   the key is present, `404` otherwise.
//! - HEAD to the same URL family responds `204` with no body if the key is
//!   present, `404` otherwise — storage-api's `stat_by_address` shape. It
//!   serves `ext_blob_exists` and the put-rollback existence probes.
//! - LIST to `<base>/v1alpha/fileobject/data-objects/<percent-encoded-prefix>`
//!   responds `200` with an `EnumerateResult` JSON computed from the map
//!   keys filtered by the percent-decoded prefix. The default fake returns
//!   all matching keys in a single page and never hands out a continuation
//!   handle (pagination is a sibling test's concern). A prefix that holds
//!   nothing answers `404` with no body, as the real service does.
//! - DELETE to `<base>/v1alpha/fileobject/by-address/<percent-encoded-address>`
//!   responds `204` if the map had the key (the entry is removed), `404`
//!   otherwise.
//!
//! The fake's correctness target is the trait-level surface — opaque-bytes
//! round-trip with last-writer-wins semantics, newest-first ordering of
//! `<oid>.<snap_seq>.ompf` filenames, and the SR-1.3 cases #1-#18
//! contract on absent-key / idempotent-delete paths. It is the moral analog
//! of LocalDisk's `TempDir` — a freshly-allocated, in-process backing store
//! whose lifetime is tied to the per-test guard.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use percent_encoding::percent_decode_str;
use serde_json::json;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

use live_edit::auth::service_token::ServiceTokenProvider;
use live_edit::checkpoint::adapters::StorageApiCheckpointStore;
use live_edit::checkpoint::CheckpointStore;

use super::conformance::{self, CheckpointStoreGuard, CheckpointStorePair};

#[cfg(feature = "test-checkpoint-fault-injection")]
use super::conformance::CheckpointStoreInjectedPair;
#[cfg(feature = "test-checkpoint-fault-injection")]
use live_edit::checkpoint::adapters::CheckpointFaultInjector;
#[cfg(feature = "test-checkpoint-fault-injection")]
use live_edit::checkpoint::CheckpointError;

// ---------------------------------------------------------------------------
// Fixed test configuration
// ---------------------------------------------------------------------------

/// Connect-phase timeout for the adapter's `reqwest::Client`. Mirrors the
/// realistic-timeout pair in
/// `src/checkpoint/adapters/storage_api.rs::tests::layer_checkpoint_method_tests`.
const CONNECT_TIMEOUT: Duration = Duration::from_millis(5_000);

/// Per-request timeout for the adapter's `reqwest::Client`. Mirrors the
/// realistic-timeout pair noted above.
const REQUEST_TIMEOUT: Duration = Duration::from_millis(30_000);

/// Fixed base path used by every conformance case. Pinning the value lets
/// the fake server's route matchers and LIST-prefix filter agree on the
/// expected address shape — every artifact will resolve under
/// `<BASE_PATH>/.live-edit/...`. Concurrent cases (#17, #18) rely on
/// the same base path so cross-case key collisions are impossible by
/// construction (per-test fresh MockServer + fresh map).
const BASE_PATH: &str = "/projects/team-a";

/// Disambiguator in the case-#27 temp's filename.
///
/// `<final-name>.<uuid>.tmp` is the shape `LocalDiskCheckpointStore` writes and
/// the shape `parse_layer_artifact_filename` recognises, so the planted temp has
/// to wear it: this adapter never writes one itself, and the artifact it must
/// reclaim is a *foreign* tool's leftover in the same namespace. Fixed rather
/// than random because the case plants exactly one.
const CASE_27_TEMP_UUID: &str = "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9";

// ---------------------------------------------------------------------------
// Per-test guard
// ---------------------------------------------------------------------------

/// `CheckpointStoreGuard` impl that owns the `wiremock::MockServer` plus the
/// `TempDir` backing the `ServiceTokenProvider`'s polling task. Dropping the
/// guard at end-of-test releases the mock server (which tears down the
/// in-process HTTP listener) and removes the temp dir (which the polling
/// task tolerates silently per SR-6.2's startup-absent contract).
///
/// The shared "fake storage server" map (`Arc<Mutex<HashMap<String, Bytes>>>`)
/// is closure-captured by every default-route mock and therefore lives as
/// long as the `MockServer` does — when the `MockServer` is dropped here,
/// every closure that captured an `Arc` of the map drops too, and the
/// underlying allocation is released.
struct MockServerGuard {
    // `MockServer`'s drop terminates the in-process HTTP listener. The
    // field is never read after construction — holding it is the entire
    // contract.
    _server: MockServer,
    // `TempDir`'s drop removes the temp dir backing the
    // `ServiceTokenProvider`'s polling-task target. The polling task
    // tolerates a missing file silently per the SR-6.2 startup-absent
    // contract; nothing observable depends on the tempdir's continued
    // existence once the test future is dropped.
    _token_dir: tempfile::TempDir,
}

impl CheckpointStoreGuard for MockServerGuard {}

// ---------------------------------------------------------------------------
// Stateful fake storage-api server
// ---------------------------------------------------------------------------

/// In-process emulation of the storage-api `by-address` + `data-objects/`
/// routes. Maps **percent-decoded** addresses to their recorded bytes; PUT
/// inserts, GET reads, HEAD stats, DELETE removes, LIST enumerates keys by
/// prefix.
///
/// Wrapped in an `Arc<Mutex<...>>` so the default-route closures share
/// state across the five method-handlers (PUT, GET, HEAD, LIST, DELETE) for
/// one `MockServer`. Per-test isolation is preserved at the `setup` level —
/// every call to `setup()` allocates a fresh `FakeStorageState` instance.
type FakeStorageState = Arc<Mutex<HashMap<String, Bytes>>>;

/// `Respond` impl backing the PUT route. Records the request body in the
/// shared map under the percent-decoded `by-address` tail, then returns 201
/// with an opaque-handle JSON payload.
///
/// The adapter does not inspect the PUT response body (only the status
/// code), so the JSON shape we return here is irrelevant beyond being valid
/// JSON; we still emit a minimal `{"resource_address": "<address>"}` shape
/// for parity with the real storage-api's response and so any future test
/// that reads the body has something sensible to parse.
struct PutHandler {
    state: FakeStorageState,
}

impl Respond for PutHandler {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let Some(address) = decode_by_address_from_path(request.url.path()) else {
            return ResponseTemplate::new(400);
        };
        let body = Bytes::copy_from_slice(&request.body);
        self.state
            .lock()
            .expect("fake storage state mutex must not be poisoned")
            .insert(address.clone(), body);
        ResponseTemplate::new(201).set_body_json(json!({
            "resource_address": address,
        }))
    }
}

/// `Respond` impl backing the GET route. Looks up the percent-decoded
/// `by-address` tail; on hit returns 200 + bytes, on miss returns 404.
struct GetHandler {
    state: FakeStorageState,
}

impl Respond for GetHandler {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let Some(address) = decode_by_address_from_path(request.url.path()) else {
            return ResponseTemplate::new(400);
        };
        let map = self
            .state
            .lock()
            .expect("fake storage state mutex must not be poisoned");
        match map.get(&address) {
            Some(bytes) => ResponseTemplate::new(200).set_body_bytes(bytes.to_vec()),
            None => ResponseTemplate::new(404),
        }
    }
}

/// `Respond` impl backing the HEAD route — storage-api's `stat_by_address`.
/// On hit returns `204` with no body, on miss `404`; never a redirect, since
/// the real HEAD has no redirect branch.
///
/// Without it an unmatched HEAD gets wiremock's default `404`, so every
/// existence probe would read "absent" whatever the map holds:
/// `ext_blob_exists` would answer `Ok(false)` for a stored blob, and a put
/// rollback would delete the companions of an OMPF that is already
/// published.
struct HeadHandler {
    state: FakeStorageState,
}

impl Respond for HeadHandler {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let Some(address) = decode_by_address_from_path(request.url.path()) else {
            return ResponseTemplate::new(400);
        };
        let map = self
            .state
            .lock()
            .expect("fake storage state mutex must not be poisoned");
        if map.contains_key(&address) {
            ResponseTemplate::new(204)
        } else {
            ResponseTemplate::new(404)
        }
    }
}

/// `Respond` impl backing the DELETE route. Removes the entry under the
/// percent-decoded `by-address` tail; on hit returns 204, on miss returns
/// 404 (the adapter swallows that 404 per the idempotent-on-missing
/// contract — `delete_layer_checkpoint`, `delete_session_checkpoint`,
/// `delete_ext_blob` all translate `NotFound` into `Ok(())`).
struct DeleteHandler {
    state: FakeStorageState,
}

impl Respond for DeleteHandler {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let Some(address) = decode_by_address_from_path(request.url.path()) else {
            return ResponseTemplate::new(400);
        };
        let mut map = self
            .state
            .lock()
            .expect("fake storage state mutex must not be poisoned");
        if map.remove(&address).is_some() {
            ResponseTemplate::new(204)
        } else {
            ResponseTemplate::new(404)
        }
    }
}

/// `Respond` impl backing the LIST (Enumerate) route. Percent-decodes the
/// prefix from the URL tail, scans the shared map for matching keys, and
/// returns a `EnumerateResult` JSON page with the matching keys as
/// `AddressInfo` entries. The default fake returns every match in one
/// page (`next_continuation_handle: null`) — pagination is a sibling
/// test's concern, not part of the SR-1.3 conformance surface.
///
/// A prefix that matches no key answers **404 with no body**, as the real
/// storage-api does for an Enumerate of a prefix holding no objects
/// (storage-core `enumerate_nonexistent_prefix`), so the empty-store
/// conformance cases run the adapter's first-page-404 → empty mapping.
///
/// The `continuation_handle` query parameter answers 404 too, as the real
/// service does once a later page holds nothing. It is unreachable: this
/// fake never hands out a continuation handle.
struct ListHandler {
    state: FakeStorageState,
}

impl Respond for ListHandler {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        // The default fake fits everything in one page and never hands out a
        // continuation handle, so a request carrying one is unreachable; it
        // answers what the real service answers for a page that holds
        // nothing.
        if request
            .url
            .query_pairs()
            .any(|(k, _)| k == "continuation_handle")
        {
            return ResponseTemplate::new(404);
        }
        let Some(prefix) = decode_data_objects_prefix_from_path(request.url.path()) else {
            return ResponseTemplate::new(400);
        };
        let map = self
            .state
            .lock()
            .expect("fake storage state mutex must not be poisoned");
        let mut items: Vec<serde_json::Value> = map
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .map(|k| json!({ "resource_address": k }))
            .collect();
        // Stable order so tests that observe LIST output indirectly (e.g.
        // via the adapter's newest-first sort) see a deterministic input.
        items.sort_by(|a, b| {
            a.get("resource_address")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .cmp(
                    b.get("resource_address")
                        .and_then(|v| v.as_str())
                        .unwrap_or(""),
                )
        });
        if items.is_empty() {
            return ResponseTemplate::new(404);
        }
        ResponseTemplate::new(200).set_body_json(json!({
            "next_continuation_handle": null,
            "items": items,
        }))
    }
}

/// Percent-decode the `by-address` tail of a request path. Returns
/// `Some(decoded_address)` for paths shaped like
/// `/v1alpha/fileobject/by-address/<encoded>`; `None` otherwise.
///
/// The adapter constructs every `by-address` URL via
/// `auth::rest::encode_resource_address(address)` over the **entire**
/// composed address (including the leading `/`), so the round-trip here
/// recovers the original decoded address verbatim.
fn decode_by_address_from_path(path: &str) -> Option<String> {
    const PREFIX: &str = "/v1alpha/fileobject/by-address/";
    let encoded = path.strip_prefix(PREFIX)?;
    let decoded = percent_decode_str(encoded).decode_utf8().ok()?;
    Some(decoded.into_owned())
}

/// Percent-decode the `data-objects/` tail of a request path. Returns
/// `Some(decoded_prefix)` for paths shaped like
/// `/v1alpha/fileobject/data-objects/<encoded>`; `None` otherwise.
fn decode_data_objects_prefix_from_path(path: &str) -> Option<String> {
    const PREFIX: &str = "/v1alpha/fileobject/data-objects/";
    let encoded = path.strip_prefix(PREFIX)?;
    let decoded = percent_decode_str(encoded).decode_utf8().ok()?;
    Some(decoded.into_owned())
}

/// Mount the fake's five default routes — PUT, GET, HEAD and DELETE on
/// `by-address/`, and the LIST on `data-objects/` — over `state`, at the
/// default priority.
///
/// Every setup that stands up the plain fake goes through here, so a route
/// added to the fake reaches all of them; a setup that hand-mounted its own
/// copy is how a route goes missing from one fixture and its tests quietly
/// read wiremock's unmatched-request `404` instead.
async fn mount_fake_storage_routes(mock_server: &MockServer, state: &FakeStorageState) {
    Mock::given(method("PUT"))
        .and(path_regex(r"^/v1alpha/fileobject/by-address/"))
        .respond_with(PutHandler {
            state: Arc::clone(state),
        })
        .mount(mock_server)
        .await;

    Mock::given(method("GET"))
        .and(path_regex(r"^/v1alpha/fileobject/by-address/"))
        .respond_with(GetHandler {
            state: Arc::clone(state),
        })
        .mount(mock_server)
        .await;

    Mock::given(method("HEAD"))
        .and(path_regex(r"^/v1alpha/fileobject/by-address/"))
        .respond_with(HeadHandler {
            state: Arc::clone(state),
        })
        .mount(mock_server)
        .await;

    Mock::given(method("DELETE"))
        .and(path_regex(r"^/v1alpha/fileobject/by-address/"))
        .respond_with(DeleteHandler {
            state: Arc::clone(state),
        })
        .mount(mock_server)
        .await;

    Mock::given(method("GET"))
        .and(path_regex(r"^/v1alpha/fileobject/data-objects/"))
        .respond_with(ListHandler {
            state: Arc::clone(state),
        })
        .mount(mock_server)
        .await;
}

/// A `ServiceTokenProvider` over a tempfile-backed, never-written token path,
/// and an adapter pointed at `mock_server` through it. The polling task
/// tolerates the missing file silently per SR-6.2, and the fake never answers
/// 401, so every request goes out without an `Authorization` header.
fn build_fake_backed_adapter(
    mock_server: &MockServer,
    metrics_label: &str,
) -> (StorageApiCheckpointStore, tempfile::TempDir) {
    let token_dir = tempfile::tempdir().expect("tempdir for service-token");
    let token_path = token_dir.path().join("service-token.json");
    let metrics = Arc::new(crate::common::metrics_scrape_only(metrics_label));
    let provider = ServiceTokenProvider::new(
        token_path,
        Duration::from_millis(50),
        metrics,
        CancellationToken::new(),
    );
    let adapter = StorageApiCheckpointStore::new(
        mock_server.uri(),
        PathBuf::from(BASE_PATH),
        Arc::new(provider),
        CONNECT_TIMEOUT,
        REQUEST_TIMEOUT,
    )
    .expect("StorageApiCheckpointStore::new must succeed on realistic inputs");
    (adapter, token_dir)
}

// ---------------------------------------------------------------------------
// Factory closure
// ---------------------------------------------------------------------------

/// Construct a fresh `StorageApiCheckpointStore` pointed at a per-test
/// `MockServer` whose default routes implement the in-process fake storage
/// server documented at the top of this file.
///
/// Each invocation:
///
/// 1. Starts a new `MockServer` via [`MockServer::start`].
/// 2. Allocates a fresh `FakeStorageState` map.
/// 3. Mounts the five default-route programs (PUT, GET, HEAD, LIST,
///    DELETE) against that map, through [`mount_fake_storage_routes`].
/// 4. Builds a `ServiceTokenProvider` over a tempfile-backed,
///    intentionally-never-written token path (the polling task tolerates
///    the missing file silently per SR-6.2; the adapter sends every
///    request without an `Authorization` header in this configuration —
///    which is fine because the fake never returns 401).
/// 5. Constructs the `StorageApiCheckpointStore` over the mock server's
///    URI, the fixed `BASE_PATH`, the token provider, and the realistic
///    timeout pair.
/// 6. Wraps both the `MockServer` and the `TempDir` in a `MockServerGuard`
///    so they live as long as the returned guard does.
///
/// The contract on `factory()` (per `conformance::case_*` docs) is: each
/// invocation MUST yield an independent adapter so the harness never
/// observes cross-case state. We satisfy that by allocating a fresh
/// `MockServer` and `FakeStorageState` on every call.
async fn setup() -> CheckpointStorePair {
    // ─── 1. Fresh MockServer ───────────────────────────────────────────────
    let mock_server = MockServer::start().await;

    // ─── 2. Fresh fake-storage state ───────────────────────────────────────
    let state: FakeStorageState = Arc::new(Mutex::new(HashMap::new()));

    // ─── 3. Mount the five default routes (PUT/GET/HEAD/DELETE on
    //     by-address, LIST on data-objects/). Each handler holds an Arc
    //     clone of the shared state. ──────────────────────────────────────
    mount_fake_storage_routes(&mock_server, &state).await;

    // ─── 4. ServiceTokenProvider. Tempfile-backed, never written — the
    //     polling task tolerates the missing file silently per SR-6.2. ───
    let token_dir = tempfile::tempdir().expect("tempdir for service-token");
    let token_path = token_dir.path().join("service-token.json");
    let metrics = Arc::new(crate::common::metrics_scrape_only(
        "storage-api-conformance-test",
    ));
    let cancel = CancellationToken::new();
    let provider =
        ServiceTokenProvider::new(token_path, Duration::from_millis(50), metrics, cancel);

    // ─── 5. Build the adapter. ────────────────────────────────────────────
    let adapter = StorageApiCheckpointStore::new(
        mock_server.uri(),
        PathBuf::from(BASE_PATH),
        Arc::new(provider),
        CONNECT_TIMEOUT,
        REQUEST_TIMEOUT,
    )
    .expect("StorageApiCheckpointStore::new must succeed on realistic inputs");

    // ─── 6. Pair adapter with guard that owns the MockServer + TempDir. ──
    let store: Arc<dyn CheckpointStore> = Arc::new(adapter);
    let guard: Box<dyn CheckpointStoreGuard> = Box::new(MockServerGuard {
        _server: mock_server,
        _token_dir: token_dir,
    });
    (store, guard)
}

// ---------------------------------------------------------------------------
// Fault-injection translation layer (cases #19, #20, #22)
//
// `StorageApiCheckpointStore` does NOT consult [`CheckpointFaultInjector`] —
// wiremock IS the fault injector for this adapter. `setup_with_injector`
// inspects the supplied injector by calling `maybe_inject(op)` for each op
// label the harness uses in Group G, classifies the returned variant, and
// mounts a high-priority wiremock override (priority `1`, beats the default
// priority `5` routes) that returns the matching HTTP status. The
// translation table below MUST stay line-for-line consistent with
// `StorageApiCheckpointStore::send_classified`'s doc-comment
// (in `src/checkpoint/adapters/storage_api.rs`) and the
// `classify_outcome` body (same file):
//
// - Injector `Transient` → wiremock 503 (the canonical "service unavailable"
//   in the SR-4.4 retriable set `{429, 500, 502, 503, 504}`). The adapter's
//   inner retry loop will attempt MAX_RETRY_ATTEMPTS = 3 times against the
//   override before surfacing `CheckpointError::Transient`; the mock has no
//   call limit, so every retry hits 503 and budget exhaustion classifies
//   exactly as the harness asserts.
// - Injector `Permanent` → wiremock 400 (any non-401 4xx outside `{404, 412}`
//   reaches the `_ => Permanent` arm of `classify_outcome`). No retry loop
//   runs — `Permanent` is terminal on the first attempt.
// - Injector `Conflict` → wiremock 412 (the canonical Conflict status). The
//   adapter's `put_layer_checkpoint` translates 412 to
//   `CheckpointError::Conflict` (the `put_ext_blob` carve-out that swallows
//   412 to `Ok(())` does NOT apply to the layer-checkpoint route case #22
//   targets).
//
// Op-target ambiguity (case #20 only). The harness's
// `OpTargetedFaultInjector` targets the `"get_latest_layer_checkpoint"` op
// label with a `Permanent` error. That op issues a LIST against
// `data-objects/` first, then (on non-empty list) a GET against `by-address`.
// We apply the failure status to the FIRST request the op issues — the LIST
// — by mounting a high-priority 400 override on
// `GET /v1alpha/fileobject/data-objects/...`. The harness asserts only the
// classifier surface (`is_permanent() == true`), not request count, so the
// choice is the only sensible interpretation: a Permanent failure on LIST
// short-circuits the method body before the GET runs, surfacing the
// `Permanent` variant the harness expects.
// ---------------------------------------------------------------------------

/// Construct a fresh `StorageApiCheckpointStore` (per the same six-step
/// pipeline as [`setup`]) and additionally mount high-priority wiremock
/// overrides for whichever op labels the supplied injector flags as
/// failing. The override status code is the
/// [`CheckpointError`] → HTTP translation documented in the comment block
/// directly above.
///
/// Probes the injector by calling `maybe_inject(op)` at setup time for each
/// op label the conformance harness injects on — `put_layer_checkpoint`,
/// `get_latest_layer_checkpoint`, `get_layer_checkpoint`,
/// `get_session_checkpoint`, `list_session_checkpoint_ids` and
/// `ext_blob_exists`. For each `Err(variant)` returned, mounts
/// the matching wiremock override before the adapter is constructed.
///
/// The overrides are route-level and unbounded in call count; the constraint
/// that places on a case body is spelled out above the probe list in the
/// function body.
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn setup_with_injector(
    injector: Arc<dyn CheckpointFaultInjector>,
) -> CheckpointStoreInjectedPair {
    // Steps 1–2: fresh MockServer + fresh fake state. Identical to `setup`.
    let mock_server = MockServer::start().await;
    let state: FakeStorageState = Arc::new(Mutex::new(HashMap::new()));

    // Step 3a: inspect the injector for each known op label, translate any
    //          returned variant to its HTTP status, and mount a
    //          high-priority override (priority 1, default routes are 5)
    //          that targets the FIRST request the op issues.
    //
    // A label present in `first_request_route_for_op` but absent from THIS
    // list is never probed, so no override is ever mounted and any case
    // injecting on it passes vacuously. Both sites must carry the label.
    //
    // ── Constraint on case bodies: these overrides are route-level, not
    //    call-ordered ────────────────────────────────────────────────────
    //
    // wiremock matches on (method, path-regex) with no call counter, so an
    // override is in force for the whole test, on every request that matches
    // the route — not just the one call the case means to fail. Concretely:
    //
    // - An override mounted for `put_layer_checkpoint` fails EVERY PUT under
    //   `by-address` — base ext-refs / meta / OMPF puts, tail meta and tail
    //   artifact puts, and ext-blob puts alike. The op labels name the
    //   *method*, never the artifact kind, and the route is shared by all of
    //   them.
    // - An override mounted for `get_layer_checkpoint` (or
    //   `get_session_checkpoint`) fails EVERY by-address GET, whichever
    //   method issued it.
    // - An override mounted for `list_session_checkpoint_ids` fails EVERY
    //   `data-objects/` LIST, `layers/` walks included; one mounted for
    //   `ext_blob_exists` fails EVERY by-address HEAD, the put-rollback
    //   existence probes included.
    //
    // Therefore a conformance case that injects on a label MUST NOT need a
    // successful call on that same HTTP route earlier in its body — e.g. it
    // cannot seed a generation with a real put and then inject on
    // `put_layer_checkpoint`, because the seeding put fails too. Arrange the
    // fixture through a different route, or through a separately-built
    // uninjected store.
    for op in [
        "put_layer_checkpoint",
        "get_latest_layer_checkpoint",
        "get_layer_checkpoint",
        "get_session_checkpoint",
        "list_session_checkpoint_ids",
        "ext_blob_exists",
    ] {
        let Err(err) = injector.maybe_inject(op) else {
            continue;
        };
        let (http_method, path_pattern) = first_request_route_for_op(op);
        let status = checkpoint_error_to_http_status(&err);
        Mock::given(method(http_method))
            .and(path_regex(path_pattern))
            .respond_with(ResponseTemplate::new(status))
            .with_priority(1)
            .mount(&mock_server)
            .await;
    }

    // Step 3b: mount the default fake-storage routes (priority 5). These
    //          serve every request that doesn't match a higher-priority
    //          override above, so unrelated ops in the test body still run
    //          against the in-process fake.
    mount_fake_storage_routes(&mock_server, &state).await;

    // Step 4: ServiceTokenProvider over a never-written tempfile.
    let token_dir = tempfile::tempdir().expect("tempdir for service-token");
    let token_path = token_dir.path().join("service-token.json");
    let metrics = Arc::new(crate::common::metrics_scrape_only(
        "storage-api-conformance-injected-test",
    ));
    let cancel = CancellationToken::new();
    let provider =
        ServiceTokenProvider::new(token_path, Duration::from_millis(50), metrics, cancel);

    // Step 5: build the adapter.
    let adapter = StorageApiCheckpointStore::new(
        mock_server.uri(),
        PathBuf::from(BASE_PATH),
        Arc::new(provider),
        CONNECT_TIMEOUT,
        REQUEST_TIMEOUT,
    )
    .expect("StorageApiCheckpointStore::new must succeed on realistic inputs");

    // Step 6: pair with guard that owns the MockServer + TempDir.
    let store: Arc<dyn CheckpointStore> = Arc::new(adapter);
    let guard: Box<dyn CheckpointStoreGuard> = Box::new(MockServerGuard {
        _server: mock_server,
        _token_dir: token_dir,
    });
    (store, guard)
}

/// Translate a `CheckpointError` variant into the wiremock HTTP status code
/// that `classify_outcome` (in `src/checkpoint/adapters/storage_api.rs`) maps back to the
/// same variant. The status choices match the canonical mapping table
/// documented above [`setup_with_injector`].
#[cfg(feature = "test-checkpoint-fault-injection")]
fn checkpoint_error_to_http_status(err: &CheckpointError) -> u16 {
    match err {
        CheckpointError::Transient { .. } => 503,
        CheckpointError::Permanent { .. } => 400,
        CheckpointError::Conflict => 412,
        // The harness only constructs Transient / Permanent / Conflict in
        // cases #19, #20, #22. NotFound / NotImplemented would indicate a
        // future case shape this translation layer does not yet cover —
        // panic loudly rather than silently misclassify.
        other => panic!(
            "setup_with_injector: unsupported injector variant {other:?} — \
             extend the translation table when a new Group G case is added",
        ),
    }
}

/// Resolve the (HTTP method, path-regex) pair for the FIRST outbound request
/// each known op label issues. Used by [`setup_with_injector`] to mount the
/// failure override on the right route.
///
/// - `"put_layer_checkpoint"` issues PUT(s) under `by-address`, but *which*
///   artifact goes first depends on the branch (read
///   `StorageApiCheckpointStore::put_layer_checkpoint`, not the ordering this
///   bullet used to claim):
///   - **Base branch** publishes ext-refs → meta → OMPF. The first PUT is the
///     `.ext-refs.json` sidecar when the caller paired one with the meta
///     (`meta.sidecar_bytes == Some`), and the `.meta.json` when it did not
///     (the OMPF-only shape). The OMPF is **last**, never first.
///   - **Tail branch** publishes meta → artifact. The first PUT is the
///     `.tail.meta.json`, always — there is no ext-refs artifact on a tail
///     under any circumstance (`validate_tail_put` rejects one).
///
///   All four of those addresses render under the same
///   `/v1alpha/fileobject/by-address/` prefix, so the single prefix regex
///   returned here matches whichever PUT actually fires first on either
///   branch. That is why one kind-blind `put_layer_checkpoint` label still
///   suffices and no kind-qualified put label is needed: the override lands on
///   the route, and the route is shared. Whatever the first PUT is, it either
///   succeeds (advancing to the next artifact) or surfaces the override status
///   and terminates the method.
/// - `"get_latest_layer_checkpoint"` issues LIST against `data-objects/`
///   first, then GET against `by-address` (read
///   `StorageApiCheckpointStore::get_latest_layer_checkpoint`). Override
///   targets the LIST so the failure short-circuits the method body before
///   any GET runs — matches the op-target-ambiguity choice documented above.
/// - `"get_layer_checkpoint"` is the **by-address GET, not the LIST**. Get-by-
///   handle addresses `(object_id, snap_seq, kind)` directly — it hands those
///   three straight to `read_generation_at`, which derives the artifact and
///   meta URLs from the kind and GETs them — so it never issues a
///   `data-objects/` enumeration at all. Overriding the LIST route (the
///   `get_latest_layer_checkpoint` choice above) would therefore leave every
///   request this op actually makes unaffected, and the case would assert its
///   error classification against a request that never failed.
/// - `"get_session_checkpoint"` (Epic 2 placeholder — the adapter currently
///   returns `NotImplemented` before any HTTP fires) is included for
///   forward-compatibility. The route returned here is the eventual session
///   GET shape; until Epic 2 wires it in, an injector targeting this op
///   simply has no observable effect because the method never reaches the
///   HTTP layer.
/// - `"list_session_checkpoint_ids"` issues only the `sessions/` LIST against
///   `data-objects/`, so the override targets that route.
/// - `"ext_blob_exists"` issues only a HEAD on the ext blob's `by-address`
///   URL — never a GET — so the override is a HEAD override. A GET override
///   would leave the probe's only request untouched and the case would read
///   the fake's honest answer instead of the injected failure.
#[cfg(feature = "test-checkpoint-fault-injection")]
fn first_request_route_for_op(op: &str) -> (&'static str, &'static str) {
    match op {
        "put_layer_checkpoint" => ("PUT", r"^/v1alpha/fileobject/by-address/"),
        "get_latest_layer_checkpoint" => ("GET", r"^/v1alpha/fileobject/data-objects/"),
        "get_layer_checkpoint" => ("GET", r"^/v1alpha/fileobject/by-address/"),
        "get_session_checkpoint" => ("GET", r"^/v1alpha/fileobject/by-address/"),
        "list_session_checkpoint_ids" => ("GET", r"^/v1alpha/fileobject/data-objects/"),
        "ext_blob_exists" => ("HEAD", r"^/v1alpha/fileobject/by-address/"),
        other => panic!(
            "setup_with_injector: unknown op label {other:?} — extend \
             first_request_route_for_op when a new Group G case adds a label",
        ),
    }
}

// ---------------------------------------------------------------------------
// Group A: round-trip puts and gets (#1–#4)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_01_put_layer_then_get_latest_round_trips_bytes() {
    conformance::case_01_put_layer_then_get_latest_round_trips_bytes(setup).await;
}

#[tokio::test]
async fn case_02_put_session_then_get_session_round_trips_bytes() {
    conformance::case_02_put_session_then_get_session_round_trips_bytes(setup).await;
}

#[tokio::test]
async fn case_03_put_ext_blob_then_get_ext_blob_round_trips_bytes() {
    conformance::case_03_put_ext_blob_then_get_ext_blob_round_trips_bytes(setup).await;
}

#[tokio::test]
async fn case_04_put_counters_sidecar_then_get_round_trips_bytes() {
    conformance::case_04_put_counters_sidecar_then_get_round_trips_bytes(setup).await;
}

// ---------------------------------------------------------------------------
// Group B: absent-key behaviour (#5–#8)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_05_get_latest_layer_checkpoint_unknown_object_returns_none() {
    conformance::case_05_get_latest_layer_checkpoint_unknown_object_returns_none(setup).await;
}

#[tokio::test]
async fn case_06_get_session_checkpoint_unknown_session_returns_none() {
    conformance::case_06_get_session_checkpoint_unknown_session_returns_none(setup).await;
}

#[tokio::test]
async fn case_07_get_ext_blob_unknown_hash_returns_none() {
    conformance::case_07_get_ext_blob_unknown_hash_returns_none(setup).await;
}

#[tokio::test]
async fn case_08_get_counters_sidecar_fresh_adapter_returns_none() {
    conformance::case_08_get_counters_sidecar_fresh_adapter_returns_none(setup).await;
}

// ---------------------------------------------------------------------------
// Group C: idempotent puts (#9–#10)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_09_put_layer_idempotent_single_generation() {
    conformance::case_09_put_layer_idempotent_single_generation(setup).await;
}

#[tokio::test]
async fn case_10_put_ext_blob_idempotent_no_duplicate_storage() {
    conformance::case_10_put_ext_blob_idempotent_no_duplicate_storage(setup).await;
}

// ---------------------------------------------------------------------------
// Group D: list ordering (#11–#12)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_11_list_layer_checkpoints_newest_first_ordering() {
    conformance::case_11_list_layer_checkpoints_newest_first_ordering(setup).await;
}

#[tokio::test]
async fn case_12_list_layer_checkpoints_unknown_object_returns_empty() {
    conformance::case_12_list_layer_checkpoints_unknown_object_returns_empty(setup).await;
}

// ---------------------------------------------------------------------------
// Group E: delete semantics (#13–#16)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_13_delete_layer_then_get_returns_next_newest_or_none() {
    conformance::case_13_delete_layer_then_get_returns_next_newest_or_none(setup).await;
}

#[tokio::test]
async fn case_14_delete_layer_nonexistent_handle_is_idempotent() {
    conformance::case_14_delete_layer_nonexistent_handle_is_idempotent(setup).await;
}

#[tokio::test]
async fn case_15_delete_session_then_get_returns_none() {
    conformance::case_15_delete_session_then_get_returns_none(setup).await;
}

#[tokio::test]
async fn case_16_delete_ext_blob_then_get_returns_none() {
    conformance::case_16_delete_ext_blob_then_get_returns_none(setup).await;
}

// ---------------------------------------------------------------------------
// Group F: concurrency safety (#17–#18)
//
// Multi-thread runtime so the spawned writer / reader / delete tasks make
// real progress in parallel and the race window the case bodies depend on
// actually opens up. Mirrors the analogous gating in
// `adapter_local_disk_test.rs` and `adapter_in_memory_test.rs`.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn case_17_concurrent_put_delete_unrelated_keys_no_interference() {
    conformance::case_17_concurrent_put_delete_unrelated_keys_no_interference(setup).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn case_18_concurrent_put_get_same_key_no_torn_read() {
    conformance::case_18_concurrent_put_get_same_key_no_torn_read(setup).await;
}

// ---------------------------------------------------------------------------
// Group G: error classification (#19-#22)
//
// Cases #19, #20, #22 drive the fault-injection seam via
// `setup_with_injector` and are cfg-gated on `test-checkpoint-fault-injection`
// to match the harness-side gating in `conformance.rs`. Case #21 is the
// natural-`NotFound`-never-surfaced contract — the default mocks already
// return 404 for absent addresses, so #21 works against the regular `setup`
// closure with zero additional wiring and is therefore NOT cfg-gated.
// ---------------------------------------------------------------------------

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test]
async fn case_19_transient_error_classification() {
    conformance::case_19_transient_error_classification(setup_with_injector).await;
}

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test]
async fn case_20_permanent_error_classification() {
    conformance::case_20_permanent_error_classification(setup_with_injector).await;
}

// Case #21 uses the regular `setup` closure: the default mocks return 404
// for absent addresses, which the adapter translates to the trait-surface
// `Ok(None)` / `Ok(())` per `send_classified`'s NotFound contract — that
// is the exact behaviour case #21 asserts across every `get_*` / `list_*`
// / `delete_*` shape. No injector wiring required, so this case is NOT
// cfg-gated on `test-checkpoint-fault-injection`.
#[tokio::test]
async fn case_21_not_found_classification() {
    conformance::case_21_not_found_classification(setup).await;
}

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test]
async fn case_22_conflict_error_classification() {
    conformance::case_22_conflict_error_classification(setup_with_injector).await;
}

// ---------------------------------------------------------------------------
// Group H: cross-cutting enumeration + size/retention (#23-#25)
//
// All three cases delegate to the regular `setup` closure — no fault injection
// involvement. The default-route fake (PUT/GET/LIST/DELETE on the by-address
// + data-objects/ routes) already covers case #23's ext-refs sidecar PUT,
// case #24's 1 MiB layer-checkpoint round-trip, and case #25's put + delete
// retention sequence; no per-case mock programming or wiremock body-cap bump
// is required (see case-body comments in `conformance.rs` for the sizing
// rationale — 1 MiB is well below any reasonable HTTP/mock body cap).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_23_list_referenced_ext_hashes_union_across_layers() {
    conformance::case_23_list_referenced_ext_hashes_union_across_layers(setup).await;
}

#[tokio::test]
async fn case_24_size_bound_sanity_accepts_large_blob() {
    conformance::case_24_size_bound_sanity_accepts_large_blob(setup).await;
}

#[tokio::test]
async fn case_25_retention_bound_sanity_after_prune() {
    conformance::case_25_retention_bound_sanity_after_prune(setup).await;
}

// Case #26 — per-handle `read_ext_refs_sidecar`. Wired here alongside cases
// #23 and #25 to round out the cross-cutting enumeration coverage. Requires
// the per-adapter override added in the cross-Epic follow-up to
// `t3.fgj.zc.56.uq`; without that override the trait default returns
// `Ok(HashSet::new())` for every handle and the case fails immediately on
// the happy-path assertion.
#[tokio::test]
async fn case_26_read_ext_refs_sidecar_per_handle() {
    conformance::case_26_read_ext_refs_sidecar_per_handle(setup).await;
}

// ---------------------------------------------------------------------------
// Group I — raw artifact enumeration (#27)
//
// This adapter can hold an orphan AND a temp, so it supplies both planters —
// and neither residual is reachable through the trait, so both are planted
// straight into the fake's map.
//
// **The orphan** the adapter deliberately cannot produce any more: since b.s5y
// a failed publish rolls its companions back. The residual this case is about
// is what survives a crash *between* two PUTs, a state no sequence of trait
// calls can reach.
//
// **The temp** the adapter never writes at all — its PUTs land by address, with
// no temp-then-rename step — but it parses the whole `layers/` subtree, so it
// enumerates one a *foreign* tool left behind, and it carries a real
// LIST-then-DELETE `delete_layer_temp_artifacts` override to reclaim it. That
// combination is the reason to plant one here rather than declare
// `plant_temp: None`: an adapter that enumerates `Temp` while inheriting the
// trait's no-op default hands the reconciliation pass a candidate it can never
// retire, and the harness's no-temp branch would have accepted exactly that
// (it only asserts `Ok(0)`, which the no-op default also returns). Planting the
// temp routes the case through the harness's reclaim branch instead, where the
// override has to actually enumerate it, delete it, and leave the live
// generation at the same stem alone.
// ---------------------------------------------------------------------------

/// Stand up the same fake as [`setup`], but keep a handle on the shared state
/// map so the planting closure can insert an artifact the adapter would never
/// write on its own.
async fn setup_artifacts() -> conformance::ArtifactFixture {
    let mock_server = MockServer::start().await;
    let state: FakeStorageState = Arc::new(Mutex::new(HashMap::new()));
    mount_fake_storage_routes(&mock_server, &state).await;
    let (adapter, token_dir) =
        build_fake_backed_adapter(&mock_server, "storage-api-conformance-test");

    let planter_state = Arc::clone(&state);
    let temp_planter_state = Arc::clone(&state);
    conformance::ArtifactFixture {
        store: Arc::new(adapter),
        guard: Box::new(MockServerGuard {
            _server: mock_server,
            _token_dir: token_dir,
        }),
        plant_orphan: Some(Box::new(move |object_id, snap_seq| {
            let state = Arc::clone(&planter_state);
            Box::pin(async move {
                let mut map = state.lock().expect("fake storage state mutex");
                map.insert(
                    format!("{BASE_PATH}/.live-edit/layers/{object_id}.{snap_seq}.ext-refs.json"),
                    conformance::ext_refs_sidecar_for(&[conformance::CASE_27_ORPHAN_HASH]),
                );
                map.insert(
                    format!("{BASE_PATH}/.live-edit/layers/{object_id}.{snap_seq}.meta.json"),
                    Bytes::from_static(b"{}"),
                );
            })
        })),
        plant_temp: Some(Box::new(move |object_id, snap_seq| {
            let state = Arc::clone(&temp_planter_state);
            Box::pin(async move {
                let mut map = state.lock().expect("fake storage state mutex");
                map.insert(
                    format!(
                        "{BASE_PATH}/.live-edit/layers/\
                         {object_id}.{snap_seq}.ompf.{CASE_27_TEMP_UUID}.tmp"
                    ),
                    Bytes::from_static(b"half-written"),
                );
            })
        })),
    }
}

#[tokio::test]
async fn case_27_list_layer_artifacts_enumerates_orphans_and_temps() {
    conformance::case_27_list_layer_artifacts_enumerates_orphans_and_temps(setup_artifacts).await;
}

// ---------------------------------------------------------------------------
// Group J — tail generations (#28–#37)
//
// Tail generations ride the regular `setup` closure. The default-route fake is
// address-shaped rather than artifact-shaped — PUT records whatever address it
// is handed, GET reads it back, LIST returns every key under the prefix — so a
// tail's `<oid>.<to_seq>.tail` and `.tail.meta.json` addresses need no new
// route programming; the kind-aware parsing that turns those addresses back
// into generations is the adapter's own, which is exactly what these cases
// exercise. Deliberately NOT skip-gated: this adapter stores tails as of Epic 2
// Task 4, so a `NotImplemented` out of a tail put would be an adapter defect,
// and gating would turn it into a silent pass.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_28_tail_put_then_get_by_handle_and_list_round_trips() {
    conformance::case_28_tail_put_then_get_by_handle_and_list_round_trips(setup).await;
}

#[tokio::test]
async fn case_29_get_layer_checkpoint_by_handle_contract_both_kinds() {
    conformance::case_29_get_layer_checkpoint_by_handle_contract_both_kinds(setup).await;
}

#[tokio::test]
async fn case_30_list_layer_checkpoints_mixed_kind_newest_first_base_before_tail() {
    conformance::case_30_list_layer_checkpoints_mixed_kind_newest_first_base_before_tail(setup)
        .await;
}

#[tokio::test]
async fn case_31_delete_tail_is_idempotent_and_kind_isolated() {
    conformance::case_31_delete_tail_is_idempotent_and_kind_isolated(setup).await;
}

#[tokio::test]
async fn case_32_tail_put_validation_rejections_are_permanent_and_write_nothing() {
    conformance::case_32_tail_put_validation_rejections_are_permanent_and_write_nothing(setup)
        .await;
}

#[tokio::test]
async fn case_33_tails_are_invisible_to_the_ext_refs_surface() {
    conformance::case_33_tails_are_invisible_to_the_ext_refs_surface(setup).await;
}

#[tokio::test]
async fn case_34_list_all_object_ids_includes_tail_only_objects() {
    conformance::case_34_list_all_object_ids_includes_tail_only_objects(setup).await;
}

#[tokio::test]
async fn case_37_list_layer_artifacts_reports_both_generation_kinds() {
    conformance::case_37_list_layer_artifacts_reports_both_generation_kinds(setup).await;
}

// ---------------------------------------------------------------------------
// Group K — lineage-aware retention (#38)
//
// The Epic-5 retention surface: a whole lineage's delete run, issued
// tails-newest-first then base-last over handles captured from an earlier
// listing, must be idempotent at every position and must not perturb the
// sibling lineage under the same `object_id`. Rides the regular `setup` fake
// with NO per-case route programming and no new fault seam: every delete in the
// run is an ordinary `DELETE by-address/`, and the fake's 404-on-missing answer
// is what the adapter's idempotent-on-missing contract is built on — which is
// precisely the behaviour under test, not a shortcut around it.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_38_lineage_prune_is_idempotent_and_lineage_isolated() {
    conformance::case_38_lineage_prune_is_idempotent_and_lineage_isolated(setup).await;
}

// ---------------------------------------------------------------------------
// Case #39 — `validate_generation_identity` (b.hka (b)). Rides the regular
// `setup` closure. The case body asserts the classification and the read
// surfaces; the adapter-specific ZERO-HTTP-REQUEST half — which is the one that
// distinguishes "rejected before any URL was derived" from "rejected after a
// partial publish" — is pinned in-crate against a `wiremock` request log, in
// `storage_api::tests::layer_checkpoint_method_tests::put_layer_checkpoint_with_an_empty_*`,
// beside the identical pin for `validate_tail_put`.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_39_identity_empty_puts_are_permanent_and_write_nothing() {
    conformance::case_39_identity_empty_puts_are_permanent_and_write_nothing(setup).await;
}

// Cases #35 / #36 extend Group G's classification surface to the tail methods
// and so ride `setup_with_injector`, cfg-gated on the same feature as #19 /
// #20 / #22. Both rely on the `"get_layer_checkpoint"` entries already carried
// by the probe list and `first_request_route_for_op`; #36 reuses the existing
// `"put_layer_checkpoint"` label, whose PUT-under-`by-address` override covers
// the tail meta / envelope puts as well as the base triplet.

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test]
async fn case_35_get_layer_checkpoint_error_classification() {
    conformance::case_35_get_layer_checkpoint_error_classification(setup_with_injector).await;
}

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test]
async fn case_36_tail_put_error_classification() {
    conformance::case_36_tail_put_error_classification(setup_with_injector).await;
}

// ---------------------------------------------------------------------------
// Group M — the DR restore Job's read surface (#40–#42)
//
// #40 plants straight into the fake's map under `sessions/`, the subtree the
// listing enumerates: a foreign tool's `<sid>.json.<uuid>.tmp` and a stray
// file, neither of which this adapter writes (its session PUT lands by
// address, with no temp step). #41 needs the HEAD route `setup` mounts — with
// no HEAD route, wiremock's unmatched-request 404 makes every blob read
// "absent" and the case's stored-blob half fails. #42 rides
// `setup_with_injector`, whose probe list and `first_request_route_for_op`
// carry both labels: the LIST override for `list_session_checkpoint_ids` and
// the HEAD override for `ext_blob_exists`.
//
// The fake's LIST answers an empty prefix with the 404 storage-api answers, so
// #40's empty-store branches run the adapter's first-page-404 → empty mapping.
// The later-page 404 is pinned in-crate, in
// `storage_api::tests::dr_restore_read_method_tests`.
// ---------------------------------------------------------------------------

/// Stand up the same fake as [`setup`], keeping a handle on the shared state
/// map so case #40 can plant files under `sessions/`.
async fn setup_session_listing() -> conformance::SessionListingFixture {
    let mock_server = MockServer::start().await;
    let state: FakeStorageState = Arc::new(Mutex::new(HashMap::new()));
    mount_fake_storage_routes(&mock_server, &state).await;
    let (adapter, token_dir) =
        build_fake_backed_adapter(&mock_server, "storage-api-conformance-test");

    let planter_state = Arc::clone(&state);
    conformance::SessionListingFixture {
        store: Arc::new(adapter),
        guard: Box::new(MockServerGuard {
            _server: mock_server,
            _token_dir: token_dir,
        }),
        plant_in_sessions: Some(Box::new(move |filename| {
            let state = Arc::clone(&planter_state);
            Box::pin(async move {
                state.lock().expect("fake storage state mutex").insert(
                    format!("{BASE_PATH}/.live-edit/sessions/{filename}"),
                    Bytes::from_static(b"not a session backup"),
                );
            })
        })),
    }
}

#[tokio::test]
async fn case_40_list_session_checkpoint_ids_returns_exactly_the_stored_session_ids() {
    conformance::case_40_list_session_checkpoint_ids_returns_exactly_the_stored_session_ids(
        setup_session_listing,
    )
    .await;
}

#[tokio::test]
async fn case_41_ext_blob_exists_is_true_only_for_stored_bytes() {
    conformance::case_41_ext_blob_exists_is_true_only_for_stored_bytes(setup).await;
}

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test]
async fn case_42_dr_restore_read_methods_surface_backend_failures_as_errors() {
    conformance::case_42_dr_restore_read_methods_surface_backend_failures_as_errors(
        setup_with_injector,
    )
    .await;
}

// ---------------------------------------------------------------------------
// Group N — the live-set walk's own evidence (#43)
//
// Rides the regular `setup` fake with no per-case route programming. Its LIST
// answers an empty prefix with the 404 storage-api answers, so the case's
// empty-store half runs the adapter's first-page-404 → `(∅, 0)` mapping; once
// the two bases are put, the same LIST names their artifacts, and the count
// comes from that one listing.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_43_list_live_ext_set_counts_its_own_listing_and_projects_to_the_live_set() {
    conformance::case_43_list_live_ext_set_counts_its_own_listing_and_projects_to_the_live_set(
        setup,
    )
    .await;
}

// ---------------------------------------------------------------------------
// Group L — the b.z22 client-side redirect protocol, end to end
// ---------------------------------------------------------------------------

mod redirect_round_trip_tests {
    //! A large generation published **and read back** through the HTTP 300
    //! protocol, against a fake that behaves the way the dev deployment does.
    //!
    //! ## Why this exists alongside the in-crate redirect tests
    //!
    //! Those tests are per-branch: each mounts a hand-written 300 body and
    //! asserts one property of the exchange it produces. None of them can see
    //! the property the ticket was actually filed about — that a checkpoint
    //! **survives** the round trip. The bytes leave through a presigned upload
    //! and come back through a presigned download, and nothing in a per-branch
    //! test pins that the two halves address the same object, that the payload
    //! is not truncated by the empty-bodied by-address PUT, or that the meta
    //! (which is small, and therefore travels inline) still pairs with an OMPF
    //! that did not.
    //!
    //! ## The fake
    //!
    //! One shared `HashMap` stands in for the storage account, and BOTH servers
    //! read and write it — which is the real-world property that makes the
    //! protocol work at all: the presigned target and the by-address API are two
    //! doors onto one object. It keeps the [`setup`] fake's HEAD, DELETE and
    //! LIST routes, and has in place of, or beside, its PUT and GET:
    //!
    //! - a **size-routing PUT**: a write whose declared `data_object_size` is at
    //!   or above the threshold, or which states `upload_preference=redirect`,
    //!   is answered with a write redirect instead of being stored. Routing on
    //!   the *declared* size rather than on the body length is what the real
    //!   server does, and is required here: the pre-flight path's by-address PUT
    //!   carries no body at all.
    //! - a **size-routing GET** that answers a large artifact with a read
    //!   redirect **despite** the declared `download_preference=body`. That is
    //!   the contract-drift path, deliberately: it is what dev did, and the read
    //!   fallback exists because it must degrade to a slower success.
    //! - the presigned target's own upload / download routes, and the
    //!   `…/redirect/complete` handshake.
    //!
    //! ## The two configurations
    //!
    //! Both tests publish and read back the same artifact; they differ only in
    //! whether the upload-options pre-flight answers. With it, the by-address
    //! PUT is empty-bodied and the payload only ever travels once; without it
    //! (an older storage-api, or one that does not grant `FetchWriteTypeInfo`)
    //! the adapter streams the payload and handles the 300 reactively. The
    //! *observable outcome* — a byte-identical read-back — must be identical,
    //! and that equivalence is the whole safety case for the pre-flight being an
    //! optimisation rather than a behaviour change.

    use percent_encoding::{utf8_percent_encode, NON_ALPHANUMERIC};

    use super::*;

    /// Size at or above which the fake routes a write to a presigned upload and
    /// a read to a presigned download. Mirrors dev's 256 KiB `readRedirectSize`
    /// (dev's 2 MiB `writeBodyMaxSize` is scaled down to the same number so one
    /// artifact exercises both directions).
    const REDIRECT_THRESHOLD_BYTES: usize = 256 * 1024;

    /// The generation payload. Above the threshold in both directions, and
    /// above the adapter's own `UPLOAD_OPTIONS_MIN_PAYLOAD_BYTES` floor so the
    /// pre-flight is consulted at all.
    const LARGE_ARTIFACT_BYTES: usize = 400 * 1024;

    /// Stands in for the SAS an Azure / S3 presigned URL carries.
    const SAS_MARKER: &str = "sig=integration-presigned-marker";

    const OBJECT_ID: u64 = 42;
    const SNAP_SEQ: u64 = 7;

    /// Deterministic, position-dependent payload: a truncated or re-ordered
    /// upload cannot compare equal to it.
    fn large_payload() -> Bytes {
        Bytes::from(
            (0..LARGE_ARTIFACT_BYTES)
                .map(|i| (i % 251) as u8)
                .collect::<Vec<u8>>(),
        )
    }

    /// The presigned path this fake mints for an artifact address. Encoding the
    /// address into the path is what lets the target's own handlers land the
    /// bytes in the shared map under the address the by-address API will read
    /// them back by.
    fn blob_path_for(address: &str) -> String {
        format!("/blob/{}", utf8_percent_encode(address, NON_ALPHANUMERIC))
    }

    /// Inverse of [`blob_path_for`].
    fn address_from_blob_path(path: &str) -> Option<String> {
        let encoded = path.strip_prefix("/blob/")?;
        Some(percent_decode_str(encoded).decode_utf8().ok()?.into_owned())
    }

    /// Value of query parameter `key` on a recorded request.
    fn query_value(request: &Request, key: &str) -> Option<String> {
        request
            .url
            .query_pairs()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.into_owned())
    }

    /// PUT `by-address` with the real server's size routing: at or above the
    /// threshold (or on an explicit redirect preference) it answers a write
    /// redirect; below it, it stores the body.
    struct SizeRoutingPutHandler {
        state: FakeStorageState,
        target_uri: String,
    }

    impl Respond for SizeRoutingPutHandler {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let Some(address) = decode_by_address_from_path(request.url.path()) else {
                return ResponseTemplate::new(400);
            };
            let declared = query_value(request, "data_object_size")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(request.body.len());
            let prefers_redirect =
                query_value(request, "upload_preference").as_deref() == Some("redirect");

            if prefers_redirect || declared >= REDIRECT_THRESHOLD_BYTES {
                return ResponseTemplate::new(300).set_body_json(json!({
                    "redirect": {
                        "redirect_target_url": format!(
                            "{}{}?{SAS_MARKER}",
                            self.target_uri,
                            blob_path_for(&address)
                        ),
                        "method": "put",
                        "additional_headers": [
                            { "name": "x-ms-blob-type", "value": "BlockBlob" }
                        ],
                        "completion_header_names": ["etag"],
                    }
                }));
            }

            self.state
                .lock()
                .expect("fake storage state mutex must not be poisoned")
                .insert(address.clone(), Bytes::copy_from_slice(&request.body));
            ResponseTemplate::new(201).set_body_json(json!({ "resource_address": address }))
        }
    }

    /// GET `by-address` with the same size routing — and deliberately
    /// **without** honouring `download_preference=body`, which is exactly the
    /// contract drift the read fallback exists for.
    struct SizeRoutingGetHandler {
        state: FakeStorageState,
        target_uri: String,
    }

    impl Respond for SizeRoutingGetHandler {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let Some(address) = decode_by_address_from_path(request.url.path()) else {
                return ResponseTemplate::new(400);
            };
            let map = self
                .state
                .lock()
                .expect("fake storage state mutex must not be poisoned");
            match map.get(&address) {
                None => ResponseTemplate::new(404),
                Some(bytes) if bytes.len() >= REDIRECT_THRESHOLD_BYTES => {
                    ResponseTemplate::new(300).set_body_json(json!({
                        "redirect_target_url": format!(
                            "{}{}?{SAS_MARKER}",
                            self.target_uri,
                            blob_path_for(&address)
                        ),
                        "additional_headers": { "x-ms-version": "2021-08-06" },
                    }))
                }
                Some(bytes) => ResponseTemplate::new(200).set_body_bytes(bytes.to_vec()),
            }
        }
    }

    /// The presigned target's upload route: lands the bytes in the SAME map the
    /// by-address API reads, under the address the redirect encoded.
    struct TargetUploadHandler {
        state: FakeStorageState,
    }

    impl Respond for TargetUploadHandler {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let Some(address) = address_from_blob_path(request.url.path()) else {
                return ResponseTemplate::new(400);
            };
            self.state
                .lock()
                .expect("fake storage state mutex must not be poisoned")
                .insert(address, Bytes::copy_from_slice(&request.body));
            ResponseTemplate::new(201).insert_header("etag", "\"etag-from-the-fake-blob-store\"")
        }
    }

    /// The presigned target's download route.
    struct TargetDownloadHandler {
        state: FakeStorageState,
    }

    impl Respond for TargetDownloadHandler {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let Some(address) = address_from_blob_path(request.url.path()) else {
                return ResponseTemplate::new(400);
            };
            let map = self
                .state
                .lock()
                .expect("fake storage state mutex must not be poisoned");
            match map.get(&address) {
                Some(bytes) => ResponseTemplate::new(200).set_body_bytes(bytes.to_vec()),
                None => ResponseTemplate::new(404),
            }
        }
    }

    /// Whether the fake's upload-options route answers.
    enum PreFlight {
        /// Publishes intervals routing anything at or above the threshold to a
        /// redirect — the shape a current storage-api answers.
        Answers,
        /// 404 — the route does not exist on this server, so the adapter falls
        /// back to the reactive write it shipped with.
        Absent,
    }

    /// Everything a round-trip test needs, alive for as long as the test body.
    struct RedirectFake {
        store: StorageApiCheckpointStore,
        storage: MockServer,
        target: MockServer,
        _token_dir: tempfile::TempDir,
    }

    /// Stand up the two servers, the shared object map, and an adapter pointed
    /// at the by-address server.
    async fn setup_redirect_fake(pre_flight: PreFlight) -> RedirectFake {
        let storage = MockServer::start().await;
        let target = MockServer::start().await;
        let state: FakeStorageState = Arc::new(Mutex::new(HashMap::new()));

        let upload_options = match pre_flight {
            PreFlight::Answers => ResponseTemplate::new(200).set_body_json(json!({
                "write_type_intervals": [
                    {
                        "minimum_data_object_size": 0,
                        "maximum_data_object_size": REDIRECT_THRESHOLD_BYTES,
                        "preferred_upload_method": "body",
                    },
                    {
                        "minimum_data_object_size": REDIRECT_THRESHOLD_BYTES,
                        "maximum_data_object_size": 1_073_741_824_u64,
                        "preferred_upload_method": "redirect",
                    },
                ]
            })),
            PreFlight::Absent => ResponseTemplate::new(404),
        };
        // Anchored on its own route prefix, and mounted before the by-address
        // GET: the two routes share their percent-encoded address tail, so an
        // unanchored matcher would let the artifact route answer the probe.
        Mock::given(method("GET"))
            .and(path_regex(
                r"^/v1alpha/fileobject/upload-options/by-address/",
            ))
            .respond_with(upload_options)
            .mount(&storage)
            .await;

        Mock::given(method("PUT"))
            .and(path_regex(r"^/v1alpha/fileobject/by-address/"))
            .respond_with(SizeRoutingPutHandler {
                state: Arc::clone(&state),
                target_uri: target.uri(),
            })
            .mount(&storage)
            .await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/v1alpha/fileobject/by-address/"))
            .respond_with(SizeRoutingGetHandler {
                state: Arc::clone(&state),
                target_uri: target.uri(),
            })
            .mount(&storage)
            .await;
        // Not size-routed: storage-api's HEAD (`stat_by_address`) has no
        // redirect branch at any size.
        Mock::given(method("HEAD"))
            .and(path_regex(r"^/v1alpha/fileobject/by-address/"))
            .respond_with(HeadHandler {
                state: Arc::clone(&state),
            })
            .mount(&storage)
            .await;
        Mock::given(method("DELETE"))
            .and(path_regex(r"^/v1alpha/fileobject/by-address/"))
            .respond_with(DeleteHandler {
                state: Arc::clone(&state),
            })
            .mount(&storage)
            .await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/v1alpha/fileobject/data-objects/"))
            .respond_with(ListHandler {
                state: Arc::clone(&state),
            })
            .mount(&storage)
            .await;
        Mock::given(method("POST"))
            .and(path_regex(r"/redirect/complete$"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&storage)
            .await;

        Mock::given(method("PUT"))
            .and(path_regex(r"^/blob/"))
            .respond_with(TargetUploadHandler {
                state: Arc::clone(&state),
            })
            .mount(&target)
            .await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/blob/"))
            .respond_with(TargetDownloadHandler {
                state: Arc::clone(&state),
            })
            .mount(&target)
            .await;

        let token_dir = tempfile::tempdir().expect("tempdir for service-token");
        let metrics = Arc::new(crate::common::metrics_scrape_only(
            "storage-api-redirect-round-trip-test",
        ));
        let provider = ServiceTokenProvider::new(
            token_dir.path().join("service-token.json"),
            Duration::from_millis(50),
            metrics,
            CancellationToken::new(),
        );
        let store = StorageApiCheckpointStore::new(
            storage.uri(),
            PathBuf::from(BASE_PATH),
            Arc::new(provider),
            CONNECT_TIMEOUT,
            REQUEST_TIMEOUT,
        )
        .expect("StorageApiCheckpointStore::new must succeed on realistic inputs");

        RedirectFake {
            store,
            storage,
            target,
            _token_dir: token_dir,
        }
    }

    /// The recorded requests of `server`, in arrival order.
    async fn recorded(server: &MockServer) -> Vec<Request> {
        server
            .received_requests()
            .await
            .expect("mock server records requests")
    }

    /// Publish the large generation, read it back, and assert byte-identity —
    /// the property the whole b.z22 fix exists to restore.
    async fn publish_and_read_back(fake: &RedirectFake, payload: &Bytes) {
        let handle = fake
            .store
            .put_layer_checkpoint(
                OBJECT_ID,
                SNAP_SEQ,
                payload.clone(),
                conformance::sample_meta(SNAP_SEQ),
            )
            .await
            .expect(
                "a generation above the deployment's write threshold must publish — before b.z22 \
                 its 300 was classified Permanent and it never published at all",
            );
        assert_eq!(handle.object_id, OBJECT_ID);
        assert_eq!(handle.snap_seq, SNAP_SEQ);

        let blob = fake
            .store
            .get_layer_checkpoint(&handle)
            .await
            .expect(
                "and it must read back — before b.z22 the read 300 was Permanent too, which is \
                 what broke DR restore of every layer above readRedirectSize",
            )
            .expect("the generation was just published, so it must be present");

        assert_eq!(
            blob.bytes.len(),
            payload.len(),
            "the read-back payload must be the same LENGTH — a truncation would otherwise stay \
             invisible until an OMPF parse failed somewhere far away"
        );
        assert_eq!(
            blob.bytes, *payload,
            "and byte-identical: the bytes left through a presigned upload and came back through \
             a presigned download, and the two must address the same object"
        );
        assert_eq!(
            blob.meta.snap_seq,
            Some(SNAP_SEQ),
            "the meta travels inline (it is small) while the OMPF does not, so this pins that a \
             split-transport generation still pairs correctly"
        );
    }

    /// With the upload-options pre-flight answering, the payload travels
    /// **once** — the by-address PUT is empty-bodied — and still round-trips.
    #[tokio::test]
    async fn a_large_generation_round_trips_with_the_pre_flight_answering() {
        let fake = setup_redirect_fake(PreFlight::Answers).await;
        let payload = large_payload();
        publish_and_read_back(&fake, &payload).await;

        let requests = recorded(&fake.storage).await;
        let ompf_put = requests
            .iter()
            .find(|r| r.method.as_str() == "PUT" && r.url.path().ends_with("42.7.ompf"))
            .expect("the OMPF must have been PUT by address");
        assert_eq!(
            query_value(ompf_put, "upload_preference").as_deref(),
            Some("redirect"),
            "the pre-flight said this size does not travel inline, so the write says so up front"
        );
        assert!(
            ompf_put.body.is_empty(),
            "and carries no payload: the point of the pre-flight is that a large artifact is \
             never streamed into a socket the server has already stopped reading. Got {} bytes",
            ompf_put.body.len()
        );
        assert_eq!(
            query_value(ompf_put, "data_object_size").as_deref(),
            Some(payload.len().to_string().as_str()),
            "while still declaring the real size, which is what the server routes on"
        );

        assert_eq!(
            requests
                .iter()
                .filter(|r| r
                    .url
                    .path()
                    .starts_with("/v1alpha/fileobject/upload-options/"))
                .count(),
            1,
            "one probe for the life of the adapter, cached across the meta and OMPF puts"
        );

        let target_requests = recorded(&fake.target).await;
        let uploaded: Vec<usize> = target_requests
            .iter()
            .filter(|r| r.method.as_str() == "PUT")
            .map(|r| r.body.len())
            .collect();
        assert_eq!(
            uploaded,
            vec![payload.len()],
            "exactly one presigned upload, carrying the whole payload once"
        );
        assert_eq!(
            target_requests
                .iter()
                .filter(|r| r.method.as_str() == "GET")
                .count(),
            1,
            "and one presigned download on the way back — the read really did take the fallback \
             rather than the fake quietly serving it inline"
        );
    }

    /// With the pre-flight absent — an older storage-api, or one that does not
    /// grant `FetchWriteTypeInfo` — the adapter streams the payload and handles
    /// the 300 reactively, and the round trip is **observably identical**.
    ///
    /// That equivalence is the safety case for the pre-flight: it is an
    /// optimisation over a path that must keep working, not a new protocol.
    #[tokio::test]
    async fn a_large_generation_round_trips_with_the_pre_flight_absent() {
        let fake = setup_redirect_fake(PreFlight::Absent).await;
        let payload = large_payload();
        publish_and_read_back(&fake, &payload).await;

        let requests = recorded(&fake.storage).await;
        let ompf_put = requests
            .iter()
            .find(|r| r.method.as_str() == "PUT" && r.url.path().ends_with("42.7.ompf"))
            .expect("the OMPF must have been PUT by address");
        assert_eq!(
            query_value(ompf_put, "upload_preference"),
            None,
            "with no answer from the pre-flight the adapter states no preference — guessing one \
             from an answer the server never gave is what could turn a writable artifact into an \
             unwritable one"
        );
        assert_eq!(
            ompf_put.body.len(),
            payload.len(),
            "so the payload rides along and the 300 is handled when it arrives: the reactive path \
             costs an extra copy on the wire, and must still publish"
        );

        let uploaded: Vec<usize> = recorded(&fake.target)
            .await
            .iter()
            .filter(|r| r.method.as_str() == "PUT")
            .map(|r| r.body.len())
            .collect();
        assert_eq!(
            uploaded,
            vec![payload.len()],
            "and the durable copy still reaches the presigned target exactly once"
        );
    }
}
