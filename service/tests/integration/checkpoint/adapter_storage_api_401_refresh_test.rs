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

//! Integration tests for the storage-api adapter's 401 → `force_refresh()` →
//! retry-once flow.
//!
//! Subtask `t3.fgj.jd.pg.xd` under Task `t2.fgj.jd.pg`, Epic `t1.fgj.jd`.
//!
//! ## Scope vs. sibling unit subtask
//!
//! The unit-level coverage of the same outer auth-attempt loop lives in
//! `src/checkpoint/adapters/storage_api.rs::tests::force_refresh_tests`
//! (sibling Subtask `t3.fgj.jd.3v.98`). Those tests construct the adapter
//! and provider in-crate and exercise the loop against a wiremock backend
//! — they pin the per-method contract.
//!
//! This file pins the **integration boundary** — the same flow driven from
//! outside the crate (`live_edit::auth::service_token` + `live_edit::checkpoint`
//! public surfaces only), via a real on-disk `ServiceTokenProvider` with its
//! polling task, a real `tempfile::TempDir`, and a real `wiremock::MockServer`.
//! A passing test here proves the engineer's `send_classified` wrapper (landed
//! in Engineer Subtask `t3.fgj.jd.3v.84`) survives the end-to-end wiring an
//! Epic 5 caller would actually use.
//!
//! ## The contract these tests pin, post-b.e49
//!
//! Two things changed under this file, and every variant below is written
//! against the new shape:
//!
//! 1. **The provider has exactly ONE bearer accessor —
//!    [`ServiceTokenProvider::cached_bearer_if_unexpired`] — and it never
//!    withholds a still-valid token.** The `refresh_lead` is a proactive
//!    re-read cadence, never a validity floor. A token 30 s from expiry is a
//!    perfectly good credential and goes out on the first request
//!    ([`near_expiry_token_is_attached_and_succeeds_in_one_round_trip`]).
//!    Under the pre-b.e49 shape a second accessor withheld that same token
//!    inside the lead: the request went out unauthenticated and the adapter
//!    bought itself a guaranteed 401 plus a `force_refresh()`, once per token
//!    lifetime, fleet-wide. b.mq8 then deleted that accessor outright, so
//!    "which one does this site call" is no longer a question a reader can
//!    ask. `None` means only "nothing cached" or "the cached token has
//!    actually expired".
//!
//! 2. **The second 401 is two different failures, and the split is decided on
//!    what the rejected request actually carried on the wire.** Every site
//!    that builds a storage-api request records whether it attached a bearer —
//!    `cached_bearer_if_unexpired().is_some()`, evaluated at the moment that
//!    one request was built — and that bool rides the 401 upward as
//!    `bearer_attached` on `InnerOutcome::Unauthorized`. The terminal verdict
//!    reads it directly. Nothing about the token cache is consulted afterwards,
//!    so the rule states an observable fact about a specific HTTP request:
//!
//!    | Terminal shape | Verdict | Variant |
//!    |---|---|---|
//!    | the rejected retry **carried a bearer**; storage-api refused it | `Permanent` | [`unauthorized_then_same_live_token_surfaces_permanent`], [`second_401_after_rotation_surfaces_permanent`] |
//!    | the rejected retry carried **no `Authorization` header** — nothing usable was cached when it was built | `Transient` | [`expired_token_second_401_surfaces_transient`], [`token_expiring_mid_ladder_second_401_surfaces_transient`] |
//!    | `force_refresh()` itself failed (I/O, malformed JSON) | `Transient` | [`force_refresh_failure_surfaces_transient_with_no_retry`] |
//!
//!    Because the rule is on-wire, the tests below assert it on-wire: every
//!    terminal-verdict variant reads the `Authorization` header back off
//!    `MockServer::received_requests()` for the request that drew the final
//!    401, and pins its presence (`Permanent`) or absence (`Transient`)
//!    alongside the verdict. Those header assertions ARE the predicate; the
//!    `cached_bearer_if_unexpired()` / `cached_token_expires_on()` reads that
//!    accompany them are fixture preconditions — they explain *why* the
//!    request came out the way it did, and catch a fixture that has drifted
//!    onto the other arm, but they are not what `send_classified` decides on.
//!
//!    Note what is NOT in that table. First, whether the refresh observed a
//!    *rotation*: splitting on a rotation — comparing the cached `expires_on`
//!    either side of the refresh — was the pre-review shape and is wrong in
//!    the steady state, where `force_refresh()` re-reads the same on-disk
//!    document and `post == pre` on nearly every 401, so a genuinely rejected
//!    live credential would retry forever as `Transient` and never escalate.
//!    Second, a **pre-retry snapshot** of the cache: reading
//!    `cached_token_expires_on()` once, just after the refresh, was the
//!    review-round-1 correction and is also wrong, because "the retry" is not
//!    one request. It is a whole transient-retry ladder — up to
//!    `MAX_RETRY_ATTEMPTS` attempts, each bounded by the request timeout and
//!    separated by backoff — so a token live at the snapshot can be dead by
//!    the attempt that actually draws the terminal 401, and that attempt goes
//!    out unauthenticated. The snapshot would call that `Permanent`: a
//!    self-healing rotation gap reported to DR-restore clients as
//!    non-retriable, which is the harm b.e49 exists to remove.
//!    [`token_expiring_mid_ladder_second_401_surfaces_transient`] is this
//!    file's witness for that case, and the reason the predicate names the
//!    wire rather than the cache.
//!
//!    Both `Transient` arms are the sidecar-has-not-caught-up condition: no
//!    usable credential existed for the retry, which self-heals within seconds
//!    of the replacement landing, so a caller should retry. Classifying those
//!    `Permanent` made the layer-recovery path surface a self-healing gap to
//!    clients as a non-retriable `INTERNAL` once b.33e stopped masking the
//!    variant (SDD §18.1). The converse error is just as real and is why the
//!    `Permanent` arm is not merely tolerated but asserted: a rejected live
//!    credential retried as `Transient` would hide an operator-actionable
//!    auth-policy fault behind an unbounded retry storm.
//!
//! ## Where the variant → gRPC-code mapping is pinned
//!
//! Here we assert the `CheckpointError` **variant**. The mapping from that
//! variant to a client-visible status on the layer-recovery path
//! (`Transient` → `UNAVAILABLE`, everything else → `INTERNAL`) is pinned
//! exhaustively against a stub store in
//! [`super::recovery_hook_test`], and is NOT re-enumerated here. The two
//! composed cases at the foot of this file — the REAL adapter, a real
//! wiremock 401 storm, a real `RecoveryHook`, and a real StatLayer RPC — pin
//! only the two rows the b.e49 predicate can reach, one each side of it:
//!
//! - [`storage_api_second_401_with_expired_token_surfaces_unavailable_to_the_client`]
//!   — expired cached token, retry unauthenticated → `Transient` →
//!   `UNAVAILABLE`. The client keeps retrying across a sidecar gap that heals
//!   itself.
//! - [`storage_api_second_401_with_live_token_surfaces_internal_to_the_client`]
//!   — live cached token attached to both attempts and refused →
//!   `Permanent` → `INTERNAL`. The escalation reaches the client instead of
//!   dissolving into a retry storm.
//!
//! Together they prove the two halves compose on the exact path b.e49
//! describes, without duplicating the matrix.
//!
//! ## Determinism
//!
//! Every [`ScaffoldWithStore`] variant — that is, every test above the composed
//! pair at the foot of this file — builds its provider with a deliberately-long
//! poll interval (`QUIET_POLL_INTERVAL`, 1 hour) so the polling task cannot
//! tick during the test window. All cache mutations there are driven explicitly
//! through `force_refresh()` calls — by the scaffold constructor (to populate
//! the cache deterministically) and by the adapter's own 401 handler.
//!
//! The two composed cases at the foot are the exception, and the claim above
//! does NOT extend to them. They use [`setup_storage_api_checkpoint_harness`],
//! whose provider polls every 250 ms (`STORAGE_API_TOKEN_POLL_INTERVAL`,
//! `tests/integration/checkpoint/helpers.rs`), so its polling task ticks many
//! times per test. Those ticks are benign because what each case depends on is
//! a property of the token *file*, not of tick timing: a tick can only re-read
//! the same on-disk document an explicit `force_refresh()` already loaded, so
//! it can only rewrite the cache with the value that is already in it.
//! Concretely — case #2 never touches the file at all (the harness's live 1 h
//! token stands for the whole test), and case #1 rewrites it exactly once, to
//! an already-expired document, then drives `force_refresh()` itself before the
//! RPC starts. A tick landing before, between, or after that write/refresh pair
//! yields the same cache contents as the explicit refresh, and the file is
//! never written again, so `cached_bearer_if_unexpired()` holds one value for
//! the entire window under test. That is why both composed cases can bracket
//! their RPC with cache assertions instead of racing the poller.
//!
//! The only `tokio::time::sleep` allowed is the post-`cancel()` yield-once
//! pattern from `service_token_rotation.rs:131`, and we keep that to 50 ms.
//!
//! One variant — [`token_expiring_mid_ladder_second_401_surfaces_transient`] —
//! must let a cached token expire *during* an adapter call, which no injectable
//! clock exists for. It buys determinism without a client-side sleep by holding
//! one response open server-side (`ResponseTemplate::set_delay`) for strictly
//! longer than the fixture token's remaining life, so the ordering it depends on
//! is enforced rather than raced. See that test's docblock for the two timing
//! assumptions and the assertions that name them.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use chrono::Utc;
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{any, header, method, path_regex};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use live_edit::auth::service_token::ServiceTokenProvider;
use live_edit::checkpoint::adapters::StorageApiCheckpointStore;
use live_edit::checkpoint::{CheckpointError, CheckpointMeta, CheckpointStore};
use live_edit::observability::metrics::Metrics;

// Composed-path (deliverable 2) imports — used only by the two
// `storage_api_second_401_*_to_the_client` cases at the foot of this file.
use super::helpers::{setup_storage_api_checkpoint_harness, wait_for_checkpoint, wipe_layer_state};
use crate::grpc::helpers::create_session_and_layer;
use live_edit::grpc::layer::handle_stat_layer;
use live_edit::grpc::proto::StatLayerRequest;
use live_edit::grpc::BearerToken;

// ---------------------------------------------------------------------------
// Fixed configuration — mirrors `adapter_storage_api_test.rs` so per-test
// behaviour matches the conformance harness's adapter wiring exactly.
// ---------------------------------------------------------------------------

/// Connect-phase timeout for the adapter's `reqwest::Client`.
const CONNECT_TIMEOUT: Duration = Duration::from_millis(5_000);

/// Per-request timeout for the adapter's `reqwest::Client`.
const REQUEST_TIMEOUT: Duration = Duration::from_millis(30_000);

/// Fixed base path for the adapter under test. Pinning this is enough to
/// derive the expected percent-encoded `by-address` route tail used by the
/// wiremock matchers below.
const BASE_PATH: &str = "/projects/team-a";

/// Deliberately-long poll interval — the background polling task is NOT
/// allowed to refresh the cache during these tests. Only explicit
/// `force_refresh()` calls (either from the test setup or from the adapter's
/// 401 handler) may mutate the cache.
const QUIET_POLL_INTERVAL: Duration = Duration::from_secs(3_600);

/// Session ID used by all three tests. The wiremock route matchers below
/// expect the percent-encoded form `sessions%2Fsess-abc.json` because the
/// adapter percent-encodes the entire composed address through
/// `auth::rest::encode_resource_address`.
const SESSION_ID: &str = "sess-abc";

/// Token file basename used inside each per-test `TempDir`. Matches the
/// conventional sidecar mount path (`/var/run/oauth/access_token`) basename.
const TOKEN_FILE_NAME: &str = "access_token";

// ---------------------------------------------------------------------------
// Test scaffolding
// ---------------------------------------------------------------------------

/// Atomically write `contents` to `path` via a sibling tempfile + rename.
///
/// Mirrors the production credential-provisioner sidecar's write contract:
/// a partial reader on the other side never observes a torn file. Using a
/// sibling in the same directory keeps `rename` cross-filesystem-safe
/// (rename across filesystems is `EXDEV`). This is the same atomic-rename
/// pattern that `tests/integration/service_token_rotation.rs:57-68` uses.
fn atomic_write(path: &Path, contents: &str) {
    let parent = path
        .parent()
        .expect("token file path must have a parent directory");
    let mut tmp = tempfile::NamedTempFile::new_in(parent)
        .expect("NamedTempFile::new_in must succeed in test tempdir");
    use std::io::Write;
    tmp.write_all(contents.as_bytes())
        .expect("write to tempfile must succeed");
    tmp.persist(path)
        .expect("persist (atomic rename) must succeed");
}

/// Route matcher for "this request carried no `Authorization` header".
///
/// The complement of `wiremock::matchers::header`, which this crate's wiremock
/// version has no negative form of. Used where the *absence* of a credential is
/// the thing under test, so a regression that starts attaching one falls
/// through to wiremock's 404 instead of quietly matching an authenticated
/// route. Mirrors the same-named helper in the sibling unit module
/// (`src/checkpoint/adapters/storage_api.rs::tests::force_refresh_tests`).
fn no_authorization_header(request: &Request) -> bool {
    !request.headers.contains_key("authorization")
}

/// Whether a **recorded** request carried an `Authorization` header.
///
/// This is the on-wire fact the b.e49 verdict turns on — the adapter's
/// `bearer_attached`, read back off the mock server's own request log instead
/// of re-derived from the token cache. Every terminal-verdict test below pairs
/// it with the verdict so the two are pinned as one statement: a `Permanent`
/// must be accompanied by a final request that carried a credential, a
/// `Transient` by one that did not. Same helper, same role, as the one in the
/// sibling unit module.
fn recorded_authorization_present(request: &Request) -> bool {
    request.headers.contains_key("authorization")
}

/// Build a JSON token document expiring `expires_in_secs` from now.
///
/// `expires_in_secs` is the only lever this file has on the expiry dimension
/// (both bearer accessors read `Utc::now()` directly; there is no injectable
/// clock), and post-b.e49 it is the lever that decides what each request puts
/// on the wire — which is in turn what decides the **verdict**:
///
/// - **A positive value is a live credential.** The adapter attaches it
///   regardless of how close to the refresh lead it sits, so the request goes
///   out bearing an `Authorization` header and a 401 against it means
///   storage-api refused a credential the provider considers valid —
///   `Permanent`.
/// - **A negative value is an already-expired credential.**
///   `cached_bearer_if_unexpired()` returns `None`, the request goes out with
///   no `Authorization` header, and a 401 against it means no usable token
///   existed when that request was built — `Transient`.
///
/// The sign of this argument is therefore the whole fixture. It is still only
/// the fixture: the classification is read off `bearer_attached`, recorded by
/// the send site for the one request that drew the terminal 401, so a token
/// whose sign flips *mid-call* moves the verdict with it (see
/// [`token_expiring_mid_ladder_second_401_surfaces_transient`]). The token
/// *bytes* select which wiremock route matches and make the on-wire story
/// legible in a failure message, but they never decide the classification —
/// and neither does whether the value differs from the previously-cached one.
fn token_json_expiring_in(access_token: &str, expires_in_secs: i64) -> String {
    let expires_on = Utc::now() + chrono::Duration::seconds(expires_in_secs);
    format!(
        "{{\"expires_on\":\"{}\",\"access_token\":\"{}\"}}",
        expires_on.to_rfc3339(),
        access_token
    )
}

/// The common case: a token an hour out. Comfortably live, and comfortably
/// clear of the provider's 60 s refresh lead so no test that uses it is
/// accidentally exercising the near-expiry boundary.
fn fresh_token_json(access_token: &str) -> String {
    token_json_expiring_in(access_token, 3_600)
}

/// Bundle of per-test resources whose lifetimes must outlive both the
/// adapter call AND the provider's polling task.
///
/// - `mock_server` owns the in-process HTTP listener; dropping it tears the
///   listener down.
/// - `provider` owns the polling task; the task watches `cancel`. Kept on
///   the scaffold so tests can read the cache through the same accessor each
///   send site reads when it decides whether to attach a bearer. Every
///   second-401 variant uses it to state its fixture precondition as an
///   assertion rather than a comment: the `Permanent` variants assert
///   `cached_bearer_if_unexpired()` still serves a token, the `Transient`
///   variants assert it does not, and the near-expiry variant additionally
///   reads `cached_token_expires_on()` to pin "this token sits inside the
///   refresh lead" — the window in which a second, withholding accessor used
///   to suppress it, before b.mq8 deleted that accessor and left the provider
///   with one. None of those reads is the
///   verdict's predicate — that is the `Authorization` header on the recorded
///   request, asserted separately through
///   [`recorded_authorization_present`]. They exist so a fixture that has
///   drifted onto the opposite arm fails as a precondition rather than as a
///   confusing verdict mismatch.
/// - `_token_dir` is the `TempDir` backing the on-disk token file; it must
///   outlive the polling task (the task tolerates a missing file silently
///   per SR-6.2's startup-absent contract, but tearing the dir down while
///   the task is mid-read is racy and not worth exercising here).
/// - `token_path` is the path inside `_token_dir` the test rewrites between
///   adapter calls.
/// - `metrics` is exposed so Variant 3 can assert the parse-error counter.
/// - `cancel` MUST be cancelled at end-of-test so the polling task exits.
/// - `store` is the trait-object adapter handle the test bodies call into.
struct ScaffoldWithStore {
    mock_server: MockServer,
    provider: Arc<ServiceTokenProvider>,
    _token_dir: TempDir,
    token_path: PathBuf,
    metrics: Arc<Metrics>,
    cancel: CancellationToken,
    store: Arc<dyn CheckpointStore>,
}

impl ScaffoldWithStore {
    /// Construct the per-test scaffold.
    ///
    /// 1. Spin up a fresh `wiremock::MockServer`.
    /// 2. Allocate a fresh `TempDir`, atomic-write the initial token JSON
    ///    (`expires_on = now + 1h`, `access_token = initial-token`) to
    ///    `<tempdir>/access_token`.
    /// 3. Build a `ServiceTokenProvider` with a deliberately-long poll
    ///    interval (1 h) so the background task cannot tick during the
    ///    test window, and prime its cache with one explicit
    ///    `force_refresh().await` so the first adapter call sees a
    ///    populated bearer.
    /// 4. Build a `StorageApiCheckpointStore` pointed at the mock server.
    async fn new() -> Self {
        Self::with_initial_expiry(3_600).await
    }

    /// [`Self::new`] with the initial token's lifetime under test control.
    ///
    /// Three variants need a value other than an hour:
    ///
    /// - `30` — inside the provider's 60 s refresh lead but still live: the
    ///   exact fixture the pre-b.e49 adapter would have refused to attach
    ///   ([`near_expiry_token_is_attached_and_succeeds_in_one_round_trip`]).
    /// - `6` — live at the first two attempts and dead by the third, so the
    ///   credential's liveness *changes inside one adapter call*. Pairs with a
    ///   longer server-side response delay (10 s) to make the transition
    ///   deterministic in both directions
    ///   ([`token_expiring_mid_ladder_second_401_surfaces_transient`]; the two
    ///   numbers are a matched pair and must move together).
    /// - `-30` — already expired. `force_refresh()` in step 3 still
    ///   *succeeds* (the document parses; `read_and_update` caches whatever
    ///   `expires_on` it finds and never rejects a past one), so the cache
    ///   ends up holding a dead token rather than nothing. That is precisely
    ///   the sidecar-gap state the issue describes:
    ///   `cached_bearer_if_unexpired()` returns `None`, every send site goes
    ///   out unauthenticated, and a second 401 must classify `Transient`
    ///   ([`expired_token_second_401_surfaces_transient`]).
    async fn with_initial_expiry(expires_in_secs: i64) -> Self {
        let mock_server = MockServer::start().await;

        let token_dir = TempDir::new().expect("tempdir for token file");
        let token_path = token_dir.path().join(TOKEN_FILE_NAME);
        atomic_write(
            &token_path,
            &token_json_expiring_in("initial-token", expires_in_secs),
        );

        let metrics = Arc::new(crate::common::metrics_scrape_only(
            "storage-api-401-refresh-int-test",
        ));
        let cancel = CancellationToken::new();
        let provider = Arc::new(ServiceTokenProvider::new(
            token_path.clone(),
            QUIET_POLL_INTERVAL,
            Arc::clone(&metrics),
            cancel.clone(),
        ));

        // Deterministically prime the cache — see file-level Determinism note.
        provider
            .force_refresh()
            .await
            .expect("initial force_refresh on a present, valid file must succeed");

        let adapter = StorageApiCheckpointStore::new(
            mock_server.uri(),
            PathBuf::from(BASE_PATH),
            Arc::clone(&provider),
            CONNECT_TIMEOUT,
            REQUEST_TIMEOUT,
        )
        .expect("storage-api adapter constructor must succeed on realistic inputs");

        ScaffoldWithStore {
            mock_server,
            provider,
            _token_dir: token_dir,
            token_path,
            metrics,
            cancel,
            store: Arc::new(adapter),
        }
    }

    /// Cancel the polling task and yield once so the task observes the
    /// cancel signal before `TempDir`'s `Drop` tears down the file. This is
    /// the post-cancel yield-once pattern from
    /// `tests/integration/service_token_rotation.rs:131` — the only sleep
    /// the determinism contract allows in this file.
    async fn shutdown(self) {
        self.cancel.cancel();
        tokio::time::sleep(Duration::from_millis(50)).await;
        // `self` (and therefore the `MockServer`, `TempDir`, and adapter)
        // drops here, after the polling task has had a chance to observe
        // the cancel signal.
    }
}

// ---------------------------------------------------------------------------
// Variant 1 — rotated-token retry succeeds (Ok + exactly 2 requests)
// ---------------------------------------------------------------------------

/// Variant 1: First attempt with `Bearer initial-token` → 401. Before the
/// adapter call, the token file is atomically rewritten to `rotated-token`.
/// The adapter's 401 handler invokes `force_refresh()`, the provider re-reads
/// the file, the cache now holds `rotated-token`, and the retry attempt
/// sends `Bearer rotated-token` → 200.
///
/// Assertions:
/// - Adapter call returns `Ok(...)`.
/// - Exactly 2 storage-api requests:
///   1. `Bearer initial-token` → 401
///   2. `Bearer rotated-token` → 200
///
/// `Mock::expect(1)` on each header-scoped route is what asserts the
/// per-route count; the combined `received_requests()` length pins the
/// total.
#[tokio::test]
async fn unauthorized_then_rotated_token_succeeds() {
    let scaffold = ScaffoldWithStore::new().await;

    // Route 1: Bearer initial-token → 401 (the stale credential).
    Mock::given(method("PUT"))
        .and(path_regex(r"sessions%2Fsess-abc\.json$"))
        .and(header("authorization", "Bearer initial-token"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&scaffold.mock_server)
        .await;

    // Route 2: Bearer rotated-token → 200 (the freshly-rotated credential).
    Mock::given(method("PUT"))
        .and(path_regex(r"sessions%2Fsess-abc\.json$"))
        .and(header("authorization", "Bearer rotated-token"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&scaffold.mock_server)
        .await;

    // Rotate the token file BEFORE the adapter call, atomically. The
    // adapter's 401 handler will invoke `force_refresh()` and re-read this
    // new content.
    atomic_write(&scaffold.token_path, &fresh_token_json("rotated-token"));

    scaffold
        .store
        .put_session_checkpoint(
            SESSION_ID,
            Bytes::from_static(b"payload"),
            CheckpointMeta::default(),
        )
        .await
        .expect("rotated-token retry must succeed");

    let received = scaffold
        .mock_server
        .received_requests()
        .await
        .expect("mock server records requests");
    assert_eq!(
        received.len(),
        2,
        "expected exactly 2 on-wire attempts (init-token 401 + rotated-token 200), got {}",
        received.len()
    );

    scaffold.shutdown().await;
}

// ---------------------------------------------------------------------------
// Variant 2 — unchanged LIVE token, second 401 surfaces as Permanent (no
// third request)
// ---------------------------------------------------------------------------

/// Variant 2: First attempt with `Bearer initial-token` → 401. The token file
/// is NOT rewritten. The adapter's 401 handler invokes `force_refresh()`, the
/// provider re-reads the same still-live `initial-token` from disk, the retry
/// attempt also sends `Bearer initial-token`, and wiremock returns 401 again.
///
/// The verdict is `CheckpointError::Permanent`. A live credential went out on
/// the wire twice and storage-api refused it both times — either the sidecar
/// is minting tokens the backend will not accept or the auth policy rejects
/// this service principal. Neither heals on its own, so the caller must stop
/// retrying and an operator must look.
///
/// This is the steady state, and that is why the classification cannot be
/// decided on a *rotation*. `force_refresh()` normally re-reads the same
/// on-disk document — nothing rotated here, and nothing rotates on the vast
/// majority of production 401s. Under the pre-review rotation predicate this
/// exact shape classified `Transient` and retried forever, which is the
/// failure mode the corrected predicate exists to prevent: an
/// operator-actionable fault dissolved into an unbounded retry storm.
///
/// Read this as a pair with [`expired_token_second_401_surfaces_transient`]:
/// identical two-401 shape, opposite verdict, and the *only* difference is
/// whether the rejected retry carried an `Authorization` header. The contrast
/// is the contract. See
/// [`storage_api_second_401_with_live_token_surfaces_internal_to_the_client`]
/// for this verdict composed all the way to a client-visible status.
///
/// Assertions:
/// - Adapter call returns `Err(CheckpointError::Permanent)`.
/// - Source string is the post-refresh-401 one, so the test cannot pass on
///   some *other* `Permanent` (a 4xx that is not 401, an invalid-bearer-bytes
///   rejection) that happens to share the variant.
/// - Exactly 2 storage-api requests, both carrying `Bearer initial-token`,
///   checked from the recorded-request side. **That is the predicate**: the
///   request that drew the terminal 401 carried a bearer, which is the
///   `bearer_attached == true` the adapter classified on — not an inference
///   from the token cache.
/// - The post-call cache still serves that credential
///   (`cached_bearer_if_unexpired()`). A fixture precondition explaining why
///   the retry attached anything, not the predicate.
/// - NO third request — the auth-attempt loop is bounded at two iterations.
#[tokio::test]
async fn unauthorized_then_same_live_token_surfaces_permanent() {
    let scaffold = ScaffoldWithStore::new().await;

    // Single header-scoped route. `Mock::expect(2)` asserts the per-route
    // count; on `MockServer` drop wiremock panics if the count is off.
    Mock::given(method("PUT"))
        .and(path_regex(r"sessions%2Fsess-abc\.json$"))
        .and(header("authorization", "Bearer initial-token"))
        .respond_with(ResponseTemplate::new(401))
        .expect(2)
        .mount(&scaffold.mock_server)
        .await;

    let err = scaffold
        .store
        .put_session_checkpoint(
            SESSION_ID,
            Bytes::from_static(b"payload"),
            CheckpointMeta::default(),
        )
        .await
        .expect_err("two consecutive 401s must surface as an error");

    assert!(
        !err.is_transient(),
        "a live credential was attached to both attempts and refused twice — that is a real \
         credential rejection and must NOT be retriable, or the steady-state 401 (where \
         force_refresh re-reads the same good file) retries forever and never escalates. \
         Got {err:?}"
    );
    let CheckpointError::Permanent { source } = &err else {
        panic!("expected CheckpointError::Permanent, got {err:?}");
    };
    let source_str = format!("{source}");
    assert!(
        source_str.contains("HTTP 401 after force_refresh"),
        "expected source to name the post-refresh-401 contract (and NOT merely to be some \
         other Permanent sharing the variant), got `{source_str}`"
    );

    // Fixture precondition, stated through the accessor each send site reads
    // when it decides whether to attach a bearer: the cache still holds a live
    // token, which is why the retry could not have gone out unauthenticated.
    // The proof that it did not is the header assertion below; this one
    // explains it, and catches a fixture whose token aged out mid-test.
    assert!(
        scaffold.provider.cached_bearer_if_unexpired().is_some(),
        "fixture precondition: the re-read token is still live, so every send site in this test \
         had a credential to attach — otherwise this would be the unauthenticated (Transient) \
         arm wearing the wrong name"
    );

    // No third request: the outer auth-attempt loop is bounded at two
    // iterations regardless of what the cache holds.
    let received = scaffold
        .mock_server
        .received_requests()
        .await
        .expect("mock server records requests");
    assert_eq!(
        received.len(),
        2,
        "second 401 must NOT trigger a third attempt; expected 2 on-wire PUTs, got {}",
        received.len()
    );
    // THE predicate, read back off the wire: the request that drew the
    // terminal 401 carried an `Authorization` header, so the adapter recorded
    // `bearer_attached == true` and owed a `Permanent`. Asserted from the
    // `received_requests()` side rather than left to the header-scoped
    // `Mock::expect(2)`, both because that route count could in principle be
    // satisfied by a wiremock change in how header-mismatched fall-through
    // requests are attributed, and because this is the fact the verdict is
    // about — it belongs next to the verdict assertion, not in a mock builder.
    assert!(
        recorded_authorization_present(&received[1]),
        "the terminal request must carry a credential — `Permanent` is exactly the claim that \
         storage-api refused one that was presented"
    );
    // And the stronger form: every request carried the *unchanged live*
    // bearer, which is what makes this the steady-state shape rather than a
    // rotation.
    for req in &received {
        let auth = req
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert_eq!(
            auth, "Bearer initial-token",
            "both requests must carry the unchanged live bearer; got `{auth}`"
        );
    }

    scaffold.shutdown().await;
}

// ---------------------------------------------------------------------------
// Variant 2b — expired cached token, unauthenticated retry, second 401
// surfaces as Transient
// ---------------------------------------------------------------------------

/// Variant 2b — the issue's own motivating scenario, and the sibling that
/// makes [`unauthorized_then_same_live_token_surfaces_permanent`] a contrast
/// rather than a blanket rule.
///
/// The scaffold's cached token is already 30 s dead before the first request.
/// `cached_bearer_if_unexpired()` therefore returns `None` and the first
/// attempt goes out with **no** `Authorization` header at all; storage-api
/// 401s it. The adapter refreshes, the provider re-reads the same expired
/// document (a real sidecar gap: the replacement has not been written yet),
/// the cache is still dead, and the retry also goes out unauthenticated and
/// 401s.
///
/// Nothing in that sequence is a statement about credential validity — no
/// credential was ever presented. It says the sidecar has not caught up,
/// which self-heals within seconds of the replacement landing, so the verdict
/// must be `Transient` and the caller must keep retrying.
///
/// Note what the fixture does NOT do: it never rewrites the token file. Under
/// the pre-review rotation predicate that made this arm indistinguishable
/// from Variant 2 (neither rotates); under the corrected predicate the two
/// separate cleanly on the only thing that was ever observable to
/// storage-api — whether the retry carried a credential.
///
/// Assertions:
/// - Both on-wire attempts carry NO `Authorization` header. For the second of
///   them **that is the predicate** — the adapter recorded
///   `bearer_attached == false` for exactly that request — so it is asserted
///   from the `received_requests()` side rather than inferred from a route
///   match or from the cache.
/// - Adapter call returns `Err(CheckpointError::Transient)`.
/// - Source string names the no-usable-token cause, so the test cannot pass
///   on some *other* `Transient` (a connect failure, the b.gg7 ceiling, a
///   failed `force_refresh()`) that happens to share the variant.
/// - Exactly 2 storage-api requests — the loop is still bounded at two
///   iterations when the cache is dead.
#[tokio::test]
async fn expired_token_second_401_surfaces_transient() {
    // 30 s past expiry: cached (the document parses, so `read_and_update`
    // stores it) but unusable.
    let scaffold = ScaffoldWithStore::with_initial_expiry(-30).await;

    // Deliberately NOT header-scoped: the contract under test is that these
    // requests carry no credential, so the route must match regardless and
    // the header assertion happens below against the recorded requests. A
    // header-scoped route would silently turn a regression (a bearer that
    // should not exist) into a wiremock 404 and a confusing failure.
    Mock::given(method("PUT"))
        .and(path_regex(r"sessions%2Fsess-abc\.json$"))
        .respond_with(ResponseTemplate::new(401))
        .expect(2)
        .mount(&scaffold.mock_server)
        .await;

    // Fixture precondition: the cache holds a token, and that token is dead.
    // Both halves matter — "nothing cached" would reach the same verdict by a
    // different route, and this test means to pin the expired-token one.
    assert!(
        scaffold
            .provider
            .cached_token_expires_on()
            .is_some_and(|e| e < Utc::now()),
        "fixture precondition: force_refresh must have cached the expired document (it parses \
         fine — the provider never rejects a past expires_on), got {:?}",
        scaffold.provider.cached_token_expires_on()
    );
    assert_eq!(
        scaffold.provider.cached_bearer_if_unexpired(),
        None,
        "fixture precondition: an expired cached token must NOT be served, so every send site \
         goes out unauthenticated"
    );

    let err = scaffold
        .store
        .put_session_checkpoint(
            SESSION_ID,
            Bytes::from_static(b"payload"),
            CheckpointMeta::default(),
        )
        .await
        .expect_err("two consecutive 401s must surface as an error");

    assert!(
        err.is_transient(),
        "no credential was ever presented, so the 401s say nothing about credential validity: \
         this is the sidecar-has-not-caught-up condition and it must stay retriable. Got {err:?}"
    );
    let CheckpointError::Transient { source } = &err else {
        panic!("expected CheckpointError::Transient, got {err:?}");
    };
    let source_str = format!("{source}");
    assert!(
        source_str.contains("no usable service token was available for the retry"),
        "expected source to name the no-usable-token cause (and NOT merely to be some other \
         Transient sharing the variant), got `{source_str}`"
    );

    let received = scaffold
        .mock_server
        .received_requests()
        .await
        .expect("mock server records requests");
    assert_eq!(
        received.len(),
        2,
        "an expired cache must still cost exactly the adapter's two attempts (initial + one \
         post-force_refresh retry), got {}",
        received.len()
    );
    // The predicate, read back off the wire: neither attempt carried a
    // credential, so the request that drew the terminal 401 was recorded with
    // `bearer_attached == false` and the verdict owed is `Transient`.
    for (i, req) in received.iter().enumerate() {
        assert!(
            !recorded_authorization_present(req),
            "attempt {} must go out with NO Authorization header (the cached token is \
             expired); got `{:?}`",
            i + 1,
            req.headers.get("authorization")
        );
    }

    scaffold.shutdown().await;
}

// ---------------------------------------------------------------------------
// Variant 3 — force_refresh failure surfaces as Transient (no retry)
// ---------------------------------------------------------------------------

/// Variant 3 (re-classified by b.e49): First attempt with
/// `Bearer initial-token` → 401. Before the adapter call, the token file is
/// atomically rewritten with malformed JSON (`"not-json-at-all"`). When the
/// adapter's 401 handler invokes `force_refresh()`, the provider's parser
/// surfaces `ServiceTokenError::Parse`; `send_classified` MUST short-circuit
/// the retry and surface `CheckpointError::Transient` carrying the documented
/// `"service-token force_refresh failed:"` source-string prefix.
///
/// `Transient` because a failed *re-read* is not a credential verdict — it
/// says nothing about whether the token is valid — and `read_and_update`'s
/// previous-good-cache invariant leaves the cache untouched, so the next poll
/// or the next call can still recover (SDD §14.6).
///
/// Assertions:
/// - Adapter call returns `Err(CheckpointError::Transient)`.
/// - Source string contains `"service-token force_refresh failed"`.
/// - Exactly 1 storage-api request (NOT 2 — the failed `force_refresh()`
///   suppresses the retry).
/// - Bonus: `live_edit::observability::metrics::test_util::sample_value(&live_edit::observability::metrics::test_util::render(&metrics), "live_edit_service_token_parse_errors_total", &[]).unwrap_or(0.0) >= 1` — at
///   least the adapter-driven `force_refresh()` call recorded a parse error.
#[tokio::test]
async fn force_refresh_failure_surfaces_transient_with_no_retry() {
    let scaffold = ScaffoldWithStore::new().await;

    // Single header-scoped route. `Mock::expect(1)` pins the one expected
    // request — if the adapter mistakenly retries (with any bearer, or
    // none), the retry falls through to wiremock's default 404 handler,
    // which both breaks the test path AND fires this mock's expect check
    // because the count is exact.
    Mock::given(method("PUT"))
        .and(path_regex(r"sessions%2Fsess-abc\.json$"))
        .and(header("authorization", "Bearer initial-token"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&scaffold.mock_server)
        .await;

    // Capture the parse-error counter BEFORE rewriting the file so we can
    // assert the adapter-driven `force_refresh()` ticks it. The pre-call
    // `provider.force_refresh().await` in `Scaffold::new` already succeeded
    // (valid initial JSON), so the counter starts at 0; the malformed
    // rewrite below is the FIRST parse failure this provider will see.
    let parse_errors_before = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&scaffold.metrics),
        "live_edit_service_token_parse_errors_total",
        &[],
    )
    .unwrap_or(0.0);

    // Atomically swap in malformed JSON BEFORE the adapter call so that
    // when the adapter's 401 handler invokes `force_refresh()`, the
    // provider hits the parse-error path.
    atomic_write(&scaffold.token_path, "not-json-at-all");

    let err = scaffold
        .store
        .put_session_checkpoint(
            SESSION_ID,
            Bytes::from_static(b"payload"),
            CheckpointMeta::default(),
        )
        .await
        .expect_err("force_refresh parse failure must surface as an error");

    assert!(
        err.is_transient(),
        "a failed token re-read is an availability condition, not a credential verdict: \
         expected CheckpointError::Transient on force_refresh parse failure, got {err:?}"
    );
    let CheckpointError::Transient { source } = &err else {
        panic!("expected CheckpointError::Transient, got {err:?}");
    };
    let source_str = format!("{source}");
    assert!(
        source_str.contains("service-token force_refresh failed"),
        "expected source to reference the refresh-failure contract, got `{source_str}`"
    );

    let received = scaffold
        .mock_server
        .received_requests()
        .await
        .expect("mock server records requests");
    assert_eq!(
        received.len(),
        1,
        "force_refresh failure MUST short-circuit the retry; expected exactly 1 on-wire PUT, got {}",
        received.len()
    );

    // Bonus: assert the parse-error counter ticked. Use `>=` rather than
    // `==` because the polling task (parked on a 1 h interval) cannot have
    // fired during this short test, but we deliberately do not encode that
    // expectation as `== 1` — a future change to the provider's startup
    // semantics that adds an early read MUST NOT silently break this
    // bonus assertion. The contract is: at least the adapter-driven
    // `force_refresh()` call ticked the counter.
    let parse_errors_after = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&scaffold.metrics),
        "live_edit_service_token_parse_errors_total",
        &[],
    )
    .unwrap_or(0.0);
    assert!(
        parse_errors_after > parse_errors_before,
        "expected service_token_parse_errors_total to tick on the adapter-driven \
         force_refresh; before={parse_errors_before} after={parse_errors_after}"
    );

    scaffold.shutdown().await;
}

// ---------------------------------------------------------------------------
// Variant 4 (b.e49) — refresh lands a DIFFERENT live token, second 401 →
// Permanent (rotation is the fixture; liveness is the predicate)
// ---------------------------------------------------------------------------

/// Variant 4: First attempt with `Bearer initial-token` → 401. Before the
/// adapter call the token file is atomically rewritten to `rotated-token`
/// with a different lifetime (2 h vs the scaffold's 1 h). The adapter's 401
/// handler refreshes, the cache moves onto the new live token, the retry
/// sends `Bearer rotated-token` — and storage-api rejects *that*.
///
/// That is the operator escalation: an unexpired credential was presented on
/// the wire and refused, so either the sidecar is minting bad tokens or the
/// storage-api auth policy rejects the service principal. Verdict
/// `Permanent`, and the client should stop retrying.
///
/// ## What the rotation is and is not doing here
///
/// The rotation is **fixture, not predicate**. `send_classified` decides on
/// `bearer_attached` — did the request that drew the terminal 401 carry an
/// `Authorization` header? — and never looks at the token cache, let alone at
/// what it held before the refresh. So this test would reach the same
/// `Permanent` verdict with the file left untouched (that is
/// [`unauthorized_then_same_live_token_surfaces_permanent`], which is the
/// steady-state shape). What the rotation buys is a *stronger* observation:
/// because the two credentials are distinct on the wire, the header-scoped
/// routes below prove the refresh actually ran and that the credential the
/// backend refused was the freshly-minted one. A rotation predicate would
/// have made this the only Permanent case; under the corrected rule it is
/// simply the loudest one.
///
/// The distinct lifetime (7200 s vs 3600 s) is retained as a fixture check
/// for the same reason: two same-lifetime writes milliseconds apart differ
/// only by sub-second jitter, so a moved `expires_on` is a crisp signal that
/// the re-read landed rather than an accident of clock resolution. It is a
/// statement about the *file*, not about the verdict.
///
/// Sibling of [`expired_token_second_401_surfaces_transient`] — identical
/// two-401 shape, opposite verdict, and the difference is whether a
/// credential reached storage-api at all.
#[tokio::test]
async fn second_401_after_rotation_surfaces_permanent() {
    let scaffold = ScaffoldWithStore::new().await;

    // Route 1: the stale credential → 401.
    Mock::given(method("PUT"))
        .and(path_regex(r"sessions%2Fsess-abc\.json$"))
        .and(header("authorization", "Bearer initial-token"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&scaffold.mock_server)
        .await;

    // Route 2: the genuinely-rotated credential → ALSO 401.
    Mock::given(method("PUT"))
        .and(path_regex(r"sessions%2Fsess-abc\.json$"))
        .and(header("authorization", "Bearer rotated-token"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&scaffold.mock_server)
        .await;

    // 7200 s vs the scaffold's 3600 s. Both are live, so both would classify
    // Permanent; the distinct lifetime exists so the fixture check below can
    // tell "the re-read landed" from "nothing happened" — see the docblock.
    let pre_expiry = scaffold.provider.cached_token_expires_on();
    atomic_write(
        &scaffold.token_path,
        &token_json_expiring_in("rotated-token", 7_200),
    );

    let err = scaffold
        .store
        .put_session_checkpoint(
            SESSION_ID,
            Bytes::from_static(b"payload"),
            CheckpointMeta::default(),
        )
        .await
        .expect_err("two consecutive 401s must surface as an error");

    let CheckpointError::Permanent { source } = &err else {
        panic!(
            "a different, unexpired credential was presented and rejected — that is a real \
             credential rejection and must classify Permanent. Got {err:?}"
        );
    };
    let source_str = format!("{source}");
    assert!(
        source_str.contains("HTTP 401 after force_refresh"),
        "expected source to reference the post-refresh-401 contract, got `{source_str}`"
    );

    // Fixture precondition: the re-read token is still unexpired. That is not
    // the predicate — it is what MAKES the predicate come out `true`, because
    // an unexpired cached token is exactly the condition under which the retry
    // attaches an `Authorization` header. The predicate itself is that header,
    // asserted on the recorded terminal request below.
    let post_expiry = scaffold.provider.cached_token_expires_on();
    assert!(
        post_expiry.is_some_and(|e| e > Utc::now()),
        "fixture precondition: the rotated token must be UNEXPIRED, or the retry would have gone \
         out unauthenticated and this test would be the Transient arm wearing the wrong name; \
         post={post_expiry:?}"
    );

    // Fixture check, NOT the predicate (see the docblock): the re-read really
    // did land the new document. Kept because it is what licenses reading the
    // `Bearer rotated-token` route match below as proof the refresh ran.
    assert!(
        post_expiry != pre_expiry,
        "fixture check: the adapter's force_refresh must have picked up the rewritten file; \
         pre={pre_expiry:?} post={post_expiry:?}"
    );

    let received = scaffold
        .mock_server
        .received_requests()
        .await
        .expect("mock server records requests");
    assert_eq!(
        received.len(),
        2,
        "second 401 must NOT trigger a third attempt; expected 2 on-wire PUTs, got {}",
        received.len()
    );
    // THE predicate: the rejected retry carried a bearer (the rotated one, per
    // the header-scoped route it matched), so the adapter recorded
    // `bearer_attached == true` and owed a `Permanent`.
    assert!(
        recorded_authorization_present(&received[1]),
        "the terminal request must carry the freshly-rotated credential — `Permanent` is the \
         claim that storage-api refused a credential that was actually presented"
    );

    scaffold.shutdown().await;
}

// ---------------------------------------------------------------------------
// Variant 4b (b.e49 review round 2) — the token dies MID-LADDER, terminal 401
// is unauthenticated → Transient
// ---------------------------------------------------------------------------

/// Variant 4b: the integration mirror of the sibling unit test
/// `storage_api.rs::tests::force_refresh_tests::
/// token_expiring_mid_ladder_second_401_surfaces_transient`, and the case that
/// separates "what the terminal request put on the wire" from "what the token
/// cache said just before the retry".
///
/// The retry is not one request. It is a whole transient-retry ladder — up to
/// `MAX_RETRY_ATTEMPTS` attempts, each bounded by `REQUEST_TIMEOUT` and
/// separated by backoff — so a token that is live when `force_refresh()`
/// returns can be dead by the attempt that actually draws the terminal 401,
/// and that attempt goes out unauthenticated. Nothing was rejected; the
/// sidecar's replacement heals it in seconds; the verdict must be `Transient`.
///
/// A pre-retry snapshot predicate — `cached_token_expires_on()` read once,
/// right after the refresh — sees the token while it is still live here and
/// returns `Permanent`, which post-b.33e reaches DR-restore clients as a
/// non-retriable `INTERNAL` on a self-healing condition. Every other variant in
/// this file classifies identically under the snapshot rule and under the
/// on-wire rule; this is the one that separates them, at the integration
/// boundary the rest of the file pins.
///
/// ## The fixture, and why it is a wall-clock one
///
/// Neither bearer accessor takes an injectable clock, so mid-ladder expiry has
/// to be produced with real time. It is produced *deterministically* rather
/// than raced, and the fixture gives real margin in BOTH directions. The two
/// numbers — the token's 6 s lifetime and the 10 s delay on attempt 2's `503` —
/// are a matched pair and must move together:
///
/// - **Attempt 3 must run after the token dies.** The 10 s server-side delay is
///   held open 4 s past the token's 6 s life, so by the time attempt 3 is built
///   `cached_bearer_if_unexpired()` is unconditionally `None`. 10 s is also
///   well under the adapter's 30 s `REQUEST_TIMEOUT`, so that response lands as
///   a real 503 on the transient ladder — at or above the timeout it would
///   become a client-side timeout instead, a different `Transient` reached for
///   a different reason, and the scenario would silently stop being covered.
/// - **Attempt 2 must be built before the token dies.** The budget is the
///   token's 6 s minus everything in front of that build: the three
///   `Mock::mount` calls, one localhost round trip for attempt 1, the
///   `force_refresh()` file re-read, and ≤ 300 ms of backoff. All of that is
///   comfortably sub-second on an idle machine, leaving ≈ 5 s of slack — sized
///   at roughly 2× the worst scheduler drift measured against this fixture
///   under 12-way CPU saturation (~2.5 s), which is what a loaded CI runner or
///   a parallel nextest shard reproduces. The earlier 3 s / 5 s pair left only
///   ~3 s here and could be eaten by that drift.
///
/// The second budget is the fragile one, so it is *named* by the assertions
/// instead of left implicit — a machine slow enough to break it fails with the
/// reason rather than as a mystery. Both the request-count assertion and the
/// `received[1]` header assertion carry the hint, because either can be the one
/// that fires: if attempt 2 itself goes out unauthenticated it matches the
/// `no_authorization_header` mock, the ladder terminates at two requests, and
/// the count assertion trips before the header assertion is ever reached.
///
/// Wall-clock cost is ~10 s (the one delayed response) plus ≤ 300 ms of
/// backoff, still far inside nextest's 40 s slow mark.
///
/// Ordered responses come from `up_to_n_times(1)` plus mount order: wiremock
/// scans mounted mocks in order and skips one that has hit its cap, so two
/// mocks with the same matcher fire in the order mounted.
#[tokio::test]
async fn token_expiring_mid_ladder_second_401_surfaces_transient() {
    // 6 s of life: live for the first two attempts, dead by the third. The
    // file is never rewritten — the adapter's `force_refresh()` re-reads this
    // same document, exactly as it does against a sidecar that has not yet
    // landed the replacement. Paired with the 10 s delay below; see the
    // docblock for how the two budgets are sized.
    let scaffold = ScaffoldWithStore::with_initial_expiry(6).await;

    // Attempt 1 (first auth attempt): the still-live token is rejected.
    Mock::given(method("PUT"))
        .and(path_regex(r"sessions%2Fsess-abc\.json$"))
        .and(header("authorization", "Bearer initial-token"))
        .respond_with(ResponseTemplate::new(401))
        .up_to_n_times(1)
        .mount(&scaffold.mock_server)
        .await;
    // Attempt 2 (post-refresh, transient-ladder attempt #1): the token is
    // still live, so it is attached — and the 503 is held open 4 s past the
    // token's expiry, which is what makes attempt 3's unauthenticated send
    // deterministic rather than racy. 10 s < the adapter's 30 s
    // REQUEST_TIMEOUT, so this still lands as a real 503.
    Mock::given(method("PUT"))
        .and(path_regex(r"sessions%2Fsess-abc\.json$"))
        .and(header("authorization", "Bearer initial-token"))
        .respond_with(ResponseTemplate::new(503).set_delay(Duration::from_millis(10_000)))
        .up_to_n_times(1)
        .mount(&scaffold.mock_server)
        .await;
    // Attempt 3 (transient-ladder attempt #2): the token has expired, so
    // `cached_bearer_if_unexpired()` returns `None` and this goes out with no
    // credential at all. THIS is the request the verdict is about.
    Mock::given(method("PUT"))
        .and(path_regex(r"sessions%2Fsess-abc\.json$"))
        .and(no_authorization_header)
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&scaffold.mock_server)
        .await;

    assert!(
        scaffold.provider.cached_bearer_if_unexpired().is_some(),
        "fixture precondition: the token must start out live, or the first attempt would already \
         be the unauthenticated one and the mid-ladder transition would never happen"
    );

    let err = scaffold
        .store
        .put_session_checkpoint(
            SESSION_ID,
            Bytes::from_static(b"payload"),
            CheckpointMeta::default(),
        )
        .await
        .expect_err("the terminal 401 must surface as an error");

    let received = scaffold
        .mock_server
        .received_requests()
        .await
        .expect("mock server records requests");
    assert_eq!(
        received.len(),
        3,
        "expected 3 on-wire PUTs (401 with bearer, 503 with bearer, 401 without), got {}. A \
         count of 2 is the slow-machine signature, not a logic failure: it means attempt 2 \
         itself went out unauthenticated, so it matched the no-authorization mock and ended \
         the ladder one request early — i.e. the 6 s fixture token expired before attempt 2 \
         was even built, which scheduler drift under CPU saturation can do. Widen the \
         `with_initial_expiry` lifetime and the 503 `set_delay` together if so (see the \
         docblock: the delay must stay strictly longer than the lifetime, and under the 30 s \
         REQUEST_TIMEOUT)",
        received.len()
    );
    assert!(
        recorded_authorization_present(&received[1]),
        "fixture precondition: the ladder's first attempt must still carry the live token. If \
         this failed, the 6 s token expired before that attempt ran and the test is no longer \
         exercising mid-ladder expiry — widen the fixture lifetime and the 503 delay together"
    );
    // THE predicate: the terminal request went out with no credential, so the
    // adapter recorded `bearer_attached == false` for it — even though the
    // cache held a live token when the refresh returned.
    assert!(
        !recorded_authorization_present(&received[2]),
        "the terminal attempt must go out unauthenticated — that is the whole scenario"
    );

    assert!(
        err.is_transient(),
        "the token expired mid-ladder and the terminal 401 was drawn by an unauthenticated \
         request, so nothing was rejected and the sidecar's replacement heals this in seconds. \
         Got {err:?} — a Permanent here means the verdict is being read off a pre-retry snapshot \
         of the token cache instead of off the wire"
    );
    let CheckpointError::Transient { source } = &err else {
        panic!("expected CheckpointError::Transient, got {err:?}");
    };
    let source_str = format!("{source}");
    assert!(
        source_str.contains("no usable service token was available for the retry"),
        "expected the second-401 no-usable-credential source string (and NOT some other \
         Transient sharing the variant), got `{source_str}`"
    );

    scaffold.shutdown().await;
}

// ---------------------------------------------------------------------------
// Variant 5 (b.e49) — a live near-expiry token is ATTACHED, one round trip
// ---------------------------------------------------------------------------

/// Variant 5, b.e49's headline adapter assertion at the integration boundary:
/// a token 30 s from expiry — INSIDE the provider's 60 s refresh lead — is
/// attached to the first request, and the call succeeds in exactly ONE round
/// trip.
///
/// Before b.e49 this fixture produced a self-inflicted extra round trip on
/// every consumer, once per token lifetime: a second, withholding accessor
/// suppressed the still-good credential inside the lead, the request went out
/// unauthenticated, storage-api 401'd by construction, and the adapter burned a
/// `force_refresh()` plus a retry to recover. When the sidecar had not yet
/// written the replacement, the retry 401'd too and the whole call died
/// terminal — which is the retry-storm the issue describes. b.mq8 removed that
/// accessor entirely, so the provider now has exactly one and the refresh lead
/// is a re-read cadence with no effect on what a request carries.
///
/// The `.expect(1)` on the single bearer-scoped route IS the assertion: there
/// is no unauthenticated route mounted, so a regression that reintroduced
/// withholding shows up as both a failed expectation and a wiremock 404 turning
/// the call into an error.
#[tokio::test]
async fn near_expiry_token_is_attached_and_succeeds_in_one_round_trip() {
    // 30 s of life left: inside the 60 s lead, and a perfectly good credential.
    let scaffold = ScaffoldWithStore::with_initial_expiry(30).await;

    // The ONLY route.
    Mock::given(method("PUT"))
        .and(path_regex(r"sessions%2Fsess-abc\.json$"))
        .and(header("authorization", "Bearer initial-token"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&scaffold.mock_server)
        .await;

    // Fixture precondition, stated arithmetically on the cached expiry: b.mq8
    // deleted the withholding accessor whose `None` used to report this, and
    // without the check the test silently stops covering the boundary it names
    // if 30 s ever stops being inside the lead.
    let remaining = scaffold
        .provider
        .cached_token_expires_on()
        .expect("the provider must have cached the fixture token")
        - chrono::Utc::now();
    assert!(
        remaining > chrono::Duration::zero() && remaining < chrono::Duration::seconds(60),
        "fixture precondition: the token must be unexpired AND inside the 60 s default refresh \
         lead — exactly the case the adapter no longer routes through; if it drifts outside, \
         this test silently stops covering the boundary it names (remaining: {remaining})"
    );
    assert!(
        scaffold.provider.cached_bearer_if_unexpired().is_some(),
        "fixture precondition: the same token is unexpired, so the accessor the adapter reads \
         MUST serve it"
    );

    scaffold
        .store
        .put_session_checkpoint(
            SESSION_ID,
            Bytes::from_static(b"payload"),
            CheckpointMeta::default(),
        )
        .await
        .expect("a live near-expiry token must be attached and accepted");

    let received = scaffold
        .mock_server
        .received_requests()
        .await
        .expect("mock server records requests");
    assert_eq!(
        received.len(),
        1,
        "a live near-expiry token must cost exactly ONE round trip — no unauthenticated probe, \
         no force_refresh, no retry. Got {} PUTs: {:?}",
        received.len(),
        received
            .iter()
            .map(|r| (
                r.method.to_string(),
                r.headers
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("<absent>")
                    .to_string()
            ))
            .collect::<Vec<_>>()
    );

    scaffold.shutdown().await;
}

// ---------------------------------------------------------------------------
// Composed path (b.e49 + b.33e) — the adapter's new Transient reaches the
// client as UNAVAILABLE on the layer-recovery path
// ---------------------------------------------------------------------------

/// Build a `StatLayerRequest` carrying a stub bearer in the extensions map —
/// same shape `e2e_stat_layer_recovery_test.rs` uses. This is the *inbound*
/// user token, entirely unrelated to the outbound service-identity token the
/// rest of this file is about.
fn stat_request(session_id: &str, layer_name: &str) -> tonic::Request<StatLayerRequest> {
    let mut request = tonic::Request::new(StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

/// Read the live label-cell counter value for `result` on
/// `live_edit_checkpoint_recovery_attempts_total{result, scope="layer",
/// service}`.
///
/// A thin harness-shaped adapter over `common::recovery_metric` rather than a
/// re-implementation of it: this file's call sites hold a harness, not the
/// `SharedMetrics` the shared helper takes, and unwrapping it at every site
/// would be noisier than unwrapping it once here.
fn recovery_metric(harness: &super::helpers::StorageApiCheckpointHarness, result: &str) -> u64 {
    crate::common::recovery_metric(&harness.metrics, result)
}

/// Composed case #1 for b.e49 — the retriable side. The real
/// `StorageApiCheckpointStore`, a real `RecoveryHook`, a real StatLayer RPC,
/// a wiremock that 401s everything, and a service-token cache holding an
/// **expired** token. The client must see `Code::Unavailable`.
///
/// ## What this proves that nothing else does
///
/// Two contracts land in different files and only compose here:
///
/// - b.e49 (this file, unit + integration variants above): a second 401 where
///   the retry could not present a live credential classifies
///   `CheckpointError::Transient`.
/// - b.33e (`recovery_hook_test.rs`, exhaustively, against a stub store):
///   `RecoveryOutcome::Error(Transient)` maps to `Code::Unavailable`, every
///   other kind to `Code::Internal`.
///
/// Neither file can see the seam. `recovery_hook_test.rs` never runs the real
/// adapter — it injects the variant directly — so it would keep passing if
/// the adapter went back to `Permanent`. The variants above never run a
/// handler, so they would keep passing if the recovery path stopped
/// classifying. This test is the single point where "the adapter produces
/// Transient" and "Transient means retry" are checked as one statement, on
/// the exact retry-storm path the issue describes: a DR restore against a
/// sidecar that has not yet written the replacement token.
///
/// Its sibling
/// [`storage_api_second_401_with_live_token_surfaces_internal_to_the_client`]
/// pins the other side of the same predicate on the same wiring. The pair is
/// the point: one fixture knob (the sign of the cached token's remaining
/// lifetime) moves the client-visible status, and nothing else does.
///
/// The kind → code matrix is deliberately NOT re-enumerated here; that lives
/// in `recovery_hook_test.rs`.
///
/// ## Shape
///
/// 1. Stand up the storage-api harness (real adapter over wiremock, real
///    checkpoint loop, real recovery hook, `service_token_provider` wired).
/// 2. CreateSession + CreateLayer, wait for the loop to land a generation —
///    so recovery has something it *could* have restored, and the failure
///    verdict later cannot be confused with "no checkpoint exists". The
///    harness's live 1 h token is still in the cache for this phase, so the
///    setup path is the ordinary authenticated one.
/// 3. Mount a priority-1 catch-all 401 that shadows every default route.
/// 4. Age the credential out: atomically rewrite the harness token file with
///    an `expires_on` 30 s in the *past*, then drive the harness provider's
///    `force_refresh()` directly so the cache holds the dead token before the
///    RPC starts. Driving the refresh rather than waiting on the provider's
///    250 ms poll is what keeps this deterministic — no sleep, no race
///    between the polling task and the handler.
/// 5. Wipe `le:{object_id}:obj:*` so StatLayer's header read misses and the
///    recovery branch fires.
/// 6. StatLayer → the hook's `list_layer_checkpoints` goes out
///    unauthenticated and 401s, `force_refresh()` re-reads the same expired
///    document, the retry also goes out unauthenticated and 401s, the adapter
///    classifies `Transient`, and the handler must surface `UNAVAILABLE`.
///
/// ## Why an expired token rather than an untouched one
///
/// This test previously left the token file alone and relied on "no rotation
/// observed" to reach `Transient`. That predicate was wrong (see the file
/// header): with the harness's live 1 h token, an untouched file now means a
/// live credential was presented on both attempts, which is `Permanent` — and
/// correctly so. Reaching the retriable arm requires reproducing the
/// condition the issue actually describes, which is a token that has expired
/// with its replacement not yet written. The rewrite is that condition,
/// stated as a fixture instead of inferred from file timing.
///
/// ## Why the checkpoint loop is left running
///
/// The harness shares ONE `CancellationToken` between the checkpoint loop and
/// the `ServiceTokenProvider`'s polling task, and `force_refresh()` is served
/// *by* that task — cancelling the loop to quiet the request stream would
/// make `force_refresh()` return `Cancelled`, and the call would then take
/// the refresh-failure arm (also `Transient`, so the test would still pass —
/// for entirely the wrong reason). Leaving the loop running costs only
/// unattributable 401 chatter in `received_requests()`, which is why the
/// count assertion below is a lower bound and the exact-count assertions live
/// in the variants above.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn storage_api_second_401_with_expired_token_surfaces_unavailable_to_the_client() {
    let harness = setup_storage_api_checkpoint_harness(1).await;

    let (session_id, object_id) = create_session_and_layer(
        &harness.tv,
        &harness.svc,
        "b-e49-401-recovery-session",
        "b-e49-401-recovery-layer",
    )
    .await;

    // A generation must exist before the wipe, otherwise a later failure
    // would be indistinguishable from "recovery had nothing to restore".
    let _handle = wait_for_checkpoint(&harness.store, object_id, 10).await;

    let error_pre = recovery_metric(&harness, "error");
    let not_recoverable_pre = recovery_metric(&harness, "not_recoverable");

    // Every subsequent request 401s. Priority 1 beats the harness's default
    // routes (wiremock's default priority is 5; lower wins).
    Mock::given(any())
        .respond_with(ResponseTemplate::new(401))
        .with_priority(1)
        .mount(&harness.mock_server)
        .await;

    let requests_before = harness
        .mock_server
        .received_requests()
        .await
        .expect("wiremock records requests")
        .len();

    // Age the credential out. The harness ships a live 1 h token; rewrite it
    // with an `expires_on` 30 s in the past and load that into the cache
    // synchronously via the provider's own `force_refresh()`. The document
    // still parses, so the refresh SUCCEEDS and the cache ends up holding a
    // dead token — exactly the sidecar-gap state the issue describes, and
    // distinct from the malformed-file arm
    // (`force_refresh_failure_surfaces_transient_with_no_retry`), which is a
    // different `Transient` reached for a different reason.
    atomic_write(
        &harness.token_path,
        &token_json_expiring_in("expired-token", -30),
    );
    harness
        .service_token_provider
        .force_refresh()
        .await
        .expect("re-reading a well-formed but expired token document must still succeed");

    // Fixture precondition, stated through the accessor every send site reads
    // when it decides whether to attach a bearer. Unlike the variants above,
    // the two composed cases cannot assert the `Authorization` header on the
    // terminal request directly: the catch-all 401 wall shares
    // `received_requests()` with the still-running checkpoint loop's chatter,
    // so no index in that log is attributable to the handler's own retry. A
    // permanently-empty cache is the next best thing — it makes an
    // unauthenticated send the only shape any request in this window can have.
    // If this ever starts serving a bearer, the test below is silently
    // exercising the Permanent arm under the wrong name.
    assert_eq!(
        harness.service_token_provider.cached_bearer_if_unexpired(),
        None,
        "fixture precondition: the cached token is expired, so every outbound storage-api \
         attempt in this test must go out unauthenticated"
    );

    wipe_layer_state(&harness.tv, object_id).await;

    let status = handle_stat_layer(
        &harness.svc,
        stat_request(&session_id, "b-e49-401-recovery-layer"),
    )
    .await
    .expect_err("StatLayer must fail: recovery cannot read the checkpoint through a 401 wall");

    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "no credential could be attached to either attempt, so the adapter's verdict is \
         Transient (b.e49) and b.33e maps Transient to UNAVAILABLE (SDD §18.1) — a \
         self-healing sidecar gap must leave the client retrying, not stop it with INTERNAL. \
         Got {status:?}"
    );

    // The recovery hook must have reported `error`, not `not_recoverable`:
    // this is a backend failure, and a `not_recoverable` here would mean the
    // hook swallowed the 401 and concluded no checkpoint existed.
    assert_eq!(
        recovery_metric(&harness, "error") - error_pre,
        1,
        "exactly one recovery attempt must have reported `error`"
    );
    assert_eq!(
        recovery_metric(&harness, "not_recoverable") - not_recoverable_pre,
        0,
        "`not_recoverable` must stay flat — a 401 wall is a backend failure, not an absent \
         checkpoint, and conflating the two is what would hide a DR outage"
    );

    // The arm under test really was the unauthenticated one, checked again
    // AFTER the RPC. Bracketing the call this way is what stands in for the
    // per-request header assertion the variants above make: the cache held no
    // usable token before the handler ran and still holds none after — the
    // adapter's own `force_refresh()` re-read the same expired document — so
    // no request the handler built in between could have attached a bearer,
    // and every one of them recorded `bearer_attached == false`.
    assert_eq!(
        harness.service_token_provider.cached_bearer_if_unexpired(),
        None,
        "fixture precondition: the adapter's force_refresh re-read the same expired document, \
         so the retry went out unauthenticated — that is what makes this the Transient arm \
         rather than the rejected-live-credential (Permanent) one"
    );

    // Lower bound, not an exact count — see the docblock's note on the shared
    // cancellation token. Two is the adapter's own budget for one logical
    // request (initial attempt + one post-refresh retry); anything less would
    // mean the retry never happened.
    let requests_after = harness
        .mock_server
        .received_requests()
        .await
        .expect("wiremock records requests")
        .len();
    assert!(
        requests_after >= requests_before + 2,
        "the recovery read must have made at least 2 on-wire attempts (initial + one \
         post-force_refresh retry); saw {} new requests",
        requests_after - requests_before
    );

    harness.cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), harness.loop_handle).await;
}

/// Composed case #2 for b.e49 — the escalating side, and the contrast that
/// makes case #1 a claim about the *predicate* rather than about 401s in
/// general.
///
/// Identical wiring to
/// [`storage_api_second_401_with_expired_token_surfaces_unavailable_to_the_client`]
/// — same harness, same catch-all 401 wall, same wipe, same StatLayer — with
/// exactly one fixture difference: the harness's live 1 h token is left
/// alone. Both attempts therefore carry `Bearer test-token`, storage-api
/// refuses a credential the provider considers valid, the adapter classifies
/// `Permanent`, and b.33e maps that to `Code::Internal`.
///
/// ## Why this half has to be pinned too
///
/// b.e49's fix is a *split*, and a split is only correct if both sides land.
/// A regression that made every second 401 `Transient` — which is precisely
/// what the pre-review rotation predicate did in the steady state, where
/// `force_refresh()` re-reads the same good document — would leave case #1
/// green while turning an operator-actionable auth-policy fault into an
/// unbounded client retry storm against a backend that will never accept this
/// service principal. Only this test fails on that regression.
///
/// The `INTERNAL` here is not a lesser outcome than case #1's `UNAVAILABLE`;
/// it is the *correct* one. The client should stop, and the redacted
/// `"internal error"` message is the b.33e contract: detail stays in the
/// operator log, never on the wire.
///
/// The one thing this test does NOT re-prove is the kind → code matrix —
/// `recovery_hook_test.rs` owns that exhaustively against a stub store. What
/// it adds is that the REAL adapter produces the kind that maps this way.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn storage_api_second_401_with_live_token_surfaces_internal_to_the_client() {
    let harness = setup_storage_api_checkpoint_harness(1).await;

    let (session_id, object_id) = create_session_and_layer(
        &harness.tv,
        &harness.svc,
        "b-e49-401-live-token-session",
        "b-e49-401-live-token-layer",
    )
    .await;

    // As in case #1: a generation must exist before the wipe so the later
    // failure cannot be confused with "recovery had nothing to restore".
    let _handle = wait_for_checkpoint(&harness.store, object_id, 10).await;

    let error_pre = recovery_metric(&harness, "error");
    let not_recoverable_pre = recovery_metric(&harness, "not_recoverable");

    // Every subsequent request 401s, including the post-force_refresh retry.
    Mock::given(any())
        .respond_with(ResponseTemplate::new(401))
        .with_priority(1)
        .mount(&harness.mock_server)
        .await;

    // The single fixture difference from case #1: the token file is left
    // untouched, so the cache keeps serving the harness's live 1 h token and
    // both attempts carry it. Asserted, not assumed — if the harness ever
    // ships a token that has aged out by this point, this test would silently
    // become a duplicate of case #1 and stop covering the escalation.
    assert!(
        harness
            .service_token_provider
            .cached_bearer_if_unexpired()
            .is_some(),
        "fixture precondition: the harness token must still be live, or this test is no longer \
         exercising the rejected-live-credential arm"
    );

    wipe_layer_state(&harness.tv, object_id).await;

    let status = handle_stat_layer(
        &harness.svc,
        stat_request(&session_id, "b-e49-401-live-token-layer"),
    )
    .await
    .expect_err("StatLayer must fail: recovery cannot read the checkpoint through a 401 wall");

    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "a live credential was presented on both attempts and refused, so the adapter's \
         verdict is Permanent (b.e49) and b.33e maps non-Transient kinds to INTERNAL — the \
         client must stop retrying and an operator must look. Got {status:?}"
    );
    assert_eq!(
        status.message(),
        "internal error",
        "the b.33e redaction contract: the operator-facing detail stays in the log, and the \
         client sees only the generic message"
    );

    // Same shape as case #1: this is a backend failure, so the recovery hook
    // reports `error`. A `not_recoverable` here would mean the hook swallowed
    // the 401 and concluded no checkpoint existed — which is exactly the DR
    // outage-hiding behaviour the counter split exists to make visible.
    assert_eq!(
        recovery_metric(&harness, "error") - error_pre,
        1,
        "exactly one recovery attempt must have reported `error`"
    );
    assert_eq!(
        recovery_metric(&harness, "not_recoverable") - not_recoverable_pre,
        0,
        "`not_recoverable` must stay flat — a 401 wall is a backend failure, not an absent \
         checkpoint"
    );

    // Post-RPC: the cache still holds a live token. Paired with the pre-RPC
    // check above, this brackets the whole call — the credential was available
    // before the handler ran and the adapter's own `force_refresh()` left it in
    // place, so every request the handler built attached it and recorded
    // `bearer_attached == true`. It stands in for the per-request header
    // assertion the variants above make, which the shared 401-wall request log
    // does not permit here (see case #1's note).
    assert!(
        harness
            .service_token_provider
            .cached_bearer_if_unexpired()
            .is_some(),
        "the adapter's force_refresh re-read the same live document, so the retry presented a \
         credential — that is what makes this the Permanent arm"
    );

    harness.cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), harness.loop_handle).await;
}
