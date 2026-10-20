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

//! Integration tests for the service-token-aware health supervisor wiring
//! added by Task `t2.fgj.dp.s7` (Subtask `t3.fgj.dp.s7.s7`).
//!
//! These tests pin Acceptance Criteria #2..#5 of the parent Task end-to-end
//! against a real Valkey + tonic `HealthServer` + `ServiceTokenProvider`
//! triple. Coverage:
//!
//! - AC #3: provider not wired → readiness model collapses to the pre-Task
//!   Valkey + auth baseline; the `live_edit_service_token_available` gauge
//!   is absent from the Prometheus registry.
//! - AC #2: provider wired against an absent token file → readiness flips
//!   `NotServing` and stays there.
//! - AC #4: token file appears → readiness flips back to `Serving` within
//!   one poll interval + slack. Token file deleted → readiness flips back
//!   to `NotServing` within two probe intervals + slack.
//! - AC #5: the gauge is registered ONLY when the provider is constructed,
//!   verified by `Registry::gather()` (absence assertion in the no-provider
//!   case, value 1/0 assertion in the provider-wired case across the
//!   `Some → None` transition).
//! - Transition WARN log carries the `service_token_ok` structured field —
//!   the operator-runbook signal the new field is keyed off of.
//! - b.e49: readiness keys on **credential death**, not on the provider's
//!   refresh lead. A token 30 s from expiry — inside the 60 s
//!   `DEFAULT_REFRESH_LEAD` — keeps the replica SERVING and the gauge at 1;
//!   only an absent or actually-expired token flips it. The supervisor reads
//!   `cached_bearer_if_unexpired()`; before b.e49 it read a second, withholding
//!   accessor (since deleted, b.mq8) that flipped a healthy replica
//!   NOT_SERVING once per token lifetime while it held a live credential.
//! - b.rgs: the supervisor's FIRST tick always publishes and logs a verdict.
//!   An unready-from-boot replica logs the NOT_SERVING WARN with
//!   `first_verdict=true`; a healthy boot logs the raised-to-SERVING line
//!   exactly once and stays silent thereafter (edge-triggered).
//!
//! These tests do not require any `LIVE_EDIT_*` env vars at runtime; they
//! construct `Config` via `crate::common::test_config()` and
//! `HealthSupervisorConfig` directly. Valkey-unreachable runs short-circuit
//! with `return` matching the project-wide pattern.

use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::Utc;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;
use tonic::transport::{Channel, Server};
use tonic_health::pb::{
    health_check_response::ServingStatus as PbServingStatus, health_client::HealthClient,
    HealthCheckRequest,
};
use tonic_health::server::HealthReporter;
use tracing_subscriber::fmt::MakeWriter;

use live_edit::auth::service_token::ServiceTokenProvider;
use live_edit::config::Config;
use live_edit::observability::health::{
    create_health_service, spawn_health_supervisor, HealthSupervisorConfig,
};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;

const READINESS_SERVICE_NAME: &str = "nvidia.omniverse.live.v1alpha.LiveService";

/// File name inside each per-test `TempDir`. Matches the conventional
/// credential-provisioner sidecar mount basename.
const TOKEN_FILE_NAME: &str = "access_token";

/// Poll interval for the `ServiceTokenProvider`'s polling task. Short so
/// the appears / disappears transitions resolve well inside each test's
/// 5 s deadline budget.
const TOKEN_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Supervisor probe interval used by every test below. The supervisor
/// reads `cached_bearer_if_unexpired()` synchronously on every tick (b.e49 —
/// it read a withholding accessor before, deleted in b.mq8), so this also
/// bounds how quickly readiness and the gauge follow the provider's cache
/// state.
const SUPERVISOR_PROBE_INTERVAL: Duration = Duration::from_millis(100);

/// Per-probe deadline. Short enough that a healthy localhost Valkey ping
/// is never racy and a dead-connection ping cannot succeed.
const SUPERVISOR_PROBE_TIMEOUT: Duration = Duration::from_millis(200);

/// Build a JSON token document expiring `expires_in_secs` from now.
///
/// The sign is the whole contract post-b.e49. The supervisor reads
/// `cached_bearer_if_unexpired()`, whose only `None` conditions are "nothing
/// cached" and "actually past `expires_on`" — so:
///
/// - **Positive** → a usable credential, however small the value.
///   `service_token_ok = true`, readiness SERVING, gauge 1. A small positive
///   value (30 s) sits inside the provider's 60 s refresh lead and is exactly
///   the fixture the pre-b.e49 supervisor flipped NOT_SERVING on while
///   holding a live credential.
/// - **Negative** → an actually-dead credential, and the only cached-token
///   shape that still drives `service_token_ok = false`.
///
/// There is no injectable clock on either accessor, so this offset is the
/// only lever these tests have on the expiry dimension.
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

/// Atomically write `contents` to `path` via a sibling tempfile + rename
/// — matches the production credential-provisioner sidecar's write
/// contract so a concurrent reader never observes a torn file. Using
/// sibling-in-same-directory keeps the rename cross-filesystem-safe.
fn atomic_write_token(path: &Path, contents: &str) {
    let parent = path
        .parent()
        .expect("token file path must have a parent directory");
    let mut tmp = tempfile::NamedTempFile::new_in(parent)
        .expect("NamedTempFile::new_in must succeed in test tempdir");
    tmp.write_all(contents.as_bytes())
        .expect("write to tempfile must succeed");
    tmp.persist(path)
        .expect("persist (atomic rename) must succeed");
}

/// Construct a `Config` clone matching the canonical integration-test
/// shape, then narrow Valkey-pool sizing for tests that don't need a
/// large pool. Mirrors the sibling `health_supervisor_test.rs` helper so
/// the two readiness-test files exercise comparable connection setups.
fn test_config() -> Config {
    let mut cfg = crate::common::test_config();
    cfg.valkey_pool_size = 1;
    cfg.subscriber_pool_size = 1;
    cfg.valkey_reconnect_max_delay_ms = 500;
    cfg
}

/// Attempt to connect to Valkey using the test config. Returns `None`
/// when Valkey is unreachable — every test below short-circuits via
/// `return` in that case, matching the existing pattern in
/// `health_supervisor_test.rs`.
async fn maybe_connect_valkey() -> Option<ValkeyClient> {
    let cfg = test_config();
    match tokio::time::timeout(
        Duration::from_secs(5),
        live_edit::storage::test_support::connect_test_client(&cfg),
    )
    .await
    {
        Ok(Ok(c)) => Some(c),
        _ => None,
    }
}

/// Spawn a minimal tonic server that exposes only the supplied
/// `HealthServer` on a random port. Returns the bound addr and a
/// oneshot sender that gracefully shuts the server down. Mirrors the
/// helper in `health_supervisor_test.rs` — duplicated here rather than
/// extracted so this Subtask stays scoped to the new file + the `main.rs`
/// registration line.
async fn spawn_health_only_server<H>(
    health_service: tonic_health::pb::health_server::HealthServer<H>,
) -> (std::net::SocketAddr, oneshot::Sender<()>)
where
    H: tonic_health::pb::health_server::Health + Send + Sync + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind health server");
    let addr = listener.local_addr().expect("local_addr");
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    // Listener-ready signal — the spawned task fires `ready_tx` before
    // entering `serve_with_incoming_shutdown` so the parent waits on an
    // actual readiness fact instead of a blanket sleep.
    let (ready_tx, ready_rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
        let _ = ready_tx.send(());
        Server::builder()
            .add_service(health_service)
            .serve_with_incoming_shutdown(incoming, async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("health server");
    });

    let _ = tokio::time::timeout(Duration::from_secs(5), ready_rx)
        .await
        .expect("health server task did not signal readiness within 5s");

    (addr, shutdown_tx)
}

async fn connect_health(addr: std::net::SocketAddr) -> HealthClient<Channel> {
    let endpoint = format!("http://{addr}");
    let channel = Channel::from_shared(endpoint)
        .expect("endpoint uri")
        .connect()
        .await
        .expect("connect health");
    HealthClient::new(channel)
}

async fn observe_status(client: &mut HealthClient<Channel>) -> PbServingStatus {
    let resp = client
        .check(HealthCheckRequest {
            service: READINESS_SERVICE_NAME.to_string(),
        })
        .await
        .expect("health Check must succeed")
        .into_inner();
    PbServingStatus::try_from(resp.status)
        .unwrap_or_else(|_| panic!("unknown ServingStatus code: {}", resp.status))
}

/// Poll `observe_status` until it returns `target` or `deadline`
/// elapses. Returns `true` on observing the target, `false` on timeout.
async fn wait_for_status(
    client: &mut HealthClient<Channel>,
    target: PbServingStatus,
    deadline: Duration,
) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < deadline {
        if observe_status(client).await == target {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    false
}

/// Publish `SERVING` on the readiness name before the supervisor starts —
/// see [`crate::common::preseed_readiness_serving`] for why the seed is
/// load-bearing since b.rgs. Thin wrapper so call sites below stay free of
/// the readiness-name argument.
async fn preseed_readiness_serving(reporter: &HealthReporter) {
    crate::common::preseed_readiness_serving(reporter, READINESS_SERVICE_NAME).await;
}

/// Build a fresh `Metrics` rooted in its own `prometheus::Registry`. The
/// caller passes `register_token_gauge = true` to mirror `main.rs`'s
/// "register iff a `ServiceTokenProvider` was constructed" gating. The
/// registry is returned so tests can inspect `gather()` output for the
/// AC #5 family-presence assertion.
fn make_metrics(_register_token_gauge: bool) -> Arc<Metrics> {
    // The `live_edit_service_token_available` gauge is now an unconditional OTel
    // instrument that emits no Prometheus family until first `record()`. The
    // old `register_token_gauge` seam is gone, so the flag is inert: family
    // presence is driven entirely by whether the supervisor records the gauge
    // (i.e. by whether a `ServiceTokenProvider` is wired into its config).
    Arc::new(crate::common::metrics_scrape_only("test"))
}

/// Read the `live_edit_service_token_available` gauge value out of the rendered
/// scrape exposition, or `None` when the family is absent (never recorded).
fn token_gauge_opt(metrics: &Arc<Metrics>) -> Option<i64> {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_service_token_available",
        &[],
    )
    .map(|v| v as i64)
}

/// Return `true` iff the rendered scrape exposition contains a family with the
/// supplied name. Used by AC #5 absence / presence assertions — the family is
/// omitted until the supervisor first records the gauge, which is the
/// record-gating contract (the OTel analogue of the old registration gate).
fn registry_has_family(metrics: &Arc<Metrics>, name: &str) -> bool {
    live_edit::observability::metrics::test_util::has_metric(
        &live_edit::observability::metrics::test_util::render(metrics),
        name,
    )
}

/// Read the current gauge value. Panics if the family is absent — tests using
/// this helper have already asserted presence.
fn read_token_gauge(metrics: &Arc<Metrics>) -> i64 {
    token_gauge_opt(metrics)
        .expect("live_edit_service_token_available family must be present on the scrape surface")
}

/// Poll the gauge until it equals `target` or `deadline` elapses. Returns the
/// last observed value (`target` on success), or `i64::MIN` if the family never
/// materialised.
async fn wait_for_gauge(metrics: &Arc<Metrics>, target: i64, deadline: Duration) -> i64 {
    let start = std::time::Instant::now();
    let mut latest = token_gauge_opt(metrics);
    while latest != Some(target) && start.elapsed() < deadline {
        tokio::time::sleep(Duration::from_millis(25)).await;
        latest = token_gauge_opt(metrics);
    }
    latest.unwrap_or(i64::MIN)
}

/// A `MakeWriter` that captures all bytes written to it into a shared
/// buffer. Mirrors the `CapturingWriter` pattern in
/// `src/auth/service_token.rs::tests` and `src/grpc/auth_interceptor.rs::tests`.
#[derive(Clone)]
struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

impl CapturingWriter {
    fn new() -> (Self, Arc<Mutex<Vec<u8>>>) {
        let buf = Arc::new(Mutex::new(Vec::new()));
        (Self(buf.clone()), buf)
    }
}

impl<'a> MakeWriter<'a> for CapturingWriter {
    type Writer = CapturingWriter;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

impl Write for CapturingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// ───────────────────────────────────────────────────────────────────────
// AC #3 — provider not wired: gauge absent, no check, readiness baseline
// ───────────────────────────────────────────────────────────────────────

/// `service_token_provider = None` (the `LIVE_EDIT_SERVICE_TOKEN_FILE`-unset
/// wiring) → readiness model degrades to the pre-Task Valkey + auth
/// baseline and the `live_edit_service_token_available` gauge is NOT
/// registered. Verified directly via `Registry::gather()` absence:
/// `main.rs` gates `register_service_token_available` on the env var
/// being set, so a `Metrics` built without that call produces a registry
/// whose `gather()` output omits the family entirely. AC #3 / AC #5.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supervisor_no_service_token_provider_omits_check() {
    let Some(valkey) = maybe_connect_valkey().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let metrics = make_metrics(/*register_token_gauge=*/ false);

    // Sanity-check the AC #5 absence half BEFORE the supervisor starts:
    // construction alone must leave the family unregistered.
    assert!(
        !registry_has_family(&metrics, "live_edit_service_token_available"),
        "AC #5: live_edit_service_token_available must be ABSENT from the scrape surface \
         before the supervisor records it (the LIVE_EDIT_SERVICE_TOKEN_FILE-unset wiring). \
         Rendered exposition:\n{}",
        live_edit::observability::metrics::test_util::render(&metrics)
    );

    let (reporter, service) = create_health_service(READINESS_SERVICE_NAME).await;
    let cancel = CancellationToken::new();
    let supervisor = spawn_health_supervisor(
        reporter.clone(),
        valkey.clone(),
        Arc::clone(&metrics),
        HealthSupervisorConfig {
            probe_interval: SUPERVISOR_PROBE_INTERVAL,
            probe_timeout: SUPERVISOR_PROBE_TIMEOUT,
            readiness_service_name: READINESS_SERVICE_NAME.to_string(),
            service_token_provider: None,
            counter_seed_ready: None,
            api_serving: None,
        },
        cancel.clone(),
    );

    let (addr, server_shutdown) = spawn_health_only_server(service).await;
    let mut client = connect_health(addr).await;

    // Since b.rgs readiness is registered NOT_SERVING and raised by the
    // supervisor's first tick, so wait for that before sampling — a bare
    // sample would race the first probe rather than test anything.
    assert!(
        wait_for_status(
            &mut client,
            PbServingStatus::Serving,
            Duration::from_secs(3)
        )
        .await,
        "precondition: with provider=None and Valkey healthy the supervisor must raise \
         readiness to SERVING"
    );

    // Across at least 5 probe intervals the supervisor must keep
    // readiness pinned at SERVING — there is no service-token check to
    // trip, no auth probes, and Valkey is healthy.
    for _ in 0..5 {
        tokio::time::sleep(SUPERVISOR_PROBE_INTERVAL + Duration::from_millis(25)).await;
        assert_eq!(
            observe_status(&mut client).await,
            PbServingStatus::Serving,
            "AC #3: supervisor with provider=None must keep SERVING (Valkey healthy, \
             no auth probes, no service-token check)"
        );
    }

    // AC #5 (post-supervisor-runs): even after multiple supervisor
    // ticks, the gauge family must still be absent. The supervisor's
    // `set_service_token_available` is a no-op when the gauge was not
    // registered — confirms the registration-gating contract is
    // load-bearing for the `/metrics` family list, not just for what
    // value gets written.
    assert!(
        !registry_has_family(&metrics, "live_edit_service_token_available"),
        "AC #5: gauge family must remain absent even after supervisor ticks when the \
         provider is None — set_service_token_available is a no-op when unregistered, \
         so the family must NOT be lazily created by the supervisor's update path"
    );

    cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
    let _ = server_shutdown.send(());
    valkey.quit().await;
}

// ───────────────────────────────────────────────────────────────────────
// AC #2 — provider with absent token file: stays NOT_SERVING
// ───────────────────────────────────────────────────────────────────────

/// Provider wired against a path that does NOT yet exist → readiness
/// flips `NotServing` on the first probe (the provider's cache is empty,
/// so `cached_bearer_if_unexpired()` returns `None`, so
/// `service_token_ok = false`,
/// so `serving = false`) and stays there across ≥ 3 probe intervals.
/// Pins AC #2: an absent token file is a hard NOT_SERVING signal.
///
/// Readiness is pre-seeded `SERVING` (see [`preseed_readiness_serving`]) so
/// both halves stay load-bearing after b.rgs: the first assertion observes a
/// real `SERVING → NOT_SERVING` transition driven by the supervisor rather
/// than the `NOT_SERVING` state `create_health_service` now registers, and
/// the hold loop then pins that the verdict does not drift back up.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supervisor_service_token_absent_file_keeps_not_serving() {
    let Some(valkey) = maybe_connect_valkey().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let dir = TempDir::new().expect("TempDir::new must succeed");
    let token_path = dir.path().join(TOKEN_FILE_NAME);
    // File deliberately NOT created.

    let metrics = make_metrics(/*register_token_gauge=*/ true);

    let provider_cancel = CancellationToken::new();
    let provider = Arc::new(ServiceTokenProvider::new(
        token_path.clone(),
        TOKEN_POLL_INTERVAL,
        Arc::clone(&metrics),
        provider_cancel.clone(),
    ));

    let (reporter, service) = create_health_service(READINESS_SERVICE_NAME).await;
    preseed_readiness_serving(&reporter).await;
    let supervisor_cancel = CancellationToken::new();
    let supervisor = spawn_health_supervisor(
        reporter.clone(),
        valkey.clone(),
        Arc::clone(&metrics),
        HealthSupervisorConfig {
            probe_interval: SUPERVISOR_PROBE_INTERVAL,
            probe_timeout: SUPERVISOR_PROBE_TIMEOUT,
            readiness_service_name: READINESS_SERVICE_NAME.to_string(),
            service_token_provider: Some(Arc::clone(&provider)),
            counter_seed_ready: None,
            api_serving: None,
        },
        supervisor_cancel.clone(),
    );

    let (addr, server_shutdown) = spawn_health_only_server(service).await;
    let mut client = connect_health(addr).await;

    // Must pull readiness down from the seeded SERVING on the very first
    // probe — the provider's cache has nothing in it, so
    // `service_token_ok = false`.
    assert!(
        wait_for_status(
            &mut client,
            PbServingStatus::NotServing,
            Duration::from_secs(3),
        )
        .await,
        "AC #2: readiness must flip SERVING → NOT_SERVING when the provider is wired but \
         the token file is absent (cached_bearer_if_unexpired() returns None on a
         cold-start cache)"
    );

    // And must stay there across multiple probe intervals — the
    // polling task swallows NotFound silently (SR-6.2), so cache stays
    // empty and `service_token_ok` stays false.
    for _ in 0..3 {
        tokio::time::sleep(SUPERVISOR_PROBE_INTERVAL + Duration::from_millis(25)).await;
        assert_eq!(
            observe_status(&mut client).await,
            PbServingStatus::NotServing,
            "AC #2: readiness must REMAIN NOT_SERVING while the token file stays absent"
        );
    }

    // Gauge must read 0 throughout — the supervisor wrote 0 on every
    // tick (every `cached_bearer_if_unexpired()` returned None).
    let gauge = read_token_gauge(&metrics);
    assert_eq!(
        gauge, 0,
        "AC #5: live_edit_service_token_available must be 0 while the token file is absent"
    );

    supervisor_cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
    provider_cancel.cancel();
    let _ = server_shutdown.send(());
    valkey.quit().await;
}

// ───────────────────────────────────────────────────────────────────────
// AC #4 — token file appears: flips Serving within poll + slack
// ───────────────────────────────────────────────────────────────────────

/// Provider wired against an initially-absent path → readiness starts
/// `NotServing`. Once the sidecar atomically writes a valid token file,
/// the provider's polling task picks it up within ~one poll interval,
/// the supervisor's next probe observes
/// `cached_bearer_if_unexpired().is_some()`,
/// and readiness flips to `Serving`. Pins AC #4 (file-appears path) and
/// the AC #5 gauge `0 → 1` transition.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supervisor_service_token_appears_flips_to_serving() {
    let Some(valkey) = maybe_connect_valkey().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let dir = TempDir::new().expect("TempDir::new must succeed");
    let token_path = dir.path().join(TOKEN_FILE_NAME);

    let metrics = make_metrics(/*register_token_gauge=*/ true);

    let provider_cancel = CancellationToken::new();
    let provider = Arc::new(ServiceTokenProvider::new(
        token_path.clone(),
        TOKEN_POLL_INTERVAL,
        Arc::clone(&metrics),
        provider_cancel.clone(),
    ));

    let (reporter, service) = create_health_service(READINESS_SERVICE_NAME).await;
    let supervisor_cancel = CancellationToken::new();
    let supervisor = spawn_health_supervisor(
        reporter.clone(),
        valkey.clone(),
        Arc::clone(&metrics),
        HealthSupervisorConfig {
            probe_interval: SUPERVISOR_PROBE_INTERVAL,
            probe_timeout: SUPERVISOR_PROBE_TIMEOUT,
            readiness_service_name: READINESS_SERVICE_NAME.to_string(),
            service_token_provider: Some(Arc::clone(&provider)),
            counter_seed_ready: None,
            api_serving: None,
        },
        supervisor_cancel.clone(),
    );

    let (addr, server_shutdown) = spawn_health_only_server(service).await;
    let mut client = connect_health(addr).await;

    // Confirm the starting state: NotServing (file absent → cache empty).
    assert!(
        wait_for_status(
            &mut client,
            PbServingStatus::NotServing,
            Duration::from_secs(3),
        )
        .await,
        "precondition: readiness must start NOT_SERVING with token file absent"
    );

    // Sidecar appears — atomic rename matches the production write
    // contract. Concurrent reads on the provider's polling task never
    // observe a torn file.
    atomic_write_token(&token_path, &fresh_token_json("token-arrival"));

    // Within ~one token poll interval (~50 ms) the provider's cache
    // flips Some; within the next supervisor probe interval (~100 ms)
    // the supervisor observes that and flips readiness. 3 s deadline is
    // ample slack for CI scheduling jitter.
    assert!(
        wait_for_status(
            &mut client,
            PbServingStatus::Serving,
            Duration::from_secs(3),
        )
        .await,
        "AC #4: readiness must flip SERVING within ~one poll interval + supervisor probe \
         interval after the token file appears"
    );

    // AC #5: gauge transitioned 0 → 1.
    let gauge = wait_for_gauge(&metrics, 1, Duration::from_secs(2)).await;
    assert_eq!(
        gauge, 1,
        "AC #5: live_edit_service_token_available must read 1 after the token file appears"
    );

    supervisor_cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
    provider_cancel.cancel();
    let _ = server_shutdown.send(());
    valkey.quit().await;
}

// ───────────────────────────────────────────────────────────────────────
// AC #4 — token file deleted: flips NotServing within probe + slack
// ───────────────────────────────────────────────────────────────────────

/// Provider starts with a valid future-expiry token → readiness boots
/// `Serving`. The cached token becomes unusable → readiness flips
/// `NotServing` within ≤ one supervisor probe interval + slack.
///
/// AC #4 wording is "deleting the file past the supervisor's
/// `probe_interval` flips readiness to NotServing". A literal `unlink`
/// does NOT cleanly exercise that path here, because the polling task
/// silently tolerates NotFound per SR-6.2 (cache untouched) — the cached
/// far-future-expiry token would survive the unlink and the supervisor
/// would keep seeing a usable credential. The operator-equivalent failure
/// mode that IS deterministically reachable inside a unit-test budget is to
/// atomically replace the file with an **already-expired** token: the
/// polling task picks the new entry up, `cached_bearer_if_unexpired()`
/// returns `None` on the next read, the supervisor observes
/// `service_token_ok = false`, and readiness flips. Same operator-visible
/// "no usable token" signal.
///
/// Pre-b.e49 this test used a token expiring in 10 s — *inside* the
/// provider's 60 s refresh lead but very much alive. That worked only
/// because the supervisor then read a withholding accessor, and
/// it encoded the defect b.e49 removed: a healthy sidecar on a short refresh
/// lead flipped the replica NOT_SERVING once per token lifetime while it held
/// a perfectly good credential. A live token must never reach this assertion
/// again, so the fixture is now a genuinely dead one (`-30 s`) and the test
/// pins credential *death*, which is the only thing readiness was ever
/// supposed to key on. The live-but-near-expiry direction is pinned by
/// [`supervisor_service_token_inside_refresh_lead_stays_serving`].
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supervisor_service_token_disappears_flips_to_not_serving() {
    let Some(valkey) = maybe_connect_valkey().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let dir = TempDir::new().expect("TempDir::new must succeed");
    let token_path = dir.path().join(TOKEN_FILE_NAME);

    // Start with a valid, far-future-expires token so readiness boots
    // SERVING.
    std::fs::write(&token_path, fresh_token_json("token-fresh"))
        .expect("initial token file write must succeed");

    let metrics = make_metrics(/*register_token_gauge=*/ true);

    let provider_cancel = CancellationToken::new();
    let provider = Arc::new(ServiceTokenProvider::new(
        token_path.clone(),
        TOKEN_POLL_INTERVAL,
        Arc::clone(&metrics),
        provider_cancel.clone(),
    ));

    let (reporter, service) = create_health_service(READINESS_SERVICE_NAME).await;
    let supervisor_cancel = CancellationToken::new();
    let supervisor = spawn_health_supervisor(
        reporter.clone(),
        valkey.clone(),
        Arc::clone(&metrics),
        HealthSupervisorConfig {
            probe_interval: SUPERVISOR_PROBE_INTERVAL,
            probe_timeout: SUPERVISOR_PROBE_TIMEOUT,
            readiness_service_name: READINESS_SERVICE_NAME.to_string(),
            service_token_provider: Some(Arc::clone(&provider)),
            counter_seed_ready: None,
            api_serving: None,
        },
        supervisor_cancel.clone(),
    );

    let (addr, server_shutdown) = spawn_health_only_server(service).await;
    let mut client = connect_health(addr).await;

    // Baseline: readiness comes up SERVING once the polling task has
    // read the fresh token into its cache.
    assert!(
        wait_for_status(
            &mut client,
            PbServingStatus::Serving,
            Duration::from_secs(3),
        )
        .await,
        "precondition: readiness must come up SERVING with a fresh token file present"
    );
    let gauge_before = wait_for_gauge(&metrics, 1, Duration::from_secs(2)).await;
    assert_eq!(
        gauge_before, 1,
        "AC #5: gauge must read 1 before the disappearance event (precondition)"
    );

    // The "file disappears" failure mode: per SR-6.2 the polling task
    // silently tolerates a literal NotFound (cache untouched), so the
    // contract-equivalent path is to atomically replace the file with an
    // ALREADY-EXPIRED token. The polling task picks the new entry up (the
    // parser caches expired tokens too); on the next
    // `cached_bearer_if_unexpired()` call the expiry check fires and returns
    // None — same operator-visible "no usable token" state as a literal
    // unlink. `expires_on = now - 30 s` is unambiguously dead, with no
    // dependence on how long the rest of this test takes to run.
    atomic_write_token(&token_path, &token_json_expiring_in("token-expired", -30));

    // The supervisor must flip NOT_SERVING within ~one poll interval
    // (polling task re-reads) + one supervisor probe interval (supervisor
    // observes `cached_bearer_if_unexpired() == None`). 3 s budget is well
    // past two intervals + slack.
    assert!(
        wait_for_status(
            &mut client,
            PbServingStatus::NotServing,
            Duration::from_secs(3),
        )
        .await,
        "AC #4: readiness must flip NOT_SERVING within ~probe_interval + slack once the \
         cached token is no longer usable (here: replaced with an already-expired token, \
         the operator-facing equivalent of `unlink` per SR-6.2's silent-NotFound contract)"
    );

    // AC #5: gauge transitioned 1 → 0.
    let gauge_after = wait_for_gauge(&metrics, 0, Duration::from_secs(2)).await;
    assert_eq!(
        gauge_after, 0,
        "AC #5: live_edit_service_token_available must transition to 0 once the cached token \
         has actually expired — which post-b.e49 is the gauge's whole contract (\"1 when a \
         non-expired token is cached\"), not \"1 until the refresh lead opens\""
    );

    supervisor_cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
    provider_cancel.cancel();
    let _ = server_shutdown.send(());
    valkey.quit().await;
}

// ───────────────────────────────────────────────────────────────────────
// b.e49 — a live token inside the refresh lead keeps the replica SERVING
// ───────────────────────────────────────────────────────────────────────

/// b.e49's headline readiness assertion at the integration boundary: a token
/// 30 s from expiry — INSIDE the provider's 60 s `DEFAULT_REFRESH_LEAD` but
/// entirely alive — keeps readiness SERVING and the gauge at 1, and keeps it
/// there across several probe intervals.
///
/// This is the exact inverse of the fixture
/// [`supervisor_service_token_disappears_flips_to_not_serving`] used to
/// employ, and it fails against the pre-b.e49 supervisor. The defect it pins
/// out is a *correlated availability* one, not a cosmetic one: a sidecar
/// whose refresh lead is under ~75 s (60 s lead + one poll) made every
/// replica in the fleet withdraw itself from load-balancer rotation once per
/// token lifetime — simultaneously, since a rolling deploy leaves their token
/// clocks in phase — while every one of them held a perfectly good
/// credential.
///
/// Early warning is not readiness's job. That is what
/// `live_edit_service_token_expires_in_seconds` and the b.is1 rotation-
/// headroom gauge are for; readiness is a hard signal and must key on
/// credential death only.
///
/// The in-crate sibling `src/observability/health.rs::tests::
/// service_token_inside_refresh_lead_but_unexpired_stays_serving` asserts the
/// same verdict on the same 30 s fixture, and reads it the same way — it too
/// spawns a real tonic health service and `Check`s it over gRPC rather than
/// calling the probe directly. The read path is therefore NOT what separates
/// the two, and this file does not claim it is.
///
/// The genuine differentiator is the **compile boundary**, which is the reason
/// this file exists at all. As an external integration crate it can name only
/// `live_edit`'s public surface — `ServiceTokenProvider` with its
/// `cached_bearer_if_unexpired` accessor, `create_health_service`,
/// `spawn_health_supervisor`, and `HealthSupervisorConfig` including its
/// `service_token_provider` field — while the in-crate sibling additionally
/// reaches `pub(crate)` items and `#[cfg(test)]` helpers that no caller outside
/// the crate has. A change that narrowed any part of this wiring below `pub`
/// would leave the sibling green and break only this file, and the callers that
/// break with it are the real ones: `src/main.rs` wires the supervisor through
/// exactly these public names.
///
/// The gauge assertion rides along because the gauge's own documented
/// contract is "1 when a non-expired token is cached", which only the
/// post-b.e49 accessor actually satisfies.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supervisor_service_token_inside_refresh_lead_stays_serving() {
    let Some(valkey) = maybe_connect_valkey().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let dir = TempDir::new().expect("TempDir::new must succeed");
    let token_path = dir.path().join(TOKEN_FILE_NAME);
    // 30 s of life left: inside the 60 s lead, and comfortably longer than
    // this test runs (the hold loop below spans ~0.6 s), so nothing here
    // races real expiry.
    std::fs::write(&token_path, token_json_expiring_in("token-near-expiry", 30))
        .expect("near-expiry token file write must succeed");

    let metrics = make_metrics(/*register_token_gauge=*/ true);

    let provider_cancel = CancellationToken::new();
    let provider = Arc::new(ServiceTokenProvider::new(
        token_path.clone(),
        TOKEN_POLL_INTERVAL,
        Arc::clone(&metrics),
        provider_cancel.clone(),
    ));

    // Fixture precondition, both halves. First: the accessor the supervisor
    // reads serves the token.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while provider.cached_bearer_if_unexpired().is_none() {
        assert!(
            std::time::Instant::now() < deadline,
            "precondition: the provider's polling task must cache the 30 s token within 5 s"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // Second: the fixture really is in the band this test names. Asserted
    // arithmetically on the cached expiry — b.mq8 deleted the withholding
    // accessor whose `None` used to report it — so that if 30 s ever stops
    // being inside the lead this fails loudly rather than the test silently
    // ceasing to cover the boundary it names.
    let remaining = provider
        .cached_token_expires_on()
        .expect("the provider must have cached the fixture token")
        - Utc::now();
    assert!(
        remaining > chrono::Duration::zero() && remaining < chrono::Duration::seconds(60),
        "fixture precondition: the token must be unexpired AND inside the 60 s default refresh \
         lead — precisely the state readiness must no longer punish; if it drifts outside, this \
         test silently stops covering the boundary it names (remaining: {remaining})"
    );

    let (reporter, service) = create_health_service(READINESS_SERVICE_NAME).await;
    let supervisor_cancel = CancellationToken::new();
    let supervisor = spawn_health_supervisor(
        reporter.clone(),
        valkey.clone(),
        Arc::clone(&metrics),
        HealthSupervisorConfig {
            probe_interval: SUPERVISOR_PROBE_INTERVAL,
            probe_timeout: SUPERVISOR_PROBE_TIMEOUT,
            readiness_service_name: READINESS_SERVICE_NAME.to_string(),
            service_token_provider: Some(Arc::clone(&provider)),
            counter_seed_ready: None,
            api_serving: None,
        },
        supervisor_cancel.clone(),
    );

    let (addr, server_shutdown) = spawn_health_only_server(service).await;
    let mut client = connect_health(addr).await;

    assert!(
        wait_for_status(
            &mut client,
            PbServingStatus::Serving,
            Duration::from_secs(3),
        )
        .await,
        "b.e49: a token 30 s from expiry is a LIVE credential — readiness must be raised to \
         SERVING. A NOT_SERVING here is the pre-fix behaviour: the supervisor treating the \
         provider's proactive re-read cadence as a validity floor"
    );

    // And must STAY there. The one-shot assertion above could be satisfied by
    // a supervisor that raised readiness once and then flipped on its next
    // tick; the pre-fix defect was precisely a steady-state flip, so the hold
    // is the load-bearing half.
    for _ in 0..5 {
        tokio::time::sleep(SUPERVISOR_PROBE_INTERVAL + Duration::from_millis(25)).await;
        assert_eq!(
            observe_status(&mut client).await,
            PbServingStatus::Serving,
            "b.e49: readiness must REMAIN SERVING for the whole time the cached token is \
             inside the refresh lead but unexpired"
        );
    }

    let gauge = wait_for_gauge(&metrics, 1, Duration::from_secs(2)).await;
    assert_eq!(
        gauge, 1,
        "live_edit_service_token_available must read 1 for a cached, non-expired token — \
         that is the gauge's documented contract, and a 0 here would page an operator over \
         a healthy sidecar"
    );

    supervisor_cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
    provider_cancel.cancel();
    let _ = server_shutdown.send(());
    valkey.quit().await;
}

// ───────────────────────────────────────────────────────────────────────
// AC #5 — gauge registration gating, end-to-end across two registries
// ───────────────────────────────────────────────────────────────────────

/// Twin-registry test of the AC #5 "registered iff provider present"
/// contract. We construct two `Metrics` + supervisor pairs in parallel:
///
/// 1. provider wired, gauge registered → after ≥ 1 probe tick the
///    registry's `gather()` output contains a metric family named
///    `live_edit_service_token_available` with one unlabeled gauge
///    series whose value is 1.
/// 2. provider not wired, gauge NOT registered → after the same number
///    of probe ticks against the same Valkey, the registry's `gather()`
///    output does NOT contain a family with that name.
///
/// Then exercise the in-supervisor `1 → 0` transition: rewrite the token
/// file with an **already-expired** token (same operator-equivalent of
/// `unlink` used in the file-disappears test), wait for the supervisor to
/// flip the gauge to 0, and assert.
///
/// Pre-b.e49 that rewrite used a token expiring in 10 s — alive, but inside
/// the provider's 60 s refresh lead, which the withholding accessor of the day
/// treated as unusable. The supervisor now reads
/// `cached_bearer_if_unexpired()`, so a live token no longer drives the gauge
/// to 0 at all; only real credential death does, and the fixture says so.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supervisor_service_token_metric_registered_only_when_provider_present() {
    let Some(valkey) = maybe_connect_valkey().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    // ── Branch 1: provider wired ────────────────────────────────────
    let dir = TempDir::new().expect("TempDir::new must succeed");
    let token_path = dir.path().join(TOKEN_FILE_NAME);
    std::fs::write(&token_path, fresh_token_json("token-wired"))
        .expect("initial token file write must succeed");

    let metrics_wired = make_metrics(/*register_token_gauge=*/ true);
    let provider_cancel = CancellationToken::new();
    let provider = Arc::new(ServiceTokenProvider::new(
        token_path.clone(),
        TOKEN_POLL_INTERVAL,
        Arc::clone(&metrics_wired),
        provider_cancel.clone(),
    ));

    let (reporter_wired, service_wired) = create_health_service(READINESS_SERVICE_NAME).await;
    let supervisor_cancel_wired = CancellationToken::new();
    let supervisor_wired = spawn_health_supervisor(
        reporter_wired.clone(),
        valkey.clone(),
        Arc::clone(&metrics_wired),
        HealthSupervisorConfig {
            probe_interval: SUPERVISOR_PROBE_INTERVAL,
            probe_timeout: SUPERVISOR_PROBE_TIMEOUT,
            readiness_service_name: READINESS_SERVICE_NAME.to_string(),
            service_token_provider: Some(Arc::clone(&provider)),
            counter_seed_ready: None,
            api_serving: None,
        },
        supervisor_cancel_wired.clone(),
    );
    let (addr_wired, server_shutdown_wired) = spawn_health_only_server(service_wired).await;
    let mut client_wired = connect_health(addr_wired).await;

    // ── Branch 2: provider NOT wired ────────────────────────────────
    let metrics_unwired = make_metrics(/*register_token_gauge=*/ false);
    let (reporter_unwired, service_unwired) = create_health_service(READINESS_SERVICE_NAME).await;
    let supervisor_cancel_unwired = CancellationToken::new();
    let supervisor_unwired = spawn_health_supervisor(
        reporter_unwired.clone(),
        valkey.clone(),
        Arc::clone(&metrics_unwired),
        HealthSupervisorConfig {
            probe_interval: SUPERVISOR_PROBE_INTERVAL,
            probe_timeout: SUPERVISOR_PROBE_TIMEOUT,
            readiness_service_name: READINESS_SERVICE_NAME.to_string(),
            service_token_provider: None,
            counter_seed_ready: None,
            api_serving: None,
        },
        supervisor_cancel_unwired.clone(),
    );
    let (addr_unwired, server_shutdown_unwired) = spawn_health_only_server(service_unwired).await;
    let mut client_unwired = connect_health(addr_unwired).await;

    // Wait for the wired branch to reach SERVING — this also guarantees
    // the supervisor has run at least one probe tick (and therefore one
    // gauge write) before we sample the family.
    assert!(
        wait_for_status(
            &mut client_wired,
            PbServingStatus::Serving,
            Duration::from_secs(3),
        )
        .await,
        "wired-branch precondition: readiness must come up SERVING"
    );

    // And the unwired branch must also be SERVING (Valkey healthy, no
    // service-token check).
    assert!(
        wait_for_status(
            &mut client_unwired,
            PbServingStatus::Serving,
            Duration::from_secs(3),
        )
        .await,
        "unwired-branch precondition: readiness must be SERVING (provider=None)"
    );

    // Run a few more probe intervals to give the unwired supervisor
    // ample opportunity to (incorrectly) lazily register the gauge if
    // a regression introduced that bug.
    tokio::time::sleep(SUPERVISOR_PROBE_INTERVAL * 3 + Duration::from_millis(50)).await;

    // Branch 1 — present, gauge value 1.
    assert!(
        registry_has_family(&metrics_wired, "live_edit_service_token_available"),
        "AC #5: wired registry must contain the live_edit_service_token_available family"
    );
    let gauge_wired = read_token_gauge(&metrics_wired);
    assert_eq!(
        gauge_wired, 1,
        "AC #5: wired registry gauge must read 1 (provider returns Some(token))"
    );

    // Branch 2 — absent.
    assert!(
        !registry_has_family(&metrics_unwired, "live_edit_service_token_available"),
        "AC #5: unwired metrics must NOT contain the live_edit_service_token_available \
         family — the supervisor never records the gauge when the provider is None, so \
         the OTel instrument emits no family. Rendered exposition:\n{}",
        live_edit::observability::metrics::test_util::render(&metrics_unwired)
    );

    // ── 1 → 0 transition on the wired branch ────────────────────────
    atomic_write_token(&token_path, &token_json_expiring_in("token-expired", -30));

    let gauge_after = wait_for_gauge(&metrics_wired, 0, Duration::from_secs(3)).await;
    assert_eq!(
        gauge_after, 0,
        "AC #5: gauge must transition 1 -> 0 once the cached token has ACTUALLY expired \
         (operator-facing equivalent of an unlink per SR-6.2). A token merely inside the \
         provider's 60 s refresh lead must NOT move this gauge — see \
         supervisor_service_token_inside_refresh_lead_stays_serving"
    );

    supervisor_cancel_wired.cancel();
    supervisor_cancel_unwired.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(2), supervisor_wired).await;
    let _ = tokio::time::timeout(Duration::from_secs(2), supervisor_unwired).await;
    provider_cancel.cancel();
    let _ = server_shutdown_wired.send(());
    let _ = server_shutdown_unwired.send(());
    valkey.quit().await;
}

// ───────────────────────────────────────────────────────────────────────
// Transition WARN log carries `service_token_ok` structured field
// ───────────────────────────────────────────────────────────────────────

/// The NOT_SERVING verdict log line in `health.rs` is a `tracing::warn!`
/// with structured fields `valkey_ok`, `service_token_ok`, and
/// `counter_seed_ok` (the auth-backend `permissions_ok` / `userinfo_ok`
/// fields were removed in b.9f4 along with auth-backend readiness probing).
/// Operator runbooks key off these field names — the test captures the
/// log output with an in-process `tracing_subscriber::fmt` writer and
/// asserts the bytes contain `service_token_ok=`.
///
/// Since b.rgs readiness is registered NOT_SERVING and the WARN fires on the
/// supervisor's FIRST tick (which always publishes a verdict) rather than on
/// a SERVING → NOT_SERVING edge. The assertions are unchanged — the point of
/// the test is that an unready replica logs which dependency is holding it
/// back, and that is now true from the very first tick instead of only after
/// a replica had once been ready.
///
/// The fmt layer's default format renders boolean fields as
/// `service_token_ok=false`, so a substring search is sufficient. We
/// trigger the transition by wiring the supervisor to a provider whose
/// file is absent (cache empty → `cached_bearer_if_unexpired() == None` → bool
/// `false` → readiness flips), then assert the WARN line was captured
/// and contains the field.
///
/// **Why `flavor = "current_thread"` (default)**: `tracing::subscriber::set_default`
/// installs the dispatcher in a thread-local, which only reaches
/// `tokio::spawn`-ed tasks running on the same OS thread as the test
/// body. The multi-thread runtime would put the supervisor task on a
/// worker thread, the dispatcher would not be visible there, and the
/// `tracing::warn!` call in `health.rs` would never reach the capture
/// buffer. The single-thread runtime cooperatively multiplexes every
/// spawned task on the test thread, so the thread-local dispatcher
/// reaches the supervisor's warn call. Same pattern as the
/// `malformed_json_logs_warn_*` test in `src/auth/service_token.rs`.
#[tokio::test]
async fn supervisor_transition_log_names_service_token_in_dependency_set() {
    let Some(valkey) = maybe_connect_valkey().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let dir = TempDir::new().expect("TempDir::new must succeed");
    let token_path = dir.path().join(TOKEN_FILE_NAME);
    // File deliberately absent → cache stays empty → first supervisor
    // probe sees `service_token_ok = false` → the first-tick verdict is
    // NOT_SERVING → WARN line emitted.

    let metrics = make_metrics(/*register_token_gauge=*/ true);

    // Install a capturing tracing subscriber as the default for this
    // thread BEFORE spawning the supervisor and provider. Under the
    // current_thread runtime (chosen above for this test), every
    // `tokio::spawn`-ed task runs on the same OS thread as the test
    // body, so `set_default`'s thread-local dispatcher reaches the
    // supervisor task's `tracing::warn!` call in
    // `src/observability/health.rs`. Mirrors the dispatcher pattern
    // proven by `src/auth/service_token.rs::tests::malformed_json_logs_warn_*`.
    let (writer, captured) = CapturingWriter::new();
    // Filter at WARN — the transition log is `tracing::warn!` and we
    // don't need TRACE noise from tonic / hyper / h2 cluttering the
    // captured buffer. The transition WARN's `service_token_ok` field
    // is the only signal under test.
    let subscriber = tracing_subscriber::fmt()
        .with_writer(writer)
        .with_max_level(tracing::Level::WARN)
        .with_ansi(false)
        .finish();
    // Use `set_default` (returns a guard) — under the current_thread
    // runtime above, `tokio::spawn` captures the current dispatcher
    // at spawn time and re-enters it inside the task body, so the
    // supervisor's WARN reaches our capturing writer. Verified by the
    // sibling pattern in
    // `src/auth/service_token.rs::tests::malformed_json_logs_warn_*`.
    let _dispatch_guard = tracing::subscriber::set_default(subscriber);

    let provider_cancel = CancellationToken::new();
    let provider = Arc::new(ServiceTokenProvider::new(
        token_path.clone(),
        TOKEN_POLL_INTERVAL,
        Arc::clone(&metrics),
        provider_cancel.clone(),
    ));

    let (reporter, service) = create_health_service(READINESS_SERVICE_NAME).await;
    let supervisor_cancel = CancellationToken::new();
    let supervisor = spawn_health_supervisor(
        reporter.clone(),
        valkey.clone(),
        Arc::clone(&metrics),
        HealthSupervisorConfig {
            probe_interval: SUPERVISOR_PROBE_INTERVAL,
            probe_timeout: SUPERVISOR_PROBE_TIMEOUT,
            readiness_service_name: READINESS_SERVICE_NAME.to_string(),
            service_token_provider: Some(Arc::clone(&provider)),
            counter_seed_ready: None,
            api_serving: None,
        },
        supervisor_cancel.clone(),
    );

    let (addr, server_shutdown) = spawn_health_only_server(service).await;
    let mut client = connect_health(addr).await;

    // Wait for the NOT_SERVING verdict. The WARN line fires with it.
    assert!(
        wait_for_status(
            &mut client,
            PbServingStatus::NotServing,
            Duration::from_secs(3),
        )
        .await,
        "supervisor must flip NOT_SERVING (absent file → cache empty → \
         service_token_ok = false) so the transition WARN fires"
    );

    // Give the subscriber a moment to flush after the transition.
    tokio::time::sleep(Duration::from_millis(100)).await;

    supervisor_cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
    provider_cancel.cancel();
    let _ = server_shutdown.send(());
    valkey.quit().await;

    // Drop the dispatcher guard so it stops capturing before we read.
    drop(_dispatch_guard);

    let buf = captured.lock().unwrap();
    let output = String::from_utf8_lossy(&buf);

    // The transition WARN line MUST be present and MUST mention the
    // new `service_token_ok` field — operator runbooks pin this name.
    // The fmt subscriber's default format renders boolean fields as
    // `service_token_ok=false`, so a substring search is sufficient.
    assert!(
        output.contains("WARN"),
        "expected a WARN event in captured output (the NOT_SERVING verdict); got: {output}"
    );
    assert!(
        output.contains("readiness flipped to NOT_SERVING"),
        "expected the transition log message text in captured output; got: {output}"
    );
    assert!(
        output.contains("service_token_ok"),
        "expected the transition WARN line to carry the structured field \
         `service_token_ok` — operator runbooks key off this name. Captured: {output}"
    );
    // Defense-in-depth: assert the other dependency-set fields are
    // also present, so a future refactor that drops one of them
    // surfaces here rather than as a silent runbook regression. Post-b.9f4
    // the auth-backend fields (permissions_ok / userinfo_ok) are gone from
    // the transition WARN — the dependency set is Valkey + service-token +
    // counter-seed.
    for field in ["valkey_ok", "counter_seed_ok"] {
        assert!(
            output.contains(field),
            "expected the transition WARN line to carry the structured field `{field}` \
             alongside `service_token_ok` — runbook contract pins the full set. Captured: {output}"
        );
    }
}

// ───────────────────────────────────────────────────────────────────────
// b.rgs — first-tick verdict is always published and always logged
// ───────────────────────────────────────────────────────────────────────

/// A replica that is unready from boot MUST still log which dependency is
/// holding it back.
///
/// This is the observability property the supervisor's `Option<bool>`
/// last-verdict seed exists to protect. Since b.rgs `create_health_service`
/// registers readiness `NOT_SERVING`, so a supervisor seeded with a plain
/// `false` would find its first verdict *agreeing* with the published state,
/// skip the write, and emit nothing at all — an operator staring at a pod
/// that never goes ready would have no log naming the cause. Tracking
/// `Option<bool>` makes the first tick a transition in both directions, and
/// the WARN carries `first_verdict=true` to distinguish "never became ready"
/// from "was ready and fell over".
///
/// The unhealthy dependency here is the counter-seed gate (`Some(false)`)
/// rather than the service token, so this test does not restate the
/// token-fixture path already covered above; the property under test is the
/// first-tick log, not which dependency tripped.
///
/// **Why the default `current_thread` flavor**: `tracing::subscriber::set_default`
/// installs the dispatcher in a thread-local, which only reaches
/// `tokio::spawn`-ed tasks running on the same OS thread as the test body.
/// Same reasoning as `supervisor_transition_log_names_service_token_in_dependency_set`.
#[tokio::test]
async fn supervisor_first_tick_unhealthy_dependency_logs_not_serving_verdict() {
    let Some(valkey) = maybe_connect_valkey().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let metrics = make_metrics(/*register_token_gauge=*/ false);

    let (writer, captured) = CapturingWriter::new();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(writer)
        .with_max_level(tracing::Level::WARN)
        .with_ansi(false)
        .finish();
    let dispatch_guard = tracing::subscriber::set_default(subscriber);

    // Counter-seed gate unsatisfied from t=0 — the boot-time analogue of a
    // seed routine that has not completed (or never will). Sender held for
    // the whole test so the channel never closes.
    let (_seed_tx, seed_rx) = tokio::sync::watch::channel(false);

    // Deliberately NOT pre-seeded to SERVING: this test is precisely about
    // the case where the supervisor's verdict AGREES with the registered
    // initial state, which is the case a `false`-seeded supervisor would
    // log nothing for.
    let (reporter, service) = create_health_service(READINESS_SERVICE_NAME).await;
    let supervisor_cancel = CancellationToken::new();
    let supervisor = spawn_health_supervisor(
        reporter.clone(),
        valkey.clone(),
        Arc::clone(&metrics),
        HealthSupervisorConfig {
            probe_interval: SUPERVISOR_PROBE_INTERVAL,
            probe_timeout: SUPERVISOR_PROBE_TIMEOUT,
            readiness_service_name: READINESS_SERVICE_NAME.to_string(),
            service_token_provider: None,
            counter_seed_ready: Some(seed_rx),
            api_serving: None,
        },
        supervisor_cancel.clone(),
    );

    let (addr, server_shutdown) = spawn_health_only_server(service).await;
    let mut client = connect_health(addr).await;

    // Let several probe intervals elapse. Readiness is expected to stay
    // NOT_SERVING throughout — the log, not the status, is the subject.
    for _ in 0..3 {
        tokio::time::sleep(SUPERVISOR_PROBE_INTERVAL + Duration::from_millis(25)).await;
        assert_eq!(
            observe_status(&mut client).await,
            PbServingStatus::NotServing,
            "precondition: an unsatisfied counter-seed gate must hold readiness NOT_SERVING"
        );
    }

    supervisor_cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
    let _ = server_shutdown.send(());
    valkey.quit().await;

    drop(dispatch_guard);

    let buf = captured.lock().unwrap();
    let output = String::from_utf8_lossy(&buf);

    assert!(
        output.contains("readiness flipped to NOT_SERVING"),
        "a supervisor whose FIRST verdict is NOT_SERVING must still publish and log it — \
         otherwise a replica that never becomes ready produces no log naming the blocking \
         dependency. Captured: {output}"
    );
    assert!(
        output.contains("first_verdict=true"),
        "the first-tick NOT_SERVING WARN must carry `first_verdict=true`, distinguishing \
         `never became ready` from `was ready and fell over` — the two have different \
         operator responses. Captured: {output}"
    );
    assert!(
        output.contains("counter_seed_ok=false"),
        "the first-tick WARN must name the dependency that produced the verdict. \
         Captured: {output}"
    );
}

/// A healthy boot logs the raised-to-SERVING line exactly once.
///
/// Two properties in one assertion target: (a) the first tick publishes an
/// affirmative SERVING verdict with its own message — readiness starts
/// `NOT_SERVING` since b.rgs, so an operator needs a log marking the moment
/// the replica became routable; and (b) the supervisor stays edge-triggered,
/// so subsequent healthy ticks are silent. A regression that dropped the
/// `last_serving` bookkeeping and wrote/logged on every tick would show up
/// here as N lines instead of 1 — invisible to every status-based test,
/// since the published status would be identical.
///
/// `current_thread` flavor for the thread-local dispatcher, as above.
#[tokio::test]
async fn supervisor_healthy_boot_logs_readiness_raised_to_serving_exactly_once() {
    let Some(valkey) = maybe_connect_valkey().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let metrics = make_metrics(/*register_token_gauge=*/ false);

    let (writer, captured) = CapturingWriter::new();
    // INFO: the raised / restored lines are `tracing::info!`; the
    // NOT_SERVING line is `tracing::warn!` and is included at this level
    // too, which is what lets the "no spurious flip" assertion below work.
    let subscriber = tracing_subscriber::fmt()
        .with_writer(writer)
        .with_max_level(tracing::Level::INFO)
        .with_ansi(false)
        .finish();
    let dispatch_guard = tracing::subscriber::set_default(subscriber);

    // Every gated dependency healthy: Valkey reachable, service-token check
    // inert, counter-seed gate inert.
    let (reporter, service) = create_health_service(READINESS_SERVICE_NAME).await;
    let supervisor_cancel = CancellationToken::new();
    let supervisor = spawn_health_supervisor(
        reporter.clone(),
        valkey.clone(),
        Arc::clone(&metrics),
        HealthSupervisorConfig {
            probe_interval: SUPERVISOR_PROBE_INTERVAL,
            probe_timeout: SUPERVISOR_PROBE_TIMEOUT,
            readiness_service_name: READINESS_SERVICE_NAME.to_string(),
            service_token_provider: None,
            counter_seed_ready: None,
            api_serving: None,
        },
        supervisor_cancel.clone(),
    );

    let (addr, server_shutdown) = spawn_health_only_server(service).await;
    let mut client = connect_health(addr).await;

    assert!(
        wait_for_status(
            &mut client,
            PbServingStatus::Serving,
            Duration::from_secs(3)
        )
        .await,
        "precondition: with every gated dependency healthy the supervisor's first tick must \
         raise readiness to SERVING"
    );

    // Five more probe intervals of steady-state health. Each is an
    // opportunity for a non-edge-triggered supervisor to log again.
    tokio::time::sleep(SUPERVISOR_PROBE_INTERVAL * 5 + Duration::from_millis(50)).await;

    supervisor_cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(2), supervisor).await;
    let _ = server_shutdown.send(());
    valkey.quit().await;

    drop(dispatch_guard);

    let buf = captured.lock().unwrap();
    let output = String::from_utf8_lossy(&buf);

    let raised = output.matches("readiness raised to SERVING").count();
    assert_eq!(
        raised, 1,
        "the raised-to-SERVING line must be logged exactly once on a healthy boot (got \
         {raised}) — more than one means the supervisor stopped being edge-triggered and \
         is now writing the reporter on every tick; zero means a replica becoming routable \
         leaves no trace in the log. Captured: {output}"
    );
    assert_eq!(
        output.matches("readiness restored to SERVING").count(),
        0,
        "a first-tick SERVING verdict must use the `raised` wording, not the `restored` \
         (dependencies recovered) wording — the latter claims an outage that never \
         happened. Captured: {output}"
    );
    assert_eq!(
        output.matches("readiness flipped to NOT_SERVING").count(),
        0,
        "no NOT_SERVING verdict may be logged while every gated dependency is healthy. \
         Captured: {output}"
    );
}
