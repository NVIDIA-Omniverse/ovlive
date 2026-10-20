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

//! End-to-end integration test for the counter-seed readiness gate
//! (SDD SR-5.3 / SR-5.4, parent Task `t2.fgj.mh.4u`, Issue `b.63b`).
//!
//! The supervisor's `counter_seed_ready: Option<tokio::sync::watch::Receiver<bool>>`
//! gate is already covered by sibling tests:
//!
//! - In-process `src/observability/health.rs` unit tests pin the supervisor's
//!   reaction to a pre-flipped watch handle (gate `true` → SERVING, gate
//!   `false` → NOT_SERVING) without involving a real `seed_object_id_counter`
//!   call or a real gRPC `Check` round-trip.
//! - `tests/integration/checkpoint/counter_seed_test.rs` drives the
//!   `seed_object_id_counter` routine against a real Valkey-backed
//!   `CounterStore` + `InMemoryCheckpointStore` and pins outcome semantics
//!   (`Seeded` / `Absent` / `NoOp` / `ParseFailed`) plus a wrapper that
//!   mirrors the `main.rs` orchestrator-side `tx.send(true)` decision —
//!   but neither stands up the supervisor or the tonic `HealthServer`.
//!
//! This test seals the gap: it wires together the **real** routine, the
//! **real** supervisor, the **real** tonic `HealthServer`, and a **real**
//! gRPC `HealthClient`, observes the `NOT_SERVING → SERVING` transition
//! across the gate flip exactly the way Kubernetes does, and confirms that
//! the routine's `Ok(_)` → `tx.send(true)` → supervisor-tick → reporter
//! flip chain works end-to-end against a real Valkey.
//!
//! The Err-path symmetric test (gate stays `false` on `SeedError::ParseFailed`)
//! is OUT OF SCOPE for this Issue (deferred per the Section 3 analysis).
//!
//! ## Why a new file vs. extending an existing one
//!
//! `health_supervisor_test.rs` is scoped to the Valkey-availability flip
//! tests. `health_supervisor_service_token_test.rs` is scoped to the
//! service-token-aware readiness wiring. Neither is a natural home for an
//! end-to-end test that involves a real `CheckpointStore` + `CounterStore`.
//! A dedicated file mirrors the per-input-signal pattern that already exists
//! and keeps the new test's blast radius contained.
//!
//! ## Helper duplication is deliberate
//!
//! The `spawn_health_only_server` / `connect_health` / `observe_status` /
//! `wait_for_status` helpers are duplicated verbatim from
//! `health_supervisor_service_token_test.rs` (which itself duplicates them
//! from `health_supervisor_test.rs`). The sibling file's load-bearing
//! comment — "duplicated here rather than extracted so this Subtask stays
//! scoped to the new file" — applies in full here. Lifting them into a
//! shared module would expand this Subtask's diff into the `common/`
//! footprint and pull every sibling supervisor test into the same change.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;
use tonic::transport::{Channel, Server};
use tonic_health::pb::{
    health_check_response::ServingStatus as PbServingStatus, health_client::HealthClient,
    HealthCheckRequest,
};

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::counters::CountersSidecar;
use live_edit::checkpoint::{seed_object_id_counter, CheckpointStore, SeedOutcome};
use live_edit::config::Config;
use live_edit::observability::health::{
    create_health_service, spawn_health_supervisor, HealthSupervisorConfig,
};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::counters::CounterStore;
use live_edit::storage::keys::KeySchema;

const READINESS_SERVICE_NAME: &str = "nvidia.omniverse.live.v1alpha.LiveService";

/// Sidecar value used by the single end-to-end test below. Picked above the
/// default Valkey counter floor (the test wipes the counter at the start)
/// so the conditional Lua advances `le:counter:object_id` to `seed`.
const SIDECAR_MAX_OBJECT_ID_OBSERVED: u64 = 5_000;

/// Safety margin passed to `seed_object_id_counter`. Pairs with
/// `SIDECAR_MAX_OBJECT_ID_OBSERVED` to produce the expected `seed = 6_000`
/// asserted in step 9 of the test outline.
const SAFETY_MARGIN: u64 = 1_000;

/// Supervisor probe cadence — short enough that the post-flip wait stays
/// well inside the 2 s `wait_for_status` deadline used by every sibling
/// supervisor integration test. Mirrors the cadence in
/// `health_supervisor_service_token_test.rs`.
const SUPERVISOR_PROBE_INTERVAL: Duration = Duration::from_millis(200);

/// Per-probe deadline. Short enough that a healthy localhost Valkey ping
/// is never racy and a dead-connection ping cannot succeed. Same posture
/// as the sibling supervisor integration test files.
const SUPERVISOR_PROBE_TIMEOUT: Duration = Duration::from_millis(50);

/// Construct a `Config` clone matching the canonical integration-test
/// shape, narrowed to a one-connection pool. Mirrors the sibling
/// `health_supervisor_test.rs` / `health_supervisor_service_token_test.rs`
/// helpers so the readiness-test files exercise comparable connection setups.
fn test_config() -> Config {
    let mut cfg = crate::common::test_config();
    cfg.valkey_pool_size = 1;
    cfg.subscriber_pool_size = 1;
    cfg.valkey_reconnect_max_delay_ms = 500;
    cfg
}

/// Attempt to connect to Valkey using the test config. Returns `None`
/// when Valkey is unreachable — the test short-circuits via `return` in
/// that case, matching the project-wide silent-skip idiom called out in
/// the Section 3 directive.
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
/// `HealthServer` on a random port. Returns the bound addr and a oneshot
/// sender that gracefully shuts the server down. Mirrors the helper in
/// `health_supervisor_test.rs` / `health_supervisor_service_token_test.rs`
/// — duplicated here rather than extracted so this Subtask stays scoped
/// to the new file.
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
/// 25 ms poll interval matches every other supervisor integration test
/// in this crate.
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

/// End-to-end pinning of the SDD SR-5.4 readiness chain: real
/// `seed_object_id_counter` + real `tokio::sync::watch::channel` + real
/// `spawn_health_supervisor` + real tonic `HealthServer` + real gRPC
/// `HealthClient`. The test reproduces the orchestrator-side decision
/// in `src/main.rs` (the `Ok(_) → tx.send(true)` branch around
/// `src/main.rs:662..690`) so the seam under test is the *full* chain a
/// production startup walks, not a mocked subset.
///
/// Sequence (mirrors the parent-Task directive):
///
/// 1. Spin up a `TestValkey`-equivalent connection (silent return if
///    Valkey is unreachable, per the project-wide pattern).
/// 2. Seed an `InMemoryCheckpointStore` with a `CountersSidecar` body
///    encoded the same way the Epic 6 producer encodes it (`serde_json`
///    round-trip through the shared `CountersSidecar` type).
/// 3. Build the real `CounterStore` against a fresh `KeySchema`.
/// 4. Create a `(watch::Sender<bool>, watch::Receiver<bool>)` with
///    initial value `false` — mirrors `src/main.rs:662`.
/// 5. Stand up the real `(reporter, service) = create_health_service`
///    pair and `spawn_health_supervisor` with `counter_seed_ready: Some(rx)`
///    + all other auth/service-token signals off.
/// 6. Confirm via the real `HealthClient` that the supervisor flips
///    `NOT_SERVING` while the gate is still `false` (pre-flip baseline).
/// 7. Call `seed_object_id_counter(...).await`. Assert outcome is
///    `Seeded { seed: 6000, new_counter: 6000 }`.
/// 8. Mirror `src/main.rs:678`'s `Ok(_) → tx.send(true)` decision.
/// 9. Confirm via the real `HealthClient` that the supervisor flips
///    back to `Serving` (post-flip assertion).
/// 10. Cancel the supervisor and `valkey.quit()`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn counter_seed_routine_drives_supervisor_readiness_flip_end_to_end() {
    // Step 1: silent-return if Valkey isn't reachable — matches the
    // project-wide integration-test idiom called out by the Section 3
    // analyst. No `eprintln!("SKIP")` here: the directive explicitly
    // calls that out as non-idiomatic for this codebase.
    let Some(valkey) = maybe_connect_valkey().await else {
        return;
    };

    // Step 2: seed the InMemoryCheckpointStore with a CountersSidecar
    // encoded via `serde_json::to_vec` — the same wire path the Epic 6
    // periodic counters-sync tick uses. Round-tripping through the
    // shared `CountersSidecar` type (vs. hand-rolled JSON) means a
    // future field rename surfaces as a compile error here, not a
    // silent recovery miss in production.
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let sidecar = CountersSidecar {
        max_object_id_observed: SIDECAR_MAX_OBJECT_ID_OBSERVED,
        // Operability-only field per `CountersSidecar` doc — the recovery
        // routine does not read it. A fixed RFC3339 string is fine; the
        // safety margin is the correctness defense, not the timestamp.
        updated_at: "2026-05-26T00:00:00Z".to_string(),
    };
    let sidecar_bytes =
        serde_json::to_vec(&sidecar).expect("CountersSidecar serialization must succeed");
    store
        .put_counters_sidecar(Bytes::from(sidecar_bytes))
        .await
        .expect("put_counters_sidecar must succeed on InMemoryCheckpointStore");

    // Step 3: build the real `CounterStore` against a fresh per-test
    // key prefix. Using a unique prefix means the test never collides
    // with sibling tests that mutate `le:counter:object_id` against the
    // same shared Valkey — same isolation invariant `TestValkey` uses.
    let prefix = format!("test_{}:", uuid::Uuid::new_v4());
    let key_schema = KeySchema::new(&prefix);
    let counters = Arc::new(CounterStore::new(valkey.clone(), key_schema));

    // Step 4: real watch channel, initial value `false`. Mirrors the
    // exact handle shape from `src/main.rs:662`.
    let (tx, rx) = tokio::sync::watch::channel(false);

    // Step 5: real metrics registry + real `create_health_service` pair
    // + real supervisor wiring. We deliberately leave every other
    // contributor to the aggregate verdict OFF (`service_token_provider =
    // None`) so the only signal that can flip readiness is the
    // `counter_seed_ready` gate — keeps the test's intent surface minimal.
    // (Auth-backend readiness probing was removed in b.9f4, so there is no
    // longer any permissions/userinfo contributor to disable.)
    let metrics = Arc::new(crate::common::metrics_scrape_only(
        "counter-seed-readiness-test",
    ));

    let (reporter, service) = create_health_service(READINESS_SERVICE_NAME).await;
    // Pre-seed readiness SERVING before the supervisor starts, so the
    // pre-flip baseline below is a supervisor-driven transition rather than a
    // restatement of the `NOT_SERVING` state `create_health_service` registers
    // since b.rgs. Rationale in `crate::common::preseed_readiness_serving`.
    crate::common::preseed_readiness_serving(&reporter, READINESS_SERVICE_NAME).await;
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
            counter_seed_ready: Some(rx),
            api_serving: None,
        },
        cancel.clone(),
    );

    let (addr, server_shutdown) = spawn_health_only_server(service).await;
    let mut client = connect_health(addr).await;

    // Step 8 (renumbered to "pre-flip baseline"): the supervisor MUST
    // flip readiness to NOT_SERVING on its first tick because the gate
    // is still `false`. The aggregate-verdict logic in
    // `src/observability/health.rs` treats `counter_seed_ready =
    // Some(rx)` + `*rx.borrow() == false` as a hard NOT_SERVING signal
    // (same posture as the service-token gate), so the verdict resolves
    // `serving = false` and the reporter flips on the very first probe.
    assert!(
        wait_for_status(
            &mut client,
            PbServingStatus::NotServing,
            Duration::from_secs(2),
        )
        .await,
        "pre-flip baseline: supervisor must drive readiness from the seeded SERVING down \
         to NOT_SERVING while the counter-seed gate is still false (mirrors the \
         pre-routine state in main.rs before `seed_object_id_counter().await` returns)"
    );

    // Step 9: drive the real routine. `seed = 5_000 + 1_000 = 6_000`,
    // the counter starts unset (a fresh `KeySchema` prefix means
    // `le:counter:object_id` is absent), so the conditional Lua MUST
    // advance the counter from "below seed" up to `seed`. Outcome
    // `Seeded { seed: 6_000, new_counter: 6_000 }`.
    let outcome = seed_object_id_counter(
        Arc::clone(&store),
        Arc::clone(&counters),
        SAFETY_MARGIN,
        Arc::clone(&metrics),
    )
    .await
    .expect("seed_object_id_counter must return Ok against the seeded sidecar + fresh counter");

    assert_eq!(
        outcome,
        SeedOutcome::Seeded {
            seed: SIDECAR_MAX_OBJECT_ID_OBSERVED + SAFETY_MARGIN,
            new_counter: SIDECAR_MAX_OBJECT_ID_OBSERVED + SAFETY_MARGIN,
        },
        "seeded outcome must equal `Seeded {{ seed: 6_000, new_counter: 6_000 }}` — \
         the routine must advance le:counter:object_id from below `seed` up to `seed` \
         via the conditional Lua, and the equality `new_counter == seed` is the \
         load-bearing fact distinguishing the `Seeded` arm from the `NoOp` arm"
    );

    // Step 10: mirror `src/main.rs:678` — flip the watch sender to
    // `true` on `Ok(_)`. This is the seam under test: the real `main.rs`
    // orchestrator inlines this decision, and the supervisor downstream
    // observes whatever value the sender most recently wrote.
    tx.send(true)
        .expect("watch send must succeed (receiver is held by the live supervisor task)");

    // Step 11: post-flip assertion. Within ~one supervisor probe
    // interval the supervisor's tick re-reads `*rx.borrow() == true`,
    // the aggregate verdict flips `serving = true`, the reporter writes
    // `SERVING`, and the real `HealthClient` observes the new state.
    // 2 s budget gives 10x slack over the 200 ms probe interval — same
    // slack as every other supervisor integration test in this crate.
    assert!(
        wait_for_status(
            &mut client,
            PbServingStatus::Serving,
            Duration::from_secs(2),
        )
        .await,
        "post-flip assertion: supervisor must observe the watch flip and restore \
         SERVING within one probe interval + slack; pinning the end-to-end \
         seed_object_id_counter → tx.send(true) → supervisor-tick → reporter flip chain"
    );

    // Step 12: graceful teardown. The supervisor join is bounded under
    // 5 s so a stuck cancel can never wedge the test process — sibling
    // supervisor integration tests use the same 2 s envelope; 5 s gives
    // generous CI headroom.
    cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), supervisor).await;
    let _ = server_shutdown.send(());
    valkey.quit().await;
}
