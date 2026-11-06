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

// Test fixture for integration tests that need a live Valkey connection.
//
// Each `TestValkey` instance uses a unique UUID-based key prefix, so tests can
// run in parallel against a shared Valkey without interfering with each other.
// The `Drop` impl scans and deletes all keys matching the prefix.

use std::env;
use std::ffi::OsString;
use std::sync::{Arc, Once, OnceLock};
use std::time::Duration;

use fred::interfaces::{ClientLike, KeysInterface};
use fred::types::config::CredentialProvider;
use fred::types::scan::ScanType;
use fred::types::Key as FredKey;
use futures::{Stream, StreamExt};
use tokio::sync::Semaphore;

use crate::config::{AuthProtocol, Config, UpdateLayerBackend};
use crate::storage::client::{ConnectError, ValkeyClient};
use crate::storage::keys::KeySchema;

#[cfg(test)]
pub(crate) mod fenced_read_progress;
pub mod relay;
pub mod structural_commit;

/// The external-value store's in-flight bound,
/// `storage::external_values::STORE_VALUES_MAX_INFLIGHT`, for tests outside
/// the crate. The constant itself stays crate-private; this is the same value,
/// so a test bounded by it follows any change to it.
pub const STORE_VALUES_MAX_INFLIGHT: usize =
    crate::storage::external_values::STORE_VALUES_MAX_INFLIGHT;

/// The production unresponsive-connection bounds
/// (`storage::client::UNRESPONSIVE_*`), for tests outside the crate. The
/// constants stay crate-private; these are the same values.
pub const UNRESPONSIVE_MAX_TIMEOUT: std::time::Duration =
    crate::storage::client::UNRESPONSIVE_MAX_TIMEOUT;
/// See [`UNRESPONSIVE_MAX_TIMEOUT`].
pub const UNRESPONSIVE_FAST_TIMEOUT: std::time::Duration =
    crate::storage::client::UNRESPONSIVE_FAST_TIMEOUT;
/// See [`UNRESPONSIVE_MAX_TIMEOUT`].
pub const UNRESPONSIVE_PEER_EVIDENCE_LAG: std::time::Duration =
    crate::storage::client::UNRESPONSIVE_PEER_EVIDENCE_LAG;
/// See [`UNRESPONSIVE_MAX_TIMEOUT`].
pub const UNRESPONSIVE_CHECK_INTERVAL: std::time::Duration =
    crate::storage::client::UNRESPONSIVE_CHECK_INTERVAL;

/// Apply the production unresponsive-connection policy — the one
/// `ValkeyClient::connect` gives every `Builder` it makes — to a raw fred
/// `ConnectionConfig`, for tests that build their own fred client (through a
/// proxy, say). A test that needs shorter bounds overrides the fields
/// afterwards: `unresponsive.max_timeout` and `unresponsive.interval` (the
/// slow bound and scan), and `live_edit.liveness.fast_timeout` and
/// `peer_evidence_lag`. Clients given the same `peer_replies` count each
/// other's replies as evidence.
pub fn apply_unresponsive_policy(
    cfg: &mut fred::types::config::ConnectionConfig,
    peer_replies: &fred::live_edit::liveness::PeerReplies,
    verdicts: crate::storage::client::UnresponsiveVerdicts,
) {
    crate::storage::client::apply_unresponsive_policy(cfg, peer_replies, verdicts);
}

/// Read an object's whole state for a test assertion. Defined in
/// `storage::diff_read` because it shares that module's private per-entry
/// decoders; see it there for its read shape and what it does not prove.
pub use crate::storage::diff_read::read_object_state_for_test;

/// The compact projection of a [`read_object_state_for_test`] snapshot: the
/// structural cache's contract oracle. Defined beside that reader in
/// `storage::diff_read`; see it there.
pub use crate::storage::diff_read::compact_snapshot;

/// Global ceiling on concurrent `TestValkey::setup_with_url` connect handshakes.
///
/// Under heavy parallel test load (~600+ tests), an unbounded set of
/// fixtures all calling `ValkeyClient::connect` simultaneously produces a
/// SYN burst that overflows the Valkey container's `tcp-backlog`, which
/// then surfaces as `TestValkey: failed to connect to Valkey: Timeout` in
/// random integration tests (b.1kf). Holding a permit across the full
/// handshake bounds the connection storm at this ceiling regardless of how
/// many test threads tokio schedules in parallel.
///
/// The permit is released *before* the fixture returns, so the bound only
/// applies during connect — normal-operation client traffic from already-
/// connected fixtures is unrestricted.
///
/// Value rationale: `16` mirrors `LIVE_EDIT_UPDATE_LAYER_PER_OBJECT_CONCURRENCY`
/// (the per-object writer-pool cap used in production; see CLAUDE.md env
/// table). Picking the same number gives future readers an anchor for *why*
/// 16 specifically. It stays well under any reasonable `tcp-backlog`
/// (`scripts/dev-valkey.sh` configures the dev container at 4096), so the
/// SYN-burst path is comfortably defended.
const TEST_VALKEY_SETUP_PERMITS: usize = 16;

fn test_valkey_setup_semaphore() -> &'static Semaphore {
    static SEMAPHORE: OnceLock<Semaphore> = OnceLock::new();
    SEMAPHORE.get_or_init(|| Semaphore::new(TEST_VALKEY_SETUP_PERMITS))
}

/// Connect a `ValkeyClient` under the b.1kf SYN-burst defense semaphore.
///
/// This is the canonical entry point for test fixtures that need to call
/// `ValkeyClient::connect` directly outside of `TestValkey::setup_*`. It
/// acquires a permit on the module-private singleton semaphore, drives the
/// connect handshake, and releases the permit on the way out — capping the
/// peak number of concurrent TCP/TLS negotiations against the dev Valkey
/// container's listen queue at `TEST_VALKEY_SETUP_PERMITS` (= 16) regardless
/// of how many test threads tokio schedules in parallel.
///
/// Every test fixture that needs to call `ValkeyClient::connect` outside of
/// `TestValkey::setup_*` MUST route through this helper to inherit the b.1kf
/// SYN-burst defenses, except for the named exceptions below. The semaphore
/// is a `'static` singleton so the cap is enforced process-wide; bypassing it
/// from a parallel fixture re-opens the SYN-burst flake class b.1kf was filed
/// against.
///
/// The helper is a pure permit-wrapper: it acquires a permit, calls
/// `ValkeyClient::connect(config)` with the caller's `Config` unmodified, and
/// releases the permit on the way out. It does NOT mutate the caller's
/// `Config` — in particular it does not impose any `valkey_connect_timeout_secs`
/// value of its own. Callers keep full control of their connect-timeout
/// budget; `TestValkey::setup_*` fixtures use `valkey_connect_timeout_secs: 30`
/// because the post-`b.fgj` workspace suite's recovery profile under
/// listen-queue pressure needs that budget to drain reliably, but that value
/// lives in the fixture's own `Config`, not here.
///
/// **Intentional exceptions** (deliberately bypass this helper):
/// - `tests/integration/storage/client_quit_timeout_test.rs` — exercises
///   pool-quit timing on a 1-pool client (two test runs × three connections
///   each); its SYN-burst contribution is negligible against the 16-permit
///   ceiling.
/// - `tests/integration/storage/client_test.rs::valkey_client_unreachable_address_returns_err_not_panic`
///   — a deliberate connect-failure-path assertion against an unreachable
///   address with a tight 2 s timeout; routing it through the helper would
///   serve no purpose since it is asserting that `connect` returns `Err`
///   rather than completing a handshake.
pub async fn connect_test_client(config: &Config) -> Result<ValkeyClient, ConnectError> {
    // `acquire()` only errors when the semaphore is closed, which never
    // happens for this `'static` singleton, so we unwrap.
    let _permit = test_valkey_setup_semaphore()
        .acquire()
        .await
        .expect("test_valkey_setup_semaphore is never closed");
    // Test fixtures always connect in static auth mode, so neither a
    // Redis-scoped access-token-file `ServiceTokenProvider` nor an IAM AWS credential
    // source is threaded in.
    ValkeyClient::connect(config, None, None).await
}

/// [`connect_test_client`] with `provider` installed as every connection's
/// credential provider through `build_fred_config`
/// ([`with_credential_provider_for_test`](crate::storage::client::with_credential_provider_for_test)),
/// under the same SYN-burst permit. Drives the passwordless-mode paths — the
/// stream readers' own re-AUTH, fred's pooled refresh — against a store with no
/// ACL: a provider answering with username `default` and any password
/// authenticates there. `config`'s own auth mode is overridden, not consulted.
pub async fn connect_test_client_with_credential_provider(
    config: &Config,
    provider: Arc<dyn CredentialProvider>,
) -> Result<ValkeyClient, ConnectError> {
    crate::storage::client::with_credential_provider_for_test(provider, connect_test_client(config))
        .await
}

/// The variable naming the Valkey the store-backed tests run against. Set
/// (CI always sets it), it makes the store required
/// ([`TestValkeyRequirement::Required`]).
pub const TEST_VALKEY_URL_VAR: &str = "TEST_VALKEY_URL";

/// How long [`connect_test_client_or_skip`] waits for one connect when the
/// store is required, before it fails the test. A hang guard, not a timing
/// claim: a healthy connect takes well under a second, and a loaded shared
/// store is waited for rather than skipped. It is the connect budget
/// `TestValkey`'s fixtures already use under suite load, and sits 90 s below
/// nextest's 120 s kill (`.config/nextest.toml`), so a connect that never
/// completes fails with its own message before the kill.
pub const TEST_VALKEY_CONNECT_HANG_GUARD: Duration = Duration::from_secs(30);

/// How long [`connect_test_client_or_skip`] tries an optional store
/// ([`TestValkeyRequirement::Optional`]) before it skips. The caller's own
/// connect budget applies inside it, unchanged.
const UNSET_STORE_PROBE: Duration = Duration::from_secs(5);

/// Whether a test that needs a Valkey must reach one, decided from the
/// environment by [`from_env`](Self::from_env).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestValkeyRequirement {
    /// [`TEST_VALKEY_URL_VAR`] is set, even to an empty value: a store was
    /// promised, so failing to reach it fails the test.
    Required,
    /// [`TEST_VALKEY_URL_VAR`] is unset outside CI: a local run that may have
    /// no store, so failing to reach one skips the test.
    Optional,
}

impl TestValkeyRequirement {
    /// [`from_vars`](Self::from_vars) on this process's
    /// [`TEST_VALKEY_URL_VAR`] and `GITLAB_CI`, read on every call.
    ///
    /// # Panics
    ///
    /// As [`from_vars`](Self::from_vars).
    pub fn from_env() -> Self {
        Self::from_vars(env::var_os(TEST_VALKEY_URL_VAR), env::var_os("GITLAB_CI"))
    }

    /// The requirement given the two variables' values: `Required` when
    /// `test_valkey_url` is present, `Optional` when both are absent.
    ///
    /// # Panics
    ///
    /// When `test_valkey_url` is absent and `gitlab_ci` present: a CI job that
    /// runs store-backed tests without naming its store is broken job wiring,
    /// and skipping there would turn every such test into a pass nobody sees.
    pub fn from_vars(test_valkey_url: Option<OsString>, gitlab_ci: Option<OsString>) -> Self {
        if test_valkey_url.is_some() {
            return Self::Required;
        }
        assert!(
            gitlab_ci.is_none(),
            "{TEST_VALKEY_URL_VAR} is unset in a CI job (GITLAB_CI is set) that runs Valkey-backed \
             tests; the job must set {TEST_VALKEY_URL_VAR} to its Valkey service"
        );
        Self::Optional
    }
}

/// Connect a bare `ValkeyClient` on `config` through [`connect_test_client`],
/// for a test that needs a store: `Some` when it connects, `None` (after one
/// `SKIP:` line) only when the store is optional and did not answer.
///
/// The requirement comes from [`TestValkeyRequirement::from_env`]:
///
/// - **Required** ([`TEST_VALKEY_URL_VAR`] set): the client is built from a
///   clone of `config` whose `valkey_connect_timeout_secs` is the hang guard's
///   ([`TEST_VALKEY_CONNECT_HANG_GUARD`]), and the connect runs inside that
///   guard, so a slow store is waited for. A connect error or the guard
///   expiring panics, naming the variable, the URL (credentials removed), the
///   guard and the error. It never returns `None`.
///
///   The returned client keeps that raised budget for its whole lifetime, not
///   only for this first connect: it is every connection's timeout on a
///   reconnect, the stream readers' lazy connect and the member replacer's
///   connect. The caller's own `valkey_connect_timeout_secs` is replaced
///   without notice, so a test whose subject depends on its connect budget (a
///   short budget to drive a reconnect path, say) calls
///   [`connect_test_client`] directly.
/// - **Optional** (unset outside CI): `config` as given, for at most 5 s; on
///   failure it prints the `SKIP:` line and returns `None`.
///
/// A test that needs two clients joins two calls (`tokio::join!`), so its
/// connect phase costs one guard, not two.
///
/// # Panics
///
/// When required and the connect fails, and as
/// [`TestValkeyRequirement::from_vars`].
pub async fn connect_test_client_or_skip(config: &Config) -> Option<ValkeyClient> {
    connect_test_client_or_skip_with(
        TestValkeyRequirement::from_env(),
        config,
        TEST_VALKEY_CONNECT_HANG_GUARD,
    )
    .await
}

/// [`connect_test_client_or_skip`] with the requirement and the required
/// mode's hang guard given, so this module's tests drive both modes without
/// touching the process environment. `hang_guard` bounds the required mode
/// only; the optional mode is bounded by [`UNSET_STORE_PROBE`].
async fn connect_test_client_or_skip_with(
    requirement: TestValkeyRequirement,
    config: &Config,
    hang_guard: Duration,
) -> Option<ValkeyClient> {
    let url = crate::storage::client::redact_url_credentials(&config.valkey_url);
    match requirement {
        TestValkeyRequirement::Required => {
            // `ValkeyClient::connect` enforces the connect budget on each of
            // its phases, so the guard alone would still fail a slow connect
            // at the caller's (shorter) budget: raise it to match. The client
            // keeps the raised budget for its lifetime (reconnects, the stream
            // readers' lazy connect, member replacement), replacing the
            // caller's.
            let mut config = config.clone();
            config.valkey_connect_timeout_secs = hang_guard.as_secs().max(1);
            match tokio::time::timeout(hang_guard, connect_test_client(&config)).await {
                Ok(Ok(client)) => Some(client),
                Ok(Err(e)) => panic!(
                    "{TEST_VALKEY_URL_VAR} is set, so this test requires Valkey, but the \
                     connect to {url} failed within the {hang_guard:?} hang guard: {e}"
                ),
                Err(_elapsed) => panic!(
                    "{TEST_VALKEY_URL_VAR} is set, so this test requires Valkey, but the \
                     connect to {url} did not complete within the {hang_guard:?} hang guard"
                ),
            }
        }
        TestValkeyRequirement::Optional => {
            let failure =
                match tokio::time::timeout(UNSET_STORE_PROBE, connect_test_client(config)).await {
                    Ok(Ok(client)) => return Some(client),
                    Ok(Err(e)) => e.to_string(),
                    Err(_elapsed) => format!("no answer within {UNSET_STORE_PROBE:?}"),
                };
            eprintln!(
                "SKIP: {TEST_VALKEY_URL_VAR} is unset and no Valkey answered at {url} \
                 ({failure}); set {TEST_VALKEY_URL_VAR} to run this test"
            );
            None
        }
    }
}

/// The variable naming the multi-shard Valkey Cluster the cluster-gated tests
/// run against. CI's `test-cluster:` and `test-cluster-disruptive:` jobs set
/// it; read it only through [`test_cluster_url`].
pub const TEST_CLUSTER_URL_VAR: &str = "LIVE_EDIT_TEST_CLUSTER_URL";

/// The name rule for a test that reads [`TEST_CLUSTER_URL_VAR`]: its full
/// nextest name (module path included) contains one of these. `cluster`
/// selects it into CI's `test-cluster:` job; `disruptive` is the
/// `#[ignore]`d family the `test-cluster-disruptive:` job runs by name.
pub const CLUSTER_GATED_NAME_MARKS: [&str; 2] = ["cluster", "disruptive"];

/// The filterset CI's `test-cluster:` job selects with: the name rule
/// [`CLUSTER_GATED_NAME_MARKS`] states, as nextest regexes. Every other test
/// runs in the `test:` job alone. The CI-wiring fence in
/// `tests/integration/storage/b_289_tests.rs` holds the job to this string.
pub const CLUSTER_GATED_FILTERSET: &str = "test(/cluster/) | test(/disruptive/)";

/// [`TEST_CLUSTER_URL_VAR`], or `None` when it is unset.
///
/// Under nextest it first checks the calling test's name against
/// [`CLUSTER_GATED_NAME_MARKS`] and panics on a miss, whether the variable is
/// set or not. CI's `test:` job runs every test without a cluster, so a
/// cluster-gated test named outside the rule fails there, rather than being
/// left out of `test-cluster:`'s [`CLUSTER_GATED_FILTERSET`] and never
/// reaching a cluster.
pub fn test_cluster_url() -> Option<String> {
    if let Err(problem) = cluster_gated_name_problem(
        env::var("NEXTEST").ok().as_deref(),
        env::var("NEXTEST_TEST_NAME").ok().as_deref(),
    ) {
        panic!("{problem}");
    }
    env::var(TEST_CLUSTER_URL_VAR).ok()
}

/// [`test_cluster_url`]'s name check, given the `NEXTEST` and
/// `NEXTEST_TEST_NAME` variables. Outside nextest (`cargo test`) there is no
/// test name to check. Under nextest the name must be there: nextest sets it
/// from 0.9.116, and a run without it would pass the check vacuously.
fn cluster_gated_name_problem(
    nextest: Option<&str>,
    test_name: Option<&str>,
) -> Result<(), String> {
    if nextest != Some("1") {
        return Ok(());
    }
    let Some(name) = test_name else {
        return Err(format!(
            "{TEST_CLUSTER_URL_VAR} is read under nextest without NEXTEST_TEST_NAME (nextest \
             sets it from 0.9.116), so the cluster-gated name rule cannot be checked; upgrade \
             cargo-nextest"
        ));
    };
    if CLUSTER_GATED_NAME_MARKS
        .iter()
        .any(|mark| name.contains(mark))
    {
        return Ok(());
    }
    Err(format!(
        "`{name}` reads {TEST_CLUSTER_URL_VAR} but its name contains none of \
         {CLUSTER_GATED_NAME_MARKS:?}, so CI's `test-cluster:` job (filterset \
         `{CLUSTER_GATED_FILTERSET}`) would never run it against a cluster. Put `cluster` in \
         the test's name or its module's"
    ))
}

/// The logical database reserved for
/// `storage::orphan_sweep::tests::scan_keyspace_yields_the_keys_before_a_failed_step_then_one_error_then_nothing`,
/// whose assertions need a keyspace that holds its own keys alone (what a
/// walk's first `SCAN` step returns must not depend on other tests' keys).
/// No other test may select this database; a test that needs a keyspace of
/// its own takes another number and documents it here beside this one.
pub const SCAN_STEP_FAULT_TEST_DATABASE: u8 = 11;

/// `TEST_VALKEY_URL` (default `redis://127.0.0.1:6379`, as `TestValkey`'s) on
/// logical database `db`: any path or query the URL carries is replaced by
/// `/<db>`. For [`TestValkey::setup_with_url`] on a database a test has
/// reserved ([`SCAN_STEP_FAULT_TEST_DATABASE`]); standalone stores only.
pub fn test_valkey_url_on_database(db: u8) -> String {
    let url = env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_owned());
    let authority = url.find("://").map_or(0, |scheme| scheme + 3);
    let base = match url[authority..].find(['/', '?']) {
        Some(end) => &url[..authority + end],
        None => url.as_str(),
    };
    format!("{base}/{db}")
}

/// Per-test Valkey fixture with isolated key namespace.
///
/// On drop, all keys matching the test prefix are deleted via SCAN + UNLINK,
/// and the fixture's client is quit (on a multi-thread runtime; see `Drop`).
pub struct TestValkey {
    client: ValkeyClient,
    key_schema: KeySchema,
    prefix: String,
}

impl TestValkey {
    /// Connect to Valkey and create an isolated test namespace.
    ///
    /// Reads `TEST_VALKEY_URL` from the environment (default: `redis://127.0.0.1:6379`).
    /// Generates a random `test_{uuid}:` prefix so that parallel tests never collide.
    ///
    /// Uses `valkey_pool_size=4` by default — enough to handle roughly 3
    /// concurrent writers plus a slot for WATCH racers — and 4 stream
    /// multiplexer readers (`subscriber_pool_size: 4`); subscriptions share the
    /// readers and hold no data or txn connection. Tests that need an
    /// explicitly different pool size (e.g. to exercise N-way WATCH contention
    /// at N ≥ 5) should call [`setup_with_pool_size`](Self::setup_with_pool_size)
    /// directly.
    pub async fn setup() -> Self {
        Self::setup_with_pool_size(4).await
    }

    /// Like [`setup`](Self::setup) but with a caller-controlled data/txn pool
    /// size. Tests that exercise N-way concurrent WATCH/MULTI/EXEC contention
    /// need `pool_size >= N` so that every racer can hold its own exclusive
    /// txn connection simultaneously; the default `setup()` uses `pool_size=4`,
    /// which covers roughly 3 concurrent writers but not the dedicated 5- or
    /// 10-writer contention tests.
    ///
    /// The reader count is fixed at 4 here (matching [`setup`](Self::setup));
    /// tests needing a different one use
    /// [`setup_with_readers`](Self::setup_with_readers).
    pub async fn setup_with_pool_size(valkey_pool_size: u32) -> Self {
        let valkey_url =
            env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
        Self::setup_with_url(valkey_url, valkey_pool_size).await
    }

    /// Like [`setup_with_pool_size`](Self::setup_with_pool_size) but accepts an
    /// explicit connection URL, bypassing `TEST_VALKEY_URL`. Intended for
    /// cluster-mode regression tests that connect via `redis-cluster://...`
    /// rather than the single-node default.
    pub async fn setup_with_url(valkey_url: String, valkey_pool_size: u32) -> Self {
        Self::setup_with_url_and_readers(valkey_url, valkey_pool_size, 4).await
    }

    /// Like [`Self::setup`] with `readers` stream-multiplexer reader
    /// connections (`subscriber_pool_size`) instead of 4.
    pub async fn setup_with_readers(readers: u32) -> Self {
        let valkey_url =
            env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
        Self::setup_with_url_and_readers(valkey_url, 4, readers).await
    }

    pub async fn setup_with_url_and_readers(
        valkey_url: String,
        valkey_pool_size: u32,
        readers: u32,
    ) -> Self {
        Self::setup_inner(valkey_url, valkey_pool_size, readers, None).await
    }

    /// Like [`Self::setup_with_readers`], with `provider` installed as every
    /// connection's credential provider
    /// ([`connect_test_client_with_credential_provider`]): the passwordless
    /// auth paths — each stream reader's own re-AUTH, fred's pooled refresh —
    /// run against the test store.
    pub async fn setup_with_readers_and_credential_provider(
        readers: u32,
        provider: Arc<dyn CredentialProvider>,
    ) -> Self {
        let valkey_url =
            env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
        Self::setup_inner(valkey_url, 4, readers, Some(provider)).await
    }

    /// Like [`Self::setup_with_url_and_readers`], with `provider` installed as
    /// every connection's credential provider, for a cluster URL.
    pub async fn setup_with_url_readers_and_credential_provider(
        valkey_url: String,
        valkey_pool_size: u32,
        readers: u32,
        provider: Arc<dyn CredentialProvider>,
    ) -> Self {
        Self::setup_inner(valkey_url, valkey_pool_size, readers, Some(provider)).await
    }

    async fn setup_inner(
        valkey_url: String,
        valkey_pool_size: u32,
        readers: u32,
        provider: Option<Arc<dyn CredentialProvider>>,
    ) -> Self {
        raise_fd_limit_for_tests();
        let prefix = format!("test_{}:", uuid::Uuid::new_v4());
        let key_schema = KeySchema::new(&prefix);

        let config = Config {
            port: 0,
            metrics_port: 0,
            valkey_url,
            valkey_pool_size,
            subscriber_pool_size: readers,
            subscriber_admission_wait_ms: 2_000,
            subscriber_max_subscriptions: 4096,
            subscriber_max_concurrent_catchups: 64,
            subscriber_admission_max_waiters: None,
            maintenance_pool_size: 1,
            catchup_snapshot_read_concurrency:
                crate::config::CATCHUP_SNAPSHOT_READ_CONCURRENCY_DEFAULT,
            catchup_fenced_read_concurrency: crate::config::CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT,
            catchup_build_memory_budget_mib: crate::config::CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT,
            snapshot_pool_size: 1,
            snapshot_atomic_read_max_items: crate::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
            snapshot_atomic_read_max_bytes: crate::config::SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT,
            structural_cache_max_mib: crate::config::STRUCTURAL_CACHE_MAX_MIB_DEFAULT,
            structural_cache_audit_rate: crate::config::STRUCTURAL_CACHE_AUDIT_RATE_DEFAULT,
            structural_cache_audit_above_ceiling: false,
            // Connect-budget headroom for parallel-suite load. Production
            // operators tune this via `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS`;
            // the test fixture pins a generous 30 s because the post-`b.fgj`
            // workspace suite peaks well above fred's 10 s built-in default
            // (b.1kf). Paired with `test_valkey_setup_semaphore` (which
            // bounds the *number* of in-flight handshakes) this gives the
            // SYN-burst recovery path enough budget to drain.
            valkey_connect_timeout_secs: 30,
            valkey_command_timeout_ms: 5000,
            valkey_reconnect_max_delay_ms: 30_000,
            valkey_auth_mode: crate::config::ValkeyAuthMode::Static,
            valkey_token_file: None,
            valkey_token_username: None,
            valkey_token_refresh_lead_secs: 180,
            valkey_iam_username: None,
            valkey_iam_region: None,
            valkey_iam_cluster_host: None,
            valkey_iam_service: crate::config::ValkeyIamService::MemoryDb,
            storage_service_endpoint: Some("http://unused.test".to_string()),
            auth_timeout_ms: 1000,
            auth_cache_ttl_secs: 60,
            auth_cache_negative_ttl_secs: 30,
            auth_userinfo_negative_ttl_secs: 5,
            auth_cache_max_entries: 1_000,
            userinfo_endpoint: Some("http://unused.test".to_string()),
            max_layer_size: 1_048_576,
            upload_memory_budget_mib:
                crate::grpc::upload_admission::UPLOAD_MEMORY_BUDGET_MIB_DEFAULT,
            upload_admission_wait_ms:
                crate::grpc::upload_admission::UPLOAD_ADMISSION_WAIT_MS_DEFAULT,
            upload_admission_max_waiters:
                crate::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT,
            layer_stream_maxlen: 4096,
            layer_stream_max_bytes: crate::config::LAYER_STREAM_MAX_BYTES_DEFAULT,
            presence_broadcast_capacity: 4096,
            presence_lease_ttl_secs: crate::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
            presence_refresh_interval_secs: crate::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
            presence_keepalive_interval_ms: crate::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
            shutdown_timeout_secs: 1,
            shutdown_overall_budget_secs: 5,
            shutdown_stream_end_rate: 25,
            shutdown_stream_end_max_window_secs: 20,
            log_level: "warn".to_string(),
            log_format: crate::config::LogFormat::Json,
            trace_filter: crate::config::DEFAULT_TRACE_FILTER.to_string(),
            sweeper_interval_secs: 15,
            session_list_timeout_ms: 1000,
            rpc_deadline_ms: 5000,
            service_label: "test".to_string(),
            valkey_memory_bytes: None,
            ext_release_max_bytes: None,
            ext_reconcile_interval_secs: 3600,
            auth_max_attempts: 3,
            auth_retry_initial_delay_ms: 100,
            authn_enabled: true,
            authz_enabled: false,
            userinfo_service_enabled: true,
            test_only_dangerously_allow_unvalidated_tokens: false,
            health_probe_interval_ms: 5_000,
            health_probe_timeout_ms: 2_000,
            auth_protocol: AuthProtocol::Rest,
            update_layer_retry_max_attempts: 128,
            update_layer_per_object_concurrency: 16,
            update_layer_backend: UpdateLayerBackend::Lua,
            ext_value_verify: crate::config::ExtValueVerifyMode::Enforce,
            session_retry_max_attempts: 32,
            session_retry_initial_delay_ms: 1,
            pod_memory: crate::config::PodMemoryPlan::reference(),
            dr_hold_poll_interval_ms: crate::config::DR_HOLD_POLL_INTERVAL_MS_DEFAULT,
            dr_restore_parallelism: crate::config::DR_RESTORE_PARALLELISM_DEFAULT,
            checkpoint_storage_api_not_found_cache_secs:
                crate::config::CHECKPOINT_STORAGE_API_NOT_FOUND_CACHE_SECS_DEFAULT,
            session_tokens_sweep_interval_secs:
                crate::grpc::session::SESSION_TOKENS_SWEEP_INTERVAL_SECS,
            checkpoint_enabled: false,
            checkpoint_adapter: crate::checkpoint::CheckpointAdapter::Disabled,
            checkpoint_interval_secs: 60,
            checkpoint_quiesce_secs: 30,
            checkpoint_tick_secs: 10,
            checkpoint_retain_generations: 2,
            checkpoint_seq_safety_margin: 1000,
            checkpoint_counters_sync_secs: 60,
            checkpoint_reconcile_interval_secs: 3600,
            checkpoint_counter_safety_margin: 1000,
            checkpoint_tail_enabled: false,
            checkpoint_tail_max_chain: 8,
            checkpoint_tail_max_entries: 4096,
            checkpoint_tail_max_bytes: 67_108_864,
            checkpoint_local_disk_path: None,
            checkpoint_storage_api_endpoint: None,
            checkpoint_base_path: None,
            checkpoint_connect_timeout_ms: 5_000,
            checkpoint_request_timeout_ms: 30_000,
            checkpoint_ext_mirror_workers: 2,
            checkpoint_ext_mirror_queue_capacity: 1_000,
            service_token_file: None,
            service_token_poll_interval_secs: 15,
            service_token_client_credentials_enabled: false,
        };

        // Route through `connect_test_client` so the SYN-burst defense
        // permit-acquire happens uniformly with every other test fixture in
        // the workspace (b.1kf, b.pws). The helper bakes in the
        // permit-around-connect-only and never-panic-on-close rules so
        // call sites cannot forget either one.
        let client = match provider {
            Some(provider) => connect_test_client_with_credential_provider(&config, provider).await,
            None => connect_test_client(&config).await,
        }
        .expect("TestValkey: failed to connect to Valkey");

        TestValkey {
            client,
            key_schema,
            prefix,
        }
    }

    /// The connected Valkey client.
    pub fn client(&self) -> &ValkeyClient {
        &self.client
    }

    /// The key schema configured with this test's unique prefix.
    pub fn key_schema(&self) -> &KeySchema {
        &self.key_schema
    }

    /// The raw prefix string (e.g. `"test_a1b2c3d4-...:"` ).
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

/// Every key matching `pattern`, as a `SCAN MATCH pattern COUNT count` stream
/// paged in the background.
///
/// Dispatches to `scan_cluster_buffered` when the client is connected to a
/// Valkey Cluster so every shard is walked; a plain `scan_buffered` on a
/// clustered client only visits one shard's slot range and would miss keys
/// on sibling shards. It is `test_support`'s one keyless `SCAN` dispatch,
/// shared by [`cleanup_keys`] and the object-keyspace dump
/// ([`structural_commit::dump_object_keyspace`]), so the b.289 source-shape
/// fence (`tests/integration/storage/b_289_tests.rs`) counts one site here.
///
/// `count` is the per-call page hint: each call is one round trip, and a
/// walk of the whole database takes about `DBSIZE / count` of them.
pub(crate) fn scan_matching(
    client: &fred::clients::Client,
    pattern: &str,
    count: u32,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<FredKey, fred::error::Error>> + Send>> {
    let pattern_owned = pattern.to_owned();
    if client.is_clustered() {
        Box::pin(client.scan_cluster_buffered(pattern_owned, Some(count), None::<ScanType>))
    } else {
        Box::pin(client.scan_buffered(pattern_owned, Some(count), None::<ScanType>))
    }
}

/// Scan all keys matching `pattern` and delete them with UNLINK.
///
/// Scans through [`scan_matching`], so every shard of a cluster is walked.
///
/// **Cluster CROSSSLOT avoidance (b.ohv item 11).** In cluster mode an UNLINK
/// with keys spanning multiple hash slots fails the entire batch with
/// `CROSSSLOT Keys in request don't hash to the same slot`. Test prefixes
/// like `test_<uuid>:` deliberately *don't* use a `{hash-tag}`, so the keys
/// we collect during cleanup land on many different slots. To stay correct
/// (and avoid silent key leaks in the cluster CI path), we route every
/// batch through [`fred::util::group_by_hash_slot`] in cluster mode and
/// issue one UNLINK per slot. On single-node the original single-UNLINK
/// path is preserved verbatim — the slot-grouping cost only materializes
/// when `is_clustered()` is true.
///
/// **Every step is bounded, not the whole walk.** `SCAN MATCH <prefix>*`
/// walks the entire keyspace, which other tests and sessions share: about
/// `DBSIZE / CLEANUP_SCAN_COUNT` pages ([`CLEANUP_SCAN_COUNT`]), plus one
/// `UNLINK` per 100 of this test's keys. The number of round trips grows with
/// the database, so no total bound is right for all of them. What a broken
/// client looks like is a step that is never answered — `cleanup_keys` scans
/// through `pool.next()`, and when that member's fred router task has ended, a
/// cluster-wide `scan_cluster_buffered` is never answered and fred's command
/// timeout does not bound it. So each step gets [`CLEANUP_STEP_BUDGET`]: each
/// `UNLINK`, and each `stream.next()`, which waits for the walk's next
/// matching key or its end. A page carries only the keys that match, and fred
/// fetches the next page in the background, so one `stream.next()` spans every
/// page with no match before the next one: on a large database, several round
/// trips. A step that times out, or that fred itself fails with a `Timeout`,
/// is reported as `Err` naming it; any other error ends the cleanup as best
/// effort, as before.
async fn cleanup_keys(pool: &fred::clients::Pool, pattern: &str) -> Result<(), String> {
    let client = pool.next();
    let is_clustered = client.is_clustered();
    let mut stream = scan_matching(client, pattern, CLEANUP_SCAN_COUNT);
    let mut batch = Vec::with_capacity(100);

    loop {
        let next = tokio::time::timeout(CLEANUP_STEP_BUDGET, stream.next())
            .await
            .map_err(|_| format!("a SCAN step got no answer within {CLEANUP_STEP_BUDGET:?}"))?;
        match next {
            None => break,
            Some(Ok(key)) => {
                batch.push(key);
                if batch.len() >= 100 {
                    unlink_batch(pool, std::mem::take(&mut batch), is_clustered).await?;
                }
            }
            Some(Err(e)) if *e.kind() == fred::error::ErrorKind::Timeout => {
                return Err(format!("a SCAN step timed out in fred: {e}"));
            }
            Some(Err(_)) => break,
        }
    }

    if !batch.is_empty() {
        unlink_batch(pool, batch, is_clustered).await?;
    }
    Ok(())
}

/// One bounded `UNLINK` of `keys` (see [`cleanup_keys`]): `Err` when it gets
/// no answer within [`CLEANUP_STEP_BUDGET`] or fred fails it with a `Timeout`;
/// any other failure is swallowed (best-effort teardown).
async fn unlink_step(pool: &fred::clients::Pool, keys: Vec<FredKey>) -> Result<(), String> {
    match tokio::time::timeout(CLEANUP_STEP_BUDGET, pool.unlink::<(), _>(keys)).await {
        Err(_) => Err(format!(
            "an UNLINK got no answer within {CLEANUP_STEP_BUDGET:?}"
        )),
        Ok(Err(e)) if *e.kind() == fred::error::ErrorKind::Timeout => {
            Err(format!("an UNLINK timed out in fred: {e}"))
        }
        Ok(_) => Ok(()),
    }
}

/// UNLINK a batch of keys, splitting by hash slot first in cluster mode.
///
/// Single-node Valkey accepts arbitrary multi-key UNLINK; cluster mode
/// rejects multi-slot batches with `CROSSSLOT`. We pay the BTreeMap split
/// only on the cluster path so the hot single-node branch stays one
/// command per batch.
async fn unlink_batch(
    pool: &fred::clients::Pool,
    batch: Vec<FredKey>,
    is_clustered: bool,
) -> Result<(), String> {
    if batch.is_empty() {
        return Ok(());
    }
    if !is_clustered {
        return unlink_step(pool, batch).await;
    }
    // Cluster mode: group by slot and issue one UNLINK per slot. Failures
    // on individual slot batches are swallowed (best-effort teardown) —
    // matches the single-node branch's semantics. Random `test_<uuid>:`
    // prefixes carry no hash tag, so this typically produces ~one
    // slot-bucket per key on small batches and one bucket per occupied
    // slot on larger ones.
    match fred::util::group_by_hash_slot(batch) {
        Ok(groups) => {
            for (_slot, keys) in groups {
                let keys_vec: Vec<FredKey> = keys.into_iter().collect();
                unlink_step(pool, keys_vec).await?;
            }
        }
        Err(_) => {
            // Unreachable in practice (FredKey is always a plain string here),
            // so we leak this batch rather than risk UNLINKing with wrong slot
            // routing. The `test_<uuid>:` prefix is UUID-collision-resistant,
            // so leaked keys never affect a future test run.
        }
    }
    Ok(())
}

/// Why [`block_on_spawned`] returned no value.
#[derive(Debug)]
pub enum SpawnedTaskEnded {
    /// The future panicked; this is its panic payload.
    Panicked(Box<dyn std::any::Any + Send + 'static>),
    /// The task ended without reporting, so the future was dropped before it
    /// finished: the runtime is shutting down, and either refused the task at
    /// spawn (a closed runtime shuts a new task down at once) or cancelled it.
    NeverReported,
}

/// Run `fut` to completion from synchronous code on a multi-thread runtime
/// and return its output, without consuming a wake-up meant for another
/// future on the calling thread.
///
/// `fut` is spawned on `handle`, and the calling thread waits for its outcome
/// on a `std::sync::mpsc` channel inside `tokio::task::block_in_place`.
///
/// - **Why not a nested `Handle::block_on`.** A thread already inside a
///   `block_on` (a `#[tokio::test]` body, say) parks on tokio's thread-local
///   parker, which holds one pending wake at most. A second `block_on` on the
///   same thread parks on that same parker, so a wake meant for the outer
///   `block_on` — a sibling future's timer firing inside a `join_all` — is
///   consumed by the inner park. The outer `block_on` then parks with nothing
///   left to wake it. tokio's `blocking_recv` parks on the same parker, so it
///   has the same flaw. A std channel's `recv` parks with std thread parking,
///   which tokio's parker does not use, so a wake that arrives while this
///   waits stays pending for the outer `block_on`.
/// - **Why `block_in_place`.** On a runtime worker it hands the worker's core
///   (its LIFO slot included) to another thread before blocking, so the
///   spawned task, and every other task, still runs while this thread waits,
///   even on a runtime with one worker. Off a worker, in a multi-thread
///   runtime's `block_on`, it only leaves the runtime context.
/// - **Outcome.** `Ok` with the output; [`SpawnedTaskEnded::Panicked`] when
///   `fut` panicked (caught with `catch_unwind`, so the payload reaches the
///   caller rather than ending the task silently);
///   [`SpawnedTaskEnded::NeverReported`] when the task was dropped before it
///   reported, which only a runtime shutting down does. The wait itself has
///   no bound: it lasts as long as `fut` does, so bound the awaits in `fut`.
///
/// `fut` is spawned only once `block_in_place` has accepted the calling
/// context, so nothing is left running when it refuses one.
///
/// # Panics
///
/// Where `block_in_place` panics: on a current-thread runtime, and inside a
/// `LocalSet`.
pub fn block_on_spawned<F>(
    handle: &tokio::runtime::Handle,
    fut: F,
) -> Result<F::Output, SpawnedTaskEnded>
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    use futures::FutureExt;
    let (tx, rx) = std::sync::mpsc::channel();
    let received = tokio::task::block_in_place(|| {
        // The task owns the only sender, so `recv` returns once the task has
        // reported or has been dropped without reporting — and in each case
        // only after `fut`, and whatever it owns, is gone: the awaited
        // `catch_unwind` future is a temporary dropped at the end of its
        // statement, before the send; a task cancelled mid-run drops its
        // locals in reverse order, `fut` before `tx`; a task refused at spawn
        // is dropped inside `spawn`, before `recv` starts. The `JoinHandle` is
        // not needed: dropping it detaches the task.
        drop(handle.spawn(async move {
            let tx = tx;
            let fut = std::panic::AssertUnwindSafe(fut);
            let outcome = fut.catch_unwind().await;
            let _ = tx.send(outcome);
        }));
        rx.recv()
    });
    match received {
        Ok(Ok(output)) => Ok(output),
        Ok(Err(payload)) => Err(SpawnedTaskEnded::Panicked(payload)),
        Err(std::sync::mpsc::RecvError) => Err(SpawnedTaskEnded::NeverReported),
    }
}

/// The message of a panic payload, for reporting a caught panic.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s
    } else {
        "a panic with a non-string payload"
    }
}

impl Drop for TestValkey {
    fn drop(&mut self) {
        // Guards:
        // - No current runtime: rare but possible (e.g. Drop from a synchronous
        //   teardown path). Skip the async cleanup — the test-prefix keys will
        //   be cleaned up by the next `TestValkey::setup()` invocation's random
        //   UUID keyspace isolation, so leaving them behind is harmless.
        // - `CurrentThread` runtime: `block_in_place` panics there, so fall
        //   back to the same skip. Integration tests use the multi-thread
        //   runtime; this branch exists for the handful of single-threaded
        //   test harnesses that happen to own a `TestValkey`.
        let Ok(rt_handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        if rt_handle.runtime_flavor() != tokio::runtime::RuntimeFlavor::MultiThread {
            return;
        }

        // Every step of the cleanup is bounded (see `cleanup_keys`), so a step
        // that is never answered — a pooled client whose router task has
        // ended — gives up instead of hanging the test until nextest kills it,
        // which would also swallow the panic message of the test failure that
        // left the client broken. The walk as a whole is not bounded: it is as
        // long as the shared keyspace is large, one `SCAN` page per
        // `CLEANUP_SCAN_COUNT` keys.
        //
        // A test that is already failing (unwinding) only reports the stuck
        // step: a panic in `Drop` while panicking would abort the process and
        // lose the first failure's message. A test that is NOT failing fails
        // here: it left a pooled client unable to answer, and that must not
        // pass silently. Either way the keys left behind are only under this
        // test's unique prefix, which nothing else reads.
        //
        // The cleanup and the quit run as one task spawned on this runtime,
        // and this thread waits for its outcome on a std channel
        // ([`block_on_spawned`]), never in a nested `Handle::block_on`. A
        // fixture is often dropped on a thread already inside a `block_on` —
        // a `#[tokio::test]` body, beside sibling futures in a `join_all` — and
        // a nested `block_on` parks on that thread's tokio parker, consuming
        // a sibling's timer wake-up and leaving the outer `block_on` parked
        // for good. The wait has no bound of its own: it lasts as long as the
        // task, whose every await is bounded as above (and which a runtime
        // shutdown cancels, ending the wait at once).
        let pattern = format!("{}*", self.prefix);
        let task_pattern = pattern.clone();
        let pool = self.client.pool.clone();
        let client = self.client.clone();
        let outcome = block_on_spawned(&rt_handle, async move {
            let cleaned = cleanup_keys(&pool, &task_pattern).await;
            // Then quit the fixture's client, bounded, so its connections
            // close with a `QUIT` while this test's runtime still runs their
            // routers. Without it, the client's last owner dropping still
            // stops its fan-out threads and the snapshot lane's routers on
            // `snapshot-io` (`storage::fanout_thread`), and the other pools'
            // routers end with this test's runtime.
            let quit = tokio::time::timeout(FIXTURE_QUIT_BUDGET, client.quit()).await;
            // Release the clones before reporting, so the fixture's own
            // `client` field stays the client's last owner and is dropped on
            // the dropping thread, after this `drop` returns.
            drop(client);
            drop(pool);
            (cleaned, quit.is_ok())
        });
        match fixture_drop_action(outcome, &pattern, std::thread::panicking()) {
            FixtureDropAction::Report(lines) => {
                for line in lines {
                    eprintln!("{line}");
                }
            }
            FixtureDropAction::Panic { report, message } => {
                for line in report {
                    eprintln!("{line}");
                }
                panic!("{message}");
            }
            FixtureDropAction::ResumeUnwind(payload) => std::panic::resume_unwind(payload),
        }
    }
}

/// What [`TestValkey`]'s drop does once its cleanup task has ended: the pure
/// decision [`fixture_drop_action`] makes, which `Drop` then carries out.
#[derive(Debug)]
pub enum FixtureDropAction {
    /// Write each line to stderr, in order, and return. Empty when there is
    /// nothing to report.
    Report(Vec<String>),
    /// Write each `report` line to stderr, in order, then panic with
    /// `message` (failing the test).
    Panic {
        /// Lines written to stderr before the panic.
        report: Vec<String>,
        /// The panic's message.
        message: String,
    },
    /// Re-raise the cleanup task's own panic with
    /// `std::panic::resume_unwind`, so the test fails with that payload.
    ResumeUnwind(Box<dyn std::any::Any + Send + 'static>),
}

/// Map the outcome of [`TestValkey`]'s drop-time task to what the drop does,
/// with no I/O, so every leg is testable without a store.
///
/// `outcome` is [`block_on_spawned`]'s result for the task: on `Ok`, the
/// cleanup's result (`Err` names the step that stopped) and whether the quit
/// finished within its budget. `pattern` is the fixture's key pattern, named
/// in every message. `panicking` is `std::thread::panicking()` on the dropping
/// thread: a test that is already failing only reports, because a panic in
/// `Drop` while unwinding would abort the process and lose the first failure.
///
/// - `Ok`: a quit that ran out of time is reported; a stopped cleanup panics,
///   or is reported when `panicking`.
/// - [`SpawnedTaskEnded::Panicked`]: re-raised, or reported when `panicking`.
/// - [`SpawnedTaskEnded::NeverReported`]: only a runtime shutting down drops
///   the task unreported (refused at spawn, or cancelled); nothing is wrong
///   with the test, so this is reported and never fails it.
pub fn fixture_drop_action(
    outcome: Result<(Result<(), String>, bool), SpawnedTaskEnded>,
    pattern: &str,
    panicking: bool,
) -> FixtureDropAction {
    match outcome {
        Ok((cleaned, quit_finished)) => {
            let mut report = Vec::new();
            if !quit_finished {
                report.push(format!(
                    "TestValkey: the fixture client's quit for `{pattern}` did not finish \
                     within {FIXTURE_QUIT_BUDGET:?}; its connections close without a `QUIT`"
                ));
            }
            let Err(step) = cleaned else {
                return FixtureDropAction::Report(report);
            };
            let message = format!(
                "TestValkey: key cleanup for `{pattern}` stopped — {step}; leaving those keys \
                 behind — a pooled client can no longer answer"
            );
            if panicking {
                report.push(message);
                FixtureDropAction::Report(report)
            } else {
                FixtureDropAction::Panic { report, message }
            }
        }
        Err(SpawnedTaskEnded::Panicked(payload)) => {
            if panicking {
                FixtureDropAction::Report(vec![format!(
                    "TestValkey: the drop-time cleanup or quit for `{pattern}` panicked — {}; \
                     keys under that prefix may be left behind",
                    panic_message(&*payload)
                )])
            } else {
                FixtureDropAction::ResumeUnwind(payload)
            }
        }
        Err(SpawnedTaskEnded::NeverReported) => FixtureDropAction::Report(vec![format!(
            "TestValkey: the drop-time cleanup or quit for `{pattern}` did not run to the end \
             — the test's runtime is shutting down; keys under that prefix may be left behind"
        )]),
    }
}

/// Bound on each step of [`TestValkey`]'s drop-time key cleanup: one `UNLINK`,
/// or one wait for the walk's next matching key or its end, which spans every
/// `SCAN` page with no match in between (see [`cleanup_keys`]). The number of
/// round trips grows with the shared keyspace, so the walk as a whole has no
/// bound; one step that is not answered within this means a pooled client can
/// no longer answer. A test that is otherwise passing then fails in `Drop` (a
/// panic), and one that is already failing only reports it on stderr.
///
/// Because a step can span many pages, a slow but answering server can also
/// run a step past this bound; the larger [`CLEANUP_SCAN_COUNT`] is, the
/// fewer pages a step spans, and the less likely that is.
///
/// Meant to equal the fixture's `valkey_command_timeout_ms: 5000` (both
/// `Config` literals in this file): a step fred times out is reported the same
/// way as one this bound times out. Changing one means changing the other.
const CLEANUP_STEP_BUDGET: std::time::Duration = std::time::Duration::from_secs(5);

/// The `SCAN` page hint of [`TestValkey`]'s drop-time key cleanup
/// ([`cleanup_keys`]).
///
/// The cleanup walks the whole shared database for its own prefix, so a walk
/// takes about `DBSIZE / CLEANUP_SCAN_COUNT` round trips, and the server
/// visits every key once per walk whatever the page size. On a contended CI
/// runner a cleanup's cost is its round trips, so a large page cuts that cost
/// and the number of pages one [`CLEANUP_STEP_BUDGET`] step can span; it costs
/// only a longer hold of the server per call. The `UNLINK` batch is 100: it
/// counts this test's own keys, not the database's.
const CLEANUP_SCAN_COUNT: u32 = 5_000;

/// Bound on [`TestValkey`]'s drop-time `quit()` of its client. `quit()`
/// bounds each of its own steps; this caps the whole, so a client a test left
/// broken cannot hang its drop.
const FIXTURE_QUIT_BUDGET: std::time::Duration = std::time::Duration::from_secs(10);

/// Raise the per-process `RLIMIT_NOFILE` so the integration suite does not
/// trip `EMFILE` ("Too many open files") under heavy parallelism.
///
/// macOS in particular ships a launchd-inherited soft cap of 256 file
/// descriptors. The test suite opens hundreds of pooled Valkey connections
/// across parallel test threads (per-test data pool, txn pool, subscriber
/// pool, plus stream consumers), and reliably exhausts that cap part-way
/// through the run on a fresh shell where `ulimit -n` has not been raised.
///
/// This helper runs at most once per test process. It is a best-effort
/// soft-bump to `min(hard, 65_536)`. A 65 K target gives the heaviest tests
/// (`update_layer_one_hundred_concurrent_writers_*`,
/// `subscribers_do_not_starve_*`) ~250x the headroom they need without
/// asking the kernel for anything unusual. Failures are logged once to
/// stderr and otherwise swallowed: a low-FD environment will still surface
/// a clean `Too many open files` error from the test it bites, which is
/// strictly more informative than panicking here.
///
/// No-op on non-Unix targets — `RLIMIT_NOFILE` does not exist on Windows
/// and `_setmaxstdio`-style equivalents do not generalise cleanly. The
/// service does not target Windows, and Windows test runs are not part of
/// CI today.
pub fn raise_fd_limit_for_tests() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        #[cfg(unix)]
        raise_fd_limit_unix();
    });
}

#[cfg(unix)]
fn raise_fd_limit_unix() {
    // 65 K is the target soft cap. Anything below this leaves the heaviest
    // parallel tests at risk on macOS (launchd default 256) and on default
    // Linux containers (1024). Anything materially above it adds nothing —
    // we have measured peak FD usage across the suite at well under 4 K.
    const TARGET: libc::rlim_t = 65_536;

    let mut rlim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit reads into a stack-local rlimit. `RLIMIT_NOFILE` is a
    // valid resource id on every Unix we target. No aliasing concerns.
    let rc = unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut rlim) };
    if rc != 0 {
        eprintln!(
            "test_support: getrlimit(RLIMIT_NOFILE) failed (errno {}); leaving FD limit unchanged",
            std::io::Error::last_os_error()
        );
        return;
    }

    if rlim.rlim_cur >= TARGET {
        return;
    }

    let new_cur = TARGET.min(rlim.rlim_max);
    if new_cur <= rlim.rlim_cur {
        // Hard cap is already at or below the current soft. Nothing to do —
        // the operator has constrained the process and we should not fight
        // it.
        return;
    }

    let new_rlim = libc::rlimit {
        rlim_cur: new_cur,
        rlim_max: rlim.rlim_max,
    };
    // SAFETY: setrlimit takes a borrowed rlimit; we pass a stack-local. The
    // call only ever raises the soft cap toward the hard cap, which never
    // requires elevated privilege.
    let rc = unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &new_rlim) };
    if rc != 0 {
        eprintln!(
            "test_support: setrlimit(RLIMIT_NOFILE -> {}) failed (errno {}); leaving FD limit at {}",
            new_cur,
            std::io::Error::last_os_error(),
            rlim.rlim_cur,
        );
    }
}

/// Bound on each wait a [`ConnStall`] performs: the `BLPOP` blocking, and the
/// `BLPOP` settling after [`ConnStall::release`].
pub const CONN_STALL_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// Server-side `BLPOP` timeout. A backstop only: a stall is ended by
/// [`ConnStall::release`] long before it.
const CONN_STALL_BLPOP_SECS: f64 = 60.0;

/// One connection held in a server-side `BLPOP`, so every later command sent
/// on it queues behind the block and gets no reply until
/// [`ConnStall::release`]. A command that waits past its own deadline fails
/// with fred's real `ErrorKind::Timeout`, which is what this exists to produce:
/// a timeout from the real mechanism, on a chosen connection, with no
/// `CLIENT PAUSE` or `DEBUG SLEEP` against the shared Valkey.
///
/// Both the `CLIENT ID` read and the `BLPOP` are pinned to the stall key's hash
/// slot, so on a cluster the stall lands on the connection to the node that
/// owns that slot — give the key the hash tag of the keys the stalled traffic
/// touches.
pub struct ConnStall {
    observer: fred::clients::Client,
    key: String,
    conn_id: String,
    blpop: tokio::task::JoinHandle<Result<fred::types::Value, fred::error::Error>>,
}

impl ConnStall {
    /// Park a `BLPOP` on `conn` for `key` and return once the server reports
    /// the connection blocked in it, so anything sent on `conn` afterwards is
    /// queued behind the block by construction rather than by timing.
    ///
    /// `observer` must be a different connection: it carries the `CLIENT LIST`
    /// polls and the releasing `LPUSH`. `key` should sit under the test's own
    /// prefix and must be empty.
    pub async fn park(
        conn: &fred::clients::Client,
        observer: &fred::clients::Client,
        key: String,
    ) -> Self {
        use fred::interfaces::{ClientInterface, ListInterface};
        let options = fred::types::config::Options {
            cluster_hash: Some(fred::types::ClusterHash::from(key.as_str())),
            max_attempts: Some(1),
            timeout: Some(std::time::Duration::from_secs(
                CONN_STALL_BLPOP_SECS as u64 + 5,
            )),
            ..Default::default()
        };
        let id: i64 = conn
            .with_options(&options)
            .client_id()
            .await
            .expect("CLIENT ID on the connection to stall must succeed");
        let conn_id = id.to_string();
        let stalled = conn.clone();
        let blpop_key = key.clone();
        let blpop = tokio::spawn(async move {
            stalled
                .with_options(&options)
                .blpop::<fred::types::Value, _>(blpop_key, CONN_STALL_BLPOP_SECS)
                .await
        });
        let stall = Self {
            observer: observer.clone(),
            key,
            conn_id,
            blpop,
        };
        let deadline = tokio::time::Instant::now() + CONN_STALL_WAIT;
        loop {
            if stall.is_blocked().await {
                return stall;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "connection {} did not block in the stalling BLPOP within {CONN_STALL_WAIT:?}",
                stall.conn_id
            );
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }

    /// Server-side id of the stalled connection.
    pub fn conn_id(&self) -> &str {
        &self.conn_id
    }

    /// `tot-net-in` of the stalled connection: every byte the server has read
    /// from it, including commands still queued behind the block. `None` once
    /// the connection is gone.
    pub async fn net_in(&self) -> Option<u64> {
        let line = self.client_list_line().await?;
        Some(
            client_list_field(&line, "tot-net-in")
                .parse()
                .expect("CLIENT LIST reports tot-net-in as an integer"),
        )
    }

    /// Whether the server reports the connection still blocked in the stalling
    /// `BLPOP`; `false` once it is not, or once the connection is gone.
    pub async fn is_blocked(&self) -> bool {
        self.client_list_line().await.is_some_and(|line| {
            client_list_field(&line, "flags").contains('b')
                && client_list_field(&line, "cmd") == "blpop"
        })
    }

    /// End the block with an `LPUSH` and wait for the `BLPOP` to settle.
    /// Returns the `BLPOP`'s own result: a non-nil value means the stall held
    /// until this release.
    pub async fn release(self) -> Result<fred::types::Value, fred::error::Error> {
        use fred::interfaces::ListInterface;
        let pushed: i64 = self
            .observer
            .lpush(&self.key, "release")
            .await
            .expect("LPUSH releasing the stall must succeed");
        assert_eq!(pushed, 1, "the release must land on an empty stall list");
        self.settle().await
    }

    /// End the block by killing the stalled connection (`CLIENT KILL ID`), so
    /// everything queued behind the `BLPOP` dies with it, and wait for the
    /// `BLPOP` to settle. Returns the `BLPOP`'s own result, which is an error
    /// when the kill ended it.
    pub async fn kill(self) -> Result<fred::types::Value, fred::error::Error> {
        use fred::interfaces::ClientInterface;
        let killed: i64 = self
            .observer
            .client_kill(vec![fred::types::client::ClientKillFilter::ID(
                self.conn_id.clone(),
            )])
            .await
            .expect("CLIENT KILL of the stalled connection must succeed");
        assert_eq!(
            killed, 1,
            "CLIENT KILL must close exactly the stalled connection"
        );
        self.settle().await
    }

    async fn settle(self) -> Result<fred::types::Value, fred::error::Error> {
        tokio::time::timeout(CONN_STALL_WAIT, self.blpop)
            .await
            .expect("the stalling BLPOP must settle once released")
            .expect("the stalling BLPOP task must not panic")
    }

    async fn client_list_line(&self) -> Option<String> {
        use fred::interfaces::ClientInterface;
        let listing: String = self
            .observer
            .client_list::<String, String>(None, Some(vec![self.conn_id.clone()]))
            .await
            .expect("CLIENT LIST ID must succeed on the observer connection");
        listing
            .lines()
            .find(|l| !l.trim().is_empty())
            .map(str::to_owned)
    }
}

/// [`ConnStall`] every connection of `pool`, observed from `observer` (which
/// must not belong to `pool`). Connection `i` blocks on `{key_prefix}{i}`.
/// Release every returned stall before the test ends.
pub async fn stall_pool(
    pool: &fred::clients::Pool,
    observer: &fred::clients::Client,
    key_prefix: &str,
) -> Vec<ConnStall> {
    let mut stalls = Vec::new();
    for (i, conn) in pool.clients().iter().enumerate() {
        stalls.push(ConnStall::park(conn, observer, format!("{key_prefix}{i}")).await);
    }
    stalls
}

impl TestValkey {
    /// [`stall_pool`] over the data pool, observed from the maintenance pool.
    pub async fn stall_data_pool(&self) -> Vec<ConnStall> {
        let observer = self.client.maintenance_pool.next().clone();
        let prefix = format!("{}conn-stall:data:", self.prefix);
        stall_pool(&self.client.pool, &observer, &prefix).await
    }

    /// [`stall_pool`] over the maintenance pool, observed from the data pool.
    pub async fn stall_maintenance_pool(&self) -> Vec<ConnStall> {
        let observer = self.client.pool.next().clone();
        let prefix = format!("{}conn-stall:maintenance:", self.prefix);
        stall_pool(&self.client.maintenance_pool, &observer, &prefix).await
    }

    /// [`stall_pool`] over the snapshot lane, observed from the maintenance
    /// pool, so a stalled snapshot read can be told apart from a stalled data
    /// pool.
    pub async fn stall_snapshot_pool(&self) -> Vec<ConnStall> {
        let observer = self.client.maintenance_pool.next().clone();
        let prefix = format!("{}conn-stall:snapshot:", self.prefix);
        stall_pool(&self.client.snapshot_pool, &observer, &prefix).await
    }
}

/// Append `value` to the list at `key` from synchronous code on a
/// multi-thread runtime worker — a fan-out sink, say — blocking the caller
/// until the store has it, so its next step cannot run before the push lands.
///
/// It bridges into the runtime through [`block_on_spawned`] (a spawned task
/// and `block_in_place`, never a nested `block_on`, so no wake-up meant for
/// another future on the calling thread is lost), which is why it lives here:
/// this file is compiled only for tests, and is the one file under `src/` the
/// blocking-bridge fence
/// (`stream_consumer_test::no_production_code_under_src_blocks_a_runtime_worker`)
/// exempts. Panics when the push fails or does not run to the end (the
/// runtime is shutting down), outside a runtime, and on a current-thread
/// runtime.
pub fn rpush_blocking(client: &ValkeyClient, key: &str, value: &str) {
    use fred::interfaces::ListInterface;
    let pool = client.pool.clone();
    let key = key.to_owned();
    let value = value.to_owned();
    let pushed = block_on_spawned(&tokio::runtime::Handle::current(), async move {
        let _: i64 = pool
            .rpush(key, value)
            .await
            .expect("RPUSH from a blocking test sink must succeed");
    });
    match pushed {
        Ok(()) => {}
        Err(SpawnedTaskEnded::Panicked(payload)) => std::panic::resume_unwind(payload),
        Err(SpawnedTaskEnded::NeverReported) => panic!(
            "RPUSH from a blocking test sink did not run to the end: the runtime is shutting down"
        ),
    }
}

/// Release every stall, asserting each held until its release (a non-nil
/// `BLPOP` reply).
pub async fn release_all(stalls: Vec<ConnStall>) {
    for stall in stalls {
        let id = stall.conn_id().to_owned();
        let result = stall.release().await;
        assert!(
            matches!(&result, Ok(v) if !v.is_null()),
            "the stall on connection {id} must have held until its release; got {result:?}"
        );
    }
}

// ── Fleet hold marker (SDD SR-4.1) ──────────────────────────────────────────

/// The hold marker [`write_hold_marker`] writes: every one of its five fields
/// (`storage::dr_hold::MARKER_FIELDS`), so every test seeds the same shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HoldMarkerSeed {
    /// `incident`. Default: `test-<8 hex digits>`, unique per seed.
    pub incident: String,
    /// `reason`. Default: `test`.
    pub reason: String,
    /// `evidence`. Default: names this helper.
    pub evidence: String,
    /// `detected_by`. Default: `test-support`.
    pub detected_by: String,
    /// `detected_at_ms`. `None` (the default) writes the store's own clock,
    /// `TIME` on the marker's node, at the write, never this process's clock;
    /// `Some(ms)` writes `ms`, for a test that needs a hold detected in the
    /// past.
    pub detected_at_ms: Option<u64>,
}

impl Default for HoldMarkerSeed {
    fn default() -> Self {
        Self {
            incident: format!("test-{}", &uuid::Uuid::new_v4().simple().to_string()[..8]),
            reason: "test".to_owned(),
            evidence: "written by storage::test_support::write_hold_marker".to_owned(),
            detected_by: "test-support".to_owned(),
            detected_at_ms: None,
        }
    }
}

impl HoldMarkerSeed {
    /// This seed with `incident`.
    pub fn with_incident(mut self, incident: impl Into<String>) -> Self {
        self.incident = incident.into();
        self
    }

    /// This seed with `reason`.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = reason.into();
        self
    }

    /// This seed with `detected_at_ms` fixed at `ms` instead of the store's
    /// clock.
    pub fn detected_at(mut self, ms: u64) -> Self {
        self.detected_at_ms = Some(ms);
        self
    }
}

/// Write the fleet hold marker ([`KeySchema::dr_hold_marker`]) with all five
/// fields of `seed`, overwriting any it already has, and return the
/// `detected_at_ms` written. Panics if the store refuses either command.
///
/// The service itself never writes the marker in this release; this helper is
/// how every test holds the fleet. With `seed.detected_at_ms` unset it reads
/// `TIME` first, pinned to the marker's slot, so the value is the clock of the
/// node that holds the marker.
pub async fn write_hold_marker(
    client: &ValkeyClient,
    keys: &KeySchema,
    seed: &HoldMarkerSeed,
) -> u64 {
    use crate::storage::dr_hold::{
        MARKER_FIELD_DETECTED_AT_MS, MARKER_FIELD_DETECTED_BY, MARKER_FIELD_EVIDENCE,
        MARKER_FIELD_INCIDENT, MARKER_FIELD_REASON,
    };
    use fred::interfaces::HashesInterface;
    let marker = keys.dr_hold_marker();
    let detected_at_ms = match seed.detected_at_ms {
        Some(ms) => ms,
        None => store_time_ms_at(client, &marker).await,
    };
    let _: i64 = client
        .pool
        .hset(
            &marker,
            vec![
                (MARKER_FIELD_INCIDENT, seed.incident.clone()),
                (MARKER_FIELD_REASON, seed.reason.clone()),
                (MARKER_FIELD_EVIDENCE, seed.evidence.clone()),
                (MARKER_FIELD_DETECTED_AT_MS, detected_at_ms.to_string()),
                (MARKER_FIELD_DETECTED_BY, seed.detected_by.clone()),
            ],
        )
        .await
        .expect("HSET of the hold marker must succeed");
    detected_at_ms
}

/// Delete the fleet hold marker, ending the hold; returns whether it existed.
/// Panics if the store refuses the `DEL`.
pub async fn remove_hold_marker(client: &ValkeyClient, keys: &KeySchema) -> bool {
    let removed: i64 = client
        .pool
        .del(keys.dr_hold_marker())
        .await
        .expect("DEL of the hold marker must succeed");
    removed == 1
}

/// The store's clock in milliseconds: `TIME`, pinned to `key`'s slot so it
/// runs on the node that owns `key` (`TIME` is keyless, and fred would
/// otherwise send it to an arbitrary primary).
async fn store_time_ms_at(client: &ValkeyClient, key: &str) -> u64 {
    use fred::types::config::Options;
    use fred::types::{ClusterHash, CustomCommand, Value};
    let pin = Options {
        cluster_hash: Some(ClusterHash::from(key)),
        ..Default::default()
    };
    let reply: Vec<String> = client
        .pool
        .next()
        .with_options(&pin)
        .custom(
            CustomCommand::new_static("TIME", ClusterHash::FirstKey, false),
            Vec::<Value>::new(),
        )
        .await
        .expect("TIME must succeed");
    match reply.as_slice() {
        [secs, usecs] => {
            let secs: u64 = secs.parse().expect("TIME seconds are an integer");
            let usecs: u64 = usecs.parse().expect("TIME microseconds are an integer");
            secs * 1000 + usecs / 1000
        }
        other => panic!("TIME must reply with two integers, got {other:?}"),
    }
}

// ── Lengthened snapshot-fence leases ────────────────────────────────────────

/// [`assert_fence_lease_lengthened`]'s read: the store's `TIME` in whole
/// milliseconds, then `HGETALL KEYS[1]` (the fence key) as flat field/value
/// pairs, in one atomic call on the key's slot. Reads only.
static FENCE_FIELDS_AND_TIME_SCRIPT: std::sync::LazyLock<fred::types::scripts::Script> =
    std::sync::LazyLock::new(|| {
        fred::types::scripts::Script::from_lua(
            "local t = redis.call('TIME') \
             local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000) \
             local reply = {string.format('%d', now)} \
             local fields = redis.call('HGETALL', KEYS[1]) \
             for i = 1, #fields do reply[#reply + 1] = fields[i] end \
             return reply",
        )
    });

/// Assert that the fenced read holding `object_id`'s snapshot fence has a
/// lengthened lease (`ObjectSnapshotReadFaultInjector::lengthen_fence_leases`):
/// the fence key's latest live `r:*` deadline, less the store's `TIME`, is
/// more than `ceil(lease(L))` of `reading`'s production coefficients
/// ([`crate::storage::snapshot_fence::coefficients`]), with `L` the size the
/// fence key stores. Panics with the fields read otherwise, or when the key
/// stores no size or holds no live reader.
///
/// `store` issues the reads and `keys` names the key; `reading` is the client
/// the read ran on. Call it once the read is parked in its held window. A
/// production lease can never pass: every grant and renewal sets its deadline
/// at most `ceil(lease(L))` past the store's clock at that moment. So it
/// catches a lever set on an injector the client no longer has, as when a
/// fixture installs a fresh injector after the test set it.
///
/// The fence key's fields and the store's clock are read in one script call
/// on the key's slot ([`FENCE_FIELDS_AND_TIME_SCRIPT`]), so no round trip
/// separates them: the remainder is exactly what the lease has at that
/// instant, and a loaded runner or store cannot eat into the margin between
/// the two reads.
pub async fn assert_fence_lease_lengthened(
    store: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
    reading: &ValkeyClient,
) {
    use crate::storage::snapshot_fence::{coefficients, LayerSize};
    let fence_key = keys.obj_snapshot_fence(&object_id.to_string());
    let reply: Vec<String> = FENCE_FIELDS_AND_TIME_SCRIPT
        .evalsha_with_reload(
            store.pool.next(),
            vec![fence_key.clone()],
            Vec::<String>::new(),
        )
        .await
        .expect("reading the snapshot fence and the store's TIME must succeed");
    let (now_ms, flat) = reply
        .split_first()
        .expect("the fence read replies with the store's TIME first");
    let now_ms: u64 = now_ms
        .parse()
        .unwrap_or_else(|_| panic!("the store's TIME must be whole ms; got {now_ms:?}"));
    assert!(
        flat.len() % 2 == 0,
        "HGETALL of the snapshot fence must reply in field/value pairs; got {flat:?}"
    );
    let fields: std::collections::HashMap<String, String> = flat
        .chunks_exact(2)
        .map(|pair| (pair[0].clone(), pair[1].clone()))
        .collect();
    let stored = |field: &str| -> u64 {
        fields
            .get(field)
            .unwrap_or_else(|| {
                panic!("the fence key {fence_key} must store {field} once granted; {fields:?}")
            })
            .parse()
            .unwrap_or_else(|_| panic!("the fence key's {field} must be an integer; {fields:?}"))
    };
    let size = LayerSize {
        items: stored("size_e"),
        bytes: stored("size_b"),
    };
    let latest_live = fields
        .iter()
        .filter(|(name, _)| name.starts_with("r:"))
        .map(|(name, value)| {
            value.parse::<u64>().unwrap_or_else(|_| {
                panic!("fence reader field {name} must hold a deadline in ms; got {value:?}")
            })
        })
        .filter(|deadline| *deadline > now_ms)
        .max()
        .unwrap_or_else(|| {
            panic!("the fence key {fence_key} holds no live reader at {now_ms} ms; {fields:?}")
        });
    let remaining_ms = latest_live - now_ms;
    let production_lease_ms = coefficients(reading).timing(size).lease_ms.ceil() as u64;
    assert!(
        remaining_ms > production_lease_ms,
        "the held read's fence lease has {remaining_ms} ms left by the store's clock, no more \
         than a production lease at its stored size ({size:?}: {production_lease_ms} ms): set \
         lengthen_fence_leases(true) on the injector the reading client has now, before the read \
         acquires; fields: {fields:?}"
    );
}

impl TestValkey {
    /// [`write_hold_marker`] with this fixture's client and prefix.
    pub async fn write_hold_marker(&self, seed: &HoldMarkerSeed) -> u64 {
        write_hold_marker(&self.client, &self.key_schema, seed).await
    }

    /// [`remove_hold_marker`] with this fixture's client and prefix.
    pub async fn remove_hold_marker(&self) -> bool {
        remove_hold_marker(&self.client, &self.key_schema).await
    }
}

/// Every `live_edit_valkey_errors_total{pool, kind="command_timeout"}` cell, in
/// `VALKEY_POOLS` order.
///
/// The whole row, never one cell: a command-timeout test proves both that its
/// pool ticked and that no other pool did — a timeout counted on the wrong
/// pool sends an operator to the wrong connection. Every cell is seeded in
/// `Metrics::new`, so an absent one panics rather than reading zero.
pub fn command_timeout_row(
    metrics: &crate::observability::metrics::Metrics,
) -> Vec<(&'static str, u64)> {
    use crate::observability::metrics::test_util::{render, sample_value};
    let text = render(metrics);
    crate::observability::metrics::VALKEY_POOLS
        .iter()
        .map(|pool| {
            let v = sample_value(
                &text,
                "live_edit_valkey_errors_total",
                &[("pool", pool), ("kind", "command_timeout")],
            )
            .unwrap_or_else(|| panic!("the {pool} command_timeout cell is seeded in Metrics::new"));
            (*pool, v as u64)
        })
        .collect()
}

/// The [`command_timeout_row`] with `n` on `pool` and zero elsewhere.
pub fn command_timeouts_only_on(pool: &str, n: u64) -> Vec<(&'static str, u64)> {
    crate::observability::metrics::VALKEY_POOLS
        .iter()
        .map(|p| (*p, if *p == pool { n } else { 0 }))
        .collect()
}

// ── Structural cache test hooks (structural-edits SDD SR-10.2, Test Fixtures) ─
//
// Each writes the object's keys directly, outside every commit path, to put
// the store in a shape a real writer never leaves it in — or leaves it in only
// through a failure (a restore, a rollback, an older pod) — so the structural
// cache's repair, audit and commit fences can be shown to notice.

/// An out-of-band change [`rewrite_out_of_band`] makes to one stored row or
/// list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutOfBandChange {
    /// Rewrite a stored section row's `name` column.
    SectionName { section_id: u64, name: String },
    /// Rewrite a stored field row's value as inline `value` bytes, keeping
    /// its `set_order`, `authoring_set_order` and `version` columns.
    InlineFieldValue {
        section_id: u64,
        key_name: String,
        value: Vec<u8>,
    },
    /// Replace a stored children list with `children`; an empty list deletes
    /// it, with its `obj:chs` member.
    Children { parent_id: u64, children: Vec<u64> },
}

/// Change one of `object_id`'s rows or children lists without a commit: no
/// header write (`seq` and `commit_chain` stay as they are) and no stream
/// entry. The cache's fences cannot see it; a structural commit's pre-image
/// digests (for a row or list the commit rewrites) and the sampled audit can.
///
/// Panics on a store error.
pub async fn rewrite_out_of_band(tv: &TestValkey, object_id: u64, change: OutOfBandChange) {
    use fred::interfaces::{HashesInterface, ListInterface, SetsInterface};

    let pool = &tv.client().pool;
    let keys = tv.key_schema();
    let oid = object_id.to_string();
    match change {
        OutOfBandChange::SectionName { section_id, name } => {
            let _: () = pool
                .hset(
                    keys.obj_section(&oid, &section_id.to_string()),
                    [("name", name)],
                )
                .await
                .expect("rewrite_out_of_band: HSET section name");
        }
        OutOfBandChange::InlineFieldValue {
            section_id,
            key_name,
            value,
        } => {
            let field_key = keys.obj_field(&oid, &section_id.to_string(), &key_name);
            let mut columns = Vec::new();
            crate::storage::layer::encode_value_fields(
                &crate::domain::types::FieldValue::Inline(value),
                &mut columns,
            );
            // Delete-then-write, like every production value write, so the
            // row keeps one value shape.
            let _: () = pool
                .hdel(&field_key, crate::storage::layer::VALUE_HFIELDS.to_vec())
                .await
                .expect("rewrite_out_of_band: HDEL field value columns");
            let _: () = pool
                .hset(&field_key, columns)
                .await
                .expect("rewrite_out_of_band: HSET field value");
        }
        OutOfBandChange::Children {
            parent_id,
            children,
        } => {
            let parent = parent_id.to_string();
            let list_key = keys.obj_children(&oid, &parent);
            let index_key = keys.obj_children_index(&oid);
            let _: () = pool
                .del(&list_key)
                .await
                .expect("rewrite_out_of_band: DEL children list");
            if children.is_empty() {
                let _: () = pool
                    .srem(&index_key, &parent)
                    .await
                    .expect("rewrite_out_of_band: SREM children index");
            } else {
                let ids: Vec<String> = children.iter().map(u64::to_string).collect();
                let _: () = pool
                    .rpush(&list_key, ids)
                    .await
                    .expect("rewrite_out_of_band: RPUSH children list");
                let _: () = pool
                    .sadd(&index_key, &parent)
                    .await
                    .expect("rewrite_out_of_band: SADD children index");
            }
        }
    }
}

/// Give `object_id` a new random `commit_chain` at its unchanged `seq` — what
/// a re-materialization (a DR restore) landing at the same `seq` looks like to
/// the identity fence (SR-4.5). Rows, lists and stream are untouched. Returns
/// the new value.
///
/// Panics on a store error.
pub async fn rematerialize_same_seq(tv: &TestValkey, object_id: u64) -> String {
    use fred::interfaces::HashesInterface;

    let commit_chain = crate::storage::commit_chain::fresh_commit_chain();
    let _: () = tv
        .client()
        .pool
        .hset(
            tv.key_schema().obj_header(&object_id.to_string()),
            [(
                crate::storage::layer::HEADER_FIELD_COMMIT_CHAIN,
                commit_chain.as_str(),
            )],
        )
        .await
        .expect("rematerialize_same_seq: HSET commit_chain");
    commit_chain
}

/// Commit an empty delta to `object_id` the way a writer that does not fold
/// `commit_chain` would (an older pod during a rolling deploy, SR-4.5):
/// `XADD` a replayable `update` entry at `{seq+1}-0` — its payload from
/// [`update_stream_payload`](crate::storage::commit_chain::update_stream_payload),
/// its `ompf` a valid OMPF carrying an OODF delta with no command — then
/// advance the header's `seq`, leaving `commit_chain` as it was. Every real
/// commit path folds since the column exists, so this is the only way to
/// produce the shape: the next repair that crosses the entry finds the fold
/// short of the header and reloads `regressed`.
///
/// The delta changes no content, so the stored rows stay exactly what a fresh
/// read at the new `seq` decodes. The stream is not trimmed. Returns the new
/// `seq`.
///
/// Panics on a store error, or on a header without a decimal `seq`.
pub async fn commit_without_fold(tv: &TestValkey, object_id: u64) -> u64 {
    use fred::interfaces::{HashesInterface, StreamsInterface};

    let pool = &tv.client().pool;
    let keys = tv.key_schema();
    let header_key = keys.obj_header(&object_id.to_string());
    let seq: Option<String> = pool
        .hget(&header_key, crate::storage::layer::HEADER_FIELD_SEQ)
        .await
        .expect("commit_without_fold: HGET seq");
    let seq: u64 = seq
        .and_then(|s| s.parse().ok())
        .expect("commit_without_fold: the header carries a decimal seq");
    let new_seq = seq + 1;

    let delta = oodf_ffi::OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: Vec::new(),
        set_fields: Vec::new(),
        time_samples: Vec::new(),
        source_format: String::new(),
    };
    let delta_bytes = oodf_ffi::build_delta(&delta).expect("commit_without_fold: build_delta");
    let ompf = crate::domain::ompf::OmpfContainer::serialize(&delta_bytes, &[])
        .expect("commit_without_fold: serialize OMPF");
    let server_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or_default();
    let payload = crate::storage::commit_chain::update_stream_payload(
        object_id,
        new_seq,
        "test-non-folding-writer",
        0,
        server_ts,
    );
    let fields: Vec<(&str, fred::types::Value)> = vec![
        (
            crate::storage::notifications::NOTIFICATION_PAYLOAD_FIELD,
            payload.as_str().into(),
        ),
        (
            crate::storage::notifications::RELAY_OMPF_FIELD,
            fred::types::Value::Bytes(ompf.into()),
        ),
    ];
    let _: () = pool
        .xadd(
            keys.layer_stream(object_id),
            false,
            None::<()>,
            format!("{new_seq}-0").as_str(),
            fields,
        )
        .await
        .expect("commit_without_fold: XADD");
    let _: () = pool
        .hset(
            &header_key,
            [(crate::storage::layer::HEADER_FIELD_SEQ, new_seq.to_string())],
        )
        .await
        .expect("commit_without_fold: HSET seq");
    new_seq
}

/// Record a landed commit of the call holding `commit_token` for `client_id`
/// on `object_id` — the `obj:wtok` record both commit backends write,
/// `<token>:<seq>:<server_ts>` — without committing anything. With it a test
/// can show a structural call answering `AlreadyLanded` before it plans.
///
/// Panics on a store error.
pub async fn write_commit_token_record(
    tv: &TestValkey,
    object_id: u64,
    client_id: &str,
    commit_token: &str,
    seq: u64,
    server_ts: u64,
) {
    use fred::interfaces::HashesInterface;

    // The record shape of a call without a client operation id.
    let record = format!("{commit_token}:{seq}:{server_ts}");
    let _: () = tv
        .client()
        .pool
        .hset(
            tv.key_schema().obj_write_token(&object_id.to_string()),
            [(client_id, record.as_str())],
        )
        .await
        .expect("write_commit_token_record: HSET obj:wtok");
}

/// The WATCH path's charged read of `object_id`
/// (`diff_read::read_object_snapshot_slim_charged`, the second
/// `structural_update` entry point) under `charge`, as a structural WATCH
/// attempt reads it: the snapshot, or `Ok(None)` when the read stopped
/// because `charge` could not grow. A read whose round 1 came back short of
/// an index's count (`ChargedSnapshotRead::Incomplete`, which the WATCH
/// attempt weighs only after its `seq` proof) is `Err(Aborted)` naming the
/// shortfall: this helper has no proof, and never returns such a state as a
/// snapshot. For a test binary comparing it with `read_object_snapshot_slim`
/// and the full read (b.26e Stage 2).
pub async fn read_object_snapshot_slim_charged_for_test(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    charge: &dyn crate::storage::commit_charge::SharedCommitCharge,
) -> Result<
    Option<crate::storage::diff_read::AtomicSnapshotOf<crate::domain::types::CompactState>>,
    crate::error::ServiceError,
> {
    use crate::storage::diff_read::{
        read_object_snapshot_slim_charged, ChargedSnapshotRead, FanoutAllowance,
    };
    let mut allowance = FanoutAllowance::new(charge);
    Ok(
        match read_object_snapshot_slim_charged(client, keys, object_id, &mut allowance).await? {
            ChargedSnapshotRead::Snapshot(snapshot) => Some(*snapshot),
            ChargedSnapshotRead::Stopped { .. } => None,
            ChargedSnapshotRead::Incomplete { shortfall, .. } => {
                return Err(crate::error::ServiceError::Aborted(format!(
                    "charged snapshot read of object {object_id}: {shortfall}"
                )))
            }
        },
    )
}

/// What a WATCH commit of one delta is predicted to queue
/// ([`watch_queue_prediction_for_test`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WatchQueuePrediction {
    /// The prediction the commit's last charge point (WA3) prices
    /// (`storage::layer::watch_queued_commands`): an upper bound of each
    /// connection's queue.
    pub queued: crate::storage::commit_charge::QueuedCommands,
    /// The commands the plan's mutations queue into the `MULTI` … `EXEC`
    /// pipeline (`storage::layer::commands_for`, summed): the pipeline's
    /// commands less its frame (`MULTI`, the commit-token record, `XADD`,
    /// `XTRIM`, the `CLIENT ID` witness, `EXEC`), its commit release ring
    /// write (`HDEL`, `HSET` and `PEXPIRE`, predicted only with a value row)
    /// and its external-value index writes.
    pub mutation_commands: u64,
}

/// The queue a WATCH commit of `delta` on an object whose state is `state`
/// is predicted to hold, for a test that compares a commit's observed queue
/// with it: the delta planned on a copy of `state` (`apply_delta`, the
/// mutations every commit path plans), priced as the commit prices them,
/// with the external-value pre-image read counted for a fast-shaped delta
/// as the commit counts it.
///
/// Panics when the delta does not apply to `state`.
pub fn watch_queue_prediction_for_test(
    state: &crate::domain::types::ObjectState,
    delta: &oodf_ffi::OwnedDelta,
) -> WatchQueuePrediction {
    let delta = Some(delta.clone());
    let reads_pre_image = crate::storage::lua::is_fast_path(&delta);
    let mut planned = state.clone();
    let mutations = crate::domain::delta::apply_delta(&mut planned, delta)
        .expect("watch_queue_prediction_for_test: the delta applies to the state");
    WatchQueuePrediction {
        queued: crate::storage::layer::watch_queued_commands(&mutations, reads_pre_image),
        mutation_commands: mutations
            .iter()
            .map(crate::storage::layer::commands_for)
            .sum(),
    }
}

/// The verdict of a watched transaction run by [`exec_watched_writes_for_test`]:
/// the crate-private `storage::watched_txn::WatchedExec`, variant for variant.
#[derive(Debug)]
pub enum WatchedExecForTest {
    /// `EXEC` returned nil: a watched key changed.
    Conflict,
    /// `EXEC` returned an array: every queued command ran.
    Committed,
    /// `EXEC` ran but reported an error, or the `MULTI` failed.
    ExecError(fred::error::Error),
    /// The server discarded the transaction and no queued command ran; the
    /// `EXEC` reply's error.
    Discarded(fred::error::Error),
    /// No full reply set: the connection died with the transaction in
    /// flight, or the reply outlived the transaction's timeout.
    InDoubt(fred::error::Error),
}

/// Send one production `storage::watched_txn::WatchedTxn` on `client`, as the
/// service's commit paths send theirs: opened with every command pinned to
/// `hash` — and with the client timeout `timeout` when one is given, as the
/// WATCH-route `UpdateLayer` commit opens it — then one `SET key value` queued
/// per entry of `writes`, then sent (`MULTI`, the `SET`s, `EXEC`, as one fred
/// pipeline) and decoded.
///
/// No `WATCH` precedes it and the in-transaction `CLIENT ID` witness is not
/// queued: what a test drives through this is the pipeline's send, not the
/// optimistic locking around it. The error is counted on no pool.
pub async fn exec_watched_writes_for_test(
    client: &fred::clients::Client,
    hash: &fred::types::ClusterHash,
    timeout: Option<std::time::Duration>,
    writes: &[(&str, &str)],
) -> WatchedExecForTest {
    use crate::storage::watched_txn::{
        WatchedExec, WatchedTxn, WitnessBaseline, BASELINE_EXEC_DISCARDED,
    };

    let baseline = WitnessBaseline::Unavailable(BASELINE_EXEC_DISCARDED);
    let opened = match timeout {
        Some(timeout) => WatchedTxn::open_with_timeout(client, hash, &baseline, timeout).await,
        None => WatchedTxn::open(client, hash, &baseline).await,
    };
    let txn: WatchedTxn = match opened {
        Ok(txn) => txn,
        Err(e) => return WatchedExecForTest::InDoubt(e),
    };
    for (key, value) in writes {
        if let Err(e) = txn
            .commands()
            .set::<(), _, _>(*key, *value, None, None, false)
            .await
        {
            return WatchedExecForTest::InDoubt(e);
        }
    }
    let origin = crate::storage::client::FredErrOrigin::metrics(
        None,
        crate::storage::client::POOL_LABEL_TXN,
        crate::storage::store_oom::StoreOomSignal::unwired(),
    );
    match txn.exec(origin).await {
        WatchedExec::Conflict => WatchedExecForTest::Conflict,
        WatchedExec::Committed { .. } => WatchedExecForTest::Committed,
        WatchedExec::ExecError(e) => WatchedExecForTest::ExecError(e),
        WatchedExec::Discarded { exec, .. } => WatchedExecForTest::Discarded(exec),
        WatchedExec::InDoubt(e) => WatchedExecForTest::InDoubt(e),
    }
}

/// `OBJECT ENCODING key`: how the store keeps `key` (`listpack`, `skiplist`,
/// `hashtable`, …), for a test whose subject differs between a compact and a
/// table-encoded collection, so it can assert the encoding it set up. Panics
/// when the key is absent or the command fails.
pub async fn object_encoding_for_test(client: &ValkeyClient, key: &str) -> String {
    let reply: fred::types::Value = client
        .pool
        .next()
        .custom(
            fred::types::CustomCommand::new_static(
                "OBJECT",
                fred::types::ClusterHash::Offset(1),
                false,
            ),
            vec![
                fred::types::Value::from("ENCODING"),
                fred::types::Value::from(key),
            ],
        )
        .await
        .unwrap_or_else(|e| panic!("OBJECT ENCODING {key} must succeed: {e}"));
    reply
        .as_string()
        .unwrap_or_else(|| panic!("OBJECT ENCODING {key} answered no encoding: {reply:?}"))
}

/// The cursor pager (the crate-private `storage::retirement::page_by_cursor`)
/// over the sorted set `key`, on one client of `client`'s data pool: a
/// guarded walk (`storage::scan_guard::KeyedWalk`) of `ZSCAN key <cursor>
/// COUNT count` steps from cursor `"0"` until the store answers `"0"`, each
/// step sent with one attempt and, after a transient error, retried in place
/// on an unchanged witness, its witness sent in one pipeline with it (the
/// first step's start witness too) and checked against the walk's start, as
/// the periodic reference-count reconcile pass walks its owed sets. Each
/// step's members are handed to `after_step` and its future awaited before
/// the next step is issued, so a test can change the set between two steps.
/// `Ok` with every member the walk returned, in order, repeats included; the
/// first error ends it: a store fault, a change of serving process
/// ([`WalkError::Changed`](crate::storage::scan_guard::WalkError::Changed))
/// or a missing witness.
pub async fn zscan_members_by_cursor_for_test<F, Fut>(
    client: &ValkeyClient,
    key: &str,
    count: u64,
    after_step: F,
) -> Result<Vec<String>, crate::storage::scan_guard::WalkError>
where
    F: FnMut(Vec<String>) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    use crate::storage::retirement::{page_by_cursor, scan_step_firsts, CursorPageReader};
    use crate::storage::scan_guard::{
        GuardedStep, KeyedWalk, WalkError, Witness, WALK_STEP_ATTEMPTS,
    };

    struct Steps<'a, F> {
        walk: KeyedWalk<'a>,
        count: u64,
        after_step: F,
        returned: Vec<String>,
    }

    impl<F, Fut> CursorPageReader for Steps<'_, F>
    where
        F: FnMut(Vec<String>) -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        type Error = WalkError;
        type Member = String;

        async fn read_step(
            &mut self,
            cursor: &str,
            start: Option<&Witness>,
        ) -> Result<GuardedStep<String>, WalkError> {
            let step = self
                .walk
                .step(cursor, self.count, start, WALK_STEP_ATTEMPTS)
                .await?;
            let key = self.walk.key();
            step.try_map_members(|elements| {
                let elements: Vec<String> = fred::types::Value::Array(elements).convert()?;
                scan_step_firsts("ZSCAN", key, elements)
            })
            .map_err(|source| WalkError::Step {
                walk: "ZSCAN",
                target: std::sync::Arc::from(key),
                source,
            })
        }

        fn changed(&self, start: &Witness, later: &Witness) -> WalkError {
            WalkError::changed(self.walk.change(start, later))
        }

        async fn take_page(&mut self, members: Vec<String>) -> Result<(), WalkError> {
            (self.after_step)(members.clone()).await;
            self.returned.extend(members);
            Ok(())
        }
    }

    let mut steps = Steps {
        walk: KeyedWalk::new(client.pool.next(), "ZSCAN", key),
        count,
        after_step,
        returned: Vec::new(),
    };
    page_by_cursor(&mut steps).await?;
    Ok(steps.returned)
}

/// One `name=value` field of a `CLIENT LIST` line; empty when absent.
fn client_list_field(line: &str, name: &str) -> String {
    line.split_whitespace()
        .find_map(|kv| kv.strip_prefix(name)?.strip_prefix('='))
        .unwrap_or_default()
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_gated_name_check_accepts_a_marked_name_and_refuses_an_unmarked_one() {
        assert_eq!(
            cluster_gated_name_problem(Some("1"), Some("storage::gc_test::cluster_gc_spreads")),
            Ok(())
        );
        assert_eq!(
            cluster_gated_name_problem(Some("1"), Some("storage::cluster_subscriber_test::a")),
            Ok(()),
            "a mark in the module path counts: nextest's name includes it"
        );
        assert_eq!(
            cluster_gated_name_problem(Some("1"), Some("storage::x::stream_mux_disruptive_a")),
            Ok(())
        );
        let refusal = cluster_gated_name_problem(Some("1"), Some("storage::b_ji8_tests::watch"))
            .expect_err("a name with no mark is refused");
        assert!(
            refusal.contains("`storage::b_ji8_tests::watch`")
                && refusal.contains(CLUSTER_GATED_FILTERSET),
            "the refusal names the test and the filterset it misses: {refusal}"
        );
    }

    #[test]
    fn cluster_gated_name_check_needs_the_name_under_nextest_only() {
        assert!(
            cluster_gated_name_problem(Some("1"), None).is_err(),
            "under nextest a missing name would pass vacuously"
        );
        assert_eq!(cluster_gated_name_problem(None, None), Ok(()));
        assert_eq!(cluster_gated_name_problem(None, Some("anything")), Ok(()));
    }

    #[test]
    fn cluster_gated_filterset_is_the_name_marks_as_nextest_regexes() {
        let from_marks = CLUSTER_GATED_NAME_MARKS
            .iter()
            .map(|mark| format!("test(/{mark}/)"))
            .collect::<Vec<_>>()
            .join(" | ");
        assert_eq!(CLUSTER_GATED_FILTERSET, from_marks);
    }

    // ── b.1kf TestValkey setup-semaphore contract ────────────────────────
    //
    // The setup-semaphore caps simultaneous `ValkeyClient::connect`
    // handshakes during fixture spin-up so the post-`b.fgj` workspace test
    // suite's SYN burst stays inside the dev container's listen queue. The
    // tests below pin the bound (16 permits) and the singleton contract
    // (`test_valkey_setup_semaphore` returns the same `'static Semaphore`
    // across every call — otherwise different `TestValkey::setup_*` call
    // sites would each get their own permit pool and the bound would be
    // meaningless).
    //
    // We intentionally do NOT race actual connect ordering or assert on
    // observed `available_permits()` of the live singleton, since the
    // singleton is shared with any other test running concurrently in the
    // same process (`cargo test` parallel scheduling) and any
    // `acquire().await` outside this test would flake the observation.

    #[test]
    fn test_valkey_setup_permits_constant_matches_documented_value() {
        // 16 is the documented bound (mirrors
        // `LIVE_EDIT_UPDATE_LAYER_PER_OBJECT_CONCURRENCY`). A future change
        // that adjusts this value MUST also update the doc-comment and the
        // CLAUDE.md / SDD narrative — failing this assertion forces that
        // synchronization.
        assert_eq!(
            TEST_VALKEY_SETUP_PERMITS, 16,
            "TEST_VALKEY_SETUP_PERMITS must stay aligned with its doc-comment \
             (mirrors LIVE_EDIT_UPDATE_LAYER_PER_OBJECT_CONCURRENCY = 16); \
             update both the constant and the doc-comment together (b.1kf)"
        );
    }

    #[test]
    fn test_valkey_setup_semaphore_is_singleton() {
        // The semaphore must be a process-wide singleton: every
        // `TestValkey::setup_*` call site needs to draw from the same
        // permit pool, otherwise the SYN-burst bound is bypassed by
        // dispatching from a fresh call site.
        let a = test_valkey_setup_semaphore() as *const Semaphore;
        let b = test_valkey_setup_semaphore() as *const Semaphore;
        assert_eq!(
            a, b,
            "test_valkey_setup_semaphore must return the same 'static \
             Semaphore reference across calls so the connect-burst cap is \
             enforced process-wide (b.1kf)"
        );
    }

    #[test]
    fn test_valkey_setup_semaphore_has_at_most_documented_permits() {
        // Upper-bound check: the singleton must never expose more permits
        // than the documented bound. We can't assert exact equality on the
        // shared singleton without serialising the entire test process
        // (other tests holding `TestValkey::setup_*` permits would race the
        // observation), so we assert the invariant that always holds:
        // `available_permits() <= TEST_VALKEY_SETUP_PERMITS`.
        let permits = test_valkey_setup_semaphore().available_permits();
        assert!(
            permits <= TEST_VALKEY_SETUP_PERMITS,
            "test_valkey_setup_semaphore exposes {permits} permits, but the \
             documented bound is {TEST_VALKEY_SETUP_PERMITS}; the constructor \
             must build the semaphore with exactly TEST_VALKEY_SETUP_PERMITS \
             permits (b.1kf)"
        );
    }

    // ── b.pws connect_test_client helper contract ────────────────────────
    //
    // `connect_test_client` is the canonical entry point for test fixtures
    // that build their own `Config` and previously called
    // `ValkeyClient::connect` directly. The test below pins the helper's
    // plumbing (semaphore acquire → connect → release) end-to-end against
    // the dev Valkey container. We intentionally do NOT try to assert
    // SYN-burst-defense semantics (which require multi-process listen-queue
    // saturation to reproduce) — instead we pin the contract that the
    // helper's happy path succeeds and that a permit is returned to the
    // pool on return (so a long-running test process doesn't slowly leak
    // permits down to zero).
    //
    // The happy-path test needs its store through
    // `connect_test_client_or_skip`: with `TEST_VALKEY_URL` set an
    // unreachable store fails it, and only an unset variable outside CI
    // (a host shell without `dev-valkey.sh up`) skips it.

    fn minimal_test_config(valkey_url: String) -> Config {
        // Mirrors the `Config` `TestValkey::setup_with_url` builds, with
        // the minimum surface needed to drive `ValkeyClient::connect`. If
        // the fixture's `Config` grows new fields, this helper picks up
        // the breakage at compile time — the test reads as "any valid
        // Config should connect", not "this exact field set".
        Config {
            port: 0,
            metrics_port: 0,
            valkey_url,
            valkey_pool_size: 1,
            subscriber_pool_size: 1,
            subscriber_admission_wait_ms: 2_000,
            subscriber_max_subscriptions: 4096,
            subscriber_max_concurrent_catchups: 64,
            subscriber_admission_max_waiters: None,
            maintenance_pool_size: 1,
            catchup_snapshot_read_concurrency:
                crate::config::CATCHUP_SNAPSHOT_READ_CONCURRENCY_DEFAULT,
            catchup_fenced_read_concurrency: crate::config::CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT,
            catchup_build_memory_budget_mib: crate::config::CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT,
            snapshot_pool_size: 1,
            snapshot_atomic_read_max_items: crate::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
            snapshot_atomic_read_max_bytes: crate::config::SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT,
            structural_cache_max_mib: crate::config::STRUCTURAL_CACHE_MAX_MIB_DEFAULT,
            structural_cache_audit_rate: crate::config::STRUCTURAL_CACHE_AUDIT_RATE_DEFAULT,
            structural_cache_audit_above_ceiling: false,
            valkey_connect_timeout_secs: 30,
            valkey_command_timeout_ms: 5000,
            valkey_reconnect_max_delay_ms: 30_000,
            valkey_auth_mode: crate::config::ValkeyAuthMode::Static,
            valkey_token_file: None,
            valkey_token_username: None,
            valkey_token_refresh_lead_secs: 180,
            valkey_iam_username: None,
            valkey_iam_region: None,
            valkey_iam_cluster_host: None,
            valkey_iam_service: crate::config::ValkeyIamService::MemoryDb,
            storage_service_endpoint: Some("http://unused.test".to_string()),
            auth_timeout_ms: 1000,
            auth_cache_ttl_secs: 60,
            auth_cache_negative_ttl_secs: 30,
            auth_userinfo_negative_ttl_secs: 5,
            auth_cache_max_entries: 1_000,
            userinfo_endpoint: Some("http://unused.test".to_string()),
            max_layer_size: 1_048_576,
            upload_memory_budget_mib:
                crate::grpc::upload_admission::UPLOAD_MEMORY_BUDGET_MIB_DEFAULT,
            upload_admission_wait_ms:
                crate::grpc::upload_admission::UPLOAD_ADMISSION_WAIT_MS_DEFAULT,
            upload_admission_max_waiters:
                crate::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT,
            layer_stream_maxlen: 4096,
            layer_stream_max_bytes: crate::config::LAYER_STREAM_MAX_BYTES_DEFAULT,
            presence_broadcast_capacity: 4096,
            presence_lease_ttl_secs: crate::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
            presence_refresh_interval_secs: crate::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
            presence_keepalive_interval_ms: crate::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
            shutdown_timeout_secs: 1,
            shutdown_overall_budget_secs: 5,
            shutdown_stream_end_rate: 25,
            shutdown_stream_end_max_window_secs: 20,
            log_level: "warn".to_string(),
            log_format: crate::config::LogFormat::Json,
            trace_filter: crate::config::DEFAULT_TRACE_FILTER.to_string(),
            sweeper_interval_secs: 15,
            session_list_timeout_ms: 1000,
            rpc_deadline_ms: 5000,
            service_label: "test".to_string(),
            valkey_memory_bytes: None,
            ext_release_max_bytes: None,
            ext_reconcile_interval_secs: 3600,
            auth_max_attempts: 3,
            auth_retry_initial_delay_ms: 100,
            authn_enabled: true,
            authz_enabled: false,
            userinfo_service_enabled: true,
            test_only_dangerously_allow_unvalidated_tokens: false,
            health_probe_interval_ms: 5_000,
            health_probe_timeout_ms: 2_000,
            auth_protocol: AuthProtocol::Rest,
            update_layer_retry_max_attempts: 128,
            update_layer_per_object_concurrency: 16,
            update_layer_backend: UpdateLayerBackend::Lua,
            ext_value_verify: crate::config::ExtValueVerifyMode::Enforce,
            session_retry_max_attempts: 32,
            session_retry_initial_delay_ms: 1,
            pod_memory: crate::config::PodMemoryPlan::reference(),
            dr_hold_poll_interval_ms: crate::config::DR_HOLD_POLL_INTERVAL_MS_DEFAULT,
            dr_restore_parallelism: crate::config::DR_RESTORE_PARALLELISM_DEFAULT,
            checkpoint_storage_api_not_found_cache_secs:
                crate::config::CHECKPOINT_STORAGE_API_NOT_FOUND_CACHE_SECS_DEFAULT,
            session_tokens_sweep_interval_secs:
                crate::grpc::session::SESSION_TOKENS_SWEEP_INTERVAL_SECS,
            checkpoint_enabled: false,
            checkpoint_adapter: crate::checkpoint::CheckpointAdapter::Disabled,
            checkpoint_interval_secs: 60,
            checkpoint_quiesce_secs: 30,
            checkpoint_tick_secs: 10,
            checkpoint_retain_generations: 2,
            checkpoint_seq_safety_margin: 1000,
            checkpoint_counters_sync_secs: 60,
            checkpoint_reconcile_interval_secs: 3600,
            checkpoint_counter_safety_margin: 1000,
            checkpoint_tail_enabled: false,
            checkpoint_tail_max_chain: 8,
            checkpoint_tail_max_entries: 4096,
            checkpoint_tail_max_bytes: 67_108_864,
            checkpoint_local_disk_path: None,
            checkpoint_storage_api_endpoint: None,
            checkpoint_base_path: None,
            checkpoint_connect_timeout_ms: 5_000,
            checkpoint_request_timeout_ms: 30_000,
            checkpoint_ext_mirror_workers: 2,
            checkpoint_ext_mirror_queue_capacity: 1_000,
            service_token_file: None,
            service_token_poll_interval_secs: 15,
            service_token_client_credentials_enabled: false,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn connect_test_client_returns_ok_on_happy_path() {
        // Drive `connect_test_client` end-to-end: acquire a permit, run
        // the connect handshake, release the permit, return an `Ok` client,
        // reached through `connect_test_client_or_skip`: with
        // `TEST_VALKEY_URL` set a failed connect fails the test, and only an
        // unset variable outside CI skips it.
        let valkey_url =
            env::var(TEST_VALKEY_URL_VAR).unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
        let config = minimal_test_config(valkey_url);

        let Some(_client) = connect_test_client_or_skip(&config).await else {
            return;
        };
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn connect_test_client_releases_permit_on_return() {
        // After `connect_test_client` returns (success OR failure), the
        // permit MUST be released back to the singleton — otherwise a
        // long-running test process slowly leaks permits down to zero and
        // every subsequent fixture stalls waiting on `acquire().await`.
        // The permit is held via `let _permit = ...` and dropped at scope
        // exit, so this test pins the "permit-around-connect-only"
        // doc-comment contract (b.pws).
        //
        // The semaphore is a *process-global* singleton shared with every
        // other `#[cfg(test)]` `--lib` test that routes through
        // `connect_test_client` (e.g. `observability::health`'s
        // `real_valkey`). A before/after gauge of `available_permits()`
        // is therefore racy: a concurrent test may legitimately hold a
        // permit at the instant we sample, dipping the count without any
        // leak. We instead drive a real leak's *monotonic* signature — a
        // per-call leak would, after `TEST_VALKEY_SETUP_PERMITS` calls,
        // permanently pin the pool at zero. A leak-free helper keeps
        // handing permits back, so after `2 × TEST_VALKEY_SETUP_PERMITS`
        // sequential acquire/connect/release cycles the pool is still
        // non-empty (and a permit is still acquirable) even with a few
        // transient concurrent holders, since pool size is 16.
        //
        // The subject is the permit, not the connect, so the connect's result
        // is ignored and its budget is 1 s: against an unreachable store the
        // cycles end in about 32 s, well inside nextest's 120 s kill, where
        // the fixture's 30 s budget (cut at 10 s) would reach it.
        let valkey_url =
            env::var(TEST_VALKEY_URL_VAR).unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
        let config = Config {
            valkey_connect_timeout_secs: 1,
            ..minimal_test_config(valkey_url)
        };

        for _ in 0..(2 * TEST_VALKEY_SETUP_PERMITS) {
            let _ = tokio::time::timeout(
                std::time::Duration::from_secs(10),
                connect_test_client(&config),
            )
            .await;
        }

        // A genuine per-call leak would have permanently drained the pool
        // to zero by now, so a fresh acquire would block until the 5 s
        // timeout fires. Transient contention from concurrent tests
        // resolves well within that budget as they release their own
        // permits. Acquire-and-drop rather than gauge `available_permits()`
        // so the assertion is immune to the exact instantaneous count.
        let acquired = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            test_valkey_setup_semaphore().acquire(),
        )
        .await;
        assert!(
            acquired.is_ok(),
            "connect_test_client leaked its permit: setup pool is exhausted \
             (acquire timed out after {} sequential acquire/release cycles) \
             — b.pws: the `let _permit = ...` binding must drop at scope \
             exit, releasing the permit on both Ok and Err return paths",
            2 * TEST_VALKEY_SETUP_PERMITS
        );
        // Drop the guard explicitly to release the permit we just took back
        // to the pool — this test must not itself leak a permit.
        drop(acquired);
    }

    // ── b.phh connect_test_client_or_skip: a required store fails, never skips ──
    //
    // The tests drive `connect_test_client_or_skip_with` with the requirement
    // given, so they cover both modes without touching the process
    // environment. Only the stall test needs a store, and it reaches it
    // through `connect_test_client_or_skip` like any other caller.

    /// How long the stall test's forwarder waits before its first accept:
    /// above the old 5 s bounds (the outer timeout and the callers' 5 s
    /// connect budget), and below fred's 10 s `internal_command_timeout`, so
    /// the setup command written into the stalled connection is still
    /// answered once the forwarder starts relaying it.
    const ACCEPT_STALL: Duration = Duration::from_secs(7);

    /// An address no store listens on, so a connect to it is refused at once.
    const UNREACHABLE_STORE_URL: &str = "redis://127.0.0.1:1";

    /// The store's `host:port`, and `url` with its host moved to loopback, when
    /// `url` is a plain `redis://` URL a TCP forwarder can stand in for (not
    /// TLS, not a cluster). The copy keeps the scheme, credentials and
    /// database; the caller sets its port.
    fn plain_redis_target(url: &str) -> Option<(String, url::Url)> {
        let parsed = url::Url::parse(url)
            .ok()
            .filter(|u| u.scheme() == "redis")?;
        let store_addr = format!("{}:{}", parsed.host_str()?, parsed.port().unwrap_or(6379));
        let mut proxy = parsed;
        proxy.set_host(Some("127.0.0.1")).ok()?;
        Some((store_addr, proxy))
    }

    /// A required store whose connect cannot finish for [`ACCEPT_STALL`] (7 s,
    /// past the old 5 s outer timeout and 5 s connect budget) is waited for,
    /// and the helper returns its client. The stall is a TCP forwarder in
    /// front of the store that sleeps before its first accept: the client's
    /// TCP connect completes in the listen backlog, and its setup command
    /// gets no reply until the forwarder starts relaying.
    ///
    /// The caller's config carries a 5 s connect budget, as
    /// `common::test_config()` does, so the test also fails if the helper
    /// stops raising the budget to its hang guard. Slow by design (about 7 s):
    /// the only test in the suite that pays the stall.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn connect_test_client_or_skip_when_required_waits_out_a_connect_stall_past_the_old_five_second_bound(
    ) {
        let store_url =
            env::var(TEST_VALKEY_URL_VAR).unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
        // The test's own need for a store: required when the variable is set,
        // a skip only when it is unset outside CI.
        let Some(probe) =
            connect_test_client_or_skip(&minimal_test_config(store_url.clone())).await
        else {
            return;
        };
        probe.quit().await;
        let Some((store_addr, mut proxy_url)) = plain_redis_target(&store_url) else {
            eprintln!(
                "SKIP: {TEST_VALKEY_URL_VAR} is not a plain redis://host:port URL, so this test \
                 cannot put a TCP forwarder in front of the store"
            );
            return;
        };

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("the forwarder must bind a loopback port");
        let proxy_port = listener
            .local_addr()
            .expect("the forwarder's listener must have an address")
            .port();
        proxy_url
            .set_port(Some(proxy_port))
            .expect("a redis:// URL with a host takes a port");
        let accepted = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // The clock starts before the forwarder's sleep does, so no connect
        // through it can complete in less than the stall.
        let started = tokio::time::Instant::now();
        let forwarder = tokio::spawn({
            let accepted = Arc::clone(&accepted);
            async move {
                // The injected stall itself, not a wait on another task.
                tokio::time::sleep(ACCEPT_STALL).await;
                // Held by this task, so aborting it ends every relay too.
                let mut relays = tokio::task::JoinSet::new();
                while let Ok((mut inbound, _)) = listener.accept().await {
                    accepted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let store_addr = store_addr.clone();
                    relays.spawn(async move {
                        if let Ok(mut outbound) = tokio::net::TcpStream::connect(&store_addr).await
                        {
                            let _ =
                                tokio::io::copy_bidirectional(&mut inbound, &mut outbound).await;
                        }
                    });
                }
            }
        });

        let config = Config {
            valkey_connect_timeout_secs: 5,
            ..minimal_test_config(proxy_url.to_string())
        };
        let client = connect_test_client_or_skip_with(
            TestValkeyRequirement::Required,
            &config,
            TEST_VALKEY_CONNECT_HANG_GUARD,
        )
        .await
        .expect("a required store returns its client or panics; it never returns None");
        let elapsed = started.elapsed();

        assert!(
            accepted.load(std::sync::atomic::Ordering::SeqCst) >= 1,
            "the client connected through the stalled forwarder"
        );
        assert!(
            elapsed >= ACCEPT_STALL,
            "the connect completed after {elapsed:?}, before the forwarder's {ACCEPT_STALL:?} \
             stall ended, so it was never held up by it"
        );
        client.quit().await;
        forwarder.abort();
    }

    /// The panic message of a required-mode connect on `config` under
    /// `hang_guard`, failing the calling test if the helper returns instead
    /// (a `None` is the silent skip every caller would take).
    async fn required_connect_panic_message(config: &Config, hang_guard: Duration) -> String {
        use futures::FutureExt;

        let outcome = std::panic::AssertUnwindSafe(connect_test_client_or_skip_with(
            TestValkeyRequirement::Required,
            config,
            hang_guard,
        ))
        .catch_unwind()
        .await;
        let payload = match outcome {
            Ok(None) => panic!(
                "a required store that could not be connected returned None, which every caller \
                 turns into a silent skip"
            ),
            Ok(Some(_)) => panic!("{} unexpectedly accepted a connect", config.valkey_url),
            Err(payload) => payload,
        };
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .expect("the helper panics with a message")
    }

    /// A required connect that fails fails the test (the helper's connect-error
    /// arm), naming the variable that made the store required. The URL is
    /// empty, as an empty `TEST_VALKEY_URL` would make it: the variable still
    /// counts as set, and fred rejects the URL while parsing it, before any
    /// connect or retry, so the full hang guard cannot expire first.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn connect_test_client_or_skip_when_required_panics_naming_test_valkey_url_for_a_failed_connect(
    ) {
        let config = minimal_test_config(String::new());
        let message = required_connect_panic_message(&config, TEST_VALKEY_CONNECT_HANG_GUARD).await;
        assert!(
            message.contains(TEST_VALKEY_URL_VAR),
            "the panic names {TEST_VALKEY_URL_VAR}, the variable that made the store required: \
             {message}"
        );
        assert!(
            message.contains("failed within") && message.contains("invalid Valkey URL"),
            "the panic comes from the connect-error arm and carries the connect's error: \
             {message}"
        );
    }

    /// A required connect still unfinished at the hang guard fails the test
    /// (the helper's guard-expiry arm), naming the variable and the URL with
    /// its credentials removed. The guard is under a second, so it expires
    /// before the connect budget the helper raises to it (at least 1 s) can
    /// end the refused connect's retries.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn connect_test_client_or_skip_when_required_panics_naming_the_redacted_url_when_the_hang_guard_expires(
    ) {
        let config = minimal_test_config("redis://user:hunter2@127.0.0.1:1".to_string());
        let message = required_connect_panic_message(&config, Duration::from_millis(500)).await;
        assert!(
            message.contains(TEST_VALKEY_URL_VAR),
            "the panic names {TEST_VALKEY_URL_VAR}, the variable that made the store required: \
             {message}"
        );
        assert!(
            message.contains("did not complete within"),
            "the panic comes from the guard-expiry arm: {message}"
        );
        assert!(
            message.contains("redis://127.0.0.1:1"),
            "the panic names the address it could not reach: {message}"
        );
        assert!(
            !message.contains("hunter2"),
            "the panic leaves the URL's password out: {message}"
        );
    }

    /// An optional store (the variable unset outside CI) that cannot be reached
    /// still skips: the helper returns `None` rather than failing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn connect_test_client_or_skip_when_optional_returns_none_for_an_unreachable_store() {
        // A 1 s connect budget: the refused connect is retried for that long,
        // well inside the 5 s probe.
        let config = Config {
            valkey_connect_timeout_secs: 1,
            ..minimal_test_config(UNREACHABLE_STORE_URL.to_string())
        };
        let client = connect_test_client_or_skip_with(
            TestValkeyRequirement::Optional,
            &config,
            TEST_VALKEY_CONNECT_HANG_GUARD,
        )
        .await;
        assert!(
            client.is_none(),
            "an optional store that refuses the connect is skipped (None)"
        );
    }

    /// The URL variable present, even empty, makes the store required whatever
    /// `GITLAB_CI` says; both absent make it optional.
    #[test]
    fn test_valkey_requirement_from_vars_is_required_with_the_url_present_and_optional_with_both_absent(
    ) {
        let present = |v: &str| Some(OsString::from(v));
        let cases = [
            (
                present("redis://valkey:6379"),
                None,
                TestValkeyRequirement::Required,
            ),
            (present(""), None, TestValkeyRequirement::Required),
            (
                present("redis://valkey:6379"),
                present("true"),
                TestValkeyRequirement::Required,
            ),
            (
                present(""),
                present("true"),
                TestValkeyRequirement::Required,
            ),
            (None, None, TestValkeyRequirement::Optional),
        ];
        for (url, gitlab_ci, expected) in cases {
            assert_eq!(
                TestValkeyRequirement::from_vars(url.clone(), gitlab_ci.clone()),
                expected,
                "{TEST_VALKEY_URL_VAR}={url:?}, GITLAB_CI={gitlab_ci:?}"
            );
        }
    }

    /// The URL variable absent in a CI job (`GITLAB_CI` present) is broken job
    /// wiring, and fails rather than skipping every store-backed test.
    #[test]
    #[should_panic(expected = "TEST_VALKEY_URL is unset in a CI job (GITLAB_CI is set)")]
    fn test_valkey_requirement_from_vars_with_the_url_absent_under_gitlab_ci_panics() {
        TestValkeyRequirement::from_vars(None, Some(OsString::from("true")));
    }
}
