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

//! Integration tests for `validate_durability_mode` (SDD §5.5 / §20.4).
//!
//! None of these tests changes the server's `appendonly` setting. It is
//! server-global, so setting it on the shared `TEST_VALKEY_URL` server would
//! change it under every other test process and suite using that server; the
//! rule and its source fence are in `shared_valkey_config_fence_test.rs`.
//!
//! **Staged replies.** Each branch test connects its own client through a
//! [`ScriptReplyProxy`] whose rule matches only the validator's probe,
//! `CONFIG GET appendonly`, and stages its answer. Every other frame goes to
//! the real server unchanged, so connecting and everything else run against
//! it. With checkpointing off (case 1b):
//!
//! | Staged reply                               | The shape of                       | Expected            |
//! |--------------------------------------------|------------------------------------|---------------------|
//! | `appendonly` → `yes`                       | self-hosted Valkey, AOF on         | `Ok`, no WARN       |
//! | `appendonly` → `no`                        | self-hosted Valkey, AOF off        | `Err` (SR-5.5)      |
//! | an empty reply                             | Azure Managed Redis, which exposes `CONFIG GET` but not the `appendonly` parameter | `Ok` + WARN |
//! | an error reply                             | AWS MemoryDB and ElastiCache, which block `CONFIG` | `Ok` + WARN |
//! | the real reply, delayed past the probe's timeout | a store that does not answer in time | `Ok` + WARN |
//!
//! The `yes` / `no` replies are a one-line script returning the pair
//! `{'appendonly', <value>}`, which on this RESP2 client is the same flat
//! two-element array a real `CONFIG GET appendonly` returns. The error text is
//! the one MemoryDB returned
//! (`docs-internal/validation/memorydb-compat-20260927/report.md`).
//!
//! The proxy counts the frames its rule matched (`hits()`), so the tests
//! observe on the wire whether the validator probed. Each case-1b test asserts
//! exactly one probe. The case-2 test (checkpointing on, adapter `Disabled`)
//! and the case-1a test (checkpointing on, a real adapter) assert none, and
//! stage an answer that would change their result had the probe been sent;
//! after the validator they probe once themselves through the same client to
//! show the rule does see that client's probe, so a zero is not the proxy
//! missing it.
//!
//! **The real server.** One test runs the validator unproxied against the
//! integration server and asserts only that the probe got a definite answer:
//! `Ok` without the WARN, or the SR-5.5 refusal. It does not assert which,
//! since another tenant of the shared server (a suite on a branch that
//! predates the rule) may change `appendonly` at any moment; so it holds
//! whatever any tenant does, and still proves the real `CONFIG GET` wiring.
//!
//! The proxy's raw connections send no `AUTH`, so `TEST_VALKEY_URL` must name
//! a store without a password; the sandbox and CI stores have none.

use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use fred::interfaces::ConfigInterface;
use tracing::instrument::WithSubscriber;
use tracing_subscriber::fmt::MakeWriter;

use live_edit::checkpoint::{validate_durability_mode, CheckpointAdapter, DurabilityGuardError};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::test_support::TestValkey;

use crate::storage::snapshot_read_support::{
    client_through, error_reply_script, test_valkey_addr, ProxyAction, ProxyRule, ScriptReplyProxy,
};

/// Verbatim SR-5.5 contract string. Must stay byte-for-byte identical to
/// `DurabilityGuardError::NoDurabilityConfigured`'s `#[error("...")]` text
/// at `src/checkpoint/durability_guard.rs`. Drift here is a Task contract
/// violation: the same string is documented in SDD §5.5 / §20.4,
/// `docs/service/docs/durability.md`, and the Helm `checkpoint.enabled` param
/// comment, and the four documentation surfaces stay in lockstep through
/// this constant.
const EXPECTED_SR_5_5_MSG: &str = "live-edit refuses to start with no durability configured: enable Valkey AOF, enable LIVE_EDIT_CHECKPOINT_ENABLED with a configured adapter, or both";

/// The substring the WARN carries when the probe cannot answer (an empty
/// reply, an error reply, a timeout): operators on managed Valkey grep for it.
/// Pinned here so a reword in the validator surfaces as a test failure.
const EXPECTED_UNVERIFIED_WARN_SUBSTRING: &str = "could not verify Valkey AOF state";

/// The error reply AWS MemoryDB returned to `CONFIG GET appendonly`, recorded in
/// `docs-internal/validation/memorydb-compat-20260927/raw/env-check.txt` (that
/// client printed it without the `ERR` code).
const MEMORYDB_CONFIG_REFUSAL: &str =
    "ERR unknown command 'CONFIG', with args beginning with: 'GET' 'appendonly' ";

/// The probe timeout the delay test hands the validator.
const SHORT_PROBE_TIMEOUT_MS: u64 = 200;

/// How long the delay test holds the probe before forwarding it: far past
/// [`SHORT_PROBE_TIMEOUT_MS`], so the timeout decides the outcome however
/// loaded the runner is. The test does not wait it out (the validator returns
/// at its timeout), so it is well inside nextest's 120 s hang-killer.
const STAGED_PROBE_DELAY: Duration = Duration::from_secs(5);

// ─────────────────────────────────────────────────────────────────────────────
// Log capture
// ─────────────────────────────────────────────────────────────────────────────

/// Capturing `MakeWriter` mirroring the `CapturingWriter` pattern used by
/// `tests/misc/service_token_startup_warns.rs` and
/// `tests/integration/health_supervisor_service_token_test.rs`. The
/// inline duplication (vs. a shared helper) follows the existing project
/// pattern — every consumer customizes the level filter slightly and the
/// helper itself is ~20 lines.
#[derive(Clone)]
struct CapturingWriter(Arc<StdMutex<Vec<u8>>>);

impl CapturingWriter {
    fn new() -> (Self, Arc<StdMutex<Vec<u8>>>) {
        let buf = Arc::new(StdMutex::new(Vec::new()));
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

/// Async log capture: install a capturing `tracing` subscriber as the
/// dispatcher for `future`, then await it. `WithSubscriber` attaches the
/// dispatcher to the future so it follows the task across worker-thread
/// migrations on the multi-thread runtime — `tracing::subscriber::set_default`
/// only installs a thread-local dispatcher and would not survive the
/// `tokio::time::timeout(...).await` inside `probe_aof_state` resuming
/// on a different worker.
///
/// `with_max_level(WARN)` filters at WARN so unrelated `info!`/`debug!`
/// events firing inside the validator body cannot mask the
/// WARN-not-firing assertions sibling tests in this file rely on.
async fn capture_log_output_async<F, T>(future: F) -> (T, String)
where
    F: std::future::Future<Output = T>,
{
    let (writer, captured) = CapturingWriter::new();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(writer)
        .with_max_level(tracing::Level::WARN)
        // ANSI escapes would otherwise pollute substring assertions on
        // environments that auto-enable colorization.
        .with_ansi(false)
        .finish();
    let dispatch = tracing::Dispatch::new(subscriber);
    let value = future.with_subscriber(dispatch).await;
    let buf = captured.lock().unwrap();
    let output = String::from_utf8_lossy(&buf).into_owned();
    (value, output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixtures
// ─────────────────────────────────────────────────────────────────────────────

/// The canonical integration `Config` (`crate::common::test_config()`) with
/// only the checkpoint fields overridden.
fn config_with_checkpoint(enabled: bool, adapter: CheckpointAdapter) -> live_edit::config::Config {
    let mut cfg = crate::common::test_config();
    cfg.checkpoint_enabled = enabled;
    cfg.checkpoint_adapter = adapter;
    cfg
}

/// A rule matching only the validator's probe frame, `CONFIG GET appendonly`.
fn appendonly_probe(action: ProxyAction) -> ProxyRule {
    Arc::new(move |args: &[Vec<u8>]| {
        let is_probe = args.len() == 3
            && args[0].eq_ignore_ascii_case(b"CONFIG")
            && args[1].eq_ignore_ascii_case(b"GET")
            && args[2].eq_ignore_ascii_case(b"appendonly");
        is_probe.then(|| action.clone())
    })
}

/// A script whose reply is the one `CONFIG GET appendonly` gives when the
/// server's setting is `value`.
fn appendonly_reply(value: &str) -> ProxyAction {
    ProxyAction::Eval(format!("return {{'appendonly', '{value}'}}"))
}

/// A proxy staging `action` for the probe, and a client connected through it.
async fn staged(action: ProxyAction) -> (ScriptReplyProxy, ValkeyClient) {
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), appendonly_probe(action)).await;
    let client = client_through(&proxy).await;
    (proxy, client)
}

/// Run the validator with `cfg` against `client`, capturing its WARN and
/// ERROR output.
async fn validate_capturing(
    cfg: &live_edit::config::Config,
    client: &ValkeyClient,
) -> (Result<(), DurabilityGuardError>, String) {
    capture_log_output_async(validate_durability_mode(cfg, client)).await
}

/// Assert `proxy` saw exactly `expected` probes.
fn assert_probes(proxy: &ScriptReplyProxy, expected: usize, context: &str) {
    assert_eq!(
        proxy.hits(),
        expected,
        "{context}: the validator must send `CONFIG GET appendonly` exactly {expected} time(s)"
    );
}

/// Probe once through `client`'s maintenance pool, as the validator would,
/// and assert the proxy saw it and staged `expected_value`. Shows that a zero
/// probe count before it means the validator sent none, not that the rule
/// missed this client's probe.
async fn assert_rule_sees_a_probe(
    proxy: &ScriptReplyProxy,
    client: &ValkeyClient,
    expected_value: &str,
) {
    let before = proxy.hits();
    let (pool, _token) = client.acquire_maintenance();
    let reply: HashMap<String, String> = pool
        .config_get("appendonly")
        .await
        .expect("the staged `CONFIG GET appendonly` must answer");
    assert_eq!(
        proxy.hits(),
        before + 1,
        "non-vacuity: the proxy's rule must see a probe sent through this client"
    );
    assert_eq!(
        reply.get("appendonly").map(String::as_str),
        Some(expected_value),
        "non-vacuity: the probe through this client must get the staged answer"
    );
}

/// Assert `result` is `Ok` and `captured` carries the unverified-AOF WARN at
/// WARN level.
fn assert_ok_with_unverified_warn(
    result: Result<(), DurabilityGuardError>,
    captured: &str,
    staged: &str,
) {
    result.unwrap_or_else(|e| {
        panic!("{staged}: a probe that cannot answer must degrade to Ok, not refuse; got {e:?}")
    });
    let warn_line = captured
        .lines()
        .find(|line| line.contains(EXPECTED_UNVERIFIED_WARN_SUBSTRING))
        .unwrap_or_else(|| {
            panic!(
                "{staged}: the captured output must contain \
                 {EXPECTED_UNVERIFIED_WARN_SUBSTRING:?}; got: {captured:?}"
            )
        });
    assert!(
        warn_line.contains("WARN"),
        "{staged}: the line must be emitted at WARN level; got: {warn_line:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Case 1b — checkpointing off: the probe decides
// ─────────────────────────────────────────────────────────────────────────────

/// AOF on is a durability layer: `Ok`, one probe, and no unverified-AOF WARN.
/// The WARN's absence is what tells this apart from a probe that answers
/// "unknown" for everything, which also returns `Ok`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn validate_durability_mode_staged_aof_yes_returns_ok_without_warn() {
    let (proxy, client) = staged(appendonly_reply("yes")).await;
    let cfg = config_with_checkpoint(false, CheckpointAdapter::Disabled);

    let (result, captured) = validate_capturing(&cfg, &client).await;

    result.expect("the validator must accept AOF=yes as the durability layer");
    assert!(
        !captured.contains(EXPECTED_UNVERIFIED_WARN_SUBSTRING),
        "a confident `yes` must not log the unverified-AOF WARN; got: {captured:?}"
    );
    assert_probes(&proxy, 1, "checkpointing off");
}

/// AOF off with checkpointing off is no durability layer at all: the SR-5.5
/// refusal, verbatim, after one probe.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn validate_durability_mode_staged_aof_no_refuses_with_sr_5_5() {
    let (proxy, client) = staged(appendonly_reply("no")).await;
    let cfg = config_with_checkpoint(false, CheckpointAdapter::Disabled);

    let err = validate_durability_mode(&cfg, &client)
        .await
        .expect_err("the validator must refuse AOF=no with checkpointing off");

    assert_eq!(
        err.to_string(),
        EXPECTED_SR_5_5_MSG,
        "the case-1 refusal must carry the verbatim SR-5.5 contract string"
    );
    assert_probes(&proxy, 1, "checkpointing off");
}

/// Azure Managed Redis answers `CONFIG GET` for a parameter it does not
/// expose with an empty reply: the probe cannot tell, so `Ok` plus the WARN.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn validate_durability_mode_staged_empty_reply_warns_and_returns_ok() {
    let (proxy, client) = staged(ProxyAction::Eval("return {}".to_string())).await;
    let cfg = config_with_checkpoint(false, CheckpointAdapter::Disabled);

    let (result, captured) = validate_capturing(&cfg, &client).await;

    assert_ok_with_unverified_warn(result, &captured, "an empty reply");
    assert_probes(&proxy, 1, "checkpointing off");
}

/// AWS MemoryDB and ElastiCache refuse the `CONFIG` command with an error
/// reply: the probe cannot tell, so `Ok` plus the WARN.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn validate_durability_mode_staged_error_reply_warns_and_returns_ok() {
    let (proxy, client) = staged(ProxyAction::Eval(error_reply_script(
        MEMORYDB_CONFIG_REFUSAL,
    )))
    .await;
    let cfg = config_with_checkpoint(false, CheckpointAdapter::Disabled);

    let (result, captured) = validate_capturing(&cfg, &client).await;

    assert_ok_with_unverified_warn(result, &captured, "an error reply");
    assert_probes(&proxy, 1, "checkpointing off");
}

/// A probe the store does not answer within the probe timeout cannot tell:
/// `Ok` plus the WARN. The proxy holds the real `CONFIG GET` for
/// [`STAGED_PROBE_DELAY`], far past the [`SHORT_PROBE_TIMEOUT_MS`] the
/// validator is given, so the timeout decides however fast the store is. A
/// `Delay`, not a `Stall`: the reply does arrive later, so no connection is
/// left stalled for good.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn validate_durability_mode_probe_delayed_past_timeout_warns_and_returns_ok() {
    let (proxy, client) = staged(ProxyAction::Delay(STAGED_PROBE_DELAY)).await;
    let mut cfg = config_with_checkpoint(false, CheckpointAdapter::Disabled);
    cfg.valkey_command_timeout_ms = SHORT_PROBE_TIMEOUT_MS;

    let (result, captured) = validate_capturing(&cfg, &client).await;

    assert_ok_with_unverified_warn(result, &captured, "a delayed reply");
    assert_probes(&proxy, 1, "checkpointing off");
}

// ─────────────────────────────────────────────────────────────────────────────
// Checkpointing on — decided from configuration, no probe
// ─────────────────────────────────────────────────────────────────────────────

/// Case 2: checkpointing on with the no-op `Disabled` adapter is no durability
/// layer, and that is known from configuration alone: the SR-5.5 refusal with
/// no probe sent. The staged `yes` would make a probe-reaching validator
/// return `Ok`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn validate_durability_mode_checkpoints_on_with_disabled_adapter_refuses_without_probing() {
    let (proxy, client) = staged(appendonly_reply("yes")).await;
    let cfg = config_with_checkpoint(true, CheckpointAdapter::Disabled);

    let err = validate_durability_mode(&cfg, &client)
        .await
        .expect_err("the validator must refuse checkpointing on with the Disabled adapter");

    assert_eq!(
        err.to_string(),
        EXPECTED_SR_5_5_MSG,
        "the case-2 refusal must carry the verbatim SR-5.5 contract string"
    );
    assert_probes(&proxy, 0, "checkpointing on, adapter Disabled");
    assert_rule_sees_a_probe(&proxy, &client, "yes").await;
}

/// Case 1a: checkpointing on with a real adapter is the durability layer, so
/// AOF does not matter: `Ok` with no probe sent, for each real adapter. The
/// staged `no` would make a probe-reaching validator refuse.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn validate_durability_mode_checkpoints_on_with_real_adapter_returns_ok_without_probing() {
    for adapter in [
        CheckpointAdapter::InMemory,
        CheckpointAdapter::LocalDisk,
        CheckpointAdapter::StorageApi,
    ] {
        let (proxy, client) = staged(appendonly_reply("no")).await;
        let cfg = config_with_checkpoint(true, adapter);

        validate_durability_mode(&cfg, &client)
            .await
            .unwrap_or_else(|e| {
                panic!(
                    "checkpointing on with {adapter:?} must be accepted whatever AOF is; got {e:?}"
                )
            });

        assert_probes(&proxy, 0, &format!("checkpointing on, adapter {adapter:?}"));
        assert_rule_sees_a_probe(&proxy, &client, "no").await;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The real server
// ─────────────────────────────────────────────────────────────────────────────

/// Unproxied, against the integration server, with checkpointing off: the
/// validator's probe gets a definite answer, so the result is either `Ok` with
/// no unverified-AOF WARN (the server said `yes`) or the verbatim SR-5.5
/// refusal (it said `no`). Which of the two is not asserted: the setting is
/// server-global and another tenant of the shared server may change it at any
/// moment, so the test holds whatever any tenant does. The staged tests pin
/// which answer maps to which result; this one proves the real `CONFIG GET`
/// wiring against a real server reply.
///
/// `Ok` with the WARN fails: the integration server is self-hosted Valkey,
/// which must answer `yes` or `no`, and a WARN there means the guard would
/// boot unchecked against that Valkey version.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn validate_durability_mode_real_server_probe_gets_a_definite_answer() {
    let tv = TestValkey::setup().await;
    let cfg = config_with_checkpoint(false, CheckpointAdapter::Disabled);

    let (result, captured) = validate_capturing(&cfg, tv.client()).await;

    match result {
        Ok(()) => assert!(
            !captured.contains(EXPECTED_UNVERIFIED_WARN_SUBSTRING),
            "the real server's `CONFIG GET appendonly` reply was not classified as a definite \
             yes or no: the validator returned Ok with the unverified-AOF WARN; got: {captured:?}"
        ),
        Err(e) => assert_eq!(
            e.to_string(),
            EXPECTED_SR_5_5_MSG,
            "the real server's `CONFIG GET appendonly` reply was not classified as a definite \
             yes or no: the validator failed with something other than the SR-5.5 refusal"
        ),
    }
}
